"""
Step 7: 视频文件管理路由

对应 PRD v0.9 §5 流程二（视频切片流水线）的上传和转写阶段。

接口列表：
  POST   /videos/upload-and-transcribe  — 上传视频文件，触发 Whisper 异步转写
  GET    /videos/list                   — 查询已上传的视频文件列表
  GET    /videos/transcripts/{show}/{ep} — 查询指定集的 Whisper 转写结果
  DELETE /videos/transcripts/{show}/{ep} — 删除转写缓存（下次触发时重新转写）
  GET    /videos/status                 — 视频和转写统计

设计说明：
  - 转写任务通过 task_runner 串行执行，避免多个 Whisper 任务并发抢占内存
  - 转写结果永久缓存到 whisper_transcripts 表，重复上传同一集时可直接复用
  - 任务内部使用独立 SessionLocal（而非依赖注入的 session），
    因为 FastAPI 的 session 在请求结束后即关闭，无法在后台任务中使用
"""

import os
import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from database import get_session
from models.video_file import VideoFile
from models.subtitle import SubtitleRaw
from models.whisper_transcript import WhisperTranscript
from services.whisper_service import transcribe_video
from tasks.task_runner import submit_task
import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/videos", tags=["videos"])

# 允许上传的视频格式
_ALLOWED_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".rmvb"}


@router.post("/upload-and-transcribe")
async def upload_and_transcribe(
    video_file: UploadFile = File(..., description="视频文件（MP4/MKV/AVI 等）"),
    subtitle_code: str = Form(..., description="关联的字幕批次编码（必须先上传字幕）"),
    force_retranscribe: bool = Form(False, description="是否强制重新转写（忽略缓存）"),
    force_overwrite: bool = Form(False, description="已存在同名视频时是否强制覆盖"),
    session: AsyncSession = Depends(get_session),
):
    """
    上传视频文件并触发 Whisper 转写任务（异步）。

    流程：
    1. 校验字幕批次编码存在（必须先上传对应字幕）
    2. 校验视频格式（.mp4/.mkv/.avi 等）
    3. 检测是否已存在同剧集同集号的视频，若存在且 force_overwrite=False，返回冲突提示
    4. 将视频文件保存到 media/{show_name}_{episode}/video{ext}
    5. 提交异步转写任务到串行任务队列
    6. 立即返回 job_id，前端通过 GET /jobs/{job_id} 轮询进度

    注意：
    - 同一剧集重复上传需显式传 force_overwrite=True 才会覆盖
    - 若已有转写缓存且 force_retranscribe=False，转写任务会直接复用缓存
    - 转写任务在后台串行执行，不阻塞当前请求
    """
    # ── 校验 subtitle_code 是否存在，并获取剧名/集号 ──
    batch_row = await session.execute(
        select(SubtitleRaw.show_name, SubtitleRaw.en_name, SubtitleRaw.episode)
        .where(SubtitleRaw.subtitle_code == subtitle_code)
        .limit(1)
    )
    batch_info = batch_row.first()
    if batch_info is None:
        raise HTTPException(
            status_code=400,
            detail=f"字幕批次 {subtitle_code} 不存在，请先上传对应字幕文件",
        )
    show_name = batch_info.show_name
    episode = batch_info.episode or ""   # 保持与 KnowledgeNode.episode 一致，空集号不替换为占位符

    # ── 校验视频格式 ──
    ext = os.path.splitext(video_file.filename or "")[1].lower()
    if ext not in _ALLOWED_VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的视频格式: {ext}，支持: {', '.join(sorted(_ALLOWED_VIDEO_EXTS))}",
        )

    # ── 检测重复视频 ──
    existing_video = (await session.execute(
        select(VideoFile).where(
            VideoFile.show_name == show_name,
            VideoFile.episode == episode,
        )
    )).scalar_one_or_none()

    if existing_video and not force_overwrite:
        # 查询已有切片数量
        from models.clip import Clip
        existing_clips = (await session.execute(
            select(func.count(Clip.id)).where(
                Clip.show_name == show_name,
                Clip.episode == episode,
                Clip.is_active == True,
            )
        )).scalar() or 0

        return {
            "status": "duplicate",
            "message": f"已上传过相同视频（{show_name} {episode}），是否覆盖之前的切片结果？",
            "existing_video": {
                "id": existing_video.id,
                "file_path": existing_video.file_path,
                "file_size_mb": round((existing_video.file_size or 0) / 1024 / 1024, 2),
                "has_transcript": existing_video.has_transcript,
                "created_at": existing_video.created_at.isoformat() if existing_video.created_at else None,
            },
            "existing_clips_count": existing_clips,
            "hint": "如需覆盖，请重新提交并设置 force_overwrite=true",
        }

    # ── 覆盖模式日志 ──
    if existing_video and force_overwrite:
        logger.info(f"覆盖模式：用户确认覆盖 {show_name} {episode} 的视频及转写结果")

    # ── 保存视频文件到 media/{show_name}_{episode}/ 目录 ──
    save_dir = os.path.join(str(config.MEDIA_DIR), f"{show_name}_{episode}")
    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(save_dir, f"video{ext}")

    # 分块流式写入，避免大文件一次性读入内存
    file_size = 0
    chunk_size = 64 * 1024  # 64KB
    try:
        with open(video_path, "wb") as f:
            while True:
                chunk = await video_file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                file_size += len(chunk)
    except Exception as e:
        logger.error(f"视频文件写入失败: {e}")
        raise HTTPException(status_code=500, detail="视频文件保存失败")

    logger.info(f"视频文件已保存: {video_path} ({file_size / 1024 / 1024:.1f}MB)")

    # ── 定义后台转写任务 ──
    # 注意：任务内部必须使用独立 SessionLocal，
    # 因为 FastAPI 依赖注入的 session 在请求结束后即关闭
    async def transcribe_task():
        """后台 Whisper 转写任务"""
        from database import SessionLocal
        async with SessionLocal() as task_session:
            result = await transcribe_video(
                video_path=video_path,
                show_name=show_name,
                episode=episode,
                session=task_session,
                force_retranscribe=force_retranscribe,
                subtitle_code=subtitle_code,
            )
            return {
                "show_name": show_name,
                "episode": episode,
                "transcript_count": len(result),
                "video_path": video_path,
            }

    # ── 提交到串行任务队列 ──
    job_id = await submit_task(
        transcribe_task(),
        description=f"视频转写: {show_name} {episode}",
    )

    logger.info(f"转写任务已入队: job_id={job_id}, {show_name} {episode}")

    return {
        "job_id": job_id,
        "status": "pending",
        "subtitle_code": subtitle_code,
        "show_name": show_name,
        "episode": episode,
        "video_path": video_path,
        "message": f"视频已上传（{file_size / 1024 / 1024:.1f}MB），转写任务已提交，请通过 /jobs/{job_id} 查询进度",
    }


@router.get("/list")
async def list_videos(
    show_name: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    """
    查询已上传的视频文件列表。

    Args:
        show_name: 可选，按剧名筛选

    Returns:
        视频文件列表，包含文件大小（MB）、时长（秒）、是否已转写等信息
    """
    stmt = select(VideoFile).order_by(VideoFile.created_at.desc())
    if show_name:
        stmt = stmt.where(VideoFile.show_name == show_name)

    result = await session.execute(stmt)
    videos = result.scalars().all()

    return {
        "total": len(videos),
        "videos": [
            {
                "id": v.id,
                "show_name": v.show_name,
                "episode": v.episode,
                "file_path": v.file_path,
                "file_size_mb": round((v.file_size or 0) / 1024 / 1024, 2),
                "duration": v.duration,           # 视频时长（秒），由 FFprobe 获取
                "has_transcript": v.has_transcript,  # 是否已完成 Whisper 转写
                "created_at": v.created_at.isoformat() if v.created_at else None,
            }
            for v in videos
        ],
    }


async def _load_transcripts(show_name: str, episode: str, session):
    stmt = (
        select(WhisperTranscript)
        .where(WhisperTranscript.show_name == show_name, WhisperTranscript.episode == episode)
        .order_by(WhisperTranscript.seq_index)
    )
    result = await session.execute(stmt)
    transcripts = result.scalars().all()
    if not transcripts and episode == "":
        stmt2 = (
            select(WhisperTranscript)
            .where(WhisperTranscript.show_name == show_name, WhisperTranscript.episode == "S00E00")
            .order_by(WhisperTranscript.seq_index)
        )
        result2 = await session.execute(stmt2)
        transcripts = result2.scalars().all()
    if not transcripts:
        raise HTTPException(
            status_code=404,
            detail=f"未找到转写结果: {show_name} {episode}，请先上传视频并等待转写完成",
        )
    return {
        "show_name": show_name,
        "episode": episode,
        "total": len(transcripts),
        "transcripts": [
            {"seq_index": t.seq_index, "text": t.text, "start_time": t.start_time,
             "end_time": t.end_time, "whisper_model": t.whisper_model}
            for t in transcripts
        ],
    }


@router.get("/transcripts-query")
async def get_transcripts_by_query(
    show_name: str,
    episode: str = "",
    session: AsyncSession = Depends(get_session),
):
    """通过 query 参数查询转写结果（支持空集号）"""
    return await _load_transcripts(show_name, episode, session)


@router.get("/transcripts/{show_name}/{episode}")
async def get_transcripts(
    show_name: str,
    episode: str,
    session: AsyncSession = Depends(get_session),
):
    """查询指定集的 Whisper 转写结果（按时间顺序）"""
    return await _load_transcripts(show_name, episode, session)


@router.delete("/transcripts/{show_name}/{episode}")
async def delete_transcripts(
    show_name: str,
    episode: str,
    session: AsyncSession = Depends(get_session),
):
    """
    删除指定集的 Whisper 转写缓存。

    使用场景：
    - 转写结果质量不佳，需要用更大的模型重新转写
    - 视频文件已更换，旧转写结果不再适用

    删除后，下次调用 /videos/upload-and-transcribe 时会重新执行转写。
    """
    from sqlalchemy import delete as sql_delete

    result = await session.execute(
        sql_delete(WhisperTranscript).where(
            WhisperTranscript.show_name == show_name,
            WhisperTranscript.episode == episode,
        )
    )
    await session.commit()

    logger.info(f"已删除转写缓存: {show_name} {episode}，共 {result.rowcount} 条")

    return {
        "deleted": result.rowcount,
        "message": f"已删除 {show_name} {episode} 的转写缓存（{result.rowcount} 条），下次上传时将重新转写",
    }


@router.delete("/clear-all")
async def clear_all_videos_and_clips(session: AsyncSession = Depends(get_session)):
    """
    一键清空：删除所有已上传视频文件、生成的切片及相关 DB 记录。

    具体操作：
    1. 删除 media 目录下所有文件（视频原件、切片视频、缩略图）
    2. 清空 clips / whisper_transcripts / video_files 三张表
    3. 重置知识节点的 has_any_clip 标志

    ⚠️ 不可逆操作，请谨慎使用。
    """
    import shutil
    from sqlalchemy import delete as sql_delete, update as sql_update
    from models.clip import Clip
    from models.knowledge_node import KnowledgeNode

    # ── 1. 删除磁盘文件 ──────────────────────────────
    deleted_files = 0
    if config.MEDIA_DIR.exists():
        for item in config.MEDIA_DIR.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                    deleted_files += 1
                else:
                    item.unlink()
                    deleted_files += 1
            except Exception as e:
                logger.warning(f"删除文件失败: {item} — {e}")

    # ── 2. 清空数据库表 ──────────────────────────────
    clips_deleted = (await session.execute(sql_delete(Clip))).rowcount
    transcripts_deleted = (await session.execute(sql_delete(WhisperTranscript))).rowcount
    videos_deleted = (await session.execute(sql_delete(VideoFile))).rowcount

    # ── 3. 重置知识节点标志 ──────────────────────────
    await session.execute(
        sql_update(KnowledgeNode).values(has_any_clip=False)
    )
    await session.commit()

    logger.info(
        f"一键清空完成：文件 {deleted_files} 个，"
        f"切片记录 {clips_deleted} 条，转写记录 {transcripts_deleted} 条，"
        f"视频记录 {videos_deleted} 条"
    )
    return {
        "deleted_disk_items": deleted_files,
        "deleted_clips": clips_deleted,
        "deleted_transcripts": transcripts_deleted,
        "deleted_videos": videos_deleted,
        "message": f"已清空：{videos_deleted} 个视频、{clips_deleted} 条切片记录、{transcripts_deleted} 条转写记录，磁盘文件 {deleted_files} 个",
    }


@router.get("/status")
async def videos_status(session: AsyncSession = Depends(get_session)):
    """
    查询视频和转写状态统计（用于管理后台概览面板）。

    返回：
    - total_videos:             已上传视频文件总数
    - transcribed_videos:       已完成 Whisper 转写的视频数
    - total_transcript_segments: 转写片段总数（所有集的语音片段之和）
    """
    # 视频文件总数
    total_videos = (
        await session.execute(select(func.count(VideoFile.id)))
    ).scalar() or 0

    # 已转写视频数（has_transcript=True）
    transcribed = (
        await session.execute(
            select(func.count(VideoFile.id)).where(VideoFile.has_transcript == True)
        )
    ).scalar() or 0

    # 转写片段总数（whisper_transcripts 表的总行数）
    total_segments = (
        await session.execute(select(func.count(WhisperTranscript.id)))
    ).scalar() or 0

    return {
        "total_videos": total_videos,
        "transcribed_videos": transcribed,
        "total_transcript_segments": total_segments,
    }
