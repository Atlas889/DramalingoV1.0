"""
DramaLingo · 视频切片路由
对应 PRD v0.9 §5 流程二 · 切片生成与管理

接口列表：
  POST   /clips/generate           — 上传视频，触发完整切片流水线（Whisper → 对齐 → FFmpeg）
  GET    /clips/list               — 分页查询切片列表（支持剧名/集号/状态筛选）
  GET    /clips/report/{show}/{ep} — 查询指定集的切片质量报告（覆盖率/置信度分布）
  DELETE /clips/{id}               — 删除单个切片记录（同时删除磁盘文件）
  POST   /clips/batch-delete       — 批量删除切片记录
  POST   /clips/clear-all          — 清空全部视频与切片数据（危险操作，需二次确认）

流水线说明（generate）：
  视频上传 → Whisper 转写（首次）/ 复用缓存 → 合并零散片段 → 序列对齐 →
  FFmpeg 并发切片（含上下文扩展 + padding） → 生成缩略图 → 更新 clips 表 + knowledge_nodes.has_any_clip
"""

import os
import logging
import subprocess
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case, delete as sql_delete, or_, update as sql_update

import config
from database import get_session, SessionLocal
from models.clip import Clip
from models.knowledge_node import KnowledgeNode
from models.video_file import VideoFile
from models.subtitle import SubtitleRaw
from services.clip_pipeline import generate_clips
from tasks.task_runner import submit_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/clips", tags=["clips"])


@router.post("/generate")
async def generate_clips_api(
    video_file: UploadFile = File(..., description="视频文件（MP4/MKV/AVI 等）"),
    subtitle_code: str = Form(..., description="关联的字幕批次编码"),
    force_retranscribe: bool = Form(False, description="是否强制重新转写"),
    force_overwrite: bool = Form(False, description="已存在同名视频时是否强制覆盖旧切片"),
    session: AsyncSession = Depends(get_session),
):
    """
    上传视频文件并触发完整切片流水线（Whisper 转写 → 序列对齐 → FFmpeg 切片）。

    若同剧集同集号已有视频和切片，且 force_overwrite=False，返回冲突提示而非直接覆盖。
    返回 job_id 供前端轮询进度。
    """
    # 校验 subtitle_code 存在，并获取剧名/集号
    batch_row = await session.execute(
        select(SubtitleRaw.show_name, SubtitleRaw.episode)
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

    # ── 检测重复视频 ──
    existing_video = (await session.execute(
        select(VideoFile).where(
            VideoFile.show_name == show_name,
            VideoFile.episode == episode,
        )
    )).scalar_one_or_none()

    if existing_video and not force_overwrite:
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

    # 验证文件格式
    allowed_exts = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".rmvb"}
    ext = os.path.splitext(video_file.filename or "")[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"不支持的视频格式: {ext}")

    # ── 覆盖模式：将旧切片标记为 is_active=False ──
    if existing_video and force_overwrite:
        deactivated = (await session.execute(
            sql_update(Clip)
            .where(
                Clip.show_name == show_name,
                Clip.episode == episode,
                Clip.is_active == True,
            )
            .values(is_active=False)
        )).rowcount
        await session.commit()
        if deactivated:
            logger.info(f"覆盖模式：已将 {show_name} {episode} 的 {deactivated} 条旧切片标记为 inactive")

    # 保存视频文件
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

    # 提交切片任务
    job_id = str(uuid4())

    async def clip_task(_jid=job_id):
        async with SessionLocal() as task_session:
            result = await generate_clips(
                video_path=video_path,
                show_name=show_name,
                episode=episode,
                session=task_session,
                force_retranscribe=force_retranscribe,
                task_id=_jid,
            )
            return result

    await submit_task(clip_task(), description=f"视频切片: {show_name} {episode}", task_id=job_id)
    
    return {
        "job_id": job_id,
        "status": "pending",
        "show_name": show_name,
        "episode": episode,
        "video_path": video_path,
        "message": f"视频已上传，切片任务已提交，请通过 /jobs/{job_id} 查询进度",
    }


@router.get("/quality-report")
async def quality_report(
    show_name: Optional[str] = None,
    episode: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    """
    查询匹配质量报告：各置信度区间的切片分布。
    
    置信度区间：
    - 高置信（≥ 0.85）：已生成切片
    - 低置信（0.60~0.85）：待人工确认
    - 失败（< 0.60 或切片失败）
    """
    stmt = select(
        func.count(Clip.id).label("total"),
        func.sum(case((Clip.clip_status == "matched", 1), else_=0)).label("matched"),
        func.sum(case((Clip.clip_status == "unmatched", 1), else_=0)).label("unmatched"),
        func.sum(case((Clip.clip_status == "failed", 1), else_=0)).label("failed"),
        func.avg(Clip.match_confidence).label("avg_confidence"),
    )

    if show_name:
        stmt = stmt.where(Clip.show_name == show_name)
    if episode:
        stmt = stmt.where(Clip.episode == episode)

    result = await session.execute(stmt)
    row = result.one()

    # 按剧集分组统计
    stmt_by_ep = (
        select(
            Clip.show_name,
            Clip.episode,
            func.count(Clip.id).label("total"),
            func.sum(case((Clip.clip_status == "matched", 1), else_=0)).label("matched"),
            func.sum(case((Clip.clip_status == "unmatched", 1), else_=0)).label("unmatched"),
            func.sum(case((Clip.clip_status == "failed", 1), else_=0)).label("failed"),
            func.avg(Clip.match_confidence).label("avg_confidence"),
        )
        .group_by(Clip.show_name, Clip.episode)
        .order_by(Clip.show_name, Clip.episode)
    )

    if show_name:
        stmt_by_ep = stmt_by_ep.where(Clip.show_name == show_name)

    result_by_ep = await session.execute(stmt_by_ep)
    episodes = result_by_ep.all()

    return {
        "summary": {
            "total": row.total or 0,
            "matched": row.matched or 0,
            "unmatched": row.unmatched or 0,
            "failed": row.failed or 0,
            "avg_confidence": round(float(row.avg_confidence or 0), 4),
        },
        "by_episode": [
            {
                "show_name": ep.show_name,
                "episode": ep.episode,
                "total": ep.total,
                "matched": ep.matched or 0,
                "unmatched": ep.unmatched or 0,
                "failed": ep.failed or 0,
                "avg_confidence": round(float(ep.avg_confidence or 0), 4),
            }
            for ep in episodes
        ],
    }


@router.get("/list")
async def list_clips(
    show_name: Optional[str] = None,
    episode: Optional[str] = None,
    clip_status: Optional[str] = None,
    node_id: Optional[int] = None,
    content_type: Optional[str] = None,
    title_q: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
):
    """查询切片列表（支持大类和标题模糊搜索过滤）"""
    stmt = select(Clip)

    # 按内容大类过滤：JOIN KnowledgeNode
    if content_type:
        stmt = stmt.join(KnowledgeNode, Clip.node_id == KnowledgeNode.id).where(
            KnowledgeNode.content_type == content_type
        )

    # 标题模糊搜索（台词 OR 剧名）
    if title_q:
        like_q = f"%{title_q}%"
        stmt = stmt.where(
            or_(Clip.subtitle_en.ilike(like_q), Clip.show_name.ilike(like_q))
        )

    if show_name:
        stmt = stmt.where(Clip.show_name == show_name)
    if episode:
        stmt = stmt.where(Clip.episode == episode)
    if clip_status:
        stmt = stmt.where(Clip.clip_status == clip_status)
    if node_id:
        stmt = stmt.where(Clip.node_id == node_id)

    total_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await session.execute(total_stmt)).scalar() or 0

    stmt = stmt.order_by(Clip.created_at.desc()).offset(offset).limit(limit)
    result = await session.execute(stmt)
    clips = result.scalars().all()

    return {
        "total": total,
        "clips": [
            {
                "id": c.id,
                "clip_code": c.clip_code or f"CLP-{c.id:08d}",
                "node_id": c.node_id,
                "show_name": c.show_name,
                "episode": c.episode,
                "subtitle_en": c.subtitle_en,
                "subtitle_zh": c.subtitle_zh,
                "clip_status": c.clip_status,
                "is_active": c.is_active,
                "duration": c.duration,
                "match_confidence": c.match_confidence,
                "whisper_start": c.whisper_start,
                "whisper_end": c.whisper_end,
                "whisper_text": c.whisper_text or "",
                "stream_url": c.stream_url,
                "thumb_url": c.thumb_url,
                "align_step": c.align_step,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in clips
        ],
    }



@router.get("/status")
async def clips_status(session: AsyncSession = Depends(get_session)):
    """切片模块状态统计"""
    total = (await session.execute(select(func.count(Clip.id)))).scalar() or 0
    matched = (await session.execute(
        select(func.count(Clip.id)).where(Clip.clip_status == "matched")
    )).scalar() or 0
    unmatched = (await session.execute(
        select(func.count(Clip.id)).where(Clip.clip_status == "unmatched")
    )).scalar() or 0
    failed = (await session.execute(
        select(func.count(Clip.id)).where(Clip.clip_status == "failed")
    )).scalar() or 0
    nodes_with_clips = (await session.execute(
        select(func.count(KnowledgeNode.id)).where(KnowledgeNode.has_any_clip == True)
    )).scalar() or 0

    return {
        "module": "clips",
        "status": "ready",
        "total_clips": total,
        "done_clips": matched,
        "pending_clips": unmatched,
        "failed_clips": failed,
        "nodes_with_clips": nodes_with_clips,
    }


@router.get("/status-breakdown")
async def clips_status_breakdown(
    show_name: Optional[str] = None,
    episode: Optional[str] = None,
    content_type: Optional[str] = None,
    title_q: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    """各状态切片数量分布（用于切片列表顶部统计栏，支持大类和关键词过滤）"""
    base = select(Clip.clip_status, func.count(Clip.id).label("cnt")).group_by(Clip.clip_status)
    if content_type:
        base = base.join(KnowledgeNode, Clip.node_id == KnowledgeNode.id).where(
            KnowledgeNode.content_type == content_type
        )
    if title_q:
        like_q = f"%{title_q}%"
        base = base.where(or_(Clip.subtitle_en.ilike(like_q), Clip.show_name.ilike(like_q)))
    if show_name:
        base = base.where(Clip.show_name == show_name)
    if episode:
        base = base.where(Clip.episode == episode)
    rows = (await session.execute(base)).all()
    return {r.clip_status: r.cnt for r in rows}


@router.post("/retry-failed")
async def retry_failed_clips(
    show_name: str = Form(...),
    episode: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    """
    重新生成当前 failed 状态的切片。
    从 video_files 表获取视频路径，提交增量补切任务。
    """
    # 找到视频文件路径
    vf_result = await session.execute(
        select(VideoFile).where(
            VideoFile.show_name == show_name,
            VideoFile.episode == episode,
        )
    )
    video_file = vf_result.scalar_one_or_none()
    if video_file is None:
        raise HTTPException(status_code=404, detail=f"未找到视频记录: {show_name} {episode}")
    if not os.path.exists(video_file.file_path):
        raise HTTPException(status_code=404, detail=f"视频文件不存在: {video_file.file_path}")

    # 收集 failed 切片对应的 node_id
    failed_rows = await session.execute(
        select(Clip.node_id).where(
            Clip.show_name == show_name,
            Clip.episode == episode,
            Clip.clip_status == "failed",
        ).distinct()
    )
    node_ids = [r[0] for r in failed_rows.all()]
    if not node_ids:
        return {"message": "没有 failed 状态的切片需要重试", "count": 0}

    video_path = video_file.file_path
    job_id = str(uuid4())

    async def retry_task(_jid=job_id):
        async with SessionLocal() as task_session:
            return await generate_clips(
                video_path=video_path,
                show_name=show_name,
                episode=episode,
                session=task_session,
                node_ids=node_ids,
                task_id=_jid,
            )

    await submit_task(
        retry_task(),
        description=f"重试失败切片: {show_name} {episode}（{len(node_ids)} 个）",
        task_id=job_id,
    )
    return {"job_id": job_id, "node_count": len(node_ids), "message": "重试任务已提交"}


@router.post("/reset-and-rerun")
async def reset_and_rerun(
    show_name: str = Form(...),
    episode: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    """
    删除指定剧集的所有切片记录并重新提交完整切片流水线。
    用于修复 Bug 后清理历史脏数据、全量重跑。
    """
    # 找视频文件：先精确匹配 show_name+episode，再按 show_name 兜底
    vf_result = await session.execute(
        select(VideoFile).where(
            VideoFile.show_name == show_name,
            VideoFile.episode == episode,
        )
    )
    video_file = vf_result.scalar_one_or_none()
    if video_file is None:
        vf_result2 = await session.execute(
            select(VideoFile).where(VideoFile.show_name == show_name).limit(1)
        )
        video_file = vf_result2.scalar_one_or_none()
    if video_file is None:
        raise HTTPException(status_code=404, detail=f"未找到视频记录，请先上传视频: {show_name}")
    if not os.path.exists(video_file.file_path):
        raise HTTPException(status_code=404, detail=f"视频文件不存在: {video_file.file_path}")
    # 用数据库里实际的 episode（防止前端传来的值有偏差）
    episode = video_file.episode

    # 删除旧切片记录
    del_result = await session.execute(
        sql_delete(Clip).where(
            Clip.show_name == show_name,
            Clip.episode == episode,
        )
    )
    deleted = del_result.rowcount

    # 重置 knowledge_nodes.has_any_clip
    from models.knowledge_node import KnowledgeNode
    from sqlalchemy import update as sql_update
    await session.execute(
        sql_update(KnowledgeNode)
        .where(KnowledgeNode.show_name == show_name, KnowledgeNode.episode == episode)
        .values(has_any_clip=False)
    )
    await session.commit()

    # 提交完整切片任务
    video_path = video_file.file_path
    job_id = str(uuid4())

    async def rerun_task(_jid=job_id):
        async with SessionLocal() as task_session:
            return await generate_clips(
                video_path=video_path,
                show_name=show_name,
                episode=episode,
                session=task_session,
                force_retranscribe=False,
                task_id=_jid,
            )

    await submit_task(
        rerun_task(),
        description=f"重置重跑: {show_name} {episode}",
        task_id=job_id,
    )
    return {
        "job_id": job_id,
        "deleted_clips": deleted,
        "message": f"已删除 {deleted} 条旧切片，完整流水线已重新提交",
    }


# ── 批量删除切片 ──────────────────────────────────────────────────────────────

class BatchDeleteClipsRequest(BaseModel):
    ids: List[int]


@router.post("/batch-delete")
async def batch_delete_clips(
    body: BatchDeleteClipsRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    批量删除指定 ID 的切片记录。
    同步更新关联 KnowledgeNode.has_any_clip（若该节点不再有任何切片则重置为 False）。
    """
    if not body.ids:
        return {"deleted": 0, "message": "未指定任何 ID"}

    # 查询这批切片对应的 node_id，便于后续更新 has_any_clip
    node_rows = await session.execute(
        select(Clip.node_id).where(Clip.id.in_(body.ids)).distinct()
    )
    affected_node_ids = [r[0] for r in node_rows.all()]

    # 删除切片
    del_result = await session.execute(
        sql_delete(Clip).where(Clip.id.in_(body.ids))
    )
    deleted = del_result.rowcount

    # 更新 has_any_clip：若节点已无任何切片则重置
    if affected_node_ids:
        from sqlalchemy import update as sql_update
        # 找出仍然有切片的节点
        remaining_rows = await session.execute(
            select(Clip.node_id)
            .where(Clip.node_id.in_(affected_node_ids))
            .distinct()
        )
        still_has = {r[0] for r in remaining_rows.all()}
        no_clips_ids = [nid for nid in affected_node_ids if nid not in still_has]
        if no_clips_ids:
            await session.execute(
                sql_update(KnowledgeNode)
                .where(KnowledgeNode.id.in_(no_clips_ids))
                .values(has_any_clip=False)
            )

    await session.commit()
    return {"deleted": deleted, "message": f"已删除 {deleted} 条切片"}


# ── 从实际文件反查真实时长（批量补填）──────────────────────────────────────────

def _probe_duration(file_path: str) -> Optional[float]:
    """用 ffprobe 读取媒体文件的真实时长（秒），失败返回 None。"""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return round(float(result.stdout.strip()), 2)
    except Exception as e:
        logger.debug(f"ffprobe 读取时长失败 {file_path}: {e}")
    return None


async def _run_backfill_durations(task_id: str):
    """后台任务：遍历所有 matched 切片，用 ffprobe 读实际文件时长并更新 DB。"""
    from tasks.task_runner import update_task_progress
    async with SessionLocal() as session:
        rows = (await session.execute(
            select(Clip.id, Clip.stream_url, Clip.duration)
            .where(Clip.clip_status == "matched", Clip.stream_url != "")
        )).all()

        total = len(rows)
        updated = 0
        skipped = 0

        update_task_progress(task_id, {"stage": "probing", "total": total, "done": 0, "updated": 0})

        for i, row in enumerate(rows):
            # stream_url 形如 /media/xxx/clip_yyy.mp4 → 转为本地路径
            relative = row.stream_url.lstrip("/")          # media/xxx/clip_yyy.mp4
            if relative.startswith("media/"):
                relative = relative[len("media/"):]        # xxx/clip_yyy.mp4
            file_path = os.path.join(str(config.MEDIA_DIR), relative)

            if not os.path.exists(file_path):
                skipped += 1
                continue

            real_dur = _probe_duration(file_path)
            if real_dur is None:
                skipped += 1
                continue

            # 只在差距超过 0.1s 时才更新，避免无意义写入
            if row.duration is None or abs(real_dur - row.duration) > 0.1:
                await session.execute(
                    sql_update(Clip).where(Clip.id == row.id).values(duration=real_dur)
                )
                updated += 1

            if (i + 1) % 50 == 0:
                await session.commit()
                update_task_progress(task_id, {"stage": "probing", "total": total,
                                               "done": i + 1, "updated": updated})

        await session.commit()
        update_task_progress(task_id, {"stage": "done", "total": total,
                                       "done": total, "updated": updated, "skipped": skipped})
        return {"total": total, "updated": updated, "skipped": skipped}


@router.post("/backfill-durations")
async def backfill_clip_durations():
    """
    后台任务：用 ffprobe 读取所有 matched 切片的真实文件时长，批量更新 duration 字段。
    适用于存量数据修复（旧切片只存了 Whisper 片段时长，未含上下文窗口）。
    """
    job_id = str(uuid4())
    await submit_task(
        _run_backfill_durations(job_id),
        description="补填切片真实时长（ffprobe）",
        task_id=job_id,
    )
    return {"job_id": job_id, "message": "时长补填任务已提交，请通过任务队列查看进度"}
