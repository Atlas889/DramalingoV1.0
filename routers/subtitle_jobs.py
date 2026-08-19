"""
DramaLingo · 字幕处理任务路由

在内容管理模块下单独管理 TED 字幕获取任务：
  POST   /subtitle-jobs              新建任务（提交 TED URL，后台获取字幕）
  GET    /subtitle-jobs              列出所有任务（支持状态过滤与分页）
  GET    /subtitle-jobs/{id}         获取单个任务详情
  GET    /subtitle-jobs/{id}/preview 预览 SRT 文件内容
  GET    /subtitle-jobs/{id}/download 下载 SRT 文件
  POST   /subtitle-jobs/{id}/send    发送到字幕处理流水线
  DELETE /subtitle-jobs/{id}         删除任务（可选删除 SRT 文件）
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

import config
from database import get_session, SessionLocal
from models.subtitle_job import SubtitleJob
from tasks.task_runner import submit_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/subtitle-jobs", tags=["subtitle-jobs"])


# ─────────────────────────────────────────────────────────────────────────────
# 内部：fetch 任务包装（更新 DB 状态）
# ─────────────────────────────────────────────────────────────────────────────

async def _fetch_and_update(job_id_db: int, ted_url: str, title_en: str,
                             title_zh: str, task_id: str) -> dict:
    """
    执行 TED 字幕获取，完成后将结果写入 subtitle_jobs 表。
    fetch 失败时同样更新状态为 failed。
    """
    from services.ted_fetcher import fetch_ted_subtitle

    try:
        result = await fetch_ted_subtitle(
            ted_url=ted_url,
            title_en=title_en,
            title_zh=title_zh,
            deepseek_api_key=None,   # 内部自动回落到 config.AI_API_KEY
            task_id=task_id,
        )
        async with SessionLocal() as sess:
            job = await sess.get(SubtitleJob, job_id_db)
            if job:
                job.status        = "ready"
                job.srt_path      = result.get("srt_path")
                job.caption_count = result.get("caption_count")
                job.zh_source     = result.get("zh_source")
                job.updated_at    = datetime.now(timezone.utc)
                await sess.commit()
        return result

    except Exception as exc:
        async with SessionLocal() as sess:
            job = await sess.get(SubtitleJob, job_id_db)
            if job:
                job.status    = "failed"
                job.error_msg = str(exc)
                job.updated_at = datetime.now(timezone.utc)
                await sess.commit()
        raise


async def _pipeline_and_update(job_id_db: int, srt_path: str,
                                title_zh: str, title_en: str,
                                pipeline_task_id: str) -> dict:
    """
    执行字幕流水线（process_subtitle_file → sync_episode_to_knowledge），
    完成后将统计结果写入 subtitle_jobs 表。
    """
    from services.subtitle_pipeline import process_subtitle_file, sync_episode_to_knowledge

    async with SessionLocal() as sess:
        result = await process_subtitle_file(
            file_path=srt_path,
            show_name=title_zh,
            episode="",
            session=sess,
            en_name=title_en,
            content_type="talk",
        )
        if result.get("error"):
            async with SessionLocal() as sess2:
                job = await sess2.get(SubtitleJob, job_id_db)
                if job:
                    job.status    = "error"
                    job.error_msg = result["error"]
                    job.updated_at = datetime.now(timezone.utc)
                    await sess2.commit()
            return result

        sync_stats = await sync_episode_to_knowledge(
            show_name=title_zh,
            episode="",
            session=sess,
            en_name=title_en,
            content_type="talk",
            task_id=pipeline_task_id,
        )
        result.update(sync_stats)
        result["knowledge_sync"] = sync_stats

    # 更新任务状态
    async with SessionLocal() as sess:
        job = await sess.get(SubtitleJob, job_id_db)
        if job:
            job.status             = "done"
            job.new_nodes          = sync_stats.get("new_nodes", 0)
            job.ai_tagging_success = sync_stats.get("ai_tagging_success", 0)
            job.rule_fallback      = sync_stats.get("rule_fallback", 0)
            job.updated_at         = datetime.now(timezone.utc)
            await sess.commit()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# POST /subtitle-jobs  — 新建任务
# ─────────────────────────────────────────────────────────────────────────────

@router.post("")
async def create_subtitle_job(
    ted_url:  str = Form(..., description="TED 演讲 URL"),
    title_zh: str = Form(..., description="演讲中文标题"),
    title_en: str = Form("",  description="演讲英文标题（选填，可从 URL 自动解析）"),
    session: AsyncSession = Depends(get_session),
):
    """
    新建字幕处理任务：
    1. 在 subtitle_jobs 表中插入状态为 fetching 的记录
    2. 启动后台任务（yt-dlp 下载 + AI 翻译 + 写 SRT）
    3. 任务完成后自动更新 DB 记录状态为 ready / failed
    """
    if "ted.com/talks/" not in ted_url:
        raise HTTPException(
            status_code=400,
            detail="无效的 TED 链接，应为 https://www.ted.com/talks/... 格式",
        )

    # 若未提供英文标题，从 URL 解析
    if not title_en:
        import re
        m = re.search(r"ted\.com/talks/([^?#/]+)", ted_url)
        if m:
            title_en = m.group(1).replace("_", " ").title()

    task_id = str(uuid.uuid4())

    # 先插入 DB 记录
    job = SubtitleJob(
        ted_url   = ted_url,
        title_zh  = title_zh,
        title_en  = title_en,
        status    = "fetching",
        task_id   = task_id,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    job_id_db = job.id

    # 提交后台任务
    async def _task(_jid=task_id, _dbid=job_id_db):
        return await _fetch_and_update(
            job_id_db=_dbid,
            ted_url=ted_url,
            title_en=title_en,
            title_zh=title_zh,
            task_id=_jid,
        )

    await submit_task(
        _task(),
        description=f"TED字幕获取: {title_zh or ted_url[:40]}",
        task_id=task_id,
    )

    return {"job": job.to_dict(), "message": "任务已提交，正在后台获取字幕"}


# ─────────────────────────────────────────────────────────────────────────────
# GET /subtitle-jobs  — 列出所有任务
# ─────────────────────────────────────────────────────────────────────────────

@router.get("")
async def list_subtitle_jobs(
    status: Optional[str] = Query(None, description="按状态过滤"),
    page:   int           = Query(1,    ge=1),
    page_size: int        = Query(20,   ge=1, le=100),
    session: AsyncSession = Depends(get_session),
):
    stmt = select(SubtitleJob).order_by(SubtitleJob.created_at.desc())
    if status:
        stmt = stmt.where(SubtitleJob.status == status)

    count_stmt = select(func.count()).select_from(SubtitleJob)
    if status:
        count_stmt = count_stmt.where(SubtitleJob.status == status)
    total = (await session.execute(count_stmt)).scalar() or 0

    stmt = stmt.offset((page - 1) * page_size).limit(page_size)
    rows = (await session.execute(stmt)).scalars().all()

    return {
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "items":     [r.to_dict() for r in rows],
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /subtitle-jobs/{id}  — 单条详情
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{job_id}")
async def get_subtitle_job(
    job_id: int,
    session: AsyncSession = Depends(get_session),
):
    job = await session.get(SubtitleJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"字幕任务 #{job_id} 不存在")
    return job.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# GET /subtitle-jobs/{id}/preview  — SRT 内容预览
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{job_id}/preview")
async def preview_subtitle_job(
    job_id: int,
    session: AsyncSession = Depends(get_session),
):
    """返回 SRT 文件文本内容（用于前端预览，安全校验路径在 ted_srt 目录下）。"""
    job = await session.get(SubtitleJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"字幕任务 #{job_id} 不存在")
    if not job.srt_path:
        raise HTTPException(status_code=400, detail="SRT 文件尚未生成")

    allowed_dir = os.path.realpath(os.path.join(config.UPLOAD_DIR, "ted_srt"))
    real_path   = os.path.realpath(job.srt_path)
    if not real_path.startswith(allowed_dir):
        raise HTTPException(status_code=403, detail="路径不合法")
    if not os.path.isfile(real_path):
        raise HTTPException(status_code=404, detail="SRT 文件不存在（可能已被清理）")

    with open(real_path, encoding="utf-8") as f:
        content = f.read()
    return {"content": content, "filename": os.path.basename(real_path)}


# ─────────────────────────────────────────────────────────────────────────────
# GET /subtitle-jobs/{id}/download  — 下载 SRT 文件
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/{job_id}/download")
async def download_subtitle_job(
    job_id: int,
    session: AsyncSession = Depends(get_session),
):
    job = await session.get(SubtitleJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"字幕任务 #{job_id} 不存在")
    if not job.srt_path:
        raise HTTPException(status_code=400, detail="SRT 文件尚未生成")

    allowed_dir = os.path.realpath(os.path.join(config.UPLOAD_DIR, "ted_srt"))
    real_path   = os.path.realpath(job.srt_path)
    if not real_path.startswith(allowed_dir):
        raise HTTPException(status_code=403, detail="路径不合法")
    if not os.path.isfile(real_path):
        raise HTTPException(status_code=404, detail="SRT 文件不存在")

    return FileResponse(
        real_path,
        filename=os.path.basename(real_path),
        media_type="text/plain; charset=utf-8",
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /subtitle-jobs/{id}/send  — 发送到字幕处理流水线
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/{job_id}/send")
async def send_to_pipeline(
    job_id: int,
    session: AsyncSession = Depends(get_session),
):
    """
    将已就绪的 SRT 文件发送到字幕处理流水线（process_subtitle_file + sync_episode_to_knowledge）。
    只允许 status=ready 的任务执行此操作。
    """
    job = await session.get(SubtitleJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"字幕任务 #{job_id} 不存在")
    if job.status not in ("ready", "error"):
        raise HTTPException(
            status_code=400,
            detail=f"任务当前状态为 {job.status}，只有 ready 状态的任务才能发送到流水线",
        )
    if not job.srt_path or not os.path.isfile(job.srt_path):
        raise HTTPException(status_code=400, detail="SRT 文件不存在，无法处理")

    pipeline_task_id = str(uuid.uuid4())

    # 更新状态为 processing
    job.status           = "processing"
    job.pipeline_task_id = pipeline_task_id
    job.updated_at       = datetime.now(timezone.utc)
    await session.commit()

    srt_path = job.srt_path
    title_zh = job.title_zh
    title_en = job.title_en
    job_id_db = job.id

    async def _task(_ptid=pipeline_task_id, _dbid=job_id_db):
        return await _pipeline_and_update(
            job_id_db=_dbid,
            srt_path=srt_path,
            title_zh=title_zh,
            title_en=title_en,
            pipeline_task_id=_ptid,
        )

    await submit_task(
        _task(),
        description=f"TED字幕入库: {title_zh}",
        task_id=pipeline_task_id,
    )

    return {
        "pipeline_task_id": pipeline_task_id,
        "message":          f"字幕流水线任务已提交，正在处理《{title_zh}》",
    }


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /subtitle-jobs/{id}  — 删除任务
# ─────────────────────────────────────────────────────────────────────────────

@router.delete("/{job_id}")
async def delete_subtitle_job(
    job_id:      int,
    delete_file: bool = Query(False, description="同时删除本地 SRT 文件"),
    session: AsyncSession = Depends(get_session),
):
    job = await session.get(SubtitleJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"字幕任务 #{job_id} 不存在")

    if job.status in ("fetching", "processing"):
        raise HTTPException(
            status_code=400,
            detail="任务正在运行中，请等待完成后再删除",
        )

    srt_path = job.srt_path
    await session.delete(job)
    await session.commit()

    deleted_file = False
    if delete_file and srt_path and os.path.isfile(srt_path):
        try:
            os.remove(srt_path)
            deleted_file = True
        except OSError as e:
            logger.warning(f"删除 SRT 文件失败: {e}")

    return {"message": f"任务 #{job_id} 已删除", "deleted_file": deleted_file}
