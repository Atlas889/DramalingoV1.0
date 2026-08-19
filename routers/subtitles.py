"""
DramaLingo · 字幕管理路由
对应 PRD v0.9 §10.1 字幕上传 / §10.3 字幕记录

功能：
- 单文件/双语字幕上传（异步任务）
- 字幕处理任务状态查询
- 原始字幕记录列表查询与统计
- 剧集列表查询
"""

import logging
import os
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, distinct

import config
from database import get_session, SessionLocal
from models.subtitle import SubtitleRaw
from models.knowledge_node import KnowledgeNode
from tasks.task_runner import submit_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/subtitles", tags=["subtitles"])


@router.post("/upload")
async def upload_subtitle(
    zh_name: str = Form(..., description="中文剧名"),
    en_name: str = Form(..., description="英文剧名"),
    episode: str = Form("", description="集号（选填，如 S01E01）"),
    content_type: str = Form("tv", description="内容大类: tv | movie | talk"),
    overwrite: bool = Form(False),
    file: UploadFile = File(...),
):
    """
    单文件字幕上传接口。

    接收字幕文件并存入临时目录，随后将处理请求推入异步任务队列。
    支持 .srt, .ass, .ssa 格式。
    """
    from services.subtitle_pipeline import process_subtitle_file

    # 验证扩展名
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".srt", ".ass", ".ssa"]:
        raise HTTPException(status_code=400, detail=f"不支持的字幕格式: {ext}")

    subtitle_code = uuid.uuid4().hex[:8].upper()

    # 保存到临时目录
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    temp_filename = f"{uuid.uuid4()}{ext}"
    temp_path = os.path.join(config.UPLOAD_DIR, temp_filename)

    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        logger.error(f"文件保存失败: {e}")
        raise HTTPException(status_code=500, detail="文件保存失败")

    # 构造后台任务协程（使用独立 session，与请求生命周期解耦）
    async def _task():
        try:
            async with SessionLocal() as session:
                return await process_subtitle_file(
                    file_path=temp_path,
                    show_name=zh_name,
                    episode=episode,
                    session=session,
                    overwrite=overwrite,
                    en_name=en_name,
                    subtitle_code=subtitle_code,
                    content_type=content_type,
                )
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    _desc = f"{zh_name} {episode}".strip() if episode else zh_name
    job_id = await submit_task(_task(), description=_desc)

    return {
        "message": "字幕已上传，正在后台处理",
        "job_id": job_id,
        "subtitle_code": subtitle_code,
        "zh_name": zh_name,
        "en_name": en_name,
        "episode": episode,
    }


@router.post("/batch-upload")
async def batch_upload_subtitles(
    zh_name: str = Form(..., description="中文剧名"),
    en_name: str = Form(..., description="英文剧名"),
    episode_map: str = Form("{}", description="JSON 字符串: {文件名: 集号}，留空则所有文件用同一集号"),
    content_type: str = Form("tv", description="内容大类: tv | movie | talk"),
    overwrite: bool = Form(False),
    enable_ai: bool = Form(True, description="是否启用 AI 打标"),
    files: List[UploadFile] = File(...),
):
    """
    批量字幕上传接口（Step 6 新增）。

    支持一次上传多个字幕文件，并根据 episode_map 自动关联集号。
    执行完整流水线（含 Embedding + 去重 + AI 打标）。
    同一批次所有文件共享一个 subtitle_code。
    """
    import json
    from services.subtitle_pipeline import process_subtitle_batch

    stripped = (episode_map or "").strip()
    if not stripped:
        ep_map = {}
    else:
        try:
            ep_map = json.loads(stripped)
        except Exception:
            raise HTTPException(status_code=400, detail="episode_map 格式错误，应为 JSON 字符串，如 {\"ep01.srt\":\"S01E01\"}")

    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    saved_paths = []
    file_code_map: dict = {}  # temp_path → per-file subtitle_code

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".srt", ".ass", ".ssa"]:
            continue

        temp_filename = f"{uuid.uuid4()}{ext}"
        temp_path = os.path.join(config.UPLOAD_DIR, temp_filename)

        try:
            with open(temp_path, "wb") as f:
                f.write(await file.read())
        except Exception as e:
            logger.error(f"文件保存失败: {file.filename} - {e}")
            continue

        # 每个文件独立生成 subtitle_code
        file_code_map[temp_path] = uuid.uuid4().hex[:8].upper()
        saved_paths.append(temp_path)
        if file.filename in ep_map:
            ep_map[temp_path] = ep_map[file.filename]

    if not saved_paths:
        raise HTTPException(status_code=400, detail="没有有效的字幕文件")

    # 预生成 task_id，使 pipeline 能在运行中直接上报进度
    task_id = str(uuid.uuid4())

    async def _task():
        try:
            async with SessionLocal() as session:
                return await process_subtitle_batch(
                    file_paths=saved_paths,
                    show_name=zh_name,
                    episode_map=ep_map,
                    session=session,
                    overwrite=overwrite,
                    enable_ai_tagging=enable_ai,
                    en_name=en_name,
                    file_code_map=file_code_map,
                    task_id=task_id,
                    content_type=content_type,
                )
        finally:
            for _p in saved_paths:
                try:
                    os.unlink(_p)
                except OSError:
                    pass

    # 构造任务描述：剧名 + 集号列表（取 ep_map 里对应 saved_paths 的值）
    _ep_vals = [ep_map.get(p, "") for p in saved_paths]
    _ep_labels = [e for e in _ep_vals if e]
    if _ep_labels:
        _ep_str = " / ".join(_ep_labels[:4])
        if len(_ep_labels) > 4:
            _ep_str += f" 等{len(_ep_labels)}集"
        _batch_desc = f"{zh_name} · {_ep_str}"
    else:
        _batch_desc = f"{zh_name} · {len(saved_paths)}个文件"
    job_id = await submit_task(_task(), description=_batch_desc, task_id=task_id)

    # 返回所有文件的编码映射（按保存顺序）
    subtitle_codes = list(file_code_map.values())
    return {
        "message": f"已接收 {len(saved_paths)} 个文件，正在后台执行完整流水线",
        "job_id": job_id,
        "subtitle_code": subtitle_codes[0] if subtitle_codes else "",
        "subtitle_codes": subtitle_codes,
        "zh_name": zh_name,
        "en_name": en_name,
    }


@router.post("/upload-bilingual")
async def upload_bilingual_subtitles(
    zh_name: str = Form(..., description="中文剧名"),
    en_name: str = Form(..., description="英文剧名"),
    episode: str = Form("", description="集号（选填）"),
    overwrite: bool = Form(False),
    en_file: UploadFile = File(...),
    zh_file: UploadFile = File(...),
):
    """
    双语字幕上传接口。

    同时接收英文和中文字幕文件，后端将尝试按时间戳对齐。
    """
    from services.subtitle_pipeline import process_subtitle_file

    subtitle_code = uuid.uuid4().hex[:8].upper()

    os.makedirs(config.UPLOAD_DIR, exist_ok=True)

    # 保存英文文件
    en_ext = os.path.splitext(en_file.filename)[1].lower()
    en_temp = os.path.join(config.UPLOAD_DIR, f"{uuid.uuid4()}_en{en_ext}")
    try:
        with open(en_temp, "wb") as f:
            f.write(await en_file.read())
    except Exception as e:
        logger.error(f"英文字幕保存失败: {e}")
        raise HTTPException(status_code=500, detail="英文字幕文件保存失败")

    # 保存中文文件
    zh_ext = os.path.splitext(zh_file.filename)[1].lower()
    zh_temp = os.path.join(config.UPLOAD_DIR, f"{uuid.uuid4()}_zh{zh_ext}")
    try:
        with open(zh_temp, "wb") as f:
            f.write(await zh_file.read())
    except Exception as e:
        logger.error(f"中文字幕保存失败: {e}")
        os.unlink(en_temp)  # 回滚已保存的英文文件
        raise HTTPException(status_code=500, detail="中文字幕文件保存失败")

    # 构造后台任务协程：解析+筛选写库后自动同步到知识库
    async def _task():
        from services.subtitle_pipeline import sync_episode_to_knowledge
        async with SessionLocal() as session:
            result = await process_subtitle_file(
                file_path=en_temp,
                show_name=zh_name,
                episode=episode,
                session=session,
                zh_file=zh_temp,
                overwrite=overwrite,
                en_name=en_name,
                subtitle_code=subtitle_code,
            )
            # 解析成功且有通过筛选的行时，自动同步到知识库（与批量上传行为对齐）
            if not result.get("skipped") and not result.get("error") and result.get("passed", 0) > 0:
                sync_stats = await sync_episode_to_knowledge(
                    show_name=zh_name,
                    episode=episode,
                    session=session,
                    en_name=en_name,
                )
                result.update(sync_stats)
                result["knowledge_sync"] = sync_stats
            return result

    _bi_desc = f"{zh_name} {episode}".strip() if episode else zh_name
    job_id = await submit_task(_task(), description=_bi_desc)

    return {
        "message": "双语字幕已上传，正在后台处理",
        "job_id": job_id,
        "subtitle_code": subtitle_code,
        "zh_name": zh_name,
        "en_name": en_name,
        "episode": episode,
    }


@router.get("/check-exists")
async def check_subtitle_exists(
    zh_name: str = Query(..., description="中文剧名"),
    en_name: str = Query("", description="英文剧名（可选）"),
    episode: str = Query("", description="单集号（双语上传用）"),
    episodes: Optional[str] = Query(None, description="逗号分隔的集号列表（批量上传用），每个集号单独检测；传空字符串表示检测无集号文件"),
    session: AsyncSession = Depends(get_session),
):
    """
    检查指定剧名下是否已有字幕记录。
    - 批量上传：传 episodes（逗号分隔），按每个集号单独检测，只有完全匹配的集号才算冲突。
    - 双语上传：传 episode，精确匹配剧名+集号。
    """
    if episodes is not None:
        ep_list = [e.strip() for e in episodes.split(",")]
        conflicts = []
        for ep in ep_list:
            stmt = select(func.count(SubtitleRaw.id)).where(
                SubtitleRaw.show_name == zh_name,
                SubtitleRaw.episode == ep,
            )
            if en_name:
                stmt = stmt.where(SubtitleRaw.en_name == en_name)
            count = (await session.execute(stmt)).scalar() or 0
            if count > 0:
                conflicts.append({"episode": ep or "（无集号）", "count": count})
        return {
            "exists": len(conflicts) > 0,
            "count": sum(c["count"] for c in conflicts),
            "conflicts": conflicts,
            "zh_name": zh_name,
        }

    # 单集号精确检测（双语上传）
    stmt = select(func.count(SubtitleRaw.id)).where(SubtitleRaw.show_name == zh_name)
    if en_name:
        stmt = stmt.where(SubtitleRaw.en_name == en_name)
    if episode:
        stmt = stmt.where(SubtitleRaw.episode == episode)
    count = (await session.execute(stmt)).scalar() or 0
    return {"exists": count > 0, "count": count, "conflicts": [], "zh_name": zh_name}


@router.get("/shows")
async def list_subtitle_shows(
    content_type: Optional[str] = Query(None, description="内容大类: tv | movie | talk"),
    session: AsyncSession = Depends(get_session),
):
    """
    获取可上传视频的剧名/演讲列表，供视频上传联动选择。

    规则：
    - 以 SubtitleRaw.content_type 为权威（上传时明确指定，不受历史默认值影响）
    - 同时要求 KnowledgeNode 中存在对应 show_name 的记录（知识库中有实际内容）
    - 可按 content_type 过滤（tv / movie / talk）
    """
    # ── Step 1：从 SubtitleRaw 取有 subtitle_code 的唯一剧名（content_type 以此为准）
    # GROUP BY 保证每个 (show_name, content_type) 只返回一行，en_name 取字典序最小值
    sr_stmt = (
        select(
            SubtitleRaw.show_name,
            func.min(SubtitleRaw.en_name).label("en_name"),
            SubtitleRaw.content_type,
        )
        .where(SubtitleRaw.subtitle_code.isnot(None))
        .group_by(SubtitleRaw.show_name, SubtitleRaw.content_type)
        .order_by(SubtitleRaw.show_name)
    )
    if content_type:
        sr_stmt = sr_stmt.where(SubtitleRaw.content_type == content_type)
    sr_rows = (await session.execute(sr_stmt)).all()

    # 按 show_name 去重（极少数情况：同一剧名有不同 content_type 记录，取第一条）
    seen: dict = {}  # show_name -> {zh_name, en_name, content_type}
    for r in sr_rows:
        if r.show_name and r.show_name not in seen:
            seen[r.show_name] = {
                "zh_name": r.show_name,
                "en_name": r.en_name or "",
                "content_type": r.content_type or "tv",
            }

    # ── Step 2：单次 IN 查询替代 N+1，过滤无知识库记录的剧名 ─────────────────────
    if seen:
        kn_names = set(
            (await session.execute(
                select(KnowledgeNode.show_name)
                .where(KnowledgeNode.show_name.in_(seen.keys()))
                .distinct()
            )).scalars().all()
        )
    else:
        kn_names = set()

    shows = sorted(
        [info for name, info in seen.items() if name in kn_names],
        key=lambda x: x["zh_name"],
    )
    return {"shows": shows}


@router.get("/batches")
async def list_subtitle_batches(
    content_type: Optional[str] = Query(None, description="内容大类: tv | movie | talk"),
    zh_name: Optional[str] = Query(None, description="按剧名/演讲名过滤"),
    session: AsyncSession = Depends(get_session),
):
    """
    获取所有字幕批次列表（每个 subtitle_code 为一批）。
    供视频上传时选择关联的字幕批次。
    支持按 content_type 和 zh_name 过滤。
    """
    stmt = (
        select(
            SubtitleRaw.subtitle_code,
            SubtitleRaw.show_name,
            SubtitleRaw.en_name,
            SubtitleRaw.episode,
            SubtitleRaw.content_type,
            func.count(SubtitleRaw.id).label("line_count"),
            func.min(SubtitleRaw.created_at).label("created_at"),
        )
        .where(SubtitleRaw.subtitle_code.isnot(None))
        .group_by(
            SubtitleRaw.subtitle_code, SubtitleRaw.show_name,
            SubtitleRaw.en_name, SubtitleRaw.episode, SubtitleRaw.content_type,
        )
        .order_by(SubtitleRaw.show_name, SubtitleRaw.episode)
    )
    if zh_name:
        # zh_name 已唯一定位剧集，无需再按 content_type 过滤
        # （避免历史记录 content_type 字段错标导致漏出）
        stmt = stmt.where(SubtitleRaw.show_name == zh_name)
    elif content_type:
        stmt = stmt.where(SubtitleRaw.content_type == content_type)
    result = await session.execute(stmt)
    rows = result.all()

    return {
        "batches": [
            {
                "subtitle_code": r.subtitle_code,
                "zh_name": r.show_name,
                "en_name": r.en_name,
                "episode": r.episode or "",
                "content_type": r.content_type or "tv",
                "line_count": r.line_count,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "label": f"{r.show_name}（{r.en_name}）[{r.subtitle_code}]",
            }
            for r in rows
        ]
    }


@router.get("/list")
async def list_subtitles(
    show_name: Optional[str] = None,
    episode: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
):
    """
    查询原始字幕记录列表（subtitles_raw 表）。
    支持按剧名、集号、过滤状态筛选。
    """
    stmt = select(SubtitleRaw)
    if show_name:
        stmt = stmt.where(SubtitleRaw.show_name == show_name)
    if episode:
        stmt = stmt.where(SubtitleRaw.episode == episode)
    if status:
        stmt = stmt.where(SubtitleRaw.filter_status == status)
    
    # 分页与排序
    stmt = stmt.order_by(SubtitleRaw.id.desc()).offset((page - 1) * page_size).limit(page_size)
    
    result = await session.execute(stmt)
    rows = result.scalars().all()
    
    # 总数统计
    count_stmt = select(func.count(SubtitleRaw.id))
    if show_name:
        count_stmt = count_stmt.where(SubtitleRaw.show_name == show_name)
    if episode:
        count_stmt = count_stmt.where(SubtitleRaw.episode == episode)
    if status:
        count_stmt = count_stmt.where(SubtitleRaw.filter_status == status)
    
    total = (await session.execute(count_stmt)).scalar() or 0
    
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [
            {
                "id": r.id,
                "show_name": r.show_name,
                "episode": r.episode,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "en_text": r.en_text,
                "zh_text": r.zh_text,
                "word_count": r.word_count,
                "filter_status": r.filter_status,
                "quality_score": r.quality_score,
                "source": r.source,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


@router.get("/shows")
async def list_shows(session: AsyncSession = Depends(get_session)):
    """获取已上传字幕的所有剧名列表"""
    stmt = select(distinct(SubtitleRaw.show_name)).order_by(SubtitleRaw.show_name)
    result = await session.execute(stmt)
    shows = result.scalars().all()
    return {"shows": [s for s in shows if s]}


@router.get("/stats")
async def get_subtitle_stats(session: AsyncSession = Depends(get_session)):
    """获取字幕库整体统计信息（总行数、各状态分布）"""
    # 各状态分布
    stmt = select(SubtitleRaw.filter_status, func.count(SubtitleRaw.id)).group_by(SubtitleRaw.filter_status)
    result = await session.execute(stmt)
    stats = {row[0]: row[1] for row in result.all()}

    total = sum(stats.values())
    passed = stats.get("passed", 0)

    return {
        "total_lines": total,
        "passed_lines": passed,
        "filter_rate": round(1 - (passed / total), 4) if total > 0 else 0,
        "status_distribution": stats,
    }

