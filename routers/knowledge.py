"""
DramaLingo · 知识库管理 API 路由
对应 PRD v0.9 §6 知识库管理功能

提供以下接口分组：

  CRUD 接口：
    GET    /knowledge/nodes                  — 分页查询知识节点（支持多维筛选）
    GET    /knowledge/nodes/{id}             — 获取单节点详情（含来源列表）
    PATCH  /knowledge/nodes/{id}             — 编辑单节点（中英文台词、标签等）
    DELETE /knowledge/nodes/{id}             — 删除单节点（级联删除 node_sources）
    POST   /knowledge/nodes/batch-delete     — 批量删除
    POST   /knowledge/nodes/batch-update     — 批量更新标签（含批量弃用）

  统计接口：
    GET    /knowledge/status                 — 知识库整体统计（难度/场景/打标来源分布）
    GET    /knowledge/shows                  — 知识库剧集列表

  晋升接口（将被过滤字幕手动加入知识库）：
    POST   /knowledge/promote                — 单条晋升
    POST   /knowledge/batch-promote          — 批量晋升

  质量评估接口：
    POST   /knowledge/full-re-evaluate       — 全库质量分重评估（后台任务）
    POST   /knowledge/nodes/{id}/re-evaluate — 单节点质量分重评估
    POST   /knowledge/nodes/batch-re-evaluate— 批量节点质量分重评估

  弃用分析接口（deprecated_analyzer.py 驱动）：
    POST   /knowledge/analyze-deprecated     — 触发全量弃用节点模式分析，返回统计报告
    GET    /knowledge/analyze-deprecated     — 轻量快速摘要（采样 30 条），供管理后台实时展示

  同步 & 补填接口：
    POST   /knowledge/sync-passed            — 一键同步 subtitles_raw.passed → knowledge_nodes
    POST   /knowledge/backfill-zh-embeddings — 批量补填中文 Embedding（zh_embedding）
    POST   /knowledge/merge-adjacent         — 对已入库节点执行相邻分句合并

  打标接口：
    POST   /knowledge/retag                  — 批量重新 AI 打标（含 speech_act）
    POST   /knowledge/retag-speech-acts      — 专项补标 speech_act（NULL 节点）

重要设计约定：
  - 所有将节点设为 is_active=False 的路径均调用 _trigger_deprecated_analysis()，
    以 fire-and-forget 方式触发弃用模式后台分析（不阻塞响应）。
  - 晋升流程：SubtitleRaw → Embedding → 去重检查 → KnowledgeNode + NodeSource + FAISS 更新
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete, update
from pydantic import BaseModel

from sqlalchemy.dialects.sqlite import insert as sqlite_insert

import config
from database import get_session, SessionLocal
from models.knowledge_node import KnowledgeNode
from models.node_source import NodeSource
from models.subtitle import SubtitleRaw
from tasks.task_runner import submit_task
from services.faiss_index import rebuild_zh_index_from_db

import asyncio
import logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def _trigger_deprecated_analysis(source: str) -> None:
    """
    在任何将节点设为 is_active=False 的操作完成后调用。
    以 fire-and-forget 方式提交后台分析任务，不阻塞当前请求。
    覆盖路径：
      - batch-update        (管理后台批量弃用)
      - single-patch        (PATCH /nodes/{id} 手动弃用)
      - re-evaluate         (单节点重评估降级)
      - batch-re-evaluate   (批量重评估降级)
      - full-re-evaluate    (全库重评估降级)
    """
    try:
        from services.deprecated_analyzer import schedule_post_deprecation_analysis
        asyncio.create_task(
            schedule_post_deprecation_analysis(trigger_source=source, delay_seconds=3.0)
        )
    except Exception as e:
        logger.warning(f"弃用分析任务触发失败（不影响主操作）: {e}")


# ──────────────────────────────────────────────
# Pydantic 请求体定义
# ──────────────────────────────────────────────

class NodeUpdateRequest(BaseModel):
    """
    单节点编辑请求体。所有字段均为可选，仅更新非 None 的字段（exclude_none=True）。

    注意：
    - 修改 en_text 不会自动重算 Embedding，需手动重建 FAISS 索引
    - 将 is_active 设为 False 会自动触发后台弃用分析（fire-and-forget）
    """
    zh_text: Optional[str] = None          # 中文翻译（填写后自动关闭 needs_translation）
    en_text: Optional[str] = None          # 英文台词（修改后 Embedding 将与实际文本不一致，慎用）
    scene: Optional[str] = None            # 场景标签（daily / work / romance / conflict / humor）
    emotion: Optional[str] = None          # 情绪标签（neutral / happy / sad / angry / surprised）
    difficulty: Optional[str] = None       # 难度标签（beginner / intermediate / advanced）
    keywords: Optional[List[str]] = None   # 关键词列表（3–5 个）
    speech_act: Optional[str] = None       # 言语行为标签（25 种枚举，见 README speech_act 表）
    is_active: Optional[bool] = None       # 是否激活（False = 弃用，触发后台分析）


class BatchDeleteRequest(BaseModel):
    """批量删除请求体"""
    ids: List[int]  # 待删除的节点 ID 列表


class BatchUpdateRequest(BaseModel):
    """批量更新标签请求体。ids 指定目标节点，其余字段为待更新值。"""
    ids: List[int]
    scene: Optional[str] = None
    emotion: Optional[str] = None
    difficulty: Optional[str] = None
    is_active: Optional[bool] = None


# ──────────────────────────────────────────────
# GET /knowledge/nodes  — 分页查询知识节点
# ──────────────────────────────────────────────

@router.get("/nodes")
async def list_nodes(
    show_name: Optional[str] = Query(None, description="按剧名筛选"),
    episode: Optional[str] = Query(None, description="按集号筛选，如 S01E01"),
    difficulty: Optional[str] = Query(None, description="按难度筛选：beginner/intermediate/advanced"),
    scene: Optional[str] = Query(None, description="按场景筛选：daily/business/romance 等"),
    emotion: Optional[str] = Query(None, description="按情绪筛选：neutral/happy/sad/angry/surprised"),
    tag_source: Optional[str] = Query(None, description="按打标来源筛选：ai/rule/manual"),
    is_active: Optional[bool] = Query(None, description="是否只查询激活节点"),
    content_type: Optional[str] = Query(None, description="内容大类: tv | movie | talk"),
    keyword: Optional[str] = Query(None, description="全文搜索英文或中文台词（ILIKE 模糊匹配）"),
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(50, ge=1, le=200, description="每页条数，最大 200"),
    session: AsyncSession = Depends(get_session),
):
    """
    分页查询知识节点，支持多维筛选。

    筛选条件均为可选，多个条件之间为 AND 关系。
    keyword 会同时搜索 en_text 和 zh_text（不区分大小写）。

    Returns:
        {total, page, page_size, items: [节点列表]}
    """
    q = select(KnowledgeNode)

    # 逐条应用筛选条件（均为精确匹配，keyword 为模糊匹配）
    if show_name:
        q = q.where(KnowledgeNode.show_name == show_name)
    if episode:
        q = q.where(KnowledgeNode.episode == episode)
    if difficulty:
        q = q.where(KnowledgeNode.difficulty == difficulty)
    if scene:
        q = q.where(KnowledgeNode.scene == scene)
    if emotion:
        q = q.where(KnowledgeNode.emotion == emotion)
    if tag_source:
        q = q.where(KnowledgeNode.tag_source == tag_source)
    if is_active is not None:
        q = q.where(KnowledgeNode.is_active == is_active)
    if content_type:
        q = q.where(KnowledgeNode.content_type == content_type)
    if keyword:
        # 转义 LIKE 通配符，防止用户输入的 % / _ 被当作模式匹配符号
        kw_escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        kw = f"%{kw_escaped}%"
        q = q.where(
            (KnowledgeNode.en_text.ilike(kw, escape="\\")) |
            (KnowledgeNode.zh_text.ilike(kw, escape="\\"))
        )

    # 先查总数（用于分页计算），再查当前页数据
    count_q = select(func.count()).select_from(q.subquery())
    total = (await session.execute(count_q)).scalar_one()

    q = q.order_by(KnowledgeNode.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    rows = (await session.execute(q)).scalars().all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": [_node_to_dict(r) for r in rows],
    }


# ──────────────────────────────────────────────
# GET /knowledge/nodes/{node_id}  — 单条详情
# ──────────────────────────────────────────────

@router.get("/nodes/{node_id}")
async def get_node(
    node_id: int,
    session: AsyncSession = Depends(get_session),
):
    """
    获取单个知识节点详情，附带所有来源记录（node_sources）。

    来源记录记录了该节点在哪些剧集/集数中出现过，
    可用于追溯原始字幕和时间戳。

    Returns:
        节点详情字典，额外包含 sources 列表
    """
    node = await session.get(KnowledgeNode, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"节点 {node_id} 不存在")

    # 查询该节点的所有来源记录（可能来自多个剧集/集数）
    sources = (
        await session.execute(
            select(NodeSource).where(NodeSource.node_id == node_id)
        )
    ).scalars().all()

    result = _node_to_dict(node)
    result["sources"] = [
        {
            "id": s.id,
            "show_name": s.show_name,
            "episode": s.episode,
            "subtitle_id": s.subtitle_id,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in sources
    ]
    return result


# ──────────────────────────────────────────────
# PATCH /knowledge/nodes/{node_id}  — 单条编辑
# ──────────────────────────────────────────────

@router.patch("/nodes/{node_id}")
async def update_node(
    node_id: int,
    body: NodeUpdateRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    编辑单个知识节点，支持修改中英文台词、标签、激活状态等。

    特殊逻辑：
    - 若 zh_text 被填写（非空），自动将 needs_translation 设为 False，
      表示该节点不再需要翻译。
    - en_text 修改不会触发 Embedding 重新计算（需手动重建索引）。

    Returns:
        {success: True, node: 更新后的节点详情}
    """
    node = await session.get(KnowledgeNode, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"节点 {node_id} 不存在")

    # 仅更新请求体中非 None 的字段（exclude_none=True 过滤未提供的字段）
    update_data = body.model_dump(exclude_none=True)
    for field, value in update_data.items():
        setattr(node, field, value)

    # 若中文翻译已填写，关闭待翻译标记
    if body.zh_text is not None and body.zh_text.strip():
        node.needs_translation = False

    await session.commit()
    await session.refresh(node)

    # 若节点被手动设为弃用，触发后台弃用分析
    if body.is_active is False:
        _trigger_deprecated_analysis("patch-node")

    return {"success": True, "node": _node_to_dict(node)}


# ──────────────────────────────────────────────
# DELETE /knowledge/nodes/{node_id}  — 单条删除
# ──────────────────────────────────────────────

@router.delete("/nodes/{node_id}")
async def delete_node(
    node_id: int,
    session: AsyncSession = Depends(get_session),
):
    """
    删除单个知识节点，同时级联删除关联的 node_sources 记录。

    注意：删除后 FAISS 索引不会立即更新，需等待下次重建（
    当下线节点占比超过 FAISS_REBUILD_THRESHOLD 时自动触发全量重建）。

    Returns:
        {success: True, deleted_id: 节点 ID}
    """
    node = await session.get(KnowledgeNode, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"节点 {node_id} 不存在")

    # 先删除关联的来源记录（外键约束）
    await session.execute(
        delete(NodeSource).where(NodeSource.node_id == node_id)
    )
    await session.delete(node)
    await session.commit()
    return {"success": True, "deleted_id": node_id}


# ──────────────────────────────────────────────
# POST /knowledge/nodes/batch-delete  — 批量删除
# ──────────────────────────────────────────────

@router.post("/nodes/batch-delete")
async def batch_delete_nodes(
    body: BatchDeleteRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    批量删除知识节点，同时级联删除关联的 node_sources 记录。

    Returns:
        {success: True, deleted_count: 实际删除数量}
    """
    if not body.ids:
        raise HTTPException(status_code=400, detail="ids 不能为空")

    # 先删除所有关联的来源记录
    await session.execute(
        delete(NodeSource).where(NodeSource.node_id.in_(body.ids))
    )
    # 再批量删除节点，rowcount 为实际删除数量
    result = await session.execute(
        delete(KnowledgeNode).where(KnowledgeNode.id.in_(body.ids))
    )
    await session.commit()
    return {"success": True, "deleted_count": result.rowcount}


# ──────────────────────────────────────────────
# POST /knowledge/nodes/batch-update  — 批量更新标签
# ──────────────────────────────────────────────

@router.post("/nodes/batch-update")
async def batch_update_nodes(
    body: BatchUpdateRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    批量更新知识节点的标签字段（scene/emotion/difficulty/is_active）。

    仅更新请求体中非 None 的字段，ids 字段本身不参与更新。

    Returns:
        {success: True, updated_count: 实际更新数量}
    """
    if not body.ids:
        raise HTTPException(status_code=400, detail="ids 不能为空")

    # 提取待更新字段（排除 ids 和 None 值）
    update_data = body.model_dump(exclude_none=True)
    update_data.pop("ids")  # ids 是目标选择器，不是更新字段
    if not update_data:
        raise HTTPException(status_code=400, detail="没有提供需要更新的字段")

    result = await session.execute(
        update(KnowledgeNode)
        .where(KnowledgeNode.id.in_(body.ids))
        .values(**update_data)
    )
    await session.commit()
    updated_count = result.rowcount

    # 若本次操作包含弃用（is_active=False），异步触发弃用模式分析
    if update_data.get("is_active") is False and updated_count > 0:
        _trigger_deprecated_analysis("batch-update")

    return {"success": True, "updated_count": updated_count}


# ──────────────────────────────────────────────
# GET /knowledge/status  — 知识库统计
# ──────────────────────────────────────────────

@router.get("/status")
async def knowledge_status(
    session: AsyncSession = Depends(get_session),
):
    """
    知识库整体统计信息，用于管理后台统计面板。

    统计维度：
    - 总节点数、已翻译数、待翻译数、翻译率
    - 按难度分布（beginner/intermediate/advanced）
    - 按场景分布（daily/business/romance 等）
    - 按打标来源分布（ai/rule/manual）
    - 按剧名分布（降序排列）

    Returns:
        多维统计字典
    """
    # 活跃节点总数
    total = (await session.execute(
        select(func.count()).where(KnowledgeNode.is_active == True)
    )).scalar_one()

    # 待翻译节点数（needs_translation=True 表示 zh_text 为空）
    needs_trans = (await session.execute(
        select(func.count()).where(
            KnowledgeNode.is_active == True,
            KnowledgeNode.needs_translation == True
        )
    )).scalar_one()

    # 已有中文翻译的节点数（zh_text 非空且非 None）
    with_zh = (await session.execute(
        select(func.count()).where(
            KnowledgeNode.is_active == True,
            KnowledgeNode.zh_text != "",
            KnowledgeNode.zh_text.isnot(None),
        )
    )).scalar_one()

    # 按难度分组统计
    diff_rows = (await session.execute(
        select(KnowledgeNode.difficulty, func.count().label("cnt"))
        .where(KnowledgeNode.is_active == True)
        .group_by(KnowledgeNode.difficulty)
    )).all()

    # 按场景分组统计
    scene_rows = (await session.execute(
        select(KnowledgeNode.scene, func.count().label("cnt"))
        .where(KnowledgeNode.is_active == True)
        .group_by(KnowledgeNode.scene)
    )).all()

    # 按打标来源分组统计（ai/rule/manual）
    tag_rows = (await session.execute(
        select(KnowledgeNode.tag_source, func.count().label("cnt"))
        .where(KnowledgeNode.is_active == True)
        .group_by(KnowledgeNode.tag_source)
    )).all()

    # 按剧名分组统计（降序）
    show_rows = (await session.execute(
        select(KnowledgeNode.show_name, func.count().label("cnt"))
        .where(KnowledgeNode.is_active == True)
        .group_by(KnowledgeNode.show_name)
        .order_by(func.count().desc())
    )).all()

    return {
        "total_nodes": total,
        "with_zh_translation": with_zh,
        "needs_translation": needs_trans,
        "translation_rate": round(with_zh / total, 4) if total > 0 else 0,
        "by_difficulty": {r.difficulty: r.cnt for r in diff_rows},
        "by_scene": {r.scene: r.cnt for r in scene_rows},
        "by_tag_source": {r.tag_source: r.cnt for r in tag_rows},
        "by_show": [{"show_name": r.show_name, "count": r.cnt} for r in show_rows],
    }


# ──────────────────────────────────────────────
# GET /knowledge/shows  — 知识库剧集列表
# ──────────────────────────────────────────────

@router.get("/shows")
async def knowledge_shows(
    content_type: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    """
    获取知识库中的剧集列表（按剧名字母序排列）。
    可通过 content_type 过滤大类（tv / movie / talk）。

    Returns:
        [{show_name, count}, ...] 列表
    """
    stmt = (
        select(KnowledgeNode.show_name, func.count().label("cnt"))
        .where(KnowledgeNode.is_active == True)
        .group_by(KnowledgeNode.show_name)
        .order_by(KnowledgeNode.show_name)
    )
    if content_type:
        stmt = stmt.where(KnowledgeNode.content_type == content_type)
    rows = (await session.execute(stmt)).all()
    return [{"show_name": r.show_name, "count": r.cnt} for r in rows]


# ──────────────────────────────────────────────
# GET /knowledge/episodes  — 某剧的集号列表
# ──────────────────────────────────────────────

@router.get("/episodes")
async def knowledge_episodes(
    show_name: str = Query(..., description="剧名"),
    session: AsyncSession = Depends(get_session),
):
    """返回指定剧名下知识库中已有的集号列表（按集号升序）。"""
    rows = (await session.execute(
        select(KnowledgeNode.episode, func.count().label("cnt"))
        .where(KnowledgeNode.show_name == show_name)
        .group_by(KnowledgeNode.episode)
        .order_by(KnowledgeNode.episode)
    )).all()
    return [{"episode": r.episode, "count": r.cnt} for r in rows]


# ──────────────────────────────────────────────
# 内部工具函数
# ──────────────────────────────────────────────

def _node_to_dict(node: KnowledgeNode) -> dict:
    """
    将 KnowledgeNode ORM 对象序列化为字典（供 API 返回使用）。

    注意：embedding 字段（BLOB）不包含在返回结果中，以减少传输体积。
    """
    return {
        "id": node.id,
        "code": node.code or "",
        "en_text": node.en_text,
        "zh_text": node.zh_text or "",
        "show_name": node.show_name,
        "episode": node.episode,
        "start_time": node.start_time,       # 字幕文件中的开始时间戳（秒）
        "end_time": node.end_time,           # 字幕文件中的结束时间戳（秒）
        "word_count": node.word_count,
        "occurrence_count": node.occurrence_count,  # 跨剧集出现次数（用于重排序）
        "scene": node.scene,
        "emotion": node.emotion,
        "difficulty": node.difficulty,
        "keywords": node.keywords or [],
        "quality_score": node.quality_score,
        "tag_source": node.tag_source,       # ai/rule/manual
        "needs_translation": node.needs_translation,
        "has_any_clip": node.has_any_clip,   # 是否已有对应视频切片
        "content_type": node.content_type or "tv",
        "is_active": node.is_active,
        "created_at": node.created_at.isoformat() if node.created_at else None,
    }


# ──────────────────────────────────────────────
# POST /knowledge/promote  — 将被过滤字幕手动加入知识库（单条）
# ──────────────────────────────────────────────
class PromoteRequest(BaseModel):
    """单条晋升请求体"""
    subtitle_id: int                          # 待晋升的 SubtitleRaw ID
    zh_text: Optional[str] = None            # 可选：手动提供中文翻译
    scene: Optional[str] = "daily"           # 场景标签（默认 daily）
    emotion: Optional[str] = "neutral"       # 情绪标签（默认 neutral）
    difficulty: Optional[str] = "intermediate"  # 难度标签（默认 intermediate）
    keywords: Optional[List[str]] = []       # 关键词列表

@router.post("/promote")
async def promote_subtitle_to_knowledge(
    req: PromoteRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    将一条被过滤掉的字幕行手动加入知识库（单条操作）。

    晋升流程：
    1. 从 subtitles_raw 中读取该字幕行
    2. 检查是否已有对应知识节点（幂等保护）
    3. Embedding 向量化
    4. 加载 FAISS 索引，执行去重检查
    5. 若与已有节点相似：累加 occurrence_count，写入 node_sources
    6. 若为新节点：规则打标 → 写入 knowledge_nodes + node_sources
    7. 更新字幕行状态为 promoted
    8. 增量更新 FAISS 索引

    Returns:
        {action: "created"|"merged", node_id, message}
    """
    from sqlalchemy import update as sa_update
    from models.subtitle import SubtitleRaw
    from models.node_source import NodeSource
    from services.embedder import encode_texts, serialize_vector
    from services.faiss_index import (
        load_index, rebuild_index_from_db, add_to_index, save_index,
        get_global_index, set_global_index
    )
    from services.deduplicator import check_duplicate, refresh_text_cache, add_to_text_cache
    from services.ai_tagger import rule_based_tag
    import config, numpy as np

    # ── 1. 查询字幕行 ──
    subtitle = (await session.execute(
        select(SubtitleRaw).where(SubtitleRaw.id == req.subtitle_id)
    )).scalar_one_or_none()
    if not subtitle:
        raise HTTPException(status_code=404, detail=f"字幕行 {req.subtitle_id} 不存在")
    if not subtitle.en_text:
        raise HTTPException(status_code=400, detail="该字幕行无英文内容，无法加入知识库")

    # ── 2. 幂等检查：该字幕行是否已有对应知识节点 ──
    existing_node = (await session.execute(
        select(KnowledgeNode).where(KnowledgeNode.subtitle_id == req.subtitle_id)
    )).scalar_one_or_none()
    if existing_node:
        raise HTTPException(
            status_code=409,
            detail=f"该字幕行已对应知识节点 id={existing_node.id}，无需重复加入"
        )

    # ── 3. Embedding 向量化 ──
    embeddings = encode_texts([subtitle.en_text])
    embedding = embeddings[0]

    # ── 4. 加载 FAISS 索引，刷新去重缓存 ──
    index_path = str(config.INDEX_DIR / "knowledge.index")
    try:
        index, mapping = load_index(index_path)
    except FileNotFoundError:
        # 索引文件不存在时，从数据库全量重建
        index, mapping = await rebuild_index_from_db(session, force_full=True, index_path=index_path)
    await refresh_text_cache(session)

    # ── 5. 去重检查（即使被过滤，也可能与已有节点语义重复）──
    is_dup, existing_node_id = await check_duplicate(subtitle.en_text, embedding, session, index, mapping)
    if is_dup and existing_node_id:
        # 已有相似节点：累加出现次数，写入来源记录
        await session.execute(
            sa_update(KnowledgeNode)
            .where(KnowledgeNode.id == existing_node_id)
            .values(occurrence_count=KnowledgeNode.occurrence_count + 1)
        )
        # 唯一约束冲突时静默忽略（同节点同集已有来源记录）
        await session.execute(
            sqlite_insert(NodeSource).values(
                node_id=existing_node_id,
                subtitle_id=subtitle.id,
                show_name=subtitle.show_name,
                episode=subtitle.episode,
            ).on_conflict_do_nothing()
        )
        # 更新字幕行状态为 promoted
        await session.execute(
            sa_update(SubtitleRaw)
            .where(SubtitleRaw.id == req.subtitle_id)
            .values(filter_status="promoted")
        )
        await session.commit()
        return {
            "action": "merged",
            "node_id": existing_node_id,
            "message": f"与已有节点 {existing_node_id} 相似度高，已合并（occurrence_count+1）"
        }

    # ── 6. 新节点：规则打标（手动加入时不调用 AI，避免延迟）──
    tags = rule_based_tag(subtitle.en_text, subtitle.word_count or len(subtitle.en_text.split()))
    # 优先使用请求体中的中文翻译，其次使用字幕文件中的中文
    zh_text_final = req.zh_text or subtitle.zh_text or ""

    node = KnowledgeNode(
        subtitle_id=subtitle.id,
        en_text=subtitle.en_text,
        zh_text=zh_text_final,
        show_name=subtitle.show_name,
        episode=subtitle.episode,
        start_time=subtitle.start_time,
        end_time=subtitle.end_time,
        word_count=subtitle.word_count or len(subtitle.en_text.split()),
        occurrence_count=1,
        scene=req.scene or tags.get("scene", "daily"),
        emotion=req.emotion or tags.get("emotion", "neutral"),
        difficulty=req.difficulty or tags.get("difficulty", "intermediate"),
        keywords=req.keywords or [],
        embedding=serialize_vector(embedding),
        quality_score=subtitle.quality_score,
        needs_translation=not bool(zh_text_final),
        tag_source="manual",  # 手动晋升标记为 manual
        is_active=True,
    )
    session.add(node)
    await session.flush()
    await session.refresh(node)

    # 写入来源记录
    ns = NodeSource(
        node_id=node.id,
        subtitle_id=subtitle.id,
        show_name=subtitle.show_name,
        episode=subtitle.episode,
    )
    session.add(ns)
    await session.flush()

    # 更新字幕行状态为 promoted
    await session.execute(
        sa_update(SubtitleRaw)
        .where(SubtitleRaw.id == req.subtitle_id)
        .values(filter_status="promoted")
    )
    await session.commit()

    # ── 7. 增量更新 FAISS 索引 ──
    add_to_text_cache(node.id, subtitle.en_text)
    new_embs = np.array([embedding], dtype="float32")
    add_to_index(index, mapping, [node.id], new_embs)
    save_index(index, mapping, index_path)
    set_global_index(index, mapping)

    return {
        "action": "created",
        "node_id": node.id,
        "en_text": node.en_text,
        "zh_text": node.zh_text,
        "tag_source": "manual",
        "message": f"已成功将字幕行 {req.subtitle_id} 加入知识库，节点 id={node.id}"
    }


# ──────────────────────────────────────────────
# POST /knowledge/batch-promote  — 批量将被过滤字幕加入知识库
# ──────────────────────────────────────────────
class BatchPromoteRequest(BaseModel):
    """批量晋升请求体"""
    subtitle_ids: List[int]                      # 待晋升的字幕行 ID 列表
    difficulty: Optional[str] = "intermediate"  # 统一设置难度标签
    scene: Optional[str] = "daily"              # 统一设置场景标签

@router.post("/batch-promote")
async def batch_promote_subtitles(
    req: BatchPromoteRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    批量将被过滤字幕行加入知识库。

    对每条字幕执行：Embedding → 去重检查 → 写入知识节点。
    最后批量更新 FAISS 索引（减少 I/O 次数）。

    Returns:
        {total, created, merged, skipped, not_found, results: [每条处理结果]}
    """
    from sqlalchemy import update as sa_update
    from models.subtitle import SubtitleRaw
    from models.node_source import NodeSource
    from services.embedder import encode_texts, serialize_vector
    from services.faiss_index import (
        load_index, rebuild_index_from_db, add_to_index, save_index, set_global_index
    )
    from services.deduplicator import check_duplicate, refresh_text_cache, add_to_text_cache
    from services.ai_tagger import rule_based_tag
    import config, numpy as np

    if not req.subtitle_ids:
        raise HTTPException(status_code=400, detail="subtitle_ids 不能为空")

    # 批量查询字幕行（一次 IN 查询，避免 N+1）
    subtitles = (await session.execute(
        select(SubtitleRaw).where(SubtitleRaw.id.in_(req.subtitle_ids))
    )).scalars().all()

    subtitle_map = {s.id: s for s in subtitles}
    # 记录请求中存在但数据库中找不到的 ID
    not_found = [sid for sid in req.subtitle_ids if sid not in subtitle_map]

    # 加载 FAISS 索引（整个批量操作共用同一个索引实例）
    index_path = str(config.INDEX_DIR / "knowledge.index")
    try:
        index, mapping = load_index(index_path)
    except FileNotFoundError:
        index, mapping = await rebuild_index_from_db(session, force_full=True, index_path=index_path)
    await refresh_text_cache(session)

    results = []
    new_node_embeddings = []  # 收集新节点 embedding，用于批量更新 FAISS
    new_node_ids = []

    # 预过滤：筛选出可处理的字幕行，并批量向量化（一次模型调用）
    valid_sids = []
    for sid in req.subtitle_ids:
        subtitle = subtitle_map.get(sid)
        if not subtitle:
            results.append({"subtitle_id": sid, "action": "error", "message": "字幕行不存在"})
            continue
        if not subtitle.en_text:
            results.append({"subtitle_id": sid, "action": "skip", "message": "无英文内容"})
            continue
        existing_node = (await session.execute(
            select(KnowledgeNode).where(KnowledgeNode.subtitle_id == sid)
        )).scalar_one_or_none()
        if existing_node:
            results.append({
                "subtitle_id": sid, "action": "skip",
                "message": f"已有知识节点 id={existing_node.id}"
            })
            continue
        valid_sids.append(sid)

    # 批量向量化（一次 SentenceTransformer 调用，替代逐条调用）
    embedding_map = {}
    if valid_sids:
        texts = [subtitle_map[sid].en_text for sid in valid_sids]
        batch_embeddings = encode_texts(texts)
        embedding_map = {sid: batch_embeddings[i] for i, sid in enumerate(valid_sids)}

    for sid in valid_sids:
        subtitle = subtitle_map[sid]
        embedding = embedding_map[sid]

        # 去重检查
        is_dup, existing_node_id = await check_duplicate(subtitle.en_text, embedding, session, index, mapping)
        if is_dup and existing_node_id:
            # 合并到已有节点
            await session.execute(
                sa_update(KnowledgeNode)
                .where(KnowledgeNode.id == existing_node_id)
                .values(occurrence_count=KnowledgeNode.occurrence_count + 1)
            )
            await session.execute(
                sqlite_insert(NodeSource).values(
                    node_id=existing_node_id, subtitle_id=sid,
                    show_name=subtitle.show_name, episode=subtitle.episode,
                ).on_conflict_do_nothing()
            )
            await session.execute(
                sa_update(SubtitleRaw).where(SubtitleRaw.id == sid).values(filter_status="promoted")
            )
            results.append({"subtitle_id": sid, "action": "merged", "node_id": existing_node_id})
            continue

        # 新节点：规则打标
        tags = rule_based_tag(subtitle.en_text, subtitle.word_count or len(subtitle.en_text.split()))
        node = KnowledgeNode(
            subtitle_id=subtitle.id,
            en_text=subtitle.en_text,
            zh_text=subtitle.zh_text or "",
            show_name=subtitle.show_name,
            episode=subtitle.episode,
            start_time=subtitle.start_time,
            end_time=subtitle.end_time,
            word_count=subtitle.word_count or len(subtitle.en_text.split()),
            occurrence_count=1,
            scene=req.scene or "daily",
            emotion="neutral",
            difficulty=req.difficulty or "intermediate",
            keywords=[],
            embedding=serialize_vector(embedding),
            quality_score=subtitle.quality_score,
            needs_translation=not bool(subtitle.zh_text),
            tag_source="manual",
            is_active=True,
        )
        session.add(node)
        await session.flush()
        await session.refresh(node)

        ns = NodeSource(node_id=node.id, subtitle_id=sid,
                        show_name=subtitle.show_name, episode=subtitle.episode)
        session.add(ns)
        await session.flush()

        await session.execute(
            sa_update(SubtitleRaw).where(SubtitleRaw.id == sid).values(filter_status="promoted")
        )

        # 收集新节点 embedding，延迟到批量操作结束后统一更新 FAISS
        add_to_text_cache(node.id, subtitle.en_text)
        new_node_embeddings.append(embedding)
        new_node_ids.append(node.id)
        results.append({"subtitle_id": sid, "action": "created", "node_id": node.id})

    await session.commit()

    # 批量更新 FAISS 索引（一次 I/O，比逐条更新更高效）
    if new_node_ids:
        new_embs = np.stack(new_node_embeddings).astype("float32")
        add_to_index(index, mapping, new_node_ids, new_embs)
        save_index(index, mapping, index_path)
        set_global_index(index, mapping)

    # 汇总统计
    created = sum(1 for r in results if r["action"] == "created")
    merged = sum(1 for r in results if r["action"] == "merged")
    skipped = sum(1 for r in results if r["action"] in ("skip", "error"))

    return {
        "total": len(req.subtitle_ids),
        "created": created,
        "merged": merged,
        "skipped": skipped,
        "not_found": not_found,
        "results": results,
    }


# ──────────────────────────────────────────────
# POST /knowledge/sync-passed  — 一键同步所有 passed 字幕到知识库
# ──────────────────────────────────────────────

@router.post("/sync-passed")
async def sync_passed_to_knowledge(
    show_name: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    """
    将所有 filter_status='passed' 且尚无对应知识节点的字幕行，
    以后台任务方式批量同步到知识库。

    适用场景：
    - 字幕流水线中途失败，passed 字幕已入库但知识节点未创建
    - 使用单文件/双语上传接口（不含 embedding 步骤）上传的字幕
    - 历史遗留数据修复

    Args:
        show_name: 可选，只同步指定剧名的字幕（为空时同步全部）

    Returns:
        job_id + 待同步条数
    """
    # 查询 passed 且无直接知识节点链接的字幕 ID
    synced_ids_subq = select(KnowledgeNode.subtitle_id).where(
        KnowledgeNode.subtitle_id.isnot(None)
    )
    stmt = select(SubtitleRaw.id).where(
        SubtitleRaw.filter_status == "passed",
        SubtitleRaw.id.notin_(synced_ids_subq),
    )
    if show_name:
        stmt = stmt.where(SubtitleRaw.show_name == show_name)

    result = await session.execute(stmt)
    unsynced_ids = [row[0] for row in result.all()]

    if not unsynced_ids:
        return {
            "job_id": None,
            "pending": 0,
            "message": "没有需要同步的字幕，知识库已是最新状态",
        }

    # 提交后台任务（复用 batch_promote 核心逻辑）
    async def _sync_task():
        from sqlalchemy import update as sa_update
        import numpy as np
        import config as _cfg
        from services.embedder import encode_texts, serialize_vector
        from services.faiss_index import load_index, rebuild_index_from_db, add_to_index, save_index, set_global_index
        from services.deduplicator import check_duplicate, refresh_text_cache, add_to_text_cache
        from services.ai_tagger import rule_based_tag

        async with SessionLocal() as task_session:
            # 批量查询字幕行
            subtitles = (await task_session.execute(
                select(SubtitleRaw).where(SubtitleRaw.id.in_(unsynced_ids))
            )).scalars().all()
            subtitle_map = {s.id: s for s in subtitles}

            index_path = str(_cfg.INDEX_DIR / "knowledge.index")
            try:
                index, mapping = load_index(index_path)
            except FileNotFoundError:
                index, mapping = await rebuild_index_from_db(task_session, force_full=True, index_path=index_path)
            await refresh_text_cache(task_session)

            # 批量 embedding
            valid_ids = [sid for sid in unsynced_ids if subtitle_map.get(sid) and subtitle_map[sid].en_text]
            texts = [subtitle_map[sid].en_text for sid in valid_ids]
            embeddings = encode_texts(texts)
            embedding_map = {sid: embeddings[i] for i, sid in enumerate(valid_ids)}

            new_node_ids, new_node_embeddings = [], []
            created = merged = 0

            for sid in valid_ids:
                subtitle = subtitle_map[sid]
                embedding = embedding_map[sid]

                is_dup, existing_id = await check_duplicate(subtitle.en_text, embedding, task_session, index, mapping)
                if is_dup and existing_id:
                    await task_session.execute(
                        sa_update(KnowledgeNode)
                        .where(KnowledgeNode.id == existing_id)
                        .values(occurrence_count=KnowledgeNode.occurrence_count + 1)
                    )
                    await task_session.execute(
                        sqlite_insert(NodeSource).values(
                            node_id=existing_id, subtitle_id=sid,
                            show_name=subtitle.show_name, episode=subtitle.episode,
                        ).on_conflict_do_nothing()
                    )
                    await task_session.execute(
                        sa_update(SubtitleRaw).where(SubtitleRaw.id == sid).values(filter_status="promoted")
                    )
                    merged += 1
                    continue

                tags = rule_based_tag(subtitle.en_text, subtitle.word_count or len(subtitle.en_text.split()))
                node = KnowledgeNode(
                    subtitle_id=subtitle.id,
                    en_text=subtitle.en_text,
                    zh_text=subtitle.zh_text or "",
                    show_name=subtitle.show_name,
                    episode=subtitle.episode,
                    start_time=subtitle.start_time,
                    end_time=subtitle.end_time,
                    word_count=subtitle.word_count or len(subtitle.en_text.split()),
                    occurrence_count=1,
                    scene=tags["scene"],
                    emotion=tags["emotion"],
                    difficulty=tags["difficulty"],
                    keywords=tags["keywords"],
                    embedding=serialize_vector(embedding),
                    quality_score=subtitle.quality_score,
                    needs_translation=not bool(subtitle.zh_text),
                    tag_source="rule",
                    is_active=True,
                )
                task_session.add(node)
                await task_session.flush()
                await task_session.refresh(node)

                task_session.add(NodeSource(
                    node_id=node.id, subtitle_id=sid,
                    show_name=subtitle.show_name, episode=subtitle.episode,
                ))
                await task_session.flush()
                await task_session.execute(
                    sa_update(SubtitleRaw).where(SubtitleRaw.id == sid).values(filter_status="promoted")
                )
                add_to_text_cache(node.id, subtitle.en_text)
                new_node_ids.append(node.id)
                new_node_embeddings.append(embedding)
                created += 1

            await task_session.commit()

            if new_node_ids:
                add_to_index(index, mapping, new_node_ids, np.stack(new_node_embeddings).astype("float32"))
                save_index(index, mapping, index_path)
                set_global_index(index, mapping)

            return {"created": created, "merged": merged, "total": len(valid_ids)}

    show_desc = show_name or "全部剧名"
    job_id = await submit_task(_sync_task(), description=f"同步 passed 字幕到知识库: {show_desc} ({len(unsynced_ids)} 条)")

    return {
        "job_id": job_id,
        "pending": len(unsynced_ids),
        "message": f"已提交同步任务，共 {len(unsynced_ids)} 条字幕待处理，请通过 /jobs/{job_id} 查询进度",
    }


# ──────────────────────────────────────────────
# POST /knowledge/retag  — 批量重新打标知识节点
# ──────────────────────────────────────────────

@router.post("/retag")
async def retag_knowledge_nodes(
    show_name: Optional[str] = Query(None, description="指定剧名，为空时处理全库"),
    episode: Optional[str] = Query(None, description="指定集号，为空时处理该剧所有集（需同时传 show_name）"),
    only_rule_tagged: bool = Query(True, description="True=仅重标 tag_source=rule 的节点；False=全部重标"),
    session: AsyncSession = Depends(get_session),
):
    """
    批量重新用 AI 对知识节点打标（scene / emotion / difficulty / keywords / tag_source）。

    适用场景：
    - 历史数据全部使用规则打标（tag_source='rule'），需补做 AI 打标
    - 修改了打标 Prompt 或枚举定义后，刷新历史标签
    - AI_API_KEY 初始未配置，配置后补跑 AI 打标

    参数：
    - show_name: 指定剧名（空=全库）
    - only_rule_tagged: True 仅处理规则打标节点，False 重标全部（含已 AI 打标的）
    """
    stmt = select(KnowledgeNode.id, KnowledgeNode.en_text, KnowledgeNode.word_count)
    if only_rule_tagged:
        stmt = stmt.where(KnowledgeNode.tag_source == "rule")
    if show_name:
        stmt = stmt.where(KnowledgeNode.show_name == show_name)
    if episode:
        stmt = stmt.where(KnowledgeNode.episode == episode)

    result = await session.execute(stmt)
    nodes = result.all()  # [(id, en_text, word_count), ...]

    if not nodes:
        scope_parts = ["规则打标节点" if only_rule_tagged else "所有节点"]
        if show_name:
            scope_parts.append(show_name)
        if episode:
            scope_parts.append(episode)
        return {"job_id": None, "pending": 0, "message": f"没有符合条件的节点（{'·'.join(scope_parts)}）"}

    import uuid as _uuid
    task_id = str(_uuid.uuid4())
    node_list = [{"id": r[0], "en_text": r[1], "word_count": r[2] or len((r[1] or "").split())} for r in nodes]

    async def _retag_task():
        from services.ai_tagger import batch_tag_and_translate, rule_based_tag
        from tasks.task_runner import update_task_progress

        _total = len(node_list)
        ai_ok = rule_fb = 0
        _CHUNK = 20  # 每批 20 条，平衡并发与进度粒度

        async with SessionLocal() as task_session:
            for chunk_start in range(0, _total, _CHUNK):
                chunk = node_list[chunk_start: chunk_start + _CHUNK]
                pct = int(100 * chunk_start / _total)

                if task_id:
                    update_task_progress(task_id, {
                        "stage": "retagging",
                        "label": f"打标进度 {chunk_start}/{_total}",
                        "pct": pct,
                        "done": chunk_start,
                        "total": _total,
                    })

                try:
                    all_tags, _ = await batch_tag_and_translate(
                        texts=[n["en_text"] for n in chunk],
                        word_counts=[n["word_count"] for n in chunk],
                        max_concurrency=5,
                    )
                except Exception:
                    all_tags = [rule_based_tag(n["en_text"], n["word_count"]) for n in chunk]

                for node_info, tags in zip(chunk, all_tags):
                    await task_session.execute(
                        update(KnowledgeNode)
                        .where(KnowledgeNode.id == node_info["id"])
                        .values(
                            scene=tags["scene"],
                            emotion=tags["emotion"],
                            difficulty=tags["difficulty"],
                            keywords=tags["keywords"],
                            speech_act=tags.get("speech_act", "neutral_statement"),
                            tag_source=tags["tag_source"],
                        )
                    )
                    if tags.get("tag_source") == "ai":
                        ai_ok += 1
                    else:
                        rule_fb += 1

                await task_session.commit()

            if task_id:
                update_task_progress(task_id, {
                    "stage": "done",
                    "label": f"打标完成：AI {ai_ok} 条，规则降级 {rule_fb} 条",
                    "pct": 100,
                    "done": _total,
                    "total": _total,
                })

        return {"total": _total, "ai_tagged": ai_ok, "rule_fallback": rule_fb}

    scope_parts = ["规则打标" if only_rule_tagged else "全量重标"]
    scope_parts.append(show_name if show_name else "全库")
    if episode:
        scope_parts.append(episode)
    scope_desc = "·".join(scope_parts)
    job_id = await submit_task(
        _retag_task(),
        description=f"批量重新打标: {scope_desc} ({len(node_list)} 条)",
        task_id=task_id,
    )

    return {
        "job_id": job_id,
        "pending": len(node_list),
        "message": f"已提交重新打标任务，共 {len(node_list)} 条节点，Job ID: {job_id}",
    }


# ──────────────────────────────────────────────
# POST /knowledge/merge-adjacent
# 对已入库的知识节点执行相邻分句合并
# ──────────────────────────────────────────────

@router.get("/merge-adjacent/preview")
async def preview_merge_adjacent(
    show_name: Optional[str] = Query(None),
    episode: Optional[str] = Query(None),
    gap_threshold: float = Query(1.0, description="相邻字幕最大间隔（秒）"),
    session: AsyncSession = Depends(get_session),
):
    """
    预览合并结果，不修改数据库。
    返回：候选对数量、各条件阻断数量、示例合并对。
    """
    from services.subtitle_parser import (
        _is_continuation, _cosine_sim, _ELLIPSIS_RE, _DIALOGUE_DASH_RE, _MERGE_MAX_WORDS,
    )
    from services.embedder import deserialize_vector

    q = (
        select(KnowledgeNode)
        .where(KnowledgeNode.is_active == True)
        .order_by(KnowledgeNode.show_name, KnowledgeNode.episode, KnowledgeNode.start_time)
    )
    if show_name:
        q = q.where(KnowledgeNode.show_name == show_name)
    if episode:
        q = q.where(KnowledgeNode.episode == episode)
    nodes = (await session.execute(q)).scalars().all()

    def _safe_deserialize(blob):
        try:
            return deserialize_vector(blob) if blob else None
        except Exception:
            return None

    node_dicts = [
        {
            "id": n.id,
            "show_name": n.show_name,
            "episode": n.episode,
            "start_time": n.start_time,
            "end_time": n.end_time,
            "en_text": n.en_text or "",
            "zh_text": n.zh_text or "",
            "emb": _safe_deserialize(n.embedding),
        }
        for n in nodes
    ]

    total_pairs = blocked_gap = blocked_terminal = blocked_dash = blocked_words = would_merge = 0
    examples = []

    for i in range(len(node_dicts) - 1):
        cur = node_dicts[i]
        nxt = node_dicts[i + 1]

        if cur["show_name"] != nxt["show_name"] or cur["episode"] != nxt["episode"]:
            continue

        total_pairs += 1
        gap = nxt["start_time"] - cur["end_time"]
        is_cont = _is_continuation(cur["en_text"], nxt["en_text"], cur["emb"], nxt["emb"])
        has_dash = bool(_DIALOGUE_DASH_RE.match(nxt["en_text"]))
        combined_words = len((cur["en_text"] + " " + nxt["en_text"]).split())
        sim = round(_cosine_sim(cur["emb"], nxt["emb"]), 3) \
            if cur["emb"] is not None and nxt["emb"] is not None else None

        if gap > gap_threshold:
            blocked_gap += 1
        elif not is_cont:
            blocked_terminal += 1
        elif has_dash:
            blocked_dash += 1
        elif combined_words > _MERGE_MAX_WORDS:
            blocked_words += 1
        else:
            would_merge += 1
            if len(examples) < 10:
                examples.append({
                    "id_a": cur["id"], "en_a": cur["en_text"], "end_a": cur["end_time"],
                    "id_b": nxt["id"], "en_b": nxt["en_text"], "start_b": nxt["start_time"],
                    "gap": round(gap, 3),
                    "similarity": sim,
                })

    return {
        "total_nodes": len(node_dicts),
        "adjacent_pairs": total_pairs,
        "gap_threshold": gap_threshold,
        "blocked_by_gap": blocked_gap,
        "blocked_by_terminal_punct": blocked_terminal,
        "blocked_by_dialogue_dash": blocked_dash,
        "blocked_by_word_count": blocked_words,
        "would_merge": would_merge,
        "examples": examples,
    }


@router.post("/merge-adjacent")
async def merge_adjacent_nodes(
    show_name: Optional[str] = Query(None, description="指定剧名，为空时处理全库"),
    episode: Optional[str] = Query(None, description="指定集号（需同时传 show_name）"),
    gap_threshold: float = Query(1.0, description="相邻字幕最大间隔（秒），默认 1.0s"),
    session: AsyncSession = Depends(get_session),
):
    """
    对已入库的知识节点执行相邻分句合并（与字幕解析时的合并规则相同）。

    合并条件（全部满足才合并）：
    1. 相邻两行时间间隔 ≤ 0.30s
    2. 前一行末尾无终止标点（. ! ? ; …）
    3. 后一行不以对话破折号「- 」开头（换人说话）
    4. 合并后英文词数 ≤ 50

    合并操作：
    - 保留时间靠前的节点，延伸 end_time 并拼接 en_text / zh_text
    - 删除被吸收的节点（及其 node_sources）
    - 重新计算合并后节点的 embedding 和 zh_embedding
    - 重建 FAISS 索引（en + zh）
    """
    stmt = select(func.count(KnowledgeNode.id)).where(KnowledgeNode.is_active == True)
    if show_name:
        stmt = stmt.where(KnowledgeNode.show_name == show_name)
    if episode:
        stmt = stmt.where(KnowledgeNode.episode == episode)
    total_count = (await session.execute(stmt)).scalar() or 0

    if total_count == 0:
        return {"job_id": None, "merged_pairs": 0, "message": "没有符合条件的节点"}

    import uuid as _uuid
    task_id = str(_uuid.uuid4())

    async def _merge_task():
        from services.subtitle_parser import (
            _is_continuation, _ELLIPSIS_RE, _DIALOGUE_DASH_RE, _MERGE_MAX_WORDS,
        )
        _MERGE_GAP_SEC = gap_threshold  # use caller-specified threshold
        from services.embedder import encode_texts, serialize_vector
        from services.faiss_index import (
            rebuild_index_from_db, set_global_index,
            rebuild_zh_index_from_db, set_global_zh_index,
        )
        from tasks.task_runner import update_task_progress
        import config as _cfg, numpy as np

        async with SessionLocal() as sess:
            # ── 1. 加载所有活跃节点，按 show/episode/start_time 排序 ──
            q = (
                select(KnowledgeNode)
                .where(KnowledgeNode.is_active == True)
                .order_by(KnowledgeNode.show_name, KnowledgeNode.episode, KnowledgeNode.start_time)
            )
            if show_name:
                q = q.where(KnowledgeNode.show_name == show_name)
            if episode:
                q = q.where(KnowledgeNode.episode == episode)
            nodes = (await sess.execute(q)).scalars().all()

            update_task_progress(task_id, {
                "stage": "scanning", "label": f"扫描 {len(nodes)} 个节点，分析合并候选对", "pct": 5,
            })

            # 转为纯字典进行合并算法，避免 ORM 追踪干扰
            from services.embedder import deserialize_vector as _dv

            def _safe_emb(blob):
                try:
                    return _dv(blob) if blob else None
                except Exception:
                    return None

            node_dicts = [
                {
                    "id": n.id,
                    "show_name": n.show_name,
                    "episode": n.episode,
                    "start_time": n.start_time,
                    "end_time": n.end_time,
                    "en_text": n.en_text or "",
                    "zh_text": n.zh_text or "",
                    "emb": _safe_emb(n.embedding),
                }
                for n in nodes
            ]

            # ── 2. 应用合并逻辑（Layer 1 + Layer 2）──
            to_update: dict = {}   # id -> {en_text, zh_text, end_time, word_count}
            to_delete: list = []   # 被吸收节点的 id

            i = 0
            while i < len(node_dicts) - 1:
                cur = node_dicts[i]
                nxt = node_dicts[i + 1]

                if cur["show_name"] != nxt["show_name"] or cur["episode"] != nxt["episode"]:
                    i += 1
                    continue

                gap = nxt["start_time"] - cur["end_time"]
                no_dash = not _DIALOGUE_DASH_RE.match(nxt["en_text"])
                combined_words = len((cur["en_text"] + " " + nxt["en_text"]).split())

                if gap <= _MERGE_GAP_SEC and no_dash and combined_words <= _MERGE_MAX_WORDS \
                        and _is_continuation(cur["en_text"], nxt["en_text"], cur["emb"], nxt["emb"]):
                    base_en = _ELLIPSIS_RE.sub("", cur["en_text"]).rstrip() \
                        if _ELLIPSIS_RE.search(cur["en_text"]) else cur["en_text"]
                    cur["en_text"] = (base_en + " " + nxt["en_text"]).strip()
                    cur["zh_text"] = (cur["zh_text"] + " " + nxt["zh_text"]).strip()
                    # 合并后 embedding 失效，置 None 避免误导下一轮 Layer 2 判断
                    cur["emb"] = None
                    cur["end_time"] = nxt["end_time"]
                    to_update[cur["id"]] = {
                        "en_text": cur["en_text"],
                        "zh_text": cur["zh_text"],
                        "end_time": cur["end_time"],
                        "word_count": len(cur["en_text"].split()),
                    }
                    to_delete.append(nxt["id"])
                    node_dicts.pop(i + 1)
                    # 不 +i，允许 cur 继续与新的 nxt 合并
                else:
                    i += 1

            update_task_progress(task_id, {
                "stage": "merging",
                "label": f"找到 {len(to_delete)} 对可合并节点，写入数据库",
                "pct": 25,
            })

            if not to_delete:
                update_task_progress(task_id, {
                    "stage": "done", "label": "未找到需要合并的相邻节点", "pct": 100,
                })
                return {"merged_pairs": 0, "nodes_deleted": 0}

            # ── 3. 写入合并结果 ──
            for nid, vals in to_update.items():
                await sess.execute(
                    update(KnowledgeNode).where(KnowledgeNode.id == nid).values(**vals)
                )

            # 删除被吸收节点的来源记录和节点本身
            await sess.execute(delete(NodeSource).where(NodeSource.node_id.in_(to_delete)))
            await sess.execute(delete(KnowledgeNode).where(KnowledgeNode.id.in_(to_delete)))
            await sess.commit()

            # ── 4. 重新计算合并节点的 embedding ──
            update_task_progress(task_id, {
                "stage": "encoding",
                "label": f"重新向量化 {len(to_update)} 个合并后节点",
                "pct": 45,
            })

            merged_ids = list(to_update.keys())
            merged_info = list(to_update.values())
            en_texts = [v["en_text"] for v in merged_info]
            zh_texts = [v["zh_text"] for v in merged_info]

            en_embeddings = encode_texts(en_texts)

            # zh_embedding 只对有中文内容的节点编码
            zh_valid_idx = [(i, t) for i, t in enumerate(zh_texts) if t.strip()]
            zh_embs_raw = encode_texts([t for _, t in zh_valid_idx]) if zh_valid_idx else []
            zh_emb_map = {i: emb for (i, _), emb in zip(zh_valid_idx, zh_embs_raw)}

            for idx, nid in enumerate(merged_ids):
                zh_emb = zh_emb_map.get(idx)
                await sess.execute(
                    update(KnowledgeNode).where(KnowledgeNode.id == nid).values(
                        embedding=serialize_vector(en_embeddings[idx]),
                        zh_embedding=serialize_vector(zh_emb) if zh_emb is not None else None,
                    )
                )
            await sess.commit()

            # ── 5. 重建 FAISS 索引 ──
            update_task_progress(task_id, {"stage": "rebuild", "label": "重建检索索引…", "pct": 80})

            index_path = str(_cfg.INDEX_DIR / "knowledge.index")
            en_index, en_mapping = await rebuild_index_from_db(sess, force_full=True, index_path=index_path)
            set_global_index(en_index, en_mapping)

            zh_index, zh_mapping = await rebuild_zh_index_from_db(sess)
            set_global_zh_index(zh_index, zh_mapping)

            update_task_progress(task_id, {
                "stage": "done",
                "label": (
                    f"完成！合并 {len(to_delete)} 对节点，"
                    f"重建索引 en={en_index.ntotal} zh={zh_index.ntotal}"
                ),
                "pct": 100,
            })
            return {"merged_pairs": len(to_delete), "nodes_deleted": len(to_delete)}

    scope = show_name or "全库"
    if episode:
        scope += f"·{episode}"
    job_id = await submit_task(
        _merge_task(),
        description=f"合并相邻分句: {scope}（{total_count} 个节点）",
        task_id=task_id,
    )
    return {
        "job_id": job_id,
        "total_nodes": total_count,
        "message": f"已提交合并任务，共 {total_count} 个节点待扫描，完成后重建索引",
    }


# ──────────────────────────────────────────────
# POST /knowledge/backfill-zh-embeddings
# 为历史节点批量计算中文向量并重建 zh FAISS 索引
# ──────────────────────────────────────────────

@router.post("/backfill-zh-embeddings")
async def backfill_zh_embeddings(
    session: AsyncSession = Depends(get_session),
):
    """
    为所有有中文翻译（zh_text 非空）但尚无中文向量（zh_embedding 为 NULL）的知识节点，
    批量计算并存储 zh_embedding，然后重建中文 FAISS 索引。

    这是一次性迁移操作，执行完成后新上传字幕将自动维护 zh_embedding。
    """
    # 统计待处理节点数（预检）
    stmt = select(func.count(KnowledgeNode.id)).where(
        KnowledgeNode.zh_text != "",
        KnowledgeNode.zh_text.isnot(None),
        KnowledgeNode.zh_embedding.is_(None),
    )
    pending = (await session.execute(stmt)).scalar() or 0

    if pending == 0:
        return {"job_id": None, "pending": 0, "message": "所有有中文翻译的节点已有中文向量，无需回填"}

    import uuid as _uuid
    task_id = str(_uuid.uuid4())

    async def _backfill_task():
        from services.embedder import encode_texts, serialize_vector
        from services.faiss_index import rebuild_zh_index_from_db, set_global_zh_index
        from tasks.task_runner import update_task_progress

        async with SessionLocal() as sess:
            # 查询所有待回填节点
            rows = (await sess.execute(
                select(KnowledgeNode.id, KnowledgeNode.zh_text).where(
                    KnowledgeNode.zh_text != "",
                    KnowledgeNode.zh_text.isnot(None),
                    KnowledgeNode.zh_embedding.is_(None),
                )
            )).all()

            total = len(rows)
            if not rows:
                return {"filled": 0}

            update_task_progress(task_id, {
                "stage": "backfill", "label": f"共 {total} 个节点待回填中文向量", "pct": 3,
            })

            _CHUNK = 128
            filled = 0
            for i in range(0, total, _CHUNK):
                chunk = rows[i: i + _CHUNK]
                chunk_ids = [r.id for r in chunk]
                chunk_texts = [r.zh_text for r in chunk]

                embeddings = encode_texts(chunk_texts)
                for nid, emb in zip(chunk_ids, embeddings):
                    await sess.execute(
                        update(KnowledgeNode)
                        .where(KnowledgeNode.id == nid)
                        .values(zh_embedding=serialize_vector(emb))
                    )
                await sess.flush()
                filled += len(chunk)

                pct = 5 + int(85 * filled / total)
                update_task_progress(task_id, {
                    "stage": "backfill",
                    "label": f"已处理 {filled}/{total} 个节点",
                    "pct": pct,
                })

            await sess.commit()

            # 重建 zh FAISS 索引
            update_task_progress(task_id, {
                "stage": "rebuild", "label": "重建中文检索索引…", "pct": 93,
            })
            index, mapping = await rebuild_zh_index_from_db(sess)
            set_global_zh_index(index, mapping)

            update_task_progress(task_id, {
                "stage": "done",
                "label": f"完成！已回填 {filled} 个节点，zh 索引共 {index.ntotal} 条",
                "pct": 100,
            })
            return {"filled": filled, "zh_index_size": index.ntotal}

    job_id = await submit_task(
        _backfill_task(),
        description=f"zh_embedding 回填（{pending} 个节点）",
        task_id=task_id,
    )

    return {
        "job_id": job_id,
        "pending": pending,
        "message": f"已提交回填任务，共 {pending} 个节点待处理，完成后将重建中文检索索引",
    }


# ──────────────────────────────────────────────
# POST /knowledge/backfill-codes
# 为历史节点批量生成 code 编码
# ──────────────────────────────────────────────

@router.post("/backfill-codes")
async def backfill_codes(
    session: AsyncSession = Depends(get_session),
):
    """
    为所有尚未生成 code 的历史节点批量生成编码（格式：剧名-集号-毫秒时间戳）。
    新上传的字幕在入库时会自动生成 code，此接口仅需执行一次用于历史数据迁移。
    """
    from models.knowledge_node import make_node_code

    pending = (await session.execute(
        select(func.count(KnowledgeNode.id)).where(KnowledgeNode.code.is_(None))
    )).scalar() or 0

    if pending == 0:
        return {"filled": 0, "message": "所有节点已有编码，无需回填"}

    rows = (await session.execute(
        select(KnowledgeNode.id, KnowledgeNode.show_name,
               KnowledgeNode.episode, KnowledgeNode.start_time)
        .where(KnowledgeNode.code.is_(None))
    )).all()

    filled = skipped = 0
    for row in rows:
        code = make_node_code(row.show_name, row.episode or "", row.start_time)
        try:
            await session.execute(
                update(KnowledgeNode).where(KnowledgeNode.id == row.id).values(code=code)
            )
            filled += 1
        except Exception:
            # 极少数情况下同一 show+episode+start_time 有重复节点，追加 id 区分
            try:
                await session.execute(
                    update(KnowledgeNode).where(KnowledgeNode.id == row.id)
                    .values(code=f"{code}-{row.id}")
                )
                filled += 1
            except Exception:
                skipped += 1

    await session.commit()
    return {
        "filled": filled,
        "skipped": skipped,
        "message": f"已为 {filled} 个节点生成编码{'，跳过 ' + str(skipped) + ' 个' if skipped else ''}",
    }


# ──────────────────────────────────────────────
# POST /knowledge/full-re-evaluate
# 对全库在用知识节点一键全量质量评估
# ──────────────────────────────────────────────

@router.post("/full-re-evaluate")
async def full_re_evaluate_active_nodes(
    threshold: Optional[float] = Query(None, description="质量分阈值，为空时用 config.QUALITY_THRESHOLD"),
    show_name: Optional[str] = Query(None, description="指定剧名，为空时处理全部在用节点"),
    session: AsyncSession = Depends(get_session),
):
    """
    对知识库中所有「在用」节点重新计算口语质量分，低于阈值的自动降级为弃用。

    以后台任务运行，避免节点量大时请求超时。

    Args:
        threshold: 质量分阈值（空=使用 config.QUALITY_THRESHOLD）
        show_name: 可选，仅处理指定剧名（空=全库）

    Returns:
        {job_id, pending, threshold, message}
    """
    threshold = threshold if threshold is not None else config.QUALITY_THRESHOLD

    # 统计待处理节点数（用于前端预览）
    count_stmt = select(func.count(KnowledgeNode.id)).where(KnowledgeNode.is_active == True)
    if show_name:
        count_stmt = count_stmt.where(KnowledgeNode.show_name == show_name)
    pending = (await session.execute(count_stmt)).scalar_one()

    if pending == 0:
        return {
            "job_id": None,
            "pending": 0,
            "threshold": threshold,
            "message": "没有在用的知识节点，无需评估",
        }

    import uuid as _uuid
    task_id = str(_uuid.uuid4())
    _threshold = threshold       # 闭包捕获，防止外部修改
    _show_name = show_name

    async def _full_eval_task():
        from services.quality_filter import compute_authenticity_score
        from tasks.task_runner import update_task_progress

        _CHUNK = 200  # 每批 200 条，平衡进度粒度与事务大小

        async with SessionLocal() as sess:
            # 查全量在用节点 id + en_text（不加载 embedding 等大字段）
            q = select(KnowledgeNode.id, KnowledgeNode.en_text, KnowledgeNode.is_active).where(
                KnowledgeNode.is_active == True
            )
            if _show_name:
                q = q.where(KnowledgeNode.show_name == _show_name)
            rows = (await sess.execute(q)).all()

            total = len(rows)
            evaluated = demoted = 0

            update_task_progress(task_id, {
                "stage": "evaluating",
                "label": f"共 {total} 个在用节点待评估（阈值 {_threshold}）",
                "pct": 2,
                "done": 0,
                "total": total,
                "demoted": 0,
            })

            for chunk_start in range(0, total, _CHUNK):
                chunk = rows[chunk_start: chunk_start + _CHUNK]
                chunk_demoted = 0

                for row in chunk:
                    score = compute_authenticity_score(row.en_text or "")
                    values: dict = {"quality_score": round(score, 4)}
                    if score < _threshold:
                        values["is_active"] = False
                        chunk_demoted += 1

                    await sess.execute(
                        update(KnowledgeNode)
                        .where(KnowledgeNode.id == row.id)
                        .values(**values)
                    )

                await sess.commit()
                evaluated += len(chunk)
                demoted += chunk_demoted

                pct = 5 + int(93 * evaluated / total)
                update_task_progress(task_id, {
                    "stage": "evaluating",
                    "label": f"已评估 {evaluated}/{total}，累计降级 {demoted} 个",
                    "pct": pct,
                    "done": evaluated,
                    "total": total,
                    "demoted": demoted,
                })

            update_task_progress(task_id, {
                "stage": "done",
                "label": f"全量评估完成：共 {total} 个节点，降级 {demoted} 个（阈值 {_threshold}）",
                "pct": 100,
                "done": total,
                "total": total,
                "demoted": demoted,
            })

            # 有节点被降级时触发弃用分析（在任务内部触发，确保数据已落库）
            if demoted > 0:
                _trigger_deprecated_analysis("full-re-evaluate")

        return {"total": total, "demoted": demoted, "threshold": _threshold}

    scope = _show_name or "全库"
    job_id = await submit_task(
        _full_eval_task(),
        description=f"全量质量评估: {scope}（{pending} 个在用节点，阈值 {threshold}）",
        task_id=task_id,
    )

    return {
        "job_id": job_id,
        "pending": pending,
        "threshold": threshold,
        "message": f"已提交全量评估任务，共 {pending} 个在用节点，完成后低于 {threshold} 的将自动降级",
    }


# ──────────────────────────────────────────────
# 质量分重新评估（在用 ↔ 弃用 降级/晋升）
# ──────────────────────────────────────────────

@router.post("/nodes/{node_id}/re-evaluate")
async def re_evaluate_node(
    node_id: int,
    threshold: Optional[float] = None,
    session: AsyncSession = Depends(get_session),
):
    """
    重新评估单个知识节点的口语质量分。

    Args:
        threshold: 质量分阈值（可选，默认使用 config.QUALITY_THRESHOLD=0.30）

    流程：
    1. 用与流水线相同的评分函数重新计算 quality_score
    2. 更新数据库中的 quality_score
    3. 若 quality_score < threshold → is_active=False（降级为弃用）
    4. 若 quality_score >= threshold 且当前为弃用 → is_active=True（恢复为在用）

    Returns:
        {node_id, quality_score, is_active, demoted, restored, threshold}
    """
    from services.quality_filter import compute_authenticity_score

    node = await session.get(KnowledgeNode, node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"节点 {node_id} 不存在")

    new_score = compute_authenticity_score(node.en_text)
    demoted = False
    restored = False

    threshold = threshold if threshold is not None else config.QUALITY_THRESHOLD
    was_active = node.is_active

    node.quality_score = round(new_score, 4)
    if new_score < threshold:
        if was_active:
            node.is_active = False
            demoted = True
    else:
        if not was_active:
            node.is_active = True
            restored = True

    await session.commit()

    if demoted:
        _trigger_deprecated_analysis("re-evaluate")

    return {
        "node_id": node_id,
        "quality_score": node.quality_score,
        "is_active": node.is_active,
        "threshold": threshold,
        "demoted": demoted,
        "restored": restored,
        "message": (
            f"已降级为弃用知识（质量分 {node.quality_score:.3f} < 阈值 {threshold}）" if demoted
            else f"已恢复为在用知识（质量分 {node.quality_score:.3f} ≥ 阈值 {threshold}）" if restored
            else f"质量分 {node.quality_score:.3f}，状态未变"
        ),
    }


# ──────────────────────────────────────────────
# POST /knowledge/retag-speech-acts
# 为 speech_act 为 NULL 或未打标的历史节点补填言语行为标签
# ──────────────────────────────────────────────

@router.post("/retag-speech-acts")
async def retag_speech_acts(
    show_name: Optional[str] = Query(None, description="指定剧名，为空时处理全库"),
    force_all: bool = Query(False, description="True=重标全部节点；False=仅补标 speech_act 为 NULL 的节点"),
    session: AsyncSession = Depends(get_session),
):
    """
    专项补标 speech_act（言语行为标签）。

    适用场景：
    - L1 功能上线前已入库的历史节点，speech_act 列为 NULL 需要补填
    - 修改了 speech_act 枚举定义后，对全库重新打标

    与 /retag 的区别：
    - 只更新 speech_act 字段，不修改 scene/emotion/difficulty/keywords
    - 仅调用 AI 打标，不触发 embedding 重建

    返回：{job_id, pending, message}
    """
    stmt = select(KnowledgeNode.id, KnowledgeNode.en_text, KnowledgeNode.word_count).where(
        KnowledgeNode.is_active == True
    )
    if not force_all:
        stmt = stmt.where(KnowledgeNode.speech_act.is_(None))
    if show_name:
        stmt = stmt.where(KnowledgeNode.show_name == show_name)

    rows = (await session.execute(stmt)).all()
    if not rows:
        return {
            "job_id": None,
            "pending": 0,
            "message": "没有需要补标的节点（所有节点已有 speech_act）",
        }

    node_list = [
        {"id": r[0], "en_text": r[1], "word_count": r[2] or len((r[1] or "").split())}
        for r in rows
    ]

    import uuid as _uuid
    task_id = str(_uuid.uuid4())

    async def _retag_sa_task():
        from services.ai_tagger import batch_tag_and_translate, rule_based_tag, VALID_SPEECH_ACTS
        from tasks.task_runner import update_task_progress

        _total = len(node_list)
        ai_ok = rule_fb = 0
        _CHUNK = 20

        async with SessionLocal() as sess:
            for chunk_start in range(0, _total, _CHUNK):
                chunk = node_list[chunk_start: chunk_start + _CHUNK]
                pct = int(100 * chunk_start / _total)
                update_task_progress(task_id, {
                    "stage": "retagging",
                    "label": f"speech_act 补标 {chunk_start}/{_total}",
                    "pct": pct,
                    "done": chunk_start,
                    "total": _total,
                })

                try:
                    all_tags, _ = await batch_tag_and_translate(
                        texts=[n["en_text"] for n in chunk],
                        word_counts=[n["word_count"] for n in chunk],
                        max_concurrency=5,
                    )
                except Exception:
                    all_tags = [rule_based_tag(n["en_text"], n["word_count"]) for n in chunk]

                for node_info, tags in zip(chunk, all_tags):
                    sa = tags.get("speech_act", "neutral_statement")
                    if sa not in VALID_SPEECH_ACTS:
                        sa = "neutral_statement"
                    await sess.execute(
                        update(KnowledgeNode)
                        .where(KnowledgeNode.id == node_info["id"])
                        .values(speech_act=sa)
                    )
                    if tags.get("tag_source") == "ai":
                        ai_ok += 1
                    else:
                        rule_fb += 1

                await sess.commit()

            update_task_progress(task_id, {
                "stage": "done",
                "label": f"speech_act 补标完成：AI {ai_ok} 条，规则降级 {rule_fb} 条",
                "pct": 100,
                "done": _total,
                "total": _total,
            })

        return {"total": _total, "ai_tagged": ai_ok, "rule_fallback": rule_fb}

    scope = show_name or "全库"
    job_id = await submit_task(
        _retag_sa_task(),
        description=f"speech_act 补标: {scope}（{len(node_list)} 个节点）",
        task_id=task_id,
    )

    return {
        "job_id": job_id,
        "pending": len(node_list),
        "message": (
            f"已提交 speech_act 补标任务，共 {len(node_list)} 个节点，"
            f"请通过 /jobs/{job_id} 查询进度"
        ),
    }


class BatchReEvaluateRequest(BaseModel):
    node_ids: List[int]
    threshold: Optional[float] = None


@router.post("/nodes/batch-re-evaluate")
async def batch_re_evaluate_nodes(
    body: BatchReEvaluateRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    批量重新评估知识节点质量分，低于阈值的自动降级为弃用。

    Body: {node_ids: [...], threshold: 0.30}（threshold 可选，默认用 config 值）
    Returns: {evaluated, demoted, restored, threshold}
    """
    from services.quality_filter import compute_authenticity_score

    if not body.node_ids:
        raise HTTPException(status_code=400, detail="node_ids 不能为空")

    threshold = body.threshold if body.threshold is not None else config.QUALITY_THRESHOLD
    result_rows = await session.execute(
        select(KnowledgeNode).where(KnowledgeNode.id.in_(body.node_ids))
    )
    nodes = result_rows.scalars().all()

    evaluated = demoted = restored = 0
    for node in nodes:
        score = compute_authenticity_score(node.en_text)
        was_active = node.is_active
        node.quality_score = round(score, 4)
        if score < threshold:
            node.is_active = False
            if was_active:
                demoted += 1
        else:
            node.is_active = True
            if not was_active:
                restored += 1
        evaluated += 1

    await session.commit()

    if demoted > 0:
        _trigger_deprecated_analysis("batch-re-evaluate")

    return {
        "evaluated": evaluated,
        "demoted": demoted,
        "restored": restored,
        "threshold": threshold,
        "message": f"已评估 {evaluated} 个节点：降级 {demoted} 个，恢复 {restored} 个",
    }


# ──────────────────────────────────────────────
# POST /knowledge/analyze-deprecated  — 弃用节点模式分析
# ──────────────────────────────────────────────

@router.post("/analyze-deprecated")
async def analyze_deprecated(
    session: AsyncSession = Depends(get_session),
):
    """
    分析 is_active=False 的弃用知识节点，识别导致节点失效的典型语言模式，
    并输出可用于反哺字幕质量过滤机制的统计报告与改进建议。

    报告结构：
    - total_deprecated: 弃用节点总数
    - patterns: 各模式的分布（count / ratio / 示例 / 过滤建议）
    - score_distribution: 质量分桶分布 + 均值/中位数
    - filter_recommendations: 过滤器改进建议列表

    适用场景：
    - 管理员手动弃用一批节点后，点击"分析弃用规律"
    - 定期（每周/每次大规模弃用后）调用，持续迭代质量过滤规则
    """
    from services.deprecated_analyzer import analyze_deprecated_nodes
    report = await analyze_deprecated_nodes(session)
    return report


# ──────────────────────────────────────────────
# GET /knowledge/analyze-deprecated  — 查询上次分析报告（轻量版）
# ──────────────────────────────────────────────

@router.get("/analyze-deprecated")
async def get_deprecated_summary(
    session: AsyncSession = Depends(get_session),
):
    """
    快速摘要：只返回弃用节点数量和顶部模式，不做全量分析。
    供管理后台侧边栏实时展示，响应更快。
    """
    from sqlalchemy import select, func
    total = (await session.execute(
        select(func.count(KnowledgeNode.id)).where(KnowledgeNode.is_active == False)
    )).scalar() or 0

    if total == 0:
        return {"total_deprecated": 0, "top_patterns": [], "hint": "暂无弃用节点"}

    # 取前 30 条弃用节点做快速扫描
    rows = (await session.execute(
        select(KnowledgeNode.en_text, KnowledgeNode.quality_score)
        .where(KnowledgeNode.is_active == False)
        .order_by(KnowledgeNode.id.desc())
        .limit(30)
    )).all()

    from services.deprecated_analyzer import _classify_node
    from collections import Counter
    counter: Counter = Counter()
    for row in rows:
        for label in _classify_node(row.en_text, row.quality_score or 0.0):
            counter[label] += 1

    top = [{"pattern": k, "count_in_sample": v} for k, v in counter.most_common(5)]
    return {
        "total_deprecated": total,
        "sample_size": len(rows),
        "top_patterns": top,
        "hint": "调用 POST /knowledge/analyze-deprecated 获取完整报告",
    }
