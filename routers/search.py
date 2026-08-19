"""
DramaLingo · 语义检索路由（多信号版）

综合三路语义信号 + 标签匹配进行知识节点检索：
  1. zh_sim:    中文查询向量 vs 中文翻译向量（zh_embedding）
  2. en_tr_sim: 中文查询翻译为英文后的向量 vs 英文字幕向量（en_embedding）
  3. en_sim:    中文查询向量跨语言 vs 英文字幕向量（baseline，不依赖翻译 API）
  4. tag_score: 场景/情绪关键词匹配
  5. quality / occurrence: 辅助质量信号

接口：GET /search?q=<中文查询>&limit=10&difficulty=&scene=&has_clip=
"""

import asyncio
import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

import config
from database import get_session
from models.knowledge_node import KnowledgeNode
from models.clip import Clip
from models.search_feedback import SearchFeedback
from services.embedder import encode_single, encode_texts
from services.faiss_index import (
    get_global_index, get_global_zh_index, search_similar,
)
from services.query_engine import (
    detect_tag_hints, compute_tag_score,
    translate_to_english, expand_query, compute_final_score,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


@router.get("/search")
async def semantic_search(
    q: str = Query(..., min_length=1, max_length=200, description="中文查询词，如「表白心意」"),
    limit: int = Query(10, ge=1, le=50, description="返回结果数量"),
    offset: int = Query(0, ge=0, description="分页偏移量，第 N 页传 (N-1)*limit"),
    difficulty: Optional[str] = Query(None, description="难度筛选: beginner/intermediate/advanced"),
    scene: Optional[str] = Query(None, description="情景筛选: daily/work/romance/conflict/humor"),
    has_clip: Optional[bool] = Query(None, description="是否只返回有视频切片的结果"),
    speech_act: Optional[str] = Query(None, description="言语行为筛选: requesting/apologizing/refusing 等 25 种值"),
    session: AsyncSession = Depends(get_session),
):
    """
    多信号语义检索接口。

    输入中文描述，综合三路语义向量相似度 + 标签匹配，返回最地道的英文表达。
    """
    q = q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="查询词不能为空")

    loop = asyncio.get_running_loop()

    # ──────────────────────────────────────────────
    # Step 1: 并行执行 — 原始查询向量化 + 标签检测 + 意图扩展
    # ──────────────────────────────────────────────
    query_vec = await loop.run_in_executor(None, encode_single, q)
    detected_scene, detected_emotion = detect_tag_hints(q)

    # 意图扩展（与后续 DB 操作并行；超时 3s 内自动降级）
    expansion_task = asyncio.create_task(expand_query(q))

    # ──────────────────────────────────────────────
    # Step 2: FAISS 三路检索
    # ──────────────────────────────────────────────
    en_index, en_mapping = get_global_index()
    zh_index, zh_mapping = get_global_zh_index()

    # 若两个索引均为空，降级为关键词搜索
    if (en_index is None or en_index.ntotal == 0) and (zh_index is None or zh_index.ntotal == 0):
        logger.warning("FAISS 索引为空，降级为关键词搜索")
        await expansion_task
        return await _keyword_fallback(q, limit, offset, difficulty, scene, has_clip, session, speech_act)

    # 分页时需要足够多的候选才能支撑 offset，动态扩大 K
    K = max(config.FAISS_RECALL_K, offset + limit + 20)

    # 信号1：跨语言 en_sim（原始中文查询 → 英文 embedding，不依赖扩展的基线）
    en_sim_map: Dict[int, float] = {}
    if en_index and en_index.ntotal > 0:
        for nid, sim in search_similar(en_index, en_mapping, query_vec, k=K):
            en_sim_map[nid] = sim

    # 等待意图扩展完成
    expansion = await expansion_task
    expanded_zh = expansion.expanded_zh
    candidate_en = expansion.candidate_en

    # 信号2：zh_sim（扩展后的中文描述 → 中文翻译 embedding）
    zh_sim_map: Dict[int, float] = {}
    if zh_index and zh_index.ntotal > 0:
        # 若 expanded_zh 与原始查询不同，重新编码
        if expanded_zh != q:
            zh_query_vec = await loop.run_in_executor(None, encode_single, expanded_zh)
        else:
            zh_query_vec = query_vec
        for nid, sim in search_similar(zh_index, zh_mapping, zh_query_vec, k=K):
            zh_sim_map[nid] = sim

    # 信号3：en_tr_sim（候选英文表达 → 英文 embedding，多路取 max）
    en_tr_sim_map: Dict[int, float] = {}
    if candidate_en and en_index and en_index.ntotal > 0:
        en_vecs = await loop.run_in_executor(None, encode_texts, candidate_en)
        for vec in en_vecs:
            for nid, sim in search_similar(en_index, en_mapping, vec, k=K):
                if sim > en_tr_sim_map.get(nid, 0.0):
                    en_tr_sim_map[nid] = sim

    # 扩展元数据（供返回值调试）
    en_query = candidate_en[0] if candidate_en else None

    # ──────────────────────────────────────────────
    # Step 3: 合并候选集 + 最低相似度过滤
    # ──────────────────────────────────────────────
    all_candidate_ids = set(en_sim_map) | set(zh_sim_map) | set(en_tr_sim_map)

    if not all_candidate_ids:
        return {"query": q, "total": 0, "results": [], "search_mode": "semantic"}

    min_sim = config.RANK_MIN_SIMILARITY
    # 至少有一路信号 >= min_sim 才保留
    filtered_ids = {
        nid for nid in all_candidate_ids
        if max(
            en_sim_map.get(nid, 0.0),
            zh_sim_map.get(nid, 0.0),
            en_tr_sim_map.get(nid, 0.0),
        ) >= min_sim
    }

    if not filtered_ids:
        return {
            "query": q, "total": 0, "results": [], "search_mode": "semantic",
            "en_query": en_query,
        }

    # ──────────────────────────────────────────────
    # Step 4: 从数据库加载候选节点详情
    # ──────────────────────────────────────────────
    stmt = select(KnowledgeNode).where(
        KnowledgeNode.id.in_(filtered_ids),
        KnowledgeNode.is_active == True,
    )
    if difficulty:
        stmt = stmt.where(KnowledgeNode.difficulty == difficulty)
    if scene:
        stmt = stmt.where(KnowledgeNode.scene == scene)
    if has_clip is True:
        stmt = stmt.where(KnowledgeNode.has_any_clip == True)
    if speech_act:
        stmt = stmt.where(KnowledgeNode.speech_act == speech_act)

    nodes = (await session.execute(stmt)).scalars().all()

    if not nodes:
        return await _keyword_fallback(q, limit, offset, difficulty, scene, has_clip, session, speech_act)

    # ──────────────────────────────────────────────
    # Step 5: 多信号综合评分 + 相对阈值过滤
    # ──────────────────────────────────────────────
    scored: List[tuple] = []
    for node in nodes:
        zh_sim = zh_sim_map.get(node.id, 0.0)
        en_sim = en_sim_map.get(node.id, 0.0)
        en_tr_sim = en_tr_sim_map.get(node.id, 0.0)
        tag_score = compute_tag_score(node.scene, node.emotion, detected_scene, detected_emotion)
        final = compute_final_score(
            zh_sim, en_sim, en_tr_sim, tag_score,
            node.quality_score or 0.5,
            node.occurrence_count or 1,
        )
        scored.append((node, zh_sim, en_sim, en_tr_sim, tag_score, final))

    # 相对阈值：仅保留与最高综合分差距在阈值内的结果
    if scored:
        top_score = max(item[-1] for item in scored)
        cutoff = top_score - config.RANK_RELATIVE_THRESHOLD
        scored = [item for item in scored if item[-1] >= cutoff]

    scored.sort(key=lambda x: x[-1], reverse=True)
    total_scored = len(scored)          # 分页前总数，返回给前端用于渲染页码
    top_results = scored[offset:offset + limit]

    # ──────────────────────────────────────────────
    # Step 6: 查询切片信息
    # ──────────────────────────────────────────────
    top_node_ids = [item[0].id for item in top_results]
    clips_stmt = (
        select(Clip)
        .where(
            Clip.node_id.in_(top_node_ids),
            Clip.is_active == True,
            Clip.clip_status == "matched",
        )
        .order_by(Clip.node_id, Clip.match_confidence.desc())
    )
    all_clips = (await session.execute(clips_stmt)).scalars().all()
    best_clip_map: Dict[int, Clip] = {}
    for clip in all_clips:
        if clip.node_id not in best_clip_map:
            best_clip_map[clip.node_id] = clip

    # ──────────────────────────────────────────────
    # Step 7: 组装返回结果
    # ──────────────────────────────────────────────
    results = []
    for node, zh_sim, en_sim, en_tr_sim, tag_score, final_score in top_results:
        clip = best_clip_map.get(node.id)
        item = {
            "node_id": node.id,
            "en_text": node.en_text,
            "zh_text": node.zh_text,
            "show_name": node.show_name,
            "episode": node.episode,
            "start_time": node.start_time,
            "end_time": node.end_time,
            "scene": node.scene,
            "emotion": node.emotion,
            "difficulty": node.difficulty,
            "keywords": node.keywords or [],
            "speech_act": node.speech_act or "neutral_statement",
            "quality_score": round(node.quality_score or 0.5, 4),
            "occurrence_count": node.occurrence_count,
            # 兼容旧前端：similarity 取各路信号中最高的单项相似度
            "similarity": round(max(zh_sim, en_sim, en_tr_sim), 4),
            "rank_score": round(final_score, 4),
            # 各信号得分明细（供调试/展示）
            "scores": {
                "final": round(final_score, 4),
                "zh_sim": round(zh_sim, 4),
                "en_sim": round(en_sim, 4),
                "en_tr_sim": round(en_tr_sim, 4),
                "tag": round(tag_score, 4),
            },
            "has_clip": clip is not None,
            "clip": None,
        }
        if clip:
            item["clip"] = {
                "clip_id": clip.id,
                "stream_url": clip.stream_url,
                "thumb_url": clip.thumb_url,
                "duration": clip.duration,
                "clip_status": clip.clip_status,
                "whisper_start": clip.whisper_start,
                "whisper_end": clip.whisper_end,
            }
        results.append(item)

    return {
        "query": q,
        "expanded_zh": expanded_zh if expanded_zh != q else None,
        "candidate_en": candidate_en,
        "en_query": en_query,
        "from_cache": expansion.from_cache,
        "detected_scene": detected_scene,
        "detected_emotion": detected_emotion,
        "total": total_scored,          # 命中总数（分页前），用于渲染页码
        "offset": offset,
        "limit": limit,
        "results": results,
        "search_mode": "semantic_multi",
    }


async def _keyword_fallback(
    q: str,
    limit: int,
    offset: int,
    difficulty: Optional[str],
    scene: Optional[str],
    has_clip: Optional[bool],
    session: AsyncSession,
    speech_act: Optional[str] = None,
) -> dict:
    """
    降级关键词搜索（FAISS 索引为空或语义检索无结果时使用）。

    使用 ILIKE 对 en_text 和 zh_text 做模糊匹配，按 quality_score 降序排列。
    返回格式与 semantic_search 一致，search_mode 标记为 "keyword_fallback"。

    Args:
        q:          用户查询文本（中文或英文）
        limit:      每页结果数
        offset:     分页偏移量
        difficulty: 难度筛选（可选）
        scene:      场景筛选（可选）
        has_clip:   是否只返回有切片的结果（可选）
        session:    SQLAlchemy 异步会话
        speech_act: 言语行为筛选（可选）

    Returns:
        与 semantic_search 结构一致的 dict，所有 similarity/scores 字段为 0.0
    """
    logger.info(f"降级为关键词搜索: {q}")
    q_escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    base_stmt = select(KnowledgeNode).where(
        KnowledgeNode.is_active == True,
        (KnowledgeNode.en_text.ilike(f"%{q_escaped}%", escape="\\"))
        | (KnowledgeNode.zh_text.ilike(f"%{q_escaped}%", escape="\\")),
    )
    if difficulty:
        base_stmt = base_stmt.where(KnowledgeNode.difficulty == difficulty)
    if scene:
        base_stmt = base_stmt.where(KnowledgeNode.scene == scene)
    if has_clip is True:
        base_stmt = base_stmt.where(KnowledgeNode.has_any_clip == True)
    if speech_act:
        base_stmt = base_stmt.where(KnowledgeNode.speech_act == speech_act)

    # 先统计总数（用于分页）
    from sqlalchemy import func as _func
    count_stmt = select(_func.count()).select_from(base_stmt.subquery())
    total_kw = (await session.execute(count_stmt)).scalar() or 0

    stmt = base_stmt.order_by(KnowledgeNode.quality_score.desc()).offset(offset).limit(limit)
    nodes = (await session.execute(stmt)).scalars().all()

    results = []
    for node in nodes:
        results.append({
            "node_id": node.id,
            "en_text": node.en_text,
            "zh_text": node.zh_text,
            "show_name": node.show_name,
            "episode": node.episode,
            "start_time": node.start_time,
            "end_time": node.end_time,
            "scene": node.scene,
            "emotion": node.emotion,
            "difficulty": node.difficulty,
            "keywords": node.keywords or [],
            "speech_act": node.speech_act or "neutral_statement",
            "quality_score": round(node.quality_score or 0.5, 4),
            "occurrence_count": node.occurrence_count,
            "similarity": 0.0,
            "rank_score": round(node.quality_score or 0.5, 4),
            "scores": {"final": 0.0, "zh_sim": 0.0, "en_sim": 0.0, "en_tr_sim": 0.0, "tag": 0.0},
            "has_clip": node.has_any_clip,
            "clip": None,
        })
    return {
        "query": q,
        "en_query": None,
        "detected_scene": None,
        "detected_emotion": None,
        "total": total_kw,
        "offset": offset,
        "limit": limit,
        "results": results,
        "search_mode": "keyword_fallback",
    }


@router.get("/search/status")
async def search_status(session: AsyncSession = Depends(get_session)):
    """检索模块状态（含 zh 索引覆盖率）。"""
    en_index, _ = get_global_index()
    zh_index, _ = get_global_zh_index()
    en_size = en_index.ntotal if en_index else 0
    zh_size = zh_index.ntotal if zh_index else 0

    total_nodes = (
        await session.execute(
            select(func.count(KnowledgeNode.id)).where(KnowledgeNode.is_active == True)
        )
    ).scalar() or 0

    zh_covered = (
        await session.execute(
            select(func.count(KnowledgeNode.id)).where(
                KnowledgeNode.is_active == True,
                KnowledgeNode.zh_embedding.isnot(None),
            )
        )
    ).scalar() or 0

    return {
        "module": "search",
        "status": "ready",
        "en_index_size": en_size,
        "zh_index_size": zh_size,
        "total_active_nodes": total_nodes,
        "zh_embedding_coverage": round(zh_covered / total_nodes, 4) if total_nodes > 0 else 0.0,
        "en_index_coverage": round(en_size / total_nodes, 4) if total_nodes > 0 else 0.0,
    }


# ──────────────────────────────────────────────
# 搜索反馈
# ──────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    node_id: int
    action: str = Field(..., pattern="^(click|play|copy)$")
    rank_position: Optional[int] = Field(None, ge=1)
    rank_score: Optional[float] = None


@router.post("/search/feedback")
async def submit_feedback(
    body: FeedbackRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    记录用户搜索交互行为，用于后续排序权重调优。

    前端在用户点击结果卡片、播放视频、复制文本时调用此接口。
    数据写入 search_feedback 表，通过 /search/feedback/stats 可查看汇总统计。

    Args:
        body: FeedbackRequest，包含 query / node_id / action / rank_position / rank_score

    Returns:
        {"status": "ok"}
    """
    fb = SearchFeedback(
        query=body.query,
        node_id=body.node_id,
        action=body.action,
        rank_position=body.rank_position,
        rank_score=body.rank_score,
    )
    session.add(fb)
    await session.commit()
    return {"status": "ok"}


@router.get("/search/feedback/stats")
async def feedback_stats(
    days: int = Query(7, ge=1, le=90, description="统计最近 N 天"),
    session: AsyncSession = Depends(get_session),
):
    """
    搜索反馈统计：最近 N 天的点击/播放分布。

    Args:
        days: 统计时间窗口（天），默认 7 天，最大 90 天

    Returns:
        dict 包含：
        - total_interactions: 总交互数
        - by_action: {"click": N, "play": M, "copy": K}
        - top_queries: 最热门的 20 个查询词及其交互次数
    """
    from datetime import timedelta
    cutoff = func.datetime("now", f"-{days} days")

    total = (await session.execute(
        select(func.count(SearchFeedback.id)).where(SearchFeedback.created_at >= cutoff)
    )).scalar() or 0

    by_action = (await session.execute(
        select(SearchFeedback.action, func.count(SearchFeedback.id))
        .where(SearchFeedback.created_at >= cutoff)
        .group_by(SearchFeedback.action)
    )).all()

    top_queries = (await session.execute(
        select(SearchFeedback.query, func.count(SearchFeedback.id).label("cnt"))
        .where(SearchFeedback.created_at >= cutoff)
        .group_by(SearchFeedback.query)
        .order_by(func.count(SearchFeedback.id).desc())
        .limit(20)
    )).all()

    return {
        "days": days,
        "total_interactions": total,
        "by_action": {row[0]: row[1] for row in by_action},
        "top_queries": [{"query": row[0], "count": row[1]} for row in top_queries],
    }
