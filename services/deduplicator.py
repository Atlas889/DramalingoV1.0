"""
DramaLingo · 两段式去重模块
对应 PRD v0.9 §5 流程一·两段式去重 / §6.1 去重参数

去重流程：
  粗筛（归一化编辑距离 < 0.15）→ 命中则直接返回重复
  精筛（FAISS Top-5 + 余弦相似度 ≥ 0.95）→ 命中则返回重复
  两者均未命中 → 判定为新节点

公开接口：
  normalize_text(text)                          -> str
  edit_distance_normalized(s1, s2)              -> float
  coarse_filter(text, existing_texts)           -> Optional[int]  (existing_texts 的索引)
  fine_filter(text, embedding, index, mapping)  -> Optional[int]  (node_id)
  check_duplicate(text, embedding, session, index, mapping)
                                                -> Tuple[bool, Optional[int]]
  refresh_text_cache(session)                   -> None（刷新内存缓存）
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 内存缓存：全库活跃节点文本（归一化后）
# {node_id -> normalized_text}
# ──────────────────────────────────────────────
_text_cache: Dict[int, str] = {}   # node_id -> normalized en_text
_text_cache_loaded: bool = False


# ──────────────────────────────────────────────
# 口语缩写归一化映射（PRD §5 流程一）
# 只处理高频口语缩写，避免误改专有名词
# ──────────────────────────────────────────────
_CONTRACTIONS = {
    # 动词缩写
    r"\bgonna\b": "going to",
    r"\bwanna\b": "want to",
    r"\bgotta\b": "got to",
    r"\bkinda\b": "kind of",
    r"\bsorta\b": "sort of",
    r"\boutta\b": "out of",
    r"\blotta\b": "lot of",
    r"\bnotta\b": "not a",
    r"\bdunno\b": "do not know",
    r"\blemme\b": "let me",
    r"\bgimme\b": "give me",
    r"\bcmon\b": "come on",
    r"\byeah\b": "yes",
    r"\bnope\b": "no",
    # 人称缩写（顺序重要：先处理长的）
    r"\bi'm\b": "i am",
    r"\bi've\b": "i have",
    r"\bi'll\b": "i will",
    r"\bi'd\b": "i would",
    r"\byou're\b": "you are",
    r"\byou've\b": "you have",
    r"\byou'll\b": "you will",
    r"\byou'd\b": "you would",
    r"\bhe's\b": "he is",
    r"\bhe'd\b": "he would",
    r"\bhe'll\b": "he will",
    r"\bshe's\b": "she is",
    r"\bshe'd\b": "she would",
    r"\bshe'll\b": "she will",
    r"\bwe're\b": "we are",
    r"\bwe've\b": "we have",
    r"\bwe'll\b": "we will",
    r"\bwe'd\b": "we would",
    r"\bthey're\b": "they are",
    r"\bthey've\b": "they have",
    r"\bthey'll\b": "they will",
    r"\bthey'd\b": "they would",
    r"\bit's\b": "it is",
    r"\bit'll\b": "it will",
    r"\bthat's\b": "that is",
    r"\bthat'll\b": "that will",
    r"\bthat'd\b": "that would",
    r"\bwhat's\b": "what is",
    r"\bwhat'll\b": "what will",
    r"\bwhere's\b": "where is",
    r"\bwho's\b": "who is",
    r"\bhow's\b": "how is",
    r"\bthere's\b": "there is",
    r"\bthere're\b": "there are",
    r"\bthere'll\b": "there will",
    # 否定缩写
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwouldn't\b": "would not",
    r"\bcouldn't\b": "could not",
    r"\bshouldn't\b": "should not",
    r"\bmightn't\b": "might not",
    r"\bmustn't\b": "must not",
    r"\bneedn't\b": "need not",
}

# 预编译正则（提升性能）
_CONTRACTION_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in _CONTRACTIONS.items()
]


# ──────────────────────────────────────────────
# 文本归一化
# ──────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """
    对英文文本做归一化处理，用于粗筛编辑距离比较。

    处理步骤：
    1. 转小写
    2. 展开口语缩写（I'm → i am 等）
    3. 去除标点符号（保留字母、数字、空格）
    4. 合并多余空格，去除首尾空格

    Args:
        text: 原始英文文本

    Returns:
        归一化后的文本
    """
    # 1. 转小写
    text = text.lower()

    # 2. 展开口语缩写
    for pattern, replacement in _CONTRACTION_PATTERNS:
        text = pattern.sub(replacement, text)

    # 3. 去除标点符号（只保留字母、数字、空格）
    text = re.sub(r"[^\w\s]", " ", text)

    # 4. 合并多余空格
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ──────────────────────────────────────────────
# 编辑距离
# ──────────────────────────────────────────────

def edit_distance_normalized(s1: str, s2: str) -> float:
    """
    计算两个字符串的归一化编辑距离。

    公式：Levenshtein距离 / max(len(s1), len(s2))
    返回范围 [0, 1]，0 表示完全相同，1 表示完全不同。

    Args:
        s1, s2: 待比较的字符串（应已归一化）

    Returns:
        float，归一化编辑距离
    """
    if s1 == s2:
        return 0.0

    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0

    try:
        from Levenshtein import distance as lev_distance
        dist = lev_distance(s1, s2)
    except ImportError:
        # 降级：使用 difflib（较慢但无需额外依赖）
        import difflib
        ratio = difflib.SequenceMatcher(None, s1, s2).ratio()
        # 统一用 max_len 作分母，与归一化步骤保持一致
        dist = round((1 - ratio) * max_len)

    return dist / max_len


# ──────────────────────────────────────────────
# 粗筛
# ──────────────────────────────────────────────

def coarse_filter(
    text: str,
    existing_texts: Dict[int, str],
) -> Optional[int]:
    """
    粗筛：用归一化编辑距离判断是否与已有节点重复。

    Args:
        text:           待检测文本（已归一化）
        existing_texts: {node_id -> normalized_text} 的字典

    Returns:
        若命中重复，返回已有节点的 node_id；否则返回 None
    """
    norm_text = normalize_text(text)
    query_len = len(norm_text)

    for node_id, norm_existing in existing_texts.items():
        # 长度预过滤：若长度差已超过阈值允许的最大编辑距离，直接跳过
        exist_len = len(norm_existing)
        max_len = max(query_len, exist_len)
        if max_len > 0 and abs(query_len - exist_len) / max_len >= config.DEDUP_EDIT_DISTANCE_THRESHOLD:
            continue

        dist = edit_distance_normalized(norm_text, norm_existing)
        if dist < config.DEDUP_EDIT_DISTANCE_THRESHOLD:
            logger.debug(
                "粗筛命中：'%s' ≈ '%s'，编辑距离=%.3f < %.3f",
                text[:30], norm_existing[:30], dist, config.DEDUP_EDIT_DISTANCE_THRESHOLD,
            )
            return node_id

    return None


# ──────────────────────────────────────────────
# 精筛
# ──────────────────────────────────────────────

async def fine_filter(
    embedding: np.ndarray,
    index,
    mapping: Dict[int, int],
    session,
) -> Optional[int]:
    """
    精筛：用 FAISS Top-K 候选 + 余弦相似度判断是否与已有节点重复。

    Args:
        embedding: 待检测文本的 Embedding 向量（未归一化，shape=(384,)）
        index:     faiss.Index
        mapping:   {faiss_internal_id -> node_id}
        session:   AsyncSession（用于获取候选节点的 embedding）

    Returns:
        若命中重复（余弦相似度 ≥ DEDUP_SIMILARITY），返回已有节点的 node_id；否则返回 None
    """
    from services.faiss_index import search_similar
    from services.embedder import deserialize_vector
    from models.knowledge_node import KnowledgeNode
    from sqlalchemy import select

    if index is None or index.ntotal == 0:
        return None

    # FAISS Top-K 候选（PRD §6.1 FAISS_DEDUP_TOP_K=5）
    candidates = search_similar(
        index, mapping, embedding, k=config.FAISS_DEDUP_TOP_K
    )

    if not candidates:
        return None

    # 归一化查询向量（用于余弦相似度计算）
    query_vec = embedding.astype(np.float32).copy()
    norm = np.linalg.norm(query_vec)
    if norm > 0:
        query_vec = query_vec / norm

    # 逐一精算余弦相似度
    candidate_ids = [node_id for node_id, _ in candidates]
    rows = (
        await session.execute(
            select(KnowledgeNode.id, KnowledgeNode.embedding).where(
                KnowledgeNode.id.in_(candidate_ids),
                KnowledgeNode.is_active == True,
                KnowledgeNode.embedding.isnot(None),
            )
        )
    ).all()

    for row in rows:
        cand_vec = deserialize_vector(row.embedding).astype(np.float32)
        cand_norm = np.linalg.norm(cand_vec)
        if cand_norm > 0:
            cand_vec = cand_vec / cand_norm

        cosine_sim = float(np.dot(query_vec, cand_vec))

        if cosine_sim >= config.DEDUP_SIMILARITY:
            logger.debug(
                f"精筛命中：node_id={row.id}，余弦相似度={cosine_sim:.4f} "
                f">= {config.DEDUP_SIMILARITY}"
            )
            return row.id

    return None


# ──────────────────────────────────────────────
# 整合函数
# ──────────────────────────────────────────────

async def check_duplicate(
    text: str,
    embedding: np.ndarray,
    session,
    index,
    mapping: Dict[int, int],
) -> Tuple[bool, Optional[int]]:
    """
    完整两段式去重判定。

    流程：
    1. 粗筛（归一化编辑距离）：从内存缓存中比较
    2. 若粗筛未命中，执行精筛（FAISS Top-5 + 余弦相似度）

    Args:
        text:      待检测的英文文本
        embedding: 对应的 Embedding 向量（shape=(384,)）
        session:   AsyncSession
        index:     faiss.Index（可为 None，此时跳过精筛）
        mapping:   {faiss_internal_id -> node_id}

    Returns:
        (is_duplicate, existing_node_id)
        - is_duplicate=True 时，existing_node_id 为已有节点 ID
        - is_duplicate=False 时，existing_node_id=None
    """
    global _text_cache, _text_cache_loaded

    # 确保缓存已加载
    if not _text_cache_loaded:
        await refresh_text_cache(session)

    # ── 粗筛 ──
    coarse_hit = coarse_filter(text, _text_cache)
    if coarse_hit is not None:
        logger.debug(f"去重判定：粗筛命中 node_id={coarse_hit}")
        return True, coarse_hit

    # ── 精筛 ──
    if index is not None and index.ntotal > 0:
        fine_hit = await fine_filter(embedding, index, mapping, session)
        if fine_hit is not None:
            logger.debug(f"去重判定：精筛命中 node_id={fine_hit}")
            return True, fine_hit

    logger.debug(f"去重判定：新节点 '{text[:40]}'")
    return False, None


# ──────────────────────────────────────────────
# 缓存管理
# ──────────────────────────────────────────────

async def refresh_text_cache(session) -> None:
    """
    从数据库刷新全库活跃节点的文本缓存（归一化后存入内存）。
    每次字幕批处理任务开始时调用一次。

    Args:
        session: AsyncSession
    """
    global _text_cache, _text_cache_loaded
    from sqlalchemy import select
    from models.knowledge_node import KnowledgeNode

    # 提前置 True，防止并发协程在 await 挂起期间重复进入刷新逻辑
    _text_cache_loaded = True

    rows = (
        await session.execute(
            select(KnowledgeNode.id, KnowledgeNode.en_text).where(
                KnowledgeNode.is_active == True,
                KnowledgeNode.en_text.isnot(None),
            )
        )
    ).all()

    _text_cache = {row.id: normalize_text(row.en_text) for row in rows}
    logger.info("refresh_text_cache: 缓存 %d 个活跃节点文本", len(_text_cache))


def add_to_text_cache(node_id: int, en_text: str) -> None:
    """
    新节点写入数据库后，同步更新内存缓存（避免重新全量加载）。

    Args:
        node_id: 新节点的数据库 ID
        en_text: 新节点的英文文本
    """
    global _text_cache
    _text_cache[node_id] = normalize_text(en_text)
    logger.debug(f"add_to_text_cache: 新增 node_id={node_id}")


def clear_text_cache() -> None:
    """清空内存缓存（测试用）"""
    global _text_cache, _text_cache_loaded
    _text_cache = {}
    _text_cache_loaded = False
