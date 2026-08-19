"""
DramaLingo · FAISS 索引管理模块
对应 PRD v0.9 §6.6 / §10.4 FAISS 索引设计约束：
  - 使用 IndexFlatIP（内积索引）
  - 向量添加前必须 L2 归一化（等价于余弦相似度）
  - 不支持原地删除，下线节点通过全量重建移出
  - 重建阈值 FAISS_REBUILD_THRESHOLD=0.05（5%）

公开接口：
  build_index(node_ids, embeddings)          -> faiss.Index
  add_to_index(index, mapping, node_ids, embeddings)
  search_similar(index, mapping, query_vec, k) -> List[Tuple[int, float]]
  save_index(index, mapping, path)
  load_index(path)                           -> Tuple[faiss.Index, dict]
  rebuild_index_from_db(session, force_full) -> None（异步）
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

import config
from services.embedder import deserialize_vector

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 内存中的全局索引实例（应用启动后懒加载）
# ──────────────────────────────────────────────
_global_index: Optional[faiss.Index] = None
_global_mapping: Dict[int, int] = {}   # {faiss_internal_id -> node_id}

# 中文翻译 embedding 的独立索引（用于中文查询的高精度匹配）
_global_zh_index: Optional[faiss.Index] = None
_global_zh_mapping: Dict[int, int] = {}

# 默认索引文件路径
_INDEX_PATH = str(config.INDEX_DIR / "knowledge.index")
_ZH_INDEX_PATH = str(config.INDEX_DIR / "knowledge_zh.index")


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def _normalize(vectors: np.ndarray) -> np.ndarray:
    """
    对向量矩阵做 L2 归一化（in-place 操作）。
    IndexFlatIP + L2 归一化 = 余弦相似度。
    """
    vectors = vectors.astype(np.float32)
    faiss.normalize_L2(vectors)
    return vectors


# ──────────────────────────────────────────────
# 核心接口
# ──────────────────────────────────────────────

def build_index(
    node_ids: List[int],
    embeddings: np.ndarray,
) -> Tuple[faiss.Index, Dict[int, int]]:
    """
    从头构建 FAISS IndexFlatIP 索引。

    Args:
        node_ids:   数据库 node_id 列表，长度 N
        embeddings: shape=(N, 384) 的 float32 向量矩阵

    Returns:
        (index, mapping)
        - index:   faiss.IndexFlatIP 实例
        - mapping: {faiss_internal_id -> node_id}
    """
    if len(node_ids) == 0:
        index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
        return index, {}

    embeddings = embeddings.astype(np.float32)
    normalized = _normalize(embeddings.copy())

    index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    index.add(normalized)

    # FAISS 内部 ID 是连续整数 0, 1, 2, ...
    mapping = {i: node_id for i, node_id in enumerate(node_ids)}

    logger.info(f"build_index: 构建索引完成，共 {index.ntotal} 条向量")
    return index, mapping


def add_to_index(
    index: faiss.Index,
    mapping: Dict[int, int],
    node_ids: List[int],
    embeddings: np.ndarray,
) -> None:
    """
    增量添加向量到已有索引（in-place 修改 index 和 mapping）。

    Args:
        index:      已有 faiss.Index
        mapping:    已有映射字典（会被修改）
        node_ids:   新增节点的 node_id 列表
        embeddings: shape=(N, 384) 的 float32 向量矩阵
    """
    if len(node_ids) == 0:
        return

    embeddings = embeddings.astype(np.float32)
    normalized = _normalize(embeddings.copy())

    # 新增向量的 FAISS 内部 ID 从当前 ntotal 开始
    start_id = index.ntotal
    index.add(normalized)

    for i, node_id in enumerate(node_ids):
        mapping[start_id + i] = node_id

    logger.debug(f"add_to_index: 新增 {len(node_ids)} 条，索引总量 {index.ntotal}")


def search_similar(
    index: faiss.Index,
    mapping: Dict[int, int],
    query_vec: np.ndarray,
    k: int,
) -> List[Tuple[int, float]]:
    """
    查询 Top-K 相似向量。

    Args:
        index:     faiss.Index
        mapping:   {faiss_internal_id -> node_id}
        query_vec: shape=(384,) 的查询向量（未归一化）
        k:         返回候选数量

    Returns:
        [(node_id, similarity_score), ...]，按相似度降序排列
        similarity_score 为余弦相似度（0~1）
    """
    if index.ntotal == 0:
        return []

    # 归一化查询向量
    query = query_vec.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query)

    # 实际 k 不超过索引大小
    actual_k = min(k, index.ntotal)
    distances, indices = index.search(query, actual_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:  # FAISS 未找到时返回 -1
            continue
        node_id = mapping.get(int(idx))
        if node_id is not None:
            results.append((node_id, float(dist)))

    return results


def save_index(
    index: faiss.Index,
    mapping: Dict[int, int],
    path: str,
) -> None:
    """
    保存索引到磁盘。
    - 索引文件：path
    - 映射文件：path + '.mapping.json'

    Args:
        index:   faiss.Index
        mapping: {faiss_internal_id -> node_id}
        path:    索引文件路径
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mapping_path = path + ".mapping.json"

    # 先写临时文件，再原子替换，防止两文件间崩溃导致不一致
    tmp_index = path + ".tmp"
    tmp_mapping = mapping_path + ".tmp"
    try:
        faiss.write_index(index, tmp_index)
        str_mapping = {str(k): v for k, v in mapping.items()}
        with open(tmp_mapping, "w", encoding="utf-8") as f:
            json.dump(str_mapping, f)
        os.replace(tmp_index, path)
        os.replace(tmp_mapping, mapping_path)
    except Exception:
        # 清理临时文件，避免残留
        for tmp in (tmp_index, tmp_mapping):
            if os.path.exists(tmp):
                os.remove(tmp)
        raise

    logger.info(f"save_index: 索引已保存到 {path}，共 {index.ntotal} 条向量")


def load_index(path: str) -> Tuple[faiss.Index, Dict[int, int]]:
    """
    从磁盘加载索引和映射。

    Args:
        path: 索引文件路径

    Returns:
        (index, mapping)
        - index:   faiss.Index
        - mapping: {faiss_internal_id(int) -> node_id(int)}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"索引文件不存在: {path}")

    index = faiss.read_index(path)

    mapping_path = path + ".mapping.json"
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            str_mapping = json.load(f)
        mapping = {int(k): int(v) for k, v in str_mapping.items()}
    else:
        # 兼容旧版本（无映射文件时，假设 FAISS ID 即 node_id）
        mapping = {i: i for i in range(index.ntotal)}
        logger.warning(f"映射文件不存在，使用默认映射: {mapping_path}")

    logger.info(f"load_index: 从 {path} 加载索引，共 {index.ntotal} 条向量")
    return index, mapping


# ──────────────────────────────────────────────
# 索引重建（异步，从数据库读取）
# ──────────────────────────────────────────────

async def rebuild_index_from_db(
    session,
    force_full: bool = False,
    index_path: Optional[str] = None,
) -> Tuple[faiss.Index, Dict[int, int]]:
    """
    从 knowledge_nodes 表重建 FAISS 索引。

    策略（PRD §10.4）：
    - 若 force_full=True，强制全量重建
    - 若下线节点占比 > FAISS_REBUILD_THRESHOLD，全量重建
    - 否则仅增量添加新节点

    Args:
        session:    AsyncSession
        force_full: 是否强制全量重建
        index_path: 索引保存路径（默认 config.INDEX_DIR/knowledge.index）

    Returns:
        (index, mapping)
    """
    from sqlalchemy import select
    from models.knowledge_node import KnowledgeNode

    if index_path is None:
        index_path = _INDEX_PATH

    # ── 查询所有活跃节点（仅取 id 和 embedding 两列，节省内存）──
    active_q = select(KnowledgeNode.id, KnowledgeNode.embedding).where(
        KnowledgeNode.is_active == True,
        KnowledgeNode.embedding.isnot(None),
    )
    active_rows = (await session.execute(active_q)).all()

    # ── 查询下线节点数量（用于判断是否需要全量重建）──
    from sqlalchemy import func
    total_count = (
        await session.execute(select(func.count()).select_from(KnowledgeNode))
    ).scalar_one()
    inactive_count = total_count - len(active_rows)

    # ── 判断重建策略 ──
    need_full_rebuild = force_full
    if not need_full_rebuild and total_count > 0:
        inactive_ratio = inactive_count / total_count
        if inactive_ratio > config.FAISS_REBUILD_THRESHOLD:
            logger.info(
                f"下线节点占比 {inactive_ratio:.2%} > 阈值 {config.FAISS_REBUILD_THRESHOLD:.2%}，触发全量重建"
            )
            need_full_rebuild = True

    if len(active_rows) == 0:
        logger.warning("rebuild_index_from_db: 无活跃节点，构建空索引")
        index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
        mapping: Dict[int, int] = {}
        save_index(index, mapping, index_path)
        return index, mapping

    # ── 反序列化 embedding，跳过损坏节点 ──
    node_ids, valid_vecs = [], []
    for r in active_rows:
        try:
            vec = deserialize_vector(r.embedding)
            node_ids.append(r.id)
            valid_vecs.append(vec)
        except Exception as e:
            logger.warning("rebuild_index_from_db: 跳过损坏节点 id=%s，原因: %s", r.id, e)

    if not node_ids:
        logger.warning("rebuild_index_from_db: 所有节点 embedding 均损坏，构建空索引")
        index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
        mapping: Dict[int, int] = {}
        save_index(index, mapping, index_path)
        return index, mapping

    embeddings = np.stack(valid_vecs).astype(np.float32)

    if need_full_rebuild:
        logger.info(f"rebuild_index_from_db: 全量重建，共 {len(node_ids)} 个节点")
        index, mapping = build_index(node_ids, embeddings)
    else:
        # 增量模式：只添加索引中尚未包含的节点
        try:
            existing_index, existing_mapping = load_index(index_path)
            existing_node_ids = set(existing_mapping.values())
            new_node_ids = [nid for nid in node_ids if nid not in existing_node_ids]

            if new_node_ids:
                new_mask = [i for i, nid in enumerate(node_ids) if nid not in existing_node_ids]
                new_embeddings = embeddings[new_mask]
                add_to_index(existing_index, existing_mapping, new_node_ids, new_embeddings)
                logger.info(f"rebuild_index_from_db: 增量添加 {len(new_node_ids)} 个节点")
            else:
                logger.info("rebuild_index_from_db: 无新增节点，跳过更新")

            index, mapping = existing_index, existing_mapping
        except FileNotFoundError:
            # 索引文件不存在，回退到全量重建
            logger.info("rebuild_index_from_db: 索引文件不存在，执行全量重建")
            index, mapping = build_index(node_ids, embeddings)

    save_index(index, mapping, index_path)
    logger.info(f"rebuild_index_from_db: 索引重建完成，共 {index.ntotal} 条向量")
    return index, mapping


# ──────────────────────────────────────────────
# 全局索引访问（供检索接口使用）
# ──────────────────────────────────────────────

def get_global_index() -> Tuple[Optional[faiss.Index], Dict[int, int]]:
    """获取英文 embedding 全局索引（懒加载）。"""
    global _global_index, _global_mapping
    if _global_index is None:
        try:
            _global_index, _global_mapping = load_index(_INDEX_PATH)
            logger.info(f"英文全局索引从磁盘加载，共 {_global_index.ntotal} 条向量")
        except FileNotFoundError:
            logger.info("英文全局索引文件不存在，将在首次重建后可用")
    return _global_index, _global_mapping


def set_global_index(index: faiss.Index, mapping: Dict[int, int]) -> None:
    """更新内存中的英文全局索引（重建后调用）。"""
    global _global_index, _global_mapping
    _global_index = index
    _global_mapping = mapping
    logger.info(f"英文全局索引已更新，共 {index.ntotal} 条向量")


# ──────────────────────────────────────────────
# 中文 embedding 索引（zh_index）
# ──────────────────────────────────────────────

def get_global_zh_index() -> Tuple[Optional[faiss.Index], Dict[int, int]]:
    """获取中文 embedding 全局索引（懒加载）。"""
    global _global_zh_index, _global_zh_mapping
    if _global_zh_index is None:
        try:
            _global_zh_index, _global_zh_mapping = load_index(_ZH_INDEX_PATH)
            logger.info(f"中文全局索引从磁盘加载，共 {_global_zh_index.ntotal} 条向量")
        except FileNotFoundError:
            logger.info("中文全局索引文件不存在，可通过后台管理触发回填任务")
    return _global_zh_index, _global_zh_mapping


def set_global_zh_index(index: faiss.Index, mapping: Dict[int, int]) -> None:
    """更新内存中的中文全局索引。"""
    global _global_zh_index, _global_zh_mapping
    _global_zh_index = index
    _global_zh_mapping = mapping
    logger.info(f"中文全局索引已更新，共 {index.ntotal} 条向量")


def add_to_zh_index_and_save(node_ids: List[int], embeddings: np.ndarray) -> None:
    """增量添加节点到中文索引并持久化。"""
    global _global_zh_index, _global_zh_mapping
    if not node_ids:
        return

    if _global_zh_index is None:
        try:
            _global_zh_index, _global_zh_mapping = load_index(_ZH_INDEX_PATH)
        except FileNotFoundError:
            _global_zh_index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
            _global_zh_mapping = {}

    add_to_index(_global_zh_index, _global_zh_mapping, node_ids, embeddings)
    save_index(_global_zh_index, _global_zh_mapping, _ZH_INDEX_PATH)
    logger.debug(f"zh 索引增量更新，当前共 {_global_zh_index.ntotal} 条向量")


async def rebuild_zh_index_from_db(
    session,
    index_path: Optional[str] = None,
) -> Tuple[faiss.Index, Dict[int, int]]:
    """从 knowledge_nodes.zh_embedding 全量重建中文 FAISS 索引。"""
    from sqlalchemy import select
    from models.knowledge_node import KnowledgeNode

    if index_path is None:
        index_path = _ZH_INDEX_PATH

    rows = (await session.execute(
        select(KnowledgeNode.id, KnowledgeNode.zh_embedding).where(
            KnowledgeNode.is_active == True,
            KnowledgeNode.zh_embedding.isnot(None),
        )
    )).all()

    node_ids, vecs = [], []
    for r in rows:
        try:
            node_ids.append(r.id)
            vecs.append(deserialize_vector(r.zh_embedding))
        except Exception as e:
            logger.warning(f"zh_embedding 反序列化失败 id={r.id}: {e}")

    if not node_ids:
        idx = faiss.IndexFlatIP(config.EMBEDDING_DIM)
        save_index(idx, {}, index_path)
        logger.info("rebuild_zh_index_from_db: 无有效 zh_embedding，构建空索引")
        return idx, {}

    embeddings = np.stack(vecs).astype(np.float32)
    idx, mapping = build_index(node_ids, embeddings)
    save_index(idx, mapping, index_path)
    logger.info(f"rebuild_zh_index_from_db: 完成，共 {idx.ntotal} 条向量")
    return idx, mapping
