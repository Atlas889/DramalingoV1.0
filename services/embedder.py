"""
DramaLingo · Embedding 向量化模块
对应 PRD v0.9 §8.1 Embedding 模型约束：
  - 模型：paraphrase-multilingual-MiniLM-L12-v2（支持中英跨语言）
  - 维度：384
  - 批量编码：batch_size=64

公开接口：
  encode_texts(texts)       -> np.ndarray  shape=(N, 384)
  serialize_vector(vec)     -> bytes
  deserialize_vector(data)  -> np.ndarray  shape=(384,)
"""

import logging
from typing import List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 模型单例（懒加载，首次调用时初始化）
# ──────────────────────────────────────────────
_model = None


def _get_model():
    """
    懒加载 SentenceTransformer 模型。
    首次调用时从 HuggingFace 加载（约 400MB），之后复用同一实例。
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"加载 Embedding 模型: {config.EMBEDDING_MODEL}")
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
        dim = _model.get_sentence_embedding_dimension()
        logger.info(f"模型加载完成，维度: {dim}")
        if dim != config.EMBEDDING_DIM:
            raise ValueError(
                f"模型维度 {dim} 与配置 EMBEDDING_DIM={config.EMBEDDING_DIM} 不符，"
                "请检查 config.py 或模型名称"
            )
    return _model


# ──────────────────────────────────────────────
# 公开接口
# ──────────────────────────────────────────────

def encode_texts(
    texts: List[str],
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """
    批量编码文本为 Embedding 向量。

    Args:
        texts:         待编码的文本列表
        batch_size:    批量大小（PRD §8.1 要求 64）
        show_progress: 是否显示进度条（大批量时可开启）

    Returns:
        np.ndarray, shape=(N, 384), dtype=float32
    """
    if not texts:
        return np.empty((0, config.EMBEDDING_DIM), dtype=np.float32)

    # 预处理：去除首尾空格、小写化（模型对大小写不敏感）
    cleaned = [t.strip().lower() for t in texts]

    # 过滤空字符串：记录非空文本的原始位置，编码后还原
    non_empty_indices = [i for i, t in enumerate(cleaned) if t]
    if not non_empty_indices:
        logger.warning("encode_texts: 所有输入文本均为空，返回零向量")
        return np.zeros((len(texts), config.EMBEDDING_DIM), dtype=np.float32)

    non_empty_texts = [cleaned[i] for i in non_empty_indices]
    empty_count = len(texts) - len(non_empty_texts)
    if empty_count > 0:
        logger.warning("encode_texts: 过滤了 %d 条空字符串", empty_count)

    model = _get_model()
    encoded = model.encode(
        non_empty_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=False,  # 归一化在 FAISS 层统一处理
    ).astype(np.float32)

    # 将编码结果还原到原始位置，空字符串位置保持零向量
    vectors = np.zeros((len(texts), config.EMBEDDING_DIM), dtype=np.float32)
    for out_pos, orig_idx in enumerate(non_empty_indices):
        vectors[orig_idx] = encoded[out_pos]

    logger.debug("encode_texts: 编码 %d 条文本，shape=%s", len(texts), vectors.shape)
    return vectors


def serialize_vector(vec: np.ndarray) -> bytes:
    """
    将 Embedding 向量序列化为二进制，用于存入数据库 BLOB 字段。

    Args:
        vec: shape=(384,) 的 float32 向量

    Returns:
        bytes，长度 = 384 * 4 = 1536 字节
    """
    vec = vec.astype(np.float32).flatten()
    if vec.shape != (config.EMBEDDING_DIM,):
        raise ValueError(
            f"向量维度错误: {vec.shape}，期望 ({config.EMBEDDING_DIM},)"
        )
    return vec.tobytes()


def deserialize_vector(data: bytes) -> np.ndarray:
    """
    从二进制反序列化为 Embedding 向量。

    Args:
        data: bytes，长度应为 384 * 4 = 1536 字节

    Returns:
        np.ndarray, shape=(384,), dtype=float32
    """
    expected_bytes = config.EMBEDDING_DIM * 4  # float32 = 4 字节
    if len(data) != expected_bytes:
        raise ValueError(
            f"反序列化失败：数据长度 {len(data)} 字节，"
            f"期望 {expected_bytes} 字节（{config.EMBEDDING_DIM} 维 float32）"
        )
    vec = np.frombuffer(data, dtype=np.float32).copy()
    return vec.reshape(config.EMBEDDING_DIM)


def encode_single(text: str) -> np.ndarray:
    """
    编码单条文本，返回 shape=(384,) 的向量（检索时使用）。
    """
    vecs = encode_texts([text])
    return vecs[0]
