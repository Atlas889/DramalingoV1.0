"""
DramaLingo · 序列对齐模块（多信号增强版）

对齐逻辑（每个知识节点 → 寻找最匹配的 Whisper 片段）：
1. 在字幕时间戳 ±window_seconds 窗口内搜索候选 Whisper 片段
2. 候选预过滤：长度比例 ≥ ALIGN_MIN_LEN_RATIO（短文本不匹配长片段）
3. 综合评分（三信号融合）：
     score = lev × ALIGN_LEV_WEIGHT
           + emb_cos × ALIGN_EMB_WEIGHT
           + order_bonus(候选在游标后) 或 -order_penalty(候选在游标前)
4. 短文本（词数 ≤ 3）要求 sim ≥ ALIGN_SHORT_TEXT_MIN_SIM，过滤 "Hi"/"Yeah" 噪声
5. best/second margin < ALIGN_AMBIGUOUS_MARGIN → 标记 align_step="ambiguous"
6. 已被占用的片段排除，防止重复匹配
7. 无论 in_order 是否成立，都更新游标，避免一次乱序拖累后续连锁错位
"""

import logging
import re
from typing import List, Optional, Tuple

import numpy as np
from Levenshtein import ratio as levenshtein_ratio

import config
from services.deduplicator import normalize_text

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 对齐专用文本归一化：在 dedup 用的 normalize_text 之上，
# 再剥离 Whisper filler 与 SRT 括号注释，提升匹配质量。
# 不复用到去重路径，避免影响节点入库判重逻辑。
# ──────────────────────────────────────────────

# Whisper 高频填充词（在归一化后的小写无标点文本上匹配）
_FILLER_PATTERNS = [
    re.compile(r"\b(?:uh|um|uhm|er|erm|ah|oh|hmm|mm|mhm)\b"),
    re.compile(r"\b(?:you know|i mean|like i said|kind of|sort of)\b"),
]

# SRT 描述性标注（在原文上匹配，归一化前先剥离）：[door slams] / (whispering) / ♪ music ♪
_ANNOTATION_PATTERNS = [
    re.compile(r"\[[^\]]*\]"),     # [ ... ]
    re.compile(r"\([^\)]*\)"),     # ( ... )
    re.compile(r"♪[^♪]*♪"),         # ♪ ... ♪
    re.compile(r"<[^>]+>"),         # <i>...</i> 等 HTML 风格标签
]


def normalize_for_alignment(text: str) -> str:
    """对齐专用归一化：normalize_text + 剥离括号注释 + 剥离 filler。"""
    if not text:
        return ""
    # 先剥离括号/方括号/音乐符号注释（保留原大小写以便后续 normalize_text 处理）
    for pat in _ANNOTATION_PATTERNS:
        text = pat.sub(" ", text)
    # 标准归一化（小写/缩写展开/去标点）
    text = normalize_text(text)
    # 剥离 filler（多遍：filler 可能相邻）
    for pat in _FILLER_PATTERNS:
        text = pat.sub(" ", text)
    # 合并多余空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


def text_similarity(text1: str, text2: str) -> float:
    """Levenshtein ratio，先做对齐专用归一化（剥离 filler / 括号注释）"""
    norm1 = normalize_for_alignment(text1)
    norm2 = normalize_for_alignment(text2)
    if not norm1 and not norm2:
        return 1.0
    if not norm1 or not norm2:
        return 0.0
    return levenshtein_ratio(norm1, norm2)


def _length_ratio(a: str, b: str) -> float:
    """词数比例 min/max，用于候选预过滤"""
    la, lb = len(a.split()), len(b.split())
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """两向量余弦相似度（输入未归一化也安全）；任一为 None 返回 0"""
    if a is None or b is None:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def is_in_covered_range(start: float, end: float, covered_ranges: List[Tuple[float, float]]) -> bool:
    """判断时间区间是否与已覆盖区间有重叠"""
    for cov_start, cov_end in covered_ranges:
        if not (end <= cov_start or start >= cov_end):
            return True
    return False


def align_seq(
    nodes: List[dict],
    transcripts: List[dict],
    window_seconds: float = None,
) -> List[dict]:
    """
    序列对齐：每个节点在字幕时间戳窗口内寻找最佳 Whisper 片段。

    Args:
        nodes: 节点列表 [{node_id, en_text, start_time_hint, embedding(optional np.ndarray)}]
        transcripts: Whisper 序列 [{text, start_time, end_time, seq_index, embedding(optional)}]
        window_seconds: 窗口半径，默认 config.WHISPER_ALIGN_WINDOW_SECONDS

    Returns:
        [{node_id, whisper_start, whisper_end, confidence, text_similarity,
          align_step, whisper_text}, ...]
    """
    if window_seconds is None:
        window_seconds = config.WHISPER_ALIGN_WINDOW_SECONDS

    if not nodes or not transcripts:
        return []

    use_emb = config.ALIGN_USE_EMBEDDING
    w_lev = config.ALIGN_LEV_WEIGHT
    w_emb = config.ALIGN_EMB_WEIGHT
    w_order_pos = config.ALIGN_ORDER_BONUS
    w_order_neg = config.ALIGN_ORDER_PENALTY
    min_len_ratio = config.ALIGN_MIN_LEN_RATIO
    margin_thresh = config.ALIGN_AMBIGUOUS_MARGIN
    short_words = config.ALIGN_SHORT_TEXT_WORDS
    short_min_sim = config.ALIGN_SHORT_TEXT_MIN_SIM

    results = []
    last_matched_seq_index = -1
    used_seq_indices: set = set()

    for node in nodes:
        node_id = node["node_id"]
        en_text = node["en_text"]
        start_hint = node.get("start_time_hint", 0.0)
        node_emb = node.get("embedding") if use_emb else None

        en_norm = normalize_for_alignment(en_text)
        node_words = len(en_norm.split())
        is_short = node_words <= short_words

        window_start = max(0.0, start_hint - window_seconds)
        window_end = start_hint + window_seconds

        # 窗口内未占用候选
        candidates = [
            t for t in transcripts
            if t["start_time"] < window_end and t["end_time"] > window_start
            and t["seq_index"] not in used_seq_indices
        ]

        if not candidates:
            logger.debug(f"节点 {node_id} 窗口内无候选片段")
            continue

        # 长度比例预过滤
        candidates = [
            t for t in candidates
            if _length_ratio(en_norm, normalize_for_alignment(t["text"])) >= min_len_ratio
        ]
        if not candidates:
            continue

        # 计算每个候选的综合分
        scored = []  # [(score, text_sim, candidate, in_order)]
        for cand in candidates:
            t_sim = text_similarity(en_text, cand["text"])
            cos_sim = _cosine(node_emb, cand.get("embedding")) if use_emb else 0.0
            in_order = cand["seq_index"] > last_matched_seq_index
            order_term = w_order_pos if in_order else -w_order_neg

            if use_emb and cos_sim > 0:
                base = w_lev * t_sim + w_emb * cos_sim
            else:
                # 无 embedding 时退化为纯 Levenshtein（权重归一）
                base = (w_lev + w_emb) * t_sim
            score = base + order_term
            scored.append((score, t_sim, cos_sim, cand, in_order))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_text_sim, best_cos, best_match, best_in_order = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        margin = best_score - second_score

        # 短文本严格门槛：避免 "Hi"/"Yeah" 在长候选中误命中
        if is_short and best_text_sim < short_min_sim:
            logger.debug(
                f"节点 {node_id} 短文本 ({node_words} 词) 相似度 {best_text_sim:.2f} 未达 {short_min_sim}"
            )
            continue

        if best_text_sim < 0.3:
            continue

        # 置信度：text_sim 主导，emb 辅助；不含 order 项（仅用于排序）
        if use_emb and best_cos > 0:
            confidence = w_lev * best_text_sim + w_emb * best_cos
            # 归一化到 [0,1]：分母 = w_lev + w_emb
            confidence = confidence / (w_lev + w_emb)
        else:
            confidence = best_text_sim

        if confidence < 0.5:
            continue

        # 模糊匹配标记：与次优差距过小
        align_step = "ambiguous" if margin < margin_thresh else "seq"

        results.append({
            "node_id": node_id,
            "whisper_start": best_match["start_time"],
            "whisper_end": best_match["end_time"],
            "confidence": round(confidence, 4),
            "text_similarity": round(best_text_sim, 4),
            "embedding_similarity": round(best_cos, 4) if use_emb else 0.0,
            "score_margin": round(margin, 4),
            "align_step": align_step,
            "whisper_text": best_match["text"],
        })

        # 始终更新游标 —— 即使本次乱序匹配也要推进，避免一次乱序拖累后续连锁错位
        used_seq_indices.add(best_match["seq_index"])
        last_matched_seq_index = max(last_matched_seq_index, best_match["seq_index"])

    logger.info(
        f"序列对齐完成: {len(nodes)} 个节点 → {len(results)} 个匹配 "
        f"(use_embedding={use_emb})"
    )
    return results
