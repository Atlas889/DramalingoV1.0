"""
DramaLingo · 视频切片流水线
对应 PRD v0.9 §5 流程二 / 开发文档 Step 9

完整流水线（generate_clips 主入口）：
  Whisper 转写 → 零散片段合并 → 查询本集知识节点 →
  序列对齐（DTW+Levenshtein） → 规划切点 → FFmpeg 并发切片 →
  生成缩略图 → 写入 clips 表 → 更新 has_any_clip 汇总字段

切片置信度门禁（C 类约束）：
  C1  置信度 ≥ 0.85 → 高置信，直接切片（align_step='seq'）
  C2  0.60 ≤ 置信度 < 0.85 → 中置信，仍切片（align_step='seq'）
  C3  置信度 < 0.60  → 不切片；若字幕时间戳可信则字幕兜底（align_step='subtitle_fallback'）

辅助函数：
  merge_transcript_segments  — 合并 Whisper 零散片段为完整句子
  ffmpeg_cut_clip            — 调用 FFmpeg 切片（自动选流复制 vs 重编码）
  ffmpeg_generate_thumbnail  — 从切片取 0.5s 处帧生成缩略图
  _find_context_window       — 在主片段前后扩展上下文窗口
"""

import asyncio
import logging
import os
import secrets
import subprocess
import time
from typing import List, Optional

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

import config
from models.clip import Clip
from models.knowledge_node import KnowledgeNode
from services.whisper_service import transcribe_video
from services.aligner import align_seq, normalize_for_alignment
from services.embedder import encode_texts, deserialize_vector

logger = logging.getLogger(__name__)

# 置信度阈值
HIGH_CONF_THRESHOLD = 0.85   # 高置信：直接切片
LOW_CONF_THRESHOLD = 0.60    # 中置信：仍切片，依赖 Whisper 合并后的时间精度


def merge_transcript_segments(
    transcripts: List[dict],
    max_gap: float = None,
    max_duration: float = None,
    max_words: int = None,
) -> List[dict]:
    """
    将 Whisper 零散转写片段合并为语义完整的句子。

    合并条件（同时满足）：
    1. 相邻片段时间间隔 ≤ max_gap
    2. 合并后总时长 ≤ max_duration
    3. 合并后词数 ≤ max_words（防止跨说话人过度合并）
    4. 当前片段末尾未出现句子结束标点（. ? ! ...）
       或：末尾为逗号但当前段已 ≥ 8 词（视为独立从句边界）

    Args:
        transcripts: Whisper 原始片段列表，每项含 seq_index/text/start_time/end_time
        max_gap:      相邻片段最大间隔（秒）
        max_duration: 合并后单段最大时长（秒）
        max_words:    合并后最大词数

    Returns:
        合并后的片段列表（seq_index 保留第一个片段的值）
    """
    if max_gap is None:
        max_gap = config.WHISPER_MERGE_GAP_SECONDS
    if max_duration is None:
        max_duration = config.WHISPER_MERGE_MAX_DURATION
    if max_words is None:
        max_words = config.WHISPER_MERGE_MAX_WORDS

    if not transcripts:
        return []

    hard_gap = config.WHISPER_MERGE_HARD_GAP

    merged: List[dict] = []
    cur = dict(transcripts[0])

    for nxt in transcripts[1:]:
        gap = nxt["start_time"] - cur["end_time"]
        combined_dur = nxt["end_time"] - cur["start_time"]
        cur_text = cur["text"].rstrip()
        nxt_text_lstripped = nxt["text"].lstrip()
        cur_words = len(cur_text.split())
        combined_words = cur_words + len(nxt_text_lstripped.split())

        # 边界检测：标点 + 静音硬边界 + 大小写信号（多信号融合）
        ends_sentence = (
            cur_text.endswith((".", "?", "!", "..."))
            or (cur_text.endswith(",") and cur_words >= 8)
        )
        # 静音硬边界：超过阈值视为场景/说话人切换，无论是否有标点都不合并
        hard_silence = gap > hard_gap
        # 大小写信号：当前段以小写收尾（非缩写），下一段以大写起首，且 gap 较长（> 0.6s）
        ends_with_lower = bool(cur_text) and cur_text[-1].islower()
        starts_with_upper = bool(nxt_text_lstripped) and nxt_text_lstripped[0].isupper()
        case_boundary = ends_with_lower and starts_with_upper and gap > 0.6

        if (gap <= max_gap
                and combined_dur <= max_duration
                and combined_words <= max_words
                and not ends_sentence
                and not hard_silence
                and not case_boundary):
            cur["text"] = cur_text + " " + nxt_text_lstripped
            cur["end_time"] = nxt["end_time"]
        else:
            merged.append(cur)
            cur = dict(nxt)

    merged.append(cur)
    return merged


def _gen_clip_code(show_name: str, episode: str) -> str:
    """生成切片业务编码，格式：CLP-{show2}-{ep2}-{rand6大写hex}。
    示例：CLP-FR-S1-A3F8C2 / CLP-CO-01-7D1E44"""
    sn = (show_name or "XX")[:2].upper()
    ep = (episode or "00")[:2].upper()
    rand6 = secrets.token_hex(3).upper()
    return f"CLP-{sn}-{ep}-{rand6}"


def _make_clip(
    node,
    show_name: str,
    episode: str,
    align_result: dict,
    clip_status: str,
    is_active: bool,
    stream_url: str = "",
    thumb_url: str = "",
    actual_duration: Optional[float] = None,
) -> Clip:
    """构造 Clip ORM 对象，避免四处重复相同字段填充。
    actual_duration: 含上下文窗口 + padding 的真实切割时长（cut_end - cut_start），
                     仅 matched 切片有效；失败/未匹配记录退化为 Whisper 片段时长。
    """
    dur = round(actual_duration, 2) if actual_duration is not None else \
          round(align_result["whisper_end"] - align_result["whisper_start"], 2)
    return Clip(
        clip_code=_gen_clip_code(show_name, episode),
        node_id=node.id,
        show_name=show_name,
        episode=episode,
        subtitle_en=node.en_text,
        subtitle_zh=node.zh_text or "",
        scene=node.scene or "daily",
        emotion=node.emotion or "neutral",
        difficulty=node.difficulty or "intermediate",
        keywords=node.keywords or [],
        stream_url=stream_url,
        thumb_url=thumb_url,
        clip_status=clip_status,
        is_active=is_active,
        duration=dur,
        word_count=node.word_count,
        match_confidence=align_result["confidence"],
        whisper_start=align_result["whisper_start"],
        whisper_end=align_result["whisper_end"],
        whisper_text=align_result.get("whisper_text", ""),
        align_step=align_result["align_step"],
    )


def _find_context_window(
    main_start: float,
    main_end: float,
    transcripts: List[dict],
) -> tuple:
    """
    在主片段 [main_start, main_end] 前后各取 config.CLIP_CONTEXT_BEFORE /
    CLIP_CONTEXT_AFTER 个相邻 Whisper 转写段，返回扩展后的时间窗口。

    场景断点保护（新增）：
    - 邻段静音 gap > CLIP_CONTEXT_SCENE_GAP → 视为场景/说话人切换，不追加
    - 邻段时长 > CLIP_CONTEXT_NEIGHBOR_MAX_DUR → 视为独立长句，不追加（避免吞掉另一句完整对白）

    其他约束：
    - 单方向额外时长不超过 CLIP_CONTEXT_MAX_EXTRA_S 秒
    - 扩展后总时长（含 padding）超过 CLIP_MAX_DURATION 时退回主片段

    Returns:
        (ctx_start, ctx_end) — 扩展后的裸时间窗口（不含 padding）
    """
    before_count = config.CLIP_CONTEXT_BEFORE
    after_count  = config.CLIP_CONTEXT_AFTER
    max_extra    = config.CLIP_CONTEXT_MAX_EXTRA_S
    scene_gap    = config.CLIP_CONTEXT_SCENE_GAP
    neighbor_max = config.CLIP_CONTEXT_NEIGHBOR_MAX_DUR

    # 在主片段开始前结束的段落（按时间逆序取最后 N 个）
    before_segs = [t for t in transcripts if t["end_time"] <= main_start + 0.05]
    before_segs = before_segs[-before_count:] if before_count and before_segs else []

    # 在主片段结束后开始的段落（取前 N 个）
    after_segs = [t for t in transcripts if t["start_time"] >= main_end - 0.05]
    after_segs = after_segs[:after_count] if after_count and after_segs else []

    ctx_start = main_start
    ctx_end   = main_end

    # 向前扩展：从最近的邻段开始，逐段判断场景断点 / 邻段过长
    if before_segs:
        # 从距离主片段最近的邻段开始（before_segs[-1]），向更早的方向扩展
        for seg in reversed(before_segs):
            gap = ctx_start - seg["end_time"]
            seg_dur = seg["end_time"] - seg["start_time"]
            if gap > scene_gap:
                break  # 场景断点：停止前扩
            if seg_dur > neighbor_max:
                break  # 邻段过长：停止前扩
            new_start = seg["start_time"]
            if (main_start - new_start) > max_extra:
                break  # 单方向额外时长超限
            ctx_start = new_start

    # 向后扩展：同理
    if after_segs:
        for seg in after_segs:
            gap = seg["start_time"] - ctx_end
            seg_dur = seg["end_time"] - seg["start_time"]
            if gap > scene_gap:
                break
            if seg_dur > neighbor_max:
                break
            new_end = seg["end_time"]
            if (new_end - main_end) > max_extra:
                break
            ctx_end = new_end

    # 整体超限则退回主片段（padding 由调用方决定）
    padded_total = (ctx_end - ctx_start) + config.CLIP_PADDING_START + config.CLIP_PADDING_END
    if padded_total > config.CLIP_MAX_DURATION:
        ctx_start = main_start
        ctx_end   = main_end

    return ctx_start, ctx_end


def _anchor_to_nearest_segment(
    target_time: float,
    transcripts: List[dict],
    search_radius: float = 5.0,
    anchor_kind: str = "start",
) -> float:
    """
    将 SRT 时间戳锚定到最近的 Whisper 段端点（仅用于 subtitle_fallback）。

    SRT 字幕的时间戳通常有 1-3s 偏差，直接使用会导致切片开头被吞或结尾拖泥带水。
    在 ±search_radius 范围内寻找最近的 Whisper 段对应端点：
    - anchor_kind="start": 寻找最近的 Whisper 段 start_time
    - anchor_kind="end":   寻找最近的 Whisper 段 end_time

    找不到合适锚点 → 返回原值
    """
    if not transcripts:
        return target_time

    field = "start_time" if anchor_kind == "start" else "end_time"
    best_time = target_time
    best_dist = search_radius
    for t in transcripts:
        candidate = t[field]
        dist = abs(candidate - target_time)
        if dist < best_dist:
            best_dist = dist
            best_time = candidate
    return best_time


def _snap_cut_to_boundary(
    cut_start: float,
    cut_end: float,
    transcripts: List[dict],
) -> tuple:
    """
    将切片端点 snap 到最近的 Whisper 段间静音边界，避免掐头去尾。

    - 在 [CLIP_SNAP_TO_BOUNDARY_MIN, CLIP_SNAP_TO_BOUNDARY_MAX] 范围内寻找
      与原 cut 点最接近的相邻段间隙中点
    - 找不到合适静音 → 保持原 cut 点（退回默认 padding）

    Returns: (snapped_start, snapped_end)
    """
    if not transcripts:
        return cut_start, cut_end

    snap_min = config.CLIP_SNAP_TO_BOUNDARY_MIN
    snap_max = config.CLIP_SNAP_TO_BOUNDARY_MAX

    # 寻找最佳起点 snap：候选 = 落在 [cut_start - snap_max, cut_start + snap_min] 区间的"段间隙"中点
    # 段间隙中点：transcripts[i].end_time 到 transcripts[i+1].start_time 之间的中点
    snap_start = cut_start
    best_dist = float("inf")
    for i in range(len(transcripts) - 1):
        gap_start = transcripts[i]["end_time"]
        gap_end = transcripts[i + 1]["start_time"]
        if gap_end <= gap_start:
            continue
        mid = (gap_start + gap_end) / 2.0
        dist = abs(mid - cut_start)
        if snap_min <= dist <= snap_max and dist < best_dist:
            best_dist = dist
            snap_start = mid

    # 同理找终点 snap
    snap_end = cut_end
    best_dist = float("inf")
    for i in range(len(transcripts) - 1):
        gap_start = transcripts[i]["end_time"]
        gap_end = transcripts[i + 1]["start_time"]
        if gap_end <= gap_start:
            continue
        mid = (gap_start + gap_end) / 2.0
        dist = abs(mid - cut_end)
        if snap_min <= dist <= snap_max and dist < best_dist:
            best_dist = dist
            snap_end = mid

    # 保险：snap 后端点交叉则退回原值
    if snap_end <= snap_start + 0.2:
        return cut_start, cut_end

    return snap_start, snap_end


# 支持流复制（无损）的容器格式；RMVB / AVI / WMV / FLV 等需要重编码
_STREAM_COPY_EXTS = {".mp4", ".mkv", ".mov", ".m4v", ".ts", ".mts", ".webm"}


def ffmpeg_cut_clip(
    video_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
) -> bool:
    """
    使用 FFmpeg 切片，自动选择编码策略：
    - MP4/MKV/MOV 等：-c copy 无损切片（快）
    - RMVB/AVI/WMV 等：libx264+aac 重编码（兼容性）

    调用方负责应用 CLIP_PADDING_START / CLIP_PADDING_END 后再传入 start_time/end_time。
    """
    actual_start = max(0.0, start_time)
    duration = end_time - actual_start

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    src_ext = os.path.splitext(video_path)[1].lower()
    if src_ext in _STREAM_COPY_EXTS:
        codec_args = ["-c", "copy", "-avoid_negative_ts", "make_zero"]
        timeout = 60
    else:
        # RMVB / AVI / WMV / FLV：重编码为 H.264+AAC MP4
        codec_args = ["-c:v", "libx264", "-c:a", "aac", "-preset", "fast", "-crf", "23"]
        timeout = 300

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(actual_start),
        "-i", video_path,
        "-t", str(duration),
        *codec_args,
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        logger.warning(f"FFmpeg 切片失败 [{src_ext}]: {stderr[-500:]}")
        return False
    return True


def ffmpeg_generate_thumbnail(clip_path: str, thumb_path: str) -> bool:
    """从视频切片取 0.5s 处帧生成缩略图（避免开头黑帧或转场帧）"""
    cmd = [
        "ffmpeg", "-y",
        "-ss", "0.5",
        "-i", clip_path,
        "-vframes", "1",
        "-q:v", "2",
        thumb_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=30)
    if result.returncode != 0:
        # 若 0.5s 处无帧（片段 < 0.5s），降级为第 1 帧
        cmd2 = ["ffmpeg", "-y", "-i", clip_path, "-vframes", "1", "-q:v", "2", thumb_path]
        result = subprocess.run(cmd2, capture_output=True, timeout=30)
    return result.returncode == 0


async def generate_clips(
    video_path: str,
    show_name: str,
    episode: str,
    session: AsyncSession,
    force_retranscribe: bool = False,
    node_ids: Optional[List[int]] = None,
    task_id: Optional[str] = None,
) -> dict:
    """
    主入口：完整的视频切片流水线
    
    流程：
    1. Whisper 转写
    2. 合并零散片段
    3. 查询本集知识节点
    4. 序列对齐
    5. 规划 + 并发 FFmpeg 切片
    6. 写入数据库
    
    Args:
        video_path: 视频文件路径
        show_name: 剧名
        episode: 集号
        session: 数据库会话
        force_retranscribe: 是否强制重新转写
    
    Returns:
        统计信息字典
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    _video_name = f"{show_name} {episode}"
    _step_results: dict = {}
    _pipeline_start = time.monotonic()

    def _rpt(stage: str, label: str, pct: int) -> None:
        if not task_id:
            return
        from tasks.task_runner import update_task_progress
        update_task_progress(task_id, {
            "task_type": "clip",
            "video_name": _video_name,
            "stage": stage,
            "label": label,
            "pct": pct,
            "step_results": dict(_step_results),
            "total_elapsed_s": round(time.monotonic() - _pipeline_start, 1),
        })

    stats = {
        "show_name": show_name,
        "episode": episode,
        "transcript_count": 0,
        "nodes_total": 0,
        "high_conf": 0,
        "low_conf": 0,
        "unmatched": 0,
        "failed": 0,
        "clips_generated": 0,
        "subtitle_fallback": 0,  # C1: 字幕时间戳兜底切片数
    }

    # ──────────────────────────────────────────────
    # Step 1: Whisper 转写
    # ──────────────────────────────────────────────
    logger.info(f"开始切片流水线: {show_name} {episode}")
    _rpt("whisper", "Whisper 转写中…", 5)
    _t0 = time.monotonic()
    transcripts = await transcribe_video(
        video_path=video_path,
        show_name=show_name,
        episode=episode,
        session=session,
        force_retranscribe=force_retranscribe,
    )
    stats["transcript_count"] = len(transcripts)
    _step_results["whisper"] = {"count": len(transcripts), "elapsed_s": round(time.monotonic() - _t0, 1)}
    _rpt("whisper", f"Whisper 转写完成：{len(transcripts)} 个片段", 20)
    logger.info(f"转写完成: {len(transcripts)} 个片段")

    if not transcripts:
        logger.warning("转写结果为空，跳过切片")
        return stats

    # ──────────────────────────────────────────────
    # Step 1b: 合并零散转写片段为完整句子
    # ──────────────────────────────────────────────
    _t0 = time.monotonic()
    raw_count = len(transcripts)
    transcripts = merge_transcript_segments(transcripts)
    _step_results["merge"] = {
        "before": raw_count,
        "after": len(transcripts),
        "elapsed_s": round(time.monotonic() - _t0, 2),
    }
    _rpt("merge", f"片段合并：{raw_count} → {len(transcripts)} 段", 25)
    logger.info(f"Whisper 片段合并: {raw_count} → {len(transcripts)} 段")

    # ──────────────────────────────────────────────
    # Step 2: 查询本集知识节点（不再跨集匹配）
    # ──────────────────────────────────────────────
    # 不按 quality_score 过滤：字幕质量门禁已在字幕处理阶段完成，
    # 切片阶段应对所有激活的本集节点生成切片。
    stmt_same = select(KnowledgeNode).where(
        KnowledgeNode.show_name == show_name,
        KnowledgeNode.episode == episode,
        KnowledgeNode.is_active == True,
        *([KnowledgeNode.id.in_(node_ids)] if node_ids else []),
    ).order_by(KnowledgeNode.start_time)

    result_same = await session.execute(stmt_same)
    same_episode_nodes = result_same.scalars().all()

    stats["nodes_total"] = len(same_episode_nodes)
    _step_results["query_nodes"] = {
        "same": len(same_episode_nodes),
        "show_name": show_name,
        "episode": episode or "(无集号)",
    }
    _rpt("query_nodes", f"知识节点：本集 {len(same_episode_nodes)} 个（{show_name} {episode or '无集号'}）", 30)
    logger.info(f"知识节点: 本集 {len(same_episode_nodes)} 个 [{show_name} episode={repr(episode)}]")

    # ──────────────────────────────────────────────
    # Step 3: 序列对齐（仅本集节点，融合 Levenshtein + Embedding 双信号）
    # ──────────────────────────────────────────────
    _rpt("alignment", "序列对齐中…", 35)
    _t0 = time.monotonic()
    loop = asyncio.get_running_loop()

    # 预计算 embedding：节点用入库时的存储向量，Whisper 段批量编码一次
    node_embeddings: dict = {}
    if config.ALIGN_USE_EMBEDDING:
        # 节点 embedding：优先复用入库时的存储向量（en_embedding 优先，回退 embedding）
        nodes_need_encode = []
        for n in same_episode_nodes:
            stored = getattr(n, "embedding", None)
            if stored:
                try:
                    node_embeddings[n.id] = deserialize_vector(stored)
                except Exception as e:
                    logger.warning(f"节点 {n.id} embedding 反序列化失败，将重新编码: {e}")
                    nodes_need_encode.append(n)
            else:
                nodes_need_encode.append(n)

        # 缺失 embedding 的节点统一批量编码
        if nodes_need_encode:
            texts = [normalize_for_alignment(n.en_text) for n in nodes_need_encode]
            vecs = await loop.run_in_executor(None, encode_texts, texts)
            for n, v in zip(nodes_need_encode, vecs):
                node_embeddings[n.id] = v

        # Whisper 段批量编码（用对齐专用归一化文本）
        transcript_texts = [normalize_for_alignment(t["text"]) for t in transcripts]
        transcript_vecs = await loop.run_in_executor(None, encode_texts, transcript_texts)
        # 写回 transcripts dict（不修改原对象的其他字段）
        transcripts = [
            {**t, "embedding": transcript_vecs[i]}
            for i, t in enumerate(transcripts)
        ]

    same_nodes_input = [
        {
            "node_id": n.id,
            "en_text": n.en_text,
            "start_time_hint": n.start_time or 0.0,
            "embedding": node_embeddings.get(n.id),
        }
        for n in same_episode_nodes
    ]

    all_results = await loop.run_in_executor(
        None, align_seq, same_nodes_input, transcripts
    )

    _step_results["alignment"] = {
        "seq": len(all_results),
        "elapsed_s": round(time.monotonic() - _t0, 1),
    }
    _rpt("alignment", f"对齐完成：{len(all_results)} 个匹配", 50)
    logger.info(f"对齐完成: {len(all_results)} 个匹配")
    
    # ──────────────────────────────────────────────
    # Step 4a: 收集所有待切任务（对齐结果 + C1 兜底）
    # ──────────────────────────────────────────────
    _rpt("collect_jobs", "规划切片任务…", 52)
    _t0 = time.monotonic()
    clip_dir = os.path.join(str(config.MEDIA_DIR), f"{show_name}_{episode}")
    os.makedirs(clip_dir, exist_ok=True)

    node_map = {n.id: n for n in same_episode_nodes}

    # cut_jobs: 每项为 dict，描述一个待执行的 FFmpeg 切片任务
    cut_jobs: List[dict] = []
    # immediate_records: 不需要 FFmpeg 的记录（超限 / 未匹配）
    immediate_records: List[Clip] = []

    # ── 对齐结果 → 生成任务 ──
    for align_result in all_results:
        node_id = align_result["node_id"]
        whisper_start = align_result["whisper_start"]
        whisper_end = align_result["whisper_end"]
        confidence = align_result["confidence"]
        node = node_map.get(node_id)
        if node is None:
            logger.warning(f"节点 {node_id} 不在查询结果中，跳过")
            continue

        # B2: 时长检查用含 padding 的实际切片时长
        padded_dur = (whisper_end - whisper_start) + config.CLIP_PADDING_START + config.CLIP_PADDING_END
        if padded_dur > config.CLIP_MAX_DURATION:
            logger.debug(f"节点 {node_id} 含padding时长 {padded_dur:.1f}s 超限")
            immediate_records.append(_make_clip(node, show_name, episode, align_result, "failed", False))
            stats["failed"] += 1
            continue
        if padded_dur < config.CLIP_MIN_DURATION:
            logger.debug(f"节点 {node_id} 含padding时长 {padded_dur:.1f}s 过短")
            stats["failed"] += 1
            continue

        if confidence >= LOW_CONF_THRESHOLD:
            clip_filename = f"clip_{node_id}.mp4"
            # 扩展上下文：在主片段前后追加相邻对话段
            ctx_start, ctx_end = _find_context_window(whisper_start, whisper_end, transcripts)
            raw_start = max(0.0, ctx_start - config.CLIP_PADDING_START)
            raw_end = ctx_end + config.CLIP_PADDING_END
            # Snap 到最近的静音边界，避免掐头去尾
            cut_start, cut_end = _snap_cut_to_boundary(raw_start, raw_end, transcripts)
            cut_jobs.append({
                "node": node,
                "align_result": align_result,
                "cut_start": cut_start,
                "cut_end": cut_end,
                "clip_path": os.path.join(clip_dir, clip_filename),
                "thumb_path": os.path.join(clip_dir, f"thumb_{node_id}.jpg"),
                "stream_url": f"/media/{show_name}_{episode}/{clip_filename}",
                "thumb_url": f"/media/{show_name}_{episode}/thumb_{node_id}.jpg",
                "is_fallback": False,
                "confidence": confidence,
            })
        else:
            stats["unmatched"] += 1

    # ── C1 字幕兜底 → 追加任务 ──
    planned_ids = {j["node"].id for j in cut_jobs}
    fallback_nodes = [
        n for n in same_episode_nodes
        if n.id not in planned_ids
        and n.start_time is not None
        and n.end_time is not None
        # B2: 兜底也用含 padding 的实际时长做校验
        and config.CLIP_MIN_DURATION <= (
            (n.end_time - n.start_time) + config.CLIP_PADDING_START + config.CLIP_PADDING_END
        ) <= config.CLIP_MAX_DURATION
    ]
    logger.info(f"C1 字幕兜底: {len(fallback_nodes)} 个节点")

    for node in fallback_nodes:
        # ── 字幕时间戳锚点修正：在 ±5s 内寻找最近的 Whisper 段端点 ──
        # SRT 字幕时间戳常有 1-3s 偏差，直接使用会导致开头被吞或结尾拖泥带水
        anchor_start = _anchor_to_nearest_segment(node.start_time, transcripts, search_radius=5.0, anchor_kind="start")
        anchor_end = _anchor_to_nearest_segment(node.end_time, transcripts, search_radius=5.0, anchor_kind="end")
        # 保险：anchor 后端点交叉则退回原值
        if anchor_end <= anchor_start + 0.2:
            anchor_start, anchor_end = node.start_time, node.end_time

        fallback_result = {
            "node_id": node.id,
            "whisper_start": anchor_start,
            "whisper_end": anchor_end,
            "confidence": 0.65,
            "text_similarity": 0.65,
            "align_step": "subtitle_fallback",
            "whisper_text": "",
        }
        clip_filename = f"clip_{node.id}.mp4"
        # 扩展上下文：在修正后的端点前后追加相邻对话段
        ctx_start, ctx_end = _find_context_window(anchor_start, anchor_end, transcripts)
        raw_start = max(0.0, ctx_start - config.CLIP_PADDING_START)
        raw_end = ctx_end + config.CLIP_PADDING_END
        cut_start, cut_end = _snap_cut_to_boundary(raw_start, raw_end, transcripts)
        cut_jobs.append({
            "node": node,
            "align_result": fallback_result,
            "cut_start": cut_start,
            "cut_end": cut_end,
            "clip_path": os.path.join(clip_dir, clip_filename),
            "thumb_path": os.path.join(clip_dir, f"thumb_{node.id}.jpg"),
            "stream_url": f"/media/{show_name}_{episode}/{clip_filename}",
            "thumb_url": f"/media/{show_name}_{episode}/thumb_{node.id}.jpg",
            "is_fallback": True,
            "confidence": 0.65,
        })

    # 上报 collect_jobs 步骤结果
    _aligned_count = len([j for j in cut_jobs if not j["is_fallback"]])
    _fallback_count = len([j for j in cut_jobs if j["is_fallback"]])
    _step_results["collect_jobs"] = {
        "aligned": _aligned_count,
        "fallback": _fallback_count,
        "total": len(cut_jobs),
        "immediate_failed": len(immediate_records),
        "elapsed_s": round(time.monotonic() - _t0, 2),
    }
    _rpt("collect_jobs",
         f"规划完成：对齐 {_aligned_count} · 字幕兜底 {_fallback_count} · 超限跳过 {len(immediate_records)}",
         54)

    # ──────────────────────────────────────────────
    # Step 4b: 并发执行所有 FFmpeg 切片任务
    # ──────────────────────────────────────────────
    _rpt("ffmpeg", f"FFmpeg 切片中…（共 {len(cut_jobs)} 个，并发 {config.FFMPEG_CONCURRENT}）", 58)
    ffmpeg_sem = asyncio.Semaphore(config.FFMPEG_CONCURRENT)

    async def _do_cut(job: dict) -> bool:
        async with ffmpeg_sem:
            ok = await loop.run_in_executor(
                None, ffmpeg_cut_clip, video_path, job["clip_path"], job["cut_start"], job["cut_end"]
            )
            if ok:
                await loop.run_in_executor(
                    None, ffmpeg_generate_thumbnail, job["clip_path"], job["thumb_path"]
                )
            return ok

    cut_results = await asyncio.gather(*[_do_cut(j) for j in cut_jobs])

    # ── 根据切片结果构建 DB 记录 ──
    clips_to_add: List[Clip] = list(immediate_records)
    nodes_with_clips: set = set()

    for job, ok in zip(cut_jobs, cut_results):
        node = job["node"]
        ar = job["align_result"]
        conf = job["confidence"]
        if ok:
            clips_to_add.append(
                _make_clip(node, show_name, episode, ar, "matched", True,
                           stream_url=job["stream_url"], thumb_url=job["thumb_url"],
                           actual_duration=job["cut_end"] - job["cut_start"])
            )
            nodes_with_clips.add(node.id)
            stats["clips_generated"] += 1
            if job["is_fallback"]:
                stats["subtitle_fallback"] += 1
            elif conf >= HIGH_CONF_THRESHOLD:
                stats["high_conf"] += 1
            else:
                stats["low_conf"] += 1
        else:
            clips_to_add.append(_make_clip(node, show_name, episode, ar, "failed", False))
            stats["failed"] += 1

    _step_results["ffmpeg"] = {
        "high_conf": stats["high_conf"],
        "low_conf": stats["low_conf"],
        "subtitle_fallback": stats["subtitle_fallback"],
        "unmatched": stats["unmatched"],
        "failed": stats["failed"],
        "clips_generated": stats["clips_generated"],
        "elapsed_s": round(time.monotonic() - _t0, 1),
    }
    _rpt("ffmpeg",
         f"切片完成：高置信 {stats['high_conf']} · 中置信 {stats['low_conf']} · "
         f"字幕兜底 {stats['subtitle_fallback']} · 失败 {stats['failed']}", 85)

    # ──────────────────────────────────────────────
    # Step 5: 写入数据库
    # ──────────────────────────────────────────────
    _rpt("writing", "写入数据库…", 90)
    _t0 = time.monotonic()
    if clips_to_add:
        session.add_all(clips_to_add)

    # 更新 knowledge_nodes.has_any_clip
    if nodes_with_clips:
        await session.execute(
            update(KnowledgeNode)
            .where(KnowledgeNode.id.in_(list(nodes_with_clips)))
            .values(has_any_clip=True)
        )

    await session.commit()
    _step_results["writing"] = {"clips": len(clips_to_add), "elapsed_s": round(time.monotonic() - _t0, 1)}
    _rpt("done",
         f"流水线完成：生成切片 {stats['clips_generated']} 个", 100)

    logger.info(
        f"切片流水线完成: 高置信={stats['high_conf']}, 中置信={stats['low_conf']}, "
        f"字幕兜底={stats['subtitle_fallback']}, 未匹配={stats['unmatched']}, "
        f"失败={stats['failed']}, 生成切片={stats['clips_generated']}"
    )

    return stats
