"""
DramaLingo · 字幕处理流水线
对应 PRD v0.9 §5 流程一 / 开发文档 Step 2~5

完整流水线（Step 5 整合版）：
  解析 → 四道质量筛选 → Embedding 向量化 → 两段式去重 → AI 打标/降级 → 写入 DB → FAISS 增量更新

主要功能：
  - process_subtitle_file: 单文件处理（向后兼容，仅解析和筛选）
  - process_subtitle_batch: 核心流水线，处理多文件并同步到知识库
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

import config
from models.subtitle import SubtitleRaw

_ENCODE_BATCH_SIZE = 64  # 单次向量化最大条数，控制内存峰值


def _encode_in_batches(encode_fn, texts: List[str], batch_size: int = _ENCODE_BATCH_SIZE):
    """分批调用 encode_fn，避免大批量时内存峰值过高。返回 (N, D) numpy float32 数组。"""
    import numpy as _np
    if not texts:
        return _np.empty((0,), dtype=_np.float32)
    results = []
    for i in range(0, len(texts), batch_size):
        results.extend(encode_fn(texts[i: i + batch_size]))
    return _np.array(results, dtype=_np.float32)
from models.knowledge_node import KnowledgeNode, make_node_code
from models.node_source import NodeSource
from services.subtitle_parser import parse_subtitle, parse_bilingual_subtitles
from services.quality_filter import filter_subtitles

logger = logging.getLogger(__name__)

# 批量写入的 chunk 大小（避免单次 INSERT 过大导致 SQLite 锁竞争或内存压力）
_BATCH_SIZE = 500


def _report(task_id: Optional[str], stage: str, label: str, pct: int, **extra) -> None:
    """向任务中心上报阶段进度，task_id 为 None 时静默跳过。"""
    if not task_id:
        return
    from tasks.task_runner import update_task_progress
    update_task_progress(task_id, {"task_type": "subtitle", "stage": stage, "label": label, "pct": pct, **extra})


# ──────────────────────────────────────────────
# Step 2 原有接口（向后兼容）
# ──────────────────────────────────────────────

async def process_subtitle_file(
    file_path: str,
    show_name: str,
    episode: str,
    session: AsyncSession,
    zh_file: Optional[str] = None,
    overwrite: bool = False,
    en_name: str = "",
    subtitle_code: Optional[str] = None,
    content_type: str = "tv",
) -> Dict:
    """
    单文件字幕处理（Step 2 基础版）。

    执行流程：
    1. 幂等检查：若不覆盖且已存在，则跳过。
    2. 解析：调用 subtitle_parser 处理 SRT/ASS 文件。
    3. 筛选：调用 quality_filter 执行四道质量门禁。
    4. 存储：将结果批量写入 subtitles_raw 表。

    Args:
        file_path:     英文字幕文件路径
        show_name:     中文剧名
        episode:       集号（如 S01E01）
        session:       SQLAlchemy 异步 session
        zh_file:       中文字幕文件路径（可选，用于双语对齐）
        overwrite:     若该集已有记录，是否覆盖（默认 False 跳过）
        en_name:       英文剧名
        subtitle_code: 批次唯一编码

    Returns:
        统计字典，包含 total/passed/filter_rate 等字段
    """
    path = Path(file_path)
    base_result = {
        "total": 0,
        "passed": 0,
        "filter_rate": 1.0,
        "rejected_by_stage": {
            "rejected_blacklist": 0,
            "rejected_wordcount": 0,
            "rejected_quality": 0,
        },
        "skipped": False,
        "error": None,
    }

    # ── 1. 幂等检查 ──
    if not overwrite:
        existing = await session.execute(
            select(SubtitleRaw.id)
            .where(SubtitleRaw.show_name == show_name)
            .where(SubtitleRaw.episode == episode)
            .limit(1)
        )
        if existing.scalar_one_or_none() is not None:
            logger.info(f"幂等跳过: {show_name} {episode} 已有字幕记录")
            base_result["skipped"] = True
            return base_result

    # ── 2. 解析字幕文件 ──
    try:
        if zh_file:
            # 双语解析模式
            lines = parse_bilingual_subtitles(file_path, zh_file)
        else:
            # 单语解析模式
            lines = parse_subtitle(file_path)
    except ValueError as e:
        logger.error(f"字幕格式不支持: {e}")
        base_result["error"] = str(e)
        return base_result
    except Exception as e:
        logger.error(f"字幕解析异常: {e}")
        base_result["error"] = f"字幕解析异常: {e}"
        return base_result

    if not lines:
        msg = f"该文件未解析到任何字幕行: {path.name}"
        logger.error(msg)
        base_result["error"] = msg
        return base_result

    # ── 3. 四道质量筛选 ──
    filter_results, stats = filter_subtitles(lines)

    if stats["passed"] == 0:
        base_result.update(stats)
        base_result["error"] = "该文件未解析到有效台词（通过率 0%），请检查字幕文件质量"
        logger.error(base_result["error"])

    # ── 4. 批量写入 subtitles_raw 表 ──
    rows = []
    for fr in filter_results:
        row = SubtitleRaw(
            show_name=show_name,
            en_name=en_name,
            episode=episode,
            subtitle_code=subtitle_code,
            start_time=fr.line.start_time,
            end_time=fr.line.end_time,
            en_text=fr.line.en_text,
            zh_text=fr.line.zh_text,
            word_count=fr.line.word_count,
            filter_status=fr.filter_status,
            quality_score=fr.quality_score if fr.quality_score > 0 else None,
            content_type=content_type,
            source="subtitle",
        )
        rows.append(row)

    # 分块写入，避免大事务锁表
    for i in range(0, len(rows), _BATCH_SIZE):
        chunk = rows[i: i + _BATCH_SIZE]
        session.add_all(chunk)
        await session.flush()

    await session.commit()

    logger.info(
        f"字幕流水线完成: {show_name} {episode} | "
        f"总计 {stats['total']} 行，通过 {stats['passed']} 行，"
        f"过滤率 {stats['filter_rate']:.1%}"
    )

    result = dict(stats)
    result["skipped"] = False
    result["error"] = base_result.get("error")
    return result


async def process_bilingual_subtitle_files(
    en_file: str,
    zh_file: str,
    show_name: str,
    episode: str,
    session: AsyncSession,
    overwrite: bool = False,
) -> Dict:
    """
    处理双语字幕（英文 + 中文）的语法糖。
    """
    return await process_subtitle_file(
        file_path=en_file,
        show_name=show_name,
        episode=episode,
        session=session,
        zh_file=zh_file,
        overwrite=overwrite,
    )


# ──────────────────────────────────────────────
# Step 5 完整流水线（含 Embedding + 去重 + AI 打标）
# ──────────────────────────────────────────────

async def process_subtitle_batch(
    file_paths: List[str],
    show_name: str,
    episode_map: Dict[str, str],
    session: AsyncSession,
    overwrite: bool = False,
    enable_ai_tagging: bool = True,
    en_name: str = "",
    subtitle_code: Optional[str] = None,
    file_code_map: Optional[Dict[str, str]] = None,
    task_id: Optional[str] = None,
    content_type: str = "tv",
) -> Dict:
    """
    完整字幕处理流水线（核心业务逻辑）。
    
    执行流程（PRD §5 流程一）：
    1. 索引准备：加载或重建 FAISS 索引。
    2. 缓存准备：刷新内存中的去重文本缓存。
    3. 逐文件处理：
       a. 幂等检查。
       b. 解析与筛选（Step 2）。
       c. 写入原始字幕表。
       d. 提取通过筛选的台词。
    4. 向量化：批量生成 Embedding（Step 3）。
    5. 去重判定：两段式去重（编辑距离 + 语义相似度）（Step 4）。
    6. 知识提取：
       a. 命中重复：更新已有节点的出现次数，添加新来源。
       b. 新节点：AI 打标（或规则降级），创建新节点。
    7. 索引同步：将新节点增量添加到 FAISS 索引。

    Args:
        file_paths:       字幕文件路径列表
        show_name:        剧名
        episode_map:      {文件名 -> 集号} 映射
        session:          AsyncSession
        overwrite:        是否覆盖已有记录
        enable_ai_tagging: 是否启用 AI 打标（若为 False 则全部使用规则打标）

    Returns:
        统计字典，包含处理过程中的各项指标
    """
    from services.embedder import encode_texts, serialize_vector
    from services.faiss_index import (
        rebuild_index_from_db, load_index, add_to_index, save_index, get_global_index, set_global_index
    )
    from services.deduplicator import check_duplicate, refresh_text_cache, add_to_text_cache, normalize_text as _norm_text
    from services.ai_tagger import batch_tag_and_translate, rule_based_tag, AITaggingError

    stats = {
        "total_files": len(file_paths),
        "total_lines": 0,
        "passed_lines": 0,
        "new_nodes": 0,
        "merged_nodes": 0,
        "filter_rate": 0.0,
        "ai_tagging_success": 0,
        "rule_fallback": 0,
        "skipped_files": 0,
        "errors": [],
    }
    # 各步骤完成后的结果快照，随每次 _report() 传递到前端
    _step_results: dict = {}
    # 跨文件累计的解析统计
    _acc_blacklist = _acc_wordcount = _acc_quality = _acc_merged = 0

    _n = len(file_paths)
    _report(task_id, "init", f"准备索引，共 {_n} 个文件", 2, files_total=_n, step_results=_step_results)

    # ── 1. 加载/重建 FAISS 索引 ──
    index_path = str(config.INDEX_DIR / "knowledge.index")
    try:
        index, mapping = load_index(index_path)
        logger.info(f"加载已有 FAISS 索引，共 {index.ntotal} 条向量")
    except FileNotFoundError:
        logger.info("FAISS 索引不存在，从数据库重建")
        index, mapping = await rebuild_index_from_db(session, force_full=True, index_path=index_path)

    # ── 2. 刷新去重文本缓存 ──
    await refresh_text_cache(session)

    # ── 3. 逐文件处理（解析与筛选） ──
    all_passed_lines = []  # 暂存所有通过筛选的行，用于后续批量 Embedding

    for _file_idx, file_path in enumerate(file_paths):
        file_name = Path(file_path).name
        episode = episode_map.get(file_name, episode_map.get(file_path, ""))

        _report(task_id, "parsing",
                f"解析文件 {_file_idx + 1}/{_n}：{file_name}",
                5 + int(25 * _file_idx / _n),
                files_done=_file_idx, files_total=_n,
                lines_total=stats["total_lines"], lines_passed=stats["passed_lines"])

        logger.info(f"开始处理: {show_name} {episode} ({file_name})")

        # Step 1：幂等检查
        if not overwrite:
            existing = await session.execute(
                select(SubtitleRaw.id)
                .where(SubtitleRaw.show_name == show_name)
                .where(SubtitleRaw.episode == episode)
                .limit(1)
            )
            if existing.scalar_one_or_none() is not None:
                logger.info(f"幂等跳过: {show_name} {episode} 已有字幕记录")
                stats["skipped_files"] += 1
                continue

        # Step 2：解析 + 质量筛选
        try:
            lines, _ps = parse_subtitle(file_path, return_stats=True)
            _acc_merged += _ps.get("merged", 0)
        except Exception as e:
            msg = f"{file_name} 解析失败: {e}"
            logger.error(msg)
            stats["errors"].append(msg)
            continue

        if not lines:
            msg = f"{file_name} 未解析到任何字幕行"
            logger.warning(msg)
            stats["errors"].append(msg)
            continue

        filter_results, file_stats = filter_subtitles(lines)
        stats["total_lines"] += file_stats["total"]
        stats["passed_lines"] += file_stats["passed"]
        _rbs = file_stats.get("rejected_by_stage", {})
        _acc_blacklist += _rbs.get("rejected_blacklist", 0)
        _acc_wordcount += _rbs.get("rejected_wordcount", 0)
        _acc_quality   += _rbs.get("rejected_quality", 0)

        # 写入 subtitles_raw（每个文件使用独立的 subtitle_code）
        _file_code = (file_code_map or {}).get(file_path) or subtitle_code
        raw_rows = []
        for fr in filter_results:
            row = SubtitleRaw(
                show_name=show_name,
                en_name=en_name,
                episode=episode,
                subtitle_code=_file_code,
                start_time=fr.line.start_time,
                end_time=fr.line.end_time,
                en_text=fr.line.en_text,
                zh_text=fr.line.zh_text,
                word_count=fr.line.word_count,
                filter_status=fr.filter_status,
                quality_score=fr.quality_score if fr.quality_score > 0 else None,
                content_type=content_type,
                source="subtitle",
            )
            raw_rows.append(row)

            if fr.filter_status == "passed":
                all_passed_lines.append({
                    "en_text": fr.line.en_text,
                    "zh_text": fr.line.zh_text,
                    "start_time": fr.line.start_time,
                    "end_time": fr.line.end_time,
                    "word_count": fr.line.word_count,
                    "quality_score": fr.quality_score,
                    "episode": episode
                })

        # 批量写入原始字幕，并立即提交（与后续 AI 打标解耦，避免 AI 超时导致字幕数据回滚）
        for i in range(0, len(raw_rows), _BATCH_SIZE):
            session.add_all(raw_rows[i: i + _BATCH_SIZE])
            await session.flush()
        await session.commit()

        _report(task_id, "parsing",
                f"文件 {_file_idx + 1}/{_n} 已入库：{file_name}  "
                f"（总 {file_stats['total']} 行 / 通过 {file_stats['passed']} 行）",
                5 + int(25 * (_file_idx + 1) / _n),
                files_done=_file_idx + 1, files_total=_n,
                lines_total=stats["total_lines"], lines_passed=stats["passed_lines"],
                step_results=_step_results)

    if not all_passed_lines:
        logger.warning("所有文件均未产生有效台词，流水线提前结束")
        return stats

    # 解析阶段完成，写入 step_results
    _step_results["parsing"] = {
        "total": stats["total_lines"],
        "passed": stats["passed_lines"],
        "blacklist": _acc_blacklist,
        "wordcount": _acc_wordcount,
        "quality": _acc_quality,
        "merged": _acc_merged,
    }

    # ── 4. 批量 Embedding 向量化 ──
    _report(task_id, "embedding",
            f"向量化中，共 {len(all_passed_lines)} 条有效台词…", 32,
            lines_passed=stats["passed_lines"], step_results=_step_results)
    logger.info(f"开始批量 Embedding: {len(all_passed_lines)} 条台词")
    texts_to_encode = [line["en_text"] for line in all_passed_lines]
    embeddings = _encode_in_batches(encode_texts, texts_to_encode)

    # 向量化完成
    _step_results["embedding"] = {"count": len(all_passed_lines)}

    # ── 4.5. 语境质量二次评分 ──
    # 将每句话放入整个剧集和知识库中横向比较，淘汰通用度高、知识库已充分覆盖的表达
    from services.quality_filter import compute_contextual_scores
    import asyncio as _asyncio

    _ctx_input = len(all_passed_lines)
    _report(task_id, "context_filter",
            f"语境评分中，{_ctx_input} 条放入剧集和知识库中横向比较…", 40,
            step_results=_step_results)

    _ctx_scores = await _asyncio.get_running_loop().run_in_executor(
        None,
        compute_contextual_scores,
        [l["en_text"] for l in all_passed_lines],
        embeddings,
        index,
    )
    _ctx_threshold = config.CONTEXT_QUALITY_THRESHOLD
    _keep = _ctx_scores >= _ctx_threshold
    all_passed_lines = [l for l, k in zip(all_passed_lines, _keep) if k]
    embeddings = embeddings[_keep]
    _ctx_rejected = int((~_keep).sum())

    _step_results["context_filter"] = {
        "input": _ctx_input,
        "passed": len(all_passed_lines),
        "rejected": _ctx_rejected,
    }
    _report(task_id, "context_filter",
            f"语境筛选完成，淘汰 {_ctx_rejected} 条低价值表达", 47,
            step_results=_step_results)

    if not all_passed_lines:
        logger.warning("语境筛选后无剩余台词，流水线提前结束")
        return stats

    # ── 5. 两段式去重：第一阶段仅判重，分离新节点和重复节点 ──
    _report(task_id, "dedup",
            f"去重判定中，共 {len(all_passed_lines)} 条…", 53,
            lines_passed=stats["passed_lines"], step_results=_step_results)
    new_node_ids = []
    new_embeddings = []
    dup_count: Dict[int, int] = {}
    new_nodes_by_episode: Dict[str, List[int]] = {}  # C4: episode → [node_id, ...]

    # 收集待打标的新节点
    new_line_items: List[dict] = []
    new_line_embeddings = []
    # 批内去重：同一批次内相同 en_text 只保留首次出现，防止多文件重复句子触发 UNIQUE 约束
    _batch_seen_texts: set = set()

    # P1 优化：批量预加载 (episode, en_text) → subtitle_id，避免写入循环中的 N+1 查询
    _all_en_texts = list({line["en_text"] for line in all_passed_lines})
    _sub_rows = await session.execute(
        select(SubtitleRaw.id, SubtitleRaw.episode, SubtitleRaw.en_text)
        .where(SubtitleRaw.show_name == show_name)
        .where(SubtitleRaw.en_text.in_(_all_en_texts))
    )
    _subtitle_id_map: Dict[tuple, int] = {
        (row.episode, row.en_text): row.id for row in _sub_rows
    }

    for i, line in enumerate(all_passed_lines):
        embedding = embeddings[i]

        # 批内精确去重（补充 check_duplicate 未能覆盖的批内重复场景）
        _norm = _norm_text(line["en_text"])
        if _norm in _batch_seen_texts:
            stats["merged_nodes"] += 1
            continue

        is_dup, existing_id = await check_duplicate(
            line["en_text"], embedding, session, index, mapping
        )

        if is_dup:
            stats["merged_nodes"] += 1
            dup_count[existing_id] = dup_count.get(existing_id, 0) + 1
            dup_subtitle_id = _subtitle_id_map.get((line["episode"], line["en_text"]))
            await session.execute(
                sqlite_insert(NodeSource).values(
                    node_id=existing_id,
                    subtitle_id=dup_subtitle_id,
                    show_name=show_name,
                    episode=line["episode"],
                ).on_conflict_do_nothing()
            )
        else:
            stats["new_nodes"] += 1
            new_line_items.append(line)
            new_line_embeddings.append(embedding)
            _batch_seen_texts.add(_norm)

    # 去重完成
    _step_results["dedup"] = {
        "new_nodes": stats["new_nodes"],
        "merged_nodes": stats["merged_nodes"],
    }

    # ── 5b. 对所有新节点并发批量打标（最多 5 并发，比逐条快 5 倍） ──
    _ai_enabled = enable_ai_tagging and bool(config.AI_API_KEY)
    if not _ai_enabled and enable_ai_tagging:
        logger.debug("AI_API_KEY 未配置，跳过 AI 打标，使用规则打标")

    _n_new = len(new_line_items)
    if new_line_items:
        if _ai_enabled:
            _report(task_id, "ai_tagging",
                    f"AI 打标中，{_n_new} 条新知识…", 60,
                    new_nodes=0, merged_nodes=stats["merged_nodes"], ai_total=_n_new,
                    step_results=_step_results)
            logger.info(f"批量打标: {_n_new} 个新节点，每批 10 条，最大并发 10")
            all_tags, _tagging_stats = await batch_tag_and_translate(
                texts=[l["en_text"] for l in new_line_items],
                word_counts=[l["word_count"] for l in new_line_items],
            )
            for t in all_tags:
                if t.get("tag_source") == "rule":
                    stats["rule_fallback"] += 1
                else:
                    stats["ai_tagging_success"] += 1
        else:
            _tagging_stats = {}
            _report(task_id, "ai_tagging",
                    f"规则打标中，{_n_new} 条新知识…", 60,
                    new_nodes=0, merged_nodes=stats["merged_nodes"], ai_total=_n_new,
                    step_results=_step_results)
            all_tags = [
                rule_based_tag(l["en_text"], l["word_count"]) for l in new_line_items
            ]
            stats["rule_fallback"] += len(new_line_items)
            _tagging_stats = {"rule": {"count": len(new_line_items), "elapsed_s": 0.0}}

        # AI 打标完成（provider_stats 供前端展示各模型打标数量和耗时）
        _step_results["ai_tagging"] = {
            "ai_success": stats["ai_tagging_success"],
            "rule_fallback": stats["rule_fallback"],
            "total": _n_new,
            "provider_stats": _tagging_stats,
        }

        # ── 5c. 批量编码中文翻译向量（供多信号检索使用）──
        _report(task_id, "writing",
                f"写入知识库，共 {_n_new} 个新节点…", 85,
                new_nodes=_n_new, merged_nodes=stats["merged_nodes"],
                step_results=_step_results)

        # 批量 encode zh_text（过滤空值，保留位置对应关系）
        _zh_texts = [l.get("zh_text") or "" for l in new_line_items]
        _zh_valid_mask = [bool(t.strip()) for t in _zh_texts]
        _zh_valid_texts = [t for t, v in zip(_zh_texts, _zh_valid_mask) if v]
        _zh_embeddings_raw = _encode_in_batches(encode_texts, _zh_valid_texts)
        _zh_iter = iter(_zh_embeddings_raw)
        _zh_embeddings_list = [next(_zh_iter) if v else None for v in _zh_valid_mask]

        for _wi, (line, embedding, tags, zh_emb) in enumerate(
            zip(new_line_items, new_line_embeddings, all_tags, _zh_embeddings_list)
        ):
            subtitle_id = _subtitle_id_map.get((line["episode"], line["en_text"]))

            # 精确文本查重（唯一约束兜底，防止跨剧集重复文本触发 UNIQUE 冲突）
            _exist_row = (await session.execute(
                select(KnowledgeNode).where(KnowledgeNode.en_text == line["en_text"])
            )).scalar_one_or_none()
            if _exist_row is not None:
                _exist_row.occurrence_count += 1
                await session.execute(
                    sqlite_insert(NodeSource).values(
                        node_id=_exist_row.id,
                        subtitle_id=subtitle_id,
                        show_name=show_name,
                        episode=line["episode"],
                    ).on_conflict_do_nothing()
                )
                stats["merged_nodes"] = stats.get("merged_nodes", 0) + 1
                continue

            new_node = KnowledgeNode(
                code=make_node_code(show_name, line["episode"], line["start_time"]),
                subtitle_id=subtitle_id,
                en_text=line["en_text"],
                zh_text=line["zh_text"],
                show_name=show_name,
                episode=line["episode"],
                start_time=line["start_time"],
                end_time=line["end_time"],
                word_count=line["word_count"],
                quality_score=line["quality_score"],
                scene=tags["scene"],
                emotion=tags["emotion"],
                difficulty=tags["difficulty"],
                keywords=tags["keywords"],
                tag_source=tags["tag_source"],
                embedding=serialize_vector(embedding),
                zh_embedding=serialize_vector(zh_emb) if zh_emb is not None else None,
                occurrence_count=1,
                content_type=content_type,
                is_active=True,
            )
            session.add(new_node)
            await session.flush()

            await session.execute(
                sqlite_insert(NodeSource).values(
                    node_id=new_node.id,
                    subtitle_id=subtitle_id,
                    show_name=show_name,
                    episode=line["episode"],
                ).on_conflict_do_nothing()
            )

            add_to_text_cache(new_node.id, line["en_text"])
            new_node_ids.append(new_node.id)
            new_embeddings.append(embedding)
            ep_key = line.get("episode", "")
            new_nodes_by_episode.setdefault(ep_key, []).append(new_node.id)

            # 每 20 个节点上报一次进度，避免前端以为任务卡死
            if (_wi + 1) % 20 == 0 or (_wi + 1) == _n_new:
                _report(task_id, "writing",
                        f"写入知识库 {_wi + 1}/{_n_new}…", 85 + int(10 * (_wi + 1) / _n_new),
                        new_nodes=_wi + 1, merged_nodes=stats["merged_nodes"],
                        step_results=_step_results)

    # 批量更新重复节点的 occurrence_count
    for dup_id, count in dup_count.items():
        await session.execute(
            update(KnowledgeNode)
            .where(KnowledgeNode.id == dup_id)
            .values(occurrence_count=KnowledgeNode.occurrence_count + count)
        )

    # 写入完成 — 先 commit 再做 FAISS，确保数据安全落盘
    _step_results["writing"] = {"count": stats["new_nodes"]}
    await session.commit()

    # ── 6. FAISS 索引增量更新（en + zh 双索引）──
    _report(task_id, "faiss", "更新检索索引…", 95,
            new_nodes=len(new_node_ids), merged_nodes=stats["merged_nodes"],
            step_results=_step_results)
    if new_node_ids:
        import numpy as np
        try:
            add_to_index(index, mapping, new_node_ids, np.stack(new_embeddings))
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, save_index, index, mapping, index_path)
            set_global_index(index, mapping)

            # zh 索引增量更新：保持 ids 和 embs 严格对齐
            from services.faiss_index import add_to_zh_index_and_save
            _zh_pairs = [
                (nid, emb)
                for nid, emb in zip(new_node_ids, _zh_embeddings_list)
                if emb is not None
            ]
            if _zh_pairs:
                await loop.run_in_executor(
                    None, add_to_zh_index_and_save,
                    [p[0] for p in _zh_pairs],
                    np.stack([p[1] for p in _zh_pairs]),
                )
        except Exception as _faiss_err:
            logger.error(f"FAISS 索引更新失败（知识库数据已安全写入，索引将在下次启动时重建）: {_faiss_err}")

    # faiss 完成
    _step_results["faiss"] = {"new_nodes": len(new_node_ids)}

    # 计算最终过滤率
    if stats["total_lines"] > 0:
        stats["filter_rate"] = 1.0 - (stats["passed_lines"] / stats["total_lines"])

    logger.info(
        f"批量处理完成: {show_name} | "
        f"新增节点 {stats['new_nodes']}, 合并节点 {stats['merged_nodes']}, "
        f"AI 打标成功 {stats['ai_tagging_success']}, 规则降级 {stats['rule_fallback']}"
    )

    # ── C4: 自动触发增量补切 ──────────────────────────────────────
    # 对每个有新节点的剧集，检查是否已有对应视频文件，有则自动入队补切
    if new_nodes_by_episode:
        try:
            from models.video_file import VideoFile
            from services.clip_pipeline import generate_clips
            from tasks.task_runner import submit_task
            from database import SessionLocal

            for ep, nids in new_nodes_by_episode.items():
                vf_row = await session.execute(
                    select(VideoFile)
                    .where(VideoFile.show_name == show_name, VideoFile.episode == ep)
                    .limit(1)
                )
                vf = vf_row.scalar_one_or_none()
                if vf and vf.file_path and os.path.exists(vf.file_path):
                    from uuid import uuid4 as _uuid4
                    _vp = vf.file_path
                    _sn, _ep, _nids = show_name, ep, list(nids)
                    _jid = str(_uuid4())

                    async def _clip_task(_vp=_vp, _sn=_sn, _ep=_ep, _nids=_nids, _jid=_jid):
                        async with SessionLocal() as _sess:
                            return await generate_clips(
                                video_path=_vp, show_name=_sn, episode=_ep,
                                session=_sess, node_ids=_nids, task_id=_jid,
                            )

                    await submit_task(
                        _clip_task(),
                        description=f"视频切片: {show_name} {ep}（增量 {len(nids)} 节点）",
                        task_id=_jid,
                    )
                    logger.info(f"C4 已自动入队补切: {show_name} {ep}，{len(nids)} 个新节点")
        except Exception as _c4_err:
            logger.warning(f"C4 自动补切入队失败（不影响字幕处理结果）: {_c4_err}")

    return stats


# ──────────────────────────────────────────────
# 辅助：将已解析入库的字幕行同步到知识库
# ──────────────────────────────────────────────

async def sync_episode_to_knowledge(
    show_name: str,
    episode: str,
    session: AsyncSession,
    en_name: str = "",
    enable_ai_tagging: bool = True,
    content_type: str = "tv",
    task_id: Optional[str] = None,
) -> Dict:
    """
    将 subtitles_raw 中已通过筛选（filter_status='passed'）但尚无知识节点的
    字幕行补全到知识库（Embedding→去重→AI打标→写 knowledge_nodes）。

    适用于双语上传后的自动同步：process_subtitle_file 已完成解析+筛选+写库，
    本函数接着执行后续 4 个步骤，使双语上传与批量上传等效。
    """
    import numpy as np
    from services.embedder import encode_texts, serialize_vector
    from services.faiss_index import (
        rebuild_index_from_db, load_index, add_to_index, save_index, set_global_index,
    )
    from services.deduplicator import check_duplicate, refresh_text_cache, add_to_text_cache
    from services.ai_tagger import batch_tag_and_translate, rule_based_tag

    stats: Dict = {"new_nodes": 0, "merged_nodes": 0, "ai_tagging_success": 0, "rule_fallback": 0}

    # 查询该集号中已通过筛选、尚无知识节点的字幕行
    synced_ids_subq = select(KnowledgeNode.subtitle_id).where(KnowledgeNode.subtitle_id.isnot(None))
    stmt = select(SubtitleRaw).where(
        SubtitleRaw.show_name == show_name,
        SubtitleRaw.episode == episode,
        SubtitleRaw.filter_status == "passed",
        SubtitleRaw.id.notin_(synced_ids_subq),
    )
    result = await session.execute(stmt)
    subtitles = result.scalars().all()

    if not subtitles:
        logger.info(f"sync_episode_to_knowledge: {show_name} {episode} 无需同步（无新增通过行）")
        return stats

    logger.info(f"sync_episode_to_knowledge: {show_name} {episode} 开始同步 {len(subtitles)} 条")

    # ── 加载/重建 FAISS 索引 ──
    _report(task_id, "embedding", f"向量化准备：加载索引…", 20)
    index_path = str(config.INDEX_DIR / "knowledge.index")
    try:
        index, mapping = load_index(index_path)
    except FileNotFoundError:
        index, mapping = await rebuild_index_from_db(session, force_full=True, index_path=index_path)

    await refresh_text_cache(session)

    # ── 向量化 ──
    _report(task_id, "embedding", f"向量化 {len(subtitles)} 条字幕…", 30)
    embeddings = _encode_in_batches(encode_texts, [s.en_text for s in subtitles])

    # ── 去重判定 ──
    _report(task_id, "dedup", f"去重判定…", 45)
    new_items: List[SubtitleRaw] = []
    new_embeddings = []
    dup_count: Dict[int, int] = {}

    for i, sub in enumerate(subtitles):
        is_dup, existing_id = await check_duplicate(sub.en_text, embeddings[i], session, index, mapping)
        if is_dup:
            stats["merged_nodes"] += 1
            dup_count[existing_id] = dup_count.get(existing_id, 0) + 1
            await session.execute(
                sqlite_insert(NodeSource).values(
                    node_id=existing_id,
                    subtitle_id=sub.id,
                    show_name=show_name,
                    episode=episode,
                ).on_conflict_do_nothing()
            )
        else:
            new_items.append(sub)
            new_embeddings.append(embeddings[i])

    # ── AI 打标（或规则降级）──
    _ai_enabled = enable_ai_tagging and bool(config.AI_API_KEY)
    # 提前初始化，避免 new_items 为空时后续 if new_node_ids 报 UnboundLocalError
    new_node_ids: List[int] = []
    new_node_embeddings = []
    _zh_new_ids_s: List[int] = []
    _zh_new_embs_s = []
    if new_items:
        _report(task_id, "ai_tagging",
                f"AI 打标中，{len(new_items)} 条新知识…", 60,
                step_results={"ai_tagging": {"ai_success": 0, "rule_fallback": 0, "total": len(new_items)}})
        if _ai_enabled:
            all_tags, _sync_tag_stats = await batch_tag_and_translate(
                texts=[s.en_text for s in new_items],
                word_counts=[s.word_count for s in new_items],
            )
            for t in all_tags:
                if t.get("tag_source") == "ai":
                    stats["ai_tagging_success"] += 1
                else:
                    stats["rule_fallback"] += 1
        else:
            all_tags = [rule_based_tag(s.en_text, s.word_count) for s in new_items]
            stats["rule_fallback"] += len(new_items)

        # ── 批量编码 zh_text（供中文检索使用）──
        _zh_texts_s = [s.zh_text or "" for s in new_items]
        _zh_valid_mask_s = [bool(t.strip()) for t in _zh_texts_s]
        _zh_valid_texts_s = [t for t, v in zip(_zh_texts_s, _zh_valid_mask_s) if v]
        _zh_embs_raw_s = _encode_in_batches(encode_texts, _zh_valid_texts_s)
        _zh_iter_s = iter(_zh_embs_raw_s)
        _zh_embs_s = [next(_zh_iter_s) if v else None for v in _zh_valid_mask_s]

        # ── 写入知识节点 ──
        for sub, embedding, tags, zh_emb in zip(new_items, new_embeddings, all_tags, _zh_embs_s):
            # 先按 en_text 精确查重（唯一约束的硬性保障）
            _exist_row = (await session.execute(
                select(KnowledgeNode).where(KnowledgeNode.en_text == sub.en_text)
            )).scalar_one_or_none()
            if _exist_row is not None:
                _exist_row.occurrence_count += 1
                await session.execute(
                    sqlite_insert(NodeSource).values(
                        node_id=_exist_row.id,
                        subtitle_id=sub.id,
                        show_name=show_name,
                        episode=episode,
                    ).on_conflict_do_nothing()
                )
                stats["merged_nodes"] += 1
                continue

            new_node = KnowledgeNode(
                code=make_node_code(show_name, episode, sub.start_time),
                subtitle_id=sub.id,
                en_text=sub.en_text,
                zh_text=sub.zh_text,
                show_name=show_name,
                episode=episode,
                start_time=sub.start_time,
                end_time=sub.end_time,
                word_count=sub.word_count,
                quality_score=sub.quality_score,
                scene=tags["scene"],
                emotion=tags["emotion"],
                difficulty=tags["difficulty"],
                keywords=tags["keywords"],
                tag_source=tags["tag_source"],
                embedding=serialize_vector(embedding),
                zh_embedding=serialize_vector(zh_emb) if zh_emb is not None else None,
                occurrence_count=1,
                content_type=content_type,
                is_active=True,
            )
            session.add(new_node)
            await session.flush()

            await session.execute(
                sqlite_insert(NodeSource).values(
                    node_id=new_node.id,
                    subtitle_id=sub.id,
                    show_name=show_name,
                    episode=episode,
                ).on_conflict_do_nothing()
            )
            add_to_text_cache(new_node.id, sub.en_text)
            new_node_ids.append(new_node.id)
            new_node_embeddings.append(embedding)
            if zh_emb is not None:
                _zh_new_ids_s.append(new_node.id)
                _zh_new_embs_s.append(zh_emb)
            stats["new_nodes"] += 1

    # 更新重复节点的出现频次
    _report(task_id, "writing",
            f"写入知识库：{stats['new_nodes']} 个新节点，{stats['merged_nodes']} 个合并…", 80,
            step_results={"writing": {"count": stats["new_nodes"]},
                          "ai_tagging": {"ai_success": stats["ai_tagging_success"],
                                         "rule_fallback": stats["rule_fallback"],
                                         "total": stats["ai_tagging_success"] + stats["rule_fallback"]}})
    for dup_id, cnt in dup_count.items():
        await session.execute(
            update(KnowledgeNode)
            .where(KnowledgeNode.id == dup_id)
            .values(occurrence_count=KnowledgeNode.occurrence_count + cnt)
        )

    await session.commit()

    # ── FAISS 增量更新（en + zh 双索引）──
    if new_node_ids:
        _report(task_id, "faiss", f"更新检索索引…", 90,
                step_results={"writing": {"count": stats["new_nodes"]},
                              "faiss": {"new_nodes": stats["new_nodes"]}})
        add_to_index(index, mapping, new_node_ids, np.stack(new_node_embeddings))
        _loop = asyncio.get_running_loop()
        await _loop.run_in_executor(None, save_index, index, mapping, index_path)
        set_global_index(index, mapping)

        if _zh_new_ids_s:
            from services.faiss_index import add_to_zh_index_and_save
            await _loop.run_in_executor(
                None, add_to_zh_index_and_save, _zh_new_ids_s, np.stack(_zh_new_embs_s)
            )

    logger.info(
        f"sync_episode_to_knowledge 完成: {show_name} {episode} | "
        f"新增 {stats['new_nodes']}, 合并 {stats['merged_nodes']}"
    )
    return stats
