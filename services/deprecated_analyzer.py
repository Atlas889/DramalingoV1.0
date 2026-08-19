"""
DramaLingo · 弃用知识节点模式分析器

定期扫描 is_active=False 的知识节点，识别导致节点失效的典型语言模式，
并输出可用于反哺字幕质量过滤机制的统计报告与建议。

调用方式：
  - API:  POST /knowledge/analyze-deprecated
  - 周期: 每次新增批量弃用操作后自动触发（在 knowledge 路由的 batch-delete/batch-update 后调用）
  - 手动: 管理后台"分析弃用知识"按钮

返回结构：
  {
    "total_deprecated": int,
    "analyzed_at": ISO8601,
    "patterns": {
      "<pattern_key>": {
        "count": int, "ratio": float,
        "description": str,
        "examples": [str, ...],   # 最多 3 条
        "suggestion": str,        # 对质量过滤的改进建议
      }
    },
    "score_distribution": { "buckets": [...], "mean": float, "median": float },
    "top_rejected_reasons": { ... },
    "filter_recommendations": [ { "signal": str, "action": str, "confidence": str } ]
  }
"""

import re
import logging
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# 模式检测正则（镜像 quality_filter.py，但独立维护，
# 分析器不依赖过滤器模块，可独立演进）
# ──────────────────────────────────────────────────────────

# 括号舞台指示
_PAREN_STAGE   = re.compile(r"^\s*\(.*\)\s*$", re.DOTALL)
_BRACKET_STAGE = re.compile(r"^\s*\[.*\]\s*$", re.DOTALL)

# 演讲者自我介绍
_SELF_INTRO    = re.compile(
    r"^(?:I(?:'m| am| was)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|"
    r"my name(?:'s| is)\s+[A-Z]|"
    r"hi,?\s+I(?:'m| am)\s+[A-Z])\.?\s*$",
    re.IGNORECASE,
)

# 末尾截断（省略号 / 悬挂逗号）
_TRAILING_TRUNC = re.compile(r"(\.{2,}|,)\s*$")

# 续接词 + 强指称开头
_CONTINUATION  = re.compile(
    r"^(and\s+(then|so|that|it|he|she|they|this|those|we|there)\b|"
    r"but\s+(then|that|it|he|she|they|this|those|there|which|what)\b|"
    r"so\s+(that|it|he|she|they|this|those|we|basically)\b|"
    r"that'?s\s+why\b|which\s+means?\b|which\s+is\s+why\b)",
    re.IGNORECASE,
)

# 悬挂条件句
_HANGING_COND  = re.compile(r"^(if|unless|even\s+if|as\s+long\s+as)\b", re.IGNORECASE)

# 因果回指
_CAUSAL_ANA    = re.compile(
    r"^(because\s+of\s+(that|it|this|those|which)|as\s+a\s+result\b|"
    r"for\s+that\s+reason\b|in\s+that\s+case\b|that\s+being\s+said\b)",
    re.IGNORECASE,
)

# 依存性介词
_DEPEND_PREP   = re.compile(
    r"^(alongside\b|together\s+with\b|along\s+with\b|"
    r"on\s+top\s+of\s+(that|it|this|all)\b|"
    r"in\s+addition\s+to\s+(that|it|this|all)\b|"
    r"apart\s+from\s+(that|it|this|what)\b|"
    r"beyond\s+(that|it|this|all)\b)",
    re.IGNORECASE,
)

# 强代词依赖（开头是 He/She/They/It + 动词，没有先行词）
_PRONOUN_OPEN  = re.compile(
    r"^(he|she|they|it|that|this|these|those)\s+(is|are|was|were|has|have|had|"
    r"will|would|can|could|should|might|may|did|does|do|said|told|knew|got|came|went)\b",
    re.IGNORECASE,
)

# 音乐符号
_MUSIC_SYM     = re.compile(r"[♪♫♬]")

# 格式垃圾
_FORMAT_JUNK   = re.compile(
    r"(www\.|http|\.com|previously on|season \d|episode \d|"
    r"subtitles? by|translated by|downloaded from)",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────────────────
# 模式分类函数
# ──────────────────────────────────────────────────────────

def _classify_node(en_text: str, quality_score: float) -> List[str]:
    """
    返回节点匹配的所有模式键列表（可多标签）。
    空列表表示没有已知模式匹配（归入 unclassified）。
    """
    t = en_text.strip()
    labels: List[str] = []

    if _PAREN_STAGE.match(t) or _BRACKET_STAGE.match(t) or _MUSIC_SYM.search(t):
        labels.append("stage_direction")
    if _SELF_INTRO.match(t):
        labels.append("self_introduction")
    if _TRAILING_TRUNC.search(t):
        labels.append("trailing_truncation")
    if _CONTINUATION.match(t):
        labels.append("continuation_opener")
    if _HANGING_COND.match(t):
        # 悬挂条件句：句内无实质主句
        cp = t.find(",")
        if not (cp > 0 and cp < len(t) - 2):
            labels.append("hanging_conditional")
    if _CAUSAL_ANA.match(t):
        labels.append("causal_anaphora")
    if _DEPEND_PREP.match(t):
        labels.append("dependency_preposition")
    if _PRONOUN_OPEN.match(t) and not labels:
        labels.append("pronoun_dependent")
    if _FORMAT_JUNK.search(t):
        labels.append("format_garbage")
    if quality_score < 0.35 and not labels:
        labels.append("low_authenticity")

    return labels or ["unclassified"]


# 模式元信息（用于报告输出）
_PATTERN_META: Dict[str, Dict] = {
    "stage_direction": {
        "description": "括号/方括号舞台指示或音效符号，非真实台词（如 (Cheers and applause)）",
        "suggestion": "已在第一道基础过滤中添加 _PAREN_STAGE_RE，可新增常见观众反应词到黑名单",
        "filter_stage": "G1（基础格式）",
        "already_fixed": True,
    },
    "self_introduction": {
        "description": "演讲者/角色自我介绍，姓名绑定导致不可复用（如 I'm Keke Palmer.）",
        "suggestion": "已在第二道黑名单过滤中添加 _SELF_INTRO_RE，可扩展 'My name is' 变体",
        "filter_stage": "G2（黑名单）",
        "already_fixed": True,
    },
    "trailing_truncation": {
        "description": "句末省略号或悬挂逗号，句子被截断（如 ...going back to how we were living be...）",
        "suggestion": "L2 P1 惩罚 -0.30 已覆盖，若该模式占比高可考虑升级为直接拒绝",
        "filter_stage": "L2（语用因子）",
        "already_fixed": True,
    },
    "continuation_opener": {
        "description": "续接词+强指称词开头，预设先行上下文（如 And that's why / But they said）",
        "suggestion": "L2 P2 惩罚 -0.25 已覆盖，可扩展更多续接词模式",
        "filter_stage": "L2（语用因子）",
        "already_fixed": True,
    },
    "hanging_conditional": {
        "description": "悬挂条件从句，缺少主句（如 If you really want to.）",
        "suggestion": "L2 P3 惩罚 -0.20 已覆盖",
        "filter_stage": "L2（语用因子）",
        "already_fixed": True,
    },
    "causal_anaphora": {
        "description": "因果回指词开头，必须有先行句才能理解（如 As a result / That being said）",
        "suggestion": "L2 P4 惩罚 -0.15 已覆盖",
        "filter_stage": "L2（语用因子）",
        "already_fixed": True,
    },
    "dependency_preposition": {
        "description": "依存性介词/副词起头，预设上文情境（如 alongside / on top of that）",
        "suggestion": "L2 P5 惩罚 -0.15 已添加（本次新增）",
        "filter_stage": "L2（语用因子）",
        "already_fixed": True,
    },
    "pronoun_dependent": {
        "description": "第三人称/指示代词开头，没有先行词导致指代不明（如 He commits. / They decided that.）",
        "suggestion": "可考虑添加 L2 P6：第三人称/指示代词开头时施加轻度惩罚 -0.10",
        "filter_stage": "待新增",
        "already_fixed": False,
    },
    "format_garbage": {
        "description": "字幕制作标注、网址、节目信息等格式垃圾",
        "suggestion": "第四道评分已归零处理，可补充更多正则模式",
        "filter_stage": "G4（质量评分）",
        "already_fixed": True,
    },
    "low_authenticity": {
        "description": "口语真实性评分过低，但未命中具体规则（缩略语/词汇密度/短语动词均不足）",
        "suggestion": "检查是否为非英语内容、代码片段或模板语言，可补充到格式垃圾正则",
        "filter_stage": "G4（质量评分）",
        "already_fixed": False,
    },
    "unclassified": {
        "description": "未命中任何已知规则，需人工复审",
        "suggestion": "积累足够样本后归纳新规则类型",
        "filter_stage": "—",
        "already_fixed": False,
    },
}


# ──────────────────────────────────────────────────────────
# 主分析函数
# ──────────────────────────────────────────────────────────

async def analyze_deprecated_nodes(session) -> Dict:
    """
    分析 knowledge_nodes 表中 is_active=False 的节点。

    Args:
        session: SQLAlchemy AsyncSession

    Returns:
        完整分析报告字典
    """
    from sqlalchemy import select, func
    from models.knowledge_node import KnowledgeNode

    # ── 查询全部弃用节点 ──
    stmt = select(
        KnowledgeNode.id,
        KnowledgeNode.en_text,
        KnowledgeNode.quality_score,
        KnowledgeNode.show_name,
        KnowledgeNode.scene,
        KnowledgeNode.difficulty,
    ).where(KnowledgeNode.is_active == False)

    rows = (await session.execute(stmt)).all()
    total = len(rows)

    if total == 0:
        return {
            "total_deprecated": 0,
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "message": "没有弃用节点，无需分析",
            "patterns": {},
            "filter_recommendations": [],
        }

    # ── 按模式分类 ──
    pattern_nodes: Dict[str, List] = {}   # pattern_key → list of (en_text, quality_score)
    scores: List[float] = []

    for row in rows:
        q = row.quality_score or 0.0
        scores.append(q)
        labels = _classify_node(row.en_text, q)
        for label in labels:
            pattern_nodes.setdefault(label, []).append((row.en_text, q))

    # ── 质量分分布 ──
    buckets = {"0.0-0.3": 0, "0.3-0.4": 0, "0.4-0.5": 0, "0.5-0.6": 0, "0.6+": 0}
    for s in scores:
        if s < 0.3:   buckets["0.0-0.3"] += 1
        elif s < 0.4: buckets["0.3-0.4"] += 1
        elif s < 0.5: buckets["0.4-0.5"] += 1
        elif s < 0.6: buckets["0.5-0.6"] += 1
        else:         buckets["0.6+"] += 1

    score_dist = {
        "buckets": buckets,
        "mean": round(statistics.mean(scores), 4) if scores else 0.0,
        "median": round(statistics.median(scores), 4) if scores else 0.0,
        "stdev": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
    }

    # ── 构建 patterns 输出 ──
    patterns_out: Dict = {}
    for key, node_list in sorted(pattern_nodes.items(), key=lambda x: -len(x[1])):
        meta = _PATTERN_META.get(key, {"description": key, "suggestion": "—", "already_fixed": False})
        examples = [en for en, _ in node_list[:3]]
        avg_q = round(statistics.mean(q for _, q in node_list), 4) if node_list else 0.0
        patterns_out[key] = {
            "count": len(node_list),
            "ratio": round(len(node_list) / total, 4),
            "avg_quality_score": avg_q,
            "description": meta["description"],
            "filter_stage": meta.get("filter_stage", "—"),
            "already_fixed": meta.get("already_fixed", False),
            "examples": examples,
            "suggestion": meta["suggestion"],
        }

    # ── 生成过滤建议 ──
    recommendations: List[Dict] = []

    # 规则1：stage_direction 占比 > 5% → 建议加入专项拒绝
    sd = pattern_nodes.get("stage_direction", [])
    if len(sd) / total > 0.05:
        recommendations.append({
            "signal": "stage_direction",
            "action": "将括号舞台指示直接归入 rejected_blacklist（已通过 _PAREN_STAGE_RE 实现）",
            "confidence": "high",
            "impact": f"可拦截 {len(sd)} 条 ({len(sd)/total:.1%}) 弃用节点",
        })

    # 规则2：self_introduction 出现 → 建议扩展黑名单
    si = pattern_nodes.get("self_introduction", [])
    if si:
        recommendations.append({
            "signal": "self_introduction",
            "action": "演讲者自我介绍已加入黑名单过滤，可进一步收集更多姓名格式变体",
            "confidence": "high",
            "impact": f"可拦截 {len(si)} 条 ({len(si)/total:.1%}) 弃用节点",
        })

    # 规则3：pronoun_dependent 占比 > 10% → 建议新增 L2 P6
    pd = pattern_nodes.get("pronoun_dependent", [])
    if len(pd) / total > 0.10:
        recommendations.append({
            "signal": "pronoun_dependent",
            "action": "建议在 L2 语用因子中新增 P6：第三人称代词开头施加 -0.10 惩罚",
            "confidence": "medium",
            "impact": f"可影响 {len(pd)} 条 ({len(pd)/total:.1%}) 弃用节点",
        })

    # 规则4：unclassified 占比 > 20% → 建议人工复审
    uc = pattern_nodes.get("unclassified", [])
    if len(uc) / total > 0.20:
        recommendations.append({
            "signal": "unclassified",
            "action": f"有 {len(uc)} 条未分类节点，建议人工复审，归纳新规则类型",
            "confidence": "low",
            "impact": f"未知，需人工判断",
        })

    # 规则5：quality_score 均值 > 0.5 → 阈值可能过松
    if score_dist["mean"] > 0.5:
        recommendations.append({
            "signal": "quality_threshold",
            "action": f"弃用节点均值质量分 {score_dist['mean']}，高于 0.5，"
                      f"建议将 QUALITY_THRESHOLD 从 {_get_threshold()} 上调 0.02~0.05",
            "confidence": "medium",
            "impact": "提高整体入库质量，减少后续人工弃用操作",
        })

    return {
        "total_deprecated": total,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "patterns": patterns_out,
        "score_distribution": score_dist,
        "filter_recommendations": recommendations,
    }


def _get_threshold() -> float:
    """安全读取当前质量阈值（config 可能未设置时返回默认值）。"""
    try:
        import config
        return getattr(config, "QUALITY_THRESHOLD", 0.38)
    except Exception:
        return 0.38


async def schedule_post_deprecation_analysis(
    trigger_source: str = "unknown",
    delay_seconds: float = 3.0,
) -> None:
    """
    弃用操作完成后，异步提交后台分析任务。

    Args:
        trigger_source: 触发来源描述（用于日志追踪），如 "batch-update" / "re-evaluate"
        delay_seconds:  等待写库事务提交的缓冲时间（默认 3s）
    """
    import asyncio
    from tasks.task_runner import submit_task
    from database import SessionLocal

    async def _run():
        await asyncio.sleep(delay_seconds)   # 等待调用方事务提交
        async with SessionLocal() as s:
            result = await analyze_deprecated_nodes(s)
            logger.info(
                f"[弃用分析 | 触发来源={trigger_source}] "
                f"完成：{result['total_deprecated']} 条弃用节点，"
                f"识别 {len(result['patterns'])} 种模式，"
                f"{len(result['filter_recommendations'])} 条改进建议"
            )
            return result

    await submit_task(_run(), description="弃用知识节点模式分析")
