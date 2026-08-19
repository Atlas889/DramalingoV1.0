"""
DramaLingo · 多信号查询引擎

综合以下三路语义信号对知识节点排序：
  1. zh_sim      — 中文查询向量 vs 节点 zh_embedding（中文翻译向量）
  2. en_tr_sim   — 中文查询翻译为英文后的向量 vs 节点 en_embedding（英文字幕向量）
  3. en_sim      — 中文查询向量跨语言 vs 节点 en_embedding（baseline，不依赖翻译）

外加辅助信号：
  4. tag_score   — 场景/情绪关键词匹配
  5. quality     — 字幕质量评分
  6. occurrence  — 出现频次

公开接口：
  detect_tag_hints(query)                         -> (scene, emotion)
  compute_tag_score(node_scene, node_emotion, ...) -> float
  translate_to_english(zh_query)                  -> Optional[str]  (async, 带缓存)
  expand_query(zh_query)                          -> QueryExpansion  (async, 带缓存)
  compute_final_score(zh_sim, en_sim, ...)         -> float
"""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 标签关键词（中文场景/情绪暗示词典）
# ──────────────────────────────────────────────

_SCENE_HINTS: Dict[str, List[str]] = {
    "romance": [
        "爱情", "恋爱", "表白", "情感", "感情", "浪漫", "心意", "情意", "暗恋",
        "告白", "约会", "示爱", "倾慕", "相爱", "情侣", "求爱", "撩", "爱意",
        "爱恋", "喜欢", "爱你", "追求", "情话", "花心思", "用情",
    ],
    "conflict": [
        "争吵", "冲突", "矛盾", "吵架", "愤怒", "争执", "对立", "指责",
        "批评", "骂人", "怼", "闹翻", "争论", "责备", "抱怨",
    ],
    "work": [
        "工作", "职场", "办公", "会议", "上班", "老板", "同事", "项目",
        "业务", "职业", "汇报", "面试", "升职", "任务", "开会", "客户",
    ],
    "humor": [
        "幽默", "搞笑", "开玩笑", "有趣", "逗", "笑话", "段子", "调侃",
        "嘲讽", "戏谑", "打趣", "逗乐",
    ],
    "daily": [
        "日常", "生活", "聊天", "闲聊", "问候", "打招呼", "叙旧",
        "话家常", "寒暄", "闲话", "说说",
    ],
}

_EMOTION_HINTS: Dict[str, List[str]] = {
    "happy": [
        "开心", "高兴", "快乐", "幸福", "喜悦", "兴奋", "激动",
        "欣喜", "愉快", "欢喜", "乐呵", "喜笑颜开",
    ],
    "sad": [
        "悲伤", "难过", "伤心", "哭泣", "失落", "沮丧", "痛苦",
        "难受", "忧伤", "伤感", "心碎", "遗憾",
    ],
    "angry": [
        "愤怒", "生气", "发火", "气愤", "恼火", "暴怒", "发怒",
        "气恼", "激怒", "怒火", "火大",
    ],
    "surprised": [
        "惊讶", "震惊", "意外", "吃惊", "出乎意料", "惊喜",
        "惊呆", "没想到", "想不到", "不敢相信",
    ],
}


def detect_tag_hints(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    从中文查询文本中检测场景和情绪暗示。

    Returns:
        (detected_scene, detected_emotion)，无匹配时对应位置为 None
    """
    scene_scores: Dict[str, int] = {}
    for scene, keywords in _SCENE_HINTS.items():
        count = sum(1 for kw in keywords if kw in query)
        if count > 0:
            scene_scores[scene] = count

    emotion_scores: Dict[str, int] = {}
    for emotion, keywords in _EMOTION_HINTS.items():
        count = sum(1 for kw in keywords if kw in query)
        if count > 0:
            emotion_scores[emotion] = count

    detected_scene = max(scene_scores, key=scene_scores.get) if scene_scores else None
    detected_emotion = max(emotion_scores, key=emotion_scores.get) if emotion_scores else None
    return detected_scene, detected_emotion


def compute_tag_score(
    node_scene: str,
    node_emotion: str,
    detected_scene: Optional[str],
    detected_emotion: Optional[str],
) -> float:
    """
    计算标签匹配得分 [0, 1]。

    场景匹配权重 60%，情绪匹配权重 40%。
    若查询只检测到一种标签，则该标签贡献全部得分。
    """
    if detected_scene is None and detected_emotion is None:
        return 0.0

    scene_match = 1.0 if (detected_scene and node_scene == detected_scene) else 0.0
    emotion_match = 1.0 if (detected_emotion and node_emotion == detected_emotion) else 0.0

    if detected_scene is not None and detected_emotion is not None:
        return scene_match * 0.6 + emotion_match * 0.4
    elif detected_scene is not None:
        return scene_match
    else:
        return emotion_match


# ──────────────────────────────────────────────
# 查询翻译（带内存缓存，异步）
# ──────────────────────────────────────────────

_translation_cache: Dict[str, Optional[str]] = {}
_MAX_CACHE_SIZE = 500


async def translate_to_english(zh_query: str) -> Optional[str]:
    """
    将中文查询翻译为英文，用于搜索英文字幕的 FAISS 索引。

    使用与 AI 打标相同的 API 配置（DeepSeek / Qwen）。
    结果缓存在内存中（最多 500 条），重启后失效。
    若 AI_API_KEY 未配置或翻译失败，返回 None（降级为纯跨语言检索）。
    """
    if not config.AI_API_KEY:
        return None

    if zh_query in _translation_cache:
        return _translation_cache[zh_query]

    provider = config.AI_PROVIDER
    if provider == "deepseek":
        base_url = config.AI_API_BASE_URL or "https://api.deepseek.com/v1"
        model = "deepseek-chat"
    elif provider == "qwen":
        base_url = config.AI_API_BASE_URL or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = "qwen-turbo"
    else:
        return None

    en_text: Optional[str] = None
    try:
        import httpx
        headers = {
            "Authorization": f"Bearer {config.AI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a translator. Translate the Chinese input to concise, "
                        "natural English. Output only the translation, no explanation."
                    ),
                },
                {"role": "user", "content": zh_query},
            ],
            "temperature": 0.1,
            "max_tokens": 80,
        }
        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions", headers=headers, json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            en_text = data["choices"][0]["message"]["content"].strip().strip('"').strip("'")
    except Exception as e:
        logger.warning(f"查询翻译失败（降级为跨语言检索）: {e}")
        en_text = None

    # 写缓存（超限时删除最早条目）
    if len(_translation_cache) >= _MAX_CACHE_SIZE:
        del _translation_cache[next(iter(_translation_cache))]
    _translation_cache[zh_query] = en_text

    if en_text:
        logger.info(f"查询翻译: '{zh_query}' → '{en_text}'")
    return en_text


# ──────────────────────────────────────────────
# 查询意图扩展（带内存缓存，异步）
# ──────────────────────────────────────────────

@dataclass
class QueryExpansion:
    expanded_zh: str          # 语义丰富化后的中文描述，用于 zh_sim 检索
    candidate_en: List[str]   # 3条候选英文表达，并发搜索 en_index 取 max
    from_cache: bool = False


_expansion_cache: Dict[str, QueryExpansion] = {}

_EXPAND_SYSTEM_PROMPT = """\
You are an English learning assistant for Chinese speakers who want to find authentic English expressions used in TV dramas.

Given a Chinese description of what the user wants to express, output a JSON object with exactly two fields:
- "expanded_zh": An enriched Chinese description (≤80 chars) that captures the intent with synonyms and related contexts. Do NOT include any English.
- "candidate_en": An array of exactly 3 natural, colloquial English phrases a native speaker would say in this situation. Keep each phrase concise (≤12 words).

Output valid JSON only. No markdown, no explanation."""


async def expand_query(zh_query: str) -> QueryExpansion:
    """
    用 DeepSeek 理解用户意图，生成：
      - expanded_zh: 语义丰富的中文描述（替换 zh_sim 查询输入）
      - candidate_en: 3条候选英文表达（并发搜索 en_index，取每个节点的最高相似度）

    若 AI_API_KEY 未配置、QUERY_EXPAND_ENABLED=false 或调用失败，
    退回 translate_to_english() 兼容模式。
    """
    if zh_query in _expansion_cache:
        cached = _expansion_cache[zh_query]
        return QueryExpansion(
            expanded_zh=cached.expanded_zh,
            candidate_en=cached.candidate_en,
            from_cache=True,
        )

    fallback_en = await translate_to_english(zh_query)
    fallback = QueryExpansion(
        expanded_zh=zh_query,
        candidate_en=[fallback_en] if fallback_en else [],
    )

    if not config.AI_API_KEY or not config.QUERY_EXPAND_ENABLED:
        return fallback

    provider = config.AI_PROVIDER
    if provider == "deepseek":
        base_url = config.AI_API_BASE_URL or "https://api.deepseek.com/v1"
        model = "deepseek-chat"
    elif provider == "qwen":
        base_url = config.AI_API_BASE_URL or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = "qwen-turbo"
    else:
        return fallback

    try:
        import httpx
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _EXPAND_SYSTEM_PROMPT},
                {"role": "user", "content": zh_query},
            ],
            "temperature": 0.3,
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        }
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.AI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            data = json.loads(raw)

        expanded_zh = str(data.get("expanded_zh") or zh_query).strip()[:120]
        raw_en = data.get("candidate_en") or []
        candidate_en = [str(e).strip() for e in raw_en if e][:3]

        if not expanded_zh:
            expanded_zh = zh_query
        if not candidate_en and fallback_en:
            candidate_en = [fallback_en]

        result = QueryExpansion(expanded_zh=expanded_zh, candidate_en=candidate_en)
        logger.info(
            f"查询扩展: '{zh_query}' → zh='{expanded_zh[:40]}…' "
            f"en={candidate_en}"
        )

    except Exception as e:
        logger.warning(f"查询扩展失败，降级为翻译模式: {e}")
        result = fallback

    if len(_expansion_cache) >= _MAX_CACHE_SIZE:
        del _expansion_cache[next(iter(_expansion_cache))]
    _expansion_cache[zh_query] = result
    return result


# ──────────────────────────────────────────────
# 综合排序公式
# ──────────────────────────────────────────────

def _occurrence_norm(count: int) -> float:
    sat = config.RANK_OCCURRENCE_SATURATION
    if sat <= 0:
        return 0.0
    return min(1.0, math.log(1 + count) / math.log(1 + sat))


def compute_final_score(
    zh_sim: float,
    en_sim: float,
    en_tr_sim: float,
    tag_score: float,
    quality: float,
    occurrence: int,
) -> float:
    """
    多信号综合排序分（满分理论值 = 1.0，实际因各信号覆盖情况会低于 1.0）。

    zh_sim×0.40 + en_tr_sim×0.25 + en_sim×0.15 + tag×0.12 + quality×0.06 + occ×0.02
    """
    return round(
        zh_sim * config.RANK_WEIGHT_ZH_SIM
        + en_tr_sim * config.RANK_WEIGHT_EN_TRANSLATED
        + en_sim * config.RANK_WEIGHT_EN_SIM
        + tag_score * config.RANK_WEIGHT_TAG
        + quality * config.RANK_WEIGHT_QUALITY
        + _occurrence_norm(occurrence) * config.RANK_WEIGHT_OCCURRENCE,
        6,
    )
