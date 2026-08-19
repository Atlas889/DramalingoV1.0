"""
DramaLingo · AI 打标与翻译模块
对应 PRD v0.9 §6.1 AI 打标 / §5 流程一·AI 打标失败降级

功能：
  - 调用 DeepSeek/Qwen API 为知识节点生成 scene/emotion/difficulty/keywords/zh_text
  - 单次调用返回全部标签（PRD §6.1 约束，不分多次调用）
  - 调用失败时自动降级为规则打标，不中断流水线

公开接口：
  tag_and_translate(en_text, provider)  -> dict  (异步)
  rule_based_tag(en_text, word_count)   -> dict
"""

import asyncio
import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 自定义异常
# ──────────────────────────────────────────────

class AITaggingError(Exception):
    """AI 打标失败，触发降级规则打标"""
    pass


# ──────────────────────────────────────────────
# 枚举值定义（PRD §7.3 knowledge_nodes 字段约束）
# ──────────────────────────────────────────────

VALID_SCENES = {"daily", "work", "romance", "conflict", "humor"}

VALID_EMOTIONS = {"neutral", "happy", "sad", "angry", "surprised"}

VALID_DIFFICULTIES = {"beginner", "intermediate", "advanced"}

# 言语行为标签（Speech Act Theory · Searle 分类体系扩展）
# 标注句子在交际中「做了什么事」，用于语用功能检索和教学分组
VALID_SPEECH_ACTS = {
    # 表达类 Expressives
    "apologizing",          # 道歉: Sorry / I didn't mean to...
    "comforting",           # 安慰: It's okay / Don't worry about it
    "complaining",          # 抱怨: This always happens / I can't believe...
    "complimenting",        # 称赞: You look amazing / Well done
    "expressing_surprise",  # 惊讶: No way! / Are you serious?
    "expressing_doubt",     # 质疑: Are you sure? / I'm not buying it
    "regretting",           # 后悔: I should have / I wish I hadn't
    # 指令类 Directives
    "requesting",           # 请求: Could you / Would you mind...
    "offering",             # 提议: Let me help / How about I...
    "warning",              # 警告: Watch out / Be careful
    "threatening",          # 威胁: If you do that... / You'll regret it
    "refusing",             # 拒绝: I can't do that / No, I won't
    # 承诺类 Commissives
    "promising",            # 承诺: I swear / You have my word
    # 断言类 Assertives
    "agreeing",             # 赞同: Exactly / That's what I thought
    "disagreeing",          # 反对: I don't think so / Actually...
    "emphasizing",          # 强调: The point is / What matters is
    "hedging",              # 缓和: Maybe / Kind of / I suppose
    "explaining",           # 解释: The reason is / What happened was
    "justifying",           # 辩护: I had to / There was no other way
    "speculating",          # 推测: I wonder / Perhaps / It could be
    "criticizing",          # 批评: That was wrong / You shouldn't have
    # 语篇控制 Discourse
    "initiating",           # 开场: Hey, listen / Can I ask you something
    "changing_topic",       # 转移话题: Anyway / Moving on / Speaking of which
    "closing",              # 结束: See you / Take care / Goodbye
    "neutral_statement",    # 中性陈述（以上均不符合）
}


# ──────────────────────────────────────────────
# AI 打标 Prompt 模板
# ──────────────────────────────────────────────

_PROMPT_TEMPLATE = '''Given the English subtitle line from a TV show:
"__EN_TEXT__"

Please provide:
1. scene: The scene type — choose ONE from exactly these values:
   daily (everyday life/casual conversation), work (workplace/office/professional),
   romance (romantic/emotional/relationships), conflict (arguments/disputes/tension),
   humor (funny/comedic/witty)
2. emotion: The emotion — choose ONE from exactly these values:
   neutral, happy, sad, angry, surprised
3. difficulty: Language difficulty — choose ONE from exactly these values:
   beginner (simple vocabulary, short sentence), intermediate (moderate complexity),
   advanced (idioms/complex grammar/slang)
4. keywords: 3-5 English keywords representing the core meaning
5. speech_act: The pragmatic function this utterance performs — choose ONE from exactly these values:
   apologizing, comforting, complaining, complimenting, expressing_surprise, expressing_doubt,
   regretting, requesting, offering, warning, threatening, refusing, promising,
   agreeing, disagreeing, emphasizing, hedging, explaining, justifying, speculating,
   criticizing, initiating, changing_topic, closing, neutral_statement

Return ONLY valid JSON format, no extra text:
{"scene": "...", "emotion": "...", "difficulty": "...", "keywords": ["...", "..."], "speech_act": "..."}'''

# 批量打标 Prompt（10 条一批，API 调用数降为 1/10）
_BATCH_PROMPT_TEMPLATE = '''You are analyzing English TV subtitle lines. For each numbered line below, output one JSON object.

Lines:
__LINES__

For EACH line provide exactly these fields:
- scene: ONE of: daily | work | romance | conflict | humor
- emotion: ONE of: neutral | happy | sad | angry | surprised
- difficulty: ONE of: beginner | intermediate | advanced
- keywords: array of 3-5 English keywords
- speech_act: ONE of: apologizing | comforting | complaining | complimenting | expressing_surprise |
  expressing_doubt | regretting | requesting | offering | warning | threatening | refusing |
  promising | agreeing | disagreeing | emphasizing | hedging | explaining | justifying |
  speculating | criticizing | initiating | changing_topic | closing | neutral_statement

Return a JSON ARRAY (same order as input, no extra text):
[{"scene":"...","emotion":"...","difficulty":"...","keywords":[...],"speech_act":"..."}, ...]'''


# ──────────────────────────────────────────────
# AI 打标（异步）
# ──────────────────────────────────────────────

async def tag_and_translate(
    en_text: str,
    provider: Optional[str] = None,
) -> dict:
    """
    调用 AI API 为字幕行生成标签和中文翻译。

    Args:
        en_text:  英文字幕文本
        provider: API 提供商（"deepseek" 或 "qwen"），默认从 config 读取

    Returns:
        dict，包含 scene/emotion/difficulty/keywords/zh_text/tag_source

    Raises:
        AITaggingError: API 调用失败或返回格式异常时抛出
    """
    if provider is None:
        provider = config.AI_PROVIDER

    if not config.AI_API_KEY:
        raise AITaggingError("AI_API_KEY 未配置，无法调用 AI 打标")

    # 用 replace 替换占位符，避免字幕文本含花括号时 str.format() 抛 KeyError
    prompt = _PROMPT_TEMPLATE.replace("__EN_TEXT__", en_text)

    # 根据 provider 选择 API 配置
    if provider == "deepseek":
        base_url = config.AI_API_BASE_URL or "https://api.deepseek.com/v1"
        model = "deepseek-chat"
    elif provider == "qwen":
        base_url = config.AI_API_BASE_URL or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = "qwen-turbo"
    else:
        raise AITaggingError(f"不支持的 AI 提供商: {provider}")

    for attempt in range(config.AI_MAX_RETRIES + 1):
        try:
            result = await _call_api(base_url, model, prompt)
            return result
        except AITaggingError as e:
            if attempt < config.AI_MAX_RETRIES:
                # 429 速率限制用指数退避，其他错误固定 1 秒
                wait = (2 ** attempt) if "429" in str(e) else 1
                logger.warning("AI 打标第 %d 次失败，%.0fs 后重试: %s", attempt + 1, wait, e)
                await asyncio.sleep(wait)
            else:
                raise


async def _call_api(base_url: str, model: str, prompt: str) -> dict:
    """
    实际调用 OpenAI 兼容 API。

    Args:
        base_url: API 基础 URL
        model:    模型名称
        prompt:   完整 Prompt

    Returns:
        解析后的标签字典

    Raises:
        AITaggingError: 调用失败或解析失败
    """
    try:
        import httpx
    except ImportError:
        # 降级：使用 requests（同步，在 asyncio 中用 run_in_executor）
        return await _call_api_sync(base_url, model, prompt)

    headers = {
        "Authorization": f"Bearer {config.AI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes English TV subtitle lines and returns structured JSON data.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,  # 低温度保证输出稳定
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=config.AI_TIMEOUT) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException as e:
        raise AITaggingError(f"AI API 超时: {e}")
    except httpx.HTTPStatusError as e:
        raise AITaggingError(f"AI API HTTP 错误 {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        raise AITaggingError(f"AI API 调用异常: {e}")

    return _parse_response(data)


async def _call_api_sync(base_url: str, model: str, prompt: str) -> dict:
    """使用 requests 的同步 API 调用（httpx 不可用时的降级方案）"""
    import requests

    headers = {
        "Authorization": f"Bearer {config.AI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes English TV subtitle lines and returns structured JSON data.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }

    loop = asyncio.get_running_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=config.AI_TIMEOUT,
            ),
        )
        response.raise_for_status()
        data = response.json()
    except requests.Timeout as e:
        raise AITaggingError(f"AI API 超时: {e}")
    except requests.HTTPError as e:
        raise AITaggingError(f"AI API HTTP 错误: {e}")
    except Exception as e:
        raise AITaggingError(f"AI API 调用异常: {e}")

    return _parse_response(data)


def _parse_response(data: dict) -> dict:
    """
    解析 API 响应，提取并验证标签字段。

    Args:
        data: API 响应的原始 JSON

    Returns:
        标准化的标签字典

    Raises:
        AITaggingError: 响应格式异常
    """
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise AITaggingError(f"API 响应格式异常: {e}，响应: {str(data)[:200]}")

    # 尝试解析 JSON（有时模型会在 JSON 前后加多余文字）
    try:
        tags = json.loads(content)
    except json.JSONDecodeError:
        # 尝试从文本中提取 JSON 块
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            try:
                tags = json.loads(json_match.group())
            except json.JSONDecodeError as e:
                raise AITaggingError(f"JSON 解析失败: {e}，内容: {content[:200]}")
        else:
            raise AITaggingError(f"响应中未找到 JSON: {content[:200]}")

    # 字段验证与修正
    scene = tags.get("scene", "daily")
    if scene not in VALID_SCENES:
        logger.warning(f"scene 值 '{scene}' 不在枚举范围，修正为 'daily'")
        scene = "daily"

    emotion = tags.get("emotion", "neutral")
    if emotion not in VALID_EMOTIONS:
        logger.warning(f"emotion 值 '{emotion}' 不在枚举范围，修正为 'neutral'")
        emotion = "neutral"

    difficulty = tags.get("difficulty", "intermediate")
    if difficulty not in VALID_DIFFICULTIES:
        logger.warning(f"difficulty 值 '{difficulty}' 不在枚举范围，修正为 'intermediate'")
        difficulty = "intermediate"

    keywords = tags.get("keywords", [])
    if not isinstance(keywords, list):
        keywords = []
    # 确保 keywords 是字符串列表，最多 5 个
    keywords = [str(k) for k in keywords[:5] if k]
    if not keywords:
        logger.warning("AI 返回的 keywords 为空，可能影响检索质量")

    speech_act = tags.get("speech_act", "neutral_statement")
    if speech_act not in VALID_SPEECH_ACTS:
        logger.warning(f"speech_act 值 '{speech_act}' 不在枚举范围，修正为 'neutral_statement'")
        speech_act = "neutral_statement"

    return {
        "scene": scene,
        "emotion": emotion,
        "difficulty": difficulty,
        "keywords": keywords,
        "speech_act": speech_act,
        "tag_source": "ai",
        "needs_translation": False,
    }


# ──────────────────────────────────────────────
# 降级规则打标
# ──────────────────────────────────────────────

def rule_based_tag(en_text: str, word_count: int) -> dict:
    """
    降级规则打标（PRD §5 流程一·降级规则打标定义）。
    当 AI 打标失败时调用，保证流水线不中断。

    难度判定规则：
    - word_count ≤ 8  → beginner
    - word_count ≤ 20 → intermediate
    - word_count > 20 → advanced

    Args:
        en_text:    英文字幕文本（当前未使用，预留扩展）
        word_count: 词数

    Returns:
        标准化的标签字典（tag_source="rule"）
    """
    if word_count <= 8:
        difficulty = "beginner"
    elif word_count <= 20:
        difficulty = "intermediate"
    else:
        difficulty = "advanced"

    return {
        "scene": "daily",
        "emotion": "neutral",
        "difficulty": difficulty,
        "keywords": [],
        "speech_act": "neutral_statement",
        "tag_source": "rule",
        "needs_translation": False,
    }


# ──────────────────────────────────────────────
# 批量打标（批量 Prompt + 并发控制）
# ──────────────────────────────────────────────

# 全局 semaphore 缓存，按 provider 名称 → Semaphore
# 懒初始化（asyncio.Semaphore 必须在事件循环内创建），跨调用复用，实现真正的全局限流
_provider_sems: Dict[str, asyncio.Semaphore] = {}

async def _tag_chunk(
    texts: List[str],
    word_counts: List[int],
    base_url: str,
    model: str,
    api_key: str = "",
    provider_name: str = "ai",
) -> List[dict]:
    """
    将一批字幕（通常 10 条）合并为单次 API 请求，返回与 texts 等长的标签列表。
    解析失败时逐条降级为规则打标，不整批放弃。
    api_key 留空时回落到 config.AI_API_KEY；provider_name 写入每条结果的 tag_source。
    """
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    prompt = _BATCH_PROMPT_TEMPLATE.replace("__LINES__", numbered)
    key = api_key or config.AI_API_KEY

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You analyze English TV subtitle lines and return structured JSON arrays.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 128 * len(texts),   # 每条约 128 tokens 足够
        "response_format": {"type": "json_object"},
    }

    try:
        import httpx
        async with httpx.AsyncClient(timeout=config.AI_TIMEOUT) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"批量打标 API 失败，整批降级规则打标: {e}")
        return [rule_based_tag(t, wc) for t, wc in zip(texts, word_counts)]

    # 解析返回的 JSON 数组
    try:
        raw = json.loads(content)
        # 模型有时把数组包在 {"results": [...]} 里
        if isinstance(raw, dict):
            for key in ("results", "data", "items", "tags"):
                if isinstance(raw.get(key), list):
                    raw = raw[key]
                    break
            else:
                raw = list(raw.values())[0] if raw else []
        if not isinstance(raw, list):
            raise ValueError("响应不是 JSON 数组")
    except Exception as e:
        logger.warning(f"批量打标 JSON 解析失败，整批降级规则打标: {e} | 内容: {content[:300]}")
        return [rule_based_tag(t, wc) for t, wc in zip(texts, word_counts)]

    # 逐条校验，数量不足时用规则补全
    results: List[dict] = []
    for i, (text, wc) in enumerate(zip(texts, word_counts)):
        item = raw[i] if i < len(raw) else {}
        try:
            scene = item.get("scene", "daily")
            if scene not in VALID_SCENES:
                scene = "daily"
            emotion = item.get("emotion", "neutral")
            if emotion not in VALID_EMOTIONS:
                emotion = "neutral"
            difficulty = item.get("difficulty", "intermediate")
            if difficulty not in VALID_DIFFICULTIES:
                difficulty = "intermediate"
            kws = item.get("keywords", [])
            keywords = [str(k) for k in (kws if isinstance(kws, list) else [])[:5] if k]
            speech_act = item.get("speech_act", "neutral_statement")
            if speech_act not in VALID_SPEECH_ACTS:
                speech_act = "neutral_statement"
            results.append({
                "scene": scene, "emotion": emotion, "difficulty": difficulty,
                "keywords": keywords, "speech_act": speech_act,
                "tag_source": provider_name, "needs_translation": False,
            })
        except Exception:
            results.append(rule_based_tag(text, wc))

    return results


def _resolve_providers() -> List[Tuple[str, str, str, str]]:
    """
    从 config 读取已配置的 provider 列表。
    每项为 (base_url, model, api_key, provider_name)。
    主 provider 优先，第二 provider 在 AI_PROVIDER_2 和 AI_API_KEY_2 均非空时追加。
    """
    _MODEL_MAP = {
        "deepseek": ("https://api.deepseek.com/v1", "deepseek-chat"),
        "qwen":     ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-turbo"),
    }

    providers: List[Tuple[str, str, str, str]] = []

    if config.AI_API_KEY and config.AI_PROVIDER in _MODEL_MAP:
        default_url, default_model = _MODEL_MAP[config.AI_PROVIDER]
        providers.append((
            config.AI_API_BASE_URL or default_url,
            default_model,
            config.AI_API_KEY,
            config.AI_PROVIDER,
        ))

    if config.AI_API_KEY_2 and config.AI_PROVIDER_2 in _MODEL_MAP:
        default_url2, default_model2 = _MODEL_MAP[config.AI_PROVIDER_2]
        providers.append((
            config.AI_API_BASE_URL_2 or default_url2,
            default_model2,
            config.AI_API_KEY_2,
            config.AI_PROVIDER_2,
        ))

    return providers


async def batch_tag_and_translate(
    texts: List[str],
    word_counts: List[int],
    max_concurrency: int = 10,
    batch_size: int = 10,
) -> Tuple[List[dict], Dict[str, dict]]:
    """
    批量 AI 打标：将 texts 切成 batch_size 大小的块，每块一次 API 请求，
    多块并发（最多 max_concurrency 个请求同时在途）。

    支持双 provider：chunks 按奇偶轮流分配给主/备两个 provider，
    各自独立限流，吞吐量接近翻倍。

    返回 (tags, provider_stats)：
      - tags: 与 texts 等长的标签列表
      - provider_stats: {"deepseek": {"count": N, "elapsed_s": X}, ...}
        rule 降级也会单独统计在 "rule" key 下
    """
    if not texts:
        return [], {}

    providers = _resolve_providers()
    if not providers:
        tags = [rule_based_tag(t, wc) for t, wc in zip(texts, word_counts)]
        return tags, {"rule": {"count": len(tags), "elapsed_s": 0.0}}

    n_providers = len(providers)

    # 切块
    chunks_texts: List[List[str]] = []
    chunks_wc: List[List[int]] = []
    for i in range(0, len(texts), batch_size):
        chunks_texts.append(texts[i: i + batch_size])
        chunks_wc.append(word_counts[i: i + batch_size])

    n_chunks = len(chunks_texts)

    # 按 provider 名称复用全局 semaphore（避免每次调用重建，保证跨并发任务的全局限流）
    for _, _, _, pname in providers:
        if pname not in _provider_sems:
            _provider_sems[pname] = asyncio.Semaphore(max_concurrency)

    provider_names = [p[3] for p in providers]
    if n_providers > 1:
        logger.info(
            f"批量打标（双 provider）：共 {len(texts)} 条，分 {n_chunks} 批，"
            f"轮流分配给 {' / '.join(provider_names)}"
        )
    else:
        logger.info(
            f"批量打标：共 {len(texts)} 条，分 {n_chunks} 批（每批 {batch_size} 条），"
            f"最大并发 {max_concurrency}"
        )

    # 每个 chunk 返回 (results, provider_name, elapsed_s)
    async def run_chunk(
        idx: int, chunk_texts: List[str], chunk_wc: List[int]
    ) -> Tuple[List[dict], str, float]:
        p_idx = idx % n_providers
        base_url, model, api_key, pname = providers[p_idx]
        sem = _provider_sems[pname]

        async with sem:
            total_elapsed = 0.0
            for attempt in range(config.AI_MAX_RETRIES + 1):
                try:
                    t0 = time.monotonic()
                    result = await _tag_chunk(
                        chunk_texts, chunk_wc, base_url, model, api_key, pname
                    )
                    total_elapsed += time.monotonic() - t0
                    logger.debug(f"批次 {idx+1}/{n_chunks} 打标完成（{pname} {total_elapsed:.1f}s）")
                    return result, pname, total_elapsed
                except Exception as e:
                    if attempt < config.AI_MAX_RETRIES:
                        wait = (2 ** attempt) if "429" in str(e) else 1
                        logger.warning(f"批次 {idx+1} 第 {attempt+1} 次失败，{wait}s 后重试: {e}")
                        await asyncio.sleep(wait)
                    else:
                        logger.warning(f"批次 {idx+1} 全部重试失败，降级规则打标")
                        return (
                            [rule_based_tag(t, wc) for t, wc in zip(chunk_texts, chunk_wc)],
                            "rule",
                            total_elapsed,
                        )

    raw_results = await asyncio.gather(
        *[run_chunk(i, ct, cw) for i, (ct, cw) in enumerate(zip(chunks_texts, chunks_wc))]
    )

    all_tags: List[dict] = []
    provider_stats: Dict[str, dict] = {}
    for chunk_res, pname, elapsed in raw_results:
        all_tags.extend(chunk_res)
        entry = provider_stats.setdefault(pname, {"count": 0, "elapsed_s": 0.0})
        entry["count"] += len(chunk_res)
        entry["elapsed_s"] += elapsed

    return all_tags, provider_stats
