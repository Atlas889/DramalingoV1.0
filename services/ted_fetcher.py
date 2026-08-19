"""
TED 演讲字幕自动获取服务

复用 TED_ZIMU/ted_subtitle_extractor.py 的核心逻辑，
封装为可供 FastAPI 后台任务调用的 async 接口。

流程：
  1. yt-dlp 获取 TED 演讲可用字幕语言列表
  2. 下载英文 VTT + 中文 VTT（若有）
  3. 有官方中文 → align_zh_to_en 对齐；
     无中文且有 DeepSeek Key → translate_with_deepseek 翻译；
     否则仅英文
  4. write_bilingual_srt 写出 SRT 文件
  5. 返回 SRT 文件绝对路径，供 subtitle_pipeline 后续处理
"""

import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)

# 将 TED_ZIMU 目录加入 Python 模块搜索路径（幂等）
_TED_DIR = Path(__file__).parent.parent / "TED_ZIMU"
if str(_TED_DIR) not in sys.path:
    sys.path.insert(0, str(_TED_DIR))


def _sync_fetch_and_write(
    ted_url: str,
    title_en: str,
    title_zh: str,
    deepseek_api_key: Optional[str],
    task_id: Optional[str],
) -> str:
    """
    同步执行 TED 字幕提取全流程（在 ThreadPoolExecutor 中调用）。
    返回生成的双语 SRT 文件的绝对路径字符串。
    """
    from ted_subtitle_extractor import (
        TedSubtitleFetcher,
        align_zh_to_en,
        translate_with_deepseek,
        polish_bilingual_with_deepseek,
        write_bilingual_srt,
        ZH_LANG_PRIORITY,
        ZH_TRADITIONAL,
    )
    from tasks.task_runner import update_task_progress

    def _prog(stage: str, msg: str, pct: int = 0, **kw):
        if task_id:
            update_task_progress(task_id, {
                "task_type": "ted_fetch",
                "stage": stage,
                "label": msg,
                "pct": pct,
                **kw,
            })

    # 临时 VTT 文件目录（yt-dlp 写中间文件）
    tmp_dir = Path(config.UPLOAD_DIR) / "ted_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # SRT 输出目录
    out_dir = Path(config.UPLOAD_DIR) / "ted_srt"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 初始化并检查可用语言 ──────────────────────────────────
    _prog("checking", "检查 TED 演讲字幕语言可用性…", 10)
    fetcher = TedSubtitleFetcher(ted_url, tmp_dir)
    available = fetcher.available_langs()
    logger.info(f"TED 可用字幕语言: {available}")

    if "en" not in available:
        raise ValueError("该 TED 演讲没有英文字幕，无法处理")

    # 仅接受简体中文作为官方字幕；繁体中文跳过，改走 AI 翻译
    zh_lang = next((c for c in ZH_LANG_PRIORITY if c in available), None)
    zh_trad = next((c for c in ZH_TRADITIONAL  if c in available), None)
    langs_to_fetch = ["en"] + ([zh_lang] if zh_lang else [])
    if zh_lang:
        logger.info(f"准备下载语言: {langs_to_fetch}，官方简体中文: {zh_lang}")
    elif zh_trad:
        logger.info(f"发现繁体中文字幕 ({zh_trad})，跳过，将由 AI 翻译为简体中文")
    else:
        logger.info("无官方中文字幕，将由 AI 翻译")

    # ── 2. 下载字幕 VTT ──────────────────────────────────────────
    _prog("downloading", f"下载字幕文件 ({', '.join(langs_to_fetch)})…", 30)
    subs = fetcher.download(langs_to_fetch)

    en_caps = subs.get("en")
    if not en_caps:
        raise ValueError("英文字幕解析失败，请确认 TED 链接有效且网络可达")

    zh_caps = subs.get(zh_lang) if zh_lang else None

    # ── 3. 中文处理：无论有无官方字幕，都经过 AI 加工 ───────────
    api_key   = deepseek_api_key or config.AI_API_KEY
    trad_note = f"（繁体 {zh_trad} 已跳过）" if zh_trad else ""

    if zh_caps:
        # 有官方简体字幕 → 先对齐时间轴，再 AI 润色
        _prog("aligning",
              f"对齐官方简体中英文字幕（en={len(en_caps)}, zh={len(zh_caps)}）…", 50)
        zh_aligned = align_zh_to_en(en_caps, zh_caps)
        logger.info(f"官方简体字幕对齐完成，语言: {zh_lang}")

        if api_key:
            key_label = "传入 Key" if deepseek_api_key else "系统配置 Key"
            _prog("translating",
                  f"AI 润色简体中文字幕（{len(en_caps)} 条）…", 70)
            zh_texts  = polish_bilingual_with_deepseek(en_caps, zh_aligned, api_key)
            zh_source = f"TED 官方简体 ({zh_lang}) + AI 润色"
            logger.info(f"AI 润色完成，使用 {key_label}")
        else:
            zh_texts  = zh_aligned
            zh_source = f"TED 官方简体 ({zh_lang})（未润色，无 AI_API_KEY）"
            logger.warning("AI_API_KEY 为空，跳过润色，直接使用对齐后的官方字幕")
    else:
        # 无简体中文字幕（含跳过繁体的情况）→ AI 翻译 + 润色一步完成
        if api_key:
            key_label = "传入 Key" if deepseek_api_key else "系统配置 Key"
            _prog("translating",
                  f"AI 翻译并润色 {len(en_caps)} 条字幕为简体中文{trad_note}…", 60)
            zh_texts  = translate_with_deepseek(en_caps, api_key)
            zh_source = f"DeepSeek 翻译润色（简体）{trad_note}"
            logger.info(f"AI 翻译润色完成，使用 {key_label}{trad_note}")
        else:
            zh_texts  = [""] * len(en_caps)
            zh_source = f"仅英文（无 AI_API_KEY）{trad_note}"
            logger.warning(f"无简体中文字幕且 AI_API_KEY 为空，仅保留英文字幕{trad_note}")

    # ── 4. 写出双语 SRT ──────────────────────────────────────────
    _prog("writing", "生成双语 SRT 文件…", 85)
    safe_name = re.sub(r"[^\w\-]", "_", title_en or title_zh or "ted_talk")[:50]
    srt_path = out_dir / f"{safe_name}_bilingual.srt"
    write_bilingual_srt(en_caps, zh_texts, srt_path)

    logger.info(f"TED 字幕生成完成: {srt_path} | {len(en_caps)} 条 | {zh_source}")
    _prog(
        "srt_done",
        f"字幕文件已就绪，等待管理员确认发送（{len(en_caps)} 条，{zh_source}）",
        100,
        caption_count=len(en_caps),
        zh_source=zh_source,
        srt_path=str(srt_path),
    )

    return {
        "srt_path":     str(srt_path),
        "caption_count": len(en_caps),
        "zh_source":    zh_source,
        "title_zh":     title_zh,
        "title_en":     title_en,
    }


async def fetch_ted_subtitle(
    ted_url: str,
    title_en: str,
    title_zh: str,
    deepseek_api_key: Optional[str] = None,
    task_id: Optional[str] = None,
) -> dict:
    """
    异步包装：在线程池中执行同步的 TED 字幕提取。

    Args:
        ted_url:          TED 演讲 URL
        title_en:         英文标题（用于 SRT 文件命名）
        title_zh:         中文标题（传给 subtitle_pipeline 作为剧名）
        deepseek_api_key: DeepSeek API Key（无官方中文字幕时需要）
        task_id:          后台任务 ID（用于进度上报）

    Returns:
        {"srt_path": str, "caption_count": int, "zh_source": str,
         "title_zh": str, "title_en": str}
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        _sync_fetch_and_write,
        ted_url,
        title_en,
        title_zh,
        deepseek_api_key,
        task_id,
    )
