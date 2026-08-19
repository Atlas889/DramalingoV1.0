#!/usr/bin/env python3
"""
TED Talk Bilingual Subtitle Extractor

Downloads subtitles from TED talks and creates bilingual (English + Chinese) SRT files.
- Uses yt-dlp to reliably download TED's HLS-segmented VTT subtitle files.
- If Chinese (zh-cn / zh-tw) subtitles exist on TED, they are merged with English.
- Otherwise, DeepSeek API translates the English subtitles to Chinese.

Usage:
    python ted_subtitle_extractor.py <ted_url> [--api-key KEY] [--output-dir DIR]

Environment:
    DEEPSEEK_API_KEY  DeepSeek API key (alternative to --api-key)
"""

import os
import re
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import httpx
import yt_dlp


# ── Constants ────────────────────────────────────────────────────────────────

DEEPSEEK_API_URL     = "https://api.deepseek.com/v1/chat/completions"
TRANSLATION_BATCH    = 40      # captions per DeepSeek request

# 仅接受简体中文字幕作为官方中文来源
ZH_LANG_PRIORITY     = ["zh-cn", "zh-hans", "zh"]
# 繁体中文代码——检测到时跳过，改用 AI 翻译
ZH_TRADITIONAL       = ["zh-tw", "zh-hant"]


# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Caption:
    index: int
    start_ms: int
    end_ms: int
    text: str


# ── Time helpers ─────────────────────────────────────────────────────────────

def ms_to_srt_time(ms: int) -> str:
    """Milliseconds → SRT timestamp HH:MM:SS,mmm."""
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vtt_time_to_ms(ts: str) -> int:
    """VTT/SRT timestamp (HH:MM:SS.mmm or MM:SS.mmm) → milliseconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = "0", parts[0], parts[1]
        else:
            return 0
        total  = int(h) * 3_600_000 + int(m) * 60_000
        s_int, _, ms_str = s.partition(".")
        total += int(s_int) * 1_000
        if ms_str:
            total += int(ms_str[:3].ljust(3, "0"))
        return total
    except ValueError:
        return 0


# ── URL parsing ───────────────────────────────────────────────────────────────

def parse_slug(url: str) -> str:
    """Extract the talk slug from any TED talk URL."""
    parsed = urlparse(url.strip())
    m = re.match(r"/talks/([^/?#]+)", parsed.path)
    if not m:
        raise ValueError(
            f"Not a valid TED talk URL: {url}\n"
            "Expected format: https://www.ted.com/talks/<slug>"
        )
    return m.group(1)


# ── .env loader ───────────────────────────────────────────────────────────────

def load_dotenv() -> None:
    """Load key=value pairs from a local .env file into os.environ (no extra deps)."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# ── VTT parser ────────────────────────────────────────────────────────────────

def parse_vtt_content(content: str) -> list[Caption]:
    """Parse a WebVTT string into Caption objects."""
    content = content.lstrip("﻿")           # strip UTF-8 BOM
    blocks  = re.split(r"\n\s*\n", content)
    captions: list[Caption] = []
    idx = 1

    for block in blocks:
        lines = [ln.rstrip() for ln in block.strip().splitlines()]
        if not lines:
            continue
        if lines[0].startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue

        # Find the timing line (may follow a cue identifier)
        timing_line: Optional[str] = None
        timing_pos = 0
        for j, ln in enumerate(lines):
            if "-->" in ln:
                timing_line = ln
                timing_pos  = j
                break
        if not timing_line:
            continue

        parts = timing_line.split("-->", 1)
        if len(parts) != 2:
            continue
        start_ms = vtt_time_to_ms(parts[0])
        end_ms   = vtt_time_to_ms(parts[1].strip().split()[0])   # strip cue settings

        # Collect and clean text lines (stop at any second timing line —
        # yt-dlp can embed duplicate cues at HLS segment boundaries)
        text_parts: list[str] = []
        for ln in lines[timing_pos + 1 :]:
            if "-->" in ln:
                break   # duplicate timing entry — discard remainder of block
            cleaned = re.sub(r"<[^>]+>", "", ln).strip()
            if cleaned:
                text_parts.append(cleaned)

        text = " ".join(text_parts).strip()
        if text and start_ms < end_ms:
            captions.append(Caption(idx, start_ms, end_ms, text))
            idx += 1

    return captions


# ── yt-dlp subtitle downloader ────────────────────────────────────────────────

class TedSubtitleFetcher:
    """
    Fetches TED subtitle VTT files using the yt-dlp Python API.
    yt-dlp handles TED's HLS-segmented subtitle format automatically.
    """

    def __init__(self, ted_url: str, output_dir: Path) -> None:
        self.ted_url    = ted_url
        self.output_dir = output_dir
        self._info: Optional[dict] = None

    # ------------------------------------------------------------------
    def _extract_info(self) -> dict:
        if self._info is None:
            opts = {"quiet": True, "no_warnings": True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                self._info = ydl.extract_info(self.ted_url, download=False)
        return self._info

    # ------------------------------------------------------------------
    def available_langs(self) -> list[str]:
        """Return the list of subtitle language codes TED provides."""
        info = self._extract_info()
        return list(info.get("subtitles", {}).keys())

    # ------------------------------------------------------------------
    def download(self, langs: list[str]) -> dict[str, list[Caption]]:
        """
        Download subtitle VTT files for the given language codes.
        Returns {lang_code: [Caption, …]}.
        """
        opts: dict = {
            "writesubtitles":  True,
            "subtitleslangs":  langs,
            "subtitlesformat": "vtt",
            "skip_download":   True,
            "outtmpl":         str(self.output_dir / "%(id)s"),
            "quiet":           True,
            "no_warnings":     True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(self.ted_url, download=True)
            self._info = info

        talk_id = info.get("id", "")
        result: dict[str, list[Caption]] = {}

        # yt-dlp writes subtitle files as  {talk_id}.{lang}.vtt
        for lang in langs:
            vtt_path = self.output_dir / f"{talk_id}.{lang}.vtt"
            if vtt_path.exists():
                content  = vtt_path.read_text(encoding="utf-8", errors="replace")
                captions = parse_vtt_content(content)
                if captions:
                    result[lang] = captions

        return result


# ── Caption alignment ─────────────────────────────────────────────────────────

def align_zh_to_en(
    en: list[Caption], zh: list[Caption], tolerance_ms: int = 2000
) -> list[str]:
    """
    Match Chinese captions to English captions.

    Strategy (in priority order):
    1. Containment: if the English start time falls inside a Chinese cue's
       time range, use that cue (handles the common case where Chinese
       subtitles use longer, coarser time segments).
    2. Nearest start-time: pick the closest Chinese cue within `tolerance_ms`.
    """
    if not zh:
        return [""] * len(en)

    # Identical count → assume 1-to-1 ordering
    if len(en) == len(zh):
        return [c.text for c in zh]

    result: list[str] = []
    for cap in en:
        # --- pass 1: containment ---
        matched = ""
        for zc in zh:
            if zc.start_ms <= cap.start_ms <= zc.end_ms:
                matched = zc.text
                break

        # --- pass 2: nearest start-time (fallback) ---
        if not matched:
            best_diff = float("inf")
            for zc in zh:
                diff = abs(zc.start_ms - cap.start_ms)
                if diff < best_diff:
                    best_diff = diff
                    matched   = zc.text
            if best_diff > tolerance_ms:
                matched = ""

        result.append(matched)
    return result


# ── DeepSeek translation & polishing ─────────────────────────────────────────

def _parse_numbered_output(output: str, fallback: list[str]) -> list[str]:
    """Parse numbered lines (「N. text」) from model output; pad with fallback if short."""
    result: list[str] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^\d+[.、：:]\s*", "", line)
        if cleaned:
            result.append(cleaned)
    while len(result) < len(fallback):
        result.append(fallback[len(result)])
    return result[: len(fallback)]


def _translate_batch(
    texts: list[str],
    api_key: str,
    prev_context: list[str] | None = None,
) -> list[str]:
    """
    Translate one batch of English subtitle lines to polished Simplified Chinese.
    Follows subtitle-polish-chinese v2.1 constraint system.
    prev_context: last few English lines from the previous batch (for continuity).
    """
    ctx_block = ""
    if prev_context:
        ctx_block = (
            "【前文参考（仅供上下文理解，无需翻译）】\n"
            + "\n".join(f"  {t}" for t in prev_context)
            + "\n\n"
        )

    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))

    prompt = (
        "你是专业的TED演讲字幕翻译润色专家，执行 subtitle-polish-chinese v2.1 标准。\n"
        "请将以下英文字幕批量翻译为简体中文，同步完成润色。\n\n"

        "【约束优先级（严格遵守）】\n"
        "① 字数上限（不可妥协）\n"
        "   · 每条译文：每行 ≤ 30 汉字，最多 2 行，优先控制在 1 行\n"
        "   · 字数比例：中文字数 ÷ 英文单词数 ≤ 1.5（理想约 1:1）\n"
        "② 上下文翻译（核心任务）\n"
        "   · 先通读全批字幕，把握演讲整体语境和情感弧线\n"
        "   · 翻译意思，不逐词直译；理解演讲者真实意图\n"
        "   · 注意前后字幕的连接词与递进关系\n"
        "   · 匹配演讲者语气：幽默段落要有趣，论证段落要清晰，呼吁段落要有力量\n"
        "③ 简洁优先\n"
        "   · 删除冗余修饰词，保留核心意思\n"
        "   · 简洁且准确 > 冗长且华丽\n"
        "④ 语言自然\n"
        "   · 口语化、地道中文，避免英文语序倒装\n"
        "   · 习语/俚语转化为对应中文地道说法\n"
        "⑤ 文化适配\n"
        "   · 在上述约束允许范围内进行文化本地化\n\n"

        "【典型对比示例】\n"
        "  'I'm blown away'          ✗ 我被吹走了        ✓ 我彻底折服了\n"
        "  'You're never asked back' ✗ 你永远不会被再次邀请  ✓ 永远得不到再邀请\n"
        "  'That's strange to me'（幽默语境）✓ 这太奇怪了\n"
        "  'That's strange to me'（论证语境）✓ 这与我的理解不符\n"
        "  'This is what I believe in' ✗ 这就是我一直相信和坚持的东西  ✓ 这是我的信念\n"
        "  'I'm leaving'（讽刺语境）  ✗ 我要离开了  ✓ 我都想溜走了\n\n"

        f"{ctx_block}"
        "【待翻译字幕】\n"
        f"{numbered}\n\n"

        "【输出格式】\n"
        "每行格式：「编号. 译文」\n"
        "条数必须与输入完全一致，不添加任何解释或备注。"
    )
    resp = httpx.post(
        DEEPSEEK_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model":       "deepseek-chat",
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens":  4096,
        },
        timeout=120,
    )
    resp.raise_for_status()
    output = resp.json()["choices"][0]["message"]["content"].strip()
    return _parse_numbered_output(output, texts)


def _polish_batch(
    en_caps: list[Caption],
    zh_texts: list[str],
    api_key: str,
    prev_en_context: list[str] | None = None,
) -> list[str]:
    """
    Polish one batch of existing Chinese subtitles against their English source.
    Follows subtitle-polish-chinese v2.1 constraint system.
    prev_en_context: last few English texts from the previous batch (for continuity).
    """
    ctx_block = ""
    if prev_en_context:
        ctx_block = (
            "【前文参考（仅供上下文理解，无需处理）】\n"
            + "\n".join(f"  {t}" for t in prev_en_context)
            + "\n\n"
        )

    numbered = "\n".join(
        f"{i + 1}. [EN] {en.text}\n   [ZH] {zh}"
        for i, (en, zh) in enumerate(zip(en_caps, zh_texts))
    )

    prompt = (
        "你是专业的TED演讲字幕润色专家，执行 subtitle-polish-chinese v2.1 标准。\n"
        "以下是英文字幕（[EN]）及现有中文翻译（[ZH]），请对中文进行润色优化。\n\n"

        "【约束优先级（严格遵守）】\n"
        "① 字数上限（不可妥协）\n"
        "   · 每条：每行 ≤ 30 汉字，最多 2 行，优先 1 行\n"
        "   · 字数比例：中文字数 ÷ 英文单词数 ≤ 1.5（理想约 1:1）\n"
        "   · 若现有中文超限，主动简化，保留核心意思\n"
        "② 上下文修正（核心任务）\n"
        "   · 先通读全批内容，把握演讲整体语境和情感弧线\n"
        "   · 若直译生硬 → 改意译；若冗长 → 主动精简\n"
        "   · 注意前后字幕的情感连贯性与逻辑递进\n"
        "   · 匹配演讲者语气（幽默/严肃/感情/论证）\n"
        "③ 简洁优先\n"
        "   · 删除冗余修饰词，避免在单条字幕中叠加过多修饰语\n"
        "④ 视觉平衡\n"
        "   · 中文视觉长度与英文大致匹配，不出现英文 1 行、中文 3 行的情况\n"
        "⑤ 语言自然\n"
        "   · 口语化、地道中文，符合演讲语境\n\n"

        "【典型修复示例】\n"
        "  原：教育对未来发展至关重要，我们必须重视它    →  改：教育对未来很重要\n"
        "  原：我已经被这一切彻底地迷住了              →  改：我被彻底吸引了\n"
        "  原：与整个教育体系格格不入，没有任何融入的空间 →  改：和体系合不来\n"
        "  原：所有孩子都拥有巨大的才能，而我们正在无情地浪费这些才能\n"
        "       →  改：孩子都有才能，我们在浪费（或拆成两条字幕）\n\n"

        f"{ctx_block}"
        "【待润色字幕】\n"
        f"{numbered}\n\n"

        "【输出格式】\n"
        "每行格式：「编号. 润色后的中文」（只输出中文，不含英文）\n"
        "条数必须与输入完全一致，不添加任何解释或备注。"
    )
    resp = httpx.post(
        DEEPSEEK_API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model":       "deepseek-chat",
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens":  4096,
        },
        timeout=120,
    )
    resp.raise_for_status()
    output = resp.json()["choices"][0]["message"]["content"].strip()
    return _parse_numbered_output(output, zh_texts)


_CTX_LOOKBACK = 3   # 传给下一批的前文参考条数


def translate_with_deepseek(captions: list[Caption], api_key: str) -> list[str]:
    """
    Translate English captions to polished Simplified Chinese, batched.
    Passes the last _CTX_LOOKBACK English lines of each batch as context
    to the next batch, so the model maintains narrative continuity.
    """
    all_zh: list[str] = []
    total = len(captions)
    prev_context: list[str] = []

    for start in range(0, total, TRANSLATION_BATCH):
        batch = captions[start : start + TRANSLATION_BATCH]
        end   = min(start + TRANSLATION_BATCH, total)
        print(f"  Translating+polishing {start + 1}–{end} / {total} …", end=" ", flush=True)
        try:
            zh_batch = _translate_batch(
                [c.text for c in batch],
                api_key,
                prev_context=prev_context or None,
            )
            print("done")
        except httpx.HTTPStatusError as exc:
            print(f"API error {exc.response.status_code}: {exc.response.text[:120]}")
            zh_batch = [c.text for c in batch]
        except httpx.RequestError as exc:
            print(f"network error: {exc}")
            zh_batch = [c.text for c in batch]

        all_zh.extend(zh_batch)
        # 保留本批最后几条英文作为下一批前文参考
        prev_context = [c.text for c in batch[-_CTX_LOOKBACK:]]
        if end < total:
            time.sleep(0.5)

    return all_zh


def polish_bilingual_with_deepseek(
    en_captions: list[Caption], zh_texts: list[str], api_key: str
) -> list[str]:
    """
    Polish existing Chinese subtitles against their English source, batched.
    Passes the last _CTX_LOOKBACK English lines of each batch as context
    to the next batch for narrative continuity.
    """
    all_polished: list[str] = []
    total = len(en_captions)
    prev_en_context: list[str] = []

    for start in range(0, total, TRANSLATION_BATCH):
        en_batch = en_captions[start : start + TRANSLATION_BATCH]
        zh_batch = zh_texts[start : start + TRANSLATION_BATCH]
        end      = min(start + TRANSLATION_BATCH, total)
        print(f"  Polishing {start + 1}–{end} / {total} …", end=" ", flush=True)
        try:
            polished = _polish_batch(
                en_batch, zh_batch, api_key,
                prev_en_context=prev_en_context or None,
            )
            print("done")
        except httpx.HTTPStatusError as exc:
            print(f"API error {exc.response.status_code}: {exc.response.text[:120]}")
            polished = zh_batch
        except httpx.RequestError as exc:
            print(f"network error: {exc}")
            polished = zh_batch

        all_polished.extend(polished)
        prev_en_context = [c.text for c in en_batch[-_CTX_LOOKBACK:]]
        if end < total:
            time.sleep(0.5)

    return all_polished


# ── SRT writer ────────────────────────────────────────────────────────────────

def write_bilingual_srt(
    captions: list[Caption],
    zh_texts: list[str],
    path: Path,
) -> None:
    """
    Write a bilingual SRT file:
        English line
        中文行
    """
    lines: list[str] = []
    for i, (cap, zh) in enumerate(zip(captions, zh_texts), 1):
        lines.append(str(i))
        lines.append(f"{ms_to_srt_time(cap.start_ms)} --> {ms_to_srt_time(cap.end_ms)}")
        lines.append(cap.text)
        if zh:
            lines.append(zh)
        lines.append("")
    # utf-8-sig adds BOM so Windows subtitle players recognise the encoding
    path.write_text("\n".join(lines), encoding="utf-8-sig")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Download TED talk subtitles and create a bilingual English+Chinese SRT file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ted_subtitle_extractor.py https://www.ted.com/talks/ken_robinson_says_schools_kill_creativity\n"
            "  python ted_subtitle_extractor.py <url> --api-key sk-xxx --output-dir ./out\n"
        ),
    )
    parser.add_argument("url", help="TED talk URL")
    parser.add_argument(
        "--api-key",
        metavar="KEY",
        help="DeepSeek API key (needed only when Chinese subtitles are unavailable on TED)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default="subtitles",
        help="Directory for output files (default: ./subtitles)",
    )
    args = parser.parse_args()

    api_key    = args.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print("TED Subtitle Extractor")
    print("=" * 52)
    print(f"URL : {args.url}")

    try:
        slug = parse_slug(args.url)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Slug: {slug}")
    print()

    fetcher = TedSubtitleFetcher(args.url, output_dir)

    # ── Step 1: discover available languages ──────────────────────────────────
    print("[1/3] Checking available subtitle languages …", end=" ", flush=True)
    try:
        available = fetcher.available_langs()
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"{len(available)} found")

    # Choose which languages to download — simplified Chinese only
    zh_lang: Optional[str] = None
    for code in ZH_LANG_PRIORITY:
        if code in available:
            zh_lang = code
            break

    # Detect traditional Chinese (will be skipped in favour of AI translation)
    zh_trad: Optional[str] = next((c for c in ZH_TRADITIONAL if c in available), None)

    langs_to_fetch = ["en"] + ([zh_lang] if zh_lang else [])
    print(f"  English : {'en' if 'en' in available else 'NOT available'}")
    if zh_lang:
        print(f"  Chinese : {zh_lang} (simplified — using official subtitles)")
    elif zh_trad:
        print(f"  Chinese : {zh_trad} (traditional — skipping, will AI-translate to simplified)")
    else:
        print("  Chinese : none")

    if "en" not in available:
        print("Error: English subtitles not available for this talk.", file=sys.stderr)
        sys.exit(1)

    # ── Step 2: download subtitle files ──────────────────────────────────────
    print("\n[2/3] Downloading subtitle VTT files …")
    try:
        subs = fetcher.download(langs_to_fetch)
    except Exception as exc:
        print(f"Error downloading subtitles: {exc}", file=sys.stderr)
        sys.exit(1)

    en_captions: Optional[list[Caption]] = subs.get("en")
    if not en_captions:
        print("Error: Failed to parse English subtitles.", file=sys.stderr)
        sys.exit(1)
    print(f"  English : {len(en_captions)} captions")

    zh_captions: Optional[list[Caption]] = subs.get(zh_lang) if zh_lang else None
    if zh_captions:
        print(f"  Chinese : {len(zh_captions)} captions  ({zh_lang})")
    else:
        print("  Chinese : not downloaded / empty")

    # ── Step 3: produce Chinese texts ────────────────────────────────────────
    print()
    if zh_captions:
        print("[3/3] Merging English + Chinese subtitles …", end=" ", flush=True)
        zh_texts = align_zh_to_en(en_captions, zh_captions)
        zh_source = f"TED official ({zh_lang})"
        print("done")
    else:
        print("[3/3] No Chinese subtitles on TED — translating via DeepSeek …")
        if not api_key:
            print(
                "\nError: Chinese subtitles unavailable and no DeepSeek API key provided.\n"
                "  Set the DEEPSEEK_API_KEY environment variable or pass --api-key KEY.",
                file=sys.stderr,
            )
            sys.exit(1)
        zh_texts  = translate_with_deepseek(en_captions, api_key)
        zh_source = "DeepSeek translation"

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = output_dir / f"{slug}_bilingual.srt"
    write_bilingual_srt(en_captions, zh_texts, out_path)

    print()
    print("=" * 52)
    print("Done!")
    print(f"  Captions : {len(en_captions)}")
    print(f"  Chinese  : {zh_source}")
    print(f"  Output   : {out_path.resolve()}")
    print()


if __name__ == "__main__":
    main()
