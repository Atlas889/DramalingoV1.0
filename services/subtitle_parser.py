"""
DramaLingo · 字幕解析模块
对应 PRD v0.9 §5 流程一 / 开发文档 Step 2

支持格式：.srt / .ass / .ssa
- 自动检测编码（chardet：UTF-8 / GBK / GB2312）
- 去除 ASS 特效标签（{...} 正则清除）
- 单文件中英双语提取：每个字幕块内自动识别英文行和中文行
  · 英文行 → en_text（含英文字母为主）
  · 中文行 → zh_text（含中文字符为主）
  · 无需跨文件对齐，无需 AI 翻译
- 返回 List[SubtitleLine] dataclass
"""

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import chardet

logger = logging.getLogger(__name__)

# ASS 特效标签正则（{...} 及 \N 换行符）
_ASS_TAG_RE = re.compile(r"\{[^}]*\}|\\N|\\n", re.IGNORECASE)
# HTML 标签
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# 多余空白
_WHITESPACE_RE = re.compile(r"\s+")
# 中文字符检测（仅汉字区块，不含全角标点，避免将含全角符号的英文行误判为中文行）
_ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
# 英文字母检测
_EN_CHAR_RE = re.compile(r"[a-zA-Z]")
# 纯数字/特殊字符行（无实质内容）
_JUNK_LINE_RE = re.compile(r"^[\d\s\W]+$")
# 句尾终止标点：单个 . 或 ! ? ; （多个点 = 省略号，视为续行，不列入此处）
_TERMINAL_PUNCT_RE = re.compile(r'[!?;][\'")\]]*\s*$|(?<!\.)\.(?!\.)[\'")\]]*\s*$')
# 省略号结尾（.. / ... / …）= 句子未完，视为续行信号
_ELLIPSIS_RE = re.compile(r'\.{2,}[\'")\]]*\s*$|…[\'")\]]*\s*$')
# 行尾续行词：连词 / 介词 / 冠词 / 不定式助动词，表示语法上句子尚未结束
_CONTINUATION_END_RE = re.compile(
    r'\b(?:'
    r'and|but|or|so|yet|nor|'
    r'because|although|though|while|whereas|whether|'
    r'that|which|who|whom|when|where|'
    r'if|unless|until|since|after|before|as|than|like|'
    r'in|on|at|to|for|of|with|about|over|under|'
    r'into|from|by|through|between|among|during|'
    r'without|within|around|'
    r'a|an|the'
    r')\s*$',
    re.IGNORECASE,
)
# 对话破折号开头（换人说话，不应合并）：- word 或 -word
_DIALOGUE_DASH_RE = re.compile(r'^\s*-\s*\w')
_MERGE_GAP_SEC: float = 1.00        # 相邻字幕最大间隔（秒）
_MERGE_MAX_WORDS: int = 50          # 合并后最大英文词数
SIM_MERGE_THRESHOLD: float = 0.68   # Layer 2：向量相似度触发合并的下限


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度，返回 [0, 1]。"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _is_continuation(
    en_a: str,
    en_b: str,
    emb_a: Optional[np.ndarray] = None,
    emb_b: Optional[np.ndarray] = None,
) -> bool:
    """
    综合语言学信号 + 向量相似度，判断行 A 是否是行 B 的前半句。

    Layer 1（语言学规则）：
      1. 省略号结尾            → 续行（直接触发）
      2. 终止标点结尾          → 非续行（直接拒绝）
      3. 行 B 首字母小写       → 强续行信号
      4. 行 A 尾词为连词/介词/冠词 → 语法上句子未结束

    Layer 2（向量相似度，仅当 Layer 1 无法判定时启用）：
      5. cosine_sim(emb_a, emb_b) ≥ SIM_MERGE_THRESHOLD → 语义高度连贯，视为续行
    """
    if not en_a or not en_b:
        return False

    # ── Layer 1 ──
    if _ELLIPSIS_RE.search(en_a):
        return True

    if _TERMINAL_PUNCT_RE.search(en_a):
        return False

    first_alpha = next((c for c in en_b if c.isalpha()), None)
    if first_alpha and first_alpha.islower():
        return True

    if _CONTINUATION_END_RE.search(en_a):
        return True

    # ── Layer 2：向量相似度兜底 ──
    if emb_a is not None and emb_b is not None:
        if _cosine_sim(emb_a, emb_b) >= SIM_MERGE_THRESHOLD:
            return True

    return False


def merge_subtitle_lines(lines: "List[SubtitleLine]") -> "List[SubtitleLine]":
    """
    将时间相邻、语义未完整的字幕行合并为一行。

    合并条件（全部满足才合并）：
      1. 相邻两行间隔 ≤ _MERGE_GAP_SEC（默认 0.30s）
      2. 前一行英文末尾无终止标点（说明表达未结束）
      3. 后一行英文不以对话破折号「- 」开头（换人说话不合并）
      4. 合并后英文总词数 ≤ _MERGE_MAX_WORDS
    """
    if not lines:
        return lines

    merged: "List[SubtitleLine]" = []
    current = lines[0]

    for nxt in lines[1:]:
        gap = nxt.start_time - current.end_time
        no_dash = not _DIALOGUE_DASH_RE.match(nxt.en_text)
        combined_words = len((current.en_text + " " + nxt.en_text).split())

        if gap <= _MERGE_GAP_SEC and no_dash and combined_words <= _MERGE_MAX_WORDS \
                and _is_continuation(current.en_text, nxt.en_text):
            base_en = _ELLIPSIS_RE.sub("", current.en_text).rstrip() \
                if _ELLIPSIS_RE.search(current.en_text) else current.en_text
            current = SubtitleLine(
                start_time=current.start_time,
                end_time=nxt.end_time,
                en_text=(base_en + " " + nxt.en_text).strip(),
                zh_text=(current.zh_text + " " + nxt.zh_text).strip() if (current.zh_text or nxt.zh_text) else "",
            )
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    return merged


@dataclass
class SubtitleLine:
    """单条字幕行数据结构"""
    start_time: float       # 秒（精度到毫秒）
    end_time: float         # 秒
    en_text: str            # 英文文本
    zh_text: str = ""       # 中文文本（从双语字幕中直接提取）
    word_count: int = 0     # 英文词数（解析后自动计算）

    def __post_init__(self):
        if self.word_count == 0 and self.en_text:
            self.word_count = len(self.en_text.split())


# ──────────────────────────────────────────────
# 语言识别工具
# ──────────────────────────────────────────────
def _is_chinese_line(text: str) -> bool:
    """判断一行文本是否以中文为主（中文字符占比 > 20% 或包含中文字符且英文字母极少）"""
    if not text:
        return False
    zh_count = len(_ZH_CHAR_RE.findall(text))
    en_count = len(_EN_CHAR_RE.findall(text))
    total = len(text.replace(" ", ""))
    if total == 0:
        return False
    # 中文字符占比超过 20%，或中文字符数 > 英文字母数
    return zh_count > 0 and (zh_count / total > 0.2 or zh_count >= en_count)


def _is_english_line(text: str) -> bool:
    """判断一行文本是否以英文为主"""
    if not text:
        return False
    zh_count = len(_ZH_CHAR_RE.findall(text))
    en_count = len(_EN_CHAR_RE.findall(text))
    # 有英文字母且中文字符极少
    return en_count > 0 and zh_count == 0


def _classify_lines(text_lines: List[str]) -> Tuple[str, str]:
    """
    将字幕块内的多行文本分类为英文和中文。
    返回 (en_text, zh_text)，均可能为空字符串。
    """
    en_parts = []
    zh_parts = []

    for line in text_lines:
        line = line.strip()
        if not line:
            continue
        if _is_chinese_line(line):
            zh_parts.append(line)
        elif _is_english_line(line):
            en_parts.append(line)
        else:
            # 混合行（如 "- Hello 你好"）：尝试拆分
            # 先尝试按常见分隔符拆分
            sub_lines = re.split(r'[\|｜]', line)
            if len(sub_lines) > 1:
                for sl in sub_lines:
                    sl = sl.strip()
                    if _is_chinese_line(sl):
                        zh_parts.append(sl)
                    elif _is_english_line(sl):
                        en_parts.append(sl)
            else:
                # 无法拆分：按字符比例归类
                if _is_chinese_line(line):
                    zh_parts.append(line)
                else:
                    en_parts.append(line)

    en_text = " ".join(en_parts).strip()
    zh_text = " ".join(zh_parts).strip()
    return en_text, zh_text


# ──────────────────────────────────────────────
# 编码检测
# ──────────────────────────────────────────────
def _detect_encoding(file_path: str) -> str:
    """使用 chardet 检测文件编码，失败时回退到 utf-8"""
    with open(file_path, "rb") as f:
        raw = f.read(min(65536, Path(file_path).stat().st_size))
    # 检测 UTF-8 BOM
    if raw.startswith(b'\xef\xbb\xbf'):
        return "utf-8-sig"
    result = chardet.detect(raw)
    encoding = result.get("encoding") or "utf-8"
    encoding = encoding.lower().replace("-", "")
    if encoding in ("gb2312", "gbk", "gb18030"):
        return "gb18030"
    if encoding in ("utf8sig", "utf8bom"):
        return "utf-8-sig"
    return encoding


def _read_text(file_path: str) -> str:
    """读取文本文件，自动检测编码"""
    encoding = _detect_encoding(file_path)
    try:
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            return f.read()
    except (LookupError, UnicodeDecodeError):
        logger.warning(f"编码 {encoding} 读取失败，回退到 utf-8: {file_path}")
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


# ──────────────────────────────────────────────
# SRT 解析（支持单文件中英双语）
# ──────────────────────────────────────────────
def _timecode_to_seconds(tc: str) -> float:
    """将 SRT 时间码 HH:MM:SS,mmm 转换为秒，兼容 HH:MM:SS.mmm"""
    tc = tc.replace(",", ".").strip()
    parts = tc.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(tc)


_SRT_TIMECODE_RE = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})"
)


def _parse_srt(file_path: str) -> List[SubtitleLine]:
    """
    解析 .srt 字幕文件，支持单文件中英双语格式。
    每个字幕块内自动识别英文行和中文行，分别提取到 en_text 和 zh_text。
    """
    text = _read_text(file_path)
    # 统一换行符
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    # 按空行分割字幕块
    blocks = re.split(r"\n\s*\n", text.strip())

    for block in blocks:
        block_lines = [l.strip() for l in block.strip().splitlines()]
        if len(block_lines) < 2:
            continue

        # 找时间轴行
        tc_idx = None
        start_time = end_time = None
        for i, bl in enumerate(block_lines):
            m = _SRT_TIMECODE_RE.match(bl)
            if m:
                tc_idx = i
                try:
                    start_time = _timecode_to_seconds(m.group(1))
                    end_time = _timecode_to_seconds(m.group(2))
                except Exception as e:
                    logger.warning("SRT 时间轴解析失败，已跳过该块: %s | 原因: %s", bl, e)
                break

        if tc_idx is None or start_time is None:
            continue

        # 时间轴之后的所有行为文本行
        text_lines = block_lines[tc_idx + 1:]
        # 清理 HTML 标签
        text_lines = [_HTML_TAG_RE.sub("", l) for l in text_lines]
        # 清理多余空白
        text_lines = [_WHITESPACE_RE.sub(" ", l).strip() for l in text_lines]
        text_lines = [l for l in text_lines if l]

        if not text_lines:
            continue

        # 分类英文/中文
        en_text, zh_text = _classify_lines(text_lines)

        # 必须有英文内容才入库
        if not en_text:
            continue

        lines.append(SubtitleLine(
            start_time=start_time,
            end_time=end_time,
            en_text=en_text,
            zh_text=zh_text,
        ))

    logger.info(f"SRT 解析完成: {Path(file_path).name}，共 {len(lines)} 行（含中文: {sum(1 for l in lines if l.zh_text)} 行）")
    return lines


# ──────────────────────────────────────────────
# ASS / SSA 解析（支持单文件中英双语）
# ──────────────────────────────────────────────
def _parse_ass(file_path: str) -> List[SubtitleLine]:
    """
    解析 .ass / .ssa 字幕文件，支持单文件中英双语格式。
    ASS 双语通常通过多个 Style 轨道实现，按时间戳合并同时刻的英中行。
    """
    try:
        import ass as ass_lib
        # 使用 _read_text 自动处理 BOM 和编码，再传给 ass 库
        text_content = _read_text(file_path)
        # ass 库要求纯文本，去除 BOM 字符
        text_content = text_content.lstrip('\ufeff')
        doc = ass_lib.parse_string(text_content)

        # 收集所有 Dialogue 行
        raw_lines = []
        for event in doc.events:
            if not (hasattr(event, "text") and hasattr(event, "start") and hasattr(event, "end")):
                continue
            text = event.text
            # 先按 \N / \n 拆行（在清除标签之前！）
            # ASS 双语格式：同一 Dialogue 行内用 \N 分隔中文和英文
            sub_parts = re.split(r'\\[Nn]', text)
            start = event.start.total_seconds()
            end = event.end.total_seconds()
            for part in sub_parts:
                # 清除 ASS 标签（{...}）
                part_clean = re.sub(r'\{[^}]*\}', ' ', part)
                part_clean = _WHITESPACE_RE.sub(' ', part_clean).strip()
                if part_clean:
                    raw_lines.append((start, end, part_clean))

        # 按时间戳分组（容差 0.1s），同组内合并英中
        # raw_lines 已排序，线性扫描与相邻组比较，O(n) 复杂度
        raw_lines.sort(key=lambda x: x[0])
        groups = []
        for start, end, text in raw_lines:
            if groups and abs(groups[-1]["start"] - start) < 0.1:
                groups[-1]["texts"].append(text)
                groups[-1]["end"] = max(groups[-1]["end"], end)
            else:
                groups.append({"start": start, "end": end, "texts": [text]})

        lines = []
        for g in groups:
            en_text, zh_text = _classify_lines(g["texts"])
            if not en_text:
                continue
            lines.append(SubtitleLine(
                start_time=g["start"],
                end_time=g["end"],
                en_text=en_text,
                zh_text=zh_text,
            ))

        logger.info(f"ASS 解析完成: {Path(file_path).name}，共 {len(lines)} 行")
        return lines
    except Exception as e:
        logger.error(f"ASS 解析失败 {file_path}: {e}")
        return []


# ──────────────────────────────────────────────
# 公共入口
# ──────────────────────────────────────────────
def parse_subtitle(file_path: str, return_stats: bool = False):
    """
    解析单个字幕文件，返回 SubtitleLine 列表。
    支持 .srt / .ass / .ssa 格式，自动检测编码。
    自动识别单文件中英双语格式，将英文提取到 en_text，中文提取到 zh_text。

    Args:
        file_path: 字幕文件路径
        return_stats: 为 True 时返回 (lines, {"merged": N}) 二元组

    Returns:
        List[SubtitleLine]，按时间戳升序排列；return_stats=True 时为 (list, dict)

    Raises:
        ValueError: 不支持的文件格式
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".srt":
        lines = _parse_srt(file_path)
    elif suffix in (".ass", ".ssa"):
        lines = _parse_ass(file_path)
    else:
        raise ValueError(f"不支持的字幕格式: {suffix}（仅支持 .srt / .ass / .ssa）")

    # 按时间戳升序排列
    lines.sort(key=lambda x: x.start_time)

    # 合并被拆分的不完整字幕行
    before_merge = len(lines)
    lines = merge_subtitle_lines(lines)
    merged_count = before_merge - len(lines)
    if merged_count > 0:
        logger.info(f"字幕合并: {before_merge} → {len(lines)} 行（合并 {merged_count} 行）")

    logger.info(f"字幕解析完成: {path.name}，共 {len(lines)} 行")
    if return_stats:
        return lines, {"merged": merged_count}
    return lines


def _parse_srt_text_only(file_path: str) -> List[Tuple[float, float, str]]:
    """
    轻量级 SRT 解析器，不要求英文内容。
    返回 List[(start_time, end_time, text)] 供双文件对齐使用。
    专为纯中文（或任意单语）字幕文件设计。
    """
    text = _read_text(file_path)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    result: List[Tuple[float, float, str]] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        block_lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        if len(block_lines) < 2:
            continue
        tc_idx = None
        start_time = end_time = None
        for i, bl in enumerate(block_lines):
            m = _SRT_TIMECODE_RE.match(bl)
            if m:
                tc_idx = i
                try:
                    start_time = _timecode_to_seconds(m.group(1))
                    end_time = _timecode_to_seconds(m.group(2))
                except Exception:
                    break
                break
        if tc_idx is None or start_time is None:
            continue
        text_lines = [_HTML_TAG_RE.sub("", l) for l in block_lines[tc_idx + 1:]]
        text_lines = [_WHITESPACE_RE.sub(" ", l).strip() for l in text_lines if l.strip()]
        content = " ".join(text_lines).strip()
        if content:
            result.append((start_time, end_time, content))
    logger.debug(f"_parse_srt_text_only: {Path(file_path).name} → {len(result)} 行")
    return result


def parse_bilingual_subtitles(
    en_file: str,
    zh_file: Optional[str] = None,
    tolerance: float = 0.5,
) -> List[SubtitleLine]:
    """
    解析双语字幕。
    模式1（单文件双语）：若字幕文件本身包含中英双语，直接调用 parse_subtitle() 即可。
    模式2（双文件对齐）：若提供独立的中文字幕文件，按时间戳对齐合并。
    Args:
        en_file: 英文字幕文件路径（可为单文件双语）
        zh_file: 独立中文字幕文件路径（可选）
        tolerance: 时间戳对齐容差（秒），默认 0.5s
    Returns:
        List[SubtitleLine]，中文已填入 zh_text 字段
    """
    en_lines = parse_subtitle(en_file)
    if not zh_file:
        return en_lines

    # 模式2：双文件时间戳对齐
    # parse_subtitle() 要求每个字幕块含英文内容，纯中文 SRT 会返回 0 行。
    # 因此直接用专用的纯文本 SRT 时间轴解析器提取中文内容。
    zh_lookup = _parse_srt_text_only(zh_file)
    if not zh_lookup:
        # 降级：尝试普通解析器（双语中文文件等场景）
        zh_lines_raw = parse_subtitle(zh_file)
        for zl in zh_lines_raw:
            zh_content = zl.zh_text or zl.en_text
            if zh_content:
                zh_lookup.append((zl.start_time, zl.end_time, zh_content))

    zh_lookup.sort(key=lambda x: x[0])
    zh_count = len(zh_lookup)

    matched = 0
    for en_line in en_lines:
        if en_line.zh_text:  # 已有中文（单文件双语已提取），跳过
            continue
        best_zh = None
        best_diff = float("inf")
        for start, end, zh_content in zh_lookup:
            diff = abs(start - en_line.start_time)
            if diff <= tolerance and diff < best_diff:
                best_diff = diff
                best_zh = zh_content
            if start > en_line.start_time + tolerance:
                break
        if best_zh:
            en_line.zh_text = best_zh
            matched += 1

    logger.info(
        f"双文件对齐完成: {len(en_lines)} 行英文 + {zh_count} 行中文 "
        f"→ 匹配 {matched} 行（容差 {tolerance}s）"
    )
    return en_lines
