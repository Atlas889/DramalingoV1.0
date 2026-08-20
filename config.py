"""
DramaLingo · 全局配置模块
所有可调参数集中管理，涉及密钥的字段从环境变量读取。
对应 PRD v0.9 §6.1 / §6.6 / §8.1 的参数定义。
"""

import os
from pathlib import Path

# 自动加载 .env 文件（开发环境）
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
except ImportError:
    pass  # 生产环境可不安装 python-dotenv

# ──────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# 文件存储根目录：优先读取环境变量，默认归档到项目 data/ 目录。
# 生产环境建议通过 DRAMALINGO_FILES_ROOT 指向独立磁盘，避免媒体文件占用代码仓库空间。
_FILES_ROOT = Path(os.getenv("DRAMALINGO_FILES_ROOT", str(BASE_DIR / "data")))

MEDIA_DIR  = _FILES_ROOT / "media"         # 视频文件 & 切片文件
UPLOAD_DIR = _FILES_ROOT / "uploads"       # 字幕/视频上传临时目录
INDEX_DIR  = _FILES_ROOT / "indexes"       # FAISS 索引文件

STATIC_DIR = BASE_DIR / "static"           # 前端静态文件
DB_PATH    = BASE_DIR / "data" / "dramalingo.db"  # SQLite 数据库与运行数据分离

# ──────────────────────────────────────────────
# 字幕筛选参数（PRD §6.1）
# ──────────────────────────────────────────────
MIN_WORDS: int = 2                          # 最少词数
MAX_WORDS: int = 100                        # 最多词数（对齐切片约束）
QUALITY_THRESHOLD: float = 0.30             # 口语真实性评分下限（基准测试最优值，F1=97.5%）
CONTEXT_QUALITY_THRESHOLD: float = float(os.getenv("CONTEXT_QUALITY_THRESHOLD", "0.30"))
DEDUP_SIMILARITY: float = 0.95             # 语义去重精筛阈值（embedding 余弦相似度）
DEDUP_EDIT_DISTANCE_THRESHOLD: float = 0.15 # 粗筛归一化编辑距离阈值（越小越严格）
BLACKLIST_VERSION: str = "v2"               # 黑名单词库版本

# ──────────────────────────────────────────────
# FAISS 参数（PRD §6.6 评审修正：召回 Top-50）
# ──────────────────────────────────────────────
FAISS_RECALL_K: int = 50                    # 语义检索召回缓冲数（不是 20）
FAISS_DEDUP_TOP_K: int = 5                  # 去重精筛候选数
FAISS_REBUILD_THRESHOLD: float = 0.05       # 增量 vs 全量重建阈值（新增占比）
RANK_MIN_SIMILARITY: float = 0.50           # 语义检索最低余弦相似度绝对门槛，低于此值直接丢弃
RANK_RELATIVE_THRESHOLD: float = 0.06      # 相对截断：只保留与最高分差距在此范围内的结果（减少长尾噪音）

# ──────────────────────────────────────────────
# Whisper 参数
# ──────────────────────────────────────────────
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "small")  # tiny/base/small/medium，small 识别质量更好
WHISPER_ALIGN_WINDOW_SECONDS: float = 30.0  # Step 2a 对齐滑动窗口（秒）；合并后字幕时间戳精度 ±5s，30s 足够

# Whisper 转写段合并参数（将零散片段拼成完整句子）
WHISPER_MERGE_GAP_SECONDS: float = 1.5     # 相邻片段最大间隔（秒），超过此值不合并
WHISPER_MERGE_HARD_GAP: float = 1.5        # 硬边界 gap：超过此值视为场景/说话人切换，无论标点都不合并
WHISPER_MERGE_MAX_DURATION: float = 12.0   # 合并后单段最大时长（秒），防止过度合并
WHISPER_MERGE_MAX_WORDS: int = 40          # 合并后最大词数，防止跨说话人过度合并

# ──────────────────────────────────────────────
# 序列对齐参数（aligner.align_seq）
# ──────────────────────────────────────────────
# 综合评分：lev × LEV_WEIGHT + emb_cos × EMB_WEIGHT + order_bonus(or penalty)
ALIGN_USE_EMBEDDING: bool = True           # 启用 embedding 余弦作为第二信号（关闭则退回纯 Levenshtein）
ALIGN_LEV_WEIGHT: float = 0.55             # Levenshtein 字面相似度权重
ALIGN_EMB_WEIGHT: float = 0.35             # Embedding 余弦语义相似度权重
ALIGN_ORDER_BONUS: float = 0.10            # 保序加分（候选 seq 在游标之后）
ALIGN_ORDER_PENALTY: float = 0.15          # 乱序惩罚（候选 seq 在游标之前）
ALIGN_MIN_LEN_RATIO: float = 0.4           # 长度比例阈值：min/max < 此值的候选直接淘汰
ALIGN_AMBIGUOUS_MARGIN: float = 0.08       # best - second_best 小于此值视为模糊匹配（标记 ambiguous）
ALIGN_SHORT_TEXT_WORDS: int = 3            # 词数 ≤ 此值的节点视为短文本，要求高相似度
ALIGN_SHORT_TEXT_MIN_SIM: float = 0.9      # 短文本节点的最低相似度门槛（避免 "Hi"/"Yeah" 噪声匹配）

# ──────────────────────────────────────────────
# 视频切片参数（PRD §5 流程二·切片门禁）
# ──────────────────────────────────────────────
CLIP_MAX_DURATION: float = 15.0             # 单个切片最大时长（秒，含 padding）
CLIP_MIN_DURATION: float = 1.0              # 单个切片最小时长（秒，含 padding）
CLIP_PADDING_START: float = 0.5            # 切片起点前置缓冲（秒）
CLIP_PADDING_END: float = 1.0              # 切片终点后置缓冲（秒）
CLIP_CONTEXT_BEFORE: int = 1              # 切片前置上下文段数（相邻 Whisper 段）
CLIP_CONTEXT_AFTER: int = 1               # 切片后置上下文段数
CLIP_CONTEXT_MAX_EXTRA_S: float = 6.0    # 单方向上下文最大额外时长（秒）；超出则不追加该方向的上下文
CLIP_CONTEXT_SCENE_GAP: float = 1.5      # 相邻段静音 > 此值视为场景断点，不再纳入上下文
CLIP_CONTEXT_NEIGHBOR_MAX_DUR: float = 3.0  # 邻段时长上限：超过此值的邻段视为独立长句，不追加
CLIP_SNAP_TO_BOUNDARY_MAX: float = 1.5   # 切片端点 snap-to-segment-boundary 最大调整量（秒）
CLIP_SNAP_TO_BOUNDARY_MIN: float = 0.3   # snap 最小调整量；小于此值认为已贴近边界，无需调整
FFMPEG_CONCURRENT: int = int(os.getenv("FFMPEG_CONCURRENT", "4"))  # 并发 FFmpeg 进程数
CLIP_MIN_QUALITY: float = float(os.getenv("CLIP_MIN_QUALITY", "0.6"))  # C3: 低于此分的节点不切片

# ──────────────────────────────────────────────
# AI 打标参数（PRD §6.1）
# ──────────────────────────────────────────────
AI_PROVIDER: str = os.getenv("AI_PROVIDER", "deepseek")  # deepseek / qwen
AI_API_KEY: str = os.getenv("AI_API_KEY", "")
AI_API_BASE_URL: str = os.getenv("AI_API_BASE_URL", "https://api.deepseek.com/v1")
# 第二 AI 提供商（可选）：同时启用两个 provider 可将打标吞吐量接近翻倍
AI_PROVIDER_2: str = os.getenv("AI_PROVIDER_2", "")      # "deepseek" / "qwen"，留空则不启用
AI_API_KEY_2: str = os.getenv("AI_API_KEY_2", "")
AI_API_BASE_URL_2: str = os.getenv("AI_API_BASE_URL_2", "")
AI_TIMEOUT: int = 30                        # API 调用超时（秒）
QUERY_EXPAND_ENABLED: bool = os.getenv("QUERY_EXPAND_ENABLED", "true").lower() != "false"
# 启用查询意图扩展：DeepSeek 将用户输入补全为更丰富的中文描述 + 3条候选英文表达
# 禁用后退回原始翻译模式（`translate_to_english`）
AI_MAX_RETRIES: int = 3                     # API 调用重试次数

# ──────────────────────────────────────────────
# 多信号检索权重（综合排序公式）
# final = zh_sim×0.40 + en_tr_sim×0.25 + en_sim×0.15 + tag×0.12 + quality×0.06 + occ×0.02
# zh_sim:    中文查询 → zh_embedding（中文翻译向量）相似度
# en_tr_sim: 中文查询翻译为英文后 → en_embedding（英文字幕向量）相似度
# en_sim:    中文查询跨语言 → en_embedding 相似度（baseline）
# tag:       场景/情绪标签关键词匹配度
# ──────────────────────────────────────────────
RANK_WEIGHT_ZH_SIM: float = 0.40
RANK_WEIGHT_EN_TRANSLATED: float = 0.25
RANK_WEIGHT_EN_SIM: float = 0.15
RANK_WEIGHT_TAG: float = 0.12
RANK_WEIGHT_QUALITY: float = 0.06
RANK_WEIGHT_OCCURRENCE: float = 0.02
RANK_OCCURRENCE_SATURATION: int = 50        # 出现频次对数归一化饱和值

# ──────────────────────────────────────────────
# SQLite 并发策略（PRD §8.1）
# ──────────────────────────────────────────────
SQLITE_BUSY_TIMEOUT: int = 30000            # 毫秒

# ──────────────────────────────────────────────
# Embedding 模型
# ──────────────────────────────────────────────
EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM: int = 384                    # 向量维度

# ──────────────────────────────────────────────
# 管理后台认证（从环境变量读取，不硬编码）
# ──────────────────────────────────────────────
ADMIN_USER: str = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS: str = os.getenv("ADMIN_PASS", "dramalingo2026")

# ──────────────────────────────────────────────
# 日志配置
# ──────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")  # DEBUG / INFO / WARNING / ERROR

# ──────────────────────────────────────────────
# 服务配置
# ──────────────────────────────────────────────
HOST: str = "0.0.0.0"
PORT: int = 8000


def validate_config() -> None:
    """启动时校验关键参数范围，发现异常立即报错。"""
    errors = []

    if not (0.0 < QUALITY_THRESHOLD < 1.0):
        errors.append(f"QUALITY_THRESHOLD={QUALITY_THRESHOLD} 必须在 (0, 1) 之间")
    if not (0.0 < DEDUP_SIMILARITY <= 1.0):
        errors.append(f"DEDUP_SIMILARITY={DEDUP_SIMILARITY} 必须在 (0, 1] 之间")
    if not (0.0 < DEDUP_EDIT_DISTANCE_THRESHOLD < 1.0):
        errors.append(f"DEDUP_EDIT_DISTANCE_THRESHOLD={DEDUP_EDIT_DISTANCE_THRESHOLD} 必须在 (0, 1) 之间")
    if CLIP_MIN_DURATION >= CLIP_MAX_DURATION:
        errors.append(f"CLIP_MIN_DURATION({CLIP_MIN_DURATION}) 必须小于 CLIP_MAX_DURATION({CLIP_MAX_DURATION})")
    if MIN_WORDS >= MAX_WORDS:
        errors.append(f"MIN_WORDS({MIN_WORDS}) 必须小于 MAX_WORDS({MAX_WORDS})")
    weight_sum = (
        RANK_WEIGHT_ZH_SIM + RANK_WEIGHT_EN_TRANSLATED + RANK_WEIGHT_EN_SIM
        + RANK_WEIGHT_TAG + RANK_WEIGHT_QUALITY + RANK_WEIGHT_OCCURRENCE
    )
    if abs(weight_sum - 1.0) > 1e-6:
        errors.append(f"多信号检索权重之和={weight_sum:.4f}，必须等于 1.0")

    if errors:
        raise ValueError("config.py 参数校验失败：\n" + "\n".join(f"  · {e}" for e in errors))
