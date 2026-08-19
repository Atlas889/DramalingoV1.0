# DramaLingo · 影视表达检索引擎

![Status](https://img.shields.io/badge/Status-MVP-success)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688)
![FAISS](https://img.shields.io/badge/FAISS-1.8.0-orange)
![SQLite](https://img.shields.io/badge/SQLite-WAL-lightgrey)

**DramaLingo** 是面向中国英语学习者的「影视表达检索引擎」——用中文描述你想说的话，系统自动从结构化的影视台词知识库中找到最地道的英文说法，连同真实的影视场景视频片段一起呈现。

---

## 目录

- [核心价值主张](#核心价值主张)
- [系统架构](#系统架构)
- [字幕质量评分与过滤机制](#字幕质量评分与过滤机制)
- [弃用节点分析机制](#弃用节点分析机制)
- [视频切片机制](#视频切片机制)
- [多信号语义检索](#多信号语义检索)
- [快速开始](#快速开始)
- [目录结构](#目录结构)
- [环境变量配置](#环境变量配置)
- [管理后台功能](#管理后台功能)
- [API 接口总览](#api-接口总览)
- [数据库表结构](#数据库表结构)
- [内容类型说明](#内容类型说明)
- [关键算法参数](#关键算法参数)
- [AI 接手指南](#ai-接手指南)
- [未来优化路线](#未来优化路线)

---

## 核心价值主张

| 维度 | 传统翻译工具 | DramaLingo |
|------|-------------|------------|
| **输入方式** | 词 / 句 | 中文自然语言场景描述 |
| **输出内容** | 机器翻译文本 | 真实影视台词 + 视频片段 |
| **语境感知** | 无 | 情景 + 情绪 + 言语行为标签 |
| **可信度来源** | 机器生成 | 影视作品背书 |
| **学习沉浸感** | 低 | 高（可边看边学） |

---

## 系统架构

DramaLingo 采用**字幕与视频处理解耦**的三阶段设计，字幕先行、知识库驱动切片，从源头过滤无效内容。

```
┌─────────────────────────────────────────────────────────────────┐
│                        三阶段解耦流水线                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  阶段一：字幕入库（纯文本，无需视频）                              │
│  ──────────────────────────────────────────────────────────────  │
│  上传 .srt/.ass                                                   │
│    → 解析（SRT/ASS/SSA，自动识别编码，提取双语）                   │
│    → 五道质量筛选（格式 → 黑名单 → 词数 → 口语真实性 × 语用完整性） │
│    → Embedding 向量化（multilingual-MiniLM，384 维）              │
│    → 两段式去重（编辑距离粗筛 + FAISS 语义精筛）                   │
│    → AI 打标（scene / emotion / difficulty / keywords /          │
│               speech_act）                                       │
│    → 写入 knowledge_nodes + FAISS 索引                           │
│                                                                  │
│  阶段二：视频切片（按需）                                          │
│  ──────────────────────────────────────────────────────────────  │
│  上传视频文件                                                      │
│    → Whisper 转写（结果永久缓存）                                  │
│    → 零散片段合并（多信号边界检测）                                 │
│    → 序列对齐（Levenshtein + Embedding + 顺序奖励）                │
│    → 置信度门控（C1 ≥ 0.85 / C2 ≥ 0.60 / C3 兜底/丢弃）          │
│    → 上下文窗口扩展 + 端点 Snap + FFmpeg 并发切片                  │
│    → 写入 clips 表 + 更新 has_any_clip                            │
│                                                                  │
│  阶段三：用户检索                                                  │
│  ──────────────────────────────────────────────────────────────  │
│  中文查询                                                          │
│    → 查询意图扩展（DeepSeek 补全描述 + 候选英文表达）              │
│    → 三路语义向量检索（zh_sim + en_tr_sim + en_sim）               │
│    → 多信号综合排序（语义 + 标签 + 质量 + 出现频次）                │
│    → 关联 clips 返回结果卡片                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 字幕质量评分与过滤机制

> 核心代码：`services/quality_filter.py`

字幕进入知识库前须通过**五道串行过滤**。前三道是快速二值拒绝，后两道产生连续质量分。每道过滤有独立的 `filter_status` 标记，方便审查和溯源。

### 第一道：基础格式过滤 `rejected_blacklist`

拒绝以下类型的行，这些行没有语言学价值：

- 纯数字、纯标点符号
- 完整方括号注释（`[applause]`、`[music]`）
- 包含音乐符号（♪ ♫）
- 去掉标点后有效字母内容 ≤ 1 个字符

### 第二道：黑名单过滤 `rejected_blacklist`

匹配约 130 个高频套话短语（归一化后精确匹配），例如：

```
hello / bye / thank you / sorry / yes / no / okay / whatever
good morning / nice to meet you / i don't know / let's go ...
```

这些表达过于泛用，脱离语境后几乎无法教学，且学习者已从教材中习得。

### 第三道：词数门控 `rejected_wordcount`

```
MIN_WORDS = 2    # 避免孤立单词噪声
MAX_WORDS = 100  # 避免叠行字幕或旁白段
```

### 第四道：口语真实性评分 × 语用完整性系数 `rejected_quality`

这是核心质量关卡，由**四维加权评分**和**语用完整性乘法系数**组成。

#### 4a. 口语真实性评分（Authenticity Score）

```
authenticity = 缩略语密度 × 0.30
             + 词汇密度   × 0.25
             + 短语动词   × 0.25
             + 句型价值   × 0.20
```

| 维度 | 计算方式 | 满分基准 |
|------|---------|---------|
| **缩略语密度** | 检测 I'm / can't / gonna / wanna 等 50+ 缩略形式；每 5 词出现 1 个缩略 = 满分 | 反映口语风格，书面语几乎不缩略 |
| **词汇密度** | 实词占比，以 0.55 为满分基准（自然口语区间 0.35–0.55） | 信息量指标，过低为无意义填充词堆砌 |
| **短语动词 + 惯用语** | 检测 200+ 短语动词（give up / figure out / move on 等）+ 30+ 口语惯用表达；每 8 词出现 1 个 = 满分，惯用语权重 ×1.5 | 母语者核心用语，教材极少覆盖 |
| **句型价值** | 陈述句基础 0.5；疑问句 +0.2；否定句 +0.15；条件句 +0.10；多子句 +0.05；超短句（≤2词）−0.30 | 句型结构的可教学价值 |

**格式垃圾直接归零**：包含 URL / `Season N` / `subtitles by` / `fade in` 等字幕制作标记的行，authenticity 直接返回 0.0。

#### 4b. 语用完整性系数（Pragmatic Autonomy Factor，L2）

语用完整性评估一句话能否**脱离原始话语语境独立使用**——这是衡量一个节点能否被学习者迁移使用的根本条件。

```
quality_score = authenticity × pragmatic_factor
pragmatic_factor ∈ [0.6, 1.0]
```

系数由五类否决信号累积惩罚决定，下界 0.6 确保单一问题不一票否决：

| 信号 | 检测条件 | 惩罚 | 典型例子 |
|------|---------|------|---------|
| **P1 末尾截断** | 以 `...` 或孤立逗号结尾 | −0.30 | `"I just think that..."` |
| **P2 续接词开头** | `but that / and so / that's why / which means / that is why` 等 | −0.25 | `"Which means we start over."` |
| **P3 悬挂条件句** | `if/unless` 开头 + 句内无逗号（无主句解析） | −0.20 | `"If you really have to."` |
| **P4 强因果回指** | `because of that / as a result / for that reason / in that case` | −0.15 | `"Because of that, we ran."` |
| **P5 依存性介词** | `alongside / together with / on top of that / apart from that` 等 | −0.15 | `"On top of that, we also need..."` |

> **选择乘法而非第五道门的原因**：语用问题是否决型信号而非程度量，乘法保持四维评分的解释性不变，同时让一句高口语质量但语用残缺的句子自然下压到阈值以下；且不引入新字段，向后兼容已有阈值（0.30）。

**通过阈值**：`quality_score ≥ 0.30` → 状态变为 `passed`

经80条人工标注基准测试：
- 旧版 TF-IDF 过滤：F1 = 75.2%，误杀率 57.5%
- 当前评分系统：F1 = 97.5%，误杀率 2.5%

### 第五道（后置）：语境质量二次评分

在 Embedding 向量化完成后执行，不拦截入库，而是生成最终写入的 `quality_score`：

```
contextual_score = 剧集内 TF-IDF 独特性 × 0.40
                 + 知识库新颖度         × 0.60
```

- **剧集内 TF-IDF**：相对于本集所有字幕，当前行使用了多少稀有词汇（信息价值高）
- **知识库新颖度**：与 FAISS 索引中现有节点的余弦相似度越低，新颖度越高（增量价值高）

---

## 弃用节点分析机制

> 核心代码：`services/deprecated_analyzer.py`，API：`POST/GET /knowledge/analyze-deprecated`

当管理员手动弃用一批节点（或质量重评估自动降级）后，系统自动分析这批弃用节点的语言规律，并输出可用于反哺质量过滤器的改进建议——形成**弃用 → 分析 → 改进过滤 → 减少弃用**的闭环。

### 自动触发条件

以下 5 个 API 操作完成后，系统以 **fire-and-forget** 方式（延迟 3 秒）自动提交后台分析任务：

| 触发路径 | source 标签 |
|---------|------------|
| `PATCH /knowledge/nodes/{id}` 将 `is_active` 设为 `false` | `patch-node` |
| `POST /knowledge/nodes/batch-update` 含 `is_active=false` | `batch-update` |
| `POST /knowledge/nodes/{id}/re-evaluate` 节点被降级 | `re-evaluate` |
| `POST /knowledge/nodes/batch-re-evaluate` 有节点被降级 | `batch-re-evaluate` |
| `POST /knowledge/full-re-evaluate` 任务内有节点被降级 | `full-re-evaluate` |

3 秒延迟确保主事务已提交；`asyncio.create_task()` 不阻塞调用方响应。

### 分析报告结构

```json
{
  "total_deprecated": 42,
  "analyzed_at": "2026-05-26T...",
  "patterns": {
    "stage_direction":   { "count": 8, "ratio": 0.19, "already_fixed": true, "examples": [...], "suggestion": "..." },
    "self_introduction": { "count": 3, "ratio": 0.07, "already_fixed": true, "examples": [...], "suggestion": "..." },
    "trailing_truncation": { ... },
    "pronoun_dependent":   { "count": 5, "ratio": 0.12, "already_fixed": false, ... }
  },
  "score_distribution": { "buckets": {...}, "mean": 0.28, "median": 0.25, "stdev": 0.09 },
  "filter_recommendations": [
    { "signal": "pronoun_dependent", "action": "建议新增 L2 P6 惩罚 -0.10", "confidence": "medium" }
  ]
}
```

### 11 种识别模式

| 模式键 | 描述 | 是否已修复 |
|--------|------|-----------|
| `stage_direction` | 括号/方括号舞台指示或音效符号 | ✅ G1 基础过滤 |
| `self_introduction` | 演讲者自我介绍（姓名绑定不可复用） | ✅ G2 黑名单 |
| `trailing_truncation` | 句末省略号/悬挂逗号（被截断） | ✅ L2 P1 |
| `continuation_opener` | 续接词 + 强指称开头 | ✅ L2 P2 |
| `hanging_conditional` | 悬挂条件从句，缺主句 | ✅ L2 P3 |
| `causal_anaphora` | 因果回指词，需先行句才能理解 | ✅ L2 P4 |
| `dependency_preposition` | 依存性介词起头，预设上文情境 | ✅ L2 P5 |
| `pronoun_dependent` | 第三人称代词开头，指代不明 | ⬜ 待新增 L2 P6 |
| `format_garbage` | 字幕制作标注、网址等格式噪声 | ✅ G4 归零 |
| `low_authenticity` | 质量分低但未命中具体规则 | ⬜ 需人工复审 |
| `unclassified` | 未命中任何规则 | ⬜ 需人工归纳 |

---

## 视频切片机制

> 核心代码：`services/clip_pipeline.py`、`services/aligner.py`、`services/whisper_service.py`

**设计原则**：字幕先行，切片从知识库反推。只为已通过质量筛选并入库的节点切片，避免处理大量低价值内容。

### 流水线总览

```
视频文件
  │
  ▼ Step 1
Whisper 转写                    ← 永久缓存，重复上传同集视频跳过
  │  原始片段数百个（单词/短句粒度）
  ▼ Step 2
零散片段合并                     ← 多信号边界检测，合并为完整句子
  │  合并后 50–200 个语义段
  ▼ Step 3
查询本集知识节点                  ← 从 knowledge_nodes 取本集所有节点
  │
  ▼ Step 4
序列对齐（逐节点匹配）            ← Levenshtein + Embedding + 顺序奖励
  │  每节点 → 最佳 Whisper 段 + 置信度
  ▼ Step 5
置信度门控                       ← C1 高置信 / C2 中置信 / C3 兜底/丢弃
  │
  ▼ Step 6
上下文窗口扩展                   ← 前后各取 ±1 邻段，场景断点保护
  │
  ▼ Step 7
端点 Snap（吸附静音边界）         ← 避免切到说话中途
  │
  ▼ Step 8
FFmpeg 并发切片 + 缩略图生成
  │
  ▼
写入 clips 表，更新 has_any_clip
```

### Step 2：零散片段合并

Whisper 原始输出为词或短句粒度，需合并为完整语义段。合并采用**多信号边界检测**，任意一个信号触发则不合并：

| 边界信号 | 条件 | 作用 |
|---------|------|------|
| **时间间隔** | gap > 1.5s（硬边界） | 场景/说话人切换 |
| **句子标点** | 当前段以 `.?!...` 结尾；或以 `,` 结尾且 ≥ 8 词 | 语义完整 |
| **大小写跳转** | 当前段以小写字母收尾，下一段以大写字母起头，且 gap > 0.6s | 说话人切换信号 |
| **时长上限** | 合并后 > 12s | 防止过度合并 |
| **词数上限** | 合并后 > 40 词 | 防止跨说话人合并 |

### Step 4：序列对齐

对每个知识节点，在其字幕时间戳 **±30s 窗口**内搜索最匹配的 Whisper 片段。

#### 三信号综合评分

```
match_score = lev_sim  × 0.55    # Levenshtein 字面相似度（对齐专用归一化后）
            + emb_cos  × 0.35    # Embedding 余弦语义相似度（FAISS dot product）
            + order    × 0.10    # 顺序奖励 +0.10 / 乱序惩罚 −0.15
```

**对齐专用归一化**在去重归一化基础上额外剥离：
- Whisper 填充词（uh / um / you know / kind of 等）
- SRT 括号注释（`[door slams]`、`(whispering)`、`♪ music ♪`）

**候选预过滤**：词数比例 `min/max < 0.4` 的候选直接淘汰（避免将短词条匹配到长片段）。

**短文本特殊处理**：词数 ≤ 3 的节点要求相似度 ≥ 0.90，防止 "Hi"/"Yeah" 之类噪声匹配。

**模糊匹配检测**：最佳分与次优分之差 < 0.08 时，标记 `align_step = "ambiguous"`，写入 clips 但提醒人工审查。

### Step 5：置信度门控

| 等级 | 置信度范围 | 处理方式 | align_step |
|------|-----------|---------|-----------|
| **C1 高置信** | ≥ 0.85 | 切片，使用 Whisper 时间戳 | `seq` |
| **C2 中置信** | 0.60 – 0.85 | 切片，仍使用 Whisper 时间戳 | `seq` |
| **C3 低置信** | < 0.60 | 尝试字幕时间戳兜底（需满足时长约束）；若不满足则不切片，仅记录 | `subtitle_fallback` / `unmatched` |

**字幕兜底（subtitle_fallback）**：使用原始 SRT 时间戳，但先将其**锚定到最近的 Whisper 段端点**（±5s 搜索半径），修正 SRT 的 1–3s 偏差。

### Step 6：上下文窗口扩展

匹配到核心片段后，向前向后各追加 1 个相邻 Whisper 段，以保留对白语境。追加受到多重保护：

| 保护条件 | 阈值 | 说明 |
|---------|------|------|
| **场景断点** | gap > 1.5s | 邻段静音过长，视为换场，停止扩展 |
| **邻段过长** | 邻段时长 > 3.0s | 独立长句，不追加（避免吞掉另一句台词） |
| **单向额外时长** | > 6.0s | 单方向上下文总量上限 |
| **总时长溢出** | 扩展后 + padding > 15s | 退回主片段，不扩展 |

### Step 7：端点 Snap（吸附静音边界）

计算出切点后，在 `[0.3s, 1.5s]` 范围内寻找最近的 **Whisper 片段间隙中点**，将切点吸附过去。

- 这一步避免在说话途中切入/切出，让切片开头不会突兀进入单词中间
- 找不到合适间隙时，退回到原切点（由 padding 缓冲吸收）

### Step 8：FFmpeg 切片策略

**起止时间**：`cut_start = ctx_start − 0.5s`，`cut_end = ctx_end + 1.0s`（padding）

**编码策略自动选择**：

| 源格式 | 策略 | 速度 |
|--------|------|------|
| `.mp4` `.mkv` `.mov` `.m4v` `.ts` `.webm` | `-c copy` 流复制无损切片 | 快（约 0.1s/片） |
| `.avi` `.rmvb` `.wmv` `.flv` | `libx264 + aac` 重编码 | 慢（约 5–30s/片） |

**缩略图**：从切片 0.5s 处取帧（避开开头黑帧），若切片不足 0.5s 则取第 1 帧。

**并发控制**：最多 `FFMPEG_CONCURRENT=4` 个 FFmpeg 进程同时运行（可通过环境变量调整）。

---

## 多信号语义检索

### 综合排序公式

```
final_score = zh_sim    × 0.40    # 中文查询 → 中文翻译 Embedding（zh_embedding）
            + en_tr_sim × 0.25    # 查询翻译为英文 → 英文字幕 Embedding
            + en_sim    × 0.15    # 中文查询跨语言 → 英文字幕 Embedding（baseline）
            + tag_score × 0.12    # 场景/情绪关键词标签匹配
            + quality   × 0.06    # 口语真实性评分
            + occ_norm  × 0.02    # 出现频次（log 归一化，饱和值 50）
```

### 查询意图扩展

用户输入中文描述后，系统调用 DeepSeek 进行意图扩展，输出：
- `expanded_zh`：语义更丰富的中文描述（用于 zh_sim 计算）
- `candidate_en`：3 条候选英文表达（用于 en_tr_sim 多路取 max）

扩展超时（3s）或失败时自动降级，不影响返回速度。

### 过滤与兜底

- **绝对门槛**：三路信号中最高者 < `RANK_MIN_SIMILARITY=0.50` 的候选直接丢弃
- **相对截断**：只保留与最高综合分差距 ≤ `RANK_RELATIVE_THRESHOLD=0.06` 的结果（减少长尾噪音）
- **关键词降级**：FAISS 索引为空或语义检索无结果时，自动降级为 ILIKE 关键词匹配

---

## 快速开始

### 环境要求

- Python 3.11+
- FFmpeg（需安装在系统 PATH 中）
- 至少一个 AI API Key（DeepSeek 或 Qwen，用于字幕打标）

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/Atlas889/Dramalingo_mvp.git
cd Dramalingo_mvp-main

# 2. 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt

# 3. 配置环境变量
copy .env.example .env          # Windows
# cp .env.example .env          # Mac/Linux
# 编辑 .env，填入 AI_API_KEY 等必要配置
```

### 启动服务

**Windows（推荐）：**

```powershell
# 方式 1：双击 start.bat
# 方式 2：PowerShell 脚本
.\start.ps1

# 方式 3：手动启动
.venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Mac/Linux：**

```bash
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 访问入口

| 入口 | 地址 | 说明 |
|------|------|------|
| 用户端 | `http://localhost:8000/` | 中文语义检索页，公开访问 |
| 管理后台 | `http://localhost:8000/admin` | 运营管理页，Basic Auth 保护 |
| API 文档 | `http://localhost:8000/docs` | Swagger 交互式文档 |
| 健康检查 | `http://localhost:8000/health` | 返回 `{"status":"healthy"}` |

> 管理后台默认账号密码见 `.env` 中的 `ADMIN_USER` / `ADMIN_PASS`（默认 `admin` / `dramalingo2026`）

---

## 目录结构

```
Dramalingo_mvp-main/
├── main.py                  # FastAPI 应用入口，lifespan 管理，路由注册，增量迁移
├── config.py                # 全局配置参数（路径、算法阈值、模型参数）
├── database.py              # SQLAlchemy 异步引擎，WAL 模式，增量迁移
├── requirements.txt         # Python 依赖
├── .env.example             # 环境变量模板
│
├── models/                  # ORM 数据模型
│   ├── subtitle.py          # subtitles_raw — 字幕原始行
│   ├── knowledge_node.py    # knowledge_nodes — 知识节点（核心表）
│   ├── node_source.py       # node_sources — 知识节点来源溯源
│   ├── clip.py              # clips — 视频切片
│   ├── video_file.py        # video_files — 已上传视频
│   ├── whisper_transcript.py# whisper_transcripts — Whisper 转写缓存
│   ├── subtitle_job.py      # subtitle_jobs — TED 字幕获取任务
│   ├── persistent_task.py   # persistent_tasks — 任务持久化（重启恢复）
│   └── search_feedback.py   # search_feedback — 用户搜索行为记录
│
├── routers/                 # FastAPI 路由
│   ├── subtitles.py         # 字幕上传、列表、剧集查询
│   ├── subtitle_jobs.py     # TED 字幕获取任务管理
│   ├── knowledge.py         # 知识库 CRUD、统计、晋升、重打标
│   ├── search.py            # 多信号语义检索
│   ├── clips.py             # 视频切片生成与管理
│   └── videos.py            # 视频文件上传与 Whisper 转写
│
├── services/                    # 核心业务逻辑
│   ├── subtitle_pipeline.py     # 字幕处理完整流水线
│   ├── subtitle_parser.py       # SRT/ASS 字幕解析（含 _parse_srt_text_only 纯文本解析器）
│   ├── quality_filter.py        # 五道质量筛选（含 L2 语用完整性系数 P1–P5）
│   ├── deprecated_analyzer.py   # 弃用节点模式分析（11 种模式，自动生成过滤改进建议）
│   ├── embedder.py              # Embedding 向量化（multilingual-MiniLM）
│   ├── deduplicator.py          # 两段式去重（编辑距离 + FAISS）
│   ├── ai_tagger.py             # AI 打标（DeepSeek/Qwen），含 speech_act 25 种枚举
│   ├── faiss_index.py           # FAISS 索引管理（增量/全量重建，en + zh 双索引）
│   ├── query_engine.py          # 多信号排序、查询扩展、标签匹配
│   ├── clip_pipeline.py         # 视频切片完整流水线
│   ├── whisper_service.py       # Whisper 转写服务（永久缓存）
│   ├── aligner.py               # 序列对齐（Levenshtein + Embedding + 顺序奖励）
│   └── ted_fetcher.py           # TED 演讲字幕自动获取
│
├── tasks/
│   └── task_runner.py       # 串行异步任务调度器（asyncio.Queue + DB 持久化）
│
└── static/
    ├── admin.html           # 管理后台单页应用（Vanilla JS）
    └── user.html            # 用户检索端
```

---

## 环境变量配置

复制 `.env.example` 为 `.env` 并按需修改：

```ini
# ── 管理后台认证 ──────────────────────────────────
ADMIN_USER=admin
ADMIN_PASS=dramalingo2026

# ── AI 打标服务（DeepSeek，推荐）────────────────────
AI_PROVIDER=deepseek
AI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
AI_API_BASE_URL=https://api.deepseek.com/v1

# ── 第二 AI 提供商（可选，启用后吞吐量接近翻倍）──────
# AI_PROVIDER_2=qwen
# AI_API_KEY_2=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# AI_API_BASE_URL_2=https://dashscope.aliyuncs.com/compatible-mode/v1

# ── 文件存储根目录（视频、切片、临时文件）───────────
DRAMALINGO_FILES_ROOT=E:\dramalingo-files

# ── HuggingFace 镜像（国内必填，解决模型下载超时）──
HF_ENDPOINT=https://hf-mirror.com

# ── 日志级别 ─────────────────────────────────────
LOG_LEVEL=INFO

# ── Whisper 模型大小（tiny/base/small/medium）────
WHISPER_MODEL=small

# ── FFmpeg 并发数 ─────────────────────────────────
FFMPEG_CONCURRENT=4
```

> **注意**：`DRAMALINGO_FILES_ROOT` 目录会在首次启动时自动创建，无需手动建立。

---

## 管理后台功能

访问 `http://localhost:8000/admin`（需要 Basic Auth 登录）。

| 模块 | 功能描述 |
|------|---------|
| **字幕获取** | 输入 TED URL，自动获取双语字幕；支持官方中文/AI 翻译/AI 润色；可预览、下载、发送到流水线 |
| **字幕处理** | 上传本地 .srt/.ass 字幕文件，触发完整流水线（筛选 → Embedding → 去重 → AI 打标 → 写库） |
| **知识库管理** | 分页浏览、多维筛选（大类/剧名/集号/难度/场景/情绪/speech_act）、批量编辑/删除、节点溯源 |
| **字幕记录** | 查看所有已入库字幕原始行，按剧集/筛选状态过滤，支持一键晋升被过滤行到知识库 |
| **视频上传** | 按大类→剧名→集号三级联动选择字幕批次，上传视频触发 Whisper 转写 + 切片流水线 |
| **切片管理** | 查看所有视频切片，可播放/删除，查看质量报告（覆盖率/置信度分布） |
| **任务队列** | 实时监控所有后台任务进度（字幕获取/字幕处理/视频切片），可查看错误详情 |
| **系统统计** | 知识库总量、剧集分布、难度分布、AI 打标率、切片覆盖率等关键指标 |

---

## API 接口总览

### 字幕管理 `/subtitles`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/subtitles/upload` | 单文件字幕上传（SRT/ASS） |
| POST | `/subtitles/upload-bilingual` | 双语字幕上传（英文 + 中文各一个） |
| GET  | `/subtitles/list` | 分页查询字幕原始记录 |
| GET  | `/subtitles/shows` | 查询有知识库记录的剧名列表 |
| GET  | `/subtitles/batches` | 查询字幕批次列表 |
| GET  | `/subtitles/stats` | 字幕记录统计 |

### TED 字幕任务 `/subtitle-jobs`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/subtitle-jobs` | 新建 TED 字幕获取任务 |
| GET  | `/subtitle-jobs` | 列出所有任务 |
| GET  | `/subtitle-jobs/{id}/preview` | 预览 SRT 文件内容 |
| POST | `/subtitle-jobs/{id}/send` | 发送到字幕处理流水线 |

### 知识库管理 `/knowledge`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/knowledge/nodes` | 分页查询（支持 speech_act / scene / difficulty 等多维筛选） |
| PATCH | `/knowledge/nodes/{id}` | 编辑单节点（设 is_active=false 自动触发弃用分析） |
| DELETE | `/knowledge/nodes/{id}` | 删除单节点 |
| POST | `/knowledge/nodes/batch-delete` | 批量删除 |
| POST | `/knowledge/nodes/batch-update` | 批量更新标签（含批量弃用，触发弃用分析） |
| GET  | `/knowledge/status` | 知识库整体统计 |
| POST | `/knowledge/promote` | 手动晋升被过滤字幕为知识节点 |
| POST | `/knowledge/retag` | 批量重新 AI 打标（含 speech_act） |
| POST | `/knowledge/retag-speech-acts` | 专项补标 speech_act（NULL 节点） |
| POST | `/knowledge/sync-passed` | 一键同步所有 passed 字幕到知识库 |
| POST | `/knowledge/backfill-zh-embeddings` | 批量补填中文 Embedding |
| POST | `/knowledge/merge-adjacent` | 对已入库节点执行相邻分句合并 |
| POST | `/knowledge/full-re-evaluate` | 全库质量分重新评估（后台任务，有降级时触发弃用分析） |
| POST | `/knowledge/nodes/{id}/re-evaluate` | 单节点质量分重评估（降级时触发弃用分析） |
| POST | `/knowledge/nodes/batch-re-evaluate` | 批量节点质量分重评估（有降级时触发弃用分析） |
| POST | `/knowledge/analyze-deprecated` | 触发全量弃用节点模式分析，返回完整统计报告 |
| GET  | `/knowledge/analyze-deprecated` | 轻量摘要（采样 30 条），供管理后台实时展示 |

### 语义检索 `/search`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/search` | 多信号语义检索（q / limit / offset / difficulty / scene / speech_act / has_clip） |
| GET  | `/search/status` | 检索模块状态（FAISS 索引大小、zh_embedding 覆盖率） |
| POST | `/search/feedback` | 记录用户搜索交互行为（click / play / copy），用于排序调优 |
| GET  | `/search/feedback/stats` | 搜索反馈统计（按 action 分布 + top 热门查询，支持 days 参数） |

### 视频切片 `/clips`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/clips/generate` | 上传视频，触发完整切片流水线 |
| GET  | `/clips/list` | 查询切片列表（支持按剧名/集号/置信度等筛选） |
| GET  | `/clips/report/{show}/{ep}` | 指定集的切片质量报告（覆盖率/置信度分布） |
| DELETE | `/clips/{id}` | 删除单个切片 |
| POST | `/clips/clear-all` | 清空全部切片数据（危险） |

### 视频管理 `/videos`

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/videos/upload-and-transcribe` | 上传视频并触发 Whisper 转写（缓存） |
| GET  | `/videos/list` | 查询已上传视频列表 |
| GET  | `/videos/transcripts/{show}/{ep}` | 查询 Whisper 转写结果 |
| DELETE | `/videos/transcripts/{show}/{ep}` | 删除转写缓存（用于重新转写） |

### 任务状态 `/jobs`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/jobs` | 列出所有后台任务（含进度、阶段、错误信息） |
| GET  | `/jobs/{job_id}` | 查询单个任务实时进度 |

---

## 数据库表结构

SQLite 数据库，WAL 模式（支持读写并发），文件路径：`dramalingo.db`。

```
subtitles_raw          — 字幕原始行（每条字幕一行，含筛选状态）
  ├─ show_name         字幕归属的中文剧名
  ├─ en_name           英文剧名
  ├─ episode           集号（如 S01E01；演讲为空）
  ├─ subtitle_code     上传批次编码（同批文件共享）
  ├─ content_type      内容大类：tv | movie | talk
  ├─ en_text / zh_text 英文/中文字幕文本
  ├─ filter_status     筛选状态：passed | rejected_blacklist |
  │                              rejected_wordcount | rejected_quality |
  │                              rejected_dedup | promoted
  └─ quality_score     口语真实性 × 语用完整性综合评分

knowledge_nodes        — 知识节点（核心表，每条英文表达全库唯一）
  ├─ code              节点唯一编码（FRIENDS-S01E01-00154520）
  ├─ en_text / zh_text 英文台词 / 中文翻译
  ├─ show_name         首次来源剧名
  ├─ content_type      内容大类：tv | movie | talk
  ├─ scene             情景标签：daily | work | romance | conflict | humor
  ├─ emotion           情绪标签：neutral | happy | sad | angry | surprised
  ├─ difficulty        难度：beginner | intermediate | advanced
  ├─ speech_act        言语行为（25 种，见下表）
  ├─ keywords          关键词数组（JSON，3–5 个）
  ├─ embedding         英文 Embedding（384 维，BLOB）
  ├─ zh_embedding      中文翻译 Embedding（384 维，BLOB）
  ├─ occurrence_count  全库出现频次（越多越地道）
  ├─ has_any_clip      是否有关联的有效切片（汇总字段）
  ├─ quality_score     口语真实性 × 语用完整性综合评分
  └─ tag_source        打标来源：ai | rule | manual

node_sources           — 知识节点来源溯源（1节点 : N来源）
clips                  — 视频切片（1节点 : N切片，保留历史轨迹）
  ├─ clip_code         切片业务编码（CLP-FR-S1-A3F8C2）
  ├─ match_confidence  对齐置信度
  ├─ align_step        对齐策略：seq | subtitle_fallback | ambiguous | unmatched
  ├─ whisper_start/end Whisper 原始时间戳（秒）
  └─ stream_url        切片文件 URL（/media/...）

video_files            — 已上传视频文件（1集 : 1视频）
whisper_transcripts    — Whisper 转写缓存（永久保留，避免重复转写）
subtitle_jobs          — TED 字幕获取任务状态

persistent_tasks       — 后台任务持久化（解决重启丢失问题）
  ├─ task_id           UUID，与内存 task_store 一致
  ├─ status            pending | running | done | failed | interrupted
  ├─ description       任务描述
  ├─ result_summary    完成时的摘要（JSON，截断 2000 字符）
  └─ error_message     失败时的错误信息

search_feedback        — 用户搜索交互行为记录（用于排序权重调优）
  ├─ query             用户查询文本
  ├─ node_id           被交互的知识节点 ID
  ├─ action            click | play | copy
  ├─ rank_position     结果在列表中的排名位置
  └─ rank_score        该结果的综合排序分
```

### speech_act 枚举值（25 种）

| 分类 | 值 | 含义 |
|------|---|------|
| **表达类** | `apologizing` | 道歉 |
| | `comforting` | 安慰 |
| | `complaining` | 抱怨 |
| | `complimenting` | 称赞 |
| | `expressing_surprise` | 惊讶 |
| | `expressing_doubt` | 质疑 |
| | `regretting` | 后悔 |
| **指令类** | `requesting` | 请求 |
| | `offering` | 提议 |
| | `warning` | 警告 |
| | `threatening` | 威胁 |
| | `refusing` | 拒绝 |
| **承诺类** | `promising` | 承诺 |
| **断言类** | `agreeing` | 赞同 |
| | `disagreeing` | 反对 |
| | `emphasizing` | 强调 |
| | `hedging` | 缓和立场 |
| | `explaining` | 解释 |
| | `justifying` | 辩护 |
| | `speculating` | 推测 |
| | `criticizing` | 批评 |
| **语篇控制** | `initiating` | 开场 |
| | `changing_topic` | 转移话题 |
| | `closing` | 结束 |
| | `neutral_statement` | 中性陈述 |

---

## 内容类型说明

系统支持三种内容大类，在上传字幕和视频时需指定：

| 大类值 | 显示名称 | 适用场景 | 集号字段 |
|--------|---------|---------|---------|
| `tv`   | 📺 电视剧 | 连续剧，有集号 | 必填，如 S01E01 |
| `movie`| 🎬 电影   | 单部电影 | 留空或填 "电影" |
| `talk` | 🎤 演讲   | TED 等演讲视频 | 留空或填演讲标题 |

---

## 关键算法参数

所有参数均在 `config.py` 中定义，可通过环境变量覆盖（标注 `[env]` 的字段）。

### 字幕质量筛选

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `QUALITY_THRESHOLD` | `0.30` | `quality_score = authenticity × pragmatic_factor` 下限 |
| `MIN_WORDS` | `2` | 最少词数 |
| `MAX_WORDS` | `100` | 最多词数 |
| `DEDUP_SIMILARITY` | `0.95` | 语义去重精筛阈值（Embedding 余弦相似度） |
| `DEDUP_EDIT_DISTANCE_THRESHOLD` | `0.15` | 编辑距离粗筛阈值 |

### 视频切片

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CLIP_MAX_DURATION` | `15.0s` | 单个切片最大时长（含 padding） |
| `CLIP_MIN_DURATION` | `1.0s` | 单个切片最小时长 |
| `CLIP_PADDING_START` | `0.5s` | 切片起点前置缓冲 |
| `CLIP_PADDING_END` | `1.0s` | 切片终点后置缓冲 |
| `CLIP_MIN_QUALITY` `[env]` | `0.60` | C3 门控：低于此分的节点不切片 |
| `WHISPER_MERGE_GAP_SECONDS` | `1.5s` | 片段合并最大间隔 |
| `WHISPER_MERGE_MAX_DURATION` | `12.0s` | 合并后单段最大时长 |
| `CLIP_CONTEXT_BEFORE/AFTER` | `1` | 上下文窗口前/后邻段数 |
| `CLIP_CONTEXT_SCENE_GAP` | `1.5s` | 场景断点检测阈值 |
| `CLIP_SNAP_TO_BOUNDARY_MAX` | `1.5s` | 端点 Snap 最大调整量 |
| `WHISPER_MODEL` `[env]` | `small` | Whisper 模型大小（tiny/base/small/medium） |
| `FFMPEG_CONCURRENT` `[env]` | `4` | 并发 FFmpeg 进程数 |

### 序列对齐

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ALIGN_LEV_WEIGHT` | `0.55` | Levenshtein 字面相似度权重 |
| `ALIGN_EMB_WEIGHT` | `0.35` | Embedding 余弦相似度权重 |
| `ALIGN_ORDER_BONUS` | `0.10` | 顺序奖励（候选在游标之后） |
| `ALIGN_ORDER_PENALTY` | `0.15` | 乱序惩罚（候选在游标之前） |
| `ALIGN_AMBIGUOUS_MARGIN` | `0.08` | 最佳分与次优分之差低于此值标记为模糊 |
| `WHISPER_ALIGN_WINDOW_SECONDS` | `30.0s` | 字幕时间戳搜索窗口 |

### FAISS 与检索

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FAISS_RECALL_K` | `50` | 语义检索召回缓冲数 |
| `RANK_MIN_SIMILARITY` | `0.50` | 最低余弦相似度绝对门槛 |
| `RANK_RELATIVE_THRESHOLD` | `0.06` | 相对截断：只保留与最高分差距在此范围内的结果 |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding 模型 |
| `EMBEDDING_DIM` | `384` | 向量维度 |

---

## 技术栈

| 类别 | 技术选型 |
|------|---------|
| **后端框架** | FastAPI 0.111 + Uvicorn |
| **数据库** | SQLite（WAL 模式）+ SQLAlchemy 2.0 异步 ORM |
| **向量检索** | FAISS-CPU（IndexFlatIP + L2 归一化 = 余弦相似度） |
| **Embedding** | Sentence-Transformers `paraphrase-multilingual-MiniLM-L12-v2`（384 维） |
| **语音转写** | OpenAI Whisper（small 模型，永久缓存） |
| **视频处理** | FFmpeg（流复制无损切片 / libx264+aac 重编码） |
| **AI 服务** | DeepSeek / Qwen（OpenAI 兼容格式，支持双 provider 并发） |
| **前端** | 单文件 HTML5 + Vanilla JS + CSS3（无框架） |
| **字幕获取** | yt-dlp（TED 演讲字幕自动下载） |

---

## AI 接手指南

> 本节专为 AI 编程助手（Claude Code、OpenAI Codex、Copilot 等）和新加入的开发工程师设计。
> 接手本项目前请通读此节，了解关键设计决策、代码路径、数据流、已知陷阱。

### 一句话理解项目

DramaLingo = 一个**影视台词知识库 + 语义检索引擎**：用户输入中文描述 → 系统从数万条预处理、打标、向量化的影视台词中找出最匹配的地道英文表达 → 连同视频片段一起返回。

### 项目状态（截至 2026-06-03）

MVP 阶段，主流程已完整打通：字幕入库 → 知识节点 → 视频切片 → 语义检索。已完成功能测试，服务可正常启动。

**最近完成的工作**：
- 修复纯中文 SRT 解析为 0 行的 bug（`_parse_srt_text_only`）
- 新增 L2 P5 依存性介词惩罚
- 新增 G1 括号舞台指示过滤、G2 自我介绍黑名单
- 新增 `deprecated_analyzer.py` 弃用节点分析模块（11 种语言模式识别）
- 5 个降级路径全部挂上 `_trigger_deprecated_analysis()` 钩子
- 修复 TF-IDF 二次评分中的 O(n^2) 性能问题（`quality_filter.py`）
- 新增任务持久化机制（`persistent_tasks` 表），进程重启后自动标记中断任务
- 新增搜索反馈 API（`POST /search/feedback`），为排序调优积累数据

### 核心数据流（必读）

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                   数据入库流程                            │
                    │                                                          │
  .srt/.ass 文件 ──→│  subtitle_parser.py                                      │
                    │     ↓                                                    │
                    │  quality_filter.py  ←── 5 道串行筛选                      │
                    │     ↓  (passed 行)                                       │
                    │  写入 subtitles_raw 表                                   │
                    │     ↓  (调用 sync-passed)                                │
                    │  embedder.py ──→ deduplicator.py ──→ ai_tagger.py        │
                    │     ↓                                                    │
                    │  写入 knowledge_nodes 表 + FAISS 索引                    │
                    └─────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────────────────┐
                    │                   视频切片流程                            │
                    │                                                          │
  视频文件 ────────→│  whisper_service.py（转写，永久缓存）                      │
                    │     ↓                                                    │
                    │  clip_pipeline.py: merge_segments()（合并零散片段）        │
                    │     ↓                                                    │
                    │  aligner.py: align_seq()（三信号序列对齐）                │
                    │     ↓                                                    │
                    │  clip_pipeline.py: 上下文窗口 → Snap → FFmpeg 切片       │
                    │     ↓                                                    │
                    │  写入 clips 表，更新 knowledge_nodes.has_any_clip         │
                    └─────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────────────────┐
                    │                   用户检索流程                            │
                    │                                                          │
  中文查询 ────────→│  query_engine.py: expand_query()（DeepSeek 意图扩展）      │
                    │     ↓                                                    │
                    │  三路 FAISS 检索（zh_sim + en_tr_sim + en_sim）           │
                    │     ↓                                                    │
                    │  compute_final_score()（六信号加权排序）                   │
                    │     ↓                                                    │
                    │  关联 clips → 返回 JSON 结果                             │
                    └─────────────────────────────────────────────────────────┘
```

### 关键代码路径（含文件行号）

**字幕入库**（最常被修改）：
```
POST /subtitles/upload-bilingual
  → routers/subtitles.py: upload_bilingual()
  → services/subtitle_pipeline.py: process_subtitle_batch()
    → subtitle_parser.py: parse_bilingual_subtitles()   ← 双语对齐解析
    → quality_filter.py: filter_subtitles()              ← 5 道质量门
    → embedder.py: encode_texts()                        ← 向量化 (384 维)
    → quality_filter.py: compute_contextual_scores()     ← 语境二次评分
    → deduplicator.py: check_duplicate()                 ← 编辑距离+FAISS 去重
    → ai_tagger.py: batch_tag_and_translate()            ← AI 打标 (10 条/批)
    → faiss_index.py: add_to_index()                     ← 写 FAISS 索引
```

**字幕 → 知识节点同步**（易忘的中间步骤）：
```
POST /knowledge/sync-passed
  → routers/knowledge.py: sync_passed_subtitles()
  — 注意：upload 接口只写 subtitles_raw，不自动同步知识库！
    需要手动调 sync-passed 或在批次上传时勾选"自动同步"
```

**语义检索**：
```
GET /search?q=...
  → routers/search.py: semantic_search()
    1. encode_single(q)                           ← 查询向量化
    2. detect_tag_hints(q)                         ← 中文场景/情绪关键词
    3. expand_query(q)                             ← DeepSeek 意图扩展（异步）
    4. search_similar(en_index, query_vec)          ← 信号1: en_sim
    5. search_similar(zh_index, zh_query_vec)       ← 信号2: zh_sim
    6. search_similar(en_index, en_candidate_vecs)  ← 信号3: en_tr_sim (多路取 max)
    7. compute_final_score(...)                     ← 六信号加权排序
    8. 关联 clips → 返回
```

**视频切片**：
```
POST /clips/generate
  → routers/clips.py: generate_clips_endpoint()
  → services/clip_pipeline.py: generate_clips()
    → whisper_service.py: transcribe_video()         ← Whisper 转写（有缓存）
    → clip_pipeline.py: merge_transcript_segments()  ← 零散片段合并
    → aligner.py: align_seq()                        ← 三信号序列对齐
    → clip_pipeline.py: _find_context_window()       ← 上下文窗口
    → clip_pipeline.py: _snap_to_boundary()          ← 端点吸附
    → clip_pipeline.py: _ffmpeg_cut()                ← FFmpeg 切片
```

### 核心算法速查

| 算法 | 文件 | 公式 / 逻辑 |
|------|------|-------------|
| **质量评分** | `quality_filter.py` | `score = (缩略语×0.30 + 词汇密度×0.25 + 短语动词×0.25 + 句型×0.20) × pragmatic_factor` |
| **语用完整性** | `quality_filter.py` | `factor ∈ [0.6, 1.0]`，5 类惩罚叠加：截断-0.30、续接-0.25、悬挂条件-0.20、因果回指-0.15、依存介词-0.15 |
| **语境二次评分** | `quality_filter.py` | `contextual = TF-IDF独特性 × 0.40 + KB新颖度 × 0.60` |
| **去重** | `deduplicator.py` | 粗筛: 编辑距离 < 0.15 → 精筛: FAISS Top-5 余弦 ≥ 0.95 |
| **序列对齐** | `aligner.py` | `score = lev × 0.55 + emb_cos × 0.35 + order × 0.10` (窗口 ±30s) |
| **综合排序** | `query_engine.py` | `final = zh_sim×0.40 + en_tr_sim×0.25 + en_sim×0.15 + tag×0.12 + quality×0.06 + occ×0.02` |

### 数据库快速上手

```bash
# 查看数据库文件（在项目根目录）
sqlite3 dramalingo.db ".tables"

# 关键查询：知识节点统计
sqlite3 dramalingo.db "SELECT is_active, COUNT(*) FROM knowledge_nodes GROUP BY is_active;"

# 查看最近弃用的节点
sqlite3 dramalingo.db "SELECT id, en_text, quality_score FROM knowledge_nodes WHERE is_active=0 ORDER BY id DESC LIMIT 20;"

# 查看中断的任务（进程异常退出后）
sqlite3 dramalingo.db "SELECT task_id, description, status, created_at FROM persistent_tasks WHERE status='interrupted';"

# 查看搜索反馈统计
sqlite3 dramalingo.db "SELECT action, COUNT(*) FROM search_feedback GROUP BY action;"
```

### 已知陷阱（坑）

| 编号 | 描述 | 位置 | 解决方案 |
|------|------|------|---------|
| T1 | `_parse_srt()` 遇到纯中文 SRT 返回 0 行 | `subtitle_parser.py` | 使用 `_parse_srt_text_only()` 解析中文 SRT |
| T2 | 上传字幕后知识库没有新节点 | `subtitle_pipeline.py` | 字幕先存 subtitles_raw，需调 `sync-passed` 才同步知识库 |
| T3 | FAISS 索引与 DB 不一致 | `main.py: lifespan` | 启动时自动校验并重建，手动重建调 `rebuild_index_from_db` |
| T4 | Windows 启动报 WinError 10013 端口占用 | `start.ps1` | 脚本已自动 kill 占用端口的旧进程 |
| T5 | PowerShell 执行 Python 多行 `-c` 语法报错 | Windows 特有 | 写临时 `.py` 文件再运行，避免 PowerShell 引号嵌套 |
| T6 | `asyncio.create_task()` 必须在事件循环中调用 | `knowledge.py` | `_trigger_deprecated_analysis` 已用 try/except 包裹，失败只记 warning |
| T7 | SQLite WAL 下写事务会锁定，避免大事务 | `database.py` | 批量操作分块（`_CHUNK = 200`），每块单独 commit |
| T8 | AI 打标每次创建新 httpx.AsyncClient | `ai_tagger.py` | 高并发时 TCP 握手开销大，待优化为模块级连接池 |
| T9 | 翻译缓存是 FIFO 而非 LRU | `query_engine.py` | 高频查询可能被低频挤掉，待改用 `OrderedDict` LRU |

### 代码规范

- **异步**：所有数据库操作用 `async with SessionLocal() as sess`（`get_session` 仅供路由依赖注入使用）
- **任务调度**：长耗时操作通过 `submit_task(coroutine, description, task_id)` 提交，进度通过 `update_task_progress()` 上报。任务生命周期自动持久化到 `persistent_tasks` 表
- **fire-and-forget**：轻量异步副作用用 `asyncio.create_task()`，包裹在 try/except 中，不允许失败中断主流程
- **增量迁移**：新增数据库列在 `main.py: lifespan` 中用 `ALTER TABLE ... ADD COLUMN`（幂等，try/except 忽略列已存在错误）；不使用 Alembic
- **新增表**：直接在 `models/` 下新建 ORM 文件，在 `models/__init__.py` 和 `database.py: init_db()` 中注册导入，`Base.metadata.create_all` 自动建表
- **注释语言**：中文（面向中国团队），代码标识符英文
- **错误处理**：DB 持久化类的辅助操作（如 `_db_update_status`）内部 try/except 吞掉异常，只 log warning，不允许辅助功能失败影响主流程

### 模块依赖关系

```
routers/          →  services/          →  models/
  subtitles.py         subtitle_pipeline.py    subtitle.py
  knowledge.py         quality_filter.py       knowledge_node.py
  search.py            embedder.py             node_source.py
  clips.py             deduplicator.py         clip.py
  videos.py            ai_tagger.py            video_file.py
                       faiss_index.py          whisper_transcript.py
                       query_engine.py         persistent_task.py
                       clip_pipeline.py        search_feedback.py
                       whisper_service.py
                       aligner.py
                       ted_fetcher.py

tasks/task_runner.py  ← 被所有 routers/ 和部分 services/ 调用
config.py             ← 被所有模块引用
database.py           ← 提供 SessionLocal, get_session, Base, engine
```

**关键约束**：`services/` 之间尽量不互相依赖。例外：
- `deduplicator.py` 被 `aligner.py` 引用（共享 `normalize_text`）
- `clip_pipeline.py` 调用 `whisper_service.py` 和 `aligner.py`
- `subtitle_pipeline.py` 编排所有 services

### 环境依赖检查

```powershell
# 检查 Python 版本
python --version  # 需要 3.11+

# 检查 FFmpeg
ffmpeg -version | Select-Object -First 1

# 检查虚拟环境
.venv\Scripts\python.exe -c "import faiss, sentence_transformers, whisper; print('OK')"

# 检查 .env 配置
Select-String "AI_API_KEY" .env | Select-String -NotMatch "sk-xxx"

# 验证所有模块可导入
.venv\Scripts\python.exe -c "from main import app; print(f'Routes: {len(app.routes)}')"
```

### 未完成 / 待改进

| 优先级 | 事项 |
|--------|------|
| P1 | `pronoun_dependent` 模式建议新增 L2 P6 惩罚 −0.10（`deprecated_analyzer.py` 已识别，弃用节点中占比 ~12%） |
| P1 | `ai_tagger.py` 中 `httpx.AsyncClient` 每次创建新连接，应改为模块级连接池 |
| P1 | `query_engine.py` 翻译缓存用 FIFO 淘汰，应改为 LRU（`OrderedDict` 或 `functools.lru_cache`） |
| P2 | 管理后台暂无"弃用分析"可视化卡片，需在 `admin.html` 中接入 `GET /knowledge/analyze-deprecated` |
| P2 | 前端 `user.html` 需接入 `POST /search/feedback` 记录用户点击/播放行为 |
| P2 | `rule_based_tag()` 降级标签全部默认 daily/neutral，影响 tag_score 召回，可加轻量关键词规则 |
| P3 | `_parse_srt_text_only` 有 fallback 逻辑，可统一到 `parse_bilingual_subtitles` 的入参控制 |
| P3 | Clip 模型有 `thumb_url` 字段但大部分切片未生成缩略图 |

---

## 未来优化路线

> 本节按三大方向梳理 DramaLingo 下一阶段可落地的技术优化，每项标注优先级（P0 紧急 / P1 重要 / P2 改进 / P3 远景）、对应代码位置、以及预期收益。

### 一、语言学视角的字幕评分机制优化

当前评分系统（`services/quality_filter.py`）以**正则 + 手工特征**为核心，F1 已达 97.5%，但在语用层面仍有盲区。以下优化可进一步减少人工弃用成本。

#### 1.1 语用自足性增强（L2 层）

| 编号 | 优化项 | 描述 | 优先级 | 代码位置 |
|------|--------|------|--------|---------|
| L2-P6 | **第三人称代词开头惩罚** | `He/She/They/It + 动词` 开头的句子通常需要先行词（指代不明），建议施加 −0.10 惩罚。`deprecated_analyzer.py` 已识别此模式（`pronoun_dependent`），弃用节点中占比约 12% | P1 | `quality_filter.py: _compute_pragmatic_factor()` |
| L2-P7 | **话语标记词开头降权** | `Well, ... / I mean, ... / You know what, ...` 等话语标记词（discourse markers）作为话轮起始时本身有教学价值，但若句子在标记词之后被截断则信息量极低，建议检测「标记词 + 不完整后续」模式 | P2 | `quality_filter.py: 新增正则` |
| L2-P8 | **省略回复检测** | `I did. / She might. / They would.` 等省略回复（elliptical response）语义完全依赖前文问句，独立使用无教学价值 | P2 | `quality_filter.py: 新增正则` |

#### 1.2 口语真实性评分维度扩展（Authenticity 层）

| 编号 | 优化项 | 描述 | 优先级 |
|------|--------|------|--------|
| A-1 | **语域检测（Register Detection）** | 当前评分未区分正式/非正式语域。可引入 bigram 特征区分学术语体（*furthermore / consequently / it is imperative*）与口语体（*gonna / figure out / no way*），对非口语语域施加降权 | P2 |
| A-2 | **情感强度因子** | 包含强情感词汇或感叹结构的句子（*I can't believe... / That's absolutely insane!*）通常具有更高的教学场景唤起度，可作为加分项 | P3 |
| A-3 | **语法复杂度评分** | 引入从句层数、非限定分句（非谓语动词短语）等句法复杂度指标。适度复杂的句子（1–2 层从句）教学价值最高，过简（单词感叹）和过复杂（嵌套 3 层以上）均降权 | P3 |
| A-4 | **搭配强度评分（Collocational Strength）** | 使用 PMI（Pointwise Mutual Information）或预训练词频表衡量搭配地道度。*make a decision* 比 *do a decision* 更地道，高搭配强度加分 | P3 |

#### 1.3 评分体系架构升级

| 编号 | 优化项 | 描述 | 优先级 |
|------|--------|------|--------|
| S-1 | **可学习权重替代手工权重** | 当前四维权重（0.30/0.25/0.25/0.20）和 L2 惩罚值为人工设定。可用逻辑回归或轻量 GBDT 在标注数据上学习最优权重，预期 F1 提升 1–2% | P2 |
| S-2 | **分剧种评分校准** | 不同内容类型（tv/movie/talk）的口语特征分布差异显著。TED 演讲偏正式，sitcom 偏口语。可为每种 `content_type` 维护独立阈值或权重组 | P2 |
| S-3 | **引入 LLM 二次审核** | 对 `quality_score ∈ [0.25, 0.35]` 的边界区间节点（约占总量 8%），调用 LLM 做语用价值判定，减少阈值附近的误杀和漏杀 | P3 |

#### 1.4 黑名单与正则维护

| 编号 | 优化项 | 描述 | 优先级 |
|------|--------|------|--------|
| B-1 | **黑名单数据驱动扩展** | 基于 `deprecated_analyzer` 周期性报告中的 `unclassified` 高频节点，自动提取候选黑名单短语，管理员审批后入库 | P1 |
| B-2 | **黑名单分级** | 当前黑名单为「出现即拒绝」的硬过滤。可引入软黑名单（常见但仍有教学价值的表达，如 *excuse me*），仅在特定语境（搭配其他台词成完整对话）下保留 | P3 |

---

### 二、视频切片质量提升

当前切片流水线（`services/clip_pipeline.py`）已实现上下文窗口扩展、端点 Snap、置信度门控。以下优化方向可进一步提升切片的观看体验和教学可用性。

#### 2.1 切片边界精度

| 编号 | 优化项 | 描述 | 优先级 | 代码位置 |
|------|--------|------|--------|---------|
| C-1 | **基于音频能量的端点 Snap** | 当前 Snap 依赖 Whisper 片段间隙中点，仅反映文字间的静默。可用 `librosa`/`pydub` 计算短时能量（RMS），在低能量区间（静音段中点）吸附切点，避免切入背景音乐或笑声中途 | P1 | `clip_pipeline.py: _snap_to_boundary()` |
| C-2 | **说话人分离（Speaker Diarization）** | 当前合并逻辑依赖 gap + 大小写信号推断说话人切换。引入 `pyannote-audio` 或 Whisper 的 `--diarize` 模式，可精确识别多说话人边界，防止合并跨角色对白、切片覆盖错误角色 | P2 | `clip_pipeline.py: merge_transcript_segments()` |
| C-3 | **场景切换检测（Scene Change Detection）** | 在视频帧层面检测场景转换点（色调突变/黑屏帧），作为切片边界的硬约束。避免切片跨越镜头切换导致观看不连贯 | P3 | `clip_pipeline.py: _find_context_window()` |

#### 2.2 切片内容质量

| 编号 | 优化项 | 描述 | 优先级 |
|------|--------|------|--------|
| C-4 | **音频质量预筛** | 部分视频片段存在背景噪声过大、多人同时说话的情况。可在切片前计算 SNR（信噪比），低于阈值的切片标记 `clip_quality=low`，管理后台可按此筛选 | P2 |
| C-5 | **字幕嵌入切片** | 当前切片为纯视频。可选在切片内烧录中英文双语字幕（hardsub），或输出 WebVTT 软字幕轨道，提升学习体验 | P2 |
| C-6 | **切片预览动图** | 当前缩略图为单帧。可生成 3–5 秒 GIF/WebP 动图作为预览，用户在搜索结果页即可预判内容相关性 | P3 |

#### 2.3 切片流水线效率

| 编号 | 优化项 | 描述 | 优先级 |
|------|--------|------|--------|
| C-7 | **FFmpeg 合并切片命令** | 当前每个切片独立调用一次 FFmpeg。对同一视频源的批量切片，可用 `-segment_list` 或 `concat` 协议一次性完成，减少 I/O 开销和进程启动成本 | P1 |
| C-8 | **GPU 加速 Whisper** | 当前使用 CPU 版 Whisper-small。可选升级为 `faster-whisper`（CTranslate2 后端）或 NVIDIA GPU 版，转写速度提升 5–10× | P1 |
| C-9 | **增量切片** | 当前 `generate_clips` 对同一视频重跑时会跳过已有切片，但仍重做对齐计算。可缓存对齐结果到 DB，仅对新增知识节点做增量对齐 | P2 |

---

### 三、字幕-视频对齐准确性与效率

对齐模块（`services/aligner.py`）是连接字幕知识库和视频切片的核心枢纽。当前三信号融合（Levenshtein + Embedding + 顺序奖励）在大部分场景下表现良好，但存在以下可优化空间。

#### 3.1 对齐信号增强

| 编号 | 优化项 | 描述 | 优先级 | 代码位置 |
|------|--------|------|--------|---------|
| A-1 | **声学特征对齐（Audio Fingerprint）** | 在文本对齐的基础上，引入音频 MFCC 或 Chromagram 特征作为第四信号。对文本相似度差异小的候选（margin < 0.08 的 ambiguous 情况），用声学特征打破平局 | P2 | `aligner.py: align_seq()` |
| A-2 | **时间戳加权余弦** | 当前 Embedding 余弦计算不考虑时间距离。可引入时间衰减因子：`adjusted_cos = cos_sim × exp(-α × abs(time_diff))`，让时间上更接近的候选获得自然优势 | P1 | `aligner.py: align_seq() 评分公式` |
| A-3 | **N-gram 重叠率作为补充信号** | Levenshtein 是字符级相似度，对词序变化敏感。补充 word-level N-gram（bigram/trigram）重叠率，更好处理同义改写（如 *I can't do this* vs *I cannot do this*）。当前 `normalize_for_alignment` 已展开部分缩写，但未覆盖全部变体 | P2 | `aligner.py: 新增 _ngram_overlap()` |

#### 3.2 对齐策略优化

| 编号 | 优化项 | 描述 | 优先级 |
|------|--------|------|--------|
| A-4 | **全局最优对齐替代贪心** | 当前采用逐节点贪心匹配（first-come-first-served），已匹配的 Whisper 段被排除。可改用匈牙利算法（Hungarian Algorithm）或动态规划求解全局最优一对一匹配，避免一个节点「抢走」后续节点的最佳候选 | P2 |
| A-5 | **多轮对齐 + 回填** | 第一轮贪心完成后，对 `ambiguous` 和 `unmatched` 节点做二次对齐：放宽窗口至 ±60s，允许复用 confidence 差距 > 0.15 的已占用片段 | P1 |
| A-6 | **自适应窗口大小** | 当前固定 ±30s 窗口。对剧集前几分钟（片头、previously on）和最后几分钟（片尾曲），字幕时间戳漂移更大，可自适应放宽窗口至 ±45s | P2 |

#### 3.3 对齐效率优化

| 编号 | 优化项 | 描述 | 优先级 |
|------|--------|------|--------|
| A-7 | **候选预索引** | 当前每个节点在 `transcripts` 列表上做线性扫描过滤窗口。可预建时间区间索引（`IntervalTree` 或按 `start_time` 排序后二分查找），将候选搜索从 O(N) 降至 O(log N + K) | P1 |
| A-8 | **批量 Embedding 计算** | 当前逐个节点计算 `_cosine()`。可在对齐前一次性 encode 所有 Whisper 段文本，构建 FAISS 临时索引，用 `search()` 一次性召回 Top-K 候选，减少重复 encode 开销 | P1 |
| A-9 | **长度预过滤加速** | 当前 `_length_ratio()` 对每个候选计算 `len(text.split())`。可预计算所有 Whisper 段词数并缓存为数组，避免重复 split | P2 |

#### 3.4 对齐质量监控

| 编号 | 优化项 | 描述 | 优先级 |
|------|--------|------|--------|
| A-10 | **对齐质量报告** | 每次切片流水线完成后自动生成对齐质量报告：matched/ambiguous/unmatched 分布、平均置信度、时间偏差分布。可作为 `POST /clips/report/{show}/{ep}` 的扩展字段 | P1 |
| A-11 | **人工校正回馈** | 在管理后台为每个 `ambiguous` 切片提供人工确认/纠正界面。校正数据可作为监督信号，用于学习更优的对齐权重（联动 S-1 可学习权重） | P3 |

---

### 优化路线图（推荐实施顺序）

```
Phase 1（近期·1–2 周）— 低成本高收益
  ├── L2-P6 第三人称代词惩罚（已有分析器支撑）
  ├── B-1  黑名单数据驱动扩展
  ├── A-2  时间戳加权余弦
  ├── A-7  候选预索引（IntervalTree）
  ├── A-8  批量 Embedding 计算
  └── C-7  FFmpeg 合并切片命令

Phase 2（中期·3–4 周）— 架构级提升
  ├── C-1  基于音频能量的端点 Snap
  ├── C-8  GPU 加速 Whisper / faster-whisper
  ├── A-5  多轮对齐 + 回填
  ├── S-1  可学习评分权重
  ├── S-2  分剧种评分校准
  └── A-10 对齐质量报告

Phase 3（远期·1–2 月）— 深度语言学增强
  ├── C-2  说话人分离（Speaker Diarization）
  ├── A-1  声学特征对齐
  ├── A-4  全局最优匹配（匈牙利算法）
  ├── A-3  语域检测 + 情感强度因子
  └── S-3  LLM 二次审核
```

---

*本项目由产品 & 架构团队设计，AI 辅助开发完成。*
