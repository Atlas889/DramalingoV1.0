# CLAUDE.md — DramaLingo 项目上下文

> 本文件供 Claude Code、OpenAI Codex 等 AI 开发工具自动读取。
> 项目完整文档见 README.md。

## 项目简介

DramaLingo = **影视台词知识库 + 中文语义检索引擎**。
用户输入中文描述 → 从数万条影视台词中找到最地道的英文表达 → 返回文本+视频片段。

## 技术栈

- Python 3.11+ / FastAPI 0.111 / Uvicorn
- SQLite (WAL) + SQLAlchemy 2.0 async ORM
- FAISS-CPU (IndexFlatIP, 384维, 余弦相似度)
- Sentence-Transformers `paraphrase-multilingual-MiniLM-L12-v2`
- OpenAI Whisper (small)
- DeepSeek / Qwen (OpenAI 兼容 API)
- FFmpeg
- 前端: Vanilla HTML + JS (无框架)

## 目录结构

```
models/          — ORM 模型（10 张表）
routers/         — FastAPI 路由（6 个模块，~70 个接口）
services/        — 核心业务逻辑（14 个服务）
tasks/           — 串行异步任务调度器（asyncio.Queue + DB 持久化）
static/          — 前端 HTML（admin.html + user.html）
data/            — SQLite、FAISS、媒体、上传和任务状态数据
tests/            — 集成测试与测试夹具
scripts/          — 启动、维护和诊断脚本
docs/reports/     — 产品测评报告和生成脚本
config.py        — 全局配置参数（所有算法阈值/权重集中管理）
database.py      — DB 引擎/Session/Base/迁移
main.py          — 应用入口/lifespan/增量迁移/路由注册
```

## 核心数据流

```
字幕入库: .srt → 解析 → 5道质量筛选 → Embedding → 去重 → AI打标 → knowledge_nodes + FAISS
视频切片: 视频 → Whisper转写 → 片段合并 → 序列对齐 → 置信度门控 → 上下文窗口 → FFmpeg切片 → clips
用户检索: 中文查询 → 意图扩展 → 三路FAISS检索 → 六信号加权排序 → 返回结果+视频
```

## 关键算法公式

- **质量评分**: `score = (缩略语×0.30 + 词汇密度×0.25 + 短语动词×0.25 + 句型×0.20) × pragmatic_factor[0.6~1.0]`
- **去重**: 编辑距离 < 0.15 (粗筛) → FAISS 余弦 ≥ 0.95 (精筛)
- **序列对齐**: `score = lev×0.55 + emb_cos×0.35 + order×0.10`
- **综合排序**: `final = zh_sim×0.40 + en_tr_sim×0.25 + en_sim×0.15 + tag×0.12 + quality×0.06 + occ×0.02`

## 开发约定

- **异步 DB**: 用 `async with SessionLocal() as sess`，路由中用 `Depends(get_session)`
- **任务调度**: 长耗时操作用 `submit_task(coro, description, task_id)` 提交到串行队列
- **增量迁移**: 新列在 `main.py: lifespan` 用 `ALTER TABLE ADD COLUMN`（try/except 幂等），新表在 models/ 建 ORM + database.py 注册导入
- **注释语言**: 中文（面向中国团队），代码标识符英文
- **错误处理**: 辅助操作（DB 持久化、日志等）内部 try/except，只 log warning，不影响主流程

## 常用命令

```powershell
# 启动（Windows）
.venv\Scripts\activate; uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 验证导入
.venv\Scripts\python.exe -c "from main import app; print(f'Routes: {len(app.routes)}')"

# 运行冒烟测试
.venv\Scripts\python.exe tests\integration\smoke_test.py
```

## 已知陷阱

1. 上传字幕后知识库没新节点 → 需调 `POST /knowledge/sync-passed` 才同步
2. FAISS 索引与 DB 不一致 → 启动时自动校验重建，手动调 `rebuild_index_from_db`
3. SQLite WAL 下写事务会锁，批量操作必须分块（每块 200 条）单独 commit
4. `asyncio.create_task()` 必须在事件循环内调用，fire-and-forget 包 try/except
5. Windows PowerShell 中 Python `-c` 多行语法容易报错，写临时 .py 文件

## 待改进项（按优先级）

- P1: 新增 L2 P6 第三人称代词惩罚（`quality_filter.py`）
- P1: `ai_tagger.py` httpx.AsyncClient 改为模块级连接池
- P1: `query_engine.py` 翻译缓存改 LRU
- P2: 前端 `user.html` 接入 `POST /search/feedback`
- P2: `rule_based_tag()` 降级标签加轻量关键词规则
- P3: Clip 缩略图生成（`thumb_url` 字段已存在但未填充）
