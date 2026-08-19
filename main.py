"""
DramaLingo · FastAPI 应用入口
对应 PRD v0.9 §4.3 两个独立入口：
  - 管理后台 /admin（Basic Auth 保护）
  - 用户端 /（公开访问）

lifespan 中完成：
  1. 数据库初始化（init_db）
  2. 启动串行任务队列 worker
"""

import logging
import os
import secrets
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

import config
from config import validate_config
from database import init_db
from tasks.task_runner import (
    start_worker, stop_worker, get_task_status, get_all_tasks,
)

# ──────────────────────────────────────────────
# 日志配置
# ──────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Lifespan（启动 / 关闭）
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("DramaLingo 启动中...")

    # 0. 校验配置参数
    validate_config()
    logger.info("配置参数校验通过")

    # 1. 初始化数据库（创建全部表 + 确保目录存在）
    await init_db()
    logger.info("数据库初始化完成")

    # 1b. 增量迁移：knowledge_nodes 添加 zh_embedding 列
    #     背景：初版只有英文 Embedding，后续引入中文查询支持时补加
    #     SQLite 不能 ALTER TABLE ADD COLUMN UNIQUE，须两步：先加列，再建索引
    try:
        from sqlalchemy import text
        from database import SessionLocal as _SL
        async with _SL() as _sess:
            await _sess.execute(text("ALTER TABLE knowledge_nodes ADD COLUMN zh_embedding BLOB"))
            await _sess.commit()
        logger.info("已添加 zh_embedding 列")
    except Exception:
        pass  # 列已存在时忽略（SQLite: "duplicate column name"）

    # 1b-2. knowledge_nodes 添加 code 列 + 唯一索引
    #       code 格式：FRIENDS-S01E01-00154520（剧名-集号-时间戳），用于防重复入库
    try:
        from sqlalchemy import text
        from database import SessionLocal as _SL
        async with _SL() as _sess:
            await _sess.execute(text("ALTER TABLE knowledge_nodes ADD COLUMN code TEXT"))
            await _sess.commit()
        logger.info("已添加 code 列")
    except Exception:
        pass  # 列已存在时忽略
    try:
        from sqlalchemy import text
        from database import SessionLocal as _SL
        async with _SL() as _sess:
            await _sess.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_kn_code ON knowledge_nodes(code)"
                " WHERE code IS NOT NULL"  # 部分索引：code IS NULL 的历史行不参与约束
            ))
            await _sess.commit()
        logger.info("已建立 code 唯一索引")
    except Exception:
        pass  # 索引已存在时忽略

    # 1c. clips 表添加 clip_code 列 + 唯一索引
    #     clip_code 格式：CLP-FR-S1-A3F8C2，用于切片 URL 生成和去重
    try:
        from sqlalchemy import text as _text
        from database import SessionLocal as _SL_cc
        async with _SL_cc() as _s:
            await _s.execute(_text("ALTER TABLE clips ADD COLUMN clip_code TEXT"))
            await _s.commit()
        logger.info("已添加 clips.clip_code 列")
    except Exception:
        pass
    try:
        from sqlalchemy import text as _text
        from database import SessionLocal as _SL_cc2
        async with _SL_cc2() as _s:
            await _s.execute(_text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_clips_code ON clips(clip_code)"
                " WHERE clip_code IS NOT NULL"
            ))
            await _s.commit()
        logger.info("已建立 clips.clip_code 唯一索引")
    except Exception:
        pass

    # 1d. 为历史存量 clips 补填 clip_code（随机 6 位十六进制，保证全局唯一）
    #     仅在有 clip_code IS NULL 的行时生效，最多处理 2000 条
    try:
        import secrets as _secrets
        from sqlalchemy import text as _text
        from database import SessionLocal as _SL_bk
        async with _SL_bk() as _s:
            rows = (await _s.execute(_text(
                "SELECT id, show_name, episode FROM clips WHERE clip_code IS NULL LIMIT 2000"
            ))).all()
            for row in rows:
                sn = (row.show_name or "XX")[:2].upper()
                ep = (row.episode or "00")[:2].upper()
                rand6 = _secrets.token_hex(3).upper()
                code = f"CLP-{sn}-{ep}-{rand6}"
                await _s.execute(_text(
                    "UPDATE clips SET clip_code=:code WHERE id=:id"
                ), {"code": code, "id": row.id})
            await _s.commit()
        logger.info(f"已为 {len(rows)} 条历史切片补填 clip_code")
    except Exception as _e:
        logger.warning(f"历史 clip_code 补填失败（不影响启动）: {_e}")

    # 1f. knowledge_nodes 添加 speech_act 列
    #     背景：引入 25 种言语行为打标体系时补加，默认值 neutral_statement 确保历史节点合法
    try:
        from sqlalchemy import text as _text
        from database import SessionLocal as _SL_sa
        async with _SL_sa() as _s:
            await _s.execute(_text(
                "ALTER TABLE knowledge_nodes ADD COLUMN speech_act TEXT DEFAULT 'neutral_statement'"
            ))
            await _s.commit()
        logger.info("已添加 knowledge_nodes.speech_act 列")
    except Exception:
        pass  # 列已存在时忽略

    # 1e. 为三张核心表添加 content_type 列（tv / movie / talk 内容大类）
    #     默认值 'tv' 确保历史数据不破坏约束
    _ct_migrations = [
        "ALTER TABLE subtitles_raw ADD COLUMN content_type TEXT NOT NULL DEFAULT 'tv'",
        "ALTER TABLE knowledge_nodes ADD COLUMN content_type TEXT NOT NULL DEFAULT 'tv'",
        "ALTER TABLE video_files ADD COLUMN content_type TEXT NOT NULL DEFAULT 'tv'",
    ]
    for _sql in _ct_migrations:
        try:
            from sqlalchemy import text as _text
            from database import SessionLocal as _SL2
            async with _SL2() as _s:
                await _s.execute(_text(_sql))
                await _s.commit()
        except Exception:
            pass  # 列已存在时忽略

    # 2. 预热 Embedding 模型（首次触发下载，避免第一次请求超时）
    try:
        import asyncio as _asyncio
        from services.embedder import _get_model
        await _asyncio.get_running_loop().run_in_executor(None, _get_model)
        logger.info("Embedding 模型预热完成")
    except Exception as e:
        logger.warning(f"Embedding 模型预热失败（不影响启动）: {e}")

    # 3. 预加载 FAISS 索引（en + zh），并校验与 DB 的一致性，不一致时自动重建
    try:
        from services.faiss_index import (
            get_global_index, get_global_zh_index, rebuild_index_from_db,
        )
        from sqlalchemy import func, select as _sa_select
        from models.knowledge_node import KnowledgeNode as _KN
        from database import SessionLocal as _SL

        en_idx, _ = get_global_index()
        zh_idx, _ = get_global_zh_index()
        logger.info(
            f"FAISS 索引预加载完成: en={en_idx.ntotal if en_idx else 0} 条，"
            f"zh={zh_idx.ntotal if zh_idx else 0} 条"
        )

        # R1：校验 index 与 DB 节点数是否一致，进程上次崩溃时可能导致不一致
        async with _SL() as _sess:
            db_count_row = await _sess.execute(
                _sa_select(func.count()).select_from(_KN).where(_KN.is_active == True)
            )
            db_count = db_count_row.scalar_one()

        index_count = en_idx.ntotal if en_idx else 0
        if db_count != index_count:
            logger.warning(
                f"FAISS index 与 DB 不一致（index={index_count}, db={db_count}），"
                "自动重建 index..."
            )
            async with _SL() as _sess:
                await rebuild_index_from_db(_sess, force_full=True)
            logger.info("FAISS index 重建完成")
        else:
            logger.info(f"FAISS index 与 DB 一致（{db_count} 条节点）")

    except Exception as e:
        logger.warning(f"FAISS 索引预加载/校验失败（不影响启动）: {e}")

    # 4. 启动串行任务队列 worker
    await start_worker()
    logger.info("任务队列 Worker 已启动")

    yield  # 应用运行中

    # 关闭时停止 worker
    await stop_worker()
    logger.info("DramaLingo 已关闭")


# ──────────────────────────────────────────────
# FastAPI 应用实例
# ──────────────────────────────────────────────
app = FastAPI(
    title="DramaLingo",
    description="影视表达检索引擎 — 用中文描述你想说的话，找到最地道的英文说法",
    version="0.1.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────
# Basic Auth 中间件（保护 /admin 路径）
# 对应 PRD §4.3 访问控制
# ──────────────────────────────────────────────
import base64


@app.middleware("http")
async def admin_auth_middleware(request: Request, call_next):
    """
    管理后台 Basic Auth 保护。
    /admin 开头的路径及 /static/admin.html 直接访问均需认证。
    用户端 / 公开访问。
    """
    # 直接访问 /static/admin.html 时重定向到受保护的 /admin 入口
    if request.url.path == "/static/admin.html":
        return Response(
            status_code=302,
            headers={"Location": "/admin"},
        )

    if request.url.path.startswith("/admin"):
        auth_header = request.headers.get("Authorization")
        if auth_header is None or not auth_header.startswith("Basic "):
            return Response(
                content="Authentication required",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="DramaLingo Admin"'},
            )

        try:
            decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
            username, password = decoded.split(":", 1)
        except Exception:
            return Response(
                content="Invalid credentials",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="DramaLingo Admin"'},
            )

        if not (
            secrets.compare_digest(username, config.ADMIN_USER)
            and secrets.compare_digest(password, config.ADMIN_PASS)
        ):
            return Response(
                content="Invalid credentials",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="DramaLingo Admin"'},
            )

    response = await call_next(request)
    return response


# ──────────────────────────────────────────────
# 挂载静态文件目录
# ──────────────────────────────────────────────
# 确保目录存在（StaticFiles 要求目录必须存在）
os.makedirs(config.MEDIA_DIR, exist_ok=True)
os.makedirs(config.STATIC_DIR, exist_ok=True)
os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.INDEX_DIR, exist_ok=True)

# 视频切片文件服务
app.mount("/media", StaticFiles(directory=str(config.MEDIA_DIR)), name="media")
# 前端静态文件
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


# ──────────────────────────────────────────────
# 注册路由
# ──────────────────────────────────────────────
from routers.subtitles import router as subtitles_router
from routers.subtitle_jobs import router as subtitle_jobs_router
from routers.clips import router as clips_router
from routers.knowledge import router as knowledge_router
from routers.search import router as search_router
from routers.videos import router as videos_router

app.include_router(subtitles_router)
app.include_router(subtitle_jobs_router)
app.include_router(clips_router)
app.include_router(knowledge_router)
app.include_router(search_router)
app.include_router(videos_router)


# ──────────────────────────────────────────────
# 根路由（用户端入口）
# ──────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    """用户端首页（公开访问）"""
    user_html = config.STATIC_DIR / "user.html"
    if user_html.exists():
        return FileResponse(str(user_html))
    return JSONResponse(
        content={
            "message": "DramaLingo API 运行中",
            "docs": "/docs",
            "version": "0.1.0",
        }
    )


# ──────────────────────────────────────────────
# 管理后台入口
# ──────────────────────────────────────────────
@app.get("/admin", include_in_schema=False)
async def admin():
    """管理后台首页（Basic Auth 保护）"""
    admin_html = config.STATIC_DIR / "admin.html"
    if admin_html.exists():
        return FileResponse(str(admin_html))
    return JSONResponse(
        content={
            "message": "DramaLingo 管理后台",
            "status": "ready",
        }
    )


# ──────────────────────────────────────────────
# 任务状态查询接口
# ──────────────────────────────────────────────
@app.get("/jobs/{job_id}", tags=["tasks"])
async def query_job_status(job_id: str):
    """
    查询后台任务状态。
    前端轮询此接口获取字幕处理 / 视频切片等异步任务的进度。
    """
    status = get_task_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"任务 {job_id} 不存在")
    return {"job_id": job_id, **status}


@app.get("/jobs", tags=["tasks"])
async def list_jobs():
    """列出所有任务状态（管理后台用）"""
    tasks = get_all_tasks()
    return {
        "total": len(tasks),
        "tasks": [{"job_id": k, **v} for k, v in tasks.items()],
    }


# ──────────────────────────────────────────────
# 健康检查
# ──────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health_check():
    """系统健康检查"""
    return {"status": "healthy", "version": "0.1.0"}
