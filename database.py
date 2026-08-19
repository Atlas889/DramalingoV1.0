"""
DramaLingo · 数据库初始化与连接管理
SQLite + WAL 模式 + busy_timeout，异步引擎。
对应 PRD v0.9 §8.1 SQLite 并发写入策略。
"""

import logging
from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import event
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

import config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# ORM 基类
# ──────────────────────────────────────────────
class Base(DeclarativeBase):
    """所有 ORM 模型的基类"""
    pass


# ──────────────────────────────────────────────
# 异步引擎与 Session 工厂
# ──────────────────────────────────────────────
DATABASE_URL = f"sqlite+aiosqlite:///{config.DB_PATH}"

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
    poolclass=NullPool,  # SQLite 不支持 pool_size/max_overflow，使用 NullPool
)

# 在每个原始 DBAPI 连接建立时设置 WAL 和 busy_timeout
@event.listens_for(engine.sync_engine, "connect")
def _set_sqlite_pragma(dbapi_conn, connection_record):
    """
    PRD §8.1：开启 WAL 模式 + busy_timeout=30s，
    允许读写并发、缓解偶发锁冲突。
    """
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    row = cursor.fetchone()
    if row and row[0] != "wal":
        logger.warning("WAL 模式未能开启，当前日志模式: %s", row[0])
    else:
        logger.debug("WAL 模式已开启")
    cursor.execute(f"PRAGMA busy_timeout={config.SQLITE_BUSY_TIMEOUT}")
    cursor.close()


SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ──────────────────────────────────────────────
# 数据库初始化
# ──────────────────────────────────────────────
async def _migrate_columns() -> None:
    """为已有表增量添加新列（SQLite 不支持 IF NOT EXISTS，用 try/except 屏蔽重复错误）。"""
    migrations = [
        "ALTER TABLE subtitles_raw ADD COLUMN en_name TEXT NOT NULL DEFAULT ''",
        "ALTER TABLE subtitles_raw ADD COLUMN subtitle_code TEXT",
        "ALTER TABLE video_files ADD COLUMN subtitle_code TEXT",
    ]
    async with engine.begin() as conn:
        for sql in migrations:
            try:
                await conn.execute(__import__("sqlalchemy").text(sql))
                logger.info("迁移成功: %s", sql)
            except Exception as e:
                if "duplicate column name" in str(e).lower():
                    logger.debug("列已存在，跳过: %s", sql)
                else:
                    logger.warning("迁移跳过（%s）: %s", e, sql)


async def init_db() -> None:
    """
    创建全部表并建立索引。
    同时确保 media/ 和 indexes/ 目录存在。
    """
    # 确保运行时目录存在
    config.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 导入所有模型，确保 Base.metadata 能看到全部表定义
    import models.subtitle          # noqa: F401
    import models.knowledge_node    # noqa: F401
    import models.node_source       # noqa: F401
    import models.clip              # noqa: F401
    import models.video_file        # noqa: F401
    import models.whisper_transcript  # noqa: F401
    import models.subtitle_job      # noqa: F401
    import models.persistent_task   # noqa: F401
    import models.search_feedback   # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await _migrate_columns()

    logger.info("数据库初始化完成，全部表已创建")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖注入用的 session 生成器"""
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
