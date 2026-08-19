"""
DramaLingo · 串行任务调度器（DB 持久化版）
对应 PRD v0.9 §8.1 SQLite 并发写入策略：
写任务串行调度（单 worker），读任务可并发。

使用 asyncio.Queue 实现：
- submit_task(coro) 将协程入队，返回 task_id
- 后台 worker 串行消费队列（worker=1）
- 任务状态同时存内存字典（秒级进度）和 DB 表（持久化生命周期）
- 进程重启时，DB 中 pending/running 的任务自动标记为 interrupted
"""

import asyncio
import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Coroutine, Dict, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 任务状态存储（内存字典，用于秒级进度查询）
# ──────────────────────────────────────────────
_task_store: Dict[str, Dict[str, Any]] = {}

_task_queue: Optional[asyncio.Queue] = None
_worker_task: Optional[asyncio.Task] = None


def _get_queue() -> asyncio.Queue:
    global _task_queue
    if _task_queue is None:
        _task_queue = asyncio.Queue()
    return _task_queue


# ──────────────────────────────────────────────
# DB 持久化辅助（fire-and-forget，不阻塞主流程）
# ──────────────────────────────────────────────

async def _db_create_task(task_id: str, description: str) -> None:
    """将新任务写入 persistent_tasks 表（status=pending）。失败仅 log warning，不中断主流程。"""
    try:
        from database import SessionLocal
        from models.persistent_task import PersistentTask
        async with SessionLocal() as sess:
            sess.add(PersistentTask(
                task_id=task_id,
                description=description,
                status="pending",
            ))
            await sess.commit()
    except Exception as e:
        logger.warning("DB 写入 persistent_task 失败（不影响任务执行）: %s", e)


async def _db_update_status(
    task_id: str,
    status: str,
    *,
    started_at: Optional[datetime] = None,
    finished_at: Optional[datetime] = None,
    result_summary: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """
    更新 persistent_tasks 表中指定任务的状态和时间戳。

    Args:
        task_id:        任务 UUID
        status:         目标状态（running / done / failed / interrupted）
        started_at:     任务开始执行的时间（仅 running 时传入）
        finished_at:    任务结束的时间（done / failed 时传入）
        result_summary: 成功时的摘要 JSON（截断 2000 字符）
        error_message:  失败时的错误信息（截断 2000 字符）
    """
    try:
        from sqlalchemy import update
        from database import SessionLocal
        from models.persistent_task import PersistentTask
        values: Dict[str, Any] = {"status": status}
        if started_at:
            values["started_at"] = started_at
        if finished_at:
            values["finished_at"] = finished_at
        if result_summary is not None:
            values["result_summary"] = result_summary[:2000]
        if error_message is not None:
            values["error_message"] = error_message[:2000]
        async with SessionLocal() as sess:
            await sess.execute(
                update(PersistentTask)
                .where(PersistentTask.task_id == task_id)
                .values(**values)
            )
            await sess.commit()
    except Exception as e:
        logger.warning("DB 更新 persistent_task 失败: %s", e)


async def _db_mark_interrupted() -> int:
    """启动时将 DB 中 pending/running 状态的任务标记为 interrupted，返回受影响行数。"""
    try:
        from sqlalchemy import update
        from database import SessionLocal
        from models.persistent_task import PersistentTask
        now = datetime.now(timezone.utc)
        async with SessionLocal() as sess:
            result = await sess.execute(
                update(PersistentTask)
                .where(PersistentTask.status.in_(["pending", "running"]))
                .values(status="interrupted", finished_at=now)
            )
            await sess.commit()
            return result.rowcount
    except Exception as e:
        logger.warning("标记中断任务失败: %s", e)
        return 0


async def _db_load_history() -> None:
    """从 DB 加载已完成/失败/中断的历史任务到内存，供 /jobs 接口查询。"""
    try:
        from sqlalchemy import select
        from database import SessionLocal
        from models.persistent_task import PersistentTask
        async with SessionLocal() as sess:
            rows = (await sess.execute(
                select(PersistentTask)
                .order_by(PersistentTask.created_at.desc())
                .limit(500)
            )).scalars().all()
            loaded = 0
            for row in rows:
                if row.task_id not in _task_store:
                    _task_store[row.task_id] = {
                        "status": row.status,
                        "description": row.description or "",
                        "progress": None,
                        "result": row.result_summary,
                        "error": row.error_message,
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        "started_at": row.started_at.isoformat() if row.started_at else None,
                        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
                    }
                    loaded += 1
            if loaded:
                logger.info("从 DB 加载 %d 个历史任务", loaded)
    except Exception as e:
        logger.warning("从 DB 加载历史任务失败: %s", e)


# ──────────────────────────────────────────────
# 提交任务
# ──────────────────────────────────────────────
async def submit_task(coro: Coroutine, description: str = "", task_id: Optional[str] = None) -> str:
    """
    将异步协程提交到串行任务队列。

    同时在 persistent_tasks 表中创建一条记录，确保任务生命周期
    在进程重启后仍可追溯。

    Args:
        coro:        要执行的异步协程
        description: 任务描述（用于日志和状态查询）
        task_id:     可选，外部预生成的任务 ID；为 None 时自动生成 UUID

    Returns:
        task_id: UUID 格式的任务 ID
    """
    if task_id is None:
        task_id = str(uuid4())
    _task_store[task_id] = {
        "status": "pending",
        "description": description,
        "progress": None,
        "result": None,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "finished_at": None,
    }

    await _db_create_task(task_id, description)

    queue = _get_queue()
    await queue.put((task_id, coro))
    logger.info(f"任务已入队: {task_id} - {description}")
    return task_id


# ──────────────────────────────────────────────
# 更新任务进度（供任务内部调用）
# ──────────────────────────────────────────────
def update_task_progress(task_id: str, progress: Any) -> None:
    """
    更新任务进度信息（由任务协程内部调用）。

    仅写内存（秒级刷新），不写 DB（避免高频 I/O）。
    前端通过 GET /jobs/{id} 轮询读取。

    Args:
        task_id:  任务 ID
        progress: 进度信息（任意格式，如 dict/str/int）
    """
    if task_id in _task_store:
        _task_store[task_id]["progress"] = progress


# ──────────────────────────────────────────────
# 查询任务状态
# ──────────────────────────────────────────────
def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """查询任务状态。返回包含 status/progress/result/error 的字典，不存在则返回 None。"""
    return _task_store.get(task_id)


def get_all_tasks() -> Dict[str, Dict[str, Any]]:
    """返回所有任务的状态（含历史和活跃任务），用于管理后台 GET /jobs 展示。"""
    return dict(_task_store)


# ──────────────────────────────────────────────
# 后台 Worker（串行消费）
# ──────────────────────────────────────────────
async def _worker() -> None:
    """
    串行任务 worker：从队列中逐个取出任务并执行。

    worker=1 保证 SQLite 写操作不并发（PRD §8.1）。
    每个任务完成/失败后同步更新 DB 持久化记录。
    内存中超过 500 条已完成任务时自动裁剪最旧条目。
    """
    queue = _get_queue()
    logger.info("任务 Worker 已启动（串行模式，worker=1）")

    while True:
        task_id, coro = await queue.get()
        logger.info(f"开始执行任务: {task_id}")

        now = datetime.now(timezone.utc)
        _task_store[task_id]["status"] = "running"
        _task_store[task_id]["started_at"] = now.isoformat()
        await _db_update_status(task_id, "running", started_at=now)

        try:
            result = await coro
            finished = datetime.now(timezone.utc)
            _task_store[task_id]["status"] = "done"
            _task_store[task_id]["result"] = result
            _task_store[task_id]["finished_at"] = finished.isoformat()
            summary = json.dumps(result, ensure_ascii=False, default=str)[:2000] if result else None
            await _db_update_status(task_id, "done", finished_at=finished, result_summary=summary)
            logger.info(f"任务完成: {task_id}")
        except Exception as e:
            finished = datetime.now(timezone.utc)
            error_info = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            _task_store[task_id]["status"] = "failed"
            _task_store[task_id]["error"] = error_info
            _task_store[task_id]["finished_at"] = finished.isoformat()
            await _db_update_status(
                task_id, "failed", finished_at=finished,
                error_message=f"{type(e).__name__}: {e}",
            )
            logger.error(f"任务失败: {task_id} - {e}", exc_info=True)
        finally:
            queue.task_done()
            # 内存裁剪：超过 500 条时移除最旧的已完成/失败任务
            finished_ids = [
                tid for tid, t in _task_store.items()
                if t["status"] in ("done", "failed", "interrupted")
            ]
            if len(finished_ids) > 500:
                finished_ids.sort(key=lambda tid: _task_store[tid].get("created_at", ""))
                for old_id in finished_ids[:-500]:
                    del _task_store[old_id]


# ──────────────────────────────────────────────
# 启动 / 停止 Worker
# ──────────────────────────────────────────────
async def start_worker() -> None:
    """
    在 FastAPI lifespan 中调用，启动后台 worker。

    启动流程：
    1. 将 DB 中 pending/running 的遗留任务标记为 interrupted
    2. 从 DB 加载历史任务到内存（供 /jobs 接口查询）
    3. 创建 asyncio.Task 运行 _worker 协程
    """
    global _worker_task
    interrupted = await _db_mark_interrupted()
    if interrupted:
        logger.warning("已标记 %d 个中断任务（上次进程未正常退出）", interrupted)
    await _db_load_history()
    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_worker())
        logger.info("任务 Worker 后台任务已创建")


async def stop_worker() -> None:
    """在 FastAPI lifespan 关闭时调用，取消 worker 并等待其退出。"""
    global _worker_task
    if _worker_task and not _worker_task.done():
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
        logger.info("任务 Worker 已停止")
