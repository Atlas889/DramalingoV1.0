"""
persistent_tasks 表 ORM 模型

持久化任务生命周期到数据库，解决进程重启后 pending/running 任务
从内存队列中丢失且不可见的问题。

状态流转: pending → running → done / failed
         pending / running → interrupted (进程重启时自动标记)
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Text, Float, DateTime, Index
from database import Base


class PersistentTask(Base):
    __tablename__ = "persistent_tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(Text, nullable=False, unique=True, comment="UUID，与内存 task_store 一致")
    description = Column(Text, nullable=False, default="", comment="任务描述")
    status = Column(
        Text, nullable=False, default="pending",
        comment="pending | running | done | failed | interrupted",
    )
    progress_pct = Column(Float, nullable=True, comment="进度百分比 0~100（可选）")
    result_summary = Column(Text, nullable=True, comment="完成时的摘要信息（JSON 字符串）")
    error_message = Column(Text, nullable=True, comment="失败时的错误信息")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_pt_status", "status"),
        Index("idx_pt_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<PersistentTask {self.task_id} status={self.status}>"
