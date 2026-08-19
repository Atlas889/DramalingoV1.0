"""
node_sources 表 ORM 模型
对应 PRD v0.9 §7.4

记录每个知识节点在哪些剧集中出现过，支持跨剧复用溯源。
联合唯一约束：同一节点在同一集只记录一次。
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, Text, DateTime, ForeignKey, Index, UniqueConstraint,
)
from database import Base


class NodeSource(Base):
    __tablename__ = "node_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    node_id = Column(
        Integer, ForeignKey("knowledge_nodes.id"), nullable=False,
        comment="关联的知识节点 ID",
    )
    subtitle_id = Column(
        Integer, ForeignKey("subtitles_raw.id"), nullable=True,
        comment="对应的字幕原始行 ID",
    )
    show_name = Column(Text, nullable=False, comment="剧名")
    episode = Column(Text, nullable=False, default="", comment="集号")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), comment="创建时间")

    __table_args__ = (
        # 同一节点在同一集只记录一次（PRD §7.4 UNIQUE 约束）
        UniqueConstraint("node_id", "show_name", "episode", name="uq_node_show_episode"),
        Index("idx_node_sources_node", "node_id"),
    )

    def __repr__(self) -> str:
        return f"<NodeSource node={self.node_id} show={self.show_name} ep={self.episode}>"
