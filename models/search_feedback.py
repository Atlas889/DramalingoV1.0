"""
search_feedback 表 ORM 模型

记录用户搜索交互行为（点击、播放视频），
为后续检索权重调优和结果排序提供数据基础。
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Text, Float, DateTime, Index
from database import Base


class SearchFeedback(Base):
    __tablename__ = "search_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False, comment="用户查询文本")
    node_id = Column(Integer, nullable=False, comment="被交互的知识节点 ID")
    action = Column(Text, nullable=False, comment="click | play | copy")
    rank_position = Column(Integer, nullable=True, comment="结果在列表中的位置（从 1 开始）")
    rank_score = Column(Float, nullable=True, comment="该结果的综合排序分")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_sf_query", "query"),
        Index("idx_sf_node", "node_id"),
        Index("idx_sf_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<SearchFeedback q='{self.query[:20]}' node={self.node_id} action={self.action}>"
