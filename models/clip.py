"""
clips 表 ORM 模型
对应 PRD v0.9 §7.7

视频切片：每个切片锚定一个知识节点（1:N 关系），
记录切片状态、对齐来源、置信度等明细信息。
每次重新切片新增记录，旧记录标记 is_active=False，保留历史轨迹。
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, Text, Float, Boolean, DateTime, JSON,
    ForeignKey, Index,
)
from database import Base


class Clip(Base):
    __tablename__ = "clips"

    id = Column(Integer, primary_key=True, autoincrement=True)
    clip_code = Column(
        Text, nullable=True, unique=True,
        comment="业务编码，格式 CLP-{show2}-{ep2}-{rand6}，全局唯一",
    )
    node_id = Column(
        Integer, ForeignKey("knowledge_nodes.id"), nullable=False,
        comment="关联的知识节点 ID（NOT NULL，每个切片必须对应知识节点）",
    )
    show_name = Column(Text, nullable=False, comment="剧名")
    episode = Column(Text, nullable=False, default="", comment="集号")
    # 字幕内容（冗余存储，方便前端直接展示）
    subtitle_en = Column(Text, nullable=False, default="", comment="英文台词")
    subtitle_zh = Column(Text, nullable=False, default="", comment="中文翻译")
    # 标签（冗余存储，来自知识节点）
    scene = Column(Text, default="daily", comment="情景标签")
    emotion = Column(Text, default="neutral", comment="情绪标签")
    difficulty = Column(Text, default="intermediate", comment="难度标签")
    keywords = Column(JSON, default=list, comment="关键词")
    actors = Column(Text, default="", comment="角色信息")
    # 切片文件路径
    stream_url = Column(Text, default="", comment="视频切片 URL 路径")
    thumb_url = Column(Text, default="", comment="缩略图 URL 路径")
    # 切片状态与约束
    clip_status = Column(
        Text, nullable=False, default="matched",
        comment=(
            "切片状态: matched（匹配成功，切片已生成）| "
            "unmatched（匹配失败，置信度不足或时长超限）| "
            "failed（切片失败，FFmpeg 出错）"
        ),
    )
    is_active = Column(
        Boolean, default=True,
        comment="是否有效（被新切片覆盖时设为 False）",
    )
    duration = Column(Float, nullable=True, comment="最终切片时长（秒）：matched 状态为含上下文窗口+padding 的真实切割时长；其他状态为 Whisper 片段时长")
    word_count = Column(Integer, nullable=True, comment="实际词数")
    is_split = Column(
        Boolean, default=False,
        comment="是否来自超长节点的拆分",
    )
    # 转写对齐字段
    match_confidence = Column(
        Float, nullable=True,
        comment="对齐置信度（0~1）；Step 2a ≥0.85，Step 2b ≥0.90",
    )
    whisper_start = Column(Float, nullable=True, comment="实际使用的 Whisper 起始时间戳")
    whisper_end = Column(Float, nullable=True, comment="实际使用的 Whisper 结束时间戳")
    whisper_text = Column(Text, nullable=True, comment="匹配到的 Whisper 转写原文（调试用）")
    align_step = Column(
        Text, nullable=True,
        comment="对齐来源: seq（序列对齐）| subtitle_fallback（字幕兜底）",
    )
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), comment="创建时间")

    # 索引
    __table_args__ = (
        Index("idx_clips_node", "node_id"),
        Index("idx_clips_show_episode", "show_name", "episode"),
        Index("idx_clips_status_active", "clip_status", "is_active"),
        Index("idx_clips_node_active", "node_id", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<Clip id={self.id} node={self.node_id} status={self.clip_status}>"
