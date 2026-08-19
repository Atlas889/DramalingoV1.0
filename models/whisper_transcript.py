"""
whisper_transcripts 表 ORM 模型
对应 PRD v0.9 §7.6

Whisper 转写结果缓存，支持重新切片时跳过转写步骤。
三元联合唯一约束：同一剧集同一集同一序号只记录一次。
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, Text, Float, DateTime, UniqueConstraint, Index,
)
from database import Base


class WhisperTranscript(Base):
    __tablename__ = "whisper_transcripts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    show_name = Column(Text, nullable=False, comment="剧名")
    episode = Column(Text, nullable=False, comment="集号")
    seq_index = Column(
        Integer, nullable=False,
        comment="转写序列中的顺序编号",
    )
    text = Column(Text, nullable=False, comment="Whisper 转写文本")
    start_time = Column(
        Float, nullable=False,
        comment="相对于视频文件起点的秒数",
    )
    end_time = Column(Float, nullable=False, comment="结束时间（秒）")
    whisper_model = Column(
        Text, nullable=False, default="base",
        comment="使用的 Whisper 模型版本",
    )
    source = Column(
        Text, nullable=False, default="clip_align",
        comment=(
            "来源: clip_align（流程二切片对齐用转写）| "
            "whisper_knowledge（流程一 Whisper 建库用转写）"
        ),
    )
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), comment="创建时间")

    __table_args__ = (
        # 三元联合唯一约束（PRD §7.6）
        UniqueConstraint(
            "show_name", "episode", "seq_index",
            name="uq_whisper_show_episode_seq",
        ),
        # 联合索引，加速按剧集查询
        Index("idx_whisper_episode", "show_name", "episode"),
    )

    def __repr__(self) -> str:
        return (
            f"<WhisperTranscript show={self.show_name} ep={self.episode} "
            f"seq={self.seq_index}>"
        )
