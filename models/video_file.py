"""
video_files 表 ORM 模型
对应 PRD v0.9 §7.5

永久记录已上传的视频文件，支持知识库更新后对存量视频重新切片。
联合唯一约束：同一剧集同一集只能有一个视频文件记录。
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, Text, Float, Boolean, DateTime, UniqueConstraint,
)
from database import Base


class VideoFile(Base):
    __tablename__ = "video_files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    show_name = Column(Text, nullable=False, comment="剧名")
    episode = Column(Text, nullable=False, comment="集号")
    file_path = Column(Text, nullable=False, comment="本地存储路径")
    file_size = Column(Integer, nullable=True, comment="文件大小（字节）")
    duration = Column(Float, nullable=True, comment="视频总时长（秒）")
    subtitle_code = Column(Text, nullable=True, comment="关联的字幕批次编码")
    content_type = Column(
        Text, nullable=False, default="tv", server_default="tv",
        comment="内容大类: tv（电视剧）| movie（电影）| talk（演讲）",
    )
    has_transcript = Column(
        Boolean, default=False,
        comment="whisper_transcripts 是否已有该集转写",
    )
    last_clipped_at = Column(
        DateTime, nullable=True,
        comment="最近一次执行切片的时间",
    )
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), comment="创建时间")

    __table_args__ = (
        UniqueConstraint("show_name", "episode", name="uq_video_show_episode"),
    )

    def __repr__(self) -> str:
        return f"<VideoFile id={self.id} show={self.show_name} ep={self.episode}>"
