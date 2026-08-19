"""
subtitles_raw 表 ORM 模型
对应 PRD v0.9 §7.2

字幕原始行存档，记录每条字幕的解析结果和筛选状态。
幂等检查由 API 层实现（上传前查 show_name + episode 是否已有记录）。
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Text, Float, DateTime, Index
from database import Base


class SubtitleRaw(Base):
    __tablename__ = "subtitles_raw"

    id = Column(Integer, primary_key=True, autoincrement=True)
    show_name = Column(Text, nullable=False, comment="中文剧名")
    en_name = Column(Text, nullable=False, default="", comment="英文剧名")
    episode = Column(Text, nullable=False, default="", comment="集号，如 S01E01")
    subtitle_code = Column(Text, nullable=True, index=True, comment="上传批次唯一编码，同批文件共享")
    start_time = Column(Float, nullable=False, comment="起始时间（秒，精度到毫秒）")
    end_time = Column(Float, nullable=False, comment="结束时间（秒）")
    en_text = Column(Text, nullable=False, comment="英文字幕文本")
    zh_text = Column(Text, nullable=False, default="", comment="中文字幕文本")
    word_count = Column(Integer, nullable=False, default=0, comment="英文词数")
    filter_status = Column(
        Text, nullable=False, default="pending",
        comment=(
            "筛选状态: pending | passed | rejected_blacklist | "
            "rejected_wordcount | rejected_quality | rejected_dedup"
        ),
    )
    quality_score = Column(Float, default=None, comment="TF-IDF 质量评分")
    content_type = Column(
        Text, nullable=False, default="tv", server_default="tv",
        comment="内容大类: tv（电视剧）| movie（电影）| talk（演讲）",
    )
    source = Column(
        Text, nullable=False, default="subtitle",
        comment="来源: subtitle（外挂字幕）| whisper_knowledge（Whisper 转录建库用）",
    )
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), comment="创建时间")

    # 索引：加速按剧集查询和按筛选状态过滤
    __table_args__ = (
        Index("idx_subtitles_show_episode", "show_name", "episode"),
        Index("idx_subtitles_filter_status", "filter_status"),
    )

    def __repr__(self) -> str:
        return f"<SubtitleRaw id={self.id} show={self.show_name} ep={self.episode}>"
