"""
knowledge_nodes 表 ORM 模型
对应 PRD v0.9 §7.3

知识节点：经过筛选、去重、AI 打标后的结构化英文表达。
每个节点全库唯一，重复出现只累加 occurrence_count。
"""

import re as _re
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, Text, Float, Boolean, DateTime, LargeBinary,
    ForeignKey, Index, JSON, UniqueConstraint,
)
from database import Base


def make_node_code(show_name: str, episode: str, start_time: float) -> str:
    """
    生成节点唯一编码：{剧名缩写}-{集号}-{毫秒时间戳}
    示例：FRIENDS-S01E01-00154520
    """
    slug = _re.sub(r'[^A-Z0-9]+', '-', show_name.upper()).strip('-')[:20]
    ep   = _re.sub(r'[^A-Z0-9]',  '', (episode or '').upper())[:8]
    ms   = int(round(start_time * 1000))
    parts = [slug] + ([ep] if ep else []) + [f'{ms:08d}']
    return '-'.join(parts)


class KnowledgeNode(Base):
    __tablename__ = "knowledge_nodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(
        Text, nullable=True, unique=True,
        comment="节点唯一编码，格式：剧名-集号-毫秒时间戳，如 FRIENDS-S01E01-00154520",
    )
    subtitle_id = Column(
        Integer, ForeignKey("subtitles_raw.id"), nullable=True,
        comment="首次来源字幕行 ID（溯源）",
    )
    en_text = Column(Text, nullable=False, comment="英文表达")
    zh_text = Column(Text, nullable=False, default="", comment="中文翻译")
    # show_name / episode 记录首次出现的剧集
    show_name = Column(Text, nullable=False, comment="首次来源剧名")
    episode = Column(Text, nullable=False, default="", comment="首次来源集号")
    start_time = Column(
        Float, nullable=False,
        comment="首次来源字幕时间戳（仅作参考，切片实际使用 Whisper 时间戳）",
    )
    end_time = Column(Float, nullable=False, comment="首次来源结束时间")
    word_count = Column(Integer, nullable=False, default=0, comment="英文词数")
    occurrence_count = Column(
        Integer, nullable=False, default=1,
        comment="在全库字幕中出现的次数（按剧集粒度计数），越多越地道",
    )
    # AI 打标字段
    scene = Column(
        Text, nullable=False, default="daily",
        comment="情景标签: workplace/daily/emotional/restaurant/travel/study/medical/legal/tech",
    )
    emotion = Column(
        Text, nullable=False, default="neutral",
        comment="情绪标签: neutral/happy/sad/angry/serious/humorous/sarcastic/anxious/surprised",
    )
    difficulty = Column(
        Text, nullable=False, default="intermediate",
        comment="难度: beginner（入门）| intermediate（进阶）| advanced（挑战）",
    )
    keywords = Column(
        JSON, nullable=False, default=list,
        comment="3~5 个英文关键词",
    )
    speech_act = Column(
        Text, nullable=True, default="neutral_statement",
        comment=(
            "言语行为标签（Speech Act Theory）: 句子在交际中执行的语用功能。"
            "枚举值见 services/ai_tagger.py VALID_SPEECH_ACTS，共 25 种，"
            "如 apologizing / requesting / disagreeing / neutral_statement 等。"
            "NULL 表示历史存量节点尚未打标，可通过 /admin/retag_speech_acts 批量补标。"
        ),
    )
    embedding = Column(
        LargeBinary, nullable=True,
        comment="英文字幕的 embedding 向量（384 维），用于跨语言检索与去重",
    )
    zh_embedding = Column(
        LargeBinary, nullable=True,
        comment="中文翻译的 embedding 向量（384 维），用于中文查询精准语义匹配",
    )
    quality_score = Column(Float, default=0.5, comment="TF-IDF 质量评分")
    needs_translation = Column(
        Boolean, default=False,
        comment="AI 打标降级时为 True，进入人工补译队列",
    )
    tag_source = Column(
        Text, default="ai",
        comment="打标来源: ai（AI 打标）| rule（降级规则打标），便于后续批量重跑",
    )
    has_any_clip = Column(
        Boolean, nullable=False, default=False,
        comment="汇总字段：全库是否存在 ≥1 条关联且有效的切片",
    )
    content_type = Column(
        Text, nullable=False, default="tv", server_default="tv",
        comment="内容大类: tv（电视剧）| movie（电影）| talk（演讲）",
    )
    is_active = Column(Boolean, default=True, comment="是否启用（下线时设为 False）")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), comment="创建时间")

    # 索引
    __table_args__ = (
        UniqueConstraint("en_text", name="uq_kn_en_text"),
        Index("idx_kn_show_episode", "show_name", "episode"),
        Index("idx_kn_is_active", "is_active"),
        Index("idx_kn_scene", "scene"),
        Index("idx_kn_difficulty", "difficulty"),
        Index("idx_kn_has_clip", "has_any_clip"),
        Index("idx_kn_subtitle_id", "subtitle_id"),
    )

    def __repr__(self) -> str:
        return f"<KnowledgeNode id={self.id} en='{self.en_text[:30]}...'>"
