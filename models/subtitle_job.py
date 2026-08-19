"""
DramaLingo · 字幕处理任务模型

存储通过 TED 链接（或其他途径）获取的字幕文件及其处理状态，
支持人工预览确认后再发送到字幕处理流水线。

状态流转：
  fetching → ready（获取成功）
  fetching → failed（获取失败）
  ready    → processing（已发送到流水线）
  processing → done（流水线完成）
  processing → error（流水线报错）
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, DateTime, Text

from database import Base


class SubtitleJob(Base):
    __tablename__ = "subtitle_jobs"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    ted_url       = Column(Text,    nullable=False,  comment="TED 演讲原始 URL")
    title_zh      = Column(String,  nullable=False,  comment="演讲中文标题（用户填写）")
    title_en      = Column(String,  nullable=False, default="", comment="演讲英文标题（yt-dlp 解析或用户填写）")

    # 状态流转：fetching → ready/failed → processing → done/error
    status        = Column(String,  nullable=False, default="fetching",
                           comment="任务状态: fetching | ready | failed | processing | done | error")

    srt_path      = Column(Text,    nullable=True,  comment="生成的双语 SRT 文件绝对路径（ready/done 时填充）")
    caption_count = Column(Integer, nullable=True,  comment="SRT 文件中的字幕条数")
    zh_source     = Column(String,  nullable=True,  comment='中文翻译来源描述，如"官方中文字幕"、"DeepSeek 翻译+润色"')

    # 关联的后台任务 ID（fetch 任务或 pipeline 任务，用于实时进度轮询）
    task_id          = Column(String, nullable=True, comment="字幕获取后台任务 ID（task_runner 任务）")
    pipeline_task_id = Column(String, nullable=True, comment="字幕处理流水线后台任务 ID（发送到流水线后填充）")

    error_msg = Column(Text, nullable=True, comment="失败原因描述（failed/error 状态时填充）")

    # 流水线结果统计（done 后填充）
    new_nodes          = Column(Integer, nullable=True, comment="本次处理新增的知识节点数")
    ai_tagging_success = Column(Integer, nullable=True, comment="AI 打标成功数（DeepSeek/Qwen）")
    rule_fallback      = Column(Integer, nullable=True, comment="降级为规则打标的数量")

    created_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict:
        return {
            "id":                self.id,
            "ted_url":           self.ted_url,
            "title_zh":          self.title_zh,
            "title_en":          self.title_en,
            "status":            self.status,
            "srt_path":          self.srt_path,
            "caption_count":     self.caption_count,
            "zh_source":         self.zh_source,
            "task_id":           self.task_id,
            "pipeline_task_id":  self.pipeline_task_id,
            "error_msg":         self.error_msg,
            "new_nodes":         self.new_nodes,
            "ai_tagging_success": self.ai_tagging_success,
            "rule_fallback":     self.rule_fallback,
            "created_at":        self.created_at.isoformat() if self.created_at else None,
            "updated_at":        self.updated_at.isoformat() if self.updated_at else None,
        }
