"""
DramaLingo · Whisper 音频转写服务
对应 PRD v0.9 §5 流程二 / 开发文档 Step 7

功能：
  1. FFmpeg 从视频文件提取 16kHz 单声道 WAV（Whisper 最佳输入格式）
  2. Whisper 模型转写，输出带精确时间戳的文本片段序列
  3. 转写结果缓存到 whisper_transcripts 表（永久保留，下次直接复用，跳过耗时转写）
  4. 视频文件元数据写入 video_files 表（文件大小、时长、字幕批次关联）

全局模型管理：
  - `_whisper_model` 懒加载（首次使用时初始化，避免非转写场景启动慢）
  - 模型大小由 config.WHISPER_MODEL 控制（tiny/base/small/medium）

公开接口：
  transcribe_video(video_path, show_name, episode, session, force)
    → List[dict]  # {seq_index, text, start_time, end_time}
"""

import os
import asyncio
import logging
import tempfile
import subprocess
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

import config
from models.whisper_transcript import WhisperTranscript
from models.video_file import VideoFile

logger = logging.getLogger(__name__)

# 全局 Whisper 模型（懒加载，首次使用时初始化；模块级不 import whisper 避免启动慢）
_whisper_model = None


def get_whisper_model():
    """懒加载 Whisper 模型，首次调用时才 import whisper 并加载权重"""
    global _whisper_model
    if _whisper_model is None:
        import whisper as _whisper   # 仅在需要时导入，避免拖慢非转写场景的启动
        logger.info(f"加载 Whisper 模型: {config.WHISPER_MODEL}")
        _whisper_model = _whisper.load_model(config.WHISPER_MODEL)
        logger.info("Whisper 模型加载完成")
    return _whisper_model


def extract_audio(video_path: str, audio_path: str) -> float:
    """
    使用 FFmpeg 从视频文件提取音频，返回视频时长（秒）
    
    Args:
        video_path: 视频文件路径
        audio_path: 输出音频文件路径（WAV 格式）
    
    Returns:
        视频时长（秒）
    """
    # 提取音频：16kHz 单声道 WAV（Whisper 最佳输入格式）
    cmd_audio = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        audio_path
    ]
    result = subprocess.run(cmd_audio, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 提取音频失败: {result.stderr[-500:]}")
    
    # 获取视频时长
    cmd_duration = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path
    ]
    result_dur = subprocess.run(cmd_duration, capture_output=True, text=True, timeout=30)
    try:
        duration = float(result_dur.stdout.strip())
    except (ValueError, AttributeError):
        duration = 0.0
    
    return duration


async def get_cached_transcripts(show_name: str, episode: str, session: AsyncSession) -> Optional[List[dict]]:
    """
    查询缓存的转写结果，若存在则直接返回
    
    Returns:
        缓存的转写序列，或 None（无缓存）
    """
    stmt = (
        select(WhisperTranscript)
        .where(
            WhisperTranscript.show_name == show_name,
            WhisperTranscript.episode == episode,
        )
        .order_by(WhisperTranscript.seq_index)
    )
    result = await session.execute(stmt)
    rows = result.scalars().all()

    # 兼容旧数据：以前空集号被存成 "S00E00"，用该别名再试一次
    if not rows and episode == "":
        stmt2 = (
            select(WhisperTranscript)
            .where(WhisperTranscript.show_name == show_name, WhisperTranscript.episode == "S00E00")
            .order_by(WhisperTranscript.seq_index)
        )
        result2 = await session.execute(stmt2)
        rows = result2.scalars().all()
        if rows:
            logger.info(f"命中旧格式转写缓存(S00E00): {show_name}，共 {len(rows)} 条")

    if not rows:
        return None

    logger.info(f"命中转写缓存: {show_name} {episode}，共 {len(rows)} 条")
    return [
        {
            "text": r.text,
            "start_time": r.start_time,
            "end_time": r.end_time,
            "seq_index": r.seq_index,
        }
        for r in rows
    ]


async def save_transcripts(
    show_name: str,
    episode: str,
    segments: List[dict],
    session: AsyncSession,
) -> None:
    """
    将 Whisper 转写结果批量写入 whisper_transcripts 表
    （先删除旧记录，再插入新记录，保证幂等）
    """
    # 删除旧记录（若存在）
    await session.execute(
        delete(WhisperTranscript).where(
            WhisperTranscript.show_name == show_name,
            WhisperTranscript.episode == episode,
        )
    )
    
    # 批量插入新记录
    records = [
        WhisperTranscript(
            show_name=show_name,
            episode=episode,
            seq_index=i,
            text=seg["text"].strip(),
            start_time=float(seg["start"]),
            end_time=float(seg["end"]),
            whisper_model=config.WHISPER_MODEL,
            source="clip_align",
        )
        for i, seg in enumerate(segments)
        if seg.get("text", "").strip()
    ]
    
    session.add_all(records)
    await session.flush()
    logger.info(f"转写结果已写入数据库: {show_name} {episode}，共 {len(records)} 条")


async def upsert_video_file(
    show_name: str,
    episode: str,
    file_path: str,
    file_size: int,
    duration: float,
    session: AsyncSession,
    subtitle_code: Optional[str] = None,
) -> VideoFile:
    """
    写入或更新 video_files 表记录
    """
    stmt = select(VideoFile).where(
        VideoFile.show_name == show_name,
        VideoFile.episode == episode,
    )
    result = await session.execute(stmt)
    video_file = result.scalar_one_or_none()

    # 兼容旧数据：以前空集号被存成 "S00E00"，找到后迁移 episode 字段
    if video_file is None and episode == "":
        stmt2 = select(VideoFile).where(
            VideoFile.show_name == show_name, VideoFile.episode == "S00E00"
        )
        result2 = await session.execute(stmt2)
        video_file = result2.scalar_one_or_none()
        if video_file:
            video_file.episode = ""   # 迁移到新格式
            logger.info(f"VideoFile episode 迁移: {show_name} S00E00 → ''")

    if video_file is None:
        video_file = VideoFile(
            show_name=show_name,
            episode=episode,
            file_path=file_path,
            file_size=file_size,
            duration=duration,
            has_transcript=True,
            subtitle_code=subtitle_code,
        )
        session.add(video_file)
    else:
        video_file.file_path = file_path
        video_file.file_size = file_size
        video_file.duration = duration
        video_file.has_transcript = True
        if subtitle_code:
            video_file.subtitle_code = subtitle_code

    await session.flush()
    return video_file


async def transcribe_video(
    video_path: str,
    show_name: str,
    episode: str,
    session: AsyncSession,
    force_retranscribe: bool = False,
    subtitle_code: Optional[str] = None,
) -> List[dict]:
    """
    主入口：对视频文件执行 Whisper 转写，返回带时间戳的文本序列
    
    Args:
        video_path: 视频文件的绝对路径
        show_name: 剧名（如 "Friends"）
        episode: 集号（如 "S01E01"）
        session: 数据库会话
        force_retranscribe: 是否强制重新转写（忽略缓存）
    
    Returns:
        转写序列：[{"text": str, "start_time": float, "end_time": float, "seq_index": int}, ...]
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 1. 检查缓存
    if not force_retranscribe:
        cached = await get_cached_transcripts(show_name, episode, session)
        if cached is not None:
            return cached
    
    logger.info(f"开始转写视频: {video_path} ({show_name} {episode})")
    
    # 2. 提取音频到临时文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        audio_path = tmp_audio.name
    
    try:
        # 在线程池中执行 CPU 密集型操作（避免阻塞事件循环）
        loop = asyncio.get_running_loop()
        
        # 提取音频
        duration = await loop.run_in_executor(
            None, extract_audio, video_path, audio_path
        )
        logger.info(f"音频提取完成，视频时长: {duration:.1f}s")
        
        # 3. 调用 Whisper 转写
        def do_transcribe():
            model = get_whisper_model()
            result = model.transcribe(
                audio_path,
                language="en",          # 强制英文，提升准确率
                verbose=False,
                beam_size=1,            # 贪心解码（速度快 30-50%，准确率损失极小）
                condition_on_previous_text=False,  # 禁用上文依赖，消除串行瓶颈
                temperature=0.0,        # 固定温度=0，与 beam_size=1 配合最稳定
            )
            return result
        
        logger.info("开始 Whisper 转写...")
        result = await loop.run_in_executor(None, do_transcribe)
        segments = result.get("segments", [])
        logger.info(f"Whisper 转写完成，共 {len(segments)} 个片段")
        
        # 4. 保存转写结果到数据库
        await save_transcripts(show_name, episode, segments, session)
        
        # 5. 更新 video_files 表
        file_size = os.path.getsize(video_path)
        await upsert_video_file(show_name, episode, video_path, file_size, duration, session, subtitle_code)
        
        await session.commit()
        
        # 6. 返回格式化结果
        transcripts = [
            {
                "text": seg["text"].strip(),
                "start_time": float(seg["start"]),
                "end_time": float(seg["end"]),
                "seq_index": i,
            }
            for i, seg in enumerate(segments)
            if seg.get("text", "").strip()
        ]
        
        logger.info(f"转写完成: {show_name} {episode}，有效片段 {len(transcripts)} 条")
        return transcripts
    
    finally:
        # 清理临时音频文件
        if os.path.exists(audio_path):
            os.unlink(audio_path)
