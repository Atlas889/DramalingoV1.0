"""
清理视频、切片、转写相关文件和数据库记录。
保留：knowledge_nodes、subtitle_raw 等知识库数据。
清除：video_files、whisper_transcripts、clips 表，以及 media/ 下的文件。
"""

import asyncio
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from database import SessionLocal
from sqlalchemy import text


async def clean():
    media_dir = config.MEDIA_DIR

    # ── 1. 清理文件系统 ─────────────────────────────────────
    if media_dir.exists():
        items = list(media_dir.iterdir())
        if not items:
            print(f"[文件] media 目录为空，跳过")
        else:
            for item in items:
                if item.is_dir():
                    # 删除切片/缩略图，保留视频文件可选（这里一起删）
                    count = sum(1 for _ in item.rglob("*") if _.is_file())
                    shutil.rmtree(item)
                    print(f"[文件] 已删除目录 {item.name}（含 {count} 个文件）")
                else:
                    item.unlink()
                    print(f"[文件] 已删除 {item.name}")
    else:
        print(f"[文件] media 目录不存在: {media_dir}")

    # ── 2. 清理数据库 ─────────────────────────────────────
    async with SessionLocal() as session:
        # clips
        r1 = await session.execute(text("DELETE FROM clips"))
        print(f"[DB] 已清除 clips 表: {r1.rowcount} 条")

        # whisper_transcripts
        r2 = await session.execute(text("DELETE FROM whisper_transcripts"))
        print(f"[DB] 已清除 whisper_transcripts 表: {r2.rowcount} 条")

        # video_files
        r3 = await session.execute(text("DELETE FROM video_files"))
        print(f"[DB] 已清除 video_files 表: {r3.rowcount} 条")

        # 重置 knowledge_nodes.has_any_clip
        r4 = await session.execute(
            text("UPDATE knowledge_nodes SET has_any_clip = FALSE")
        )
        print(f"[DB] 已重置 knowledge_nodes.has_any_clip: {r4.rowcount} 条")

        await session.commit()

    print("\n✅ 清理完成。知识库（knowledge_nodes / subtitle_raw）已保留。")
    print("   现在可以重新上传视频文件并触发转写 + 切片流水线。")


if __name__ == "__main__":
    asyncio.run(clean())
