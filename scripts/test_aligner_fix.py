"""验证 aligner 修复后对 'Like there's a rule' 的匹配结果"""
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import SessionLocal
from sqlalchemy import text
from services.aligner import align_seq, text_similarity
import config

async def test():
    async with SessionLocal() as s:
        # 取 node_id=91 所在集的所有节点（按字幕时间顺序）
        nodes_raw = (await s.execute(text(
            "SELECT id, en_text, start_time, end_time FROM knowledge_nodes "
            "WHERE show_name='老友记' AND episode='S01E01' AND is_active=TRUE "
            "ORDER BY start_time NULLS LAST"
        ))).all()

        # 取 Whisper 转写
        transcripts_raw = (await s.execute(text(
            "SELECT seq_index, text, start_time, end_time FROM whisper_transcripts "
            "WHERE show_name='老友记' AND episode='S01E01' ORDER BY seq_index"
        ))).all()

    nodes = [
        {"node_id": r[0], "en_text": r[1], "start_time_hint": r[2] or 0.0}
        for r in nodes_raw
    ]
    transcripts = [
        {"seq_index": r[0], "text": r[1], "start_time": r[2], "end_time": r[3]}
        for r in transcripts_raw
    ]

    print(f"节点数: {len(nodes)}, 转写片段数: {len(transcripts)}")
    print(f"对齐窗口: ±{config.WHISPER_ALIGN_WINDOW_SECONDS}s\n")

    results = align_seq(nodes, transcripts, window_seconds=config.WHISPER_ALIGN_WINDOW_SECONDS)

    # 找 node_id=91 的结果
    target = next((r for r in results if r["node_id"] == 91), None)
    print("=== node_id=91 ('Like there's a rule') ===")
    if target:
        print(f"  matched to: [{target['whisper_start']:.1f}~{target['whisper_end']:.1f}s]")
        print(f"  whisper_text: {target['whisper_text']}")
        print(f"  text_similarity: {target['text_similarity']}")
        print(f"  confidence: {target['confidence']}")
        print(f"  align_step: {target['align_step']}")
    else:
        print("  ❌ 未出现在对齐结果中（置信度低于 0.5 阈值）")

    # 统计总体结果
    print(f"\n总匹配数: {len(results)} / {len(nodes)}")
    high = sum(1 for r in results if r['confidence'] >= 0.85)
    low  = sum(1 for r in results if 0.60 <= r['confidence'] < 0.85)
    print(f"高置信（≥0.85）: {high}")
    print(f"低置信（0.60-0.85）: {low}")

asyncio.run(test())
