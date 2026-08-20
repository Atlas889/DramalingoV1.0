"""
调试指定台词的匹配失败原因
用法: python scripts/debug_match.py "Like there's a rule"
"""
import asyncio, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from database import SessionLocal
from sqlalchemy import text
from Levenshtein import ratio as levenshtein_ratio
import re

QUERY = sys.argv[1] if len(sys.argv) > 1 else "Like there's a rule"

def normalize(t):
    t = t.lower().strip()
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def sim(a, b):
    na, nb = normalize(a), normalize(b)
    if not na and not nb: return 1.0
    if not na or not nb: return 0.0
    return levenshtein_ratio(na, nb)

async def debug():
    async with SessionLocal() as s:
        # 1. 知识库中匹配的节点
        nodes = (await s.execute(text(
            "SELECT id, show_name, episode, en_text, start_time, end_time "
            "FROM knowledge_nodes WHERE LOWER(en_text) LIKE :q ORDER BY id"
        ), {"q": f"%{QUERY.lower()[:20]}%"})).all()

        print(f"\n=== 知识库节点 (搜索: '{QUERY}') ===")
        for n in nodes:
            print(f"  node_id={n[0]} {n[1]} {n[2]}  [{n[4]:.2f}s ~ {n[5]:.2f}s]")
            print(f"  en_text: {n[3]}")

        if not nodes:
            print("  (无匹配节点)")
            return

        # 取第一个节点做详细分析
        node = nodes[0]
        node_id, show, ep, en_text, hint_start, hint_end = node

        # 2. clips 表里该节点的切片状态
        clips = (await s.execute(text(
            "SELECT id, clip_status, match_confidence, align_step, whisper_start, whisper_end "
            "FROM clips WHERE node_id=:nid ORDER BY id DESC LIMIT 5"
        ), {"nid": node_id})).all()

        print(f"\n=== clips 表记录 (node_id={node_id}) ===")
        if clips:
            for c in clips:
                print(f"  clip_id={c[0]} status={c[1]} conf={c[2]} step={c[3]} "
                      f"[{c[4]}s ~ {c[5]}s]")
        else:
            print("  (无切片记录，节点未进入 unmatched 也未进切片)")

        # 3. Whisper 转写中找相似片段
        import config
        window = config.WHISPER_ALIGN_WINDOW_SECONDS
        win_start = max(0, hint_start - window) if hint_start else 0
        win_end = (hint_end or 0) + window

        print(f"\n=== Whisper 转写搜索 (窗口 [{win_start:.1f}s, {win_end:.1f}s]) ===")
        transcripts = (await s.execute(text(
            "SELECT seq_index, text, start_time, end_time FROM whisper_transcripts "
            "WHERE show_name=:show AND episode=:ep "
            "AND start_time < :we AND end_time > :ws "
            "ORDER BY start_time"
        ), {"show": show, "ep": ep, "ws": win_start, "we": win_end})).all()

        if transcripts:
            print(f"  窗口内找到 {len(transcripts)} 个片段:")
            for t in transcripts:
                s_ = sim(en_text, t[1])
                print(f"  seq={t[0]} sim={s_:.3f} [{t[2]:.2f}~{t[3]:.2f}s] {t[1][:80]}")
        else:
            print("  窗口内无转写片段！")

        # 4. 全量搜索最相似的 Whisper 片段（不限窗口）
        print(f"\n=== 全量 Whisper 相似度 Top-5 ===")
        all_t = (await s.execute(text(
            "SELECT seq_index, text, start_time, end_time FROM whisper_transcripts "
            "WHERE show_name=:show AND episode=:ep ORDER BY start_time"
        ), {"show": show, "ep": ep})).all()

        scored = sorted(all_t, key=lambda t: sim(en_text, t[1]), reverse=True)[:5]
        for t in scored:
            s_ = sim(en_text, t[1])
            print(f"  seq={t[0]} sim={s_:.3f} [{t[2]:.2f}~{t[3]:.2f}s] {t[1][:80]}")

        print(f"\n  节点原文: {en_text}")
        print(f"  字幕时间提示: {hint_start:.2f}s ~ {hint_end:.2f}s" if hint_start else "  字幕时间提示: 无")
        print(f"  对齐窗口: ±{window}s")

asyncio.run(debug())
