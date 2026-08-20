import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.clip_pipeline import merge_transcript_segments

segs = [
    {"seq_index": 0, "text": "Like",           "start_time": 1.0, "end_time": 1.5},
    {"seq_index": 1, "text": "there's a",      "start_time": 1.6, "end_time": 2.2},
    {"seq_index": 2, "text": "rule,",           "start_time": 2.3, "end_time": 2.8},
    {"seq_index": 3, "text": "or something?",  "start_time": 2.9, "end_time": 3.8},
    {"seq_index": 4, "text": "I got it.",      "start_time": 6.0, "end_time": 7.0},
    {"seq_index": 5, "text": "Please",         "start_time": 7.2, "end_time": 7.5},
    {"seq_index": 6, "text": "don't do that again.", "start_time": 7.6, "end_time": 8.5},
]

merged = merge_transcript_segments(segs)
print(f"原始片段: {len(segs)}  合并后: {len(merged)}\n")
for m in merged:
    print(f"  [{m['start_time']:.1f}~{m['end_time']:.1f}s] {m['text']}")
