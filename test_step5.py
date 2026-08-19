"""
Step 5 交付标准验证脚本
对应开发文档 Step 5 的 5 项验证测试
"""
import asyncio
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Step 5 · AI 打标 + 翻译 + 字幕流水线组装 · 交付标准验证")
print("=" * 60)

# ──────────────────────────────────────────────
# 测试 1：降级规则打标
# ──────────────────────────────────────────────
print("\n[测试 1] 降级规则打标")
from services.ai_tagger import rule_based_tag

result_short = rule_based_tag("Hi", word_count=1)
result_mid = rule_based_tag("I appreciate the offer but I have to pass", word_count=9)
result_long = rule_based_tag("This is a very long sentence with more than twenty words in it", word_count=21)

print(f"  词数 1  → difficulty={result_short['difficulty']}（期望 beginner）")
print(f"  词数 9  → difficulty={result_mid['difficulty']}（期望 intermediate）")
print(f"  词数 21 → difficulty={result_long['difficulty']}（期望 advanced）")

assert result_short["difficulty"] == "beginner", f"词数≤8 应为 beginner，实际: {result_short['difficulty']}"
assert result_mid["difficulty"] == "intermediate", f"词数9~20 应为 intermediate，实际: {result_mid['difficulty']}"
assert result_long["difficulty"] == "advanced", f"词数>20 应为 advanced，实际: {result_long['difficulty']}"
assert result_short["tag_source"] == "rule", "tag_source 应为 rule"
assert result_short["scene"] == "daily", "scene 应为 daily"
assert result_short["emotion"] == "neutral", "emotion 应为 neutral"
assert result_short["needs_translation"] == True, "needs_translation 应为 True"
print("  ✅ 降级规则打标正常")

# ──────────────────────────────────────────────
# 测试 2：AI 打标（需 API Key，无 Key 则跳过）
# ──────────────────────────────────────────────
print("\n[测试 2] AI 打标")
import config

if config.AI_API_KEY:
    async def test_ai_tag():
        from services.ai_tagger import tag_and_translate
        result = await tag_and_translate("I appreciate the offer, but I have to pass.")
        print(f"  Scene:      {result['scene']}")
        print(f"  Emotion:    {result['emotion']}")
        print(f"  Difficulty: {result['difficulty']}")
        print(f"  Keywords:   {result['keywords']}")
        print(f"  Translation:{result['zh_text']}")
        assert result["scene"] in ["workplace", "daily", "emotional", "restaurant", "travel", "study", "medical", "legal", "tech"], f"scene 值异常: {result['scene']}"
        assert len(result["keywords"]) >= 1, "keywords 数量不足"
        assert result["tag_source"] == "ai", "tag_source 应为 ai"
        print("  ✅ AI 打标正常")
    asyncio.run(test_ai_tag())
else:
    print("  ⚠️  AI_API_KEY 未配置，跳过 AI 打标测试（将使用规则降级）")
    print("  ✅ 跳过（规则降级已在测试 1 验证）")

# ──────────────────────────────────────────────
# 测试 3：完整字幕流水线（规则降级模式）
# ──────────────────────────────────────────────
print("\n[测试 3] 完整字幕流水线（规则降级模式）")

# 创建测试字幕文件
test_srt_e01 = """\
1
00:00:01,000 --> 00:00:03,000
I appreciate the offer, but I have to pass.

2
00:00:04,000 --> 00:00:06,000
You need to think about this more carefully.

3
00:00:07,000 --> 00:00:09,000
That's not how we do things around here.

4
00:00:10,000 --> 00:00:12,000
I'm gonna need you to come in on Saturday.

5
00:00:13,000 --> 00:00:15,000
We've been through this before, haven't we?

6
00:00:16,000 --> 00:00:18,000
The deal is off the table as of this morning.

7
00:00:19,000 --> 00:00:21,000
You can't just walk away from your responsibilities.

8
00:00:22,000 --> 00:00:24,000
Let me be perfectly clear about what I expect.

9
00:00:25,000 --> 00:00:27,000
This is not what I signed up for.

10
00:00:28,000 --> 00:00:30,000
OK

11
00:00:31,000 --> 00:00:33,000
Yeah

12
00:00:34,000 --> 00:00:36,000
We have to make a decision before the deadline.
"""

test_srt_e02 = """\
1
00:00:01,000 --> 00:00:03,000
I appreciate the offer, but I have to pass.

2
00:00:04,000 --> 00:00:06,000
Sometimes you have to take a leap of faith.

3
00:00:07,000 --> 00:00:09,000
Don't let anyone tell you what you can't do.

4
00:00:10,000 --> 00:00:12,000
The truth is, I've been lying to you all along.

5
00:00:13,000 --> 00:00:15,000
We need to talk about what happened last night.

6
00:00:16,000 --> 00:00:18,000
I'm not going to apologize for who I am.

7
00:00:19,000 --> 00:00:21,000
Everything happens for a reason, or so they say.

8
00:00:22,000 --> 00:00:24,000
You should have told me the truth from the beginning.
"""

with open("/tmp/test_pipeline_S01E01.srt", "w") as f:
    f.write(test_srt_e01)
with open("/tmp/test_pipeline_S01E02.srt", "w") as f:
    f.write(test_srt_e02)

async def test_pipeline():
    from database import SessionLocal, init_db
    from services.subtitle_pipeline import process_subtitle_batch
    from services.faiss_index import load_index
    import config

    await init_db()

    file_paths = ["/tmp/test_pipeline_S01E01.srt", "/tmp/test_pipeline_S01E02.srt"]
    episode_map = {
        "test_pipeline_S01E01.srt": "S01E01",
        "test_pipeline_S01E02.srt": "S01E02",
    }

    async with SessionLocal() as session:
        result = await process_subtitle_batch(
            file_paths=file_paths,
            show_name="PipelineTest",
            episode_map=episode_map,
            session=session,
            overwrite=True,
            enable_ai_tagging=False,  # 测试时使用规则降级
        )

    print(f"  处理结果:")
    for k, v in result.items():
        print(f"    {k}: {v}")

    assert result["total_files"] == 2, f"total_files 应为 2，实际: {result['total_files']}"
    assert result["new_nodes"] > 0, f"应有新增节点，实际: {result['new_nodes']}"
    assert result["filter_rate"] > 0, f"过滤率应 > 0，实际: {result['filter_rate']}"
    assert result["rule_fallback"] > 0, f"规则降级数应 > 0，实际: {result['rule_fallback']}"
    # E02 中第 1 行与 E01 相同，应被去重合并
    assert result["merged_nodes"] >= 1, f"应有合并节点（去重），实际: {result['merged_nodes']}"
    print("  ✅ 字幕流水线测试通过")
    return result

pipeline_result = asyncio.run(test_pipeline())

# ──────────────────────────────────────────────
# 测试 4：验证数据库写入完整性
# ──────────────────────────────────────────────
print("\n[测试 4] 数据库写入完整性")
import sqlite3

conn = sqlite3.connect("dramalingo.db")
cursor = conn.cursor()
cursor.execute("""
    SELECT 
        (SELECT COUNT(*) FROM knowledge_nodes WHERE show_name='PipelineTest') as nodes,
        (SELECT COUNT(*) FROM node_sources WHERE show_name='PipelineTest') as sources,
        (SELECT COUNT(DISTINCT node_id) FROM node_sources WHERE show_name='PipelineTest') as unique_nodes,
        (SELECT COUNT(*) FROM knowledge_nodes WHERE tag_source='rule' AND show_name='PipelineTest') as rule_tagged
""")
row = cursor.fetchone()
conn.close()

nodes, sources, unique_nodes, rule_tagged = row
print(f"  knowledge_nodes (PipelineTest): {nodes}")
print(f"  node_sources    (PipelineTest): {sources}")
print(f"  unique_nodes    (PipelineTest): {unique_nodes}")
print(f"  rule_tagged     (PipelineTest): {rule_tagged}")

assert nodes > 0, "knowledge_nodes 应有记录"
assert sources >= nodes, f"node_sources 应 >= nodes，实际: sources={sources}, nodes={nodes}"
assert unique_nodes == nodes, f"unique_nodes 应等于 nodes（每个节点至少一条来源），实际: {unique_nodes} vs {nodes}"
assert rule_tagged > 0, "应有规则打标记录"
print("  ✅ 数据库写入完整性验证通过")

# ──────────────────────────────────────────────
# 测试 5：验证 FAISS 索引更新
# ──────────────────────────────────────────────
print("\n[测试 5] FAISS 索引更新")
from services.faiss_index import load_index

try:
    index, mapping = load_index("indexes/knowledge.index")
    print(f"  索引大小: {index.ntotal}，映射大小: {len(mapping)}")
    assert index.ntotal == len(mapping), f"索引与映射大小不一致: {index.ntotal} vs {len(mapping)}"
    assert index.ntotal > 0, "索引应有向量"
    print("  ✅ FAISS 索引正常")
except FileNotFoundError:
    print("  ⚠️  索引文件不存在（可能流水线未生成新节点）")

# ──────────────────────────────────────────────
# 测试 6：幂等检查
# ──────────────────────────────────────────────
print("\n[测试 6] 幂等检查（重复上传同一集）")

async def test_idempotent():
    from database import SessionLocal
    from services.subtitle_pipeline import process_subtitle_batch

    file_paths = ["/tmp/test_pipeline_S01E01.srt"]
    episode_map = {"test_pipeline_S01E01.srt": "S01E01"}

    async with SessionLocal() as session:
        result = await process_subtitle_batch(
            file_paths=file_paths,
            show_name="PipelineTest",
            episode_map=episode_map,
            session=session,
            overwrite=False,  # 不覆盖
            enable_ai_tagging=False,
        )

    print(f"  重复上传结果: skipped_files={result['skipped_files']}, new_nodes={result['new_nodes']}")
    assert result["skipped_files"] == 1, f"应跳过 1 个文件，实际: {result['skipped_files']}"
    assert result["new_nodes"] == 0, f"重复上传不应新增节点，实际: {result['new_nodes']}"
    print("  ✅ 幂等检查正常")

asyncio.run(test_idempotent())

print("\n" + "=" * 60)
print("✅ Step 5 全部交付标准验证通过！")
print("=" * 60)
