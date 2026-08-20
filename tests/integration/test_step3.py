"""
Step 3 交付标准验证脚本
对应开发文档 Step 3 的 5 项验证测试
"""
import asyncio
import os
import sys
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
from config import INDEX_DIR

print("=" * 60)
print("Step 3 · Embedding + FAISS 索引 · 交付标准验证")
print("=" * 60)

# ──────────────────────────────────────────────
# 测试 1：Embedding 编码
# ──────────────────────────────────────────────
print("\n[测试 1] Embedding 编码")
from services.embedder import encode_texts, serialize_vector, deserialize_vector
import numpy as np

texts = ['Hello world', '你好世界', 'I appreciate the offer']
vecs = encode_texts(texts)
print(f"  Shape: {vecs.shape}")
assert vecs.shape == (3, 384), f"Embedding 维度错误: {vecs.shape}"
assert vecs.dtype == np.float32, f"dtype 错误: {vecs.dtype}"
print("  ✅ Embedding 编码正常")

# ──────────────────────────────────────────────
# 测试 2：序列化 / 反序列化
# ──────────────────────────────────────────────
print("\n[测试 2] 向量序列化 / 反序列化")
vec = vecs[0]
data = serialize_vector(vec)
print(f"  序列化字节数: {len(data)}（期望 {384*4}=1536）")
assert len(data) == 384 * 4, f"序列化长度错误: {len(data)}"

restored = deserialize_vector(data)
print(f"  反序列化 shape: {restored.shape}")
assert restored.shape == (384,), f"反序列化 shape 错误: {restored.shape}"
assert np.allclose(vec, restored, atol=1e-6), "序列化/反序列化精度损失"
print("  ✅ 序列化/反序列化正常")

# ──────────────────────────────────────────────
# 测试 3：FAISS 索引构建与检索
# ──────────────────────────────────────────────
print("\n[测试 3] FAISS 索引构建与检索")
from services.faiss_index import build_index, search_similar, add_to_index

test_texts = ['Hello', 'Hi there', 'Goodbye', 'Good morning', 'See you']
node_ids = [1, 2, 3, 4, 5]
test_vecs = encode_texts(test_texts)
index, mapping = build_index(node_ids, test_vecs)

print(f"  索引大小: {index.ntotal}（期望 5）")
assert index.ntotal == 5, f"索引大小错误: {index.ntotal}"
assert len(mapping) == 5, f"映射大小错误: {len(mapping)}"

query = encode_texts(['Hello'])[0]
results = search_similar(index, mapping, query, k=3)
print(f"  Top-3: {results}")
assert len(results) == 3, f"结果数量错误: {len(results)}"
assert results[0][0] == 1, f"Top-1 应该是自己(node_id=1)，实际: {results[0][0]}"
assert results[0][1] > 0.99, f"自身相似度应接近 1，实际: {results[0][1]}"
print("  ✅ FAISS 检索正常")

# ──────────────────────────────────────────────
# 测试 4：增量添加
# ──────────────────────────────────────────────
print("\n[测试 4] 增量添加向量")
new_texts = ['Nice to meet you', 'How are you doing']
new_ids = [6, 7]
new_vecs = encode_texts(new_texts)
add_to_index(index, mapping, new_ids, new_vecs)

print(f"  增量后索引大小: {index.ntotal}（期望 7）")
assert index.ntotal == 7, f"增量后索引大小错误: {index.ntotal}"
assert len(mapping) == 7, f"增量后映射大小错误: {len(mapping)}"
assert mapping[5] == 6 and mapping[6] == 7, f"增量映射错误: {mapping}"

# 验证新增节点可检索
results2 = search_similar(index, mapping, encode_texts(['Nice to meet you'])[0], k=1)
assert results2[0][0] == 6, f"新增节点检索失败: {results2}"
print("  ✅ 增量添加正常")

# ──────────────────────────────────────────────
# 测试 5：索引持久化
# ──────────────────────────────────────────────
print("\n[测试 5] 索引持久化（保存 + 加载）")
from services.faiss_index import save_index, load_index

save_path = str(INDEX_DIR / "test.index")
save_index(index, mapping, save_path)
print(f"  索引已保存到 {save_path}")
assert os.path.exists(save_path), f"索引文件不存在: {save_path}"
assert os.path.exists(save_path + ".mapping.json"), "映射文件不存在"

loaded_index, loaded_mapping = load_index(save_path)
print(f"  加载后索引大小: {loaded_index.ntotal}（期望 7）")
assert loaded_index.ntotal == 7, f"加载后索引大小错误: {loaded_index.ntotal}"
assert loaded_mapping == mapping, f"加载后映射不一致"

# 验证加载后检索结果一致
results3 = search_similar(loaded_index, loaded_mapping, query, k=3)
assert results3[0][0] == 1, f"加载后检索 Top-1 错误: {results3[0][0]}"
print("  ✅ 持久化正常")

# ──────────────────────────────────────────────
# 测试 6：从数据库重建索引
# ──────────────────────────────────────────────
print("\n[测试 6] 从数据库重建索引")
from database import SessionLocal, init_db
from models.knowledge_node import KnowledgeNode
from services.faiss_index import rebuild_index_from_db

async def test_rebuild():
    await init_db()
    async with SessionLocal() as session:
        # 插入测试节点
        node = KnowledgeNode(
            en_text="Test sentence for FAISS rebuild",
            zh_text="FAISS 重建测试句子",
            show_name="TestFAISS",
            episode="S01E01",
            start_time=10.0,
            end_time=12.0,
            word_count=6,
            embedding=serialize_vector(encode_texts(["Test sentence for FAISS rebuild"])[0]),
            is_active=True,
        )
        session.add(node)
        await session.commit()
        await session.refresh(node)
        print(f"  插入测试节点 id={node.id}")

        # 重建索引
        rebuilt_index, rebuilt_mapping = await rebuild_index_from_db(
            session,
            force_full=True,
            index_path=str(INDEX_DIR / "knowledge.index"),
        )
        print(f"  重建后索引大小: {rebuilt_index.ntotal}")
        assert rebuilt_index.ntotal >= 1, "重建后索引为空"
        assert node.id in rebuilt_mapping.values(), f"节点 {node.id} 未出现在映射中"
        print("  ✅ 索引重建成功")

asyncio.run(test_rebuild())

# ──────────────────────────────────────────────
# 验证索引文件生成
# ──────────────────────────────────────────────
print("\n[验证] 索引文件生成")
import subprocess
result = subprocess.run(["ls", "-lh", str(INDEX_DIR)], capture_output=True, text=True)
print(result.stdout)
assert "knowledge.index" in result.stdout, "knowledge.index 文件不存在"
assert "knowledge.index.mapping.json" in result.stdout, "映射文件不存在"

print("=" * 60)
print("✅ Step 3 全部交付标准验证通过！")
print("=" * 60)
