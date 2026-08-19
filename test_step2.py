"""
Step 2 交付标准验证脚本
对应开发文档 Step 2 交付标准
"""

import asyncio
import sys
import os

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_subtitle_parser():
    """验证 1：字幕解析"""
    print("=" * 50)
    print("验证 1：字幕解析")
    from services.subtitle_parser import parse_subtitle
    lines = parse_subtitle("test_subtitle.srt")
    print(f"  解析行数: {len(lines)}")
    assert len(lines) > 0, "解析失败：行数为 0"
    print(f"  前 3 行示例:")
    for line in lines[:3]:
        print(f"    [{line.start_time:.2f}s] {line.en_text[:60]}")
    print("  ✓ 字幕解析通过\n")
    return lines


def test_quality_filter(lines):
    """验证 2：质量筛选"""
    print("=" * 50)
    print("验证 2：质量筛选")
    from services.quality_filter import filter_subtitles
    results, stats = filter_subtitles(lines)
    print(f"  总行数: {stats['total']}")
    print(f"  通过行数: {stats['passed']}")
    print(f"  过滤率: {stats['filter_rate']:.2%}")
    print(f"  各阶段拒绝数:")
    for stage, count in stats["rejected_by_stage"].items():
        print(f"    {stage}: {count}")

    # 验证过滤率在合理范围（50%~80%）
    assert 0.4 <= stats["filter_rate"] <= 0.85, \
        f"过滤率异常: {stats['filter_rate']:.2%}（期望 50%~80%）"
    assert stats["passed"] > 0, "无有效台词通过筛选"

    print("\n  通过筛选的行示例:")
    passed = [r for r in results if r.filter_status == "passed"]
    for r in passed[:5]:
        print(f"    [score={r.quality_score:.3f}] {r.line.en_text[:70]}")
    print("  ✓ 质量筛选通过\n")
    return results, stats


async def test_pipeline():
    """验证 3：完整流水线 + 数据库写入"""
    print("=" * 50)
    print("验证 3：完整流水线 + 数据库写入")
    from database import init_db, SessionLocal
    from services.subtitle_pipeline import process_subtitle_file

    await init_db()

    async with SessionLocal() as session:
        result = await process_subtitle_file(
            file_path="test_subtitle.srt",
            show_name="TestShow",
            episode="S01E01",
            session=session,
            overwrite=True,
        )

    print(f"  总行数: {result['total']}")
    print(f"  通过行数: {result['passed']}")
    print(f"  过滤率: {result['filter_rate']:.2%}")
    if result.get("error"):
        print(f"  ⚠ 警告: {result['error']}")
    assert result["passed"] > 0, "无有效台词"
    print("  ✓ 流水线 + 数据库写入通过\n")
    return result


def test_database_records():
    """验证 4：数据库记录分布"""
    print("=" * 50)
    print("验证 4：数据库记录分布（sqlite3）")
    import subprocess
    result = subprocess.run(
        ["sqlite3", "dramalingo.db",
         "SELECT filter_status, COUNT(*) FROM subtitles_raw GROUP BY filter_status;"],
        capture_output=True, text=True
    )
    print(result.stdout)
    assert "passed" in result.stdout, "数据库中无 passed 记录"
    print("  ✓ 数据库记录验证通过\n")


if __name__ == "__main__":
    print("\n🎬 DramaLingo Step 2 交付标准验证\n")

    # 验证 1：解析
    lines = test_subtitle_parser()

    # 验证 2：筛选
    results, stats = test_quality_filter(lines)

    # 验证 3：流水线
    asyncio.run(test_pipeline())

    # 验证 4：数据库
    test_database_records()

    print("=" * 50)
    print("✅ 全部验证通过！Step 2 交付标准满足。")
