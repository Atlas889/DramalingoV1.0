"""
DramaLingo 冒烟测试脚本
运行方式：.venv\Scripts\python.exe tests\integration\smoke_test.py
要求：服务已在 http://127.0.0.1:8000 运行
"""

import io
import json
import sys
import time
import httpx

BASE_URL        = "http://127.0.0.1:8000"
TIMEOUT         = 15.0
SEARCH_TIMEOUT  = 120.0   # 首次调用需加载 embedding 模型，最长等待 2 分钟

# ── 测试统计 ────────────────────────────────────────────────
_passed = 0
_failed = 0
_results: list[dict] = []


def _print_sep(title: str = ""):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * pad}")
    else:
        print("─" * width)


def check(name: str, passed: bool, detail: str = ""):
    global _passed, _failed
    status = "✓ PASS" if passed else "✗ FAIL"
    color  = "\033[32m" if passed else "\033[31m"
    reset  = "\033[0m"
    print(f"  {color}{status}{reset}  {name}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")
    if passed:
        _passed += 1
    else:
        _failed += 1
    _results.append({"name": name, "passed": passed, "detail": detail})


def get(path: str, **params) -> tuple[int, dict | None]:
    try:
        r = httpx.get(f"{BASE_URL}{path}", params=params, timeout=TIMEOUT)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"_raw": r.text[:300]}
    except Exception as e:
        return -1, {"error": str(e)}


def post_json(path: str, body: dict) -> tuple[int, dict | None]:
    try:
        r = httpx.post(f"{BASE_URL}{path}", json=body, timeout=TIMEOUT)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"_raw": r.text[:300]}
    except Exception as e:
        return -1, {"error": str(e)}


def post_form(path: str, data: dict, files: dict | None = None) -> tuple[int, dict | None]:
    try:
        r = httpx.post(f"{BASE_URL}{path}", data=data, files=files, timeout=TIMEOUT)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"_raw": r.text[:300]}
    except Exception as e:
        return -1, {"error": str(e)}


# ────────────────────────────────────────────────────────────
# P0 · 系统基线
# ────────────────────────────────────────────────────────────
def test_p0_health():
    _print_sep("P0 · 系统健康检查")
    code, data = get("/health")
    check("GET /health 返回 200",           code == 200,           f"status={code}")
    check("返回 status=healthy",            data and data.get("status") == "healthy",
          json.dumps(data, ensure_ascii=False))
    check("返回 version 字段",              data and "version" in data)


def test_p0_docs():
    _print_sep("P0 · API 文档可访问")
    try:
        r = httpx.get(f"{BASE_URL}/docs", timeout=TIMEOUT)
        check("GET /docs 返回 200", r.status_code == 200, f"status={r.status_code}")
    except Exception as e:
        check("GET /docs 返回 200", False, str(e))


# ────────────────────────────────────────────────────────────
# P0 · 语义检索
# ────────────────────────────────────────────────────────────
def test_p0_search():
    _print_sep("P0 · 语义检索")

    print("  [提示] 首次调用需加载 Embedding 模型，最长等待 2 分钟...")

    # 基础检索（空库应返回空列表或关键词降级）—— 使用专用超时
    try:
        r = httpx.get(f"{BASE_URL}/search", params={"q": "表达感谢"}, timeout=SEARCH_TIMEOUT)
        try:
            code, data = r.status_code, r.json()
        except Exception:
            code, data = r.status_code, {"_raw": r.text[:300]}
    except Exception as e:
        code, data = -1, {"error": str(e)}
        print(f"  [异常] {e}")
    check("GET /search?q=表达感谢 返回 200",    code == 200, f"status={code}")
    check("返回 query 字段",                   data and "query" in data)
    check("返回 results 数组",                 data and isinstance(data.get("results"), list))
    check("返回 search_mode 字段",             data and "search_mode" in data,
          f"search_mode={data.get('search_mode') if data else 'N/A'}")

    # 参数校验：空查询词
    code2, data2 = get("/search", q="")
    check("空查询词返回 400 或 422",            code2 in (400, 422), f"status={code2}")

    # 检索状态接口
    code3, data3 = get("/search/status")
    check("GET /search/status 返回 200",       code3 == 200, f"status={code3}")
    check("返回 index_size 字段",              data3 and "index_size" in data3,
          json.dumps(data3, ensure_ascii=False) if data3 else "")


# ────────────────────────────────────────────────────────────
# P1 · 知识节点管理
# ────────────────────────────────────────────────────────────
def test_p1_knowledge():
    _print_sep("P1 · 知识节点管理")

    # 节点列表
    code, data = get("/knowledge/nodes")
    check("GET /knowledge/nodes 返回 200",     code == 200, f"status={code}")
    check("返回 items 数组",                   data and isinstance(data.get("items"), list))
    check("返回 total 字段",                   data and "total" in data,
          f"total={data.get('total') if data else 'N/A'}")

    # 知识库统计
    code2, data2 = get("/knowledge/status")
    check("GET /knowledge/status 返回 200",    code2 == 200, f"status={code2}")
    check("返回 total_nodes 字段",             data2 and "total_nodes" in data2,
          json.dumps(data2, ensure_ascii=False) if data2 else "")

    # 剧集列表
    code3, data3 = get("/knowledge/shows")
    check("GET /knowledge/shows 返回 200",     code3 == 200, f"status={code3}")

    # 不存在的节点
    code4, _ = get("/knowledge/nodes/999999")
    check("查询不存在的节点返回 404",           code4 == 404, f"status={code4}")


# ────────────────────────────────────────────────────────────
# P1 · 字幕上传
# ────────────────────────────────────────────────────────────
_TEST_SRT = """\
1
00:00:01,000 --> 00:00:03,500
Thank you so much for everything.

2
00:00:04,000 --> 00:00:07,000
I really appreciate your help.

3
00:00:08,000 --> 00:00:11,000
You are the best friend I have ever had.
"""

def test_p1_subtitle_upload():
    _print_sep("P1 · 字幕上传")

    srt_bytes = _TEST_SRT.encode("utf-8")

    # 单语字幕上传
    code, data = post_form(
        "/subtitles/upload",
        data={"show_name": "SmokeTest", "episode": "S01E01", "overwrite": "true"},
        files={"file": ("test.srt", io.BytesIO(srt_bytes), "text/plain")},
    )
    check("POST /subtitles/upload 返回 200",   code == 200, f"status={code}")
    check("返回 job_id",                       data and "job_id" in data,
          json.dumps(data, ensure_ascii=False) if data else "")

    job_id = data.get("job_id") if data else None

    # 不支持的格式
    code2, data2 = post_form(
        "/subtitles/upload",
        data={"show_name": "SmokeTest", "episode": "S01E01"},
        files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    check("上传不支持格式返回 400",             code2 == 400, f"status={code2}")

    # 双语上传接口
    code3, data3 = post_form(
        "/subtitles/upload-bilingual",
        data={"show_name": "SmokeTest", "episode": "S01E01", "overwrite": "true"},
        files={
            "en_file": ("en.srt", io.BytesIO(srt_bytes), "text/plain"),
            "zh_file": ("zh.srt", io.BytesIO("1\n00:00:01,000 --> 00:00:03,500\n非常感谢你的一切。\n\n".encode()), "text/plain"),
        },
    )
    check("POST /subtitles/upload-bilingual 返回 200", code3 == 200, f"status={code3}")

    return job_id


# ────────────────────────────────────────────────────────────
# P1 · 字幕记录查询
# ────────────────────────────────────────────────────────────
def test_p1_subtitle_list():
    _print_sep("P1 · 字幕记录查询")

    # 字幕列表
    code, data = get("/subtitles/list")
    check("GET /subtitles/list 返回 200",      code == 200, f"status={code}")
    check("返回 items 数组",                   data and isinstance(data.get("items"), list))
    check("返回 total 字段",                   data and "total" in data,
          f"total={data.get('total') if data else 'N/A'}")

    # 字幕统计
    code2, data2 = get("/subtitles/stats")
    check("GET /subtitles/stats 返回 200",     code2 == 200, f"status={code2}")
    check("返回 total_lines 字段",             data2 and "total_lines" in data2,
          json.dumps(data2, ensure_ascii=False) if data2 else "")

    # 剧名列表
    code3, data3 = get("/subtitles/shows")
    check("GET /subtitles/shows 返回 200",     code3 == 200, f"status={code3}")


# ────────────────────────────────────────────────────────────
# P2 · 任务队列
# ────────────────────────────────────────────────────────────
def test_p2_jobs(job_id: str | None = None):
    _print_sep("P2 · 任务队列")

    # 任务列表
    code, data = get("/jobs")
    check("GET /jobs 返回 200",               code == 200, f"status={code}")
    check("返回 tasks 数组",                  data and isinstance(data.get("tasks"), list),
          f"total={data.get('total') if data else 'N/A'}")

    # 查询不存在的任务
    code2, _ = get("/jobs/nonexistent-job-id")
    check("查询不存在任务返回 404",            code2 == 404, f"status={code2}")

    # 轮询刚提交的任务
    if job_id:
        time.sleep(2)
        code3, data3 = get(f"/jobs/{job_id}")
        check(f"GET /jobs/{job_id[:8]}... 返回 200", code3 == 200, f"status={code3}")
        check("任务状态字段存在",              data3 and "status" in data3,
              json.dumps(data3, ensure_ascii=False) if data3 else "")


# ────────────────────────────────────────────────────────────
# P2 · 视频管理
# ────────────────────────────────────────────────────────────
def test_p2_videos():
    _print_sep("P2 · 视频管理")

    code, data = get("/videos/list")
    check("GET /videos/list 返回 200",        code == 200, f"status={code}")

    code2, data2 = get("/videos/status")
    check("GET /videos/status 返回 200",      code2 == 200, f"status={code2}")
    check("返回 total_videos 字段",           data2 and "total_videos" in data2,
          json.dumps(data2, ensure_ascii=False) if data2 else "")


# ────────────────────────────────────────────────────────────
# P2 · 切片管理
# ────────────────────────────────────────────────────────────
def test_p2_clips():
    _print_sep("P2 · 切片管理")

    code, data = get("/clips/quality-report")
    check("GET /clips/quality-report 返回 200", code == 200, f"status={code}")
    check("返回 summary 字段",                data and "summary" in data,
          json.dumps(data, ensure_ascii=False)[:200] if data else "")


# ────────────────────────────────────────────────────────────
# 汇总报告
# ────────────────────────────────────────────────────────────
def print_summary():
    _print_sep("测试结果汇总")
    total = _passed + _failed
    print(f"\n  总计：{total} 项   通过：{_passed}   失败：{_failed}\n")

    if _failed > 0:
        print("  ── 失败项明细 ──")
        for r in _results:
            if not r["passed"]:
                print(f"  ✗ {r['name']}")
                if r["detail"]:
                    print(f"    {r['detail'][:120]}")
    print()
    return _failed == 0


# ────────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  DramaLingo 冒烟测试")
    print("=" * 60)

    # 确认服务在线
    try:
        httpx.get(f"{BASE_URL}/health", timeout=5)
    except Exception:
        print(f"\n[错误] 无法连接到 {BASE_URL}，请先启动服务。\n")
        sys.exit(1)

    test_p0_health()
    test_p0_docs()
    test_p0_search()
    test_p1_knowledge()
    job_id = test_p1_subtitle_upload()
    test_p1_subtitle_list()
    test_p2_jobs(job_id)
    test_p2_videos()
    test_p2_clips()

    ok = print_summary()
    sys.exit(0 if ok else 1)
