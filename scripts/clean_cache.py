"""缓存清理 —— P4 待办（审查 L6）

⭐ 起因（2026-10-25 审查 L6）：
   data/index/ 里堆积了 118 MB 的过期货：
     · rerank_scores_d2_dev_debug500q.json  2.4 MB  500 条调试缓存（已被 8640q 取代）
     · bm25_dev_k11.5_b0.75.pkl           116.3 MB  慢版索引（仅等价性验证用）
     · run_bm25_dev_*.json                 30 MB   中间产物

⭐ 安全原则（重要）：
   · 默认 --dry-run，只报告不删除
   · 只删【可重新生成】的缓存：索引、分数、中间 run
   · 【绝不删】results/*.json（实验结论）与 results/TEST_USED.lock
   · 删前逐项列出大小与理由

用法：
  .venv\\Scripts\\python.exe scripts/clean_cache.py             # 只报告
  .venv\\Scripts\\python.exe scripts/clean_cache.py --apply     # 真删
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "index"
RESULTS = ROOT / "results"


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def collect() -> list[dict]:
    """收集候选删除项（含理由）。"""
    items: list[dict] = []

    # ① 调试子集缓存（已被全量取代）
    for p in CACHE.glob("*debug*q.json"):
        items.append({"path": p, "reason": "调试子集缓存，已被全量取代",
                      "size": p.stat().st_size})
    for p in CACHE.glob("*_500q.json"):
        items.append({"path": p, "reason": "500 条子集缓存", "size": p.stat().st_size})

    # ② 慢版 BM25 索引（仅等价性验证用，可重建）
    for p in CACHE.glob("bm25_dev_k1*.pkl"):
        items.append({"path": p, "reason": "rank_bm25 慢版索引（仅验证用，可重建）",
                      "size": p.stat().st_size})

    # ②b 已被快版取代的旧命名（bm25fast_dev_k1*，实际用的是 bm25fast_base_dev）
    for p in CACHE.glob("bm25fast_dev_k1*.pkl"):
        items.append({"path": p, "reason": "旧命名的 BM25 索引（已被 base 版取代）",
                      "size": p.stat().st_size})

    # ②c E4 分块缓存（4 组策略 × BM25/稠密，约 2 GB，可重跑生成）
    for pattern, reason in [
        ("bm25fast_chunked_*.pkl", "E4 分块 BM25 索引（可重跑生成）"),
        ("dense_base_chunked_*.faiss", "E4 分块稠密索引（可重跑生成，最占空间）"),
        ("dense_base_chunked_*.pkl", "E4 分块稠密元数据"),
    ]:
        for p in CACHE.glob(pattern):
            items.append({"path": p, "reason": reason, "size": p.stat().st_size})

    # ③ ⚠️ 不删除 run 文件！
    #    2026-10-25 教训：初版规则把 *_run.json 当"中间产物"删了，
    #    导致 run_e2_hybrid.py 等脚本立刻报 FileNotFoundError ——
    #    这些 run 是【下游实验的输入】，不是可丢弃的中间件。
    #    （且它们已在 .gitignore 中，可用 --regen 从索引缓存重算）

    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正删除（默认只报告）")
    args = ap.parse_args()

    print("=" * 76)
    print(f"  缓存清理  {'【真删模式】' if args.apply else '【演练模式，不删任何文件】'}")
    print("=" * 76)

    # 安全检查：test 的 run 与锁必须在
    lock = RESULTS / "TEST_USED.lock"
    test_runs = list(RESULTS.glob("*test_final*_run.json"))
    print(f"\n  安全检查：")
    print(f"    results/TEST_USED.lock        存在: {lock.exists()}")
    print(f"    test 的 run 文件数               : {len(test_runs)}（必须全部保留）")

    items = collect()
    if not items:
        print("\n  ✅ 无可清理项")
        return 0

    total = sum(i["size"] for i in items)
    print(f"\n  可清理 {len(items)} 项，合计 {human(total)}：\n")
    print(f"  {'文件':<48} {'大小':>10}  理由")
    print("  " + "-" * 92)
    for it in sorted(items, key=lambda x: -x["size"]):
        name = it["path"].name
        print(f"  {name[:48]:<48} {human(it['size']):>10}  {it['reason']}")

    # ⭐ 再次确认：绝不碰 test 的东西
    for it in items:
        assert "TEST_USED" not in it["path"].name, "安全检查失败：不能删锁文件"
        assert "test_final" not in it["path"].name, "安全检查失败：不能删 test run"

    print(f"\n  ⭐ 保留确认：")
    print(f"    · results/*_test_final*_run.json（{len(test_runs)} 个）—— test 唯一原始证据")
    print(f"    · results/TEST_USED.lock —— 一次性锁")
    print(f"    · results/*.json（非 run）—— 实验结论")

    if not args.apply:
        print(f"\n  这是演练模式。确认删除请加 --apply")
        print(f"  预计可释放 {human(total)}")
    else:
        freed = 0
        for it in items:
            try:
                size = it["path"].stat().st_size
                it["path"].unlink()
                freed += size
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠️ 删除失败 {it['path'].name}: {e}")
        print(f"\n  ✅ 已释放 {human(freed)}")

    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())