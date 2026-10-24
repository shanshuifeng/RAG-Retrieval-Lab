"""E1 · BM25 超参网格搜索（k1 × b）—— P3

⭐ 设计要点：
   · 分词与 df/tf 统计只做【一次】（build_base），之后每组参数只重建加权矩阵
     → 16 组从 5.6 小时降到约 2 分钟
   · 只在 dev 上跑（⭐ 铁律）；test 集在配置锁定后才碰
   · 每组记录完整指标 + 耗时，写入 results/
   · 用主指标 nDCG@10 选最优（与 docs/05 §3.5 约定一致）

用法：
  .venv\\Scripts\\python.exe scripts/grid_bm25.py --split dev
  .venv\\Scripts\\python.exe scripts/grid_bm25.py --split dev --quick   # 4 组粗调
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_split  # noqa: E402
from src.eval.metrics import evaluate  # noqa: E402
from src.retrieval.bm25_fast import BM25Fast  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "index"

# docs/06 §1.3 规定的网格
K1_GRID = [0.9, 1.2, 1.5, 2.0]
B_GRID = [0.3, 0.5, 0.75, 0.9]
# 粗调（每组取中间值，4 组）
K1_QUICK = [0.9, 1.5, 2.0]
B_QUICK = [0.3, 0.75]


def main() -> int:
    ap = argparse.ArgumentParser(description="BM25 网格搜索")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--quick", action="store_true", help="粗调网格（6 组）")
    ap.add_argument("--primary", default="ndcg@10",
                    help="选优主指标（默认 ndcg@10）")
    args = ap.parse_args()

    if args.split == "test":
        print("❌ 网格调参禁止在 test 上运行（docs/05 铁律）")
        return 2

    k1s = K1_QUICK if args.quick else K1_GRID
    bs = B_QUICK if args.quick else B_GRID
    combos = list(itertools.product(k1s, bs))

    print("=" * 78)
    print(f"  E1 · BM25 网格搜索  |  split={args.split}  "
          f"|  {len(combos)} 组（{len(k1s)} k1 × {len(bs)} b）")
    print("=" * 78)

    t_start = time.time()

    # ── 1. 加载数据 ──
    print("\n[1/4] 加载数据 ...")
    split = load_split(args.split)
    queries = {q: split.relevant_of(q) for q in split.query_list()}
    print(f"  语料 {split.n_passages:,} 段 | 评测 query {len(queries):,} 条")

    # ── 2. 基础统计（⭐ 只做一次，与 k1/b 无关）──
    print("\n[2/4] 分词 + 基础统计（一次性）...")
    base_cache = CACHE / f"bm25fast_base_{args.split}.pkl"
    retriever = BM25Fast()
    if base_cache.exists():
        print(f"  命中基础统计缓存 {base_cache.name}")
        retriever = BM25Fast.load(base_cache)
        if not retriever.can_set_params:
            print("  ⚠️ 旧缓存无基础统计，重建 ...")
            retriever.build_base(split, verbose=True)
            retriever.save(base_cache)
    else:
        retriever.build_base(split, verbose=True)
        retriever.save(base_cache)
        print(f"  已缓存 → {base_cache.name}")
    print(f"  词表 {len(retriever.vocab):,} | tokens {retriever._n_tokens:,}")

    # ── 3. 逐组调参 ──
    print(f"\n[3/4] 遍历 {len(combos)} 组参数 ...")
    print(f"  {'#':>3}  {'k1':>5}  {'b':>5}  {'R@10':>8}  {'nDCG@10':>9}  "
          f"{'MRR@10':>8}  {'耗时':>8}")
    print("  " + "-" * 62)

    rows: list[dict] = []
    for i, (k1, b) in enumerate(combos, start=1):
        t0 = time.time()
        retriever.set_params(k1=k1, b=b)
        run = retriever.run_all(split, top_k=args.top_k, verbose=False)
        res = evaluate(queries, run, ks=[10, 100],
                       metrics=("recall", "ndcg", "mrr"))
        dt = time.time() - t0

        agg = res["aggregate"]
        rec = {
            "k1": k1, "b": b,
            "recall@10": round(agg["recall@10"]["mean"], 6),
            "recall@100": round(agg["recall@100"]["mean"], 6),
            "ndcg@10": round(agg["ndcg@10"]["mean"], 6),
            "ndcg@100": round(agg["ndcg@100"]["mean"], 6),
            "mrr@10": round(agg["mrr@10"]["mean"], 6),
            "mrr@100": round(agg["mrr@100"]["mean"], 6),
            "sec": round(dt, 2),
        }
        rows.append(rec)
        print(f"  {i:>3}  {k1:>5.1f}  {b:>5.2f}  {rec['recall@10']:>8.4f}  "
              f"{rec['ndcg@10']:>9.4f}  {rec['mrr@10']:>8.4f}  {dt:>7.1f}s")

    # ── 4. 选优 ──
    print(f"\n[4/4] 按 {args.primary} 选优 ...")
    primary = args.primary
    if primary not in rows[0]:
        print(f"❌ 未知主指标 {primary}，可选：{list(rows[0].keys())}")
        return 1
    best = max(rows, key=lambda r: r[primary])
    default = next((r for r in rows if r["k1"] == 1.5 and r["b"] == 0.75), None)

    print(f"  ⭐ 最优：k1={best['k1']}, b={best['b']}  "
          f"{primary}={best[primary]:.6f}")
    if default:
        gain = best[primary] - default[primary]
        rel = 100 * gain / default[primary] if default[primary] else 0
        print(f"  默认（k1=1.5,b=0.75）：{primary}={default[primary]:.6f}")
        print(f"  提升：{gain:+.6f}（{rel:+.2f}%）")
        if abs(rel) < 1:
            print("\n  ⚠️ 提升 < 1%，按 docs/05 要求应做显著性检验后再下结论")

    # ── 汇总输出 ──
    print()
    print("  完整结果（按主指标降序）：")
    print(f"  {'k1':>5}  {'b':>5}  {'R@10':>8}  {'nDCG@10':>9}  {'MRR@10':>8}")
    print("  " + "-" * 46)
    for r in sorted(rows, key=lambda x: -x[primary]):
        mark = " ⭐" if r is best else ""
        print(f"  {r['k1']:>5.1f}  {r['b']:>5.2f}  {r['recall@10']:>8.4f}  "
              f"{r['ndcg@10']:>9.4f}  {r['mrr@10']:>8.4f}{mark}")

    total = time.time() - t_start
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = RESULTS / f"{stamp}_E1_bm25_grid_{args.split}.json"
    payload = {
        "experiment_id": "E1/B1",
        "method": "bm25_fast（与 rank_bm25 等价，已验证）",
        "split": args.split,
        "date": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "corpus_size": split.n_passages,
            "queries": len(queries),
            "tokenizer": "jieba",
            "top_k": args.top_k,
            "k1_grid": k1s,
            "b_grid": bs,
            "primary_metric": primary,
        },
        "grid": rows,
        "best": best,
        "baseline_default": default,
        "total_sec": round(total, 1),
        "note": "⭐ 仅在 dev 上调参；test 集未使用",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  结果 → {out.relative_to(ROOT)}")
    print(f"  总用时 {total:.1f}s（{len(combos)} 组）")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())