"""验证 BM25Fast 与 rank_bm25 的等价性 + 速度对比

⭐ 这是 P1 的验收脚本，必须在全量数据上通过才算提速成功。
   判据：
     1. 同参数下 top-100 重合率 > 99%（理想情况 100%）
     2. 分数排序一致（允许浮点误差）
     3. 速度提升显著（目标 <10 ms/query）

用法：
  # 小数据快速验证（先跑这个，几秒钟）
  .venv\\Scripts\\python.exe scripts\\verify_bm25_fast.py --split dev --n 2000

  # 全量验证（10 万段，约 3-5 分钟）
  .venv\\Scripts\\python.exe scripts\\verify_bm25_fast.py --split dev --n 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_split  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.bm25_fast import BM25Fast  # noqa: E402

CACHE = ROOT / "data" / "index"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=2000,
                    help="验证的 query 数（0 = 全部）")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    print("=" * 74)
    print(f"  BM25Fast 等价性验证  split={args.split}  n={args.n or 'all'}")
    print("=" * 74)

    split = load_split(args.split)
    print(f"  语料 {split.n_passages:,} 段 | query {split.n_queries:,}")

    # ── 1. 建快版索引（带缓存，避免重复 160s）──
    print("\n[1/3] 建 BM25Fast 索引 ...")
    fast_cache = CACHE / f"bm25fast_{args.split}_k1{args.k1}_b{args.b}.pkl"
    if fast_cache.exists():
        print(f"  命中缓存 {fast_cache.name}")
        t0 = time.time()
        fast = BM25Fast.load(fast_cache)
        print(f"  载入 {time.time()-t0:.1f}s")
    else:
        t0 = time.time()
        fast = BM25Fast(k1=args.k1, b=args.b)
        fast.build(split, verbose=True)
        print(f"  BM25Fast 建索引 {time.time()-t0:.1f}s")
        fast.save(fast_cache)
        print(f"  已缓存 → {fast_cache.name}")
    st = fast.stats()
    print(f"  词表 {st['n_vocab']:,} | nnz {st['nnz']:,} | tokens {st['n_tokens']:,}")

    # ── 2. 载入/建慢版索引（复用已有缓存）──
    print("\n[2/3] 准备 rank_bm25 基准 ...")
    slow_cache = CACHE / f"bm25_{args.split}_k1{args.k1}_b{args.b}.pkl"
    if slow_cache.exists():
        print(f"  命中缓存 {slow_cache.name}")
        slow = BM25Retriever.load(slow_cache)
    else:
        print("  无缓存，现场建（约 70s）...")
        slow = BM25Retriever(k1=args.k1, b=args.b)
        slow.build(split, verbose=True)
        slow.save(slow_cache)

    # ── 3. 逐 query 对比 ──
    print("\n[3/3] 逐 query 对比 ...")
    qids = split.query_list(with_relevant_only=True)
    if args.n:
        qids = qids[:args.n]
    print(f"  验证 {len(qids):,} 条 query，top_k={args.top_k}")

    t_fast = 0.0
    t_slow = 0.0
    overlap_sum = 0.0
    exact = 0
    max_score_diff = 0.0

    for i, qid in enumerate(qids, start=1):
        q = split.queries[qid]

        s = time.perf_counter()
        res_fast = fast.search(q, top_k=args.top_k)
        t_fast += time.perf_counter() - s

        s = time.perf_counter()
        res_slow = slow.search(q, top_k=args.top_k)
        t_slow += time.perf_counter() - s

        ids_fast = [p for p, _ in res_fast]
        ids_slow = [p for p, _ in res_slow]

        # 重合率（用集合交）
        inter = len(set(ids_fast) & set(ids_slow))
        denom = max(len(ids_slow), 1)
        overlap_sum += inter / denom
        if ids_fast == ids_slow:
            exact += 1

        # 分数对比（同一 docid）
        d_fast = dict(res_fast)
        for pid, sc in res_slow:
            if pid in d_fast:
                max_score_diff = max(max_score_diff, abs(d_fast[pid] - sc))

        if i % 500 == 0:
            print(f"    {i:,}/{len(qids):,} ...", flush=True)

    n = len(qids)
    avg_overlap = overlap_sum / max(n, 1)
    ms_fast = 1000 * t_fast / max(n, 1)
    ms_slow = 1000 * t_slow / max(n, 1)
    speedup = ms_slow / max(ms_fast, 1e-9)

    print()
    print("  " + "-" * 60)
    print(f"  top-{args.top_k} 平均重合率 : {avg_overlap*100:.3f}%")
    print(f"  完全一致的 query 数        : {exact:,} / {n:,} "
          f"({100*exact/max(n,1):.2f}%)")
    print(f"  最大分数差                 : {max_score_diff:.3e}")
    print(f"  rank_bm25 耗时             : {ms_slow:.2f} ms/query")
    print(f"  BM25Fast 耗时              : {ms_fast:.2f} ms/query")
    print(f"  ⭐ 加速比                   : {speedup:.1f}×")
    print("  " + "-" * 60)

    # ── 验收 ──
    ok = True

    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
        ok = ok and cond

    print()
    check("top-k 重合率 > 99%", avg_overlap > 0.99, f"{avg_overlap*100:.3f}%")
    check("分数差 < 1e-6", max_score_diff < 1e-6, f"{max_score_diff:.3e}")
    check("速度提升 > 5×", speedup > 5, f"{speedup:.1f}×")

    print()
    print("=" * 74)
    print("  ⭐ " + ("P1 验收通过 —— 可替换后端" if ok
                     else "验收未通过，需排查公式对齐"))
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())