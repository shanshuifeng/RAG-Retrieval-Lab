"""E1 · BM25 基线实验 —— M2.4

⭐ 本脚本是项目【第一个真实实验】。它的作用不只是拿一个数字，而是：
   1. 验证「数据 → 检索 → 评测」整条链路正确
   2. 给出与文献可对照的基线量级
   3. 为后续所有方法（稠密/混合/重排）提供参照系

纪律（docs/05 §五）：
   · 先在 dev 上跑（默认），test 只在配置锁定后跑一次
   · 记录全部配置参数 + 成本（建索引/检索耗时、内存）
   · 结果写入 results/

用法：
  # dev 上跑（调参/开发用）
  .venv\\Scripts\\python.exe scripts\\run_e1_bm25.py --split dev

  # test 上跑（⭐ 仅配置锁定后，且只跑一次）
  .venv\\Scripts\\python.exe scripts\\run_e1_bm25.py --split test --confirm-final
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_split  # noqa: E402
from src.eval.metrics import evaluate, format_table  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.bm25_fast import BM25Fast  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "index"


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 BM25 基线")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--ks", default="1,5,10,20,100")
    ap.add_argument("--no-cache", action="store_true", help="强制重建索引")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条 query（调试用）")
    ap.add_argument("--backend", default="fast", choices=["fast", "slow"],
                    help="⭐ fast=稀疏矩阵实现(0.8ms/q)  slow=rank_bm25(147ms/q)，"
                         "两者已验证等价（top-100 重合率 100%%）")
    ap.add_argument("--confirm-final", action="store_true",
                    help="⭐ 确认这是最终 test 评测（test 只该跑一次）")
    args = ap.parse_args()

    if args.split == "test" and not args.confirm_final:
        print("❌ 拒绝在 test 上运行：test 集只该在配置锁定后跑一次。")
        print("   调参请用 --split dev；确认最终评测请加 --confirm-final")
        return 2

    ks = [int(x) for x in args.ks.split(",")]
    print("=" * 74)
    print(f"  E1 · BM25 基线  |  split={args.split}  k1={args.k1}  b={args.b}")
    print("=" * 74)

    t_all = time.time()
    print("\n[1/4] 加载数据 ...")
    split = load_split(args.split)
    print(f"  语料 {split.n_passages:,} 段 | 查询 {split.n_queries:,} 条 "
          f"| qrels {split.n_relevant_pairs:,} 对")

    print("\n[2/4] 建 BM25 索引 ...")
    tag = "bm25fast" if args.backend == "fast" else "bm25"
    cache = CACHE / f"{tag}_{args.split}_k1{args.k1}_b{args.b}.pkl"
    if cache.exists() and not args.no_cache:
        print(f"  命中缓存 {cache.relative_to(ROOT)}")
        retriever = (BM25Fast.load(cache) if args.backend == "fast"
                     else BM25Retriever.load(cache))
        print(f"  {args.backend} 后端 | k1={retriever.k1} b={retriever.b} "
              f"（建索引原始耗时 {retriever._build_sec:.1f}s）")
    else:
        retriever = ((BM25Fast(k1=args.k1, b=args.b) if args.backend == "fast"
                      else BM25Retriever(k1=args.k1, b=args.b)))
        retriever.build(split, verbose=True)
        retriever.save(cache)
        print(f"  索引已缓存 → {cache.relative_to(ROOT)}")

    print("\n[3/4] 批量检索 ...")
    run_cache = CACHE / (f"run_{tag}_{args.split}_k1{args.k1}"
                         f"_b{args.b}_top{args.top_k}.json")
    if run_cache.exists() and not args.no_cache:
        print(f"  命中检索缓存 {run_cache.relative_to(ROOT)}")
        run = json.loads(run_cache.read_text(encoding="utf-8"))
        ret_sec = 0.0
        print(f"  已加载 {len(run):,} 条 query 的检索结果（跳过 18 分钟检索）")
    else:
        t_ret = time.time()
        run = retriever.run_all(split, top_k=args.top_k, verbose=True)
        ret_sec = time.time() - t_ret
        run_cache.parent.mkdir(parents=True, exist_ok=True)
        run_cache.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
        print(f"  检索结果已缓存 → {run_cache.relative_to(ROOT)}")

    print("\n[4/4] 评测 ...")
    queries = {q: split.relevant_of(q) for q in split.query_list()}
    res = evaluate(queries, run, ks=ks, metrics=("recall", "mrr", "ndcg"),
                   skip_empty_relevant=True)
    print(f"  评测 query 数：{res['filtered']['queries_evaluated']:,}")
    print(f"  过滤（无标注）：{res['filtered']['empty_relevant']}")
    print()
    print(format_table(res["aggregate"], ks, ("recall", "mrr", "ndcg")))

    # ── 成本指标（⭐ docs/05 纪律④必须报告）──
    n_q = res["filtered"]["queries_evaluated"]
    cost = {
        "build_sec": round(retriever._build_sec, 2),
        "search_sec_total": round(ret_sec, 2),
        "search_ms_per_query": round(1000 * ret_sec / max(n_q, 1), 3),
        "n_tokens": retriever._n_tokens,
    }
    print(f"\n  成本：建索引 {cost['build_sec']}s | "
          f"检索 {cost['search_ms_per_query']:.2f} ms/query "
          f"（{n_q} 条共 {cost['search_sec_total']}s）")

    # ── 落盘 ──
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = RESULTS / f"{stamp}_E1_bm25_{args.split}.json"
    payload = {
        "experiment_id": "E1",
        "method": "bm25",
        "split": args.split,
        "date": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "dataset": "dureader-retrieval (C-MTEB test / ranking dev)",
            "corpus_size": split.n_passages,
            "queries": res["filtered"]["queries_evaluated"],
            "bm25_k1": args.k1,
            "bm25_b": args.b,
            "tokenizer": "jieba",
            "top_k": args.top_k,
            "ks": ks,
            "seed": 42,
        },
        "results": {k: round(v["mean"], 6) for k, v in res["aggregate"].items()},
        "cost": cost,
        "filtered": res["filtered"],
        "total_sec": round(time.time() - t_all, 1),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    run_out = RESULTS / f"{stamp}_E1_bm25_{args.split}_run.json"
    run_out.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")

    print(f"\n  结果 → {out.relative_to(ROOT)}")
    print(f"  排序 → {run_out.relative_to(ROOT)}")
    print(f"  总用时 {payload['total_sec']}s")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())