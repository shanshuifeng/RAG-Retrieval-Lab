"""E4 · 分块策略消融 —— P9-1（docs/06 §四）

⭐ 回答的问题（Q4）：文档怎么切最好？
   主指标 Recall@10（docs/06 §四 约定），辅助 nDCG@10。

⭐ 评测口径（关键，错则全错）：
   分块后检索单位是「块」，但 qrels 标注的是「段落」。
   判定规则：块级排名 → 投影回段落级（任一 chunk 命中 = 父段落召回，
   同一父段落的多个块只占一个位置）→ 再按段落级 qrels 算指标。
   该投影逻辑有独立单测保障（tests/test_chunking.py）。

实验组（dev，test 已锁定）：
   A. native（不切分）—— 基线，直接复用已有结果
   B. fixed256（256 字 + 重叠 50）
   C. fixed512（512 字 + 重叠 50）
   D. paragraph（\\n\\n 切，超长兜底）
   每个策略 × 两种检索器（BM25 调优 / D2）

纪律：
  · 只在 dev 上跑（test 已锁）
  · 与 native 基线做配对显著性检验
  · 记录分块后语料规模变化 + 成本

用法：
  .venv\\Scripts\\python.exe scripts\\run_e4_chunking.py --split dev
  .venv\\Scripts\\python.exe scripts\\run_e4_chunking.py --split dev --retriever bm25  # 只跑 BM25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.chunking import build_chunked_corpus, chunked_split_view  # noqa: E402
from src.data.loader import load_split  # noqa: E402
from src.eval.metrics import ndcg_at_k, recall_at_k  # noqa: E402
from src.retrieval.bm25_fast import BM25Fast  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "index"

STRATEGIES = ["native", "fixed256", "fixed512", "paragraph"]


def project_run(run: dict, cc) -> dict:
    """把块级 run 投影回段落级 run。"""
    return {qid: cc.project_ranking(ids) for qid, ids in run.items()}


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    diff = b - a
    n = len(diff)
    t, p = stats.ttest_rel(b, a)
    sd = diff.std(ddof=1)
    d = diff.mean() / sd if sd > 0 else 0.0
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(10000, n))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return {"diff": round(float(diff.mean()), 6),
            "rel_pct": round(100 * float(diff.mean()) / float(a.mean()), 2)
            if a.mean() else 0.0,
            "p_value": float(p), "cohens_d": round(float(d), 4),
            "ci95": [round(float(lo), 6), round(float(hi), 6)]}


def main() -> int:
    ap = argparse.ArgumentParser(description="E4 分块消融")
    ap.add_argument("--split", default="dev", choices=["dev"])
    ap.add_argument("--retriever", default="both", choices=["bm25", "dense", "both"])
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    print("=" * 80)
    print(f"  E4 · 分块策略消融  |  split={args.split}（test 已锁，本实验只在 dev）")
    print("=" * 80)

    t_all = time.time()
    split = load_split(args.split)
    queries = {q: split.relevant_of(q) for q in split.query_list()}
    qids = list(queries)
    print(f"  语料 {split.n_passages:,} 段 | query {len(queries):,}")

    # ── 基线（native）：BM25 调优 / D2 的已有 run ──
    print("\n[基线] native（不切分，复用已有结果）...")
    baseline_runs = {}
    if args.retriever in ("bm25", "both"):
        baseline_runs["bm25"] = json.loads(
            (RESULTS / "2026-10-24_E1_bm25_tuned_dev_run.json")
            .read_text(encoding="utf-8"))
    if args.retriever in ("dense", "both"):
        baseline_runs["dense"] = json.loads(
            sorted(RESULTS.glob("*E1_dense_base_dev_run.json"))[-1]
            .read_text(encoding="utf-8"))

    def eval_projected(run: dict, split) -> dict:
        rec = [recall_at_k(run.get(q, []), queries[q], 10) for q in qids]
        nd = [ndcg_at_k(run.get(q, []), queries[q], 10) for q in qids]
        return {"recall@10": float(np.mean(rec)), "ndcg@10": float(np.mean(nd)),
                "per_q_recall": rec, "per_q_ndcg": nd}

    baseline_scores = {}
    print("  native 基线：")
    for name, run in baseline_runs.items():
        m = eval_projected(run, split)
        baseline_scores[name] = m
        print(f"    {name}: Recall@10={m['recall@10']:.4f}  nDCG@10={m['ndcg@10']:.4f}")

    # ── 逐策略分块 + 检索 ──
    results = []
    dense_model = None
    for strategy in STRATEGIES:
        if strategy == "native":
            continue
        print(f"\n{'='*60}\n  策略 {strategy}\n{'='*60}")

        t0 = time.time()
        cc = build_chunked_corpus(split, strategy, verbose=True)
        view = chunked_split_view(split, cc)
        chunk_stats = cc.stats()
        print(f"  分块完成：{chunk_stats['n_chunks']:,} 块"
              f"（父段 {chunk_stats['n_parents']:,}，"
              f"均值 {chunk_stats['chunks_per_parent_mean']} 块/段）"
              f"，用时 {time.time()-t0:.0f}s")

        row = {"strategy": strategy, "corpus": chunk_stats}

        # ── BM25 ──
        if args.retriever in ("bm25", "both"):
            print("  [BM25 调优] ...")
            cache = CACHE / f"bm25fast_chunked_{strategy}_dev.pkl"
            bm = BM25Fast()
            t0 = time.time()
            if cache.exists() and not args.no_cache:
                bm = BM25Fast.load(cache)
                print("    命中缓存")
            else:
                bm.build_base(view, verbose=True)
                bm.save(cache)
            bm.set_params(k1=0.7, b=0.75)
            build_sec = time.time() - t0
            t0 = time.time()
            run_chunks = bm.run_all(view, top_k=args.top_k, verbose=False)
            search_sec = time.time() - t0
            run_proj = project_run(run_chunks, cc)
            m = eval_projected(run_proj, split)
            row["bm25"] = {"recall@10": round(m["recall@10"], 6),
                           "ndcg@10": round(m["ndcg@10"], 6),
                           "build_sec": round(build_sec, 1),
                           "search_sec": round(search_sec, 1)}
            row["bm25_sig"] = paired_test(
                np.array(baseline_scores["bm25"]["per_q_ndcg"]),
                np.array(m["per_q_ndcg"]))
            print(f"    Recall@10={m['recall@10']:.4f}  nDCG@10={m['ndcg@10']:.4f}"
                  f"（基线 {baseline_scores['bm25']['ndcg@10']:.4f}）")

        # ── 稠密 D2 ──
        if args.retriever in ("dense", "both"):
            print("  [D2 稠密] ...")
            cache = CACHE / f"dense_base_chunked_{strategy}_dev_noinstr.pkl"
            t0 = time.time()
            if cache.exists() and not args.no_cache:
                dr = DenseRetriever.load(cache)
                print("    命中缓存")
            else:
                dr = DenseRetriever("BAAI/bge-base-zh-v1.5")
                dr.build(view, verbose=True)
                dr.save(cache)
            build_sec = time.time() - t0
            t0 = time.time()
            run_chunks = dr.run_all(view, top_k=args.top_k, verbose=False)
            search_sec = time.time() - t0
            run_proj = project_run(run_chunks, cc)
            m = eval_projected(run_proj, split)
            row["dense"] = {"recall@10": round(m["recall@10"], 6),
                            "ndcg@10": round(m["ndcg@10"], 6),
                            "build_sec": round(build_sec, 1),
                            "search_sec": round(search_sec, 1)}
            row["dense_sig"] = paired_test(
                np.array(baseline_scores["dense"]["per_q_ndcg"]),
                np.array(m["per_q_ndcg"]))
            print(f"    Recall@10={m['recall@10']:.4f}  nDCG@10={m['ndcg@10']:.4f}"
                  f"（基线 {baseline_scores['dense']['ndcg@10']:.4f}）")
            del dr   # 释放显存

        results.append(row)

    # ── 汇总 ──
    print("\n" + "=" * 80)
    print("  E4 汇总（dev，段落级投影后评测）")
    print("=" * 80)
    for name in baseline_runs:
        print(f"\n  检索器：{name}")
        print(f"  {'策略':<12} {'Recall@10':>10} {'nDCG@10':>10} {'vs native':>11} "
              f"{'p 值':>10} {'块数':>8}")
        print("  " + "-" * 66)
        base = baseline_scores[name]
        print(f"  {'native':<12} {base['recall@10']:>10.4f} {base['ndcg@10']:>10.4f} "
              f"{'—':>11} {'—':>10} {split.n_passages:>8,}")
        key = "bm25" if name == "bm25" else "dense"
        for r in results:
            if key not in r:
                continue
            sig = r[f"{key}_sig"]
            m = r[key]
            mark = "***" if sig["p_value"] < 0.001 else ("**" if sig["p_value"] < 0.01
                       else ("*" if sig["p_value"] < 0.05 else "ns"))
            print(f"  {r['strategy']:<12} {m['recall@10']:>10.4f} "
                  f"{m['ndcg@10']:>10.4f} {sig['rel_pct']:>+10.2f}% "
                  f"{sig['p_value']:>10.2e} {r['corpus']['n_chunks']:>8,} {mark}")

    out = RESULTS / f"{datetime.now():%Y-%m-%d_%H%M%S}_E4_chunking_dev.json"
    out.write_text(json.dumps({
        "experiment_id": "E4", "split": args.split,
        "date": datetime.now().isoformat(timespec="seconds"),
        "note": "⭐ test 已锁定，本实验只在 dev；评测为块级检索后投影回段落级",
        "baseline_native": {n: {"recall@10": round(b['recall@10'], 6),
                                "ndcg@10": round(b['ndcg@10'], 6)}
                            for n, b in baseline_scores.items()},
        "strategies": [{k: v for k, v in r.items()} for r in results],
        "total_sec": round(time.time() - t_all, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 → {out.relative_to(ROOT)}")
    print(f"  总用时 {round(time.time()-t_all, 1)}s")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())