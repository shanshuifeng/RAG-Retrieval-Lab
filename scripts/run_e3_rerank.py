"""E3 · 重排序（Reranking）—— ⭐ 本项目最有价值的实验（docs/09 P1 底线）

⭐ 回答的问题（docs/06 §3.1）：加重排序值不值？
   → 效果：nDCG@10 提升多少？（重排只改顺序不改召回，故主指标是 nDCG）
   → 成本：每 query 多花多少时间？K（候选数）如何影响效果与成本？

⭐ 关键实现决策（效率核心）：
   cross-encoder 对 (query, passage) 对打分，8640 query × 100 候选 = 864,000 对。
   如果对每个 K 档（10/20/50/100）都重新打分，要跑 4 遍，浪费 4 倍时间。

   优化：对 top-100 候选【只打一次分】，存下全部 (qid, pid) → score；
   之后每个 K 档只是取该 query 重排结果的前 K 个，免重算。
   → 4 档评测只需 1 次打分。

⭐ 延迟的诚实计量：
   实际部署中延迟 ∝ K（候选数），故按"总耗时 / 总对数"得每对延迟，
   再按 K 分摊估算各档延迟，并在报告中注明这是估算值。

纪律：
  · 第一段用纯 D2（E2 已证伪混合）
  · 只在 dev 上跑
  · 效果与成本【必须一起报】（docs/06 §3.6）
  · 与第一段（不重排）做配对显著性检验

用法：
  .venv\\Scripts\\python.exe scripts\\run_e3_rerank.py --split dev
  .venv\\Scripts\\python.exe scripts\\run_e3_rerank.py --split dev --max-queries 500  # 快速调试
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

from src.data.loader import load_split  # noqa: E402
from src.eval.metrics import evaluate, ndcg_at_k  # noqa: E402
from src.rerank.cross_encoder import CrossEncoderReranker  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "index"

# ⭐ 重排候选数网格（docs/06 §3.5 关键实验）
K_GRID = [10, 20, 50, 100]
FIRST_STAGE_TOP = 100     # 第一段检索的深度（重排只在这 100 个候选内进行）


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    diff = b - a
    n = len(diff)
    t, p = stats.ttest_rel(b, a)
    sd = diff.std(ddof=1)
    d = diff.mean() / sd if sd > 0 else 0.0
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(10000, n))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return {"n": int(n), "diff": round(float(diff.mean()), 6),
            "rel_pct": round(100 * float(diff.mean()) / float(a.mean()), 2)
            if a.mean() else 0.0,
            "p_value": float(p), "cohens_d": round(float(d), 4),
            "ci95": [round(float(lo), 6), round(float(hi), 6)],
            "better": int((diff > 1e-12).sum()),
            "worse": int((diff < -1e-12).sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description="E3 重排序")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--first-stage", default="d2", choices=["d1", "d2"],
                    help="第一段检索器（E2 已证伪混合，默认用最强的 D2）")
    ap.add_argument("--max-queries", type=int, default=0, help="调试用，只跑前 N 条")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-cache", action="store_true", help="重算重排分数")
    args = ap.parse_args()

    if args.split == "test":
        print("❌ E3 调参禁止在 test 上运行")
        return 2

    print("=" * 80)
    print(f"  E3 · 重排序  |  split={args.split}  第一段={args.first_stage.upper()}")
    print("=" * 80)

    t_all = time.time()
    split = load_split(args.split)
    queries = {q: split.relevant_of(q) for q in split.query_list()}
    print(f"  语料 {split.n_passages:,} | query {len(queries):,}")

    # ── 1. 第一段检索结果（用 E1 已落盘的 D2 run）──
    print("\n[1/5] 第一段检索结果 ...")
    pat = ("*E1_dense_base_dev_run.json" if args.first_stage == "d2"
           else "*E1_dense_small_dev_run.json")
    first_run = json.loads(sorted(RESULTS.glob(pat))[-1].read_text(encoding="utf-8"))
    print(f"  第一段 run：{len(first_run):,} query × top-{FIRST_STAGE_TOP}")

    qids = [q for q in split.query_list(with_relevant_only=True) if q in first_run]
    if args.max_queries:
        qids = qids[:args.max_queries]
    print(f"  本实验 query 数：{len(qids):,}")

    # 第一段基线指标（不重排）
    base_scores = {qid: ndcg_at_k(first_run[qid][:10],
                                  split.relevant_of(qid), 10) for qid in qids}
    base_ndcg = float(np.mean(list(base_scores.values())))
    print(f"  第一段（不重排）nDCG@10 = {base_ndcg:.6f}")

    # ── 2. 重排打分（⭐ 对 top-100 只打一次，缓存复用）──
    print("\n[2/5] Cross-encoder 打分（top-100 候选，全量一次）...")
    # ⭐ 缓存键必须包含 query 范围，否则调试子集会被全量误命中
    #    （2026-10-24 踩过：500 条调试缓存险些污染全量评测）
    qscope = f"{len(qids)}q"
    score_cache = CACHE / f"rerank_scores_{args.first_stage}_{args.split}_{qscope}.json"
    latency_per_pair_ms = None

    if score_cache.exists() and not args.no_cache:
        print(f"  命中缓存 {score_cache.name}")
        cached = json.loads(score_cache.read_text(encoding="utf-8"))
        all_scores = cached["scores"]          # {qid: {pid: score}}
        latency_per_pair_ms = cached.get("latency_per_pair_ms")
        n_pairs = cached.get("n_pairs", 0)
        print(f"  载入 {len(all_scores):,} query × {n_pairs:,} 对分数")
    else:
        reranker = CrossEncoderReranker("BAAI/bge-reranker-base",
                                        batch_size=args.batch_size)
        reranker._load()
        st = reranker.stats()
        print(f"  模型 {st['model']} | {st['device']} | fp16 {st['fp16']} "
              f"| batch {st['batch_size']}")

        # 构造全部 (qid, query, pid, text) 对
        print(f"  构造打分对：{len(qids):,} query × top-{FIRST_STAGE_TOP} ...")
        tasks = []
        for qid in qids:
            q = split.queries[qid]
            for pid in first_run[qid][:FIRST_STAGE_TOP]:
                tasks.append((qid, q, pid, split.corpus.get(pid, "")))
        n_pairs = len(tasks)
        print(f"  共 {n_pairs:,} 对")

        # 全局大批量打分（GPU 利用率最高），按 qid 分组记录耗时
        t_score = time.time()
        all_scores: dict[str, dict[str, float]] = {}
        B = args.batch_size * 8      # 全局大批次
        for s in range(0, n_pairs, B):
            chunk = tasks[s:s + B]
            # 按 query 分组打分（reranker.score 是单 query 多 passage）
            # 把同 query 的连续对合并
            i = 0
            while i < len(chunk):
                qid, q = chunk[i][0], chunk[i][1]
                j = i
                texts = []
                pids = []
                while j < len(chunk) and chunk[j][0] == qid:
                    texts.append(chunk[j][3])
                    pids.append(chunk[j][2])
                    j += 1
                scores = reranker.score(q, texts)
                all_scores.setdefault(qid, {}).update(zip(pids, scores))
                i = j
            if (s // B) % 200 == 0:
                el = time.time() - t_score
                done = min(s + B, n_pairs)
                print(f"      打分 {done:,}/{n_pairs:,}  ({el:.0f}s, "
                      f"{done/max(el,1e-9):.0f} 对/s) ...", flush=True)

        total_score_sec = time.time() - t_score
        latency_per_pair_ms = 1000 * total_score_sec / n_pairs
        print(f"  打分完成：{n_pairs:,} 对，{total_score_sec:.0f}s "
              f"({n_pairs/max(total_score_sec,1e-9):.0f} 对/s，"
              f"每对 {latency_per_pair_ms:.2f} ms）")

        score_cache.parent.mkdir(parents=True, exist_ok=True)
        score_cache.write_text(json.dumps({
            "scores": all_scores,
            "latency_per_pair_ms": latency_per_pair_ms,
            "n_pairs": n_pairs,
            "total_score_sec": round(total_score_sec, 1),
            "model": st,
        }, ensure_ascii=False), encoding="utf-8")
        print(f"  分数已缓存 → {score_cache.name}")

    # ── 3. 对每个 K 档重排并评测 ──
    print(f"\n[3/5] K 档重排与评测（K ∈ {K_GRID}）...")
    print(f"  {'K':>5} {'nDCG@10':>9} {'R@10':>8} {'vs 第一段':>10} "
          f"{'延迟/查询':>10}")
    print("  " + "-" * 50)

    rows = []
    reranked_scores_per_k = {}
    for K in K_GRID:
        # 用已算好的分数重排：取第一段 top-K 候选，按 cross-encoder 分数降序
        reranked = {}
        for qid in qids:
            cand_pids = first_run[qid][:K]
            sc = all_scores.get(qid, {})
            order = sorted(cand_pids,
                           key=lambda p: (-sc.get(p, float("-inf")), p))
            reranked[qid] = order
        reranked_scores_per_k[K] = {qid: ndcg_at_k(reranked[qid],
                                                   split.relevant_of(qid), 10)
                                    for qid in qids}
        res = evaluate({q: split.relevant_of(q) for q in qids}, reranked,
                       ks=[10], metrics=("ndcg", "recall"))
        nd = res["aggregate"]["ndcg@10"]["mean"]
        rc = res["aggregate"]["recall@10"]["mean"]
        gain = nd - base_ndcg
        # 延迟估算：K 个候选 × 每对延迟（线性假设，报告中注明）
        est_latency = K * latency_per_pair_ms if latency_per_pair_ms else None
        lat_str = f"{est_latency:.0f} ms" if est_latency else "—"
        rows.append({"K": K, "ndcg@10": round(nd, 6), "recall@10": round(rc, 6),
                     "gain_vs_first": round(gain, 6),
                     "gain_pct": round(100 * gain / base_ndcg, 2),
                     "est_latency_ms_per_query": round(est_latency, 1)
                     if est_latency else None})
        print(f"  {K:>5} {nd:>9.4f} {rc:>8.4f} {gain:>+10.4f} {lat_str:>10}")

    # ── 4. 显著性检验（最优 K vs 第一段）──
    print("\n[4/5] 显著性检验 ...")
    best_row = max(rows, key=lambda r: r["ndcg@10"])
    best_k = best_row["K"]
    a = np.array([base_scores[q] for q in qids])
    b = np.array([reranked_scores_per_k[best_k][q] for q in qids])
    test = paired_test(a, b)
    sig = "***" if test["p_value"] < 0.001 else ("**" if test["p_value"] < 0.01
                                                 else ("*" if test["p_value"] < 0.05 else "ns"))
    print(f"  K={best_k} vs 第一段：差值 {test['diff']:+.6f} "
          f"（{test['rel_pct']:+.2f}%），p={test['p_value']:.2e}，"
          f"d={test['cohens_d']:.3f}，CI=[{test['ci95'][0]:+.6f}, "
          f"{test['ci95'][1]:+.6f}] {sig}")
    print(f"  变好 {test['better']:,} / 变差 {test['worse']:,} / "
          f"不变 {test['n'] - test['better'] - test['worse']:,}")

    # ── 5. 汇总：效果-成本曲线 ──
    print("\n[5/5] 效果-成本权衡（E3 的核心结论素材）...")
    print(f"  {'K':>5} {'nDCG@10':>9} {'提升':>8} {'估算延迟':>10}")
    print("  " + "-" * 38)
    print(f"  {'0(不重排)':>5} {base_ndcg:>9.4f} {'—':>8} {'~0 ms':>10}")
    for r in rows:
        lat = f"{r['est_latency_ms_per_query']:.0f} ms" \
            if r['est_latency_ms_per_query'] else "—"
        print(f"  {r['K']:>5} {r['ndcg@10']:>9.4f} "
              f"{r['gain_pct']:>+7.2f}% {lat:>10}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = RESULTS / f"{stamp}_E3_rerank_{args.first_stage}_{args.split}.json"
    payload = {
        "experiment_id": "E3",
        "method": f"rerank:bge-reranker-base over {args.first_stage.upper()}",
        "split": args.split,
        "date": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "first_stage": args.first_stage,
            "first_stage_top": FIRST_STAGE_TOP,
            "k_grid": K_GRID,
            "reranker": "BAAI/bge-reranker-base",
            "queries": len(qids),
        },
        "first_stage_baseline_ndcg10": round(base_ndcg, 6),
        "k_sweep": rows,
        "best_k": best_k,
        "significance_vs_first_stage": test,
        "latency_model": {
            "note": "每对延迟 × K 估算；实际部署延迟 ∝ K（线性假设）",
            "per_pair_ms": latency_per_pair_ms,
        },
        "total_sec": round(time.time() - t_all, 1),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 → {out.relative_to(ROOT)}")
    print(f"  总用时 {payload['total_sec']}s")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())