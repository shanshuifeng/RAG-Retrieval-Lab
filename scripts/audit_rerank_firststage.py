"""审查补充实验：重排在【弱第一段】下是否有效（验证天花板效应 S1）

⭐ 起因（2026-10-25 项目审查）：
   E3 只在 D2（test nDCG@10=0.8809）之后接重排，结论是"test 上不显著"。
   但 0.8809 意味着理想上限只剩 11.9% —— 排序改进几乎【没有空间】。
        ↓
   这可能是"天花板效应"，而不是"重排无效"。

⭐ 本脚本做的事（dev 上，test 已锁）：
   把同一个 cross-encoder 接在三个不同强度的第一段之后，比较提升幅度：
     · BM25(调优)  nDCG@10 = 0.5305  →  弱第一段
     · D1/small    nDCG@10 = 0.7332  →  中第一段
     · D2/base     nDCG@10 = 0.8139  →  强第一段（E3 已做）

   若弱第一段的提升 >> 强第一段 → 证实天花板效应
   若各段提升相近 → 说明重排确实收益有限，与第一段强度无关

⭐ 复用已有分数缓存（D2 的 8640q 已打过分），其余两段需新打分。
   为控制时间，默认只跑 1500 条 query（足够看趋势，且显著性可判）。

用法：
  .venv\\Scripts\\python.exe scripts\\audit_rerank_firststage.py --n 1500
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
from src.eval.metrics import ndcg_at_k  # noqa: E402
from src.rerank.cross_encoder import CrossEncoderReranker  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "index"
K = 20            # E3 已锁定的甜点
FIRST_TOP = 100   # 第一段深度


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    diff = b - a
    t, p = stats.ttest_rel(b, a)
    sd = diff.std(ddof=1)
    d = diff.mean() / sd if sd > 0 else 0.0
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(diff), size=(10000, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return {"diff": round(float(diff.mean()), 6),
            "rel_pct": round(100 * float(diff.mean()) / float(a.mean()), 2),
            "p_value": float(p), "cohens_d": round(float(d), 4),
            "ci95": [round(float(lo), 6), round(float(hi), 6)]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500, help="query 数（控制耗时）")
    args = ap.parse_args()

    print("=" * 80)
    print("  审查补充实验：重排在弱/中/强第一段下的收益对比")
    print("=" * 80)

    split = load_split("dev")
    queries = {q: split.relevant_of(q) for q in split.query_list()}

    # 三个第一段的 run
    first_runs = {
        "BM25(调优)": json.loads(
            (RESULTS / "2026-10-24_E1_bm25_tuned_dev_run.json")
            .read_text(encoding="utf-8")),
        "D1/small": json.loads(
            sorted(RESULTS.glob("*E1_dense_small_dev_run.json"))[-1]
            .read_text(encoding="utf-8")),
        "D2/base": json.loads(
            sorted(RESULTS.glob("*E1_dense_base_dev_run.json"))[-1]
            .read_text(encoding="utf-8")),
    }

    qids = [q for q in split.query_list() if q in first_runs["D2/base"]][:args.n]
    print(f"  query 数：{len(qids):,}（控制耗时，看趋势即可）")

    reranker = CrossEncoderReranker("BAAI/bge-reranker-base")
    reranker._load()
    print(f"  模型已加载：{reranker.stats()['model']}")

    rows = []
    for name, run in first_runs.items():
        print(f"\n{'='*60}\n  第一段：{name}\n{'='*60}")

        # 基线
        base = np.array([ndcg_at_k(run.get(q, []), queries[q], 10) for q in qids])
        print(f"  不重排 nDCG@10 = {base.mean():.6f}")

        # 打分缓存（按第一段区分）
        key = name.replace("/", "_").replace("(", "").replace(")", "")
        sc_cache = CACHE / f"audit_rerank_{key}_{len(qids)}q.json"
        if sc_cache.exists():
            all_scores = json.loads(sc_cache.read_text(encoding="utf-8"))
            print(f"  命中缓存 {sc_cache.name}")
        else:
            t0 = time.time()
            all_scores = {}
            n_pairs = 0
            for i, qid in enumerate(qids, 1):
                cands = run[qid][:FIRST_TOP]
                texts = [split.corpus.get(p, "") for p in cands]
                sc = reranker.score(split.queries[qid], texts)
                all_scores[qid] = dict(zip(cands, sc))
                n_pairs += len(cands)
                if i % 300 == 0:
                    el = time.time() - t0
                    print(f"    打分 {i:,}/{len(qids):,} ({el:.0f}s) ...", flush=True)
            print(f"  打分完成 {n_pairs:,} 对，{time.time()-t0:.0f}s")
            sc_cache.write_text(json.dumps(all_scores, ensure_ascii=False),
                                encoding="utf-8")

        # 重排（K=20）
        reranked = {}
        for qid in qids:
            cands = run[qid][:K]
            sc = all_scores.get(qid, {})
            reranked[qid] = sorted(cands, key=lambda p: (-sc.get(p, float("-inf")), p))

        after = np.array([ndcg_at_k(reranked[q], queries[q], 10) for q in qids])
        sig = paired(base, after)
        mark = "***" if sig["p_value"] < 0.001 else ("**" if sig["p_value"] < 0.01
                   else ("*" if sig["p_value"] < 0.05 else "ns"))

        print(f"  重排后 nDCG@10 = {after.mean():.6f}")
        print(f"  提升：{sig['diff']:+.6f}（{sig['rel_pct']:+.2f}%），"
              f"p={sig['p_value']:.2e}，d={sig['cohens_d']:.3f} {mark}")
        print(f"  剩余空间：{(1.0 - base.mean())*100:.1f}% "
              f"→ 提升占剩余空间的 {sig['diff']/(1.0-base.mean())*100:.1f}%")

        rows.append({
            "first_stage": name,
            "baseline_ndcg10": round(float(base.mean()), 6),
            "reranked_ndcg10": round(float(after.mean()), 6),
            "gain": sig["diff"],
            "gain_pct": sig["rel_pct"],
            "p_value": sig["p_value"],
            "cohens_d": sig["cohens_d"],
            "headroom_pct": round((1.0 - base.mean()) * 100, 2),
            "gain_as_pct_of_headroom": round(
                sig["diff"] / (1.0 - base.mean()) * 100, 2),
        })

    # ── 对比总结 ──
    print("\n" + "=" * 80)
    print("  ⭐ 三档第一段的对比（验证天花板效应）")
    print("=" * 80)
    print(f"  {'第一段':<14} {'基线':>9} {'重排后':>9} {'提升':>9} {'相对':>8} "
          f"{'剩余空间':>9} {'占空间':>8} {'p 值':>10}")
    print("  " + "-" * 78)
    for r in rows:
        print(f"  {r['first_stage']:<14} {r['baseline_ndcg10']:>9.4f} "
              f"{r['reranked_ndcg10']:>9.4f} {r['gain']:>+9.4f} "
              f"{r['gain_pct']:>+7.2f}% {r['headroom_pct']:>8.1f}% "
              f"{r['gain_as_pct_of_headroom']:>7.1f}% {r['p_value']:>10.2e}")

    weak = next(r for r in rows if "BM25" in r["first_stage"])
    strong = next(r for r in rows if "D2" in r["first_stage"])
    print()
    print(f"  ⭐ 弱段（BM25）提升 {weak['gain_pct']:+.2f}%  vs  "
          f"强段（D2）提升 {strong['gain_pct']:+.2f}%")
    if weak["gain_pct"] > strong["gain_pct"] * 2:
        print("  ✅ 证实地板效应：弱第一段下重排收益显著更大")
        print("     → 项目原结论「重排未复现」应表述为「强第一段下无空间」，")
        print("       而非「重排无效」")
    elif abs(weak["gain_pct"] - strong["gain_pct"]) < 1:
        print("  → 各段提升相近，说明重排收益与第一段强度关系不大")
        print("    项目原结论基本成立")
    else:
        print("  → 趋势不明显，需更多样本")

    out = RESULTS / f"{datetime.now():%Y-%m-%d_%H%M%S}_audit_rerank_firststage.json"
    out.write_text(json.dumps({
        "note": "审查补充实验：重排在弱/中/强第一段下的收益（dev，K=20）",
        "n_queries": len(qids), "K": K,
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 → {out.relative_to(ROOT)}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())