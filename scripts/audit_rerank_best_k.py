"""审查补充实验 ②：弱/中/强第一段下的【最优 K】（补全 S1 的实践建议）

⭐ 起因（2026-10-25 审查 P0 待办）：
   审查实验①证实了「第一段越弱，重排提升越大」（K=20 固定）。
   但 D2 下 K=20 是甜点、更大的 K 更差——这个结论是在【强第一段】得到的。
   弱第一段（BM25）下候选质量差，可能需要【更大的 K】让重排挑到好的。
        ↓
   ⭐ 若不测，实践建议就是残缺的："第一段弱就用重排" 但不知道该用多大 K。

⭐ 高效做法（零额外打分成本）：
   审查实验①已对三个第一段的 top-100 全部打分并缓存，
   本脚本直接复用 → 各 K 档只是"取重排结果的前 K 个"。
   6 个 K 档 × 3 个第一段 = 18 组评测，全部在秒级完成。

用法：
  .venv\\Scripts\\python.exe scripts\\audit_rerank_best_k.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_split  # noqa: E402
from src.eval.metrics import ndcg_at_k  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "index"

# ⭐ K 网格：覆盖很小的 K 到接近第一段深度
K_GRID = [5, 10, 20, 30, 50, 100]
N_QUERIES = 1500     # 与审查实验①一致，保证可比

FIRST_STAGES = {
    "BM25(调优)": ("audit_rerank_BM25调优_1500q.json",
                   "2026-10-24_E1_bm25_tuned_dev_run.json"),
    "D1/small": ("audit_rerank_D1_small_1500q.json",
                 None),      # 用 glob 找最新的 dense small run
    "D2/base": ("audit_rerank_D2_base_1500q.json", None),
}


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
    print("=" * 88)
    print("  审查补充实验②：弱/中/强第一段下的最优 K（复用①的打分缓存，零额外打分）")
    print("=" * 88)

    split = load_split("dev")
    queries = {q: split.relevant_of(q) for q in split.query_list()}

    # 载入三个第一段的 run
    run_bm25 = json.loads((RESULTS / "2026-10-24_E1_bm25_tuned_dev_run.json")
                          .read_text(encoding="utf-8"))
    run_d1 = json.loads(sorted(RESULTS.glob("*E1_dense_small_dev_run.json"))[-1]
                        .read_text(encoding="utf-8"))
    run_d2 = json.loads(sorted(RESULTS.glob("*E1_dense_base_dev_run.json"))[-1]
                        .read_text(encoding="utf-8"))
    runs = {"BM25(调优)": run_bm25, "D1/small": run_d1, "D2/base": run_d2}

    qids = [q for q in split.query_list() if q in run_d2][:N_QUERIES]
    print(f"  query 数：{len(qids):,}（与审查实验①相同，保证可比）")
    print(f"  K 网格：{K_GRID}")

    summary = []
    for stage_name, (cache_name, _) in FIRST_STAGES.items():
        cache_path = CACHE / cache_name
        if not cache_path.exists():
            print(f"\n⚠️ 缺少缓存 {cache_name}，跳过 {stage_name}")
            continue
        scores = json.loads(cache_path.read_text(encoding="utf-8"))
        run = runs[stage_name]

        base = np.array([ndcg_at_k(run.get(q, []), queries[q], 10) for q in qids])
        base_mean = float(base.mean())

        print(f"\n{'='*70}")
        print(f"  第一段：{stage_name}  基线 nDCG@10 = {base_mean:.6f}"
              f"（剩余空间 {(1-base_mean)*100:.1f}%）")
        print(f"{'='*70}")
        print(f"  {'K':>5} {'nDCG@10':>9} {'提升':>9} {'相对':>8} "
              f"{'占剩余空间':>10} {'Cohen d':>8} {'p 值':>10}")
        print("  " + "-" * 66)

        rows = []
        best = None
        for K in K_GRID:
            reranked = {}
            for qid in qids:
                cands = run[qid][:K]
                sc = scores.get(qid, {})
                reranked[qid] = sorted(
                    cands, key=lambda p: (-sc.get(p, float("-inf")), p))
            after = np.array([ndcg_at_k(reranked[q], queries[q], 10)
                              for q in qids])
            sig = paired(base, after)
            used = sig["diff"] / (1 - base_mean) * 100
            mark = "***" if sig["p_value"] < 0.001 else (
                "**" if sig["p_value"] < 0.01 else (
                    "*" if sig["p_value"] < 0.05 else "ns"))
            print(f"  {K:>5} {after.mean():>9.4f} {sig['diff']:>+9.4f} "
                  f"{sig['rel_pct']:>+7.2f}% {used:>9.1f}% "
                  f"{sig['cohens_d']:>8.3f} {sig['p_value']:>10.2e} {mark}")
            row = {"K": K, "ndcg@10": round(float(after.mean()), 6), **sig,
                   "pct_of_headroom": round(used, 2)}
            rows.append(row)
            if best is None or row["ndcg@10"] > best["ndcg@10"]:
                best = row

        print(f"\n  ⭐ 该第一段的最优 K = {best['K']}"
              f"（nDCG@10 = {best['ndcg@10']:.6f}，{best['rel_pct']:+.2f}%）")
        summary.append({"first_stage": stage_name,
                        "baseline_ndcg10": round(base_mean, 6),
                        "headroom_pct": round((1 - base_mean) * 100, 2),
                        "best_k": best["K"], "best_ndcg10": best["ndcg@10"],
                        "best_rel_pct": best["rel_pct"],
                        "best_cohens_d": best["cohens_d"],
                        "k_sweep": rows})

    # ── 跨第一段对比 ──
    print("\n" + "=" * 88)
    print("  ⭐⭐ 综合结论：最优 K 是否随第一段强度变化？")
    print("=" * 88)
    print(f"  {'第一段':<14} {'基线':>9} {'最优 K':>7} {'重排后':>9} "
          f"{'提升':>9} {'Cohen d':>8} {'剩余空间':>9}")
    print("  " + "-" * 72)
    for s in summary:
        print(f"  {s['first_stage']:<14} {s['baseline_ndcg10']:>9.4f} "
              f"{s['best_k']:>7} {s['best_ndcg10']:>9.4f} "
              f"{s['best_rel_pct']:>+8.2f}% {s['best_cohens_d']:>8.3f} "
              f"{s['headroom_pct']:>8.1f}%")

    if len(summary) >= 2:
        ks = [s["best_k"] for s in summary]
        print()
        if len(set(ks)) == 1:
            print(f"  → ⭐ 最优 K 恒为 {ks[0]}，不随第一段强度变化")
            print("     实践建议可简化为：任何第一段都用同一个 K")
        else:
            weakest = summary[0]
            strongest = summary[-1]
            print(f"  → ⭐⭐ 最优 K 随第一段变化：")
            print(f"     弱段（{weakest['first_stage']}）最优 K = {weakest['best_k']}")
            print(f"     强段（{strongest['first_stage']}）最优 K = {strongest['best_k']}")
            if weakest["best_k"] > strongest["best_k"]:
                print("     ✅ 证实推断：第一段越弱，需要越大的 K")
                print("        （候选质量差 → 需要更多候选让重排挑出好的）")
            else:
                print("     ⚠️ 与推断相反：第一段越弱反而用更小的 K 更好")

    out = RESULTS / f"{datetime.now():%Y-%m-%d_%H%M%S}_audit_best_k_by_firststage.json"
    out.write_text(json.dumps({
        "note": "审查补充实验②：弱/中/强第一段下的最优 K（dev，1500 query）",
        "k_grid": K_GRID, "n_queries": len(qids),
        "summary": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 → {out.relative_to(ROOT)}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())