"""审查补充实验③：质量接近时混合是否有效（验证项目的判断准则）

⭐ 起因（2026-10-25 审查 M4）：
   E2 只测了 BM25 × D2（质量差 1.53 倍），结论是"混合负优化"，
   并提出准则："当两方法质量悬殊时，RRF 融合是负优化"。
        ↓
   ⚠️ 但项目【没测质量接近的组合】——而它自己的准则预测
      在这个场景下混合【可能有效】。这是准则无法自我验证的循环。
        ↓
   本实验补测 D1(0.8151) × D2(0.8875)，两方法质量只差 8.9%。

⭐ 三个组合对照（dev）：
   · BM25 × D2    质量差 1.53×  → E2 已测：-12.96%
   · D1 × D2      质量差 1.089× → 本实验：???
   · 若 D1×D2 能提升 → 证实准则（质量接近时融合有效）
     若 D1×D2 也下降 → 准则需修正（质量悬殊不是唯一原因）

用法：
  .venv\\Scripts\\python.exe scripts\\run_e2b_close_quality.py
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
from src.retrieval.hybrid import rrf_fuse  # noqa: E402

RESULTS = ROOT / "results"

K_GRID = [10, 20, 40, 60, 100]
WEIGHT_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


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
    print("=" * 84)
    print("  审查补充实验③：质量接近时混合是否有效（验证项目判断准则）")
    print("=" * 84)

    split = load_split("dev")
    queries = {q: split.relevant_of(q) for q in split.query_list()}

    run_bm25 = json.loads((RESULTS / "2026-10-24_E1_bm25_tuned_dev_run.json")
                          .read_text(encoding="utf-8"))
    run_d1 = json.loads(sorted(RESULTS.glob("*E1_dense_small_dev_run.json"))[-1]
                        .read_text(encoding="utf-8"))
    run_d2 = json.loads(sorted(RESULTS.glob("*E1_dense_base_dev_run.json"))[-1]
                        .read_text(encoding="utf-8"))

    def score(run):
        return {q: ndcg_at_k(run.get(q, []), rel, 10) for q, rel in queries.items()}

    s_bm25, s_d1, s_d2 = score(run_bm25), score(run_d1), score(run_d2)
    print(f"\n单方法 nDCG@10（dev）：")
    print(f"  BM25(调优)  {np.mean(list(s_bm25.values())):.4f}")
    print(f"  D1/small    {np.mean(list(s_d1.values())):.4f}")
    print(f"  D2/base     {np.mean(list(s_d2.values())):.4f}")

    COMBOS = [
        ("BM25(调优)", "D2/base", run_bm25, run_d2, s_bm25, s_d2),
        ("D1/small", "D2/base", run_d1, run_d2, s_d1, s_d2),
    ]

    summary = []
    for name_a, name_b, ra, rb, sa, sb in COMBOS:
        qa = np.mean(list(sa.values()))
        qb = np.mean(list(sb.values()))
        strong_score = max(qa, qb)
        weak_name = name_a if qa < qb else name_b
        print(f"\n{'='*76}")
        print(f"  组合：{name_a} × {name_b}")
        print(f"  质量比：{max(qa,qb)/min(qa,qb):.3f}×  "
              f"（{'悬殊' if max(qa,qb)/min(qa,qb) > 1.3 else '⭐ 接近'}）")
        print(f"{'='*76}")

        # ── K 网格（等权）──
        print(f"  {'K':>4} {'nDCG@10':>9} {'vs 最强单方法':>14} {'Cohen d':>8} {'p 值':>10}")
        print("  " + "-" * 52)
        best = None
        for k in K_GRID:
            fused = rrf_fuse({name_a: ra, name_b: rb}, k=k, top_k=100)
            sf = {q: ndcg_at_k(fused.get(q, []), rel, 10)
                  for q, rel in queries.items()}
            sig = paired(np.array(list((sa if qa >= qb else sb).values())),
                         np.array(list(sf.values())))
            v = float(np.mean(list(sf.values())))
            mark = "***" if sig["p_value"] < 0.001 else (
                "**" if sig["p_value"] < 0.01 else (
                    "*" if sig["p_value"] < 0.05 else "ns"))
            print(f"  {k:>4} {v:>9.4f} {sig['rel_pct']:>+13.2f}% "
                  f"{sig['cohens_d']:>8.3f} {sig['p_value']:>10.2e} {mark}")
            row = {"k": k, "ndcg@10": round(v, 6), **sig}
            if best is None or v > best["ndcg@10"]:
                best = row

        print(f"\n  最优 K（等权）= {best['k']}，nDCG@10 = {best['ndcg@10']:.6f}"
              f"（{best['rel_pct']:+.2f}% vs 最强单方法）")

        # ── 权重网格（在最优 K 下）──
        print(f"\n  权重扫描（K={best['k']}，给弱方法 {weak_name} 加权）:")
        print(f"  {'w_弱':>6} {'nDCG@10':>9} {'vs 最强':>9}")
        print("  " + "-" * 28)
        w_rows = []
        for w in WEIGHT_GRID:
            weights = ({name_a: w, name_b: 1.0} if weak_name == name_a
                       else {name_a: 1.0, name_b: w})
            fused = rrf_fuse({name_a: ra, name_b: rb}, k=best["k"],
                             weights=weights, top_k=100)
            v = float(np.mean([ndcg_at_k(fused.get(q, []), rel, 10)
                               for q, rel in queries.items()]))
            d = v - strong_score
            print(f"  {w:>6.2f} {v:>9.4f} {100*d/strong_score:>+8.2f}%")
            w_rows.append({"w_weak": w, "ndcg@10": round(v, 6)})

        best_w = max(w_rows, key=lambda x: x["ndcg@10"])
        improvement = best_w["ndcg@10"] - strong_score
        print(f"\n  ⭐ 最优权重 w_弱 = {best_w['w_weak']}，"
              f"nDCG@10 = {best_w['ndcg@10']:.6f}"
              f"（{100*improvement/strong_score:+.2f}% vs 最强单方法）")
        if improvement > 0:
            print("     → ✅ 混合【有效】")
        else:
            print("     → ❌ 混合【无效】（最优权重仍在'不混合'附近）")

        summary.append({
            "pair": f"{name_a} × {name_b}",
            "quality_ratio": round(max(qa, qb) / min(qa, qb), 3),
            "strong_single": round(strong_score, 6),
            "best_k_equal_weight": best["k"],
            "best_ndcg_equal_weight": best["ndcg@10"],
            "best_rel_pct_equal": best["rel_pct"],
            "best_weight": best_w["w_weak"],
            "best_ndcg_with_weight": best_w["ndcg@10"],
            "best_rel_pct_weight": round(100 * improvement / strong_score, 2),
            "hybrid_effective": bool(improvement > 0),
        })

    # ── 对照结论 ──
    print("\n" + "=" * 84)
    print("  ⭐⭐ 综合结论：验证项目的判断准则")
    print("=" * 84)
    print(f"  {'组合':<26} {'质量比':>8} {'最优 K':>7} {'最优 w':>7} "
          f"{'相对最强':>9} {'混合有效':>8}")
    print("  " + "-" * 72)
    # E2 结果（已知）
    e2 = json.loads(sorted(RESULTS.glob("*_E2_rrf_dev.json"))[-1]
                    .read_text(encoding="utf-8"))
    print(f"  {'BM25 × D2（E2 已测）':<26} {'1.53':>8} {e2['best']['k']:>7} "
          f"{'1.0':>7} {e2['verdict_vs_best_single']['gain_pct']:>+8.2f}% {'否':>8}")
    for s in summary:
        print(f"  {s['pair']:<26} {s['quality_ratio']:>8.3f} "
              f"{s['best_k_equal_weight']:>7} {s['best_weight']:>7.2f} "
              f"{s['best_rel_pct_weight']:>+8.2f}% "
              f"{'✅ 是' if s['hybrid_effective'] else '❌ 否':>8}")

    print()
    close = next((s for s in summary if s["quality_ratio"] < 1.3), None)
    if close:
        if close["hybrid_effective"]:
            print("  ✅ 判断准则【成立】：质量接近时（1.089×）混合确实有效")
        else:
            print("  ⚠️ 判断准则【不成立】：即使质量接近（1.089×）混合仍无效")
            print("     → 说明「质量悬殊」不是混合失败的唯一/主要原因")
            print("     → 准则需修正：可能真正的决定因素是【两方法的重合度】")
            print("        或【RRF 排名融合本身的特性】，而非质量差")

    out = RESULTS / f"{datetime.now():%Y-%m-%d_%H%M%S}_audit_close_quality_hybrid.json"
    out.write_text(json.dumps({
        "note": "审查补充实验③：质量接近时混合是否有效（验证项目判断准则）",
        "summary": summary,
        "e2_reference": {"pair": "BM25 × D2", "quality_ratio": 1.53,
                         "gain_pct": e2["verdict_vs_best_single"]["gain_pct"]},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 → {out.relative_to(ROOT)}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())