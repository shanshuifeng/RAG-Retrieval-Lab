"""E1 核心对比与显著性检验 —— 回答 Q1

⭐ Q1：稠密向量检索比 BM25 好多少？

本脚本做三件事：
  1. 汇总 B0/B1/D1/D2 四个方法的指标与成本（一张对比表）
  2. ⭐ 逐 query 配对检验：BM25 vs D1、D1 vs D2、BM25 vs D2
  3. ⭐ 互补性分析：两方法的检索结果重合度（为 E2 混合检索提供依据）

用法：
  .venv\\Scripts\\python.exe scripts/compare_e1.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_split  # noqa: E402
from src.eval.metrics import ndcg_at_k, recall_at_k  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "index"


def load_latest(pattern: str) -> dict | None:
    files = sorted(RESULTS.glob(pattern))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def load_run_latest(pattern: str) -> dict | None:
    files = sorted(RESULTS.glob(pattern))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def paired(a: dict[str, float], b: dict[str, float], name: str) -> dict:
    common = sorted(set(a) & set(b))
    x = np.array([a[q] for q in common])
    y = np.array([b[q] for q in common])
    diff = y - x
    t, p = stats.ttest_rel(y, x)
    sd = diff.std(ddof=1)
    d = diff.mean() / sd if sd > 0 else 0.0
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(diff), size=(10000, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return {
        "comparison": name,
        "n": len(common),
        "a_mean": round(float(x.mean()), 6),
        "b_mean": round(float(y.mean()), 6),
        "diff": round(float(diff.mean()), 6),
        "rel_pct": round(100 * float(diff.mean()) / float(x.mean()), 2)
        if x.mean() else 0.0,
        "p_value": float(p),
        "cohens_d": round(float(d), 4),
        "ci95": [round(float(lo), 6), round(float(hi), 6)],
        "better": int((diff > 1e-12).sum()),
        "worse": int((diff < -1e-12).sum()),
        "same": int(len(diff) - (diff > 1e-12).sum() - (diff < -1e-12).sum()),
    }


def effect_label(d: float) -> str:
    a = abs(d)
    if a < 0.2:
        return "可忽略"
    if a < 0.5:
        return "小"
    if a < 0.8:
        return "中"
    return "大"


def main() -> int:
    print("=" * 80)
    print("  E1 · 基线对比总结与显著性检验（回答 Q1）")
    print("=" * 80)

    split = load_split("dev")
    queries = {q: split.relevant_of(q) for q in split.query_list()}

    # ── 1. 汇总指标 ──
    print("\n[1/4] 汇总各方法结果 ...")
    runs = {
        "BM25(默认)": load_run_latest("*E1_bm25_dev_run.json"),
        "BM25(调优)": load_run_latest("*E1_bm25fast_dev*k11.5_b0.75_top100.json"),
        "D1/BGE-small": load_run_latest("*E1_dense_small_dev_run.json"),
        "D2/BGE-base": load_run_latest("*E1_dense_base_dev_run.json"),
    }
    # BM25 调优结果需要单独重跑 k1=0.7（缓存里是 1.5）
    from src.retrieval.bm25_fast import BM25Fast
    tuned = BM25Fast.load(CACHE / "bm25fast_base_dev.pkl")
    tuned.set_params(k1=0.7, b=0.75)
    runs["BM25(调优)"] = tuned.run_all(split, top_k=100, verbose=False)

    runs = {k: v for k, v in runs.items() if v}
    print(f"  可用方法：{list(runs)}")

    per_q_ndcg: dict[str, dict[str, float]] = {}
    per_q_recall: dict[str, dict[str, float]] = {}
    table = {}
    for name, run in runs.items():
        nd, rc = {}, {}
        for qid, rel in queries.items():
            if not rel or qid not in run:
                continue
            nd[qid] = ndcg_at_k(run[qid], rel, 10)
            rc[qid] = recall_at_k(run[qid], rel, 10)
        per_q_ndcg[name] = nd
        per_q_recall[name] = rc
        table[name] = {
            "recall@10": round(float(np.mean(list(rc.values()))), 6),
            "ndcg@10": round(float(np.mean(list(nd.values()))), 6),
        }

    print(f"\n  {'方法':<14} {'Recall@10':>11} {'nDCG@10':>10}")
    print("  " + "-" * 38)
    for name in runs:
        print(f"  {name:<14} {table[name]['recall@10']:>11.4f} "
              f"{table[name]['ndcg@10']:>10.4f}")

    # ── 2. 配对检验 ──
    print("\n[2/4] 逐 query 配对显著性检验（nDCG@10）...")
    pairs = [
        ("BM25(调优)", "D1/BGE-small"),
        ("D1/BGE-small", "D2/BGE-base"),
        ("BM25(调优)", "D2/BGE-base"),
    ]
    tests = []
    for a, b in pairs:
        if a in per_q_ndcg and b in per_q_ndcg:
            tests.append(paired(per_q_ndcg[a], per_q_ndcg[b], f"{b} vs {a}"))

    print(f"\n  {'对比':<30} {'差值':>9} {'相对':>8} {'p 值':>11} "
          f"{'d':>7} {'效应':>6}")
    print("  " + "-" * 78)
    for t_ in tests:
        sig = "***" if t_["p_value"] < 0.001 else ("**" if t_["p_value"] < 0.01
                                                   else ("*" if t_["p_value"] < 0.05 else "ns"))
        print(f"  {t_['comparison']:<30} {t_['diff']:>+9.4f} "
              f"{t_['rel_pct']:>+7.1f}% {t_['p_value']:>11.2e} "
              f"{t_['cohens_d']:>7.3f} {effect_label(t_['cohens_d']):>6} {sig}")

    # ── 3. 互补性分析（为 E2 提供依据）──
    print("\n[3/4] 互补性分析（Top-10 重合度）...")
    overlap = {}
    if "BM25(调优)" in runs and "D1/BGE-small" in runs:
        for a, b in [("BM25(调优)", "D1/BGE-small"),
                     ("D1/BGE-small", "D2/BGE-base")]:
            if a not in runs or b not in runs:
                continue
            ious = []
            union_hit_gain = 0
            for qid, rel in queries.items():
                if not rel:
                    continue
                ra = set(runs[a].get(qid, [])[:10])
                rb = set(runs[b].get(qid, [])[:10])
                if not ra and not rb:
                    continue
                ious.append(len(ra & rb) / len(ra | rb))
                # 并集能否带来额外召回
                relset = set(rel)
                hit_a = len(ra & relset)
                hit_union = len((ra | rb) & relset)
                if hit_union > hit_a:
                    union_hit_gain += 1
            overlap[f"{a} vs {b}"] = {
                "mean_iou_top10": round(float(np.mean(ious)), 4),
                "queries_where_union_beats_a": union_hit_gain,
            }
            print(f"  {a} vs {b}:")
            print(f"    Top-10 平均 IoU        : {np.mean(ious):.4f}")
            print(f"    并集优于 {a} 的 query 数: {union_hit_gain:,} / {len(ious):,}")

    # ── 4. 成本对比 ──
    print("\n[4/4] 成本对比 ...")
    bm25_meta = load_latest("*E1_bm25_dev.json")
    d1_meta = load_latest("*E1_dense_small_dev.json")
    d2_meta = load_latest("*E1_dense_base_dev.json")
    print(f"\n  {'方法':<14} {'建索引':>12} {'检索(ms/q)':>12} {'设备':>8}")
    print("  " + "-" * 50)
    if bm25_meta:
        c = bm25_meta["cost"]
        print(f"  {'BM25':<14} {c['build_sec']:>11.1f}s "
              f"{c['search_ms_per_query']:>11.2f} {'CPU':>8}")
    for tag, meta in [("D1/BGE-small", d1_meta), ("D2/BGE-base", d2_meta)]:
        if meta:
            c = meta["cost"]
            print(f"  {tag:<14} {c['build_sec_total']:>11.1f}s "
                  f"{c['search_ms_per_query']:>11.2f} {c['device']:>8}")

    # ── 输出 ──
    out = RESULTS / "2026-10-24_E1_comparison_dev.json"
    out.write_text(json.dumps({
        "note": "E1 四方法对比 + 配对显著性检验 + 互补性分析",
        "split": "dev",
        "n_queries": len(queries),
        "metrics": table,
        "significance_tests": tests,
        "complementarity": overlap,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  结果 → {out.relative_to(ROOT)}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())