"""BM25 调参的显著性检验 —— docs/05 §六要求

⭐ 问题：nDCG@10 从 0.5199（默认）升到 0.5305（k1=0.7,b=0.75），+2.05%。
   这是真提升还是随机波动？

做法：配对 t 检验（每个 query 一组配对分数）+ 效应量 + bootstrap 置信区间。
      三者一起看，避免只看 p 值。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_split  # noqa: E402
from src.eval.metrics import ndcg_at_k, recall_at_k  # noqa: E402
from src.retrieval.bm25_fast import BM25Fast  # noqa: E402


def per_query_scores(retriever, split, queries, ks=(10,), metrics=("ndcg",)) -> dict:
    """跑一遍检索，返回每条 query 的指标（配对检验需要逐 query 分数）。"""
    out: dict[str, dict[str, float]] = {}
    fns = {"ndcg": ndcg_at_k, "recall": recall_at_k}
    for qid, rel in queries.items():
        if not rel:
            continue
        ranked = [p for p, _ in retriever.search(split.queries[qid], top_k=100)]
        d = {}
        for k in ks:
            for m in metrics:
                d[f"{m}@{k}"] = fns[m](ranked, rel, k)
        out[qid] = d
    return out


def paired_test(a: np.ndarray, b: np.ndarray, name: str) -> dict:
    """a=基线，b=新方法。返回 t 检验 + 效应量 + bootstrap CI。"""
    diff = b - a
    n = len(diff)
    t_stat, p = stats.ttest_rel(b, a)

    # Cohen's d（配对版：用差值的标准差）
    sd = diff.std(ddof=1)
    d_eff = diff.mean() / sd if sd > 0 else 0.0

    # bootstrap 95% CI（10000 次重采样）
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(10000, n))
    means = diff[idx].mean(axis=1)
    ci_lo, ci_hi = np.percentile(means, [2.5, 97.5])

    # 有多少 query 变好/变差/不变
    better = int((diff > 1e-12).sum())
    worse = int((diff < -1e-12).sum())
    same = n - better - worse

    return {
        "metric": name,
        "n_queries": int(n),
        "baseline_mean": round(float(a.mean()), 6),
        "new_mean": round(float(b.mean()), 6),
        "diff": round(float(diff.mean()), 6),
        "rel_pct": round(100 * float(diff.mean()) / float(a.mean()), 2)
        if a.mean() else 0.0,
        "t_stat": round(float(t_stat), 3),
        "p_value": float(p),
        "cohens_d": round(float(d_eff), 4),
        "ci95": [round(float(ci_lo), 6), round(float(ci_hi), 6)],
        "better": better, "worse": worse, "same": same,
        "significant": bool(p < 0.05),
    }


def main() -> int:
    print("=" * 78)
    print("  BM25 调参显著性检验  |  dev 集")
    print("=" * 78)

    split = load_split("dev")
    queries = {q: split.relevant_of(q) for q in split.query_list()}
    print(f"  语料 {split.n_passages:,} | query {len(queries):,}")

    r = BM25Fast.load(Path("data/index/bm25fast_base_dev.pkl"))

    # 基线：默认参数
    print("\n[1/3] 跑基线（k1=1.5, b=0.75）...")
    r.set_params(k1=1.5, b=0.75)
    base = per_query_scores(r, split, queries, ks=(10, 100),
                            metrics=("ndcg", "recall"))

    # 新方法：调参最优
    print("[2/3] 跑调参最优（k1=0.7, b=0.75）...")
    r.set_params(k1=0.7, b=0.75)
    tuned = per_query_scores(r, split, queries, ks=(10, 100),
                             metrics=("ndcg", "recall"))

    # 检验
    print("[3/3] 检验 ...\n")
    common = sorted(set(base) & set(tuned))
    results = []
    for metric in ("ndcg@10", "recall@10", "ndcg@100", "recall@100"):
        a = np.array([base[q][metric] for q in common])
        b = np.array([tuned[q][metric] for q in common])
        results.append(paired_test(a, b, metric))

    print(f"  {'指标':<12} {'基线':>9} {'调参后':>9} {'差值':>9} {'相对':>8} "
          f"{'p 值':>11} {'效应量':>8}")
    print("  " + "-" * 72)
    for r_ in results:
        sig = "***" if r_["p_value"] < 0.001 else ("**" if r_["p_value"] < 0.01
                                                   else ("*" if r_["significant"] else ""))
        print(f"  {r_['metric']:<12} {r_['baseline_mean']:>9.4f} "
              f"{r_['new_mean']:>9.4f} {r_['diff']:>+9.4f} "
              f"{r_['rel_pct']:>+7.2f}% {r_['p_value']:>11.2e} "
              f"{r_['cohens_d']:>8.3f} {sig}")

    print()
    main_r = results[0]
    print("  ⭐ 主指标 ndcg@10 的详细结论：")
    print(f"     95% 置信区间 : [{main_r['ci95'][0]:+.6f}, {main_r['ci95'][1]:+.6f}]")
    print(f"     变好的 query : {main_r['better']:,}")
    print(f"     变差的 query : {main_r['worse']:,}")
    print(f"     不变的 query : {main_r['same']:,}")
    print(f"     效应量 d     : {main_r['cohens_d']} "
          f"（{'可忽略' if abs(main_r['cohens_d']) < 0.2 else '小' if abs(main_r['cohens_d']) < 0.5 else '中' if abs(main_r['cohens_d']) < 0.8 else '大'}）")
    print()
    if main_r["significant"] and main_r["ci95"][0] > 0:
        print("  ✅ 判定：提升【显著】且置信区间不含 0 —— 调参确实有效")
    elif main_r["significant"]:
        print("  ⚠️ p<0.05 但置信区间含 0，结论不稳定，建议扩大样本")
    else:
        print("  ⚠️ 判定：提升【不显著】—— 按 docs/05 应如实报告为噪声")

    out = ROOT / "results" / "2026-10-24_significance_bm25_tuning_dev.json"
    out.write_text(json.dumps({
        "note": "BM25 调参（k1 1.5→0.7, b 不变）的配对 t 检验",
        "baseline": "k1=1.5, b=0.75",
        "tuned": "k1=0.7, b=0.75",
        "split": "dev",
        "tests": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 → {out.relative_to(ROOT)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())