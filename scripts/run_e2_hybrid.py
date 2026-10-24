"""E2 · 混合检索（RRF）—— P5

⭐ 量化动机（E1 实测）：BM25 与 D1 的 Top-10 平均 IoU 仅 0.2143，
   即两方法结果只有 21.4% 重合，36.7% 的 query 上取并集能多命中。

实验组（docs/06 §2.4）：
  · 单方法基线：BM25(调优) / D1 / D2
  · RRF(BM25调优, D1)、RRF(BM25调优, D2)
  · RRF 的 k 网格：{20, 40, 60, 80, 100}
  · ⭐ 消融：加权分数融合 vs RRF（验证"只用排名"是否更合适）

纪律：
  · 只在 dev 上跑
  · 每个 RRF 结果与两个单方法都做配对显著性检验
  · 融合是纯计算（不重跑检索），成本 ≈ 0 —— 这正是 E2 的第一个结论素材

用法：
  .venv\\Scripts\\python.exe scripts\\run_e2_hybrid.py --split dev
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
from src.retrieval.bm25_fast import BM25Fast  # noqa: E402
from src.retrieval.hybrid import rrf_fuse, weighted_score_fuse  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "index"

K_GRID = [20, 40, 60, 80, 100]
# ⭐ 权重网格：验证"等权 RRF 失败是否只是权重问题"
#    w_weak = 0 即退化为纯强方法（对照），往上加看是否改善
WEAK_WEIGHT_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]


def load_run(pattern: str) -> dict[str, list[str]]:
    files = sorted(RESULTS.glob(pattern))
    if not files:
        raise FileNotFoundError(f"找不到 {pattern}")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """a=基线, b=新方法。逐 query 配对 t 检验 + 效应量 + bootstrap CI。"""
    diff = b - a
    n = len(diff)
    t, p = stats.ttest_rel(b, a)
    sd = diff.std(ddof=1)
    d = diff.mean() / sd if sd > 0 else 0.0
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(10000, n))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return {
        "n": int(n),
        "diff": round(float(diff.mean()), 6),
        "rel_pct": round(100 * float(diff.mean()) / float(a.mean()), 2)
        if a.mean() else 0.0,
        "p_value": float(p),
        "cohens_d": round(float(d), 4),
        "ci95": [round(float(lo), 6), round(float(hi), 6)],
        "better": int((diff > 1e-12).sum()),
        "worse": int((diff < -1e-12).sum()),
    }


def per_q_ndcg(run: dict, queries: dict) -> dict[str, float]:
    return {qid: ndcg_at_k(run.get(qid, []), rel, 10)
            for qid, rel in queries.items() if rel}


def main() -> int:
    ap = argparse.ArgumentParser(description="E2 RRF 混合检索")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    if args.split == "test":
        print("❌ E2 调参禁止在 test 上运行")
        return 2

    print("=" * 80)
    print(f"  E2 · 混合检索（RRF）  |  split={args.split}")
    print("=" * 80)

    t_all = time.time()
    split = load_split(args.split)
    queries = {q: split.relevant_of(q) for q in split.query_list()}
    print(f"  语料 {split.n_passages:,} | query {len(queries):,}")

    # ── 1. 准备单方法 run ──
    print("\n[1/5] 准备单方法检索结果 ...")
    run_bm25_default = load_run("*E1_bm25_dev_run.json")
    run_d1 = load_run("*E1_dense_small_dev_run.json")
    run_d2 = load_run("*E1_dense_base_dev_run.json")

    # BM25 调优：现场重算（基础统计缓存 + set_params，约 5 秒）
    print("  BM25(调优 k1=0.7) 现场重算 ...")
    tuned = BM25Fast.load(CACHE / "bm25fast_base_dev.pkl")
    tuned.set_params(k1=0.7, b=0.75)
    run_bm25_tuned = tuned.run_all(split, top_k=args.top_k, verbose=False)
    # ⭐ 顺手补上缺失的存档（compare_e1.py 此前只在内存里算过）
    tuned_run_out = RESULTS / "2026-10-24_E1_bm25_tuned_dev_run.json"
    if not tuned_run_out.exists():
        tuned_run_out.write_text(json.dumps(run_bm25_tuned, ensure_ascii=False),
                                 encoding="utf-8")
        print(f"  ⭐ 补存档 → {tuned_run_out.name}")

    singles = {
        "BM25(调优)": run_bm25_tuned,
        "D1/small": run_d1,
        "D2/base": run_d2,
    }
    single_scores = {}
    print("\n  单方法基线（nDCG@10）：")
    for name, run in singles.items():
        res = evaluate(queries, run, ks=[10, 100], metrics=("recall", "ndcg"))
        single_scores[name] = per_q_ndcg(run, queries)
        print(f"    {name:<12} nDCG@10={res['aggregate']['ndcg@10']['mean']:.6f}  "
              f"Recall@10={res['aggregate']['recall@10']['mean']:.6f}")

    # ── 2. RRF 融合 + k 网格 ──
    print("\n[2/5] RRF 融合 + k 网格 ...")
    combos = [("BM25(调优)", "D1/small"), ("BM25(调优)", "D2/base")]
    grid_rows = []
    best = None
    for a_name, b_name in combos:
        ra, rb = singles[a_name], singles[b_name]
        for k in K_GRID:
            fused = rrf_fuse({a_name: ra, b_name: rb}, k=k, top_k=args.top_k)
            res = evaluate(queries, fused, ks=[10], metrics=("ndcg", "recall"))
            row = {
                "pair": f"{a_name}+{b_name}",
                "k": k,
                "ndcg@10": round(res["aggregate"]["ndcg@10"]["mean"], 6),
                "recall@10": round(res["aggregate"]["recall@10"]["mean"], 6),
            }
            grid_rows.append(row)
            print(f"    {row['pair']:<26} k={k:>3}  "
                  f"nDCG@10={row['ndcg@10']:.4f}  R@10={row['recall@10']:.4f}")
            if best is None or row["ndcg@10"] > best["ndcg@10"]:
                best = row | {"fused": fused}

    # ── 3. 消融①：权重扫描（⭐ 验证"等权失败是否只是权重问题"）──
    #    背景：等权 RRF 输了 → 可能因为两方法质量悬殊（D2 比 BM25 高 53%），
    #    弱方法拖累强方法。若调权重能救回，说明只是配置问题；救不回才是真失败。
    print("\n[3/6] 消融①：RRF 权重扫描（弱方法 BM25 的权重从 0 到 1）...")
    weight_rows = []
    for a_name, b_name in combos:
        ra, rb = singles[a_name], singles[b_name]
        print(f"    {a_name}（弱）+ {b_name}（强），k=20：")
        print(f"      {'w_弱':>6} {'w_强':>5} {'nDCG@10':>9}")
        for w_weak in WEAK_WEIGHT_GRID:
            fused = rrf_fuse({a_name: ra, b_name: rb}, k=20,
                             weights={a_name: w_weak, b_name: 1.0},
                             top_k=args.top_k)
            res = evaluate(queries, fused, ks=[10], metrics=("ndcg",))
            v = round(res["aggregate"]["ndcg@10"]["mean"], 6)
            weight_rows.append({"pair": f"{a_name}+{b_name}",
                                "w_weak": w_weak, "w_strong": 1.0, "ndcg@10": v})
            print(f"      {w_weak:>6.2f} {1.0:>5.1f} {v:>9.4f}")

    # ── 4. 消融②：分数加权融合 vs RRF ──
    print("\n[4/6] 消融②：分数加权融合 vs RRF（用最优 pair 与 k）...")
    a_name, b_name = best["pair"].split("+")
    ra, rb = singles[a_name], singles[b_name]
    # 需要带分数的 run 才能做分数融合——这里只有排序，故用排名构造伪分数
    # （run 文件只存了 pid 序列，没存原始分数；rank 倒数即"排名分数"）
    # ⭐ 诚实的做法：明确说明这是"排名分数"融合，不是原始分数
    def rank_as_score(run: dict) -> dict:
        return {q: [(p, 1.0 / (i + 1)) for i, p in enumerate(pids)]
                for q, pids in run.items()}
    sf = weighted_score_fuse({"a": rank_as_score(ra), "b": rank_as_score(rb)},
                             top_k=args.top_k)
    res_sf = evaluate(queries, sf, ks=[10], metrics=("ndcg", "recall"))
    print(f"    分数融合（排名倒数当分数）: nDCG@10="
          f"{res_sf['aggregate']['ndcg@10']['mean']:.6f}  "
          f"R@10={res_sf['aggregate']['recall@10']['mean']:.6f}")
    print(f"    RRF（k={best['k']}）          : nDCG@10={best['ndcg@10']:.6f}  "
          f"R@10={best['recall@10']:.6f}")

    # ── 5. 显著性检验（最优混合 vs 两个单方法）──
    print("\n[5/6] 显著性检验 ...")
    fused_scores = per_q_ndcg(best["fused"], queries)
    common = lambda m: np.array([single_scores[m][q] for q in fused_scores])
    fused_arr = np.array([fused_scores[q] for q in fused_scores])
    tests = []
    for m in best["pair"].split("+"):
        tests.append({"comparison": f"混合 vs {m}", **paired_test(common(m), fused_arr)})

    print(f"\n  {'对比':<26} {'差值':>9} {'相对':>8} {'p 值':>11} {'d':>7} "
          f"{'变好':>6} {'变差':>6}")
    print("  " + "-" * 74)
    for t_ in tests:
        sig = "***" if t_["p_value"] < 0.001 else ("**" if t_["p_value"] < 0.01
                                                   else ("*" if t_["p_value"] < 0.05 else "ns"))
        print(f"  {t_['comparison']:<26} {t_['diff']:>+9.4f} "
              f"{t_['rel_pct']:>+7.2f}% {t_['p_value']:>11.2e} "
              f"{t_['cohens_d']:>7.3f} {t_['better']:>6} {t_['worse']:>6} {sig}")

    # ── 6. 汇总输出 ──
    print("\n[6/6] 结论 ...")
    winner = best["pair"]
    best_single = max((n for n in winner.split("+")),
                      key=lambda m: float(np.mean(list(single_scores[m].values()))))
    best_single_val = float(np.mean(list(single_scores[best_single].values())))
    gain = best["ndcg@10"] - best_single_val
    print(f"  最优混合：{winner}  k={best['k']}")
    print(f"  最优混合 nDCG@10 = {best['ndcg@10']:.6f}")
    print(f"  最强单方法（{best_single}）nDCG@10 = {best_single_val:.6f}")
    print(f"  ⭐ 混合提升 = {gain:+.6f}（{100*gain/best_single_val:+.2f}%）")
    t_test = next(t for t in tests if best_single in t["comparison"])
    print(f"  ⭐ vs 最强单方法：p={t_test['p_value']:.2e}，d={t_test['cohens_d']:.3f}，"
          f"CI=[{t_test['ci95'][0]:+.6f}, {t_test['ci95'][1]:+.6f}]")
    if t_test["ci95"][0] > 0:
        print("  ✅ 混合显著优于最强单方法")
    elif t_test["ci95"][1] < 0:
        print("  ⚠️ 混合显著【差于】最强单方法（负面结果，如实报告）")
    else:
        print("  → 混合与最强单方法差异不显著（如实报告）")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = RESULTS / f"{stamp}_E2_rrf_{args.split}.json"
    payload = {
        "experiment_id": "E2",
        "method": "rrf",
        "split": args.split,
        "date": datetime.now().isoformat(timespec="seconds"),
        "config": {"k_grid": K_GRID, "top_k": args.top_k, "weights": "equal"},
        "singles_baseline": {
            n: round(float(np.mean(list(v.values()))), 6)
            for n, v in single_scores.items()},
        "grid": [{k: v for k, v in r.items() if k != "fused"} for r in grid_rows],
        "weight_sweep": weight_rows,
        "best": {k: v for k, v in best.items() if k != "fused"},
        "score_fusion_ablation": {
            "ndcg@10": round(res_sf["aggregate"]["ndcg@10"]["mean"], 6),
            "recall@10": round(res_sf["aggregate"]["recall@10"]["mean"], 6),
            "note": "用排名倒数当分数（run 文件未存原始分数），仅作 RRF 对照"},
        "significance": tests,
        "verdict_vs_best_single": {
            "best_single": best_single,
            "gain": round(gain, 6),
            "gain_pct": round(100 * gain / best_single_val, 2),
        },
        "cost_note": "融合是纯排名计算，无需重跑检索，额外成本 ≈ 0",
        "total_sec": round(time.time() - t_all, 1),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # 最优混合的 run 也落盘（E3 重排序要用它当第一段）
    run_out = RESULTS / f"{stamp}_E2_rrf_{args.split}_best_run.json"
    run_out.write_text(json.dumps(best["fused"], ensure_ascii=False), encoding="utf-8")

    print(f"\n  结果 → {out.relative_to(ROOT)}")
    print(f"  最优 run → {run_out.relative_to(ROOT)}")
    print(f"  总用时 {payload['total_sec']}s（融合零检索成本）")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())