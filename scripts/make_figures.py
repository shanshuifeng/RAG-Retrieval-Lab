"""图表生成 —— P9-2（results/figures/）

⭐ 生成最终报告所需的全部图表（数据全部来自 results/ 里的权威 JSON，
   不写死任何数字——数据变了图自动跟着变）。

产出：
  fig1_methods_test.png      各方法在 test 上的 nDCG@10 对比（最终结论）
  fig2_e3_k_sweep.png        E3 重排的 K 效果-成本权衡曲线
  fig3_bm25_grid.png         BM25 调参网格热力图（k1 × b）
  fig4_dev_vs_test.png       dev 结论在 test 上的复现检验
  fig5_effect_sizes.png      ⭐ 三手段效应量对比（本项目最有价值的一张图）

用法：
  .venv\\Scripts\\python.exe scripts\\make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

# ⭐ 中文显示（Windows 自带字体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def latest(pattern: str) -> dict:
    files = sorted(RESULTS.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return json.loads(files[-1].read_text(encoding="utf-8"))


def fig1_methods_test() -> None:
    """图 1：test 集各方法 nDCG@10（最终结论图）。"""
    d = latest("*_test_final.json")
    methods = list(d["methods"].keys())
    vals = [d["methods"][m]["results"]["ndcg@10"] for m in methods]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#888", "#4a90d9", "#2e7d32", "#c0c0c0", "#e65100"]
    bars = ax.barh(methods, vals, color=colors[:len(methods)])
    ax.set_xlabel("nDCG@10（test 集，已锁定）")
    ax.set_title("检索方法对比 —— test 集最终评测")
    ax.set_xlim(0, 1.0)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=10)
    ax.axvline(d["methods"]["BM25(调优)"]["results"]["ndcg@10"],
               color="#888", linestyle="--", alpha=0.5, label="BM25 基线")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fig1_methods_test.png", dpi=150)
    plt.close()


def fig2_e3_k_sweep() -> None:
    """图 2：E3 重排的 K 效果-成本权衡。"""
    d = latest("*_E3_rerank_d2_dev.json")
    ks = [r["K"] for r in d["k_sweep"]]
    ndcg = [r["ndcg@10"] for r in d["k_sweep"]]
    lat = [r["est_latency_ms_per_query"] for r in d["k_sweep"]]
    base = d["first_stage_baseline_ndcg10"]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(ks, ndcg, "o-", color="#2e7d32", lw=2, label="nDCG@10")
    ax1.axhline(base, color="#888", linestyle="--", alpha=0.6,
                label=f"第一段基线 {base:.4f}")
    ax2.plot(ks, lat, "s--", color="#c62828", lw=2, label="估算延迟")
    ax1.set_xlabel("重排候选数 K")
    ax1.set_ylabel("nDCG@10", color="#2e7d32")
    ax2.set_ylabel("估算延迟 ms/query", color="#c62828")
    ax1.set_title("E3 重排序：K 的效果-成本权衡（dev）")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="lower right")
    plt.tight_layout()
    plt.savefig(FIGURES / "fig2_e3_k_sweep.png", dpi=150)
    plt.close()


def fig3_bm25_grid() -> None:
    """图 3：BM25 调参网格热力图。"""
    d = latest("*_E1_bm25_grid_dev.json")
    rows = d["grid"]
    k1s = sorted({r["k1"] for r in rows})
    bs = sorted({r["b"] for r in rows})
    z = np.zeros((len(bs), len(k1s)))
    for r in rows:
        z[bs.index(r["b"]), k1s.index(r["k1"])] = r["ndcg@10"]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(z, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(k1s)), [str(k) for k in k1s])
    ax.set_yticks(range(len(bs)), [str(b) for b in bs])
    ax.set_xlabel("k1（词频饱和）")
    ax.set_ylabel("b（长度归一化）")
    ax.set_title("BM25 网格调参 nDCG@10（dev）")
    for i in range(len(bs)):
        for j in range(len(k1s)):
            ax.text(j, i, f"{z[i, j]:.3f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig3_bm25_grid.png", dpi=150)
    plt.close()


def fig4_dev_vs_test() -> None:
    """图 4：dev 结论在 test 上的复现检验。"""
    comp = latest("*E1_comparison_dev.json")
    test = latest("*_test_final.json")

    groups = ["D1 vs BM25", "D2 vs BM25", "RRF vs D2", "重排 vs D2"]
    dev_rel = []
    for t_ in comp["significance_tests"]:
        dev_rel.append(t_["rel_pct"])
    # dev 的 RRF vs D2 与 重排 vs D2 需要单独从 E2/E3 结果取
    e2 = latest("*_E2_rrf_dev.json")
    e3 = latest("*_E3_rerank_d2_dev.json")
    dev_rel = [
        next(t for t in comp["significance_tests"]
             if "D1/BGE-small vs BM25" in t["comparison"])["rel_pct"],
        next(t for t in comp["significance_tests"]
             if "D2/BGE-base vs BM25" in t["comparison"])["rel_pct"],
        e2["verdict_vs_best_single"]["gain_pct"],
        e3["significance_vs_first_stage"]["rel_pct"],
    ]
    test_rel_map = {t["comparison"]: t["rel_pct"] for t in test["significance"]}
    test_rel = [
        test_rel_map["D1/BGE-small vs BM25(调优)"],
        test_rel_map["D2/BGE-base vs BM25(调优)"],
        # RRF vs D2 在 test 上没直接测，从指标算
        round(100 * (test["methods"]["RRF(BM25+D2)"]["results"]["ndcg@10"]
                     - test["methods"]["D2/BGE-base"]["results"]["ndcg@10"])
              / test["methods"]["D2/BGE-base"]["results"]["ndcg@10"], 2),
        test_rel_map["D2+重排(K=20) vs D2/BGE-base"],
    ]

    x = np.arange(len(groups))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, dev_rel, w, label="dev", color="#90caf9")
    b2 = ax.bar(x + w / 2, test_rel, w, label="test（已锁定）", color="#e65100")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_ylabel("相对提升 %（nDCG@10）")
    ax.set_title("dev 结论在 test 上的复现检验")
    ax.set_xticks(x, groups)
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + (1.5 if h > 0 else -4),
                    f"{h:+.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig4_dev_vs_test.png", dpi=150)
    plt.close()


def fig5_effect_sizes() -> None:
    """图 5：⭐ 三手段效应量对比（本项目最有价值的一张图）。"""
    comp = latest("*E1_comparison_dev.json")
    e2 = latest("*_E2_rrf_dev.json")
    e3 = latest("*_E3_rerank_d2_dev.json")
    sig_bm25 = latest("*significance_bm25_tuning_dev.json")

    labels = ["BM25 调参", "重排序（K=20）", "混合检索（RRF）", "换稠密检索（D1）"]
    ds = [
        next(t for t in sig_bm25["tests"] if t["metric"] == "ndcg@10")["cohens_d"],
        e3["significance_vs_first_stage"]["cohens_d"],
        next(t for t in e2["significance"] if "D2" in t["comparison"])["cohens_d"],
        next(t for t in comp["significance_tests"]
             if "D1/BGE-small vs BM25" in t["comparison"])["cohens_d"],
    ]
    colors = ["#90a4ae", "#ffb74d", "#c62828", "#2e7d32"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, ds, color=colors)
    ax.set_xlabel("Cohen's d（效应量，dev 集）")
    ax.set_title("四种改进手段的效应量对比 ——「换方法」远大于「调参数/重排」")
    ax.axvline(0.2, color="#888", linestyle=":", label="d=0.2（可忽略/小 分界）")
    ax.axvline(0.5, color="#555", linestyle=":", label="d=0.5（小/中 分界）")
    ax.axvline(0.8, color="#333", linestyle=":", label="d=0.8（中/大 分界）")
    for bar, v in zip(bars, ds):
        ax.text(v + 0.02 if v > 0 else v - 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center",
                ha="left" if v > 0 else "right", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(min(ds) - 0.15, max(ds) + 0.15)
    plt.tight_layout()
    plt.savefig(FIGURES / "fig5_effect_sizes.png", dpi=150)
    plt.close()


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("生成图表（数据全部来自 results/ 权威 JSON）...")
    fig1_methods_test()
    print("  ✅ fig1_methods_test.png")
    fig2_e3_k_sweep()
    print("  ✅ fig2_e3_k_sweep.png")
    fig3_bm25_grid()
    print("  ✅ fig3_bm25_grid.png")
    fig4_dev_vs_test()
    print("  ✅ fig4_dev_vs_test.png")
    fig5_effect_sizes()
    print("  ✅ fig5_effect_sizes.png")
    print(f"\n全部图表 → {FIGURES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())