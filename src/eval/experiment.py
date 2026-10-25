"""公共实验工具 —— 消除 scripts/ 下 5 处重复实现

⭐ 起因（2026-10-25 审查 L1）：
   `paired_test` 出现在 5 个脚本里（run_e2_hybrid / run_e3_rerank /
   run_e4_chunking / run_final_test / significance_test），
   评估逻辑也重复了 5-6 处。
        ↓
   风险：改一处忘另一处 → 各实验口径漂移。
   本项目已因"手抄数字"踩过坑（MRR 误抄）。

⭐ 本模块提供：
   · paired_test()      配对 t 检验 + 效应量 + bootstrap CI（单一实现）
   · per_query_metric() 逐 query 指标（统一口径）
   · evaluate_run()     一次算完多个 K 的指标
   · effect_label()     效应量等级（统一分级标准）

⚠️ 已有脚本【不强制改回】——它们已产出结果并入库，改动会引入风险。
   本模块供【后续新增实验】使用，并在文档中标注"新旧并存"的过渡状态。
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from src.eval.metrics import METRICS, ndcg_at_k, recall_at_k

# 效应量分级（Cohen 惯例）
EFFECT_THRESHOLDS = [(0.2, "可忽略"), (0.5, "小"), (0.8, "中"), (float("inf"), "大")]


def effect_label(d: float) -> str:
    """把 Cohen's d 映射到等级标签。"""
    a = abs(d)
    for th, label in EFFECT_THRESHOLDS:
        if a < th:
            return label
    return "大"


def paired_test(a: np.ndarray, b: np.ndarray,
                n_boot: int = 10000, seed: int = 42) -> dict:
    """配对 t 检验 + Cohen's d + bootstrap 95% CI。

    Args:
        a: 基线方法在每个 query 上的指标（长度 = query 数）
        b: 新方法在同一批 query 上的指标（顺序必须与 a 对齐）

    Returns:
        dict（含 diff / rel_pct / p_value / cohens_d / ci95 / better / worse）
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"a 与 b 长度不一致：{a.shape} vs {b.shape}")
    if len(a) < 2:
        raise ValueError(f"配对检验至少需要 2 个样本，收到 {len(a)}")

    diff = b - a
    t, p = stats.ttest_rel(b, a)
    sd = diff.std(ddof=1)
    d = diff.mean() / sd if sd > 0 else 0.0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])

    return {
        "n": int(len(diff)),
        "a_mean": round(float(a.mean()), 6),
        "b_mean": round(float(b.mean()), 6),
        "diff": round(float(diff.mean()), 6),
        "rel_pct": round(100 * float(diff.mean()) / float(a.mean()), 2)
        if a.mean() else 0.0,
        "t_stat": round(float(t), 3),
        "p_value": float(p),
        "cohens_d": round(float(d), 4),
        "effect": effect_label(d),
        "ci95": [round(float(lo), 6), round(float(hi), 6)],
        "significant": bool(p < 0.05),
        "better": int((diff > 1e-12).sum()),
        "worse": int((diff < -1e-12).sum()),
        "same": int(len(diff) - (diff > 1e-12).sum() - (diff < -1e-12).sum()),
    }


def per_query_metric(run: dict[str, list[str]],
                     qrels: dict[str, list[str]],
                     metric: str = "ndcg", k: int = 10) -> dict[str, float]:
    """计算每个 query 的指标，返回 {qid: score}（qrels 为空的 query 被跳过）。"""
    if metric not in METRICS:
        raise ValueError(f"未知指标 {metric}，可选：{list(METRICS)}")
    fn = METRICS[metric]
    out: dict[str, float] = {}
    for qid, rel in qrels.items():
        if not rel:
            continue
        out[qid] = fn(run.get(qid, []), rel, k)
    return out


def evaluate_run(run: dict[str, list[str]], qrels: dict[str, list[str]],
                 ks: tuple[int, ...] = (1, 5, 10, 20, 100),
                 metrics: tuple[str, ...] = ("recall", "ndcg", "mrr")) -> dict:
    """一次算完多个 K × 多个指标的均值（统一口径，避免各脚本自己拼）。"""
    result: dict[str, float] = {}
    for k in ks:
        for m in metrics:
            if m not in METRICS:
                continue
            vals = [METRICS[m](run.get(q, []), rel, k)
                    for q, rel in qrels.items() if rel]
            if vals:
                result[f"{m}@{k}"] = round(float(np.mean(vals)), 6)
    return result


def k_sweep(run_builder, qrels: dict[str, list[str]], k_values: list[int],
            metric: str = "ndcg", eval_k: int = 10) -> list[dict]:
    """对一组 K 值做扫描评测（重排/候选数实验的通用模式）。

    Args:
        run_builder: 可调用对象，输入 k 返回 {qid: [pid]} 的 run
        k_values:   要扫描的 K 列表
        metric:     用哪个指标评估（默认 ndcg）
        eval_k:     评估的 @K（默认 10）
    """
    base = per_query_metric(run_builder(0) if False else
                            {q: [] for q in qrels}, qrels, metric, eval_k)
    rows = []
    for k in k_values:
        run = run_builder(k)
        scores = per_query_metric(run, qrels, metric, eval_k)
        rows.append({
            "k": k,
            "mean": round(float(np.mean(list(scores.values()))), 6),
            "n": len(scores),
        })
    return rows


__all__ = ["paired_test", "per_query_metric", "evaluate_run",
           "effect_label", "k_sweep", "EFFECT_THRESHOLDS"]