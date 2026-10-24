"""检索评测指标 —— Recall@K / Precision@K / MRR@K / nDCG@K

⭐ 本模块是项目的【正确性地基】：指标算错则后续所有结论都是错的。
   定义严格遵循 docs/05-评测协议.md 第三节，实现上做如下明确选择：

   1. 相关性为【二值】：qrels 中出现的 docid 记为相关（score=1），未出现记为不相关。
      DuReader/C-MTEB 的 qrels 只有 score=1，故二值化无损。
   2. 排名结果先去重【保序】（重复 docid 只保留首次出现），避免重复项虚增指标。
   3. Precision@K 的分母：当返回结果少于 K 时用实际返回数（len(topk)），
      否则用 K —— 这样"只返回 3 条却全对"不会被误判为 100% 精确率。
   4. nDCG 的 IDCG 必须由【同一组 label 的最优排序】算出，
      实现中用 sum(sorted(gains)[::-1][:k])，并断言 DCG <= IDCG + 1e-9。
   5. 无相关文档的 query：单指标函数返回 0.0（或在 evaluate 层被过滤）。

用法：
    from src.eval.metrics import ndcg_at_k, recall_at_k, evaluate
    ndcg_at_k(["A","D","B","E","F"], {"A","B","C"}, 5)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

# ⭐ 浮点比较容差（docs/05 明确要求 DCG <= IDCG + 1e-9）
EPS = 1e-9


def _dedup(ids: Iterable[str]) -> list[str]:
    """去重保序。重复的 docid 不应在排名中被重复计分。"""
    seen: set[str] = set()
    out: list[str] = []
    for d in ids:
        s = str(d)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _check_k(k: int) -> None:
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"K 必须是正整数，收到 {k!r}")


# ────────────────────────────────────────────────────────────
# 基础指标（单 query）
# ────────────────────────────────────────────────────────────
def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Recall@K = |Top-K ∩ Relevant| / |Relevant|

    相关文档一个都没有时返回 0.0（调用方应先过滤这类 query）。
    """
    _check_k(k)
    rel = {str(x) for x in relevant}
    if not rel:
        return 0.0
    topk = set(_dedup(ranked)[:k])
    return len(topk & rel) / len(rel)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Precision@K = |Top-K ∩ Relevant| / K（返回不足 K 时分母用实际条数）"""
    _check_k(k)
    rel = {str(x) for x in relevant}
    topk = _dedup(ranked)[:k]
    if not topk:
        return 0.0
    denom = min(k, len(topk))
    return len(set(topk) & rel) / denom


def mrr_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """MRR@K = 1 / rank（第一个相关文档的位置，从 1 起）；K 内没有则 0.0

    ⚠️ 2026-10-24 修复：此处曾写成 `rel = (str(x) for x in relevant)`（生成器），
       导致两个后果：
         1) `if not rel` 恒为 False（生成器对象永远为真），空标注不再被正确返回 0；
         2) 生成器只能消费一次，`for` 循环里的 `d in rel` 会边判断边耗尽它，
            使本该命中的文档被判为不命中 → MRR 系统性偏低。
       必须用【集合推导】，与 recall/precision/ndcg 保持一致。
    """
    _check_k(k)
    rel = {str(x) for x in relevant}
    if not rel:
        return 0.0
    for i, d in enumerate(_dedup(ranked)[:k], start=1):
        if d in rel:
            return 1.0 / i
    return 0.0

def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """DCG@K = Σ_{i=1..K} (2^g_i - 1) / log2(i+1)"""
    _check_k(k)
    total = 0.0
    for i, g in enumerate(gains[:k], start=1):
        total += (2.0 ** g - 1.0) / math.log2(i + 1)
    return total


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """nDCG@K = DCG@K / IDCG@K ∈ [0, 1]

    二值相关性：命中为增益 1.0，未命中为 0.0。
    """
    _check_k(k)
    rel = {str(x) for x in relevant}
    if not rel:
        return 0.0

    topk = _dedup(ranked)[:k]
    gains = [1.0 if d in rel else 0.0 for d in topk]
    dcg = dcg_at_k(gains, k)

    # ⭐ IDCG：同一组 label 的【最优排序】——相关文档全部排在最前
    n_ideal = min(len(rel), k)
    ideal_gains = [1.0] * n_ideal
    idcg = dcg_at_k(ideal_gains, k)

    if idcg <= 0.0:
        return 0.0
    # ⭐ 硬断言：DCG 不可能超过 IDCG（docs/05 明确要求的自检）
    assert dcg <= idcg + EPS, (
        f"DCG({dcg}) > IDCG({idcg})，实现有 bug："
        f"ranked={list(ranked)[:k]}, relevant={sorted(rel)[:10]}..."
    )
    return dcg / idcg


# ────────────────────────────────────────────────────────────
# 批量评测
# ────────────────────────────────────────────────────────────
METRICS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}


def evaluate(
    queries: dict[str, Iterable[str]],
    run: dict[str, Sequence[str]],
    ks: Sequence[int] = (1, 5, 10, 20, 100),
    metrics: Sequence[str] = ("recall", "mrr", "ndcg"),
    skip_empty_relevant: bool = True,
) -> dict:
    """批量评测。

    Args:
        queries: {qid: relevant_ids}
        run:     {qid: ranked_docid 列表（按分数降序）}
        ks:      K 值列表（所有方法必须一致）
        metrics: 要算的指标名（见 METRICS）
        skip_empty_relevant: 过滤「无相关标注」的 query 并在结果中记录数量

    Returns:
        {
          "per_query": {qid: {("ndcg",10): 0.5, ...}},
          "aggregate": {("ndcg",10): {"mean": ..., "n": ...}, ...},
          "filtered": {...计数...}
        }
    """
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise ValueError(f"未知指标 {unknown}，可选：{list(METRICS)}")
    ks = list(ks)

    per_query: dict[str, dict[str, float]] = {}
    missing_run: list[str] = []
    n_empty = 0
    n_used = 0

    for qid, rel_ids in queries.items():
        rel = {str(x) for x in rel_ids}
        if not rel and skip_empty_relevant:
            n_empty += 1
            continue
        if qid not in run:
            missing_run.append(qid)
            continue
        ranked = run[qid]
        scores: dict[str, float] = {}
        for k in ks:
            for m in metrics:
                scores[f"{m}@{k}"] = METRICS[m](ranked, rel, k)
        per_query[qid] = scores
        n_used += 1

    aggregate: dict[str, dict[str, float]] = {}
    all_keys = sorted({k for s in per_query.values() for k in s})
    for key in all_keys:
        vals = [s[key] for s in per_query.values() if key in s]
        aggregate[key] = {
            "mean": sum(vals) / len(vals) if vals else 0.0,
            "n": len(vals),
        }

    return {
        "per_query": per_query,
        "aggregate": aggregate,
        "filtered": {
            "empty_relevant": n_empty,
            "missing_in_run": len(missing_run),
            "missing_in_run_qids": missing_run[:20],
            "queries_evaluated": n_used,
        },
        "config": {
            "ks": ks,
            "metrics": list(metrics),
            "skip_empty_relevant": skip_empty_relevant,
        },
    }


def format_table(aggregate: dict[str, dict[str, float]], ks: Sequence[int],
                 metrics: Sequence[str] = ("recall", "mrr", "ndcg")) -> str:
    """把 aggregate 渲染成对齐的文本表（用于日志/报告草稿）。

    ⚠️ header 与 rows 必须按【同一顺序】构建：这里统一为
       行 = 指标（metric），列 = K 值。K 值顺序由 ks 决定。
    """
    ks = list(ks)
    metrics = list(metrics)
    header = [f"K={k}" for k in ks]
    rows: list[tuple[str, list[str]]] = []
    for m in metrics:
        row = [f"{aggregate.get(f'{m}@{k}', {}).get('mean', 0.0):.4f}" for k in ks]
        rows.append((m.upper(), row))
    widths = [max(len(header[i]),
                  *(len(r[1][i]) for r in rows) if rows else 0)
              for i in range(len(header))]
    name_w = max([len("METRIC")] + [len(r[0]) for r in rows])
    out = [" " * name_w + "  " + "  ".join(h.ljust(w) for h, w in zip(header, widths))]
    for name, row in rows:
        out.append(name.ljust(name_w) + "  "
                   + "  ".join(v.ljust(w) for v, w in zip(row, widths)))
    return "\n".join(out)