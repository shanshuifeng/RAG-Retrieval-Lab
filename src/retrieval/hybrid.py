"""混合检索（Hybrid Retrieval）—— RRF 融合

⭐ RRF（Reciprocal Rank Fusion）：
   score(d) = Σ_i  w_i / (k + rank_i(d))

   · rank_i(d)：文档 d 在第 i 个检索器结果里的排名（从 1 起；没出现则不计入）
   · k：平滑常数（越大越弱化头部差异；惯例 60）
   · w_i：第 i 个检索器的权重（默认等权）

⭐ 为什么用 RRF 而不是分数加权：
   BM25 的分数和稠密检索的余弦相似度【量纲完全不同】，直接加权不可比。
   RRF 只用【排名】，天然免疫量纲差异——这正是混合检索选它的原因
   （docs/06 §2.3）。

⭐ E2 的量化动机（E1 实测，非推测）：
   BM25 与 D1 的 Top-10 平均 IoU 仅 0.2143 —— 两者结果只重合 21.4%，
   3,169/8,640（36.7%）的 query 上取并集能多命中相关文档。

用法：
    from src.retrieval.hybrid import rrf_fuse
    fused = rrf_fuse({"bm25": run_bm25, "dense": run_dense}, k=60)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

DEFAULT_K = 60


def rrf_fuse(
    runs: Mapping[str, Mapping[str, Sequence[str]]],
    k: int = DEFAULT_K,
    weights: Mapping[str, float] | None = None,
    top_k: int = 100,
) -> dict[str, list[str]]:
    """RRF 融合多个检索结果。

    Args:
        runs:    {方法名: {qid: [pid 按分数降序]}}
        k:       平滑常数（>0）。越大，头部排名的优势越被弱化。
        weights: 各方法权重（缺省等权 1.0）
        top_k:   融合后每个 query 返回的最大长度

    Returns:
        {qid: [pid 按 RRF 分数降序]}

    ⚠️ 实现约束：
      · 排名从 1 开始（rank 1 = 第一名）
      · 某方法没召回的文档不参与该方法的贡献（不是 rank=∞，而是没有这项）
      · 分数相同的文档按 pid 字典序稳定排序（可复现）
    """
    if k <= 0:
        raise ValueError(f"RRF 的 k 必须 > 0，收到 {k}")
    if not runs:
        return {}

    methods = list(runs.keys())
    if weights is None:
        weights = {m: 1.0 for m in methods}
    else:
        unknown = set(weights) - set(methods)
        if unknown:
            raise ValueError(f"weights 含未知方法: {unknown}")

    # 收集所有 qid（各方法的 query 集应一致；不一致时取并集并记录）
    all_qids: list[str] = []
    seen: set[str] = set()
    for m in methods:
        for qid in runs[m]:
            if qid not in seen:
                seen.add(qid)
                all_qids.append(qid)

    fused: dict[str, list[str]] = {}
    for qid in all_qids:
        score: dict[str, float] = {}
        for m in methods:
            ranked = runs[m].get(qid) or []
            w = float(weights.get(m, 1.0))
            for rank, pid in enumerate(ranked, start=1):
                score[pid] = score.get(pid, 0.0) + w / (k + rank)
        # 分数降序；并列时按 pid 字典序（保证可复现）
        order = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
        fused[qid] = [pid for pid, _ in order[:top_k]]
    return fused


def rrf_fuse_with_scores(
    runs: Mapping[str, Mapping[str, Sequence[str]]],
    k: int = DEFAULT_K,
    weights: Mapping[str, float] | None = None,
    top_k: int = 100,
) -> dict[str, list[tuple[str, float]]]:
    """同 rrf_fuse，但返回 (pid, rrf_score)，用于调试与报告。"""
    if k <= 0:
        raise ValueError(f"RRF 的 k 必须 > 0，收到 {k}")
    if not runs:
        return {}
    methods = list(runs.keys())
    if weights is None:
        weights = {m: 1.0 for m in methods}

    fused: dict[str, list[tuple[str, float]]] = {}
    all_qids: set[str] = set()
    for m in methods:
        all_qids.update(runs[m].keys())
    for qid in all_qids:
        score: dict[str, float] = {}
        for m in methods:
            ranked = runs[m].get(qid) or []
            w = float(weights.get(m, 1.0))
            for rank, pid in enumerate(ranked, start=1):
                score[pid] = score.get(pid, 0.0) + w / (k + rank)
        order = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
        fused[qid] = order[:top_k]
    return fused


def weighted_score_fuse(
    runs_with_scores: Mapping[str, Mapping[str, Sequence[tuple[str, float]]]],
    weights: Mapping[str, float] | None = None,
    top_k: int = 100,
    normalize: str = "minmax",
) -> dict[str, list[str]]:
    """⭐ 备选：归一化分数加权融合（不是 RRF）。

    供 E2 消融对比：验证"只用排名的 RRF"与"用归一化分数"的差异。
    默认 min-max 归一化到 [0,1] 再加权。

    ⚠️ 分数融合对量纲敏感，必须归一化；这也是选 RRF 当主线的原因。
    """
    if not runs_with_scores:
        return {}
    methods = list(runs_with_scores.keys())
    if weights is None:
        weights = {m: 1.0 for m in methods}

    all_qids: set[str] = set()
    for m in methods:
        all_qids.update(runs_with_scores[m].keys())

    fused: dict[str, list[str]] = {}
    for qid in all_qids:
        score: dict[str, float] = {}
        for m in methods:
            ranked = runs_with_scores[m].get(qid) or []
            if not ranked:
                continue
            vals = [s for _, s in ranked]
            lo, hi = min(vals), max(vals)
            rng = hi - lo
            w = float(weights.get(m, 1.0))
            for pid, s in ranked:
                norm = (s - lo) / rng if rng > 1e-12 else 0.0
                score[pid] = score.get(pid, 0.0) + w * norm
        order = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
        fused[qid] = [pid for pid, _ in order[:top_k]]
    return fused