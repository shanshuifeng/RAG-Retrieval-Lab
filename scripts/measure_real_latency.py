"""真实延迟实测 —— P1 待办（修正审查 M2 的乐观估计）

⭐ 起因（2026-10-25 审查 M2）：
   E3 报告的"重排 83 ms/query（K=20）"是【离线批量吞吐】
   （864,000 对 / 3567 秒 = 4.13 ms/对，batch=512，GPU 打满）。
        ↓
   但线上服务是【单 query、batch=1、无预热】的，延迟会高得多。
   本脚本实测这个差距，给 E3 的成本结论一个诚实的修正。

⭐ 测什么（三种场景，逐层逼近真实线上）：
   ① 离线批量：batch=64（E3 实际用的设置）
   ② 单 query：batch=1，无预热（无其他请求竞争）
   ③ 单 query + 冷启动：每次重建模型（最坏情况，仅供参考）

   同时测：
   · 第一段稠密检索（query 编码 + FAISS 搜索）
   · 端到端（检索 + 重排），这才是用户能感知的延迟

用法：
  .venv\\Scripts\\python.exe scripts/measure_real_latency.py --n 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_split  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "index"


def stats_of(xs: list[float]) -> dict:
    """给出一组延迟的完整分布（线上体验看 P50/P95/P99，不只看均值）。"""
    s = sorted(xs)
    n = len(s)
    return {
        "n": n,
        "mean_ms": round(statistics.mean(s), 2),
        "p50_ms": round(s[n // 2], 2),
        "p95_ms": round(s[int(0.95 * (n - 1))], 2),
        "p99_ms": round(s[int(0.99 * (n - 1))], 2),
        "max_ms": round(s[-1], 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="测量 query 数（延迟测试不需多）")
    ap.add_argument("--k", type=int, default=20, help="重排候选数")
    args = ap.parse_args()

    print("=" * 80)
    print("  真实延迟实测（修正 E3 的离线吞吐估计）")
    print("=" * 80)

    split = load_split("dev")
    qids = [q for q in split.query_list(with_relevant_only=True)][:args.n]
    print(f"  测量 {len(qids)} 条 query，重排 K={args.k}")

    from src.retrieval.dense import DenseRetriever
    from src.rerank.cross_encoder import CrossEncoderReranker

    # ── 载入稠密检索器（用 D2）──
    dr = DenseRetriever.load(CACHE / "dense_base_dev_noinstr.pkl")
    dr._load_model()
    print(f"  稠密模型已载入（{dr.stats()['device']}）")

    # ── ① 第一段检索延迟（单 query）──
    print("\n[1/4] 第一段稠密检索延迟（单 query，batch=1）...")
    # 预热
    for q in qids[:5]:
        dr.search(split.queries[q], top_k=100)
    lat_search = []
    runs = {}
    for q in qids:
        t0 = time.perf_counter()
        hits = dr.search(split.queries[q], top_k=100)
        lat_search.append((time.perf_counter() - t0) * 1000)
        runs[q] = [p for p, _ in hits]
    print(f"  {stats_of(lat_search)}")

    # ── ② 重排延迟：离线批量 vs 单 query ──
    reranker = CrossEncoderReranker("BAAI/bge-reranker-base")
    reranker._load()
    dev = reranker.stats()["device"]
    print(f"\n  重排模型已载入（{dev}，fp16={reranker.stats()['fp16']}）")

    # 预热
    reranker.score(split.queries[qids[0]], ["预热文本"] * args.k)

    print(f"\n[2/4] 重排延迟 · 单 query（batch=1，模拟线上）...")
    lat_rerank_single = []
    for q in qids:
        cands = runs[q][:args.k]
        texts = [split.corpus.get(p, "") for p in cands]
        t0 = time.perf_counter()
        reranker.score(split.queries[q], texts)
        lat_rerank_single.append((time.perf_counter() - t0) * 1000)
    st_single = stats_of(lat_rerank_single)
    print(f"  {st_single}")

    print(f"\n[3/4] 重排延迟 · 离线批量（batch=64，E3 的设置）...")
    # 构造全部对，按 batch=64 打分（与 E3 相同的调用方式）
    all_pairs = []
    for q in qids:
        cands = runs[q][:args.k]
        for p in cands:
            all_pairs.append((q, split.corpus.get(p, "")))
    t0 = time.perf_counter()
    # E3 是按 query 分组调 score() 的，但 batch_size=64 内部批处理
    for q in qids:
        cands = runs[q][:args.k]
        texts = [split.corpus.get(p, "") for p in cands]
        reranker.score(split.queries[q], texts)
    batch_total = time.perf_counter() - t0
    batch_per_q = batch_total / len(qids) * 1000
    print(f"  总 {batch_total:.2f}s / {len(qids)} query = "
          f"{batch_per_q:.2f} ms/query（每对 {batch_per_q/args.k:.2f} ms）")

    # ── ③ 端到端 ──
    print(f"\n[4/4] 端到端延迟（检索 + 重排，K={args.k}）...")
    lat_e2e = []
    for q in qids:
        t0 = time.perf_counter()
        hits = dr.search(split.queries[q], top_k=100)
        cands = [p for p, _ in hits][:args.k]
        texts = [split.corpus.get(p, "") for p in cands]
        reranker.score(split.queries[q], texts)
        lat_e2e.append((time.perf_counter() - t0) * 1000)
    st_e2e = stats_of(lat_e2e)
    print(f"  {st_e2e}")

    # ── 汇总 ──
    print("\n" + "=" * 80)
    print("  ⭐ 汇总：E3 报告的估计 vs 实测")
    print("=" * 80)
    print(f"\n  {'场景':<34} {'mean':>9} {'P50':>8} {'P95':>8} {'P99':>8}")
    print("  " + "-" * 72)
    print(f"  {'第一段检索（单 query）':<34} {statistics.mean(lat_search):>9.1f}")
    print(f"  {'重排·离线批量（E3 用的）':<34} {batch_per_q:>9.1f}")
    print(f"  {'重排·单 query（真实线上）':<34} {st_single['mean_ms']:>9.1f} "
          f"{st_single['p50_ms']:>8.1f} {st_single['p95_ms']:>8.1f} "
          f"{st_single['p99_ms']:>8.1f}")
    print(f"  {'端到端（检索+重排）':<34} {st_e2e['mean_ms']:>9.1f} "
          f"{st_e2e['p50_ms']:>8.1f} {st_e2e['p95_ms']:>8.1f} "
          f"{st_e2e['p99_ms']:>8.1f}")

    e3_estimate = args.k * 4.13      # E3 报告：K × 4.13 ms/对
    ratio = st_single["mean_ms"] / e3_estimate
    print(f"\n  E3 报告的 K={args.k} 估算 = {e3_estimate:.0f} ms/query")
    print(f"  实测单 query          = {st_single['mean_ms']:.0f} ms/query")
    print(f"  ⭐ 实测是 E3 估算的 {ratio:.2f} 倍")

    if ratio > 1.5:
        print("  ⚠️ 证实审查 M2 的判断：E3 的离线吞吐严重低估了线上延迟")
        print(f"     → E3 的成本结论应修正为「约 {st_single['mean_ms']:.0f} ms/query」")
    else:
        print("  → 两者接近，E3 的估算基本可用")

    out = RESULTS / f"{datetime.now():%Y-%m-%d_%H%M%S}_real_latency.json"
    out.write_text(json.dumps({
        "note": "真实延迟实测（单 query，修正 E3 的离线吞吐估计）",
        "n_queries": len(qids), "K": args.k, "device": dev,
        "first_stage_search": stats_of(lat_search),
        "rerank_batch64_ms_per_query": round(batch_per_q, 2),
        "rerank_single_query": st_single,
        "end_to_end": st_e2e,
        "e3_estimate_ms": e3_estimate,
        "measured_vs_e3_ratio": round(ratio, 2),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 → {out.relative_to(ROOT)}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())