"""P7 · 公平成本测量 —— 回答风险 R8

⭐ 要解决的问题：
   E1 里 BM25 用 CPU（0.63 ms/q）、稠密用 GPU（0.62 ms/q）——
   硬件不同，检索速度不可直接比较。

⭐ 本脚本的做法（公平的关键）：
   FAISS 向量检索本身是 CPU 的（装的是 faiss-cpu），所以两端唯一不同的是
   【query 编码】这一步的设备。把同一个稠密检索分别在 GPU 和 CPU 上测一遍
   query 编码 + 检索的端到端延迟，就得到公平的对比。

   段落向量用已缓存的索引（不重新编码 10 万段，那与设备无关、已一次性付过）。

测量项：
   · 稠密检索在 GPU 上的每 query 端到端延迟（编码 + 检索）
   · 稠密检索在 CPU 上的每 query 端到端延迟（编码 + 检索）
   · BM25 在 CPU 上的每 query 延迟（已有缓存，直接测）
   → 三者同表对比，这才是"贵多少"的诚实答案

用法：
  .venv\\Scripts\\python.exe scripts\\measure_fair_cost.py --n 300
"""

from __future__ import annotations

import argparse
import json
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


def measure_dense(split, qids: list[str], device: str,
                  model: str = "small", top_k: int = 100) -> dict:
    """在指定设备上测稠密检索的端到端延迟。"""
    import torch
    from src.retrieval.dense import DenseRetriever

    r = DenseRetriever.load(CACHE / f"dense_{model}_{split.name}_noinstr.pkl")
    # 强制设备
    r._device = device
    # 重新加载模型到目标设备
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(r.model_name, device=device,
                            cache_folder=r.cache_folder)
    m.max_seq_length = r.max_length
    if r.use_fp16 and device == "cuda":
        m.half()
    r._model = m

    texts = [split.queries[q] for q in qids]
    # 预热（GPU 首次有初始化开销）
    _ = r.search_batch(texts[:8], top_k=top_k)

    t0 = time.perf_counter()
    r.search_batch(texts, top_k=top_k)
    total = time.perf_counter() - t0
    return {
        "device": device,
        "model": model,
        "n_queries": len(qids),
        "total_sec": round(total, 2),
        "ms_per_query": round(1000 * total / len(qids), 3),
    }


def measure_bm25(split, qids: list[str], top_k: int = 100) -> dict:
    from src.retrieval.bm25_fast import BM25Fast
    r = BM25Fast.load(CACHE / "bm25fast_base_dev.pkl")
    r.set_params(k1=0.7, b=0.75)
    texts = [split.queries[q] for q in qids]
    # 预热
    _ = r.search(texts[0], top_k=top_k)
    t0 = time.perf_counter()
    for q in texts:
        r.search(q, top_k=top_k)
    total = time.perf_counter() - t0
    return {
        "device": "cpu",
        "method": "BM25(调优 k1=0.7)",
        "n_queries": len(qids),
        "total_sec": round(total, 2),
        "ms_per_query": round(1000 * total / len(qids), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="测量 query 数")
    ap.add_argument("--model", default="small", choices=["small", "base"])
    args = ap.parse_args()

    print("=" * 78)
    print("  P7 · 公平成本测量（同任务、同数据、分设备）")
    print("=" * 78)

    split = load_split("dev")
    qids = split.query_list(with_relevant_only=True)[:args.n]
    print(f"  测量 {len(qids)} 条 query，模型 {args.model}")

    rows = []

    print("\n[1/3] BM25（CPU）...")
    rows.append(measure_bm25(split, qids))
    print(f"  {rows[-1]['ms_per_query']:.3f} ms/query")

    print(f"\n[2/3] 稠密 {args.model}（GPU）...")
    rows.append(measure_dense(split, qids, "cuda", args.model)
              | {"method": f"dense/{args.model}"})
    print(f"  {rows[-1]['ms_per_query']:.3f} ms/query")

    print(f"\n[3/3] 稠密 {args.model}（CPU）...")
    rows.append(measure_dense(split, qids, "cpu", args.model)
              | {"method": f"dense/{args.model}"})
    print(f"  {rows[-1]['ms_per_query']:.3f} ms/query")

    # 汇总
    print("\n" + "=" * 78)
    print("  公平成本对比（端到端每 query 延迟）")
    print("=" * 78)
    print(f"  {'方法':<18} {'设备':>6} {'ms/query':>10} {'相对 BM25':>12}")
    print("  " + "-" * 50)
    bm25_ms = rows[0]["ms_per_query"]
    for r in rows:
        ratio = r["ms_per_query"] / bm25_ms if bm25_ms else 0
        print(f"  {r['method']:<18} {r['device']:>6} {r['ms_per_query']:>10.3f} "
              f"{ratio:>11.1f}×")

    out = RESULTS / f"{datetime.now():%Y-%m-%d_%H%M%S}_fair_cost_dev.json"
    out.write_text(json.dumps({
        "note": "公平成本测量：同任务同数据，分设备",
        "split": "dev", "n_queries": len(qids), "model": args.model,
        "rows": rows,
        "interpretation": ("FAISS 检索本身两端都是 CPU；唯一差别是 query 编码设备。"
                           "稠密在 CPU 的延迟才是与 BM25 公平对比的数字。"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 → {out.relative_to(ROOT)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())