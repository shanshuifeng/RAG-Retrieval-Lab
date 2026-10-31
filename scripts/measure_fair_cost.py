"""P7 · 公平成本测量 —— 回答风险 R8（2026-10-31 修正口径）

⭐ 要解决的问题：
   E1 里 BM25 用 CPU（0.63 ms/q）、稠密用 GPU（0.62 ms/q）——
   硬件不同，检索速度不可直接比较。

⭐ 本脚本的做法（公平的关键）：
   FAISS 向量检索本身是 CPU 的（装的是 faiss-cpu），所以两端唯一不同的是
   【query 编码】这一步的设备。把同一个稠密检索分别在 GPU 和 CPU 上测一遍
   query 编码 + 检索的端到端延迟，就得到公平的对比。

   段落向量用已缓存的索引（不重新编码 10 万段，那与设备无关、已一次性付过）。

⭐⭐ 2026-10-31 修正：测量口径必须分【单 query】与【批量摊销】两种
   旧版本 BM25 是逐条 search（batch=1），稠密却用 search_batch（batch=300），
   两者放在同一张"公平对比"表里 → 稠密的 4.1 ms/q 是批处理摊销值，
   真实单条约 25 ms，被低估约 6 倍。
        ↓
   现在两端各测两种口径，并在每条结果里记录 batch_size：
     · single（batch=1）  ：线上真实延迟，用于回答"慢多少"
     · batch（batch=N）   ：离线吞吐，用于回答"跑批要多久"
   ⚠️ 两个数字都对，但混在一起看就是错的。详见 docs/05 §7.3。

⭐⭐ 2026-10-31 实测暴露的两个不确定性（必须一并记录，否则会误引用）
   1) 同一 batch=300 口径，本日测得稠密 GPU 4.50 ms/q，
      而 2026-10-25 测得 0.706 ms/q，相差 6.4 倍，⭐ 原因未定位
      （环境：torch 2.14.1+cu130，GPU 时钟 2370/3105 MHz）。
      → 因此本项目只用【同一进程内的口径倍数关系】，不引用绝对 ms。
   2) fp32 与 fp16 在单条口径下几乎相同（91.9 vs 92.6 ms），
      批量为 4.50 vs 4.94 ms/q。
      → 早期一次 20 次循环的诊断曾观察到 fp16 单条更慢（47 vs 35 ms），
        正式 300 条测量【未能复现】，故不作任何 fp16 结论。

测量项（BM25 × 1 + 稠密 × 3 配置，各测单条 + 批量）：
   · BM25（CPU，倒排索引，无批处理概念）
   · 稠密 GPU · fp32 / fp16
   · 稠密 CPU · fp32

用法：
  .venv\\Scripts\\python.exe scripts\\measure_fair_cost.py --n 300 --model base
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_split  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "index"


def measure_dense_modes(split, qids: list[str], device: str,
                        model: str = "small", top_k: int = 100,
                        dtype: str = "auto") -> list[dict]:
    """在指定设备+精度下，分别测【单 query】与【批量摊销】两种口径。

    ⭐ 模型只加载一次，两种口径复用同一实例（否则重复加载会占掉大半时间）。

    ⭐ dtype 为什么要分开测：
       fp16 的收益高度依赖批大小与硬件，不能想当然。
       本日 300 条实测：单条 fp32 91.9 ms vs fp16 92.6 ms（无差异），
       批量 fp32 4.50 ms/q vs fp16 4.94 ms/q（fp16 略慢）。
       ⚠️ 一次 20 次循环的诊断曾观察到 fp16 单条更慢，正式测量未复现
          → 不作结论，只保留测量能力供后续复测。
    """
    from src.retrieval.dense import DenseRetriever
    from sentence_transformers import SentenceTransformer

    r = DenseRetriever.load(CACHE / f"dense_{model}_{split.name}_noinstr.pkl")
    # 强制设备
    r._device = device
    # 重新加载模型到目标设备
    m = SentenceTransformer(r.model_name, device=device,
                            cache_folder=r.cache_folder)
    m.max_seq_length = r.max_length
    if dtype == "fp16":
        m.half()
    elif dtype == "fp32":
        m.float()
    elif r.use_fp16 and device == "cuda":      # auto：沿用索引记录的配置
        m.half()
    r._model = m
    used_dtype = "fp16" if dtype == "fp16" else (
        "fp16" if dtype == "auto" and r.use_fp16 and device == "cuda" else "fp32")

    texts = [split.queries[q] for q in qids]
    rows = []

    # ── 口径 1：单 query（batch=1）—— 线上真实延迟 ──
    _ = r.search(texts[0], top_k=top_k)          # 预热（首次有初始化开销）
    t0 = time.perf_counter()
    for t in texts:
        r.search(t, top_k=top_k)
    total = time.perf_counter() - t0
    rows.append({
        "device": device, "model": model, "method": f"dense/{model}",
        "dtype": used_dtype,
        "mode": "single", "batch_size": 1,
        "n_queries": len(qids), "total_sec": round(total, 2),
        "ms_per_query": round(1000 * total / len(qids), 3),
    })

    # ── 口径 2：批量摊销（batch=len(qids)）—— 离线吞吐 ──
    _ = r.search_batch(texts[:8], top_k=top_k)   # 预热
    t0 = time.perf_counter()
    r.search_batch(texts, top_k=top_k)
    total = time.perf_counter() - t0
    rows.append({
        "device": device, "model": model, "method": f"dense/{model}",
        "dtype": used_dtype,
        "mode": "batch", "batch_size": len(qids),
        "n_queries": len(qids), "total_sec": round(total, 2),
        "ms_per_query": round(1000 * total / len(qids), 3),
    })
    return rows


def measure_bm25(split, qids: list[str], top_k: int = 100) -> dict:
    """BM25 在 CPU 上的每 query 延迟。

    ⭐ BM25 的检索是逐条进行的（倒排索引没有批处理概念），
       所以它的"单条"与"批量"口径本质相同，只有一个数字。
       这里显式记为 mode=single，避免与稠密的两种口径混淆。
    """
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
        "mode": "single", "batch_size": 1,
        "n_queries": len(qids),
        "total_sec": round(total, 2),
        "ms_per_query": round(1000 * total / len(qids), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="测量 query 数")
    ap.add_argument("--model", default="base", choices=["small", "base"])
    args = ap.parse_args()

    print("=" * 84)
    print("  P7 · 公平成本测量（同任务、同数据、分设备、⭐ 分口径）")
    print("=" * 84)

    split = load_split("dev")
    qids = split.query_list(with_relevant_only=True)[:args.n]
    print(f"  测量 {len(qids)} 条 query，模型 {args.model}")

    rows = []

    print("\n[1/4] BM25（CPU，倒排索引逐条）...")
    rows.append(measure_bm25(split, qids))
    print(f"  {rows[-1]['ms_per_query']:.3f} ms/query")

    for i, (dev, dt) in enumerate((("cuda", "fp32"), ("cuda", "fp16"),
                                   ("cpu", "fp32")), start=2):
        print(f"\n[{i}/4] 稠密 {args.model}（{dev} · {dt}：单 query + 批量）...")
        part = measure_dense_modes(split, qids, dev, args.model, dtype=dt)
        rows.extend(part)
        for x in part:
            print(f"  {x['mode']:<6} batch={x['batch_size']:<5} "
                  f"{x['ms_per_query']:>9.3f} ms/query")

    # ── 汇总：按口径分组，避免跨口径比较 ──
    print("\n" + "=" * 84)
    print("  公平成本对比（⭐ 只有同一 mode 内才可比）")
    print("=" * 84)
    for mode, title in (("single", "口径 A：单 query（线上真实延迟，batch=1）"),
                        ("batch", "口径 B：批量摊销（离线吞吐，batch=N）")):
        sub = [r for r in rows if r["mode"] == mode]
        if not sub:
            continue
        print(f"\n  {title}")
        print(f"  {'方法':<14} {'设备':>5} {'精度':>5} {'batch':>7} "
              f"{'ms/query':>10} {'相对':>9}")
        print("  " + "-" * 62)
        base = next((r for r in sub if "BM25" in r["method"]), None)
        base_ms = base["ms_per_query"] if base else 0.0
        for r in sub:
            ratio = r["ms_per_query"] / base_ms if base_ms else 0.0
            rel = f"{ratio:>8.1f}×" if base else "—"
            dt = r.get("dtype", "—")
            print(f"  {r['method']:<14} {r['device']:>5} {dt:>5} "
                  f"{r['batch_size']:>7} {r['ms_per_query']:>10.3f} {rel:>9}")

    out = RESULTS / f"{datetime.now():%Y-%m-%d_%H%M%S}_fair_cost_dev.json"
    out.write_text(json.dumps({
        "note": "公平成本测量：同任务同数据，分设备 + ⭐ 分口径（2026-10-31 口径修正）",
        "split": "dev", "n_queries": len(qids), "model": args.model,
        "rows": rows,
        "interpretation": (
            "FAISS 检索本身两端都是 CPU；唯一差别是 query 编码设备与精度。"
            "⭐ 必须按 mode 分组看：single=线上真实延迟（batch=1），"
            "batch=离线吞吐摊销（batch=N）。跨 mode 比较会把稠密低估数十倍。"
            "BM25 是倒排索引逐条检索，无批处理概念，只有 single 口径。"
            "⚠️ 绝对 ms 依赖当前环境（同一 batch=300 口径本日比 10-25 慢 6.4 倍，"
            "原因未定位），故只使用同一进程内的【口径倍数关系】。"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 → {out.relative_to(ROOT)}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
