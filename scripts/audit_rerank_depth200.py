"""审查补充实验④：弱第一段（BM25）下重排的 K 上界在哪里？

⭐ 起因（docs/18 §五之三 的明确未解点）：
   审查实验② 扫了 K ∈ {5,10,20,30,50,100}，结论是"弱第一段用大 K"，
   但 BM25 档的 nDCG@10 在 K=100 时【仍在单调上升】（0.7776，+36.25%），
   而 K=100 正好是打分缓存的边界（= 第一段检索深度）。
        ↓
   ⚠️ 这意味着「K=100 是最优」这个说法是【网格边界的产物】，不是结论。
   若就此写成实践建议（"弱段用 K≈100"），可能严重低估，也可能过冲。
        ↓
   ⭐ 本实验只补一个点：K=200。回答"是否仍在上升"，据此定出上界。

⭐ 成本控制：
   · 只在 BM25 弱段上做（强段 D2 的 K=20 甜点已有定论，无需扩展）
   · 复用审查实验① 的 top-100 分数缓存（7 MB），只对 101–200 补打分
   · 新分数增量缓存到独立文件，支持中断续跑

⭐ 边界：
   · 只补 K=200 一个点，不扫 200 以上（每加 100 就多 50% 打分成本）
   · 延迟只是估算（按 K 线性外推），不重测；用途是提示成本量级
   · 仅 dev；test 已锁定

用法：
  .venv\\Scripts\\python.exe scripts\\audit_rerank_depth200.py --n 1500
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
from src.eval.experiment import paired_test  # noqa: E402
from src.eval.metrics import ndcg_at_k  # noqa: E402
from src.rerank.cross_encoder import CrossEncoderReranker  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "index"

FIRST_TOP = 200                       # 第一段检索深度（只为覆盖 K=200）
K_GRID = [20, 50, 100, 200]
BM25_K1, BM25_B = 0.7, 0.75


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500,
                    help="query 数（与审查实验①一致，保证可比）")
    args = ap.parse_args()

    print("=" * 84)
    print("  审查补充实验④：弱第一段（BM25）下重排的 K 上界")
    print("=" * 84)

    t_all = time.time()
    split = load_split("dev")
    qrels = {q: split.relevant_of(q) for q in split.query_list()}

    # 与审查实验①完全同一批 query（用 D2 run 过滤后再取前 n 条）
    run_d2 = json.loads(sorted(RESULTS.glob("*E1_dense_base_dev_run.json"))[-1]
                        .read_text(encoding="utf-8"))
    qids = [q for q in split.query_list()
            if qrels[q] and q in run_d2][:args.n]
    print(f"  query 数：{len(qids):,}（与审查实验①相同）")

    # ── 第一段：BM25 检索到 top-200 ──
    from src.retrieval.bm25_fast import BM25Fast
    bm = BM25Fast.load(CACHE / "bm25fast_dev_k11.5_b0.75.pkl")
    bm.set_params(k1=BM25_K1, b=BM25_B)
    t0 = time.time()
    run200 = bm.run_all(split, top_k=FIRST_TOP, verbose=False)
    print(f"  BM25 top-{FIRST_TOP} 检索 {len(run200):,} query，{time.time()-t0:.0f}s")

    # ── 自检：top-100 必须与 E1 锁定的 run 逐位一致 ──
    run100_lock = json.loads(
        (RESULTS / "2026-10-24_E1_bm25_tuned_dev_run.json")
        .read_text(encoding="utf-8"))
    checked = [q for q in qids[:200] if q in run100_lock]
    mism = sum(1 for q in checked if run200[q][:100] != run100_lock[q][:100])
    print(f"  自检：{len(checked)} 条 query 的 top-100 与 E1 锁定 run 比对，"
          f"不一致 {mism} 条")
    if mism:
        raise SystemExit("⚠️ top-100 不可复现，k1/b 或索引版本不一致，中止")

    # ── 分数：旧缓存（top-100）+ 新补（101-200）──
    # 审查实验① 的缓存固定是 1500q；--n 更小时仍复用它（只取前 n 条 query）
    old_cache = CACHE / "audit_rerank_BM25调优_1500q.json"
    new_cache = CACHE / "audit_rerank_BM25调优_1500q_top200.json"
    if not old_cache.exists():
        raise SystemExit(f"⚠️ 缺少审查实验①的分数缓存 {old_cache.name}")
    scores = json.loads(old_cache.read_text(encoding="utf-8"))
    print(f"  载入旧缓存 {old_cache.name}（{len(scores):,} query 的 top-100 分数）")

    if new_cache.exists():
        extra = json.loads(new_cache.read_text(encoding="utf-8"))
        print(f"  载入补打分缓存 {new_cache.name}（{len(extra):,} query）")
    else:
        extra = {}

    todo = [q for q in qids if q not in extra]
    if todo:
        reranker = CrossEncoderReranker("BAAI/bge-reranker-base")
        reranker._load()
        print(f"  模型：{reranker.stats()['model']} | "
              f"device={reranker.stats()['device']} | "
              f"fp16={reranker.stats()['fp16']}")
        print(f"  需补打分 {len(todo):,} query × 100 对 = {len(todo)*100:,} 对")
        t0 = time.time()
        for i, qid in enumerate(todo, 1):
            cands = run200[qid][:FIRST_TOP]
            have = scores.get(qid, {})
            need = [p for p in cands if p not in have]
            texts = [split.corpus.get(p, "") for p in need]
            if texts:
                sc = reranker.score(split.queries[qid], texts)
                extra[qid] = dict(zip(need, sc))
            else:
                extra[qid] = {}
            if i % 100 == 0:
                el = time.time() - t0
                eta = el / i * (len(todo) - i)
                print(f"    打分 {i:,}/{len(todo):,}（{el:.0f}s，"
                      f"预计还需 {eta/60:.1f} 分钟）...", flush=True)
                new_cache.write_text(json.dumps(extra, ensure_ascii=False),
                                     encoding="utf-8")
        new_cache.write_text(json.dumps(extra, ensure_ascii=False),
                             encoding="utf-8")
        print(f"  补打分完成，{time.time()-t0:.0f}s → {new_cache.name}")
    else:
        print("  ✅ 补打分已完成，直接评测")

    # 合并分数
    merged = {}
    for q in qids:
        d = dict(scores.get(q, {}))
        d.update(extra.get(q, {}))
        merged[q] = d

    # ── 评测 ──
    base = np.array([ndcg_at_k(run200[q][:10], qrels[q], 10) for q in qids])
    print(f"\n  基线（不重排）nDCG@10 = {base.mean():.6f}"
          f"（剩余空间 {(1-base.mean())*100:.1f}%）")
    print(f"\n  {'K':>5} {'nDCG@10':>9} {'提升':>9} {'相对':>8} "
          f"{'占剩余空间':>10} {'Cohen d':>8} {'p 值':>10}")
    print("  " + "-" * 68)

    rows = []
    best = None
    for K in K_GRID:
        reranked = {}
        for q in qids:
            cands = run200[q][:K]
            sc = merged[q]
            reranked[q] = sorted(cands, key=lambda p: (-sc.get(p, float("-inf")), p))
        after = np.array([ndcg_at_k(reranked[q], qrels[q], 10) for q in qids])
        sig = paired_test(base, after)
        used = sig["diff"] / (1 - base.mean()) * 100
        mark = "***" if sig["p_value"] < 0.001 else (
            "**" if sig["p_value"] < 0.01 else (
                "*" if sig["p_value"] < 0.05 else "ns"))
        print(f"  {K:>5} {after.mean():>9.4f} {sig['diff']:>+9.4f} "
              f"{sig['rel_pct']:>+7.2f}% {used:>9.1f}% "
              f"{sig['cohens_d']:>8.3f} {sig['p_value']:>10.2e} {mark}")
        row = {"K": K, "ndcg@10": round(float(after.mean()), 6),
               "gain": sig["diff"], "rel_pct": sig["rel_pct"],
               "cohens_d": sig["cohens_d"], "p_value": sig["p_value"],
               "pct_of_headroom": round(used, 2)}
        rows.append(row)
        if best is None or row["ndcg@10"] > best["ndcg@10"]:
            best = row

    # ── 判定：K=200 是否仍在上升 ──
    k100 = next(r for r in rows if r["K"] == 100)
    k200 = next(r for r in rows if r["K"] == 200)
    still_rising = k200["ndcg@10"] > k100["ndcg@10"]
    print()
    print(f"  K=100 → {k100['ndcg@10']:.4f}（{k100['rel_pct']:+.2f}%）")
    print(f"  K=200 → {k200['ndcg@10']:.4f}（{k200['rel_pct']:+.2f}%）")
    if still_rising:
        print("  ⚠️ K=200 仍在上升 → 最优点仍未被定界（只是把边界推到了 200）")
        print("     实践建议应改为：按【延迟预算】选 K，而不是找最优 K")
    else:
        print("  ✅ K=200 不再上升 → 弱段的最优 K 在 (100, 200] 内")
        print("     可给出上界：弱第一段用 K≈100–200，再大无收益")

    del bm
    out = RESULTS / f"{datetime.now():%Y-%m-%d_%H%M%S}_audit_rerank_depth200.json"
    out.write_text(json.dumps({
        "note": "审查补充实验④：弱第一段（BM25）下重排 K 的上界（dev）",
        "n_queries": len(qids), "first_top": FIRST_TOP, "k_grid": K_GRID,
        "baseline_ndcg10": round(float(base.mean()), 6),
        "rows": rows, "still_rising_at_200": bool(still_rising),
        "latency_note": ("延迟按 E3 的 K 线性关系外推：K=20 实测重排约 75 ms，"
                         "K=200 约 750 ms（估算，非实测）"),
        "total_sec": round(time.time() - t_all, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 → {out.relative_to(ROOT)}")
    print(f"  总用时 {time.time()-t_all:.0f}s")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
