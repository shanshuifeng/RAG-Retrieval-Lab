"""审查补充实验③：分块的劣势是「位置挤压」还是「检索质量下降」？

⭐ 起因（docs/16 §原因③ 遗留的未判定问题）：
   E4 结论是"分块全面劣于不切分"。报告给出的机理之一是
   「段落级投影的位置稀释」：同一父段落的多个块在投影后只占一个位置，
   有效候选变少，@10 的召回空间被压缩。
        ↓
   但这个解释【从未被判定】—— 它有一个可直接检验的推论：
     若劣势真来自"top-10 槽位稀缺"，那么放宽 K 后槽位不再稀缺，
     劣势应当【收敛】甚至反转。
     若劣势来自"块本身的检索信号变差"，那么放宽 K 后应【等比例保持】。
        ↓
   ⭐ 两个假设给出相反的预测，所以这是一个有判别力的实验
      H1（位置挤压）：劣势随 K 增大而缩小
      H2（质量下降）：劣势随 K 增大而保持

⭐ 做法（只改一个变量：检索深度 K）：
   · 不做块级宽松评测（会引入新的相关性口径，无法与 E4 主表对比）
   · 不改分词/参数/模型 —— 与 E4 完全相同（BM25 调优 k1=0.7, b=0.75）
   · 只把评测的 K 从 10 扫到 100，语言仍是"块级检索 → 投影回段落 → 段落级 qrels"
     → 与 docs/16 的 @10 数字【逐位可比】（本脚本会先复现 @10 作为自检）

⭐ 附带机理证据（零额外成本）：
   `dup@10` = top-10 块里有几个位置被同一父段落重复占用
   → 直接量化"位置稀释"的强度。若 H1 成立，dup 应与劣势幅度正相关。

⚠️ 边界：仅 dev；仅 BM25；K 上限 100（第一段检索深度）；
   不宣称"分块更好"，只判定"劣势是否随深度收敛"。

用法：
  .venv\\Scripts\\python.exe scripts\\audit_e4_depth.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.chunking import build_chunked_corpus, chunked_split_view  # noqa: E402
from src.data.loader import load_split  # noqa: E402
from src.eval.experiment import paired_test, per_query_metric  # noqa: E402
from src.retrieval.bm25_fast import BM25Fast  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "index"

DEPTH = 100                      # 第一段检索深度（与 E4 一致）
K_GRID = [10, 20, 50, 100]       # 评测深度
STRATEGIES = ["fixed256", "fixed512", "paragraph"]
BM25_K1, BM25_B = 0.7, 0.75      # E1 锁定值，与 E4 相同


def dup_rate(run_chunk: dict[str, list[str]], cc, k: int = 10) -> float:
    """top-k 块中，被同一父段落重复占用的位置比例。

    例：top-10 块来自 8 个不同段落 → 2 个位置被重复占用 → 0.20
    native（不切分）恒为 0。
    """
    vals = []
    for ids in run_chunk.values():
        top = ids[:k]
        if not top:
            vals.append(0.0)
            continue
        uniq = len({cc.chunk_to_parent[c] for c in top})
        vals.append(1.0 - uniq / len(top))
    return float(np.mean(vals)) if vals else 0.0


def main() -> int:
    print("=" * 88)
    print("  审查补充实验③：分块劣势是「位置挤压」(H1) 还是「质量下降」(H2)？")
    print("=" * 88)

    t_all = time.time()
    split = load_split("dev")
    qrels = {q: split.relevant_of(q) for q in split.query_list()}

    # native 基线：E1 已锁定的 BM25 调优 run（top-100）
    native_run = json.loads(
        (RESULTS / "2026-10-24_E1_bm25_tuned_dev_run.json")
        .read_text(encoding="utf-8"))

    qids = [q for q in qrels if qrels[q] and q in native_run]
    print(f"  query 数：{len(qids):,}（有相关标注且 native 有检索结果）")
    print(f"  评测深度 K：{K_GRID}（第一段检索深度上限 {DEPTH}）")

    def per_q(run: dict, metric: str, k: int) -> np.ndarray:
        d = per_query_metric({q: run.get(q, []) for q in qids},
                             {q: qrels[q] for q in qids}, metric, k)
        return np.array([d[q] for q in qids])

    # ── native 基线 ──
    print("\n[基线] native（不切分，复用 E1 run）")
    native = {}
    for metric in ("recall", "ndcg", "mrr"):
        for k in K_GRID:
            native[(metric, k)] = per_q(native_run, metric, k)
    print(f"  Recall@10={native[('recall', 10)].mean():.4f}  "
          f"nDCG@10={native[('ndcg', 10)].mean():.4f}  "
          f"Recall@100={native[('recall', 100)].mean():.4f}")

    rows = []
    for strategy in STRATEGIES:
        print(f"\n{'=' * 70}\n  策略 {strategy}\n{'=' * 70}")
        t0 = time.time()

        cc = build_chunked_corpus(split, strategy, verbose=False)
        view = chunked_split_view(split, cc)
        print(f"  分块：{cc.stats()['n_chunks']:,} 块"
              f"（{cc.stats()['chunks_per_parent_mean']} 块/段）")

        cache = CACHE / f"bm25fast_chunked_{strategy}_dev.pkl"
        bm = BM25Fast()
        if cache.exists():
            bm = BM25Fast.load(cache)
            print(f"  命中索引缓存 {cache.name}")
        else:
            bm.build_base(view, verbose=False)
            bm.save(cache)
        bm.set_params(k1=BM25_K1, b=BM25_B)

        run_chunk = bm.run_all(view, top_k=DEPTH, verbose=False)
        run_proj = {q: cc.project_ranking(ids) for q, ids in run_chunk.items()}
        del bm

        dup10 = dup_rate({q: run_chunk.get(q, []) for q in qids}, cc, k=10)
        dup100 = dup_rate({q: run_chunk.get(q, []) for q in qids}, cc, k=100)
        print(f"  块冲突率：top-10 = {dup10:.1%}，top-100 = {dup100:.1%}"
              f"（构建+检索 {time.time()-t0:.0f}s）")

        row = {"strategy": strategy, "corpus": cc.stats(),
               "dup_rate@10": round(dup10, 4),
               "dup_rate@100": round(dup100, 4), "by_k": {}}

        print(f"\n  {'K':>4} {'Recall':>9} {'vs native':>11} {'nDCG':>9} "
              f"{'vs native':>11} {'MRR':>8} {'p(Recall)':>11}")
        print("  " + "-" * 74)
        for k in K_GRID:
            cell = {}
            for metric in ("recall", "ndcg", "mrr"):
                cur = per_q(run_proj, metric, k)
                sig = paired_test(native[(metric, k)], cur)
                cell[metric] = {"mean": round(float(cur.mean()), 6),
                                "rel_pct": sig["rel_pct"],
                                "p_value": sig["p_value"],
                                "cohens_d": sig["cohens_d"]}
            mark = "***" if cell["recall"]["p_value"] < 0.001 else (
                "**" if cell["recall"]["p_value"] < 0.01 else (
                    "*" if cell["recall"]["p_value"] < 0.05 else "ns"))
            print(f"  {k:>4} {cell['recall']['mean']:>9.4f} "
                  f"{cell['recall']['rel_pct']:>+10.2f}% "
                  f"{cell['ndcg']['mean']:>9.4f} "
                  f"{cell['ndcg']['rel_pct']:>+10.2f}% "
                  f"{cell['mrr']['mean']:>8.4f} "
                  f"{cell['recall']['p_value']:>10.2e} {mark}")
            row["by_k"][k] = cell
        rows.append(row)

    # ── ⭐ 判定：劣势是否随 K 收敛 ──
    print("\n" + "=" * 88)
    print("  ⭐ 判定：分块劣势（Recall vs native）随深度 K 的变化")
    print("=" * 88)
    print(f"  {'策略':<12} " + " ".join(f"{'R@'+str(k):>10}" for k in K_GRID)
          + f" {'趋势':>14} {'dup@10':>8}")
    print("  " + "-" * 82)

    verdicts = {}
    for r in rows:
        gaps = [r["by_k"][k]["recall"]["rel_pct"] for k in K_GRID]
        abs_gaps = [abs(g) for g in gaps]
        # 收敛判据：最强劣势点在 K=10，且 K=100 时劣势缩小一半以上
        shrunk = abs_gaps[-1] < abs_gaps[0] * 0.5
        trend = "收敛（H1）" if shrunk else (
            "保持（H2）" if abs_gaps[-1] > abs_gaps[0] * 0.8 else "部分收敛")
        verdicts[r["strategy"]] = {
            "gaps_by_k": dict(zip([str(k) for k in K_GRID], gaps)),
            "verdict": trend,
        }
        print(f"  {r['strategy']:<12} "
              + " ".join(f"{g:>+9.2f}%" for g in gaps)
              + f" {trend:>14} {r['dup_rate@10']:>7.1%}")

    print()
    worst = max(rows, key=lambda r: abs(r["by_k"][10]["recall"]["rel_pct"]))
    wgaps = [worst["by_k"][k]["recall"]["rel_pct"] for k in K_GRID]
    print(f"  最差策略 {worst['strategy']} 的劣势序列："
          + " → ".join(f"{g:+.2f}%" for g in wgaps))
    print(f"  其 top-10 块冲突率：{worst['dup_rate@10']:.1%}"
          f"（即 {worst['dup_rate@10']*10:.1f}/10 个位置被同一段落重复占用）")
    print()
    if verdicts[worst["strategy"]]["verdict"].startswith("收敛"):
        print("  ✅ 支持 H1：劣势随 K 增大而收敛")
        print("     → docs/16 的「位置稀释」解释成立，分块的问题主要在 top-10 槽位竞争")
    else:
        print("  ✅ 支持 H2：劣势随 K 增大仍保持")
        print("     → 分块损害的是块本身的检索信号，不是评测口径造成的假象")
        print("     → docs/16 的「位置稀释」只是次要因素，主要原因是切碎了语义单元")

    out = RESULTS / f"{datetime.now():%Y-%m-%d_%H%M%S}_audit_e4_depth.json"
    out.write_text(json.dumps({
        "note": "审查补充实验③：分块劣势随检索深度 K 的收敛性（dev，仅 BM25）",
        "n_queries": len(qids), "k_grid": K_GRID, "depth": DEPTH,
        "bm25_params": {"k1": BM25_K1, "b": BM25_B},
        "native": {f"{m}@{k}": round(float(native[(m, k)].mean()), 6)
                   for m in ("recall", "ndcg", "mrr") for k in K_GRID},
        "strategies": rows,
        "verdicts": verdicts,
        "total_sec": round(time.time() - t_all, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 → {out.relative_to(ROOT)}")
    print(f"  总用时 {time.time()-t_all:.0f}s")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    sys.exit(main())
