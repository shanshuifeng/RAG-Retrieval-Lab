"""指标真实数据交叉验证 —— M2.3 的第二道保险

单元测试用的是人工算例；本脚本用【真实 dev/test 数据】做端到端自检：
  · oracle 排序（把 relevant 全排最前）→ Recall@K/nDCG@K 必须 = 1.0（K 足够大时）
  · random 排序 → 指标应接近随机基线（约 |rel|/|corpus| 量级）
  · 反 oracle（相关文档排最后）→ nDCG 应显著低于 random
  · ⭐ DCG <= IDCG 断言在真实数据上是否始终成立

这一步能抓出「单元测试没想到但在真实分布下暴露」的 bug。

用法：
  .venv\\Scripts\\python.exe scripts\\verify_metrics_real.py --split dev --n 500
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.metrics import ndcg_at_k, precision_at_k, recall_at_k  # noqa: E402

PROC = ROOT / "data" / "processed"


def load_split(name: str, n_limit: int) -> tuple[list[dict], list[str]]:
    d = PROC / name
    corpus_ids: list[str] = []
    with open(d / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus_ids.append(json.loads(line)["id"])
    queries: list[dict] = []
    with open(d / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            if q["relevant_ids"]:
                queries.append(q)
            if n_limit and len(queries) >= n_limit:
                break
    return queries, corpus_ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--n", type=int, default=500, help="评测 query 数上限")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 70)
    print(f"  真实数据指标交叉验证 —— split={args.split}, n={args.n}")
    print("=" * 70)

    t0 = time.time()
    queries, corpus_ids = load_split(args.split, args.n)
    print(f"  语料 {len(corpus_ids):,} 段；有正例的 query {len(queries):,} 条"
          f"（用时 {time.time()-t0:.1f}s）")

    corpus_set = set(corpus_ids)
    rng = random.Random(args.seed)

    # 检查正例都在语料里
    missing = 0
    for q in queries:
        missing += sum(1 for r in q["relevant_ids"] if r not in corpus_set)
    print(f"  ⭐ 正例悬空引用：{missing}")

    ks = [1, 10, 100]
    metric_names = ("recall", "ndcg", "precision")
    results: dict[str, dict[str, list[float]]] = {
        tag: {f"{m}@{k}": [] for m in metric_names for k in ks}
        for tag in ("oracle", "random", "anti")}

    for q in queries:
        rel = [r for r in q["relevant_ids"] if r in corpus_set]
        rel_set = set(rel)
        if not rel_set:
            continue
        k_max = 100
        # ── oracle：相关全在最前，其余随机填充 ──
        others = [c for c in rng.sample(corpus_ids, min(len(corpus_ids), k_max * 4))
                  if c not in rel_set]
        oracle_rank = (rel + others)[:k_max]
        # ── anti：相关全在最后 ──
        #   ⚠️ 注意：若直接用 others+rel 截断到 k_max，rel 会被完全截掉，
        #   结果与 random 一样是 0，无法区分「排序差」与「没召回」。
        #   正确做法：构造恰好 k_max 长的列表，让正例排在最后但仍被计入。
        filler_needed = max(0, k_max - len(rel))
        anti_rank = (others[:filler_needed] + rel)[:k_max]
        # ── random：全体随机 ──
        rand_rank = rng.sample(corpus_ids, k_max)

        for tag, ranked in (("oracle", oracle_rank), ("random", rand_rank),
                            ("anti", anti_rank)):
            for k in ks:
                results[tag][f"recall@{k}"].append(recall_at_k(ranked, rel_set, k))
                results[tag][f"ndcg@{k}"].append(ndcg_at_k(ranked, rel_set, k))
                results[tag][f"precision@{k}"].append(precision_at_k(ranked, rel_set, k))

    print()
    print(f"  {'方法':<8} {'Recall@10':>12} {'nDCG@10':>10} {'P@10':>8} "
          f"{'nDCG@100':>10}")
    print("  " + "-" * 56)
    summary = {}
    for tag in ("oracle", "random", "anti"):
        r = results[tag]
        mean = {key: sum(v) / len(v) for key, v in r.items()}
        summary[tag] = mean
        print(f"  {tag:<8} {mean['recall@10']:>12.4f} {mean['ndcg@10']:>10.4f} "
              f"{mean['precision@10']:>8.4f} {mean['ndcg@100']:>10.4f}")

    print()
    ok = True

    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
        ok = ok and cond

    # ⭐ 断言 1：oracle 的 Recall@100 必须是 1.0（只要 |rel| <= 100）
    n_rel_lte100 = sum(1 for q in queries if len(q["relevant_ids"]) <= 100)
    check("oracle Recall@100 == 1.0",
          abs(summary["oracle"]["recall@100"] - 1.0) < 1e-9
          and n_rel_lte100 == len(queries),
          f"{summary['oracle']['recall@100']:.6f}（全部 {len(queries)} 条 query 正例数 ≤100）")

    # ⭐ 断言 2：oracle 的 nDCG@100 必须是 1.0
    check("oracle nDCG@100 == 1.0",
          abs(summary["oracle"]["ndcg@100"] - 1.0) < 1e-9,
          f"{summary['oracle']['ndcg@100']:.6f}")

    # ⭐ 断言 3：oracle nDCG@10 应很高但可能 < 1（正例数可能 > 10）
    check("oracle nDCG@10 >= 0.99",
          summary["oracle"]["ndcg@10"] >= 0.99,
          f"{summary['oracle']['ndcg@10']:.6f}")

    # ⭐ 断言 4：排序必须单调 oracle > random > anti
    #   注：random 在 10 万段/取 100 的池子里命中率极低（≈0.1%），
    #   anti 把正例压到末尾但仍在 Top-100 内 → anti 的 Recall@100 应接近 1
    #   而 nDCG@100 很低。故用 nDCG@100 比较 anti vs random 更稳定。
    check("nDCG@10 单调 oracle > random >= anti",
          summary["oracle"]["ndcg@10"] > summary["random"]["ndcg@10"]
          >= summary["anti"]["ndcg@10"] - 1e-12,
          f"{summary['oracle']['ndcg@10']:.4f} > "
          f"{summary['random']['ndcg@10']:.4f} >= {summary['anti']['ndcg@10']:.4f}")

    # ⭐ 断言 4b：anti 的 Recall@100 应接近 1（正例都在池内），但 nDCG@100 明显低
    #   这正是「排序质量」与「召回」分离的关键证据
    check("anti Recall@100 ≈ 1 但 nDCG@100 明显偏低（排序≠召回）",
          summary["anti"]["recall@100"] > 0.99
          and summary["anti"]["ndcg@100"] < summary["oracle"]["ndcg@100"] - 0.1,
          f"Recall@100={summary['anti']['recall@100']:.4f}, "
          f"nDCG@100={summary['anti']['ndcg@100']:.4f}")

    # ⭐ 断言 5：random 的 nDCG@100 应很小（100 段 / 10 万段，命中概率 ~0.1%）
    check("random nDCG@100 接近随机基线 (<0.01)",
          summary["random"]["ndcg@100"] < 0.01,
          f"{summary['random']['ndcg@100']:.6f}")

    # ⭐ 断言 6：所有指标值都在 [0,1]
    all_ok = all(0.0 <= v <= 1.0 for tag in results for vs in results[tag].values()
                 for v in vs)
    check("所有指标 ∈ [0,1]", all_ok, f"共 {sum(len(v) for t in results.values() for v in t.values())} 个值")

    out = PROC / f"metrics_verify_{args.split}.json"
    out.write_text(json.dumps({
        "split": args.split,
        "n_queries": len(queries),
        "corpus": len(corpus_ids),
        "positive_refs_missing": missing,
        "summary": summary,
        "all_checks_passed": ok,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"  报告 → {out.relative_to(ROOT)}")
    print("=" * 70)
    print("  ⭐ " + ("全部检查通过 —— 指标实现可信" if ok else "存在失败项，必须修复"))
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())