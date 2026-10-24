"""query 指令前缀的消融实验 —— 修正 D1/D2 的一个实现选择

⭐ 背景：BGE 官方文档建议检索时给 query 加指令前缀
      "为这个句子生成表示以用于检索相关文章："
   我在 dense.py 里【默认启用了】它。

⭐ 实测（2026-10-24，dev 全量 8,640 query）：
      带指令 nDCG@10 = 0.723848
      无指令 nDCG@10 = 0.733224
      差（无 - 带）= +0.009376（+1.30%），p = 8.0e-13，CI [+0.0069, +0.0120]
    → ⭐ 无指令【显著更好】，与"官方推荐加指令"的直觉相反

⭐ 说明：BGE v1.5 的模型卡明确写了「v1.5 版本即使不加指令也不会有明显性能下降」，
   本数据上不加反而更好。这是一个【必须如实报告】的负面结果——
   官方推荐的默认做法，在特定数据/评测口径下未必最优。

本脚本可复现上述结论。

用法：
  .venv\\Scripts\\python.exe scripts/ablate_query_instruction.py
  .venv\\Scripts\\python.exe scripts/ablate_query_instruction.py --model base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_split  # noqa: E402
from src.eval.metrics import ndcg_at_k, recall_at_k  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402

CACHE = ROOT / "data" / "index"
RESULTS = ROOT / "results"

BGE_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def eval_instr(r: DenseRetriever, split, qids: list[str], instr: str,
               batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
    r.query_instruction = instr
    nd, rc = [], []
    for s in range(0, len(qids), batch):
        chunk = qids[s:s + batch]
        texts = [split.queries[q] for q in chunk]
        hits_list = r.search_batch(texts, top_k=100)
        for qid, hits in zip(chunk, hits_list):
            ids = [p for p, _ in hits]
            rel = split.relevant_of(qid)
            nd.append(ndcg_at_k(ids, rel, 10))
            rc.append(recall_at_k(ids, rel, 10))
    return np.array(nd), np.array(rc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", choices=["small", "base"])
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    print("=" * 78)
    print(f"  query 指令前缀消融  |  model={args.model}  split={args.split}")
    print("=" * 78)

    split = load_split(args.split)
    qids = split.query_list(with_relevant_only=True)
    print(f"  query {len(qids):,}")

    idx = CACHE / f"dense_{args.model}_{args.split}.pkl"
    if not idx.exists():
        print(f"❌ 找不到索引缓存 {idx}，请先跑 run_e1_dense.py")
        return 1
    r = DenseRetriever.load(idx)

    print("\n[1/2] 带指令 ...")
    nd_a, rc_a = eval_instr(r, split, qids, BGE_INSTRUCTION)
    print("[2/2] 无指令 ...")
    nd_b, rc_b = eval_instr(r, split, qids, "")

    print()
    print(f"  {'配置':<10} {'nDCG@10':>10} {'Recall@10':>11}")
    print("  " + "-" * 34)
    print(f"  {'带指令':<10} {nd_a.mean():>10.6f} {rc_a.mean():>11.6f}")
    print(f"  {'无指令':<10} {nd_b.mean():>10.6f} {rc_b.mean():>11.6f}")

    # 配对检验（无指令 - 带指令）
    out = {"split": args.split, "model": args.model, "n_queries": len(qids)}
    print()
    for name, a, b in [("nDCG@10", nd_a, nd_b), ("Recall@10", rc_a, rc_b)]:
        diff = b - a
        t, p = stats.ttest_rel(b, a)
        d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0
        rng = np.random.default_rng(42)
        ii = rng.integers(0, len(diff), size=(10000, len(diff)))
        lo, hi = np.percentile(diff[ii].mean(axis=1), [2.5, 97.5])
        better = int((diff > 1e-12).sum())
        worse = int((diff < -1e-12).sum())
        out[name] = {
            "with_instruction": round(float(a.mean()), 6),
            "without_instruction": round(float(b.mean()), 6),
            "diff_without_minus_with": round(float(diff.mean()), 6),
            "rel_pct": round(100 * float(diff.mean()) / float(a.mean()), 2),
            "p_value": float(p),
            "cohens_d": round(float(d), 4),
            "ci95": [round(float(lo), 6), round(float(hi), 6)],
            "without_better": better,
            "with_better": worse,
        }
        print(f"  {name}: 无指令 {b.mean():.6f} vs 带指令 {a.mean():.6f}  "
              f"差 {diff.mean():+.6f} ({100*diff.mean()/a.mean():+.2f}%)")
        print(f"    p={p:.2e}  d={d:+.4f}  CI=[{lo:+.6f}, {hi:+.6f}]")
        print(f"    无指令更好 {better:,} 条 / 带指令更好 {worse:,} 条")

    # 判定
    nd = out["nDCG@10"]
    print()
    if nd["ci95"][0] > 0:
        verdict = ("无指令显著更好 —— 应关闭指令前缀（与 BGE 官方推荐相反，"
                   "但符合 v1.5 模型卡「不加指令也不明显下降」的说明）")
        print(f"  ⚠️ 判定：{verdict}")
    elif nd["ci95"][1] < 0:
        verdict = "带指令显著更好 —— 保持当前实现"
        print(f"  ✅ 判定：{verdict}")
    else:
        verdict = "两者差异不显著 —— 可任选，建议保持与官方一致（带指令）"
        print(f"  → 判定：{verdict}")
    out["verdict"] = verdict

    RESULTS.mkdir(parents=True, exist_ok=True)
    p_out = RESULTS / f"2026-10-24_ablation_query_instruction_{args.model}.json"
    p_out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结果 → {p_out.relative_to(ROOT)}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())