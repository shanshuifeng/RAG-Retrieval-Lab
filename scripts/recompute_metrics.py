"""重算已有 run 文件的指标（指标代码修复后使用）

⭐ 为什么需要它：2026-10-24 发现 mrr_at_k 有生成器 bug，已完成的 BM25 结果里
   MRR 三项是错的。但【检索结果本身没问题】（run 文件存的是排序，与指标无关），
   所以只要用修复后的指标重算即可，无需重跑 21 分钟检索。

用法：
  .venv\\Scripts\\python.exe scripts/recompute_metrics.py
  .venv\\Scripts\\python.exe scripts/recompute_metrics.py --run <run.json> --split dev
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_split  # noqa: E402
from src.eval.metrics import evaluate, format_table  # noqa: E402

RESULTS = ROOT / "results"


def main() -> int:
    ap = argparse.ArgumentParser(description="用修复后的指标重算 run 文件")
    ap.add_argument("--run", default=None, help="run 文件路径（默认自动找最新）")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--ks", default="1,5,10,20,100")
    ap.add_argument("--old", default=None, help="旧结果 JSON（用于对比差异）")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]

    if args.run:
        run_path = Path(args.run)
    else:
        cands = sorted(RESULTS.glob(f"*_E1_bm25_{args.split}_run.json"))
        if not cands:
            print(f"❌ 找不到 {args.split} 的 run 文件")
            return 1
        run_path = cands[-1]

    print("=" * 74)
    print("  重算指标（修复后）")
    print("=" * 74)
    print(f"  run 文件：{run_path.relative_to(ROOT)}")

    run = json.loads(run_path.read_text(encoding="utf-8"))
    split = load_split(args.split)
    queries = {q: split.relevant_of(q) for q in split.query_list()}
    print(f"  语料 {split.n_passages:,} | query {len(queries):,} | run {len(run):,}")

    res = evaluate(queries, run, ks=ks, metrics=("recall", "mrr", "ndcg"))
    new = {k: round(v["mean"], 6) for k, v in res["aggregate"].items()}

    print()
    print(format_table(res["aggregate"], ks, ("recall", "mrr", "ndcg")))

    # 与旧结果对比（重点看 MRR 变化）
    old_path = Path(args.old) if args.old else None
    if old_path is None:
        cands = sorted(RESULTS.glob(f"*_E1_bm25_{args.split}.json"))
        old_path = cands[-1] if cands else None

    if old_path and old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8")).get("results", {})
        print()
        print("  与旧结果的差异（只列有变化的）：")
        print(f"  {'指标':<12} {'旧值':>10} {'新值':>10} {'变化':>10}")
        print("  " + "-" * 46)
        changed = 0
        for k in sorted(new):
            o = old.get(k)
            if o is None:
                continue
            if abs(o - new[k]) > 1e-6:
                changed += 1
                print(f"  {k:<12} {o:>10.6f} {new[k]:>10.6f} "
                      f"{new[k]-o:>+10.6f}")
        if not changed:
            print("  （无变化）")
        print(f"\n  共 {changed} 项指标变化")

    # 写出新结果
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = RESULTS / f"{stamp}_E1_bm25_{args.split}_recomputed.json"
    payload = {
        "experiment_id": "E1",
        "method": "bm25",
        "split": args.split,
        "date": datetime.now().isoformat(timespec="seconds"),
        "note": "⭐ 2026-10-24 修复 mrr_at_k 生成器 bug 后用原 run 重算",
        "source_run": str(run_path.name),
        "results": new,
        "filtered": res["filtered"],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  新结果 → {out.relative_to(ROOT)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())