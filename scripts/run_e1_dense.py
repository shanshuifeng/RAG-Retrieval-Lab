"""E1 · 稠密检索（D1/D2）—— P2

⭐ 与 BM25 用完全相同的数据、K 值、指标口径，只改检索方法（一次一个变量）。

用法：
  .venv\\Scripts\\python.exe scripts\\run_e1_dense.py --model small
  .venv\\Scripts\\python.exe scripts\\run_e1_dense.py --model base
  .venv\\Scripts\\python.exe scripts\\run_e1_dense.py --model small --split test --confirm-final
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
from src.eval.metrics import evaluate, format_table  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "index"

MODELS = {
    "small": "BAAI/bge-small-zh-v1.5",
    "base": "BAAI/bge-base-zh-v1.5",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 稠密检索")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--model", default="small", choices=list(MODELS))
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--ks", default="1,5,10,20,100")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--no-instruction", action="store_true",
                    help="强制关闭 BGE query 指令前缀（默认已关闭，见 dense.py 注释）")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--confirm-final", action="store_true")
    args = ap.parse_args()

    if args.split == "test" and not args.confirm_final:
        print("❌ 拒绝在 test 上运行：test 集只该在配置锁定后跑一次。")
        print("   确认最终评测请加 --confirm-final")
        return 2

    ks = [int(x) for x in args.ks.split(",")]
    model_id = MODELS[args.model]
    print("=" * 78)
    print(f"  E1 · 稠密检索 D{'1' if args.model == 'small' else '2'}  |  "
          f"split={args.split}  model={model_id}")
    print("=" * 78)

    t_all = time.time()

    print("\n[1/4] 加载数据 ...")
    split = load_split(args.split)
    print(f"  语料 {split.n_passages:,} 段 | 查询 {split.n_queries:,} 条")

    print("\n[2/4] 编码 + 建索引 ...")
    # ⭐ 缓存键必须包含「影响检索结果」的配置，否则改了默认值仍会命中旧缓存
    #    （2026-10-24 踩过：改了 query_instruction 默认值，结果没变）
    from src.retrieval.dense import DEFAULT_QUERY_INSTRUCTION
    instr = "" if args.no_instruction else DEFAULT_QUERY_INSTRUCTION
    instr_tag = "noinstr" if not instr else "instr"
    tag = f"dense_{args.model}_{args.split}_{instr_tag}"
    idx_path = CACHE / f"{tag}.pkl"
    retriever = None
    if idx_path.exists() and not args.no_cache:
        print(f"  命中缓存 {idx_path.name}")
        try:
            retriever = DenseRetriever.load(idx_path)
            cached_instr = retriever.query_instruction
            if cached_instr != instr:
                print(f"  ⚠️ 缓存配置不符（缓存={cached_instr!r} 期望={instr!r}），重建")
                retriever = None
            else:
                print(f"  已载入 {retriever._index.ntotal:,} 条向量 "
                      f"（指令={instr!r}）")
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ 缓存载入失败（{type(e).__name__}），重建")
            retriever = None

    if retriever is None:
        retriever = DenseRetriever(
            model_id, batch_size=args.batch_size, use_fp16=not args.no_fp16,
            query_instruction=instr,
        )
        retriever.build(split, verbose=True)
        retriever.save(idx_path)
        print(f"  索引已缓存 → {idx_path.relative_to(ROOT)}")

    st = retriever.stats()
    print(f"  设备 {st['device']} | fp16 {st['fp16']} | dim {st['dim']} "
          f"| 向量 {st['n_vectors']:,}")

    print("\n[3/4] 批量检索 ...")
    t_ret = time.time()
    run = retriever.run_all(split, top_k=args.top_k, verbose=True)
    ret_sec = time.time() - t_ret

    print("\n[4/4] 评测 ...")
    queries = {q: split.relevant_of(q) for q in split.query_list()}
    res = evaluate(queries, run, ks=ks, metrics=("recall", "mrr", "ndcg"))
    print(f"  评测 query 数：{res['filtered']['queries_evaluated']:,}")
    print()
    print(format_table(res["aggregate"], ks, ("recall", "mrr", "ndcg")))

    n_q = res["filtered"]["queries_evaluated"]
    cost = {
        "encode_sec": round(retriever._encode_sec, 2),
        "build_sec_total": round(retriever._build_sec, 2),
        "search_sec_total": round(ret_sec, 2),
        "search_ms_per_query": round(1000 * ret_sec / max(n_q, 1), 3),
        "device": st["device"],
        "fp16": st["fp16"],
        "dim": st["dim"],
    }
    print(f"\n  成本：编码 {cost['encode_sec']}s | "
          f"检索 {cost['search_ms_per_query']:.2f} ms/query")

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = RESULTS / f"{stamp}_E1_dense_{args.model}_{args.split}.json"
    payload = {
        "experiment_id": "E1",
        "method": f"dense:{model_id}",
        "group": f"D{'1' if args.model == 'small' else '2'}",
        "split": args.split,
        "date": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "corpus_size": split.n_passages,
            "queries": res["filtered"]["queries_evaluated"],
            "model": model_id,
            "dim": st["dim"],
            "max_length": st["max_length"],
            "normalize": True,
            "index": "faiss.IndexFlatIP",
            "query_instruction": retriever.query_instruction,
            "top_k": args.top_k,
            "ks": ks,
        },
        "results": {k: round(v["mean"], 6) for k, v in res["aggregate"].items()},
        "cost": cost,
        "filtered": res["filtered"],
        "total_sec": round(time.time() - t_all, 1),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    run_out = RESULTS / f"{stamp}_E1_dense_{args.model}_{args.split}_run.json"
    run_out.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")

    print(f"\n  结果 → {out.relative_to(ROOT)}")
    print(f"  总用时 {payload['total_sec']}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())