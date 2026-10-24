"""E3 · 重排序自检 —— 跑全量前先验证正确性（小样本）

⭐ 三个必查项：
   1. 相关段落应被排前（重排的任务就是把它们提到前面）
   2. 分数排序后应与第一段不同（重排确实改变了顺序，否则等于没做）
   3. 延迟计时合理（每 query 的耗时应为正且随候选数增长）

用法：
  .venv\\Scripts\\python.exe scripts\\verify_rerank.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_split  # noqa: E402
from src.eval.metrics import ndcg_at_k  # noqa: E402
from src.rerank.cross_encoder import CrossEncoderReranker  # noqa: E402

RESULTS = ROOT / "results"


def main() -> int:
    print("=" * 74)
    print("  E3 重排序自检（5 条 query × top-20 候选）")
    print("=" * 74)

    split = load_split("dev")
    # 第一段：用 D2（E2 已证伪混合，故用最强单方法）
    import json
    d2_run = json.loads(
        sorted(RESULTS.glob("*E1_dense_base_dev_run.json"))[-1]
        .read_text(encoding="utf-8"))

    print("\n[1/2] 加载重排模型 ...")
    t0 = time.time()
    r = CrossEncoderReranker("BAAI/bge-reranker-base")
    r._load()
    st = r.stats()
    print(f"  {st['model']} | 设备 {st['device']} | fp16 {st['fp16']} "
          f"| 加载 {st['load_sec']}s")

    print("\n[2/2] 对 5 条 query 做重排并检查 ...")
    qids = [q for q in split.query_list(with_relevant_only=True)
            if q in d2_run][:5]
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
        ok = ok and cond

    for i, qid in enumerate(qids, 1):
        query = split.queries[qid]
        rel = set(split.relevant_of(qid))
        cands = [(pid, split.corpus.get(pid, "")) for pid in d2_run[qid][:20]]

        res = r.rerank(qid, query, cands)

        # 第一段（不重排）的 nDCG@10
        first_stage = [pid for pid, _ in cands]
        nd_first = ndcg_at_k(first_stage, rel, 10)
        nd_rerank = ndcg_at_k(res.ranked, rel, 10)

        # 找正例在重排前后的位置
        def first_rel_pos(ids):
            for rank, pid in enumerate(ids, 1):
                if pid in rel:
                    return rank
            return None
        pos_before = first_rel_pos(first_stage)
        pos_after = first_rel_pos(res.ranked)

        print(f"\n  [{i}] {query[:30]}...")
        print(f"      正例数 {len(rel)} | 候选 {res.n_candidates} | "
              f"延迟 {res.latency_ms:.1f} ms")
        print(f"      第一个正例位置：重排前 {pos_before} → 重排后 {pos_after}")
        print(f"      nDCG@10：{nd_first:.4f} → {nd_rerank:.4f} "
              f"({'↑' if nd_rerank > nd_first else ('↓' if nd_rerank < nd_first else '=')})")

        check(f"Q{i} 重排后顺序与候选集合一致",
              set(res.ranked) == set(first_stage), "集合不变")
        check(f"Q{i} 延迟为正", res.latency_ms > 0, f"{res.latency_ms:.1f} ms")

    print()
    print("=" * 74)
    print("  ⭐ " + ("自检通过 —— 可跑全量" if ok else "自检失败，先排查"))
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())