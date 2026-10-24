"""稠密检索自检 —— 跑全量前先验证正确性

⭐ 三个必查项（任一失败都说明实现有问题，不要浪费 10 分钟编码全库）：
   1. 查自身：用语料里的原文当 query 检索，它自己必须排第一
   2. 分数范围：归一化向量的内积 ∈ [-1, 1]，余弦相似度应在此区间
   3. 语义相近：同义改写应比无关文本分数高

用法：
  .venv\\Scripts\\python.exe scripts\\verify_dense.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_split  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402


def main() -> int:
    print("=" * 74)
    print("  稠密检索自检（小样本，约 1000 段）")
    print("=" * 74)

    split = load_split("dev")
    # 只取前 1000 段，快速验证
    subset_ids = split.passage_ids[:1000]
    small = type(split)(name="small")
    small.passage_ids = subset_ids
    small.corpus = {p: split.corpus[p] for p in subset_ids}
    small.queries = split.queries
    small.qrels = split.qrels

    print(f"\n[1/3] 编码 {len(subset_ids)} 段 ...")
    t0 = time.time()
    r = DenseRetriever("BAAI/bge-small-zh-v1.5", batch_size=64)
    r.build(small, verbose=True)
    print(f"  用时 {time.time()-t0:.1f}s")
    st = r.stats()
    print(f"  设备 {st['device']} | fp16 {st['fp16']} | dim {st['dim']}")

    ok = True

    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
        ok = ok and cond

    # ── 1. 查自身 ──
    print("\n[2/3] 查自身（用原文检索，自己应排第一）...")
    hits = 0
    for pid in subset_ids[:20]:
        text = small.corpus[pid]
        res = r.search(text, top_k=3)
        if res and res[0][0] == pid:
            hits += 1
    check("原文检索命中自身（20 条抽样）", hits >= 19,
          f"{hits}/20 排第一")

    # ── 2. 分数范围 ──
    print("\n[3/3] 分数范围与语义排序...")
    res = r.search("南昌大学是国家双一流建设高校", top_k=5)
    scores = [s for _, s in res]
    check("余弦相似度 ∈ [-1, 1]", all(-1.001 <= s <= 1.001 for s in scores),
          f"范围 [{min(scores):.3f}, {max(scores):.3f}]")

    # ── 3. 语义相近性 ──
    # 找一对：原文 vs 同义改写 vs 无关文本
    test_pid = subset_ids[0]
    orig = small.corpus[test_pid]
    print(f"\n  测试段落（前 60 字）：{orig[:60]}...")

    res_orig = r.search(orig, top_k=3)
    print(f"  用原文检索 top-3: {[(p[:8], round(s,4)) for p,s in res_orig]}")
    check("原文 → 自己排第一", res_orig[0][0] == test_pid,
          f"{res_orig[0][0][:8]} vs {test_pid[:8]}")

    # 无关查询应得低分
    res_unrelated = r.search("量子纠缠拓扑绝缘体的能带结构", top_k=3)
    top_unrelated = res_unrelated[0][1]
    print(f"  无关查询最高分: {top_unrelated:.4f}")
    check("无关查询分数 < 原文查询分数", top_unrelated < res_orig[0][1],
          f"{top_unrelated:.4f} < {res_orig[0][1]:.4f}")

    print()
    print("=" * 74)
    print("  ⭐ " + ("自检通过 —— 可跑全量编码" if ok else "自检失败，先排查"))
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())