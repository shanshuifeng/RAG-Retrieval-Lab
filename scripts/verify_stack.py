"""栈验证脚本 —— 确认检索链路依赖全部可用。

用法:
    .venv\\Scripts\\python.exe scripts\\verify_stack.py

⚠️ 只验证导入与基本运行，不下载模型（模型下载请用 --model）。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# 让 src 包可导入（本文件在 scripts/ 下，向上一级是项目根）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, FAIL = "[OK]  ", "[FAIL]"


def check(name: str, fn) -> bool:
    t = time.time()
    try:
        info = fn()
        print(f"  {OK} {name:24s} {info}  ({time.time() - t:.1f}s)")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  {FAIL} {name:24s} {type(e).__name__}: {e}")
        return False


def main() -> int:
    print("\n  检索栈验证（venv）\n" + "  " + "-" * 60)

    results = []

    def _torch():
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.randn(64, 512, device=dev)
        y = x @ x.T
        return f"{torch.__version__} device={dev} matmul={tuple(y.shape)}"

    def _st():
        import sentence_transformers as st
        return f"sentence-transformers {st.__version__}"

    def _tf():
        import transformers
        return f"transformers {transformers.__version__}"

    def _faiss():
        import numpy as np
        import faiss
        idx = faiss.IndexFlatIP(64)
        v = np.random.randn(1000, 64).astype("float32")
        idx.add(v)
        _, ids = idx.search(v[:5], 10)
        return f"faiss {faiss.__version__} ntotal={idx.ntotal} top1={ids[0][0]}"

    def _bm25():
        from rank_bm25 import BM25Okapi
        import jieba
        corpus = [list(jieba.cut("中文检索增强生成实验台")) for _ in range(5)]
        bm = BM25Okapi(corpus)
        return "jieba+BM25 " + str(bm.get_scores(list(jieba.cut("检索")))[:2])

    def _proj():
        from src.retrieval.bm25 import BM25Retriever
        from src.eval.metrics import evaluate
        from src.data.loader import load_split
        return "src.* 可导入（bm25 / metrics / loader）"

    results.append(check("torch + CUDA", _torch))
    results.append(check("sentence-transformers", _st))
    results.append(check("transformers", _tf))
    results.append(check("faiss 向量检索", _faiss))
    results.append(check("jieba + BM25", _bm25))
    results.append(check("项目 src 模块", _proj))

    if "--model" in sys.argv:
        def _model():
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer("BAAI/bge-small-zh-v1.5")
            v = m.encode(["中文检索实验"], normalize_embeddings=True)
            return f"bge-small-zh-v1.5 dim={v.shape}"
        results.append(check("嵌入模型（需下载）", _model))

    print("  " + "-" * 60)
    n_ok = sum(results)
    print(f"\n  {n_ok}/{len(results)} 项通过\n")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
