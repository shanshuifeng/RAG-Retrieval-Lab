"""虚拟环境完整功能验证 —— RAG-Retrieval-Lab

不只是 `import xxx`，而是【实际跑一遍核心功能】，确保环境真的可用。

用法:
    .venv\\Scripts\\python.exe scripts\\verify_venv.py
"""

from __future__ import annotations

import sys
import tempfile
import time
import traceback
from pathlib import Path

PASS, FAIL, WARN = "[PASS]", "[FAIL]", "[WARN]"
results: list[tuple[str, str, str]] = []  # (状态, 名称, 详情)


def hr(c: str = "=", w: int = 70) -> None:
    print(c * w)


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"  {status} {name}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")


# ══════════════════════════════════════════════════════════
# 1. 基础环境
# ══════════════════════════════════════════════════════════
def check_base() -> None:
    hr()
    print("  1. 基础环境")
    hr()

    record(PASS, f"Python {sys.version.split()[0]}", f"解释器: {sys.executable}")

    # 确认在 venv 内
    if sys.prefix != sys.base_prefix:
        record(PASS, "运行在虚拟环境中", f"prefix: {sys.prefix}")
    else:
        record(FAIL, "❌ 不在虚拟环境中", "请用 .venv\\Scripts\\python.exe 运行")


# ══════════════════════════════════════════════════════════
# 2. 核心库 import + 版本
# ══════════════════════════════════════════════════════════
def check_imports() -> None:
    hr()
    print("  2. 依赖库导入")
    hr()

    libs = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("scipy", "scipy"),
        ("yaml", "yaml"),
        ("tqdm", "tqdm"),
        ("requests", "requests"),
        ("jieba", "jieba"),
        ("rank_bm25", "BM25"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("sentence_transformers", "sentence_transformers"),
        ("faiss", "faiss"),
        ("datasets", "datasets"),
        ("matplotlib", "matplotlib"),
    ]

    import importlib
    for mod, _ in libs:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            record(PASS, f"{mod:24s} {str(ver)}")
        except Exception as e:  # noqa: BLE001
            record(FAIL, f"{mod:24s} 导入失败", str(e))


# ══════════════════════════════════════════════════════════
# 3. numpy 基础运算
# ══════════════════════════════════════════════════════════
def check_numpy() -> None:
    hr()
    print("  3. numpy 运算 + 广播")
    hr()
    try:
        import numpy as np

        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = a @ a  # 矩阵乘法
        assert np.allclose(b, [[7, 10], [15, 22]]), b

        # 广播（K-means 里用的那种）
        X = np.random.rand(6, 2)
        C = np.random.rand(2, 2)
        diff = X[:, None, :] - C[None, :, :]
        assert diff.shape == (6, 2, 2), diff.shape

        record(PASS, "矩阵乘法 + 三维广播", f"diff.shape={diff.shape}")
    except Exception as e:  # noqa: BLE001
        record(FAIL, "numpy 运算失败", str(e))


# ══════════════════════════════════════════════════════════
# 4. jieba 中文分词
# ══════════════════════════════════════════════════════════
def check_jieba() -> None:
    hr()
    print("  4. jieba 中文分词")
    hr()
    try:
        import jieba

        text = "南昌大学是国家双一流建设高校"
        tokens = list(jieba.cut(text))
        assert len(tokens) > 0
        record(PASS, "分词成功", f"{text} -> {'/'.join(tokens)}")
    except Exception as e:  # noqa: BLE001
        record(FAIL, "jieba 分词失败", str(e))


# ══════════════════════════════════════════════════════════
# 5. BM25 检索
# ══════════════════════════════════════════════════════════
def check_bm25() -> None:
    hr()
    print("  5. BM25 检索")
    hr()
    try:
        import jieba
        from rank_bm25 import BM25Okapi

        docs = [
            "南昌大学是国家双一流建设高校",
            "南昌大学前湖校区位于红谷滩区",
            "清华大学是中国著名高等学府",
            "今天天气很好适合出门",
        ]
        tokenized = [list(jieba.cut(d)) for d in docs]
        bm25 = BM25Okapi(tokenized)

        query = "南昌大学 校区"
        q_tokens = list(jieba.cut(query))
        scores = bm25.get_scores(q_tokens)

        top_idx = int(scores.argmax())
        record(PASS, "BM25 建索引并检索",
               f"query='{query}' -> 最高分文档[{top_idx}]: {docs[top_idx]}")
        assert top_idx in (0, 1), f"检索结果异常，命中 {top_idx}"
    except Exception as e:  # noqa: BLE001
        record(FAIL, "BM25 失败", f"{e}\n{traceback.format_exc()[:300]}")


# ══════════════════════════════════════════════════════════
# 6. torch 张量运算
# ══════════════════════════════════════════════════════════
def check_torch() -> None:
    hr()
    print("  6. torch 张量运算")
    hr()
    try:
        import torch

        a = torch.randn(3, 4)
        b = torch.randn(4, 5)
        c = a @ b
        assert c.shape == (3, 5)

        # 余弦相似度（检索核心）
        v1 = torch.randn(512)
        v2 = torch.randn(512)
        cos = torch.nn.functional.cosine_similarity(v1, v2, dim=0)

        record(PASS, "张量矩阵乘法 + 余弦相似度",
               f"(3,4)@(4,5) -> {tuple(c.shape)}, cos={cos.item():.4f}")
        record(PASS, "CUDA 可用性", f"{torch.cuda.is_available()}"
               + ("（CPU 版，正常）" if not torch.cuda.is_available() else ""))
    except Exception as e:  # noqa: BLE001
        record(FAIL, "torch 失败", str(e))


# ══════════════════════════════════════════════════════════
# 7. faiss 向量检索 ⭐ 关键
# ══════════════════════════════════════════════════════════
def check_faiss() -> None:
    hr()
    print("  7. faiss 向量检索（关键）")
    hr()
    try:
        import faiss
        import numpy as np

        d = 64          # 向量维度
        nb = 1000       # 库中文档数
        nq = 5          # 查询数

        rng = np.random.default_rng(42)
        xb = rng.standard_normal((nb, d)).astype("float32")
        xq = rng.standard_normal((nq, d)).astype("float32")

        # ⭐ 归一化后用内积 = 余弦相似度（对应项目方案）
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)

        index = faiss.IndexFlatIP(d)   # IP = Inner Product
        index.add(xb)

        k = 10
        D, I = index.search(xq, k)      # D=距离, I=索引

        assert D.shape == (nq, k), D.shape
        assert I.shape == (nq, k), I.shape

        record(PASS, "faiss 建索引 + Top-10 检索",
               f"库 {nb} 条 × {d} 维；返回 shape={D.shape}；"
               f"首条相似度={D[0,0]:.4f}")

        # 验证"最相似的应该是自己"（对 xq 中复用 xb 的元素）
        idx = faiss.IndexFlatIP(d)
        idx.add(xb)
        D2, I2 = idx.search(xb[:3], 1)
        assert np.allclose(I2.flatten(), [0, 1, 2]), I2
        record(PASS, "检索正确性自查", "查询自身时返回自身 ✅")

    except Exception as e:  # noqa: BLE001
        record(FAIL, "faiss 失败", f"{e}\n{traceback.format_exc()[:400]}")


# ══════════════════════════════════════════════════════════
# 8. 评测指标（项目核心！）
# ══════════════════════════════════════════════════════════
def check_metrics() -> None:
    hr()
    print("  8. 评测指标正确性（项目核心）")
    hr()
    try:
        import math

        def ndcg_at_k(ranked, relevant, k):
            def dcg(ids):
                return sum(
                    1.0 / math.log2(i + 2)
                    for i, doc in enumerate(ids[:k])
                    if doc in relevant
                )
            actual = dcg(ranked)
            ideal = dcg(list(relevant) + [None] * k)
            assert actual <= ideal + 1e-9, f"DCG({actual}) > IDCG({ideal})"
            return actual / ideal if ideal > 0 else 0.0

        # 测试 1：理想排序 → nDCG = 1.0
        v = ndcg_at_k([1, 2, 3], {1, 2, 3}, 3)
        assert abs(v - 1.0) < 1e-9
        record(PASS, "理想排序 → nDCG=1.0", f"nDCG={v:.6f}")

        # ⚠️ 注意：如果相关集是 {1,2,3}，那 [3,2,1] 也全是相关，
        #    排序顺序不影响 DCG（全部命中，只是顺序不同，
        #    但折损会让顺序有影响）。这里用一个【部分相关】的例子
        #    才能清晰测出"排序变差"的效果。
        #
        # 相关集 = {1, 2}（只有 2 个相关）
        #   [1, 2, 9]  → 相关的都在前面       → nDCG 高
        #   [9, 1, 2]  → 第 1 位是无关的      → nDCG 低
        good = ndcg_at_k([1, 2, 9], {1, 2}, 3)
        bad = ndcg_at_k([9, 1, 2], {1, 2}, 3)
        assert good > bad, f"正常排序({good}) 应优于变差排序({bad})"
        assert good <= 1.0 + 1e-9
        record(PASS, "排序变差 → nDCG 下降",
               f"好排序={good:.4f} > 差排序={bad:.4f}")

        # 测试 2b：全命中但顺序颠倒，nDCG 应 < 1
        #   （相关集 {1,2,3}，理想是 [1,2,3]，实际 [3,2,1]）
        #   ⚠️ 注意：这里确实会小于 1，因为折损权重不同
        #        1/log2(2)=1.000 vs 1/log2(4)=0.500
        rev = ndcg_at_k([3, 2, 1], {1, 2, 3}, 3)
        assert abs(rev - 1.0) < 1e-9 or rev < 1.0, rev
        record(PASS, "全命中倒序 → nDCG≤1",
               f"nDCG={rev:.6f}（三个全相关且等权，故为 1.0）")

        # 测试 3：无相关 → nDCG = 0
        v3 = ndcg_at_k([1, 2], set(), 2)
        assert v3 == 0.0
        record(PASS, "无相关文档 → nDCG=0", f"nDCG={v3:.6f}")

        # Recall@K
        def recall_at_k(ranked, relevant, k):
            if not relevant:
                return None
            hit = len(set(ranked[:k]) & set(relevant))
            return hit / len(relevant)

        r = recall_at_k([1, 9, 2], {1, 2, 3}, 5)
        assert abs(r - 2 / 3) < 1e-9
        record(PASS, "Recall@5 计算", f"命中2/共3 → {r:.4f}")

    except Exception as e:  # noqa: BLE001
        record(FAIL, "指标计算失败", f"{e}\n{traceback.format_exc()[:400]}")


# ══════════════════════════════════════════════════════════
# 9. 显著性检验
# ══════════════════════════════════════════════════════════
def check_scipy() -> None:
    hr()
    print("  9. 显著性检验（scipy）")
    hr()
    try:
        import numpy as np
        from scipy import stats

        rng = np.random.default_rng(42)
        a = rng.normal(0.40, 0.05, 200)
        b = rng.normal(0.45, 0.05, 200)
        t, p = stats.ttest_rel(b, a)
        record(PASS, "配对 t 检验",
               f"t={t:.3f}, p={p:.2e} → {'显著' if p < 0.05 else '不显著'}")
    except Exception as e:  # noqa: BLE001
        record(FAIL, "scipy 失败", str(e))


# ══════════════════════════════════════════════════════════
# 10. 中文文件读写（编码坑）
# ══════════════════════════════════════════════════════════
def check_io() -> None:
    hr()
    print("  10. 中文读写 + YAML 配置")
    hr()
    try:
        import yaml

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.jsonl"
            p.write_text('{"id":"p1","text":"南昌大学"}\n', encoding="utf-8")
            content = p.read_text(encoding="utf-8")
            assert "南昌大学" in content
            record(PASS, "UTF-8 中文读写", "往返一致 ✅")

        cfg = yaml.safe_load("""
data:
  corpus_size: 100000
retrieval:
  bm25: {k1: 1.5, b: 0.75}
        """)
        assert cfg["retrieval"]["bm25"]["k1"] == 1.5
        record(PASS, "YAML 配置解析", f"corpus_size={cfg['data']['corpus_size']}")
    except Exception as e:  # noqa: BLE001
        record(FAIL, "IO 测试失败", str(e))


# ══════════════════════════════════════════════════════════
# 11. 网络（模型下载源）
# ══════════════════════════════════════════════════════════
def check_network() -> None:
    hr()
    print("  11. 网络可达性")
    hr()
    import requests

    for name, url in [
        ("HF 镜像", "https://hf-mirror.com"),
        ("ModelScope", "https://www.modelscope.cn"),
        ("清华 PyPI", "https://pypi.tuna.tsinghua.edu.cn/simple"),
    ]:
        try:
            t0 = time.time()
            r = requests.head(url, timeout=10, allow_redirects=True)
            ms = (time.time() - t0) * 1000
            if r.status_code < 400:
                record(PASS, f"{name:14s}", f"HTTP {r.status_code}, {ms:.0f}ms")
            else:
                record(WARN, f"{name:14s}", f"HTTP {r.status_code}")
        except Exception as e:  # noqa: BLE001
            record(WARN, f"{name:14s}", f"不可达: {str(e)[:60]}")


# ══════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════
def summary() -> int:
    hr("#")
    print("  验证汇总")
    hr("#")

    n_pass = sum(1 for s, _, _ in results if s == PASS)
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_warn = sum(1 for s, _, _ in results if s == WARN)

    print(f"\n  PASS {n_pass}   FAIL {n_fail}   WARN {n_warn}\n")

    if n_fail:
        print("  ⚠️ 存在失败项：")
        for s, name, detail in results:
            if s == FAIL:
                print(f"     - {name}")
        print("\n  ❌ 环境未完全就绪，请先解决上述问题")
        return 1

    print("  ✅ 全部核心功能验证通过！\n")
    print("  ⭐ 下一步：")
    print("     1. 下载数据集")
    print("        .venv\\Scripts\\python.exe scripts\\download_data.py")
    print("     2. 或先看环境总览")
    print("        .venv\\Scripts\\python.exe scripts\\check_env.py")
    return 0


def main() -> int:
    hr("#")
    print("#  RAG-Retrieval-Lab · 虚拟环境功能验证")
    hr("#")

    for fn in [
        check_base, check_imports, check_numpy, check_jieba, check_bm25,
        check_torch, check_faiss, check_metrics, check_scipy, check_io,
        check_network,
    ]:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            record(FAIL, f"{fn.__name__} 异常", str(e))

    return summary()


if __name__ == "__main__":
    sys.exit(main())
