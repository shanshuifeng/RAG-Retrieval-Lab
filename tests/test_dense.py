"""稠密检索器测试

⚠️ 本测试【不加载真实 BGE 模型】（太慢，且依赖网络/GPU）。
   用 mock 替换编码器，只验证：
     · 索引与检索的接口契约（归一化、Top-K、排序）
     · 持久化往返
     · 错误处理
   真实模型的行为由 scripts/verify_dense.py 做端到端验证。

运行：
    .venv\\Scripts\\python.exe -m unittest tests.test_dense -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.data.loader import Split
from src.retrieval.dense import DenseRetriever


def make_split(corpus: dict[str, str], queries: dict[str, str],
               qrels: dict[str, list[str]]) -> Split:
    s = Split(name="toy")
    s.corpus = dict(corpus)
    s.passage_ids = list(corpus.keys())
    s.queries = dict(queries)
    s.qrels = dict(qrels)
    return s


class FakeEncoder:
    """确定性假编码器：用字符 hash 生成向量，语义无关但可复现。

    ⭐ 目的：让检索结果【可预测】——同一 token 集合的文本得到相近向量。
    """

    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode_vectors(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for ch in t:
                out[i, ord(ch) % self.dim] += 1.0
        # L2 归一化（与真实实现一致）
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype(np.float32)


class TestDenseContract(unittest.TestCase):
    def _make_retriever(self, dim: int = 8) -> DenseRetriever:
        r = DenseRetriever(query_instruction="")   # 关掉指令，便于预测
        fake = FakeEncoder(dim)

        # 替换 _load_model 与 _encode
        r._dim = dim
        r._model = object()      # 非 None 即可
        r._encode = lambda texts, desc="": fake.encode_vectors(texts)  # type: ignore
        return r

    def setUp(self):
        self.split = make_split(
            corpus={
                "p1": "abc",
                "p2": "abd",
                "p3": "xyz",
            },
            queries={"q1": "abc"},
            qrels={"q1": ["p1"]},
        )

    def test_build_and_search(self):
        r = self._make_retriever()
        r.build(self.split, verbose=False)
        self.assertEqual(r._index.ntotal, 3)
        res = r.search("abc", top_k=3)
        self.assertEqual(len(res), 3)
        # 与 "abc" 完全相同 → 应排第一
        self.assertEqual(res[0][0], "p1")

    def test_scores_descending(self):
        r = self._make_retriever()
        r.build(self.split, verbose=False)
        scores = [s for _, s in r.search("abc", top_k=3)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_scores_in_cosine_range(self):
        # ⭐ 归一化后内积必须 ∈ [-1, 1]
        r = self._make_retriever()
        r.build(self.split, verbose=False)
        for _, s in r.search("abc", top_k=3):
            self.assertGreaterEqual(s, -1.001)
            self.assertLessEqual(s, 1.001)

    def test_top_k_truncation(self):
        r = self._make_retriever()
        r.build(self.split, verbose=False)
        self.assertEqual(len(r.search("abc", top_k=2)), 2)

    def test_top_k_larger_than_corpus(self):
        r = self._make_retriever()
        r.build(self.split, verbose=False)
        self.assertEqual(len(r.search("abc", top_k=100)), 3)

    def test_search_before_build_raises(self):
        r = self._make_retriever()
        with self.assertRaises(RuntimeError):
            r.search("abc")

    def test_run_all(self):
        r = self._make_retriever()
        r.build(self.split, verbose=False)
        run = r.run_all(self.split, top_k=2, verbose=False)
        self.assertEqual(set(run), {"q1"})
        self.assertLessEqual(len(run["q1"]), 2)

    def test_search_batch_matches_single(self):
        r = self._make_retriever()
        r.build(self.split, verbose=False)
        single = [p for p, _ in r.search("abc", top_k=3)]
        batch = [[p for p, _ in hits]
                 for hits in r.search_batch(["abc"], top_k=3)]
        self.assertEqual(single, batch[0])

    def test_stats(self):
        r = self._make_retriever()
        r.build(self.split, verbose=False)
        st = r.stats()
        self.assertEqual(st["method"], "dense")
        self.assertEqual(st["n_passages"], 3)
        self.assertEqual(st["n_vectors"], 3)
        self.assertEqual(st["dim"], 8)


class TestDensePersistence(unittest.TestCase):
    def test_save_load_roundtrip(self):
        split = make_split(
            corpus={"p1": "abc", "p2": "xyz"},
            queries={"q1": "abc"},
            qrels={"q1": ["p1"]},
        )
        fake = FakeEncoder(8)
        r = DenseRetriever(query_instruction="")
        r._dim = 8
        r._model = object()
        r._encode = lambda texts, desc="": fake.encode_vectors(texts)  # type: ignore
        r.build(split, verbose=False)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dense.pkl"
            r.save(p)
            self.assertTrue(p.exists())
            self.assertTrue(p.with_suffix(".faiss").exists())

            r2 = DenseRetriever.load(p)
            self.assertEqual(r2.passage_ids, r.passage_ids)
            self.assertEqual(r2._dim, r._dim)

            # ⚠️ load() 不带编码器（真实场景会按 model_name 重新加载模型）。
            #    测试里必须同样注入假编码器，否则维度不匹配。
            r2._model = object()
            r2._encode = lambda texts, desc="": fake.encode_vectors(texts)  # type: ignore

            self.assertEqual([x[0] for x in r2.search("abc", 2)],
                             [x[0] for x in r.search("abc", 2)])

    def test_load_preserves_config(self):
        """load 后配置项必须完整恢复（否则复现性丢失）。"""
        split = make_split(
            corpus={"p1": "a", "p2": "b"},
            queries={"q1": "a"},
            qrels={"q1": ["p1"]},
        )
        r = DenseRetriever("fake-model", batch_size=32, max_length=256,
                           use_fp16=False, query_instruction="Q:")
        fake = FakeEncoder(8)
        r._dim = 8
        r._model = object()
        r._encode = lambda texts, desc="": fake.encode_vectors(texts)  # type: ignore
        r.build(split, verbose=False)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.pkl"
            r.save(p)
            r2 = DenseRetriever.load(p)
            self.assertEqual(r2.model_name, "fake-model")
            self.assertEqual(r2.batch_size, 32)
            self.assertEqual(r2.max_length, 256)
            self.assertFalse(r2.use_fp16)
            self.assertEqual(r2.query_instruction, "Q:")


class TestQueryInstruction(unittest.TestCase):
    def test_instruction_prepended(self):
        """⭐ 确认 query 指令确实被加上了（影响效果一致性）。"""
        captured = {}

        def fake_encode(texts, desc=""):
            captured["texts"] = texts
            return np.zeros((len(texts), 4), dtype=np.float32)

        r = DenseRetriever(query_instruction="PREFIX:")
        r._dim = 4
        r._model = object()
        r._encode = fake_encode  # type: ignore
        import faiss
        r._index = faiss.IndexFlatIP(4)
        r._index.add(np.eye(4, dtype=np.float32))
        r.passage_ids = ["a", "b", "c", "d"]

        r.search("我的查询", top_k=2)
        self.assertEqual(captured["texts"], ["PREFIX:我的查询"])

    def test_empty_instruction(self):
        captured = {}

        def fake_encode(texts, desc=""):
            captured["texts"] = texts
            return np.zeros((len(texts), 4), dtype=np.float32)

        r = DenseRetriever(query_instruction="")
        r._dim = 4
        r._model = object()
        r._encode = fake_encode  # type: ignore
        import faiss
        r._index = faiss.IndexFlatIP(4)
        r._index.add(np.eye(4, dtype=np.float32))
        r.passage_ids = ["a", "b", "c", "d"]

        r.search("我的查询", top_k=2)
        self.assertEqual(captured["texts"], ["我的查询"])


if __name__ == "__main__":
    unittest.main(verbosity=2)