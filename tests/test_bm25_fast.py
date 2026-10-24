"""BM25Fast 单元测试 —— 与 rank_bm25 的等价性是核心验收点

运行：
    .venv\\Scripts\\python.exe -m unittest tests.test_bm25_fast -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.loader import Split
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.bm25_fast import BM25Fast


def make_split(corpus: dict[str, str], queries: dict[str, str],
               qrels: dict[str, list[str]]) -> Split:
    s = Split(name="toy")
    s.corpus = dict(corpus)
    s.passage_ids = list(corpus.keys())
    s.queries = dict(queries)
    s.qrels = dict(qrels)
    return s


class TestEquivalenceWithRankBM25(unittest.TestCase):
    """⭐ 核心验收：两套实现必须给出相同的排序与分数。

    这是 P1 提速能成立的前提——如果结果不同，提速毫无意义。
    """

    def setUp(self):
        self.split = make_split(
            corpus={
                "p1": "板蓝根颗粒的功效与作用是清热解毒",
                "p2": "紫薯的药理作用是抗氧化",
                "p3": "板蓝根颗粒可以治疗感冒发热",
                "p4": "感冒发热的症状与治疗方法",
                "p5": "清热解毒的中药有哪些",
            },
            queries={
                "q1": "板蓝根颗粒的功效",
                "q2": "紫薯抗氧化",
                "q3": "感冒发热怎么治疗",
            },
            qrels={"q1": ["p1", "p3"], "q2": ["p2"], "q3": ["p4"]},
        )
        self.fast = BM25Fast(k1=1.5, b=0.75)
        self.fast.build(self.split, verbose=False)
        self.slow = BM25Retriever(k1=1.5, b=0.75)
        self.slow.build(self.split, verbose=False)

    def test_same_ranking(self):
        for qid in self.split.query_list():
            q = self.split.queries[qid]
            r_fast = [p for p, _ in self.fast.search(q, top_k=5)]
            r_slow = [p for p, _ in self.slow.search(q, top_k=5)]
            self.assertEqual(r_fast, r_slow,
                             f"query {qid!r} 排序不一致：{r_fast} vs {r_slow}")

    def test_same_scores(self):
        for qid in self.split.query_list():
            q = self.split.queries[qid]
            d_fast = dict(self.fast.search(q, top_k=5))
            d_slow = dict(self.slow.search(q, top_k=5))
            for pid, sc in d_slow.items():
                self.assertAlmostEqual(d_fast[pid], sc, places=9,
                                       msg=f"query {qid} 的 {pid} 分数不一致")

    def test_params_affect_both_identically(self):
        # 换参数后两者仍应一致（证明公式参数化正确）
        fast2 = BM25Fast(k1=0.9, b=0.3)
        fast2.build(self.split, verbose=False)
        slow2 = BM25Retriever(k1=0.9, b=0.3)
        slow2.build(self.split, verbose=False)
        for qid in self.split.query_list():
            q = self.split.queries[qid]
            self.assertEqual([p for p, _ in fast2.search(q, 5)],
                             [p for p, _ in slow2.search(q, 5)])

    def test_negative_idf_floor(self):
        # 构造一个高频词（出现在过半文档），触发负 IDF → epsilon 下限
        corpus = {f"d{i}": "常见词 常见词 特有词" + str(i) for i in range(10)}
        corpus["d0"] = "常见词 常见词 稀有词"
        split = make_split(corpus, {"q1": "常见词"}, {"q1": ["d0"]})
        fast = BM25Fast(k1=1.5, b=0.75)
        fast.build(split, verbose=False)
        slow = BM25Retriever(k1=1.5, b=0.75)
        slow.build(split, verbose=False)
        self.assertEqual([p for p, _ in fast.search("常见词", 5)],
                         [p for p, _ in slow.search("常见词", 5)],
                         "负 IDF 情形下两者应一致（epsilon 下限需对齐）")


class TestBM25FastBasic(unittest.TestCase):
    def setUp(self):
        self.split = make_split(
            corpus={"p1": "机器学习", "p2": "深度学习", "p3": "机器翻译"},
            queries={"q1": "机器学习"},
            qrels={"q1": ["p1"]},
        )
        self.r = BM25Fast()
        self.r.build(self.split, verbose=False)

    def test_finds_relevant_first(self):
        res = self.r.search("机器学习", top_k=3)
        self.assertTrue(res)
        self.assertEqual(res[0][0], "p1")

    def test_scores_descending(self):
        scores = [s for _, s in self.r.search("机器学习", top_k=3)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_query(self):
        self.assertEqual(self.r.search("", top_k=5), [])

    def test_oov_query_all_zero(self):
        # 全部词都不在词表 → 返回空（而非崩溃）
        self.assertEqual(self.r.search("量子纠缠拓扑绝缘体", top_k=5), [])

    def test_top_k_truncation(self):
        self.assertLessEqual(len(self.r.search("机器", top_k=2)), 2)

    def test_search_before_build_raises(self):
        with self.assertRaises(RuntimeError):
            BM25Fast().search("任意")

    def test_run_all(self):
        run = self.r.run_all(self.split, top_k=3, verbose=False)
        self.assertEqual(set(run), {"q1"})
        self.assertLessEqual(len(run["q1"]), 3)

    def test_stats(self):
        st = self.r.stats()
        self.assertEqual(st["n_passages"], 3)
        self.assertGreater(st["n_vocab"], 0)
        self.assertGreater(st["nnz"], 0)


class TestPersistence(unittest.TestCase):
    def test_roundtrip(self):
        split = make_split(
            corpus={"p1": "机器学习", "p2": "深度学习"},
            queries={"q1": "学习"},
            qrels={"q1": ["p1", "p2"]},
        )
        r = BM25Fast()
        r.build(split, verbose=False)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "fast.pkl"
            r.save(p)
            r2 = BM25Fast.load(p)
            self.assertEqual(r2.passage_ids, r.passage_ids)
            self.assertEqual([x[0] for x in r2.search("机器学习", 2)],
                             [x[0] for x in r.search("机器学习", 2)])
            self.assertEqual(r2.vocab, r.vocab)

    def test_base_survives_roundtrip(self):
        # ⭐ 载入后必须还能 set_params（网格调参依赖这一点）
        split = make_split(
            corpus={"p1": "机器学习", "p2": "深度学习", "p3": "机器学习"},
            queries={"q1": "机器学习"},
            qrels={"q1": ["p1"]},
        )
        r = BM25Fast()
        r.build(split, verbose=False)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "fast.pkl"
            r.save(p)
            r2 = BM25Fast.load(p)
            self.assertTrue(r2.can_set_params, "载入后应保留基础统计")
            r2.set_params(k1=0.9, b=0.3)
            self.assertEqual(r2.k1, 0.9)
            self.assertGreater(len(r2.search("机器学习", 3)), 0)


class TestSetParams(unittest.TestCase):
    """⭐ set_params 是网格调参能加速的关键：必须与完整重建结果完全一致。"""

    def setUp(self):
        # 多文档、有词频差异，才能体现 k1/b 的作用
        corpus = {
            "p1": "苹果 苹果 苹果 香蕉",
            "p2": "苹果 香蕉 香蕉 香蕉",
            "p3": "苹果 橘子 香蕉",
            "p4": "橘子 橘子 西瓜",
        }
        self.split = make_split(
            corpus=corpus,
            queries={"q1": "苹果", "q2": "香蕉 橘子"},
            qrels={"q1": ["p1"], "q2": ["p3", "p4"]},
        )

    def test_set_params_equals_full_build(self):
        reused = BM25Fast()
        reused.build_base(self.split, verbose=False)

        for k1, b in [(1.5, 0.75), (0.9, 0.3), (2.0, 0.9), (1.2, 0.5)]:
            reused.set_params(k1=k1, b=b)

            fresh = BM25Fast(k1=k1, b=b)
            fresh.build(self.split, verbose=False)

            for qid in self.split.query_list():
                q = self.split.queries[qid]
                self.assertEqual(
                    [p for p, _ in reused.search(q, 5)],
                    [p for p, _ in fresh.search(q, 5)],
                    f"k1={k1}, b={b} 时 set_params 与完整重建排序不一致")

    def test_set_params_before_base_raises(self):
        with self.assertRaises(RuntimeError):
            BM25Fast().set_params(k1=1.0)

    def test_set_params_keeps_other_param(self):
        r = BM25Fast(k1=1.5, b=0.75)
        r.build_base(self.split, verbose=False)
        r.set_params(k1=2.0)          # 只改 k1
        self.assertEqual(r.k1, 2.0)
        self.assertEqual(r.b, 0.75)   # b 应保持


if __name__ == "__main__":
    unittest.main(verbosity=2)