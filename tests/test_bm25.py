"""BM25 检索器测试

运行：
    .venv\\Scripts\\python.exe -m unittest tests.test_bm25 -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data.loader import Split
from src.retrieval.bm25 import BM25Retriever, tokenize_zh


def make_split(corpus: dict[str, str], queries: dict[str, str],
               qrels: dict[str, list[str]]) -> Split:
    s = Split(name="toy")
    s.corpus = dict(corpus)
    s.passage_ids = list(corpus.keys())
    s.queries = dict(queries)
    s.qrels = dict(qrels)
    return s


class TestTokenizer(unittest.TestCase):
    def test_chinese_sentence(self):
        toks = tokenize_zh("南昌大学是国家双一流建设高校")
        self.assertIn("南昌大学", toks)
        # ⚠️ 实测（docs/10 §五）：jieba 默认词典把「双一流」切成 双 / 一流，
        #    不会作为整词保留。这里按【真实行为】断言，而不是按期望。
        self.assertIn("双", toks)
        self.assertIn("一流", toks)
        self.assertTrue(all(t.strip() for t in toks), "不应含纯空白 token")

    def test_custom_word_added(self):
        # ⭐ 若要「双一流」成词，须显式 jieba.add_word —— 记录该做法以备 E1 调优
        import jieba
        jieba.add_word("双一流")
        try:
            toks = tokenize_zh("南昌大学是国家双一流建设高校")
            self.assertIn("双一流", toks)
        finally:
            jieba.del_word("双一流")

    def test_empty(self):
        self.assertEqual(tokenize_zh(""), [])
        self.assertEqual(tokenize_zh("   "), [])


class TestBM25Basic(unittest.TestCase):
    def setUp(self):
        # 3 段：查询词只出现在 p1 和 p3
        self.split = make_split(
            corpus={
                "p1": "板蓝根颗粒的功效与作用是清热解毒",
                "p2": "紫薯的药理作用是抗氧化",
                "p3": "板蓝根颗粒可以治疗感冒发热",
            },
            queries={"q1": "板蓝根颗粒的功效", "q2": "紫薯抗氧化"},
            qrels={"q1": ["p1", "p3"], "q2": ["p2"]},
        )
        self.r = BM25Retriever(k1=1.5, b=0.75)
        self.r.build(self.split, verbose=False)

    def test_finds_relevant_passage_first(self):
        res = self.r.search("板蓝根颗粒的功效", top_k=3)
        self.assertTrue(res, "检索结果不应为空")
        top_ids = [pid for pid, _ in res]
        self.assertEqual(top_ids[0], "p1", f"最相关段落应排第一，实际 {top_ids}")
        self.assertIn("p3", top_ids)

    def test_scores_descending(self):
        res = self.r.search("板蓝根颗粒", top_k=3)
        scores = [s for _, s in res]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_irrelevant_query_no_dense_hit(self):
        # 查询词完全不在语料里 → 所有分数应约为 0
        res = self.r.search("量子纠缠拓扑绝缘体", top_k=3)
        self.assertTrue(all(s <= 1e-9 for _, s in res))

    def test_top_k_truncation(self):
        res = self.r.search("板蓝根", top_k=2)
        self.assertEqual(len(res), 2)

    def test_empty_query_returns_empty(self):
        self.assertEqual(self.r.search("", top_k=5), [])

    def test_run_all_covers_all_queries(self):
        run = self.r.run_all(self.split, top_k=3, verbose=False)
        self.assertEqual(set(run), {"q1", "q2"})
        for qid, pids in run.items():
            self.assertTrue(len(pids) <= 3)

    def test_stats_recorded(self):
        st = self.r.stats()
        self.assertEqual(st["n_passages"], 3)
        self.assertGreater(st["n_tokens"], 0)
        self.assertEqual(st["k1"], 1.5)


class TestBM25Params(unittest.TestCase):
    def test_k1_b_affect_scores(self):
        split = make_split(
            corpus={"p1": "苹果 苹果 苹果 香蕉", "p2": "苹果 香蕉 香蕉 香蕉"},
            queries={"q1": "苹果"},
            qrels={"q1": ["p1"]},
        )
        low = BM25Retriever(k1=0.9, b=0.3)
        low.build(split, verbose=False)
        high = BM25Retriever(k1=2.0, b=0.9)
        high.build(split, verbose=False)
        s_low = dict(low.search("苹果", 2))
        s_high = dict(high.search("苹果", 2))
        # 参数不同 → 分数不同（证明参数确实生效，不是摆设）
        self.assertNotAlmostEqual(s_low["p1"], s_high["p1"], places=6)


class TestPersistence(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        split = make_split(
            corpus={"p1": "机器学习", "p2": "深度学习"},
            queries={"q1": "学习"},
            qrels={"q1": ["p1", "p2"]},
        )
        r = BM25Retriever()
        r.build(split, verbose=False)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bm25.pkl"
            r.save(p)
            self.assertTrue(p.exists())
            r2 = BM25Retriever.load(p)
            self.assertEqual(r2.passage_ids, r.passage_ids)
            self.assertEqual([x[0] for x in r2.search("机器学习", 2)],
                             [x[0] for x in r.search("机器学习", 2)])


class TestValidation(unittest.TestCase):
    def test_search_before_build_raises(self):
        r = BM25Retriever()
        with self.assertRaises(RuntimeError):
            r.search("任意查询")


if __name__ == "__main__":
    unittest.main(verbosity=2)