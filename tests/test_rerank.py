"""Cross-encoder 重排器单元测试

⚠️ 不加载真实模型（太慢、依赖网络/GPU）。用 mock 替换打分函数，
   只验证：接口契约、排序逻辑、延迟计时、并列稳定性、错误处理。
   真实模型行为由 scripts/verify_rerank.py 做端到端验证。

运行：
    .venv\\Scripts\\python.exe -m unittest tests.test_rerank -v
"""

from __future__ import annotations

import unittest

from src.rerank.cross_encoder import CrossEncoderReranker


def make_reranker_with_fake_scores(scores_map: dict[str, float]) -> CrossEncoderReranker:
    """注入确定性假打分：分数由段落文本查表决定（可预测、可复现）。"""
    r = CrossEncoderReranker()

    def fake_score(query, passages, show_progress=False):
        return [scores_map.get(t, 0.0) for t in passages]

    r.score = fake_score  # type: ignore
    return r


class TestRerankLogic(unittest.TestCase):
    def setUp(self):
        # 假分数：p3 最高，p1 次之，p2 最低
        self.scores = {"段落一": 0.5, "段落二": 0.1, "段落三": 0.9}
        self.r = make_reranker_with_fake_scores(self.scores)

    def test_rerank_orders_by_score(self):
        res = self.r.rerank("q1", "查询",
                            [("p1", "段落一"), ("p2", "段落二"), ("p3", "段落三")])
        self.assertEqual(res.ranked, ["p3", "p1", "p2"])

    def test_rerank_records_scores(self):
        res = self.r.rerank("q1", "查询",
                            [("p1", "段落一"), ("p3", "段落三")])
        self.assertEqual(res.scores["p1"], 0.5)
        self.assertEqual(res.scores["p3"], 0.9)

    def test_rerank_records_cost(self):
        res = self.r.rerank("q1", "查询", [("p1", "段落一")])
        self.assertGreaterEqual(res.latency_ms, 0.0)
        self.assertEqual(res.n_candidates, 1)
        self.assertEqual(res.qid, "q1")

    def test_empty_candidates(self):
        res = self.r.rerank("q1", "查询", [])
        self.assertEqual(res.ranked, [])
        self.assertEqual(res.n_candidates, 0)

    def test_tie_break_deterministic(self):
        # 分数相同 → 按 pid 字典序（可复现）
        scores = {"x": 0.5, "y": 0.5, "z": 0.5}
        r = make_reranker_with_fake_scores(scores)
        res1 = r.rerank("q", "查询", [("b", "y"), ("a", "x"), ("c", "z")])
        res2 = r.rerank("q", "查询", [("c", "z"), ("a", "x"), ("b", "y")])
        self.assertEqual(res1.ranked, res2.ranked)
        self.assertEqual(res1.ranked, ["a", "b", "c"])

    def test_output_length_matches_input(self):
        cands = [(f"p{i}", f"text{i}") for i in range(7)]
        res = self.r.rerank("q", "查询", cands)
        self.assertEqual(len(res.ranked), 7)
        self.assertEqual(set(res.ranked), {f"p{i}" for i in range(7)})


class TestScoreContract(unittest.TestCase):
    def test_score_returns_same_length(self):
        r = make_reranker_with_fake_scores({"a": 1.0})
        out = r.score("q", ["a", "b", "c"])
        self.assertEqual(len(out), 3)

    def test_score_values_pass_through(self):
        r = make_reranker_with_fake_scores({"a": 1.5, "b": -0.3})
        out = r.score("q", ["a", "b"])
        self.assertEqual(out, [1.5, -0.3])


if __name__ == "__main__":
    unittest.main(verbosity=2)