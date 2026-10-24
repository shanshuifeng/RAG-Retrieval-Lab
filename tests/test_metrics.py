"""评测指标单元测试 —— M2.3 硬门槛

⭐ 为什么必须写：指标算错则后面所有实验结论都是错的（docs/09 M2.3）。
   本测试覆盖 docs/05 给定的人工算例 + 边界条件 + 常见错误。

运行（无需 pytest，标准库即可）：
    .venv\\Scripts\\python.exe -m unittest tests.test_metrics -v
"""

from __future__ import annotations

import math
import unittest

from src.eval.metrics import (
    dcg_at_k,
    evaluate,
    format_table,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

EPS = 1e-9


class TestRecall(unittest.TestCase):
    """docs/05 §3.1 的人工算例。"""

    def test_doc_example_at_5(self):
        # 3 个相关 {A,B,C}，返回 Top-5=[A,D,B,E,F] → 命中 2/3
        self.assertAlmostEqual(recall_at_k(["A", "D", "B", "E", "F"], {"A", "B", "C"}, 5),
                               2 / 3, places=9)

    def test_doc_example_at_10(self):
        ranked = ["A", "D", "B", "E", "F", "G", "H", "C"]
        self.assertAlmostEqual(recall_at_k(ranked, {"A", "B", "C"}, 10), 1.0, places=9)

    def test_perfect_and_zero(self):
        self.assertEqual(recall_at_k(["A", "B", "C"], {"A", "B", "C"}, 3), 1.0)
        self.assertEqual(recall_at_k(["X", "Y"], {"A", "B"}, 10), 0.0)

    def test_empty_relevant(self):
        self.assertEqual(recall_at_k(["A"], set(), 5), 0.0)

    def test_duplicates_not_double_counted(self):
        # 同一 docid 重复出现只能算一次，否则会虚增指标
        self.assertAlmostEqual(recall_at_k(["A", "A", "A"], {"A", "B"}, 10), 0.5, places=9)

    def test_k_larger_than_run(self):
        self.assertAlmostEqual(recall_at_k(["A"], {"A", "B"}, 100), 0.5, places=9)


class TestPrecision(unittest.TestCase):
    """docs/05 §3.2 的人工算例。"""

    def test_doc_example(self):
        self.assertAlmostEqual(precision_at_k(["A", "D", "B", "E", "F"], {"A", "B", "C"}, 5),
                               2 / 5, places=9)

    def test_short_run_uses_actual_length(self):
        # 只返回 3 条且全对：不应被算成 3/10
        self.assertEqual(precision_at_k(["A", "B", "C"], {"A", "B", "C"}, 10), 1.0)

    def test_empty_run(self):
        self.assertEqual(precision_at_k([], {"A"}, 5), 0.0)


class TestMRR(unittest.TestCase):
    """docs/05 §3.3 的人工算例。"""

    def test_doc_example(self):
        vals = [mrr_at_k(["X", "Y", "A"], {"A"}, 10),      # 1/3
                mrr_at_k(["A"], {"A"}, 10),                # 1/1
                mrr_at_k(list("123456789A"), {"A"}, 10)]   # 1/10
        self.assertAlmostEqual(sum(vals) / 3, (1 / 3 + 1.0 + 0.1) / 3, places=9)

    def test_first_hit_only(self):
        # 多个相关文档，MRR 只看第一个
        self.assertEqual(mrr_at_k(["A", "B"], {"A", "B"}, 10), 1.0)

    def test_outside_k(self):
        self.assertEqual(mrr_at_k(["X", "Y", "A"], {"A"}, 2), 0.0)


class TestDCG(unittest.TestCase):
    """docs/05 §3.4 的折损权重。"""

    def test_first_position_weight_one(self):
        self.assertAlmostEqual(dcg_at_k([1.0], 5), 1.0, places=9)

    def test_position_weights(self):
        # 位置 i 的折损 = log2(i+1)
        gains = [1.0, 0.0, 1.0]
        expected = 1 / math.log2(2) + 0 + 1 / math.log2(4)
        self.assertAlmostEqual(dcg_at_k(gains, 3), expected, places=9)

    def test_zero_gains(self):
        self.assertEqual(dcg_at_k([0.0, 0.0, 0.0], 3), 0.0)


class TestNDCG(unittest.TestCase):
    """docs/05 §3.4 完整算例（含文档里那段自我纠错）。"""

    def test_doc_example_0886(self):
        # 3 个相关，Top-5 label = [1,0,1,0,1] → nDCG@5 = 1.887/2.131 ≈ 0.8856
        ranked = ["A", "X", "B", "Y", "C"]
        val = ndcg_at_k(ranked, {"A", "B", "C"}, 5)
        self.assertAlmostEqual(val, 1.8866 / 2.1309, places=3)
        self.assertAlmostEqual(val, 0.8856, places=3)

    def test_perfect_ranking_is_one(self):
        # 理想排序 → nDCG = 1.0（M2.3 验收标准）
        self.assertAlmostEqual(ndcg_at_k(["A", "B", "C"], {"A", "B", "C"}, 3), 1.0, places=9)

    def test_no_hits_is_zero(self):
        self.assertEqual(ndcg_at_k(["X", "Y", "Z"], {"A", "B"}, 3), 0.0)

    def test_all_relevant_order_irrelevant_binary(self):
        # ⭐ docs/10 §六「发现 2」：二值相关性下全命中时 nDCG=1.0，与顺序无关
        self.assertAlmostEqual(ndcg_at_k([3, 2, 1], {1, 2, 3}, 3), 1.0, places=9)
        self.assertAlmostEqual(ndcg_at_k([1, 2, 3], {1, 2, 3}, 3), 1.0, places=9)

    def test_partial_relevant_below_one(self):
        # 部分相关时排序才影响指标
        v = ndcg_at_k(["X", "A"], {"A", "B", "C"}, 2)
        self.assertLess(v, 1.0)
        self.assertGreater(v, 0.0)

    def test_never_exceeds_one(self):
        # 随机若干组，nDCG 必须始终 <= 1（含断言检查）
        import random
        rng = random.Random(0)
        all_ids = [f"d{i}" for i in range(50)]
        for _ in range(200):
            n_rel = rng.randint(1, 15)
            rel = set(rng.sample(all_ids, n_rel))
            ranked = rng.sample(all_ids, rng.randint(1, 50))
            for k in (1, 3, 5, 10, 20, 100):
                v = ndcg_at_k(ranked, rel, k)
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0 + EPS)

    def test_k_truncation_monotone_hit(self):
        # 理想位置提前时 nDCG 不应下降
        rel = {"A", "B"}
        early = ndcg_at_k(["A", "B", "X"], rel, 3)
        late = ndcg_at_k(["X", "A", "B"], rel, 3)
        self.assertGreater(early, late)

    def test_empty_relevant(self):
        self.assertEqual(ndcg_at_k(["A"], set(), 5), 0.0)


class TestEvaluate(unittest.TestCase):
    def test_aggregate_and_filters(self):
        queries = {"q1": {"A"}, "q2": {"B"}, "q3": set()}
        run = {"q1": ["A"], "q2": ["X", "B"]}
        res = evaluate(queries, run, ks=[1, 10], metrics=["recall", "ndcg"])
        self.assertEqual(res["filtered"]["empty_relevant"], 1)  # q3 被过滤
        self.assertEqual(res["filtered"]["queries_evaluated"], 2)
        self.assertAlmostEqual(res["aggregate"]["recall@1"]["mean"], 0.5, places=9)
        self.assertAlmostEqual(res["aggregate"]["recall@10"]["mean"], 1.0, places=9)
        # ⭐ q1: A 在第 1 位 → nDCG=1.0；q2: B 在第 2 位 → nDCG=1/log2(3)
        expected_ndcg = (1.0 + 1.0 / math.log2(3)) / 2
        self.assertAlmostEqual(res["aggregate"]["ndcg@10"]["mean"], expected_ndcg, places=9)
        self.assertAlmostEqual(expected_ndcg, 0.8154648767857288, places=9)

    def test_missing_run_counted(self):
        queries = {"q1": {"A"}, "q2": {"B"}}
        run = {"q1": ["A"]}
        res = evaluate(queries, run, ks=[1])
        self.assertEqual(res["filtered"]["missing_in_run"], 1)
        self.assertEqual(res["filtered"]["queries_evaluated"], 1)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            evaluate({"q": {"A"}}, {"q": ["A"]}, metrics=["bogus"])

    def test_empty_result_still_has_keys(self):
        res = evaluate({}, {}, ks=[1, 5])
        self.assertEqual(res["aggregate"], {})
        self.assertEqual(res["filtered"]["queries_evaluated"], 0)

    def test_format_table_runs(self):
        res = evaluate({"q1": {"A"}}, {"q1": ["A"]}, ks=[1, 5], metrics=["recall"])
        text = format_table(res["aggregate"], [1, 5], ["recall"])
        # 行 = 指标，列 = K 值
        self.assertIn("K=1", text)
        self.assertIn("K=5", text)
        self.assertIn("RECALL", text)
        self.assertIn("1.0000", text)


class TestKValidation(unittest.TestCase):
    def test_invalid_k(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                recall_at_k(["A"], {"A"}, bad)
            with self.assertRaises(ValueError):
                ndcg_at_k(["A"], {"A"}, bad)


class TestInputTypeRobustness(unittest.TestCase):
    """⭐ 输入类型多样性测试 —— 防止「生成器被耗尽」类 bug 复发。

    2026-10-24 事故：mrr_at_k 曾把 rel 写成生成器 `(x for x in relevant)`，
    导致 `if not rel` 恒为假、且循环中 `d in rel` 会边判断边耗尽生成器，
    MRR 系统性偏低且【不报错】。

    本测试强制所有指标函数对 list / set / tuple / 生成器 / frozenset
    都返回【完全相同】的结果。
    """

    RANKED = ["X", "A", "Y", "B", "Z"]
    REL = ["A", "B"]

    def _variants(self, rel):
        return {
            "list": list(rel),
            "set": set(rel),
            "tuple": tuple(rel),
            "frozenset": frozenset(rel),
            "generator": (x for x in rel),
            "dict_keys": dict.fromkeys(rel).keys(),
        }

    def test_all_metrics_type_agnostic(self):
        from src.eval.metrics import METRICS
        for name, fn in METRICS.items():
            baseline = None
            for variant, rel_obj in self._variants(self.REL).items():
                val = fn(self.RANKED, rel_obj, 10)
                if baseline is None:
                    baseline = val
                    self.assertGreater(val, 0.0, f"{name} 应命中，却得 0")
                else:
                    self.assertAlmostEqual(
                        val, baseline, places=12,
                        msg=f"⚠️ {name} 对 {variant} 输入的结果不一致："
                            f"{val} != {baseline}（生成器陷阱？）")

    def test_empty_relevant_all_types(self):
        from src.eval.metrics import METRICS
        for name, fn in METRICS.items():
            for variant, rel_obj in self._variants([]).items():
                self.assertEqual(
                    fn(self.RANKED, rel_obj, 10), 0.0,
                    f"⚠️ {name} 对空标注（{variant}）应返回 0.0")

    def test_generator_not_consumed_twice(self):
        # 直接针对事故形态：生成器作为 relevant 传入，必须能正确命中第 2 位
        gen = (x for x in ["A", "B"])
        # RANKED 中 A 在第 2 位 → MRR 应为 1/2
        self.assertAlmostEqual(mrr_at_k(self.RANKED, gen, 10), 0.5, places=12)

    def test_ranked_as_generator(self):
        # ranked 也应是可迭代即用，不应因生成器而漏结果
        gen_ranked = (x for x in self.RANKED)
        self.assertAlmostEqual(recall_at_k(gen_ranked, {"A", "B"}, 10), 1.0, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)