"""公共实验工具测试 —— 保证与各脚本内联实现口径一致

⭐ 关键验收：本模块的 paired_test 必须与项目里已用的内联实现
   给出【完全相同】的结果，否则新老实验不可比。

运行：
    .venv\\Scripts\\python.exe -m unittest tests.test_experiment -v
"""

from __future__ import annotations

import unittest

import numpy as np
from scipy import stats

from src.eval.experiment import (
    EFFECT_THRESHOLDS,
    effect_label,
    evaluate_run,
    paired_test,
    per_query_metric,
)


def inline_paired(a: np.ndarray, b: np.ndarray) -> dict:
    """项目里 5 个脚本用的内联实现（用于对照）。"""
    diff = b - a
    t, p = stats.ttest_rel(b, a)
    sd = diff.std(ddof=1)
    d = diff.mean() / sd if sd > 0 else 0.0
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(diff), size=(10000, len(diff)))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return {"diff": round(float(diff.mean()), 6),
            "p_value": float(p), "cohens_d": round(float(d), 4),
            "ci95": [round(float(lo), 6), round(float(hi), 6)]}


class TestPairedTestConsistency(unittest.TestCase):
    """⭐ 核心：新模块必须与内联实现结果一致。"""

    def test_matches_inline_implementation(self):
        rng = np.random.default_rng(7)
        a = rng.random(500) * 0.5 + 0.3
        b = a + rng.normal(0.02, 0.1, 500)   # 有提升 + 噪声
        new = paired_test(a, b)
        old = inline_paired(a, b)
        for key in ("diff", "p_value", "cohens_d", "ci95"):
            self.assertAlmostEqual(new[key], old[key], places=6,
                                   msg=f"{key} 与内联实现不一致")

    def test_matches_when_no_difference(self):
        rng = np.random.default_rng(11)
        a = rng.random(300)
        new = paired_test(a, a.copy())
        self.assertEqual(new["diff"], 0.0)
        self.assertEqual(new["cohens_d"], 0.0)
        self.assertEqual(new["better"], 0)
        self.assertEqual(new["worse"], 0)


class TestPairedTestStats(unittest.TestCase):
    def test_basic_fields(self):
        a = np.array([0.5, 0.6, 0.7, 0.8])
        b = np.array([0.6, 0.7, 0.8, 0.9])
        r = paired_test(a, b)
        self.assertEqual(r["n"], 4)
        self.assertAlmostEqual(r["diff"], 0.1, places=9)
        self.assertAlmostEqual(r["rel_pct"],
                               100 * 0.1 / np.mean(a), places=1)
        self.assertEqual(r["better"], 4)
        self.assertEqual(r["worse"], 0)
        self.assertIn("ci95", r)
        self.assertIn("effect", r)

    def test_effect_labels(self):
        self.assertEqual(effect_label(0.1), "可忽略")
        self.assertEqual(effect_label(0.3), "小")
        self.assertEqual(effect_label(0.6), "中")
        self.assertEqual(effect_label(1.0), "大")
        # 负效应量按绝对值分级
        self.assertEqual(effect_label(-0.6), "中")

    def test_significance_flag(self):
        # 差异极大 → 应显著
        a = np.full(100, 0.5)
        b = np.full(100, 0.6)
        r = paired_test(a, b)
        self.assertTrue(r["significant"])

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            paired_test(np.array([1.0, 2.0]), np.array([1.0]))

    def test_too_few_samples_raises(self):
        with self.assertRaises(ValueError):
            paired_test(np.array([1.0]), np.array([2.0]))


class TestPerQueryMetric(unittest.TestCase):
    def setUp(self):
        self.run = {"q1": ["a", "b", "c"], "q2": ["x"]}
        self.qrels = {"q1": ["a", "b"], "q2": ["x"], "q3": []}   # q3 无标注

    def test_skips_empty_qrels(self):
        out = per_query_metric(self.run, self.qrels, "recall", 10)
        self.assertNotIn("q3", out)
        self.assertEqual(len(out), 2)

    def test_recall_values(self):
        out = per_query_metric(self.run, self.qrels, "recall", 10)
        self.assertAlmostEqual(out["q1"], 1.0)      # a,b 都在 top-3
        self.assertAlmostEqual(out["q2"], 1.0)

    def test_unknown_metric_raises(self):
        with self.assertRaises(ValueError):
            per_query_metric(self.run, self.qrels, "bogus", 10)

    def test_missing_query_in_run_is_zero(self):
        out = per_query_metric({"q1": ["a", "b"]}, self.qrels, "recall", 10)
        self.assertAlmostEqual(out["q2"], 0.0)      # run 里没有 q2


class TestEvaluateRun(unittest.TestCase):
    def test_multi_k_multi_metric(self):
        run = {"q1": ["a", "b", "c"]}
        qrels = {"q1": ["a", "b", "c"]}
        out = evaluate_run(run, qrels, ks=(1, 3), metrics=("recall", "ndcg"))
        self.assertIn("recall@1", out)
        self.assertIn("ndcg@3", out)
        self.assertAlmostEqual(out["recall@3"], 1.0)
        self.assertAlmostEqual(out["ndcg@3"], 1.0)

    def test_empty_qrels_gives_empty(self):
        out = evaluate_run({"q1": ["a"]}, {"q1": []})
        self.assertEqual(out, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)