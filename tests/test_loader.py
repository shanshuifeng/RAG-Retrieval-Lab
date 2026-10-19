"""数据加载层测试 —— 确保 dev/test 读取口径一致且能抓出脏数据

运行：
    .venv\\Scripts\\python.exe -m unittest tests.test_loader -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data.loader import DataError, load_split


def make_split_dir(root: Path, name: str, corpus, queries, qrels_tsv=None) -> None:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "corpus.jsonl", "w", encoding="utf-8") as f:
        for pid, text in corpus:
            f.write(json.dumps({"id": pid, "text": text}, ensure_ascii=False) + "\n")
    with open(d / "queries.jsonl", "w", encoding="utf-8") as f:
        for qid, q, rel in queries:
            f.write(json.dumps({"qid": qid, "query": q, "relevant_ids": rel},
                               ensure_ascii=False) + "\n")
    if qrels_tsv is not None:
        (d / "qrels.tsv").write_text(qrels_tsv, encoding="utf-8")


class TestLoader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_and_summarizes(self):
        make_split_dir(
            self.root, "mini",
            corpus=[("p1", "段落一"), ("p2", "段落二"), ("p3", "段落三")],
            queries=[("q1", "查询一", ["p1", "p2"]), ("q2", "查询二", ["p3"])],
        )
        s = load_split("mini", base=self.root)
        self.assertEqual(s.n_passages, 3)
        self.assertEqual(s.n_queries, 2)
        self.assertEqual(s.n_relevant_pairs, 3)
        self.assertEqual(s.query_list(), ["q1", "q2"])
        self.assertEqual(s.corpus["p1"], "段落一")

    def test_detects_dangling_reference(self):
        # ⭐ 最危险的一类脏数据：qrels 指向不存在的段落 → 指标悄悄变差
        make_split_dir(
            self.root, "bad",
            corpus=[("p1", "x")],
            queries=[("q1", "查询", ["p1", "p_missing"])],
        )
        with self.assertRaises(DataError) as cm:
            load_split("bad", base=self.root)
        self.assertIn("不存在的段落", str(cm.exception))

    def test_detects_duplicate_passage_id(self):
        make_split_dir(
            self.root, "dup",
            corpus=[("p1", "x"), ("p1", "y")],
            queries=[("q1", "查询", ["p1"])],
        )
        with self.assertRaises(DataError) as cm:
            load_split("dup", base=self.root)
        self.assertIn("重复", str(cm.exception))

    def test_detects_qrels_tsv_mismatch(self):
        make_split_dir(
            self.root, "mis",
            corpus=[("p1", "x"), ("p2", "y")],
            queries=[("q1", "查询", ["p1"])],
            qrels_tsv="# qid\tpid\tscore\nq1\tp2\t1\n",
        )
        with self.assertRaises(DataError) as cm:
            load_split("mis", base=self.root)
        self.assertIn("不一致", str(cm.exception))

    def test_accepts_matching_tsv(self):
        make_split_dir(
            self.root, "ok",
            corpus=[("p1", "x"), ("p2", "y")],
            queries=[("q1", "查询", ["p1", "p2"])],
            qrels_tsv="# qid\tpid\tscore\nq1\tp1\t1\nq1\tp2\t1\n",
        )
        s = load_split("ok", base=self.root)
        self.assertEqual(s.n_relevant_pairs, 2)

    def test_missing_dir_raises(self):
        with self.assertRaises(DataError):
            load_split("nope", base=self.root)

    def test_query_without_relevant_is_kept_but_filtered(self):
        make_split_dir(
            self.root, "mixed",
            corpus=[("p1", "x")],
            queries=[("q1", "有标注", ["p1"]), ("q2", "无标注", [])],
        )
        s = load_split("mixed", base=self.root)
        self.assertEqual(s.n_queries, 2)          # 都保留
        self.assertEqual(s.query_list(True), ["q1"])   # 评测时过滤
        self.assertEqual(s.query_list(False), ["q1", "q2"])


class TestRealData(unittest.TestCase):
    """对真实 dev / test 数据的冒烟测试（若数据存在）。"""

    @classmethod
    def setUpClass(cls):
        from src.data.loader import PROCESSED
        cls.proc = PROCESSED

    def _skip_if_missing(self, name: str):
        if not (self.proc / name / "corpus.jsonl").exists():
            self.skipTest(f"缺少 {name} 数据，跳过")

    def test_dev_real(self):
        self._skip_if_missing("dev")
        s = load_split("dev")
        self.assertEqual(s.n_queries, 8640)
        self.assertEqual(s.n_passages, 100000)
        self.assertEqual(s.n_relevant_pairs, 22470)
        self.assertEqual(len(s.query_list()), 8640)

    def test_test_real(self):
        self._skip_if_missing("test")
        s = load_split("test")
        self.assertEqual(s.n_queries, 2000)
        self.assertEqual(s.n_passages, 100001)
        self.assertEqual(s.n_relevant_pairs, 9839)

    def test_dev_test_queries_disjoint(self):
        # ⭐ dev 与 test 的 query 不得重叠（否则等于在测试集上调参）
        for n in ("dev", "test"):
            self._skip_if_missing(n)
        dev = load_split("dev")
        test = load_split("test")
        overlap = set(dev.queries.values()) & set(test.queries.values())
        self.assertEqual(overlap, set(), f"dev/test query 重叠 {len(overlap)} 条")


if __name__ == "__main__":
    unittest.main(verbosity=2)