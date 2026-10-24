"""分块模块测试 —— E4 的最大坑（chunk↔parent 映射）有专门保障

运行：
    .venv\\Scripts\\python.exe -m unittest tests.test_chunking -v
"""

from __future__ import annotations

import unittest

from src.data.chunking import (
    build_chunked_corpus,
    chunk_fixed,
    chunk_native,
    chunk_paragraph,
    chunked_split_view,
)
from src.data.loader import Split


def make_split(corpus: dict[str, str]) -> Split:
    s = Split(name="toy")
    s.corpus = dict(corpus)
    s.passage_ids = list(corpus.keys())
    return s


class TestChunkFixed(unittest.TestCase):
    def test_short_text_single_chunk(self):
        self.assertEqual(chunk_fixed("短文本", size=256, overlap=50), ["短文本"])

    def test_empty(self):
        self.assertEqual(chunk_fixed("", 256, 50), [])
        self.assertEqual(chunk_fixed("   ", 256, 50), [])

    def test_length_and_overlap(self):
        text = "x" * 600
        chunks = chunk_fixed(text, size=256, overlap=50)
        # 步长 206：0, 206, 412, 618（618>=600 停）→ [0:256],[206:462],[412:600]
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0], text[0:256])
        self.assertEqual(chunks[1], text[206:462])
        self.assertEqual(chunks[2], text[412:600])
        self.assertEqual(len(chunks[0]), 256)

    def test_overlap_actually_overlaps(self):
        text = "".join(chr(65 + (i % 26)) for i in range(300))
        chunks = chunk_fixed(text, size=256, overlap=50)
        # 第二块的前 50 字 = 第一块的 [206:256] 末尾 50 字
        self.assertEqual(chunks[1][:50], chunks[0][-50:])

    def test_full_coverage(self):
        text = "y" * 1000
        chunks = chunk_fixed(text, size=256, overlap=50)
        # 所有块拼接（去重叠）应覆盖全文
        covered = chunks[0]
        for c in chunks[1:]:
            covered += c[50:]
        self.assertEqual(covered, text)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            chunk_fixed("x", size=0)
        with self.assertRaises(ValueError):
            chunk_fixed("x", size=100, overlap=100)


class TestChunkParagraph(unittest.TestCase):
    def test_splits_on_blank_line(self):
        text = "第一段\n\n第二段\n\n\n第三段"
        self.assertEqual(chunk_paragraph(text), ["第一段", "第二段", "第三段"])

    def test_no_blank_line_single(self):
        self.assertEqual(chunk_paragraph("没有空行的段落"), ["没有空行的段落"])

    def test_long_paragraph_fallback(self):
        long_para = "z" * 1200
        text = f"短段\n\n{long_para}"
        chunks = chunk_paragraph(text, max_len=512)
        self.assertEqual(chunks[0], "短段")
        self.assertTrue(all(len(c) <= 512 for c in chunks))
        self.assertGreater(len(chunks), 2)

    def test_empty(self):
        self.assertEqual(chunk_paragraph(""), [])


class TestChunkNative(unittest.TestCase):
    def test_passthrough(self):
        self.assertEqual(chunk_native("原文"), ["原文"])
        self.assertEqual(chunk_native(""), [])


class TestMapping(unittest.TestCase):
    """⭐ chunk ↔ parent 映射是 E4 的最大坑。"""

    def setUp(self):
        self.split = make_split({
            "p1": "短段落",
            "p2": "x" * 600,          # 会分成 3 块
            "p3": "a\n\nb\n\nc",      # 段落分成 3 块
        })

    def test_chunk_ids_contain_parent(self):
        cc = build_chunked_corpus(self.split, "fixed256", verbose=False)
        for cid in cc.passage_ids:
            self.assertIn("#c", cid)
            self.assertTrue(cid.startswith(cc.chunk_to_parent[cid]))

    def test_every_chunk_maps_to_existing_parent(self):
        cc = build_chunked_corpus(self.split, "fixed256", verbose=False)
        parent_set = set(self.split.passage_ids)
        for cid in cc.passage_ids:
            self.assertIn(cc.chunk_to_parent[cid], parent_set)

    def test_parent_chunks_inverse_consistent(self):
        cc = build_chunked_corpus(self.split, "fixed256", verbose=False)
        # 双向映射必须互逆
        for pid, chunks in cc.parent_chunks.items():
            for cid in chunks:
                self.assertEqual(cc.chunk_to_parent[cid], pid)
        # 块总数一致
        self.assertEqual(sum(len(v) for v in cc.parent_chunks.values()),
                         len(cc.passage_ids))

    def test_project_ranking_dedups_parents(self):
        cc = build_chunked_corpus(self.split, "fixed256", verbose=False)
        # p2 的多个块出现在排名中 → 投影后 p2 只占一个位置
        ranked = ["p2#c0", "p1#c0", "p2#c1", "p2#c2"]
        projected = cc.project_ranking(ranked)
        self.assertEqual(projected, ["p2", "p1"])

    def test_project_ranking_unknown_chunk_raises(self):
        cc = build_chunked_corpus(self.split, "native", verbose=False)
        with self.assertRaises(KeyError):
            cc.project_ranking(["不存在#c0"])

    def test_native_strategy_one_chunk_per_parent(self):
        cc = build_chunked_corpus(self.split, "native", verbose=False)
        self.assertEqual(cc.stats()["chunks_per_parent_max"], 1)

    def test_chunked_view_preserves_queries_and_qrels(self):
        cc = build_chunked_corpus(self.split, "fixed256", verbose=False)
        self.split.queries = {"q1": "查询"}
        self.split.qrels = {"q1": ["p2"]}
        view = chunked_split_view(self.split, cc)
        self.assertEqual(view.queries, {"q1": "查询"})
        self.assertEqual(view.qrels, {"q1": ["p2"]})   # ⭐ qrels 保持段落级
        self.assertGreater(len(view.passage_ids), 3)   # 块数 > 段数


if __name__ == "__main__":
    unittest.main(verbosity=2)