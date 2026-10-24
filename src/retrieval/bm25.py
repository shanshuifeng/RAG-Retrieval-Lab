"""BM25 检索器 —— E1 的强基线

⭐ 为什么 BM25 是必做的第一件事（docs/06 §1.2）：
   · 它是稀疏检索的经典强基线，没有 BM25 的"提升"毫无意义
   · 它实现简单，跑通即证明「评测链路」正确
   · 它与稠密检索形成对照，回答"向量检索到底值多少"

实现要点（用 rank_bm25 库，不自造轮子）：
   · 分词：jieba（中文必须分词；BM25 本身是词袋模型）
   · 参数：k1（词频饱和）、b（长度归一化）—— 这两个是 E1 要调的变量
   · 索引：BM25Okapi（Lucene 式 IDF 变体，社区最常用）

⚠️ 已知取舍：
   · 这里用纯 Python 分词 + rank_bm25，10 万段语料建索引约 1-3 分钟、内存约 2-4 GB。
   · 若后续要上 100 万段，需换成更省内存的实现（见 docs/03 §七 性能预算）。
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from src.data.loader import Split

#  jieba 首次导入会加载词典（约 1s），且默认往 stderr 打日志
jieba.setLogLevel(60)  # 静默


def tokenize_zh(text: str) -> list[str]:
    """中文分词。

    ⚠️ 只保留非空白 token；标点保留（BM25 的 IDF 会自动把它们压到很低权重）。
    """
    return [w for w in jieba.lcut(text) if w.strip()]


@dataclass
class BM25Retriever:
    """BM25 检索器（稀疏检索基线）。"""

    k1: float = 1.5
    b: float = 0.75
    name: str = "bm25"

    passage_ids: list[str] = field(default_factory=list)
    _bm25: BM25Okapi | None = None
    _build_sec: float = 0.0
    _n_tokens: int = 0

    # ── 建索引 ──
    def build(self, split: Split, verbose: bool = True) -> None:
        """在给定语料的【全部段落】上建索引。

         必须用全语料建索引：检索任务的语料库是固定的，
        所有方法（BM25/稠密/混合）都在同一个库上检索，才能横向比较。
        """
        t0 = time.time()
        self.passage_ids = list(split.passage_ids)
        corpus_tokens = []
        total_tokens = 0
        for i, pid in enumerate(self.passage_ids):
            toks = tokenize_zh(split.corpus[pid])
            total_tokens += len(toks)
            corpus_tokens.append(toks)
            if verbose and (i + 1) % 20000 == 0:
                print(f"    分词 {i+1:,}/{len(self.passage_ids):,} ...", flush=True)

        self._n_tokens = total_tokens
        self._bm25 = BM25Okapi(corpus_tokens, k1=self.k1, b=self.b)
        self._build_sec = time.time() - t0
        if verbose:
            print(f"    BM25 索引完成：{len(self.passage_ids):,} 段，"
                  f"{total_tokens:,} tokens，用时 {self._build_sec:.1f}s", flush=True)

    # ── 检索 ──
    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """返回 [(passage_id, score), ...]，按分数降序。"""
        if self._bm25 is None:
            raise RuntimeError("请先调用 build() 建索引")
        toks = tokenize_zh(query)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        # argsort 取前 top_k（numpy 已在依赖内）
        import numpy as np
        k = min(top_k, len(scores))
        idx = np.argpartition(-np.asarray(scores), k - 1)[:k]
        idx = idx[np.argsort(-np.asarray(scores)[idx])]
        return [(self.passage_ids[int(i)], float(scores[int(i)])) for i in idx]

    def run_all(self, split: Split, top_k: int = 100,
                verbose: bool = True) -> dict[str, list[str]]:
        """对全部有标注的 query 批量检索，返回 {qid: [pid, ...]}。"""
        qids = split.query_list(with_relevant_only=True)
        out: dict[str, list[str]] = {}
        t0 = time.time()
        for i, qid in enumerate(qids, start=1):
            res = self.search(split.queries[qid], top_k=top_k)
            out[qid] = [pid for pid, _ in res]
            if verbose and i % 1000 == 0:
                el = time.time() - t0
                print(f"    检索 {i:,}/{len(qids):,}  ({el:.1f}s, "
                      f"{i/el:.0f} q/s) ...", flush=True)
        if verbose:
            el = time.time() - t0
            print(f"    检索完成：{len(qids):,} query，用时 {el:.1f}s "
                  f"({len(qids)/max(el,1e-9):.0f} q/s)", flush=True)
        return out

    # ── 持久化（⭐ 可复现性：索引建一次就别重建）──
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "k1": self.k1, "b": self.b,
                "passage_ids": self.passage_ids,
                "bm25": self._bm25,
                "build_sec": self._build_sec,
                "n_tokens": self._n_tokens,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> BM25Retriever:
        with open(path, "rb") as f:
            d = pickle.load(f)
        r = cls(k1=d["k1"], b=d["b"])
        r.passage_ids = d["passage_ids"]
        r._bm25 = d["bm25"]
        r._build_sec = d.get("build_sec", 0.0)
        r._n_tokens = d.get("n_tokens", 0)
        return r

    # ── 成本指标（⭐ 必须报告，docs/06 §3.6）──
    def stats(self) -> dict:
        return {
            "method": "bm25",
            "k1": self.k1,
            "b": self.b,
            "n_passages": len(self.passage_ids),
            "n_tokens": self._n_tokens,
            "build_sec": round(self._build_sec, 2),
        }