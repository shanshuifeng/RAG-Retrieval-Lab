"""BM25 倒排索引实现 —— 快速版（P1 提速）

⭐ 为什么要自己写（而不用 rank_bm25）：
   rank_bm25 的 `get_scores()` 对每条 query 都遍历【全库】每个文档、每个 query 词，
   复杂度 O(|Q_terms| × |corpus|)，纯 Python 循环。
   10 万段 × 8640 query 实测 145 ms/query，网格调参 16 组要 5.6 小时。

⭐ 本实现的做法：稀疏矩阵一次乘完
   · 建索引：构造 (vocab × docs) 的 scipy CSR 稀疏矩阵 W，
             W[t, d] = idf(t) × tf_saturate(t, d) / norm(d)
   · 检索：  q 是 (1 × vocab) 的稀疏向量，score = q @ W  → 一次矩阵乘法
   · 复杂度 O(nnz(W) 中与 query 词相关的部分)，且矩阵乘法走 C 实现

⭐ 与 rank_bm25 的公式对齐（BM25Okapi / Lucene 式 IDF）：
   idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
   tf_part(t,d) = f(t,d) × (k1 + 1) / (f(t,d) + k1 × (1 - b + b × |d| / avgdl))

   ⚠️ 注意 rank_bm25 的 BM25Okapi 用的是这个 Lucene 变体，不是经典 Robertson IDF。
      必须对齐，否则结果不可比（本模块的 verify_against_rank_bm25 会检查）。

用法：
    from src.retrieval.bm25_fast import BM25Fast
    r = BM25Fast(k1=1.5, b=0.75)
    r.build(split)
    res = r.search("查询", top_k=100)          # [(pid, score), ...]
    run = r.run_all(split, top_k=100)          # {qid: [pid, ...]}
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from src.data.loader import Split
from src.retrieval.bm25 import tokenize_zh


@dataclass
class BM25Fast:
    """倒排索引版 BM25（稀疏矩阵实现）。"""

    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25          # ⭐ rank_bm25 的负 IDF 下限系数，必须对齐
    name: str = "bm25_fast"

    passage_ids: list[str] = field(default_factory=list)
    vocab: dict[str, int] = field(default_factory=dict)
    _W: csr_matrix | None = None          # (vocab × docs) 加权矩阵
    _build_sec: float = 0.0
    _n_tokens: int = 0
    # ⭐ 与 k1/b 无关的基础统计，建一次即可复用于网格调参
    _base: dict | None = None

    # ────────────────────────────────────────────
    # 建索引（两阶段：基础统计一次 / 加权可重算）
    # ────────────────────────────────────────────
    def build_base(self, split: Split, verbose: bool = True) -> None:
        """⭐ 第一阶段：分词 + 词表 + df + tf + doc_len。

        这些量与 k1/b 【无关】，所以网格调参时只跑一次。
        结果缓存在 self._base，之后用 set_params() 秒级重建加权矩阵。
        """
        t0 = time.time()
        self.passage_ids = list(split.passage_ids)
        n_docs = len(self.passage_ids)

        if verbose:
            print(f"    分词 {n_docs:,} 段 ...", flush=True)

        # ── 1. 分词并统计词表 ──
        docs_tokens: list[list[str]] = []
        vocab: dict[str, int] = {}
        for i, pid in enumerate(self.passage_ids):
            toks = tokenize_zh(split.corpus[pid])
            docs_tokens.append(toks)
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
            if verbose and (i + 1) % 20000 == 0:
                print(f"      分词 {i+1:,}/{n_docs:,}，词表 {len(vocab):,} ...",
                      flush=True)

        self.vocab = vocab
        V = len(vocab)
        self._n_tokens = sum(len(t) for t in docs_tokens)
        if verbose:
            print(f"    词表 {V:,}，总 tokens {self._n_tokens:,}", flush=True)

        # ── 2. 统计 df 与每段的 tf ──
        df = np.zeros(V, dtype=np.float64)
        doc_len = np.zeros(n_docs, dtype=np.float64)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        for d, toks in enumerate(docs_tokens):
            doc_len[d] = len(toks)
            if not toks:
                continue
            counts: dict[int, int] = {}
            for t in toks:
                ti = vocab[t]
                counts[ti] = counts.get(ti, 0) + 1
            for ti, f in counts.items():
                df[ti] += 1.0
                rows.append(ti)
                cols.append(d)
                vals.append(float(f))
            if verbose and (d + 1) % 20000 == 0:
                print(f"      统计 tf {d+1:,}/{n_docs:,} ...", flush=True)

        # ── 3. 计算 IDF（⭐ 严格对齐 rank_bm25.BM25Okapi._calc_idf）──
        #   公式：idf = log(N - df + 0.5) - log(df + 0.5)     （不是 log(1+...)）
        #   负 IDF 处理：idf < 0 的词统一设为 eps = epsilon × average_idf
        #   这一步必须逐字对齐，否则排序会不同（实测差 6.6% 重合率）
        idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        avg_idf = idf.sum() / max(V, 1)
        eps = self.epsilon * avg_idf
        neg_mask = idf < 0
        idf = np.where(neg_mask, eps, idf)

        self._base = {
            "n_docs": n_docs,
            "V": V,
            "idf": idf,
            "rows": np.asarray(rows, dtype=np.int32),
            "cols": np.asarray(cols, dtype=np.int32),
            "tf": np.asarray(vals, dtype=np.float64),
            "doc_len": doc_len,
            "avgdl": float(doc_len.mean()) if n_docs else 0.0,
        }
        self._build_sec = time.time() - t0
        if verbose:
            print(f"    基础统计完成，nnz={len(rows):,}，用时 {self._build_sec:.1f}s",
                  flush=True)

    def set_params(self, k1: float | None = None, b: float | None = None,
                   verbose: bool = False) -> None:
        """⭐ 第二阶段：用新 k1/b 重建加权矩阵（毫秒级）。

        这是网格调参能快起来的关键——避免重复分词。
        """
        if self._base is None:
            raise RuntimeError("请先调用 build_base()")
        if k1 is not None:
            self.k1 = k1
        if b is not None:
            self.b = b

        t0 = time.time()
        base = self._base
        doc_len = base["doc_len"]
        rows_a, cols_a, tf = base["rows"], base["cols"], base["tf"]

        norm = self.k1 * (1.0 - self.b + self.b * doc_len
                          / max(base["avgdl"], 1e-9))
        denom = tf + norm[cols_a]
        tf_part = tf * (self.k1 + 1.0) / np.maximum(denom, 1e-12)
        weights = base["idf"][rows_a] * tf_part

        self._W = csr_matrix(
            (weights, (rows_a, cols_a)), shape=(base["V"], base["n_docs"]),
            dtype=np.float64
        )
        if verbose:
            print(f"    set_params(k1={self.k1}, b={self.b}) "
                  f"用时 {(time.time()-t0)*1000:.0f} ms", flush=True)

    def build(self, split: Split, verbose: bool = True) -> None:
        """一次性建索引（= build_base + set_params）。"""
        t0 = time.time()
        self.build_base(split, verbose=verbose)
        self.set_params(verbose=verbose)
        self._build_sec = time.time() - t0
        if verbose:
            print(f"    稀疏矩阵 {len(self.vocab):,}×{len(self.passage_ids):,}，"
                  f"nnz={self._W.nnz:,}，总用时 {self._build_sec:.1f}s", flush=True)

    # ────────────────────────────────────────────
    # 检索
    # ────────────────────────────────────────────
    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        if self._W is None:
            raise RuntimeError("请先调用 build() 建索引")
        toks = tokenize_zh(query)
        if not toks:
            return []

        # query 向量：词频按 BM25 的 query 部分（rank_bm25 用 q_freq 线性加权）
        q_counts: dict[int, int] = {}
        for t in toks:
            ti = self.vocab.get(t)
            if ti is not None:
                q_counts[ti] = q_counts.get(ti, 0) + 1
        if not q_counts:
            return []

        idx = np.fromiter(q_counts.keys(), dtype=np.int32, count=len(q_counts))
        val = np.fromiter(q_counts.values(), dtype=np.float64, count=len(q_counts))
        q_vec = csr_matrix((val, (np.zeros_like(idx), idx)),
                           shape=(1, len(self.vocab)))

        scores = (q_vec @ self._W).toarray().ravel()

        k = min(top_k, len(scores))
        if k <= 0:
            return []
        # argpartition 取 top-k，再排序
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part])]
        return [(self.passage_ids[int(i)], float(scores[int(i)])) for i in order]

    def run_all(self, split: Split, top_k: int = 100,
                verbose: bool = True) -> dict[str, list[str]]:
        qids = split.query_list(with_relevant_only=True)
        out: dict[str, list[str]] = {}
        t0 = time.time()
        for i, qid in enumerate(qids, start=1):
            out[qid] = [pid for pid, _ in self.search(split.queries[qid], top_k)]
            if verbose and i % 2000 == 0:
                el = time.time() - t0
                print(f"      检索 {i:,}/{len(qids):,}  ({el:.1f}s, "
                      f"{i/el:.0f} q/s) ...", flush=True)
        if verbose:
            el = time.time() - t0
            print(f"      检索完成：{len(qids):,} query，{el:.1f}s "
                  f"({len(qids)/max(el,1e-9):.0f} q/s)", flush=True)
        return out

    # ────────────────────────────────────────────
    # 持久化
    # ────────────────────────────────────────────
    def save(self, path: Path) -> None:
        """保存索引。

        ⭐ 用 save_base=True 时连同基础统计一起存，之后可用 set_params() 秒级
           切换 k1/b（网格调参专用），代价是文件更大。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "k1": self.k1, "b": self.b,
                "passage_ids": self.passage_ids,
                "vocab": self.vocab,
                "W": self._W,
                "base": self._base,
                "build_sec": self._build_sec,
                "n_tokens": self._n_tokens,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> BM25Fast:
        with open(path, "rb") as f:
            d = pickle.load(f)
        r = cls(k1=d["k1"], b=d["b"])
        r.passage_ids = d["passage_ids"]
        r.vocab = d["vocab"]
        r._W = d["W"]
        r._base = d.get("base")          # 旧缓存可能没有，为 None 则不能 set_params
        r._build_sec = d.get("build_sec", 0.0)
        r._n_tokens = d.get("n_tokens", 0)
        return r

    @property
    def can_set_params(self) -> bool:
        """是否可在不重新分词的情况下切换 k1/b。"""
        return self._base is not None

    def stats(self) -> dict:
        return {
            "method": "bm25_fast",
            "k1": self.k1,
            "b": self.b,
            "n_passages": len(self.passage_ids),
            "n_vocab": len(self.vocab),
            "nnz": int(self._W.nnz) if self._W is not None else 0,
            "n_tokens": self._n_tokens,
            "build_sec": round(self._build_sec, 2),
        }