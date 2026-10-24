"""Cross-encoder 重排序器 —— E3 核心（docs/06 §三）

⭐ 为什么必须两阶段（docs/06 §3.3）：
   第一段（bi-encoder 稠密检索）：query 和 passage 各自独立编码，快但精度有限
   第二段（cross-encoder 重排）：query 和 passage 拼接进同一模型，慢但精度高

   bi-encoder：q → [enc] → vq ;  p → [enc] → vp ;  相似度 = vq·vp   （不能交互）
   cross-enc： [CLS] q [SEP] p [SEP] → [enc] → 分数                  （深度交互）

⭐ 成本结构（这正是 E3 要量化的）：
   第一段：N 段编码一次（离线） + 每 query 一次编码 + 一次向量检索 → 极快
   第二段：每 query 要对 top-K 候选【逐个】过一遍 cross-encoder → K 次前向
   → K 越大越好（候选多），但延迟线性增长 → 效果-成本权衡是 E3 的主问题

用法：
    from src.rerank.cross_encoder import CrossEncoderReranker
    r = CrossEncoderReranker("BAAI/bge-reranker-base")
    scores = r.score("查询", ["段落1", "段落2"])   # [s1, s2]
    reranked = r.rerank("查询", [(pid, text), ...])  # 按分数降序
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class RerankResult:
    """一次重排的结果与成本。"""
    qid: str
    ranked: list[str]              # 重排后的 pid 序列（分数降序）
    scores: dict[str, float]       # pid → cross-encoder 分数
    latency_ms: float              # 该 query 的重排耗时
    n_candidates: int              # 重排的候选数


@dataclass
class CrossEncoderReranker:
    """BGE cross-encoder 重排器。"""

    model_name: str = "BAAI/bge-reranker-base"
    max_length: int = 512
    batch_size: int = 32
    use_fp16: bool = True
    cache_folder: str = "models"
    name: str = "reranker"

    _tokenizer: AutoTokenizer | None = None
    _model: AutoModelForSequenceClassification | None = None
    _device: str = "cpu"
    _load_sec: float = 0.0

    def _load(self) -> None:
        if self._model is not None:
            return
        t0 = time.time()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_folder)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, cache_dir=self.cache_folder)
        self._model.to(self._device)
        if self.use_fp16 and self._device == "cuda":
            self._model.half()
        self._model.eval()
        self._load_sec = time.time() - t0

    @torch.no_grad()
    def score(self, query: str, passages: list[str],
              show_progress: bool = False) -> list[float]:
        """对 (query, passage) 对逐个打分，返回与 passages 同序的分数列表。"""
        self._load()
        assert self._tokenizer is not None and self._model is not None
        out: list[float] = []
        for s in range(0, len(passages), self.batch_size):
            chunk = passages[s:s + self.batch_size]
            pairs = [[query, p] for p in chunk]
            feats = self._tokenizer(
                pairs, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self._device)
            logits = self._model(**feats).logits
            scores = logits.view(-1).float().cpu().tolist()
            out.extend(scores)
            if show_progress:
                print(f"      打分 {min(s+self.batch_size, len(passages))}"
                      f"/{len(passages)} ...", flush=True)
        return out

    def rerank(self, qid: str, query: str,
               candidates: list[tuple[str, str]]) -> RerankResult:
        """对单个 query 的候选段落重排。

        Args:
            candidates: [(pid, passage_text), ...]（第一段检索的输出）

        Returns:
            RerankResult（含重排顺序 + 每条分数 + 耗时）
        """
        t0 = time.perf_counter()
        pids = [p for p, _ in candidates]
        texts = [t for _, t in candidates]
        scores = self.score(query, texts)
        # 分数降序；并列按 pid 字典序（可复现）
        order = sorted(zip(pids, scores), key=lambda kv: (-kv[1], kv[0]))
        return RerankResult(
            qid=qid,
            ranked=[p for p, _ in order],
            scores=dict(zip(pids, scores)),
            latency_ms=(time.perf_counter() - t0) * 1000,
            n_candidates=len(candidates),
        )

    def stats(self) -> dict:
        return {
            "method": "rerank",
            "model": self.model_name,
            "device": self._device,
            "fp16": self.use_fp16 and self._device == "cuda",
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "load_sec": round(self._load_sec, 2),
        }