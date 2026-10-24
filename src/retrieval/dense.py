"""稠密向量检索器 —— E1 的 D1/D2 组

⭐ 核心设计（对应 docs/06 §1.5）：
   离线：passage → 编码器 → 向量 → 归一化 → FAISS 内积索引
   在线：query → 编码器 → 向量 → 归一化 → FAISS Top-K

⭐ 关键实现细节（这些都影响指标可比性）：

  1. **归一化 + 内积 = 余弦相似度**
     BGE 官方建议先 L2 归一化，再用内积（等价于余弦）。
     用 faiss.IndexFlatIP 而非 IndexFlatL2，保证与其他方法口径一致。

  2. **BGE 的 query 指令前缀**
     BGE 官方文档建议对 query 加指令：
       "为这个句子生成表示以用于检索相关文章："
     ⭐ 但本项目实测表明**不加更好**（见下方 BGE_QUERY_INSTRUCTION 注释）：
        BGE-small +1.30%、BGE-base +2.56%，均显著。
     故 DEFAULT_QUERY_INSTRUCTION = ""，并通过
     scripts/ablate_query_instruction.py 可复现该结论。

  3. **max_seq_length = 512 tokens**
     超过会被截断 → 长段落信息丢失。这正是 docs/04 指出的
     "段落最长 87,586 字" 带来的隐患，需在报告中作为局限声明。

  4. **批量编码 + 半精度（GPU）**
     RTX 4060 8GB 用 fp16 可显著提速，且对检索指标影响 <0.1%（需实测验证）。

用法：
    from src.retrieval.dense import DenseRetriever
    r = DenseRetriever("BAAI/bge-small-zh-v1.5")
    r.build(split)                      # 编码全库 + 建索引
    res = r.search("查询", top_k=100)   # [(pid, score), ...]
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.data.loader import Split

# ⭐ BGE 中文模型的检索指令前缀
#
# ⚠️ 2026-10-24 消融实验的重要发现：**本项目默认【不启用】该指令**。
#
#    BGE 官方文档建议检索时给 query 加此前缀；v1.5 模型卡则说明
#    「v1.5 版本即使不加指令也不会有明显性能下降」。
#    实测（dev 8,640 query，100,000 段）表明本数据上**不加更好**：
#
#      BGE-small: nDCG@10  0.723848（带） vs 0.733224（不带）  +1.30%
#                 p = 8.0e-13, CI [+0.0069, +0.0120] → 显著
#      BGE-base : nDCG@10  0.793555（带） vs 0.813905（不带）  +2.56%
#                 p = 3.8e-51, CI [+0.0177, +0.0230] → 显著
#
#    ⇒ 故 DEFAULT_QUERY_INSTRUCTION = ""（空），并且这是一个
#      【必须写进报告】的负面结果：官方推荐的默认做法，
#      在特定数据/评测口径下未必最优。
#
#    复现：scripts/ablate_query_instruction.py
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
DEFAULT_QUERY_INSTRUCTION = ""      # ⭐ 实测最优（见上方说明）


@dataclass
class DenseRetriever:
    """稠密检索器（BGE + FAISS）。"""

    model_name: str = "BAAI/bge-small-zh-v1.5"
    batch_size: int = 64
    max_length: int = 512
    use_fp16: bool = True
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    cache_folder: str = "models"
    name: str = "dense"

    passage_ids: list[str] = field(default_factory=list)
    _model: SentenceTransformer | None = None
    _index: faiss.Index | None = None
    _dim: int = 0
    _build_sec: float = 0.0
    _encode_sec: float = 0.0

    # ────────────────────────────────────────────
    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(
                self.model_name, device=device, cache_folder=self.cache_folder
            )
            self._model.max_seq_length = self.max_length
            if self.use_fp16 and device == "cuda":
                self._model.half()
            self._dim = self._model.get_embedding_dimension()
        return self._model

    def _encode(self, texts: list[str], desc: str = "") -> np.ndarray:
        """编码并 L2 归一化（归一化后内积 = 余弦相似度）。

        ⚠️ sentence-transformers 6.x 的 encode() 不再接受 desc 参数
           （只有模型声明了 model_kwargs 才允许额外 kwargs），故这里不用。
        """
        m = self._load_model()
        vecs = m.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,     # ⭐ 关键：归一化
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)

    # ────────────────────────────────────────────
    def build(self, split: Split, verbose: bool = True) -> None:
        """编码全部段落并建 FAISS 内积索引。"""
        t0 = time.time()
        self.passage_ids = list(split.passage_ids)
        texts = [split.corpus[pid] for pid in self.passage_ids]

        if verbose:
            m = self._load_model()
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"    模型 {self.model_name} | 设备 {dev} | dim {self._dim} "
                  f"| fp16 {self.use_fp16 and dev == 'cuda'}", flush=True)
            print(f"    编码 {len(texts):,} 段（batch={self.batch_size}, "
                  f"max_len={self.max_length}）...", flush=True)

        t_enc = time.time()
        vecs = self._encode(texts, desc="passages")
        self._encode_sec = time.time() - t_enc

        # ⭐ 内积索引（向量已归一化，内积即余弦）
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(vecs)
        self._build_sec = time.time() - t0

        if verbose:
            n = self._index.ntotal
            print(f"    编码完成 {self._encode_sec:.1f}s "
                  f"({len(texts)/max(self._encode_sec,1e-9):.0f} 段/s)", flush=True)
            print(f"    FAISS IndexFlatIP：{n:,} 条向量，dim={self._dim}，"
                  f"总用时 {self._build_sec:.1f}s", flush=True)

    # ────────────────────────────────────────────
    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        if self._index is None:
            raise RuntimeError("请先调用 build() 建索引")
        q = self.query_instruction + query if self.query_instruction else query
        qv = self._encode([q])
        k = min(top_k, self._index.ntotal)
        scores, idx = self._index.search(qv, k)
        return [(self.passage_ids[int(i)], float(s))
                for i, s in zip(idx[0], scores[0]) if i >= 0]

    def search_batch(self, queries: list[str], top_k: int = 100,
                     verbose: bool = False) -> list[list[tuple[str, float]]]:
        """批量检索（比逐条快很多：一次编码多条 query）。"""
        if self._index is None:
            raise RuntimeError("请先调用 build() 建索引")
        qs = ([self.query_instruction + q for q in queries]
              if self.query_instruction else list(queries))
        vecs = self._encode(qs)
        k = min(top_k, self._index.ntotal)
        scores, idx = self._index.search(vecs, k)
        out = []
        for i in range(len(queries)):
            out.append([(self.passage_ids[int(j)], float(s))
                        for j, s in zip(idx[i], scores[i]) if j >= 0])
        if verbose:
            print(f"      批量检索 {len(queries):,} query 完成", flush=True)
        return out

    def run_all(self, split: Split, top_k: int = 100,
                verbose: bool = True, batch: int = 256) -> dict[str, list[str]]:
        """对全部有标注 query 批量检索，返回 {qid: [pid, ...]}。"""
        qids = split.query_list(with_relevant_only=True)
        out: dict[str, list[str]] = {}
        t0 = time.time()
        for s in range(0, len(qids), batch):
            chunk = qids[s:s + batch]
            texts = [split.queries[q] for q in chunk]
            res = self.search_batch(texts, top_k=top_k)
            for qid, hits in zip(chunk, res):
                out[qid] = [pid for pid, _ in hits]
            if verbose and (s // batch + 1) % 5 == 0:
                el = time.time() - t0
                print(f"      检索 {len(out):,}/{len(qids):,}  ({el:.1f}s, "
                      f"{len(out)/el:.0f} q/s) ...", flush=True)
        if verbose:
            el = time.time() - t0
            print(f"      检索完成：{len(qids):,} query，{el:.1f}s "
                  f"({len(qids)/max(el,1e-9):.0f} q/s)", flush=True)
        return out

    # ────────────────────────────────────────────
    # 持久化：向量索引单独存（模型不存，按名字重新加载）
    # ────────────────────────────────────────────
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "use_fp16": self.use_fp16,
            "query_instruction": self.query_instruction,
            "passage_ids": self.passage_ids,
            "dim": self._dim,
            "build_sec": self._build_sec,
            "encode_sec": self._encode_sec,
        }
        # FAISS 索引与元数据分开存
        idx_path = path.with_suffix(".faiss")
        faiss.write_index(self._index, str(idx_path))
        payload["index_file"] = idx_path.name
        path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

    @classmethod
    def load(cls, path: Path) -> DenseRetriever:
        payload = pickle.loads(path.read_bytes())
        r = cls(
            model_name=payload["model_name"],
            batch_size=payload.get("batch_size", 64),
            max_length=payload.get("max_length", 512),
            use_fp16=payload.get("use_fp16", True),
            query_instruction=payload.get("query_instruction", ""),
        )
        r.passage_ids = payload["passage_ids"]
        r._dim = payload["dim"]
        r._build_sec = payload.get("build_sec", 0.0)
        r._encode_sec = payload.get("encode_sec", 0.0)
        idx_path = path.parent / payload["index_file"]
        r._index = faiss.read_index(str(idx_path))
        return r

    def stats(self) -> dict:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return {
            "method": "dense",
            "model": self.model_name,
            "device": dev,
            "fp16": self.use_fp16 and dev == "cuda",
            "max_length": self.max_length,
            "dim": self._dim,
            "n_passages": len(self.passage_ids),
            "n_vectors": int(self._index.ntotal) if self._index else 0,
            "build_sec": round(self._build_sec, 2),
            "encode_sec": round(self._encode_sec, 2),
        }