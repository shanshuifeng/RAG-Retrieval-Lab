"""分块策略 —— E4 分块消融（docs/06 §四）

⭐ 回答的问题（Q4）：文档怎么切最好？
   段落长短不一（实测 1 ~ 87,586 字），长段落信息密度低、
   且稠密模型 max_length=512 tokens 会截断 → 分块可能改善检索。

⭐ 本项目比较的策略（docs/04 §4.2）：
   A. 不切分（原生 passage）—— E1/E2/E3 的基线，必须与论文可比
   B. 固定长度：256 字 + 重叠 50 字
   C. 固定长度：512 字 + 重叠 50 字
   D. 按段落：以 \\n\\n 切分，超长段落再按 512 字兜底切

⭐⭐ 最大的坑：chunk id 与 qrels 的映射
   分块后检索单位从「段落」变成「块」：
     · 块 id 形如 {pid}#c{序号}
     · qrels 标注的是【段落级】相关 → 评评测时必须把块映射回父段落
     · 判定规则：任一 chunk 被检索到 = 父段落被召回
   映射错一位，指标就全错。因此本模块的映射有独立测试保障。

⚠️ 纪律：test 集已锁定（TEST_USED.lock），E4 只能在 dev 上做。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.data.loader import Split

PARA_SPLIT_RE = re.compile(r"\n\s*\n+")


# ────────────────────────────────────────────
# 分块策略
# ────────────────────────────────────────────
def chunk_fixed(text: str, size: int = 256, overlap: int = 50) -> list[str]:
    """固定长度 + 重叠。短文本（<= size）保持单块。"""
    if size <= 0:
        raise ValueError(f"size 必须 > 0，收到 {size}")
    if overlap >= size:
        raise ValueError(f"overlap({overlap}) 必须小于 size({size})")
    if len(text) <= size:
        return [text] if text.strip() else []
    step = size - overlap
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + size]
        if chunk.strip():
            chunks.append(chunk)
        if start + size >= len(text):
            break
        start += step
    return chunks


def chunk_paragraph(text: str, max_len: int = 512) -> list[str]:
    """按段落（\\n\\n）切；超过 max_len 的段落再按固定长度兜底切。"""
    if not text.strip():
        return []
    paras = [p for p in PARA_SPLIT_RE.split(text) if p.strip()]
    chunks: list[str] = []
    for p in paras:
        if len(p) <= max_len:
            chunks.append(p)
        else:
            # 超长段落按固定长度兜底（重叠取 max_len 的 10%）
            chunks.extend(chunk_fixed(p, size=max_len, overlap=max_len // 10))
    return chunks


def chunk_native(text: str) -> list[str]:
    """策略 A：不切分。"""
    return [text] if text.strip() else []


STRATEGIES = {
    "native": lambda: (lambda t: chunk_native(t)),
    "fixed256": lambda: (lambda t: chunk_fixed(t, size=256, overlap=50)),
    "fixed512": lambda: (lambda t: chunk_fixed(t, size=512, overlap=50)),
    "paragraph": lambda: (lambda t: chunk_paragraph(t, max_len=512)),
}


# ────────────────────────────────────────────
# 分块语料（含映射）
# ────────────────────────────────────────────
@dataclass
class ChunkedCorpus:
    """分块后的语料 + 块到父段落的映射。"""
    corpus: dict[str, str] = field(default_factory=dict)       # chunk_id → text
    passage_ids: list[str] = field(default_factory=list)       # 保持构建顺序
    chunk_to_parent: dict[str, str] = field(default_factory=dict)  # chunk_id → pid
    parent_chunks: dict[str, list[str]] = field(default_factory=dict)  # pid → [chunk_id]
    strategy: str = ""
    n_parents: int = 0

    def project_ranking(self, ranked_chunk_ids: list[str]) -> list[str]:
        """⭐ 把块级排名投影回段落级排名（去重保序）。

        规则：块在排名里第一次出现时，其父段落获得该位置；
        同一父段落的后续块不再占位。
        """
        seen: set[str] = set()
        out: list[str] = []
        for cid in ranked_chunk_ids:
            pid = self.chunk_to_parent.get(cid)
            if pid is None:
                # 未知的块 id → 映射缺失，必须显式报错而不是静默跳过
                raise KeyError(f"chunk id {cid} 不在映射表里（映射表损坏？）")
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
        return out

    def stats(self) -> dict:
        lens = [len(t) for t in self.corpus.values()]
        lens.sort()
        n = len(lens)
        chunks_per_parent = [len(v) for v in self.parent_chunks.values()]
        return {
            "strategy": self.strategy,
            "n_parents": self.n_parents,
            "n_chunks": len(self.corpus),
            "chunks_per_parent_mean": round(sum(chunks_per_parent) / max(self.n_parents, 1), 2),
            "chunks_per_parent_max": max(chunks_per_parent) if chunks_per_parent else 0,
            "chunk_len_mean": round(sum(lens) / max(n, 1), 1),
            "chunk_len_median": lens[n // 2] if n else 0,
            "chunk_len_p95": lens[int(0.95 * (n - 1))] if n else 0,
            "chunk_len_max": lens[-1] if n else 0,
        }


def build_chunked_corpus(split: Split, strategy: str,
                         verbose: bool = True) -> ChunkedCorpus:
    """把 Split 的语料按策略分块，返回 ChunkedCorpus。"""
    if strategy not in STRATEGIES:
        raise ValueError(f"未知策略 {strategy}，可选：{list(STRATEGIES)}")
    fn = STRATEGIES[strategy]()
    out = ChunkedCorpus(strategy=strategy)
    out.n_parents = split.n_passages

    for i, pid in enumerate(split.passage_ids):
        chunks = fn(split.corpus[pid])
        for j, text in enumerate(chunks):
            cid = f"{pid}#c{j}"
            out.corpus[cid] = text
            out.passage_ids.append(cid)
            out.chunk_to_parent[cid] = pid
            out.parent_chunks.setdefault(pid, []).append(cid)
        if verbose and (i + 1) % 20000 == 0:
            print(f"    分块 {i+1:,}/{split.n_passages:,} ...", flush=True)

    if verbose:
        st = out.stats()
        print(f"    策略 {strategy}：{st['n_parents']:,} 段 → {st['n_chunks']:,} 块"
              f"（均值 {st['chunks_per_parent_mean']} 块/段，"
              f"块长中位 {st['chunk_len_median']} 字）", flush=True)
    return out


def chunked_split_view(split: Split, cc: ChunkedCorpus) -> Split:
    """构造一个"分块版 Split"视图，供检索器直接 build。

    ⚠️ queries/qrels 保持【段落级】——检索在块级，评测时投影回段落级。
    """
    view = Split(name=f"{split.name}_{cc.strategy}")
    view.corpus = cc.corpus
    view.passage_ids = cc.passage_ids
    view.queries = split.queries
    view.qrels = split.qrels          # 段落级标注，评测时投影
    return view
