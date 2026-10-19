"""统一数据加载层 —— 让 dev / test 用【完全相同】的代码读取

⭐ 设计原则：
   · dev 与 test 的目录结构、文件格式完全一致（见 data/DATACARD.md 第三节），
     因此本模块对两者一视同仁——避免「换数据集就改代码」导致的口径漂移。
   · 所有读取都做完整性自检（id 唯一、qrels 引用存在），失败立刻报错而非静默继续。
   · 只读，不修改任何数据文件。

用法：
    from src.data.loader import load_split
    split = load_split("dev")            # 默认读 data/processed/dev/
    split.corpus      # {pid: text}
    split.queries     # {qid: query_text}
    split.qrels       # {qid: [pid, ...]}
    split.passage_ids # [pid, ...]（保持文件顺序，用于构建索引）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = ROOT / "data" / "processed"


class DataError(RuntimeError):
    """数据完整性错误 —— 必须立刻停下，不能带着脏数据继续实验。"""


@dataclass
class Split:
    """一个数据划分（dev 或 test）。"""

    name: str
    corpus: dict[str, str] = field(default_factory=dict)
    queries: dict[str, str] = field(default_factory=dict)
    qrels: dict[str, list[str]] = field(default_factory=dict)
    passage_ids: list[str] = field(default_factory=list)

    # ── 便捷属性 ──
    @property
    def n_passages(self) -> int:
        return len(self.passage_ids)

    @property
    def n_queries(self) -> int:
        return len(self.queries)

    @property
    def n_relevant_pairs(self) -> int:
        return sum(len(v) for v in self.qrels.values())

    def relevant_of(self, qid: str) -> list[str]:
        return self.qrels.get(qid, [])

    def query_list(self, with_relevant_only: bool = True) -> list[str]:
        """返回 qid 列表；默认只保留有相关标注的（无标注无法算指标）。"""
        if with_relevant_only:
            return [q for q in self.queries if self.qrels.get(q)]
        return list(self.queries)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "passages": self.n_passages,
            "queries": self.n_queries,
            "queries_with_relevant": len(self.query_list(True)),
            "relevant_pairs": self.n_relevant_pairs,
        }

    def verify(self) -> None:
        """完整性自检；任何问题直接抛 DataError。

        ⭐ 为什么必须在加载时校验：qrels 指向不存在的段落会让 Recall 永久偏低，
           而这种错误【不会报错】——只会让指标悄悄变差，是最难查的一类 bug。
        """
        problems: list[str] = []

        if not self.passage_ids:
            problems.append("语料为空")
        elif len(self.passage_ids) != len(set(self.passage_ids)):
            dup = len(self.passage_ids) - len(set(self.passage_ids))
            problems.append(f"语料存在 {dup} 个重复 id")

        if not self.queries:
            problems.append("查询为空")

        missing_q = [q for q in self.qrels if q not in self.queries]
        if missing_q:
            problems.append(f"{len(missing_q)} 个 qrels 的 qid 不在查询集（例：{missing_q[:3]}）")

        pid_set = set(self.passage_ids)
        dangling = 0
        samples: list[str] = []
        for qid, pids in self.qrels.items():
            for p in pids:
                if p not in pid_set:
                    dangling += 1
                    if len(samples) < 3:
                        samples.append(f"{qid}->{p}")
        if dangling:
            problems.append(f"{dangling} 条 qrels 指向不存在的段落（例：{samples}）")

        empty_rel = [q for q in self.queries if not self.qrels.get(q)]
        if empty_rel:
            # 不是错误（评测层会过滤），但必须知情
            problems.append(
                f"警告：{len(empty_rel)} 个 query 无相关标注（评测时将被过滤）"
            )

        hard = [p for p in problems if not p.startswith("警告")]
        if hard:
            raise DataError(f"[{self.name}] 数据完整性校验失败：\n  - "
                            + "\n  - ".join(problems))
        for p in problems:
            print(f"  ⚠️ [{self.name}] {p}")


def _iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise DataError(f"{path.name} 第 {lineno} 行 JSON 解析失败：{e}") from e


def load_split(name: str = "dev", base: Path | None = None,
               verify: bool = True) -> Split:
    """加载一个数据划分。

    Args:
        name: "dev" 或 "test"（也可以是 data/processed 下任意子目录名）
        base: 覆盖默认的 data/processed
        verify: 是否做完整性自检（默认 True；强烈建议保持开启）
    """
    root = (base or PROCESSED) / name
    if not root.is_dir():
        raise DataError(f"找不到数据目录：{root}")

    split = Split(name=name)

    corpus_path = root / "corpus.jsonl"
    if not corpus_path.exists():
        raise DataError(f"缺少 {corpus_path}")
    for rec in _iter_jsonl(corpus_path):
        pid = str(rec["id"])
        split.passage_ids.append(pid)
        split.corpus[pid] = rec.get("text") or ""

    queries_path = root / "queries.jsonl"
    if not queries_path.exists():
        raise DataError(f"缺少 {queries_path}")
    for rec in _iter_jsonl(queries_path):
        qid = str(rec["qid"])
        split.queries[qid] = rec.get("query") or ""
        rel = [str(x) for x in (rec.get("relevant_ids") or [])]
        if rel:
            split.qrels[qid] = rel

    # qrels.tsv 若存在则交叉校验（应与 queries.jsonl 中的 relevant_ids 一致）
    qrels_path = root / "qrels.tsv"
    if qrels_path.exists():
        from_tsv: dict[str, list[str]] = {}
        with open(qrels_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                q, p = parts[0], parts[1]
                from_tsv.setdefault(q, []).append(p)
        if from_tsv and set(from_tsv) != set(split.qrels):
            raise DataError(
                f"[{name}] qrels.tsv 与 queries.jsonl 的 query 集合不一致："
                f"仅 TSV 有 {len(set(from_tsv) - set(split.qrels))} 个，"
                f"仅 JSONL 有 {len(set(split.qrels) - set(from_tsv))} 个"
            )
        for q, pids in from_tsv.items():
            if set(pids) != set(split.qrels.get(q, [])):
                raise DataError(f"[{name}] query {q} 的 qrels.tsv 与 JSONL 标注不一致")
        split.n_relevant_pairs  # noqa: B018  (触发属性计算，无副作用)

    if verify:
        split.verify()
    return split


def load_run(path: Path) -> dict[str, list[str]]:
    """读取检索结果文件（{qid: [pid, ...]}，保持分数降序）。"""
    if not path.exists():
        raise DataError(f"找不到结果文件：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DataError(f"{path} 应是 {{qid: [pid,...]}} 结构")
    return {str(k): [str(x) for x in v] for k, v in data.items()}


__all__ = ["Split", "DataError", "load_split", "load_run", "PROCESSED"]