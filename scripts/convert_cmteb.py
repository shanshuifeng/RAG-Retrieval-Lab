"""C-MTEB/DuRetrieval → 统一 JSONL（test 集）—— M1.3 / M1.4

把已下载的 parquet 转成与 dev 完全相同的格式，使 dev / test 同构可比：
  data/processed/test/corpus.jsonl    {"id","text"}
  data/processed/test/queries.jsonl   {"qid","query","relevant_ids":[...]}
  data/processed/test/qrels.tsv       qid<TAB>pid<TAB>score
  data/processed/test/stats.json

⚠️ 纪律提醒：test 集一旦锁定，不得用于任何调参（docs/05 第五条）。
   本脚本只做格式转换与统计，不参与任何超参选择。

用法：
  .venv\\Scripts\\python.exe scripts\\convert_cmteb.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "dureval"
OUT = ROOT / "data" / "processed" / "test"
LOG = ROOT / "logs" / "convert_cmteb.log"


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hist(lens: list[int]) -> dict:
    if not lens:
        return {}
    s = sorted(lens)
    return {
        "min": s[0], "max": s[-1],
        "mean": round(mean(s), 2), "median": median(s),
        "p95": s[int(0.95 * (len(s) - 1))],
    }


def main() -> int:
    log("=" * 66)
    log("  C-MTEB/DuRetrieval → 统一 JSONL（test 集）")
    log("=" * 66)

    cf = sorted((RAW / "data").glob("corpus-*.parquet"))
    qf = sorted((RAW / "data").glob("queries-*.parquet"))
    rf = sorted((RAW / "qrels").rglob("*.parquet"))
    if not (cf and qf and rf):
        log(f"❌ 缺文件：corpus={len(cf)} queries={len(qf)} qrels={len(rf)}")
        return 1

    corpus = pd.read_parquet(cf[0])
    queries = pd.read_parquet(qf[0])
    qrels = pd.read_parquet(rf[0])
    log(f"corpus {len(corpus):,} 行，queries {len(queries):,} 行，qrels {len(qrels):,} 行")

    OUT.mkdir(parents=True, exist_ok=True)

    # ── corpus.jsonl ──
    lens: list[int] = []
    seen: set[str] = set()
    dup = 0
    with open(OUT / "corpus.jsonl", "w", encoding="utf-8") as f:
        for i, row in enumerate(corpus.itertuples(index=False)):
            d = str(getattr(row, "id"))
            t = getattr(row, "text") or ""
            if d in seen:
                dup += 1
                continue
            seen.add(d)
            lens.append(len(t))
            f.write(json.dumps({"id": d, "text": t}, ensure_ascii=False) + "\n")
            if i % 20000 == 0 and i:
                log(f"  corpus 已写 {i:,} ...")
    log(f"corpus 写出 {len(seen):,} 段（重复 {dup}）")

    # ── qrels 映射 ──
    qcol = next(c for c in ("qid", "query-id", "query_id") if c in qrels.columns)
    pcol = next(c for c in ("pid", "corpus-id", "corpus_id", "doc_id") if c in qrels.columns)
    scol = "score" if "score" in qrels.columns else None
    rel: dict[str, list[str]] = {}
    for row in qrels.itertuples(index=False):
        q = str(getattr(row, qcol))
        p = str(getattr(row, pcol))
        rel.setdefault(q, [])
        if p not in rel[q]:
            rel[q].append(p)

    # ── queries.jsonl + qrels.tsv ──
    rel_per_q: list[int] = []
    n_written = 0
    with open(OUT / "queries.jsonl", "w", encoding="utf-8") as fq, \
            open(OUT / "qrels.tsv", "w", encoding="utf-8") as fr:
        fr.write("# qid\tpid\tscore\n")
        for row in queries.itertuples(index=False):
            qid = str(getattr(row, "id"))
            text = getattr(row, "text") or ""
            ids = rel.get(qid, [])
            rel_per_q.append(len(ids))
            n_written += 1
            fq.write(json.dumps({"qid": qid, "query": text, "relevant_ids": ids},
                                ensure_ascii=False) + "\n")
            for p in ids:
                fr.write(f"{qid}\t{p}\t1\n")
    log(f"queries 写出 {n_written:,} 条")

    # ── 完整性校验 ──
    ref_ids = set().union(*[set(v) for v in rel.values()]) if rel else set()
    missing = ref_ids - seen
    log(f"⭐ qrels 引用 {len(ref_ids):,} 个 pid，其中不在语料中的：{len(missing):,}")
    if missing:
        log(f"  ⚠️ 示例：{sorted(missing)[:3]}")

    stats = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "dataset_id": "C-MTEB/DuRetrieval (+ C-MTEB/DuRetrieval-qrels)",
            "corpus_parquet_sha256": sha256_file(cf[0]),
            "queries_parquet_sha256": sha256_file(qf[0]),
            "qrels_parquet_sha256": sha256_file(rf[0]),
            "note": "⚠️ test 集：锁定后不得用于任何调参（docs/05）",
        },
        "test": {
            "queries": n_written,
            "qrels_rows": int(len(qrels)),
            "corpus_passages": len(seen),
            "corpus_duplicates": dup,
            "relevant_per_query": hist(rel_per_q),
            "zero_relevant_queries": sum(1 for x in rel_per_q if x == 0),
            "passage_len_chars": hist(lens),
            "qrels_pid_not_in_corpus": len(missing),
        },
    }
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    log("=" * 66)
    log(f"✅ test 转换完成 → {OUT.relative_to(ROOT)}")
    log(f"  queries {n_written:,} / corpus {len(seen):,} / qrels {len(qrels):,}")
    log(f"  每 query 相关数: {stats['test']['relevant_per_query']}")
    log(f"  段落长度(字)   : {stats['test']['passage_len_chars']}")
    log("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())