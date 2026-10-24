"""核验 ModelScope/HF 上的 DuReader-retrieval 训练与 dev 数据（M1 决策依据）

背景：
  现有 C-MTEB/DuRetrieval 只有 corpus(100,001) + queries(2,000)，
  没有 train split。用户决定「另下 ModelScope 训练集作 dev」。
  但探查发现 HF 镜像上有两个仓库：
    · zyznull/dureader-retrieval-ranking : train.jsonl.gz 3.4GB + dev.jsonl.gz 34MB
    · zyznull/dureader-retrieval-corpus  : passage_collection.tsv.gz 3.3GB + queries.*.tsv.gz

本脚本做的事（只下载小的 dev，不下载 3.4GB train）：
  1. 下载 dev.jsonl.gz（34MB）
  2. 统计 dev 的 query 数、正/负 passage 数分布
  3. ⭐ 关键：dev 的 docid 是否落在已下载的 C-MTEB 100,001 段落语料里？
  4. 用 HTTP Range 只取 train.jsonl.gz 的前几 MB，解出前若干行看结构
  5. 输出 JSON 结果供决策

用法：
  .venv\\Scripts\\python.exe scripts\\fetch_modelscope_dev.py
"""

from __future__ import annotations

import gzip
import json
import sys
import zlib
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT_DIR = RAW / "dureader-ranking"
DEV_GZ = OUT_DIR / "dev.jsonl.gz"
TRAIN_GZ = OUT_DIR / "train.jsonl.gz"
REPORT = OUT_DIR / "probe_report.json"

HF = "https://hf-mirror.com/datasets/zyznull/dureader-retrieval-ranking/resolve/main"

S = requests.Session()
S.headers.update({"User-Agent": "RAG-Retrieval-Lab/0.1"})


def hr(c: str = "=", w: int = 74) -> None:
    print(c * w)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download(url: str, dest: Path, limit: int | None = None) -> int:
    """流式下载；limit 为字节上限（用于只取开头）。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    if limit is not None:
        headers["Range"] = f"bytes=0-{limit - 1}"
    with S.get(url, stream=True, timeout=120, headers=headers) as r:
        if r.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {r.status_code} for {url}")
        total = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                total += len(chunk)
                print(f"\r    {human(total)}", end="", flush=True)
                if limit is not None and total >= limit:
                    break
    print(f"\r    {human(total)} 已保存 -> {dest.name}")
    return total


def load_corpus_ids() -> set[str]:
    files = sorted((RAW / "dureval" / "data").glob("corpus-*.parquet"))
    if not files:
        return set()
    df = pd.read_parquet(files[0], columns=["id"])
    return set(df["id"].astype(str))


def analyze_dev() -> dict:
    hr()
    print("  1. dev.jsonl.gz 分析")
    hr()
    n_lines = 0
    n_pos: list[int] = []
    n_neg: list[int] = []
    qids: list[str] = []
    docids: set[str] = set()
    pos_docids: set[str] = set()
    sample = None
    with gzip.open(DEV_GZ, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            n_lines += 1
            if sample is None:
                sample = {
                    "query_id": d.get("query_id"),
                    "query": d.get("query"),
                    "n_positive": len(d.get("positive_passages") or []),
                    "n_negative": len(d.get("negative_passages") or []),
                }
            pos = d.get("positive_passages") or []
            neg = d.get("negative_passages") or []
            n_pos.append(len(pos))
            n_neg.append(len(neg))
            qids.append(str(d.get("query_id")))
            for p in pos:
                if p and p.get("docid") is not None:
                    pos_docids.add(str(p["docid"]))
                    docids.add(str(p["docid"]))
            for p in neg:
                if p and p.get("docid") is not None:
                    docids.add(str(p["docid"]))

    print(f"  dev query 数        : {n_lines:,}")
    print(f"  dev 唯一 query_id 数: {len(set(qids)):,}")
    print(f"  正例数/query        : min={min(n_pos)} max={max(n_pos)} "
          f"avg={sum(n_pos)/len(n_pos):.2f}")
    c = Counter(n_pos)
    print(f"  正例数分布(前5)     : {c.most_common(5)}")
    print(f"  负例数/query        : min={min(n_neg)} max={max(n_neg)} "
          f"avg={sum(n_neg)/len(n_neg):.2f}")
    print(f"  出现过的唯一 docid  : {len(docids):,}（其中正例 {len(pos_docids):,}）")
    print(f"  首条示例            : {json.dumps(sample, ensure_ascii=False)}")

    corpus_ids = load_corpus_ids()
    print(f"\n  C-MTEB 语料唯一 id  : {len(corpus_ids):,}")
    if corpus_ids:
        inter_all = docids & corpus_ids
        inter_pos = pos_docids & corpus_ids
        print(f"  ⭐ dev 全部 docid 命中 C-MTEB 语料: {len(inter_all):,} / {len(docids):,}"
              f"  ({100*len(inter_all)/max(1,len(docids)):.1f}%)")
        print(f"  ⭐ dev 正例 docid 命中 C-MTEB 语料: {len(inter_pos):,} / {len(pos_docids):,}"
              f"  ({100*len(inter_pos)/max(1,len(pos_docids)):.1f}%)")

    return {
        "dev_queries": n_lines,
        "dev_unique_qids": len(set(qids)),
        "pos_per_query_avg": round(sum(n_pos) / len(n_pos), 3),
        "pos_per_query_max": max(n_pos),
        "neg_per_query_avg": round(sum(n_neg) / len(n_neg), 3),
        "neg_per_query_max": max(n_neg),
        "unique_docids": len(docids),
        "unique_pos_docids": len(pos_docids),
        "cmteb_corpus_ids": len(corpus_ids),
        "dev_docid_hit_cmteb": len(docids & corpus_ids) if corpus_ids else None,
        "dev_pos_docid_hit_cmteb": len(pos_docids & corpus_ids) if corpus_ids else None,
        "sample_record": sample,
    }


def peek_train(nbytes: int = 4 << 20) -> dict:
    hr()
    print(f"  2. train.jsonl.gz 头部 {human(nbytes)} 探针（不下载 3.4GB）")
    hr()
    tmp = OUT_DIR / "train_head.gz"
    got = download(f"{HF}/train.jsonl.gz", tmp, limit=nbytes)
    raw = tmp.read_bytes()
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    text = d.decompress(raw).decode("utf-8", errors="replace")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    print(f"  解出完整行数: {len(lines)}（末行可能被截断）")
    first = json.loads(lines[0])
    print(f"  train 字段: {list(first.keys())}")
    print(f"  train 首条 query: {first.get('query')!r}")
    print(f"  train 首条 正例数: {len(first.get('positive_passages') or [])}"
          f"  负例数: {len(first.get('negative_passages') or [])}")
    lens = []
    for ln in lines[:200]:
        try:
            j = json.loads(ln)
            lens.append((len(j.get('positive_passages') or []),
                         len(j.get('negative_passages') or [])))
        except Exception:  # noqa: BLE001
            pass
    if lens:
        p = [x[0] for x in lens]
        q = [x[1] for x in lens]
        print(f"  前 {len(lens)} 条: 正例 min={min(p)} max={max(p)} avg={sum(p)/len(p):.2f}"
              f" | 负例 min={min(q)} max={max(q)} avg={sum(q)/len(q):.2f}")
    return {
        "train_head_bytes": got,
        "train_fields": list(first.keys()),
        "train_first_query": first.get("query"),
        "train_head_pos_avg": round(sum(p) / len(p), 3) if lens else None,
        "train_head_neg_avg": round(sum(q) / len(q), 3) if lens else None,
    }


def main() -> int:
    hr("#")
    print("  DuReader-retrieval 训练/dev 数据核验")
    hr("#")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DEV_GZ.exists() or DEV_GZ.stat().st_size < 30_000_000:
        hr()
        print("  0. 下载 dev.jsonl.gz")
        hr()
        download(f"{HF}/dev.jsonl.gz", DEV_GZ)
    else:
        print(f"\n  已存在 dev.jsonl.gz（{human(DEV_GZ.stat().st_size)}），跳过下载")

    report = analyze_dev()
    report.update(peek_train())

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    hr("#")
    print(f"  报告已写入 {REPORT.relative_to(ROOT)}")
    hr("#")
    return 0


if __name__ == "__main__":
    sys.exit(main())