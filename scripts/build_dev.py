"""构建 dev 集（D2 · 语料库级 dev，与 test 同构）—— M1.3 / M1.4 / M1.5

设计（2026-10-17 与用户确认）：
  · dev query 来源：ModelScope/HF `zyznull/dureader-retrieval-ranking` 的 train.jsonl.gz
    （86,395 条）。⚠️ 该仓库的 dev.jsonl.gz 已核实为【官方测试集】（与 C-MTEB 的
    2,000 query 文本交集 100%），绝不可用作 dev。
  · 抽样规则：按文件顺序【每 10 条取 1 条】→ 8,640 条（10.0%）。确定性、可复现。
  · dev qrels = 这些 query 的 positive_passages 全部保留（保证相关段落必在库内）
  · dev 语料 = 上一步的正例（约 2.5 万段） + 从这些 query 的 negative_passages 中
    用固定种子 42 采样补齐到【10 万段】——与 test 的 C-MTEB 语料（100,001 段）同规模。
  · ⚠️ 必须声明的非对称：dev 每 query 平均正例 2.89 条，test 为 4.92 条；
    故 dev 与 test 的【绝对】指标不可直接比较，dev 仅用于调参与方法横比。

两遍扫描（单遍需要把 86 万条负例文本全驻内存，不划算）：
  第 1 遍：抽出 dev query → 写 queries.jsonl / qrels.tsv；收集 正例 id 集、负例 id 集
  采样   ：seed=42 从负例池抽 k 条（k = 10 万 − 正例数）
  第 2 遍：写 corpus.jsonl（正例全收 + 被抽中的负例），按 id 去重

产出：
  data/processed/dev/corpus.jsonl      {"id","text"}
  data/processed/dev/queries.jsonl     {"qid","query","relevant_ids":[...]}
  data/processed/dev/qrels.tsv         qid<TAB>pid<TAB>1
  data/processed/dev/stats.json        统计（供 DATACARD 使用）
  data/processed/dev/build_manifest.json

用法：
  .venv\\Scripts\\python.exe scripts\\build_dev.py
  .venv\\Scripts\\python.exe scripts\\build_dev.py --every 10 --corpus-size 100000
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "dureader-ranking" / "train.jsonl.gz"
OUT = ROOT / "data" / "processed" / "dev"
LOG = ROOT / "logs" / "build_dev.log"

EVERY = 10
CORPUS_SIZE = 100_000
SEED = 42


def log(msg: str, quiet: bool = False) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    if not quiet:
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


def iter_records(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            yield i, json.loads(line)


def pass1(every: int) -> dict:
    log(f"第 1 遍：扫描 train，每 {every} 条取 1 条作为 dev query ...")
    OUT.mkdir(parents=True, exist_ok=True)
    q_path = OUT / "queries.jsonl"
    r_path = OUT / "qrels.tsv"

    pos_ids: set[str] = set()
    neg_ids: set[str] = set()
    n_records = 0
    n_dev = 0
    n_rel_rows = 0
    pos_per_q: list[int] = []
    # 记录每个 dev query 的正例，用于校验「正例全部落地」
    per_q_pos: dict[str, set[str]] = {}

    with open(q_path, "w", encoding="utf-8") as fq, \
            open(r_path, "w", encoding="utf-8") as fr:
        fr.write("# qid\tpid\tscore\n")
        for i, rec in iter_records(SRC):
            n_records += 1
            if i % every != 0:
                # 仍要收集负例池（来自所有 train query，负例池越大越有代表性）
                for p in rec.get("negative_passages") or []:
                    if p and p.get("docid") is not None:
                        neg_ids.add(str(p["docid"]))
                continue

            qid = str(rec.get("query_id"))
            query = rec.get("query") or ""
            positives = rec.get("positive_passages") or []
            negatives = rec.get("negative_passages") or []
            rel = []
            for p in positives:
                if p and p.get("docid") is not None:
                    d = str(p["docid"])
                    if d not in rel:
                        rel.append(d)
                    pos_ids.add(d)
            for p in negatives:
                if p and p.get("docid") is not None:
                    neg_ids.add(str(p["docid"]))

            per_q_pos[qid] = set(rel)
            pos_per_q.append(len(rel))
            n_dev += 1
            n_rel_rows += len(rel)
            fq.write(json.dumps({"qid": qid, "query": query, "relevant_ids": rel},
                                ensure_ascii=False) + "\n")
            for d in rel:
                fr.write(f"{qid}\t{d}\t1\n")
            if n_dev % 1000 == 0:
                log(f"  dev query {n_dev:,} / 记录 {n_records:,} ...", quiet=True)

    log(f"第 1 遍完成：train 记录 {n_records:,} 条，dev query {n_dev:,} 条，"
        f"qrels 行 {n_rel_rows:,}")
    log(f"  正例唯一 id {len(pos_ids):,}；负例 id 池 {len(neg_ids):,}")
    return {
        "n_records": n_records,
        "n_dev": n_dev,
        "n_rel_rows": n_rel_rows,
        "pos_ids": pos_ids,
        "neg_ids": neg_ids - pos_ids,
        "pos_per_q": pos_per_q,
        "pos_per_q_list": pos_per_q,
    }


def pass2(pos_ids: set[str], neg_pool: set[str], corpus_size: int, seed: int,
          every: int) -> dict:
    k_pos = len(pos_ids)
    k_neg = max(0, corpus_size - k_pos)
    if k_neg > len(neg_pool):
        log(f"⚠️ 负例池不足：需要 {k_neg:,}，只有 {len(neg_pool):,}，语料将小于目标")
        k_neg = len(neg_pool)

    log(f"采样负例：从 {len(neg_pool):,} 条中抽 {k_neg:,} 条（seed={seed}）")
    rng = random.Random(seed)
    neg_sorted = sorted(neg_pool)
    picked = set(rng.sample(neg_sorted, k_neg))
    log(f"  抽中 {len(picked):,} 条（其中与正例重叠 0 条：已预先扣除）")

    c_path = OUT / "corpus.jsonl"
    written: set[str] = set()
    n = 0
    lens: list[int] = []
    t0 = time.time()
    with open(c_path, "w", encoding="utf-8") as fc:
        for i, rec in iter_records(SRC):
            items = []
            # ⚠️ 只收 dev query 的正例；非 dev 记录的正例不得入语料（否则语料规模失控）
            if i % every == 0:
                for p in (rec.get("positive_passages") or []):
                    if p and p.get("docid") is not None:
                        items.append((str(p["docid"]), p.get("text") or ""))
            for p in (rec.get("negative_passages") or []):
                if p and p.get("docid") is not None:
                    d = str(p["docid"])
                    if d in picked:
                        items.append((d, p.get("text") or ""))
            for d, t in items:
                if d in written:
                    continue
                written.add(d)
                lens.append(len(t))
                fc.write(json.dumps({"id": d, "text": t}, ensure_ascii=False) + "\n")
                n += 1
            if i % 10000 == 0 and i:
                log(f"  已写 {n:,} 段 ...", quiet=True)
    log(f"第 2 遍完成：语料 {n:,} 段（用时 {time.time()-t0:.0f}s）")

    missing_pos = pos_ids - written
    if missing_pos:
        log(f"⚠️ 有 {len(missing_pos):,} 个正例未写入语料！")
    return {"n_corpus": n, "missing_pos": len(missing_pos), "lens": lens}


def summarize(stats: dict, p1: dict, p2: dict, corpus_size: int, every: int,
              src_sha: str) -> dict:
    lens = sorted(p2["lens"])
    plens = p1["pos_per_q_list"]
    dens = lambda l: {  # noqa: E731
        "min": min(l), "max": max(l),
        "mean": round(mean(l), 2), "median": median(l),
        "p95": l[int(0.95 * (len(l) - 1))],
    }
    doc = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "file": str(SRC.relative_to(ROOT)),
            "dataset_id": "zyznull/dureader-retrieval-ranking (hf-mirror)",
            "sha256": src_sha,
            "train_records": p1["n_records"],
            "note": "该仓库 dev.jsonl.gz 已核实为官方测试集，不可用作 dev",
        },
        "selection": {
            "rule": f"按文件顺序每 {every} 条取 1 条",
            "dev_queries": p1["n_dev"],
            "fraction": round(p1["n_dev"] / p1["n_records"], 4),
            "seed_for_negatives": SEED,
        },
        "dev": {
            "queries": p1["n_dev"],
            "qrels_rows": p1["n_rel_rows"],
            "corpus_passages": p2["n_corpus"],
            "corpus_target": corpus_size,
            "missing_positive_passages": p2["missing_pos"],
            "relevant_per_query": dens(plens),
            "zero_relevant_queries": sum(1 for x in plens if x == 0),
            "passage_len_chars": dens(lens),
            "total_chars": sum(lens),
        },
        "asymmetry_vs_test": {
            "dev_relevant_per_query_mean": round(mean(plens), 2),
            "test_relevant_per_query_mean": 4.92,
            "dev_passages": p2["n_corpus"],
            "test_passages": 100001,
            "warning": "正例数不同 → dev 与 test 的绝对值不可直接比较；dev 仅用于调参与横比",
        },
    }
    return doc


def main() -> int:
    global SEED
    ap = argparse.ArgumentParser(description="构建 D2 语料库级 dev 集")
    ap.add_argument("--every", type=int, default=EVERY)
    ap.add_argument("--corpus-size", type=int, default=CORPUS_SIZE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--reuse-pass1", action="store_true",
                    help="复用已有 queries.jsonl/qrels.tsv（跳过第 1 遍）")
    args = ap.parse_args()
    SEED = args.seed

    log("=" * 66)
    log("  构建 dev 集（D2 · 语料库级）")
    log("=" * 66)
    if not SRC.exists():
        log(f"❌ 源文件不存在：{SRC}")
        return 1
    log(f"源文件 {SRC.name} 大小 {SRC.stat().st_size/2**30:.2f} GB")

    t0 = time.time()
    log("计算源文件 sha256（一次性，约 30s）...")
    src_sha = sha256_file(SRC)
    log(f"sha256 = {src_sha}")

    p1 = pass1(args.every)
    p2 = pass2(p1["pos_ids"], p1["neg_ids"], args.corpus_size, args.seed, args.every)

    doc = summarize({}, p1, p2, args.corpus_size, args.every, src_sha)
    (OUT / "stats.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    (OUT / "build_manifest.json").write_text(
        json.dumps({**doc, "cli": {"every": args.every, "corpus_size": args.corpus_size,
                                   "seed": args.seed}}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    log("=" * 66)
    log(f"✅ dev 构建完成，总用时 {time.time()-t0:.0f}s")
    log(f"  dev query      : {doc['dev']['queries']:,}")
    log(f"  dev 语料       : {doc['dev']['corpus_passages']:,} 段")
    log(f"  qrels          : {doc['dev']['qrels_rows']:,} 行")
    log(f"  每 query 正例  : {doc['dev']['relevant_per_query']}")
    log(f"  段落长度(字)   : {doc['dev']['passage_len_chars']}")
    log(f"  产物目录       : {OUT.relative_to(ROOT)}")
    log("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())