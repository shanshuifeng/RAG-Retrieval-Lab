"""数据质量核查 —— 产出 DATACARD 第二节需要的数字

对 data/processed/{dev,test}/ 做只读核查：
  · 段落数与长度分布（min/mean/median/p95/max）
  · 过短(<20 字) / 过长(>1000 字) / 极端长(>5000 字) 段落数
  · 含 HTML 标签的段落数、含连续空白/换行的段落数
  · 文本重复率（md5 去重）、id 重复数
  · query 数与长度分布、空 query、重复 query
  · ⭐ qrels 覆盖性：relevant_ids 是否全部指向语料中存在的段落（文档规范第七节要求）
  · 每 query 相关段落数分布 + 零相关 query 数

⚠️ 本脚本不修改任何数据，只统计。是否清洗/过滤由 docs/04 4.1 与实际策略决定，
   基线阶段（E1）按 docs/04 4.2 策略 A 使用原生段落，故此处只记录不删除。

用法：
  .venv\\Scripts\\python.exe scripts\\data_quality.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = PROC / "quality_report.json"
LOG = ROOT / "logs" / "data_quality.log"

HTML_RE = re.compile(r"<[a-zA-Z!/][^>]{0,120}>")
BLANK_RE = re.compile(r"[ \t\u3000]{2,}|\n{2,}")


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def dist(vals: list[int]) -> dict:
    if not vals:
        return {}
    s = sorted(vals)
    return {
        "min": s[0], "max": s[-1],
        "mean": round(mean(s), 2), "median": median(s),
        "p95": s[int(0.95 * (len(s) - 1))],
    }


def read_jsonl(p: Path):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def check_split(name: str) -> dict | None:
    d = PROC / name
    if not (d / "corpus.jsonl").exists():
        log(f"  [{name}] 跳过：无 corpus.jsonl")
        return None

    log(f"  [{name}] 核查语料 ...")
    lens: list[int] = []
    ids: set[str] = set()
    dup_ids = 0
    text_hashes: set[bytes] = set()
    dup_texts = 0
    n_short = n_long = n_extreme = 0
    n_html = n_blank = 0
    html_sample: list[str] = []
    longest: tuple[int, str] = (0, "")
    n = 0
    for rec in read_jsonl(d / "corpus.jsonl"):
        n += 1
        i = str(rec.get("id"))
        t = rec.get("text") or ""
        if i in ids:
            dup_ids += 1
        ids.add(i)
        h = hashlib.md5(t.encode("utf-8")).digest()
        if h in text_hashes:
            dup_texts += 1
        text_hashes.add(h)
        L = len(t)
        lens.append(L)
        if L < 20:
            n_short += 1
        if L > 1000:
            n_long += 1
        if L > 5000:
            n_extreme += 1
        if HTML_RE.search(t):
            n_html += 1
            if len(html_sample) < 3:
                html_sample.append(HTML_RE.search(t).group(0)[:40])
        if BLANK_RE.search(t):
            n_blank += 1
        if L > longest[0]:
            longest = (L, i)

    log(f"  [{name}] 核查查询与 qrels ...")
    q_n = 0
    q_lens: list[int] = []
    q_empty = 0
    q_hashes: set[bytes] = set()
    q_dup = 0
    rel_per_q: list[int] = []
    zero_rel = 0
    missing_refs = 0
    qids: set[str] = set()
    for rec in read_jsonl(d / "queries.jsonl"):
        q_n += 1
        q = rec.get("query") or ""
        qid = str(rec.get("qid"))
        if qid in qids:
            log(f"  ⚠️ [{name}] qid 重复：{qid}")
        qids.add(qid)
        q_lens.append(len(q))
        if not q.strip():
            q_empty += 1
        hh = hashlib.md5(q.encode("utf-8")).digest()
        if hh in q_hashes:
            q_dup += 1
        q_hashes.add(hh)
        rel = rec.get("relevant_ids") or []
        rel_per_q.append(len(rel))
        if not rel:
            zero_rel += 1
        for r in rel:
            if str(r) not in ids:
                missing_refs += 1

    res = {
        "corpus": {
            "passages": n,
            "unique_ids": len(ids),
            "duplicate_ids": dup_ids,
            "duplicate_texts": dup_texts,
            "length_chars": dist(lens),
            "passages_lt_20_chars": n_short,
            "passages_gt_1000_chars": n_long,
            "passages_gt_5000_chars": n_extreme,
            "passages_with_html_tag": n_html,
            "html_tag_samples": html_sample,
            "passages_with_blank_runs": n_blank,
            "longest_passage": {"chars": longest[0], "id": longest[1]},
        },
        "queries": {
            "count": q_n,
            "unique_qids": len(qids),
            "empty_queries": q_empty,
            "duplicate_query_texts": q_dup,
            "length_chars": dist(q_lens),
        },
        "qrels": {
            "relevant_per_query": dist(rel_per_q),
            "zero_relevant_queries": zero_rel,
            "relevant_refs_missing_from_corpus": missing_refs,
        },
    }
    c = res["corpus"]
    log(f"  [{name}] 语料 {n:,} 段 | 文本重复 {dup_texts:,} 段 | 含 HTML {n_html:,} 段"
        f" | <20 字 {n_short:,} | >1000 字 {n_long:,} | >5000 字 {n_extreme:,}")
    log(f"  [{name}] query {q_n:,} 条 | 零相关 {zero_rel} | "
        f"qrels 悬空引用 {missing_refs}")
    return res


def main() -> int:
    log("=" * 66)
    log("  数据质量核查（只读）")
    log("=" * 66)

    splits = [n for n in ("dev", "test") if (PROC / n / "corpus.jsonl").exists()]
    if not splits:
        log("❌ 没有可核查的 processed 数据集，先跑 convert_cmteb.py / build_dev.py")
        return 1

    report = {"checked_at": datetime.now().isoformat(timespec="seconds"), "splits": {}}
    for s in splits:
        report["splits"][s] = check_split(s)

    old = {}
    if OUT.exists():
        try:
            old = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            old = {}
    merged = {**old, **report}
    merged["splits"] = {**old.get("splits", {}), **report["splits"]}
    OUT.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"报告 → {OUT.relative_to(ROOT)}")

    # ⭐ 一句话判断：达到 docs/04 第九节检查清单的哪几项
    for s, r in report["splits"].items():
        if not r:
            continue
        ok = r["qrels"]["relevant_refs_missing_from_corpus"] == 0
        log(f"  [{s}] qrels 全部指向存在段落: {'✅' if ok else '❌'}"
            f" | 零相关 query: {r['qrels']['zero_relevant_queries']}")
    log("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())