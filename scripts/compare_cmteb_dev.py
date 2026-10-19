"""比对：C-MTEB/DuRetrieval 的 2000 query 与 ModelScope dev.jsonl 的 2000 query 是否同一批

用途：
  若两者相同 → ModelScope 的 dev.jsonl.gz 其实就是「官方测试集」，
  不能当我们的 dev（否则 = 在测试集上调参，违反 docs/05 的铁律）。
  那么用户的选择「另下训练集作 dev」就是唯一正确路径。

输出：
  data/raw/dureader-ranking/compare_cmteb_dev.json
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DEV = RAW / "dureader-ranking" / "dev.jsonl.gz"
OUT = RAW / "dureader-ranking" / "compare_cmteb_dev.json"


def hr(c: str = "=", w: int = 74) -> None:
    print(c * w)


def main() -> int:
    hr("#")
    print("  C-MTEB 2000 query  vs  ModelScope dev 2000 query")
    hr("#")

    qf = sorted((RAW / "dureval" / "data").glob("queries-*.parquet"))[0]
    cm = pd.read_parquet(qf)
    print(f"  C-MTEB queries 列: {list(cm.columns)}  行数: {len(cm):,}")
    print(f"  C-MTEB 首条: {cm.iloc[0].to_dict()}")

    cm_texts = set(cm["text"].astype(str).str.strip())
    cm_ids = set(cm["id"].astype(str))

    dev_texts: set[str] = set()
    dev_ids: set[str] = set()
    dev_q_text: dict[str, str] = {}
    dev_pos: dict[str, set[str]] = {}
    with gzip.open(DEV, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            qid = str(d.get("query_id"))
            txt = str(d.get("query", "")).strip()
            dev_texts.add(txt)
            dev_ids.add(qid)
            dev_q_text[qid] = txt
            dev_pos[qid] = {str(p["docid"]) for p in (d.get("positive_passages") or [])
                            if p and p.get("docid") is not None}

    hr()
    print("  1. query 集合比对")
    hr()
    print(f"  C-MTEB query 数        : {len(cm_texts):,}")
    print(f"  ModelScope dev query 数: {len(dev_texts):,}")
    inter_t = cm_texts & dev_texts
    print(f"  ⭐ 文本交集            : {len(inter_t):,}"
          f"  (= C-MTEB 的 {100*len(inter_t)/max(1,len(cm_texts)):.1f}%)")
    print(f"  C-MTEB 独有            : {len(cm_texts - dev_texts):,}")
    print(f"  dev 独有               : {len(dev_texts - cm_texts):,}")
    same_id_space = bool(cm_ids & dev_ids)
    print(f"  id 空间是否重合        : {same_id_space}"
          f"（交集 {len(cm_ids & dev_ids):,}）")

    # 若 id 空间一致，进一步做「按 id 对齐的文本一致性」
    aligned_equal = None
    if same_id_space:
        cm_map = dict(zip(cm["id"].astype(str), cm["text"].astype(str).str.strip()))
        both = cm_ids & dev_ids
        aligned_equal = sum(1 for q in both if cm_map.get(q) == dev_q_text.get(q))
        print(f"  同 id 的 query 文本一致: {aligned_equal:,} / {len(both):,}")

    hr()
    print("  2. qrels 与 dev 正例比对")
    hr()
    qrf = sorted((RAW / "dureval" / "qrels").rglob("*.parquet"))[0]
    qr = pd.read_parquet(qrf)
    print(f"  C-MTEB qrels 列: {list(qr.columns)}  行数: {len(qr):,}")
    print(f"  qrels 首 3 行:\n{qr.head(3).to_string(index=False)}")

    qcol = next((c for c in ("qid", "query-id", "query_id") if c in qr.columns), None)
    pcol = next((c for c in ("pid", "corpus-id", "corpus_id", "doc_id") if c in qr.columns), None)
    cm_rels: dict[str, set[str]] = {}
    if qcol and pcol:
        for q, p in zip(qr[qcol].astype(str), qr[pcol].astype(str)):
            cm_rels.setdefault(q, set()).add(p)
        print(f"  qrels 覆盖 query 数: {len(cm_rels):,}")
        print(f"  qrels 唯一 pid 数  : {len(set(qr[pcol].astype(str))):,}")
        if "score" in qr.columns:
            print(f"  score 取值分布     : "
                  f"{qr['score'].value_counts().to_dict()}")

        # 用文本对齐来比较（id 空间不同时唯一可行的办法）
        cm_id2text = dict(zip(cm["id"].astype(str), cm["text"].astype(str).str.strip()))
        dev_text2qid = {v: k for k, v in dev_q_text.items()}
        common_texts = inter_t
        agree = 0
        cm_only = 0
        dev_only = 0
        checked = 0
        for t in common_texts:
            cm_qid = next((q for q, tt in cm_id2text.items() if tt == t), None)
            dev_qid = dev_text2qid.get(t)
            if cm_qid is None or dev_qid is None:
                continue
            checked += 1
            a = cm_rels.get(cm_qid, set())
            b = dev_pos.get(dev_qid, set())
            if a and b and a == b:
                agree += 1
            elif b - a:
                dev_only += 1
            elif a - b:
                cm_only += 1
        print(f"\n  可比对 query 数    : {checked:,}")
        print(f"  ⭐ qrels 集合完全相同: {agree:,}"
              f"  ({100*agree/max(1,checked):.1f}%)")
        print(f"  dev 正例多于 qrels : {dev_only:,}")
        print(f"  qrels 多于 dev 正例: {cm_only:,}")

    result = {
        "cmteb_queries": len(cm_texts),
        "dev_queries": len(dev_texts),
        "text_overlap": len(inter_t),
        "cmteb_only": len(cm_texts - dev_texts),
        "dev_only": len(dev_texts - cm_texts),
        "id_space_overlap": len(cm_ids & dev_ids),
        "same_id_aligned_equal": aligned_equal,
        "cmteb_qrels_rows": int(len(qr)),
        "cmteb_qrels_queries": len(cm_rels),
        "qrels_equals_dev_positives": agree if qcol and pcol else None,
        "compared_queries": checked if qcol and pcol else None,
        "verdict": None,
    }
    if len(inter_t) == len(cm_texts) == len(dev_texts):
        result["verdict"] = ("IDENTICAL: ModelScope dev.jsonl.gz 与 C-MTEB 的 2000 查询是同一批 "
                             "→ 它就是我们的测试集，不能当 dev")
    elif len(inter_t) > 0.9 * min(len(cm_texts), len(dev_texts)):
        result["verdict"] = "HIGH_OVERLAP: 高度重合 → 同样不可当 dev"
    else:
        result["verdict"] = "DISJOINT: 两批查询不同 → dev 可用"

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    hr()
    print(f"  ⭐ 判定: {result['verdict']}")
    print(f"  报告: {OUT.relative_to(ROOT)}")
    hr("#")
    return 0


if __name__ == "__main__":
    sys.exit(main())