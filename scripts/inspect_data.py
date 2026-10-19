"""检查已下载数据集的实际内容 —— M1 数据核验

用法:
    .venv\\Scripts\\python.exe scripts\\inspect_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def hr(c: str = "=", w: int = 74) -> None:
    print(c * w)


def find_parquet(base: Path) -> dict[str, Path]:
    """在目录下按名字归类 parquet 文件。"""
    out: dict[str, Path] = {}
    for p in base.rglob("*.parquet"):
        name = p.name.lower()
        if "corpus" in name:
            out["corpus"] = p
        elif "queries" in name or "query" in name:
            out["queries"] = p
        elif "dev" in name or "qrel" in name or "test" in name:
            out.setdefault("qrels", p)
    return out


def show_df(name: str, df: pd.DataFrame, n: int = 3) -> None:
    hr()
    print(f"  【{name}】")
    hr()
    print(f"  行数: {len(df):,}")
    print(f"  列  : {list(df.columns)}")
    print(f"  dtypes:")
    for c in df.columns:
        print(f"      {c:16s} {str(df[c].dtype)}")

    print(f"\n  前 {n} 行:")
    for i in range(min(n, len(df))):
        row = df.iloc[i].to_dict()
        print(f"    [{i}]")
        for k, v in row.items():
            s = str(v)
            if len(s) > 160:
                s = s[:160] + f"…(共{len(str(v))}字)"
            print(f"        {k:14s}: {s}")

    # 关键统计
    print(f"\n  统计:")
    for c in df.columns:
        if df[c].dtype == object:
            lens = df[c].astype(str).str.len()
            print(f"      {c:14s} 长度 均值={lens.mean():.1f} "
                  f"中位={lens.median():.0f} 最大={lens.max()}")
        elif pd.api.types.is_numeric_dtype(df[c]):
            print(f"      {c:14s} 均值={df[c].mean():.3f} "
                  f"最小={df[c].min()} 最大={df[c].max()}")


def check_integrity(corpus: pd.DataFrame, queries: pd.DataFrame,
                    qrels: pd.DataFrame | None) -> None:
    """⭐ 校验 id 引用完整性（评测协议里的检查项）。"""
    hr()
    print("  ⭐ 引用完整性校验")
    hr()

    # 找 id 列名
    def pick_id_col(df: pd.DataFrame, cands: list[str]) -> str | None:
        for c in cands:
            if c in df.columns:
                return c
        # 退而求其次：第一个 object 列
        for c in df.columns:
            if df[c].dtype == object:
                return c
        return None

    c_id = pick_id_col(corpus, ["_id", "id", "pid", "doc_id", "corpus_id"])
    q_id = pick_id_col(queries, ["_id", "id", "qid", "query_id"])
    print(f"  corpus id 列 : {c_id}")
    print(f"  queries id 列: {q_id}")

    if not (c_id and q_id):
        print("  [!] 无法定位 id 列，跳过校验")
        return

    corpus_ids = set(corpus[c_id].astype(str))
    print(f"\n  corpus 唯一 id 数: {len(corpus_ids):,}")

    if qrels is not None:
        print(f"  qrels 列: {list(qrels.columns)}")
        # 常见列名: query-id / corpus-id / score, 或 qid / pid / label
        def find(cols: list[str], cands: list[str]) -> str | None:
            for c in cands:
                if c in cols:
                    return c
            return None

        qc = find(list(qrels.columns), ["query-id", "qid", "query_id"])
        pc = find(list(qrels.columns), ["corpus-id", "pid", "corpus_id", "doc_id"])
        print(f"  qrels query 列: {qc}")
        print(f"  qrels corpus 列: {pc}")

        if qc and pc:
            n_q = qrels[qc].nunique()
            print(f"\n  qrels 覆盖 query 数: {n_q:,}")

            # ⭐ 检查 qrels 里的 corpus id 是否都存在
            pid_set = set(qrels[pc].astype(str))
            missing = pid_set - corpus_ids
            print(f"  qrels 引用的 corpus id 数: {len(pid_set):,}")
            if missing:
                print(f"  ❌ 有 {len(missing):,} 个 corpus id 在语料库中不存在！")
                print(f"      示例: {list(missing)[:5]}")
            else:
                print(f"  ✅ 全部 corpus id 都能在语料库中找到")

            # ⭐ 检查 queries 覆盖
            qid_set = set(qrels[qc].astype(str))
            q_set = set(queries[q_id].astype(str))
            print(f"\n  queries 表 id 数: {len(q_set):,}")
            print(f"  qrels 中但没有 query 文本的: {len(qid_set - q_set):,}")
            print(f"  queries 中但没有 qrels 的  : {len(q_set - qid_set):,}")

            # ⭐ 每个 query 的相关文档数分布（评测协议要求）
            per_q = qrels.groupby(qc).size()
            print(f"\n  ⭐ 每 query 的相关文档数:")
            print(f"      均值={per_q.mean():.2f}  中位={per_q.median():.0f}  "
                  f"最大={per_q.max()}  最小={per_q.min()}")
            print(f"      为 0 的 query 数: {(per_q == 0).sum()}")

            # score 列分布（可能是二值或分级）
            if "score" in qrels.columns:
                print(f"\n  qrels score 分布:")
                print(qrels["score"].value_counts().to_string())


def main() -> int:
    hr("#")
    print("#  RAG-Retrieval-Lab · 数据内容检查 (M1)")
    hr("#")

    ds_dir = RAW / "dureval"
    if not ds_dir.exists():
        print(f"  ❌ 数据目录不存在: {ds_dir}")
        print(f"     请先运行: scripts/download_data.py")
        return 1

    files = find_parquet(ds_dir)
    print(f"\n  发现文件:")
    for k, p in files.items():
        print(f"    {k:10s} {p.relative_to(ds_dir)}  "
              f"({p.stat().st_size / 1024**2:.1f} MB)")

    if "corpus" not in files:
        print("\n  ❌ 找不到 corpus 文件")
        return 1

    corpus = pd.read_parquet(files["corpus"])
    queries = pd.read_parquet(files["queries"]) if "queries" in files else None
    qrels = pd.read_parquet(files["qrels"]) if "qrels" in files else None

    show_df("CORPUS（语料库）", corpus)
    if queries is not None:
        show_df("QUERIES（查询）", queries)
    if qrels is not None:
        show_df("QRELS（相关性标注）", qrels, n=5)

    if queries is not None:
        check_integrity(corpus, queries, qrels)

    # 保存一份摘要
    summary = {
        "dataset": "C-MTEB/DuRetrieval",
        "corpus_rows": len(corpus),
        "corpus_cols": list(corpus.columns),
        "queries_rows": len(queries) if queries is not None else None,
        "queries_cols": list(queries.columns) if queries is not None else None,
        "qrels_rows": len(qrels) if qrels is not None else None,
        "qrels_cols": list(qrels.columns) if qrels is not None else None,
    }
    out = ds_dir / "inspect_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  ⭐ 摘要已保存: {out}")

    hr("#")
    return 0


if __name__ == "__main__":
    sys.exit(main())
