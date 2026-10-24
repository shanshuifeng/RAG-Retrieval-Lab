"""二轮复核：文档数字 ↔ 结果 JSON 一致性交叉比对（自动，不靠肉眼）

⭐ 上一轮发现：docs/11 里 BM25 调优的 MRR@10 写错（0.5743 应为 0.5737）。
   这类"手抄数字"的错误肉眼难查，必须程序化比对。

本脚本做的事：
  1. 从 results/ 的最新 JSON 提取权威数字
  2. 扫描 docs/*.md 中所有形如 0.xxxx 的小数
  3. 报告"文档数字与权威数字不一致"的候选（允许 4 位小数四舍五入）

⚠️ 这是启发式检查：只报告疑点，是否真错误需人工判断。
   目的不是全对，而是把所有可疑数字列出来逐一过。

用法：
  .venv\\Scripts\\python.exe scripts\\audit_doc_numbers.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"


def latest(pattern: str) -> dict | None:
    files = sorted(RESULTS.glob(pattern))
    if not files:
        return None
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    return d


def collect_truth() -> dict[str, float]:
    """收集权威数字：指标名 → 值。键形如 bm25_default/recall@10。"""
    truth: dict[str, float] = {}

    def add(prefix: str, d: dict | None):
        if not d:
            return
        for k, v in (d.get("results") or {}).items():
            truth[f"{prefix}/{k}"] = float(v)

    add("bm25_default", latest("*E1_bm25_dev.json"))
    add("dense_small", latest("*E1_dense_small_dev.json"))
    add("dense_base", latest("*E1_dense_base_dev.json"))

    grid = latest("*E1_bm25_grid_dev.json")
    if grid and grid.get("best"):
        best = grid["best"]
        truth["bm25_tuned/k1"] = best["k1"]
        truth["bm25_tuned/b"] = best["b"]
        for k, v in best.items():
            if isinstance(v, (int, float)) and "@" in str(k):
                truth[f"bm25_tuned_grid/{k}"] = float(v)

    # 最终锁定值（来自扩展搜索 + 显著性检验）
    sig = latest("*significance_bm25_tuning_dev.json")
    if sig:
        for t in sig.get("tests", []):
            truth[f"bm25_tuned/{t['metric']}"] = t["new_mean"]
            truth[f"bm25_default_sig/{t['metric']}"] = t["baseline_mean"]

    comp = latest("*E1_comparison_dev.json")
    if comp:
        m = comp.get("metrics", {})
        name_map = {
            "BM25(默认)": "bm25_default",
            "BM25(调优)": "bm25_tuned",
            "D1/BGE-small": "dense_small",
            "D2/BGE-base": "dense_base",
        }
        for k, v in m.items():
            prefix = name_map.get(k)
            if prefix:
                for mk, mv in v.items():
                    truth[f"{prefix}/{mk}"] = float(mv)
    return truth


def main() -> int:
    print("=" * 78)
    print("  二轮复核：文档数字 ↔ 结果 JSON 一致性")
    print("=" * 78)

    truth = collect_truth()
    print(f"\n  权威数字共 {len(truth)} 个")
    for k in sorted(truth):
        print(f"    {k:<28} = {truth[k]}")

    # ⭐ 重点核对项：文档里"应当出现"的数字
    checks = [
        # (文件, 权威键, 说明)
        ("11-E1实验报告草稿.md", "bm25_default/ndcg@10", "BM25 默认"),
        ("11-E1实验报告草稿.md", "bm25_tuned/ndcg@10", "BM25 调优"),
        ("11-E1实验报告草稿.md", "dense_small/ndcg@10", "D1"),
        ("11-E1实验报告草稿.md", "dense_base/ndcg@10", "D2"),
        ("11-E1实验报告草稿.md", "bm25_default/recall@10", "BM25 默认 Recall"),
        ("11-E1实验报告草稿.md", "bm25_tuned/recall@10", "BM25 调优 Recall"),
        ("11-E1实验报告草稿.md", "dense_small/recall@10", "D1 Recall"),
        ("11-E1实验报告草稿.md", "dense_base/recall@10", "D2 Recall"),
        ("11-E1实验报告草稿.md", "bm25_tuned/mrr@10", "⭐ BM25 调优 MRR（上轮发现写错的那个）"),
        ("12-项目计划报告.md", "dense_small/ndcg@10", "计划报告 D1"),
        ("12-项目计划报告.md", "dense_base/ndcg@10", "计划报告 D2"),
    ]

    num_re = re.compile(r"\b0\.\d{3,6}\b")
    issues = []
    for doc, key, note in checks:
        want = truth.get(key)
        if want is None:
            continue
        p = DOCS / doc
        if not p.exists():
            issues.append((doc, key, want, None, "文件不存在"))
            continue
        text = p.read_text(encoding="utf-8")
        found = [float(x) for x in num_re.findall(text)]
        # 期望文档中出现 want（允许 4 位四舍五入：|x - want| < 0.0001）
        hit = any(abs(x - want) < 5e-5 for x in found)
        issues.append((doc, key, want, hit, note))

    print(f"\n  文档中应出现的数字核对：")
    print(f"  {'文档':<26} {'键':<24} {'权威值':>10}  在文档中?  说明")
    print("  " + "-" * 90)
    all_ok = True
    for doc, key, want, hit, note in issues:
        mark = "✅" if hit else "❌"
        if not hit:
            all_ok = False
        print(f"  {doc[:26]:<26} {key:<24} {want:>10.6f}  {mark}      {note}")

    # ⭐ 反向检查：文档里出现了哪些权威数字里【没有】的指标值
    #    （上轮发现的 0.5743 就是这种：既不在 grid 也不在 sig 里）
    print(f"\n  ⭐ 反向检查：文档中的数字是否都能对应到权威结果")
    stale_candidates = []
    known_vals = set(truth.values())
    # 4 位小数容差集合
    def matches_known(x: float) -> bool:
        return any(abs(x - v) < 5e-5 for v in known_vals)
    for doc in ["11-E1实验报告草稿.md", "12-项目计划报告.md"]:
        text = (DOCS / doc).read_text(encoding="utf-8")
        for m in num_re.finditer(text):
            x = float(m.group())
            # 只看指标量级的数（0.2–0.99），排除占比/效应量等
            if 0.20 <= x <= 0.99 and not matches_known(x):
                line_start = text.rfind("\n", 0, m.start()) + 1
                line = text[line_start:text.find("\n", m.end())][:90]
                stale_candidates.append((doc, m.group(), line.strip()))
    if stale_candidates:
        print(f"  发现 {len(stale_candidates)} 个可疑数字（需人工判断）：")
        seen = set()
        for doc, num, line in stale_candidates:
            key = (doc, num)
            if key in seen:
                continue
            seen.add(key)
            print(f"    [{doc}] {num}  ← {line}")
    else:
        print("  ✅ 未发现可疑数字")

    print()
    print("=" * 78)
    print("  ⭐ " + ("一致性核对通过" if all_ok and not stale_candidates
                         else "发现疑点，见上表"))
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())