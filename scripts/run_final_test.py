"""⭐⭐⭐ test 集最终评测 —— 全项目只允许跑一次（P8）

⚠️ 纪律（docs/05 §五 Step 1 + DATACARD §五）：
   · test 集【只用一次】，所有调参都在 dev 上完成
   · 本脚本用【机械锁】强制执行：成功跑完后写 results/TEST_USED.lock，
     之后再次运行会被拒绝（确需重跑必须先手动删锁——这是刻意设置的摩擦）
   · 所有配置在 dev 上锁定，本脚本【只读配置、不做任何选择】

锁定的配置（全部来自 dev 的结论，见各实验报告）：
   ┌─────────────────────────────────────────────────────────────┐
   │ 1. BM25          k1=0.7, b=0.75（P3 网格+扩展+显著性检验锁定）│
   │ 2. D1            bge-small-zh-v1.5，无指令，fp16（复核消融锁定）│
   │ 3. D2            bge-base-zh-v1.5，无指令，fp16              │
   │ 4. RRF           BM25(调优)+D2，k=20，等权（E2 负面结果对照）  │
   │ 5. Rerank        D2 第一段 top-100 → K=20 重排（E3 甜点锁定） │
   └─────────────────────────────────────────────────────────────┘

产出：
   results/YYYY-MM-DD_test_final.json     —— 汇总（指标+成本+显著性）
   results/YYYY-MM-DD_test_final_<方法>_run.json —— 各方法的完整排序
   results/TEST_USED.lock                 —— ⭐ 一次性锁

用法：
  .venv\\Scripts\\python.exe scripts\\run_final_test.py --confirm-final
"""

from __future__ import annotations

# ⭐ 模型已全部下载到 models/，强制离线，避免网络抖动导致的缓存回退错误
#    （2026-10-24 首次 test 评测时，hf-mirror 超时 → transformers 回退缓存
#     → AutoProcessor 无法识别 bge-base，在 D2 处崩溃）
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_split  # noqa: E402
from src.eval.metrics import evaluate, format_table, ndcg_at_k  # noqa: E402
from src.retrieval.bm25_fast import BM25Fast  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402
from src.retrieval.hybrid import rrf_fuse  # noqa: E402
from src.rerank.cross_encoder import CrossEncoderReranker  # noqa: E402

RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "index"
LOCK = RESULTS / "TEST_USED.lock"

# ⭐ 锁定的配置（dev 上已确定，本脚本不得修改）
LOCKED = {
    "bm25": {"k1": 0.7, "b": 0.75},
    "dense_d1": {"model": "BAAI/bge-small-zh-v1.5"},
    "dense_d2": {"model": "BAAI/bge-base-zh-v1.5"},
    "rrf": {"pair": ("BM25(调优)", "D2"), "k": 20, "weights": "equal"},
    "rerank": {"first_stage": "D2", "first_top": 100, "K": 20,
               "model": "BAAI/bge-reranker-base"},
}
KS = [1, 5, 10, 20, 100]


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    diff = b - a
    n = len(diff)
    t, p = stats.ttest_rel(b, a)
    sd = diff.std(ddof=1)
    d = diff.mean() / sd if sd > 0 else 0.0
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(10000, n))
    lo, hi = np.percentile(diff[idx].mean(axis=1), [2.5, 97.5])
    return {"n": int(n), "diff": round(float(diff.mean()), 6),
            "rel_pct": round(100 * float(diff.mean()) / float(a.mean()), 2)
            if a.mean() else 0.0,
            "p_value": float(p), "cohens_d": round(float(d), 4),
            "ci95": [round(float(lo), 6), round(float(hi), 6)]}


def per_q(run: dict, queries: dict, k: int = 10) -> dict[str, float]:
    return {qid: ndcg_at_k(run.get(qid, []), rel, k)
            for qid, rel in queries.items() if rel}


def eval_method(name: str, run: dict, split, cost: dict) -> dict:
    queries = {q: split.relevant_of(q) for q in split.query_list()}
    res = evaluate(queries, run, ks=KS, metrics=("recall", "mrr", "ndcg"))
    return {
        "method": name,
        "results": {k: round(v["mean"], 6) for k, v in res["aggregate"].items()},
        "per_query_ndcg10": per_q(run, queries, 10),
        "cost": cost,
        "n_evaluated": res["filtered"]["queries_evaluated"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="⭐ test 集最终评测（仅一次）")
    ap.add_argument("--confirm-final", action="store_true",
                    help="确认：我已知悉这会消耗 test 集的唯一一次机会")
    args = ap.parse_args()

    print("=" * 80)
    print("  ⭐⭐⭐ test 集最终评测（全项目仅允许一次）")
    print("=" * 80)

    # ── 机械锁 ──
    if LOCK.exists():
        info = json.loads(LOCK.read_text(encoding="utf-8"))
        print(f"\n❌ 拒绝运行：test 集已于 {info.get('used_at')} 使用过")
        print(f"   当时结果：{info.get('result_file')}")
        print(f"\n   若确需重跑（如发现 bug），必须【手动】删除 {LOCK.name} 并在报告中声明。")
        print("   这是刻意设置的摩擦——test 集的权威性来自“只跑一次”。")
        return 2
    if not args.confirm_final:
        print("\n❌ 缺少 --confirm-final。test 集只在配置全部锁定后跑一次。")
        print("   确认运行请显式加该参数。")
        return 2

    t_all = time.time()
    print("\n[0/6] 加载 test 数据（只读）...")
    split = load_split("test")
    queries = {q: split.relevant_of(q) for q in split.query_list()}
    print(f"  语料 {split.n_passages:,} 段 | 评测 query {len(queries):,} 条")
    print(f"  ⭐ 锁定配置：{json.dumps(LOCKED, ensure_ascii=False)}")

    methods: dict[str, dict] = {}

    # ── 1. BM25（调优参数）──
    print("\n[1/6] BM25（k1=0.7, b=0.75）...")
    t0 = time.time()
    base_cache = CACHE / "bm25fast_base_test.pkl"
    bm = BM25Fast()
    if base_cache.exists():
        bm = BM25Fast.load(base_cache)
        print("  命中基础统计缓存")
    else:
        bm.build_base(split, verbose=True)
        bm.save(base_cache)
    bm.set_params(k1=LOCKED["bm25"]["k1"], b=LOCKED["bm25"]["b"])
    t_ret = time.time()
    run_bm25 = bm.run_all(split, top_k=100, verbose=True)
    bm25_search_sec = time.time() - t_ret
    methods["BM25(调优)"] = eval_method(
        "BM25(调优 k1=0.7,b=0.75)", run_bm25, split,
        {"build_base_sec": round(bm._build_sec, 2),
         "search_sec": round(bm25_search_sec, 2),
         "search_ms_per_query": round(1000 * bm25_search_sec / len(queries), 3),
         "device": "cpu"})
    print(f"  nDCG@10={methods['BM25(调优)']['results']['ndcg@10']:.4f}")

    # ── 2-3. 稠密 D1 / D2 ──
    dense_runs = {}
    for tag, key, mid in [("small", "dense_d1", LOCKED["dense_d1"]["model"]),
                          ("base", "dense_d2", LOCKED["dense_d2"]["model"])]:
        print(f"\n  稠密 {tag}（{mid}，无指令）...")
        idx_path = CACHE / f"dense_{tag}_test_noinstr.pkl"
        if idx_path.exists():
            dr = DenseRetriever.load(idx_path)
            print("  命中索引缓存")
        else:
            dr = DenseRetriever(mid)
            dr.build(split, verbose=True)
            dr.save(idx_path)
        t_ret = time.time()
        run = dr.run_all(split, top_k=100, verbose=True)
        sec = time.time() - t_ret
        name = f"D{'1' if tag == 'small' else '2'}/BGE-{tag}"
        methods[name] = eval_method(
            name, run, split,
            {"encode_sec": round(dr._encode_sec, 2),
             "search_sec": round(sec, 2),
             "search_ms_per_query": round(1000 * sec / len(queries), 3),
             "device": dr.stats()["device"], "fp16": dr.stats()["fp16"]})
        dense_runs[tag] = run
        print(f"  nDCG@10={methods[name]['results']['ndcg@10']:.4f}")
        del dr   # 释放显存再加载下一个模型

    # ── 4. RRF 混合（E2 负面结果的 test 对照）──
    print("\n[4/6] RRF 混合（BM25+D2, k=20，等权）...")
    t0 = time.time()
    run_rrf = rrf_fuse({"bm25": run_bm25, "d2": dense_runs["base"]},
                       k=LOCKED["rrf"]["k"], top_k=100)
    rrf_sec = time.time() - t0
    methods["RRF(BM25+D2)"] = eval_method(
        "RRF(BM25调优+D2, k=20)", run_rrf, split,
        {"fuse_sec": round(rrf_sec, 2), "note": "纯排名计算，零检索成本"})
    print(f"  nDCG@10={methods['RRF(BM25+D2)']['results']['ndcg@10']:.4f}")

    # ── 5. 重排（D2 第一段 top-100 → K=20）──
    print("\n[5/6] 重排（D2 top-100 → K=20）...")
    first_run = dense_runs["base"]
    score_cache = CACHE / f"rerank_scores_d2_test_{len(queries)}q.json"
    if score_cache.exists():
        cached = json.loads(score_cache.read_text(encoding="utf-8"))
        all_scores = cached["scores"]
        pair_ms = cached.get("latency_per_pair_ms")
        print(f"  命中分数缓存（{len(all_scores):,} query）")
    else:
        reranker = CrossEncoderReranker(LOCKED["rerank"]["model"])
        reranker._load()
        print(f"  模型 {LOCKED['rerank']['model']} 已加载")
        qids = [q for q in split.query_list() if q in first_run]
        tasks = [(qid, split.queries[qid], pid, split.corpus.get(pid, ""))
                 for qid in qids for pid in first_run[qid][:LOCKED["rerank"]["first_top"]]]
        n_pairs = len(tasks)
        print(f"  打分对：{n_pairs:,}")
        t_score = time.time()
        all_scores = {}
        B = 512
        for s in range(0, n_pairs, B):
            chunk = tasks[s:s + B]
            i = 0
            while i < len(chunk):
                qid, q = chunk[i][0], chunk[i][1]
                j = i
                texts, pids = [], []
                while j < len(chunk) and chunk[j][0] == qid:
                    texts.append(chunk[j][3]); pids.append(chunk[j][2]); j += 1
                sc = reranker.score(q, texts)
                all_scores.setdefault(qid, {}).update(zip(pids, sc))
                i = j
            if (s // B) % 100 == 0:
                el = time.time() - t_score
                print(f"      打分 {min(s+B, n_pairs):,}/{n_pairs:,} "
                      f"({el:.0f}s) ...", flush=True)
        total_sec = time.time() - t_score
        pair_ms = 1000 * total_sec / n_pairs
        print(f"  打分完成：{total_sec:.0f}s，每对 {pair_ms:.2f} ms")
        score_cache.write_text(json.dumps({
            "scores": all_scores, "latency_per_pair_ms": pair_ms,
            "n_pairs": n_pairs, "total_score_sec": round(total_sec, 1),
        }, ensure_ascii=False), encoding="utf-8")

    K = LOCKED["rerank"]["K"]
    run_rerank = {}
    for qid in first_run:
        cands = first_run[qid][:K]
        sc = all_scores.get(qid, {})
        run_rerank[qid] = sorted(cands,
                                 key=lambda p: (-sc.get(p, float("-inf")), p))
    est_lat = K * pair_ms if pair_ms else None
    methods["D2+重排(K=20)"] = eval_method(
        "D2+重排(K=20)", run_rerank, split,
        {"score_pairs": len(queries) * LOCKED["rerank"]["first_top"],
         "est_latency_ms_per_query": round(est_lat, 1) if est_lat else None,
         "latency_model": "每对延迟×K 估算"})
    print(f"  nDCG@10={methods['D2+重排(K=20)']['results']['ndcg@10']:.4f}")

    # ── 6. 汇总 + 显著性检验 ──
    print("\n[6/6] 汇总与显著性检验 ...")
    ks_show = KS
    print()
    header = ["方法"] + [f"nDCG@{k}" for k in ks_show] + ["R@10", "检索ms/q"]
    print("  " + "  ".join(h.ljust(10) for h in header))
    print("  " + "-" * 78)
    for name, m in methods.items():
        r = m["results"]
        row = [name[:16].ljust(16)]
        row += [f"{r.get(f'ndcg@{k}', 0):.4f}".ljust(10) for k in ks_show]
        row.append(f"{r.get('recall@10', 0):.4f}".ljust(10))
        ms = m["cost"].get("search_ms_per_query")
        row.append(f"{ms:.2f}" if ms is not None else "—")
        print("  " + "  ".join(row))

    # 配对检验：vs BM25，以及重排 vs D2
    tests = []
    def sig(a_name, b_name):
        a = np.array([methods[a_name]["per_query_ndcg10"][q] for q in queries if q in methods[a_name]["per_query_ndcg10"] and q in methods[b_name]["per_query_ndcg10"]])
        b = np.array([methods[b_name]["per_query_ndcg10"][q] for q in queries if q in methods[a_name]["per_query_ndcg10"] and q in methods[b_name]["per_query_ndcg10"]])
        return paired_test(a, b) | {"comparison": f"{b_name} vs {a_name}"}
    for m in methods:
        if m != "BM25(调优)":
            tests.append(sig("BM25(调优)", m))
    tests.append(sig("D2/BGE-base", "D2+重排(K=20)"))

    print(f"\n  显著性检验（nDCG@10，逐 query 配对）：")
    print(f"  {'对比':<30} {'差值':>9} {'相对':>8} {'p 值':>11} {'d':>7}")
    print("  " + "-" * 72)
    for t_ in tests:
        mark = "***" if t_["p_value"] < 0.001 else ("**" if t_["p_value"] < 0.01 else ("*" if t_["p_value"] < 0.05 else "ns"))
        print(f"  {t_['comparison']:<30} {t_['diff']:>+9.4f} {t_['rel_pct']:>+7.2f}% "
              f"{t_['p_value']:>11.2e} {t_['cohens_d']:>7.3f} {mark}")

    # ── 落盘 + 写锁 ──
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = RESULTS / f"{stamp}_test_final.json"
    payload = {
        "experiment_id": "FINAL",
        "split": "test",
        "date": datetime.now().isoformat(timespec="seconds"),
        "note": "⭐ test 集最终评测，全项目仅一次。所有配置在 dev 上锁定。",
        "locked_config": LOCKED,
        "data": {"corpus": split.n_passages, "queries": len(queries),
                 "qrels": split.n_relevant_pairs},
        "methods": {name: {"results": m["results"], "cost": m["cost"],
                           "n_evaluated": m["n_evaluated"]}
                    for name, m in methods.items()},
        "significance": tests,
        "total_sec": round(time.time() - t_all, 1),
    }
    # per_query 序列太大，不放进汇总（run 文件已另存）
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  汇总 → {out.relative_to(ROOT)}")

    for name, m in methods.items():
        run_file = RESULTS / f"{stamp}_test_final_{name.replace('/', '_')}_run.json"
        # methods 里没存 run，重新引用
    # run 文件单独存
    runs_to_save = {
        "bm25_tuned": run_bm25,
        "dense_d1": dense_runs["small"],
        "dense_d2": dense_runs["base"],
        "rrf": run_rrf,
        "rerank_k20": run_rerank,
    }
    for key, run in runs_to_save.items():
        rp = RESULTS / f"{stamp}_test_final_{key}_run.json"
        rp.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
    print(f"  各方法 run 已存档（前缀 {stamp}_test_final_*_run.json）")

    LOCK.write_text(json.dumps({
        "used_at": datetime.now().isoformat(timespec="seconds"),
        "result_file": out.name,
        "total_sec": round(time.time() - t_all, 1),
        "note": "test 集已于本次使用。再次评测须手动删除本锁并在报告中声明原因。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  ⭐ 已写入一次性锁：results/{LOCK.name}")
    print(f"  test 集自此锁定，任何后续评测请求都将被拒绝。")
    print(f"  总用时 {payload['total_sec']}s")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())