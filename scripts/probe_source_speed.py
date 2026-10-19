"""测速 + 跨源一致性校验（决定 train.jsonl.gz 从哪个源下、能否续传）

背景：
  hf-mirror 单连接只有 ~500 KB/s，3.36 GB 要近 2 小时。
  本项目是国内环境，ModelScope 直连通常更快。

本脚本：
  1. 对比 hf-mirror 单连接 / hf-mirror 4 并发 / ModelScope 直连 的实测速度
  2. 取两个源的头部 4 MB 做 sha256 比对 —— 一致才允许跨源断点续传
  3. 打印各源 Content-Length，验证文件是否同一份

用法：
  .venv\\Scripts\\python.exe scripts\\probe_source_speed.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PARTIAL = ROOT / "data" / "raw" / "dureader-ranking" / "train.jsonl.gz"
OUT = ROOT / "data" / "raw" / "dureader-ranking" / "source_speed.json"

HF = ("https://hf-mirror.com/datasets/zyznull/dureader-retrieval-ranking"
      "/resolve/main/train.jsonl.gz")
MS = ("https://www.modelscope.cn/api/v1/datasets/zyznull/dureader-retrieval-ranking"
      "/repo?Revision=master&FilePath=train.jsonl.gz")

S = requests.Session()
S.headers.update({"User-Agent": "RAG-Retrieval-Lab/0.1"})
CHUNK = 1 << 20


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def content_length(url: str) -> tuple[int | None, bool]:
    """返回 (大小, 是否支持 Range)。"""
    try:
        r = S.get(url, headers={"Range": "bytes=0-1023"}, timeout=60, stream=True)
        size = None
        if r.status_code == 206:
            cr = r.headers.get("Content-Range", "")
            if "/" in cr:
                try:
                    size = int(cr.split("/")[-1])
                except ValueError:
                    size = None
        elif r.status_code == 200:
            cl = r.headers.get("Content-Length")
            size = int(cl) if cl else None
        r.close()
        if size is None:
            h = S.head(url, timeout=60, allow_redirects=True)
            cl = h.headers.get("Content-Length")
            size = int(cl) if cl else None
        return size, r.status_code == 206
    except Exception as e:  # noqa: BLE001
        print(f"    [ERR] {type(e).__name__}: {e}")
        return None, False


def timed_get(url: str, nbytes: int, tag: str, headers: dict | None = None) -> dict:
    """下载 nbytes，返回速度与 sha256。"""
    h = dict(headers or {})
    h["Range"] = f"bytes=0-{nbytes - 1}" if not headers else h.get("Range", "")
    h = {k: v for k, v in h.items() if v}
    got = 0
    hsh = hashlib.sha256()
    t0 = time.time()
    try:
        with S.get(url, headers=h, stream=True, timeout=120) as r:
            code = r.status_code
            for chunk in r.iter_content(CHUNK):
                hsh.update(chunk)
                got += len(chunk)
                if got >= nbytes:
                    break
    except Exception as e:  # noqa: BLE001
        print(f"    [ERR] {tag}: {type(e).__name__}: {e}")
        return {"tag": tag, "bytes": got, "sec": 0, "speed": 0, "sha256": None,
                "status": None}
    dt = max(1e-6, time.time() - t0)
    res = {"tag": tag, "bytes": got, "sec": round(dt, 2),
           "speed_kbps": round(got / dt / 1024, 1), "sha256": hsh.hexdigest(),
           "status": code}
    print(f"    {tag}: {human(got)} in {dt:.1f}s = {res['speed_kbps']:.0f} KB/s"
          f"  (HTTP {code})")
    return res


def parallel_test(url: str, nthreads: int, per_thread: int, total_size: int) -> float:
    """并发分段下载，返回合计 KB/s。"""
    step = max(per_thread, total_size // (nthreads * 10))
    spans = [(i * step, i * step + per_thread - 1) for i in range(nthreads)]

    def one(span: tuple[int, int]) -> int:
        a, b = span
        try:
            with S.get(url, headers={"Range": f"bytes={a}-{b}"},
                       stream=True, timeout=120) as r:
                n = 0
                for chunk in r.iter_content(CHUNK):
                    n += len(chunk)
                return n
        except Exception:  # noqa: BLE001
            return 0

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        counts = list(ex.map(one, spans))
    dt = max(1e-6, time.time() - t0)
    total = sum(counts)
    speed = total / dt / 1024
    print(f"    {nthreads} 并发 × {human(per_thread)}: 共 {human(total)} in {dt:.1f}s"
          f" = {speed:.0f} KB/s")
    return speed


def main() -> int:
    print("=" * 74)
    print("  源测速 + 一致性校验")
    print("=" * 74)

    report: dict = {}

    print("\n  [1] Content-Length 与 Range 支持")
    hf_size, hf_range = content_length(HF)
    ms_size, ms_range = content_length(MS)
    print(f"    hf-mirror  : {human(hf_size) if hf_size else '?'}  Range={hf_range}")
    print(f"    ModelScope : {human(ms_size) if ms_size else '?'}  Range={ms_range}")
    same_size = hf_size is not None and hf_size == ms_size
    print(f"    两源大小一致: {same_size}")
    report["hf_size"] = hf_size
    report["ms_size"] = ms_size
    report["same_size"] = same_size

    print("\n  [2] 头部 4 MB 一致性（跨源续传的前提）")
    hf_head = timed_get(HF, 4 << 20, "hf-mirror 头4MB")
    ms_head = timed_get(MS, 4 << 20, "ModelScope 头4MB")
    same_head = (hf_head["sha256"] == ms_head["sha256"]) if (
        hf_head["sha256"] and ms_head["sha256"]) else None
    print(f"    头部 sha256 一致: {same_head}")
    if PARTIAL.exists():
        with open(PARTIAL, "rb") as f:
            local = hashlib.sha256(f.read(4 << 20)).hexdigest()
        print(f"    已下载分片头4MB 与 hf 一致: {local == hf_head['sha256']}")
        report["local_head_matches_hf"] = local == hf_head["sha256"]
    report["same_head"] = same_head
    report["hf_head_speed_kbps"] = hf_head["speed_kbps"]
    report["ms_head_speed_kbps"] = ms_head["speed_kbps"]

    print("\n  [3] 并发能力（hf-mirror）")
    total = hf_size or (3 << 30)
    s4 = parallel_test(HF, 4, 8 << 20, total)
    s8 = parallel_test(HF, 8, 8 << 20, total)
    report["hf_parallel4_kbps"] = round(s4, 1)
    report["hf_parallel8_kbps"] = round(s8, 1)

    print("\n  [4] 并发能力（ModelScope）")
    m4 = parallel_test(MS, 4, 8 << 20, ms_size or (3 << 30))
    m8 = parallel_test(MS, 8, 8 << 20, ms_size or (3 << 30))
    report["ms_parallel4_kbps"] = round(m4, 1)
    report["ms_parallel8_kbps"] = round(m8, 1)

    print("\n" + "=" * 74)
    print("  结论")
    print("=" * 74)
    best_single = max(hf_head["speed_kbps"], ms_head["speed_kbps"])
    best_par = max(s4, s8, m4, m8)
    for name, sp in (("hf单连接", hf_head["speed_kbps"]),
                     ("ms单连接", ms_head["speed_kbps"]),
                     ("hf并发", max(s4, s8)), ("ms并发", max(m4, m8))):
        eta = (hf_size / (sp * 1024) / 60) if hf_size and sp else float("inf")
        print(f"    {name:<10} {sp:>8.0f} KB/s   ETA {eta:>6.1f} 分钟")
    report["best_single_kbps"] = best_single
    report["best_parallel_kbps"] = best_par
    report["cross_source_resume_safe"] = bool(same_size and same_head)

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  报告: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())