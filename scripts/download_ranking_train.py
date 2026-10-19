"""下载 DuReader-retrieval 训练集（ModelScope/HF 镜像 train.jsonl.gz, 3.4GB）

背景（2026-10-17 实测）：
  · C-MTEB/DuRetrieval 的 2,000 query 与 zyznull/dureader-retrieval-ranking
    的 dev.jsonl.gz 是【同一批 query】（文本交集 100%）→ 那就是官方测试集，
    ⚠️ 绝不可当 dev 用。
  · 因此 dev 必须从 train（86,395 query）里切 → 本脚本下载训练集。

特性：
  · 断点续传（HTTP Range；若服务器不支持则重头下）
  · 边下边写进度到 logs/，可用 --tail 查看
  · 完成后写 sha256 + manifest.json

用法：
  .venv\\Scripts\\python.exe scripts\\download_ranking_train.py
  .venv\\Scripts\\python.exe scripts\\download_ranking_train.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "dureader-ranking"
DEST = OUT_DIR / "train.jsonl.gz"
MANIFEST = OUT_DIR / "manifest.json"
LOG = ROOT / "logs" / "download_train.log"

URL = ("https://hf-mirror.com/datasets/zyznull/dureader-retrieval-ranking"
       "/resolve/main/train.jsonl.gz")
EXPECTED = 3_400_000_000  # 约 3.4 GB，仅用于进度显示

S = requests.Session()
S.headers.update({"User-Agent": "RAG-Retrieval-Lab/0.1"})


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def remote_size() -> int | None:
    try:
        r = S.head(URL, timeout=60, allow_redirects=True)
        n = r.headers.get("Content-Length")
        return int(n) if n else None
    except Exception as e:  # noqa: BLE001
        log(f"HEAD 失败: {type(e).__name__}: {e}")
        return None


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_remote = remote_size()
    if total_remote:
        log(f"远端大小: {human(total_remote)}")

    have = DEST.stat().st_size if DEST.exists() else 0
    if total_remote and have == total_remote:
        log("文件已完整，跳过下载")
    else:
        headers = {}
        mode = "wb"
        if have and total_remote and have < total_remote:
            headers["Range"] = f"bytes={have}-"
            log(f"断点续传：已有 {human(have)}")
        else:
            have = 0

        t0 = time.time()
        with S.get(URL, stream=True, timeout=180, headers=headers) as r:
            if headers and r.status_code == 200:
                log("服务器不支持 Range（200），从零重下")
                have = 0
                mode = "wb"
            elif r.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {r.status_code}")
            else:
                mode = "ab" if headers and r.status_code == 206 else "wb"
            cl = r.headers.get("Content-Length")
            expect = int(cl) + have if cl else None
            if expect:
                log(f"本次需下 {human(int(cl) if cl else 0)}，目标总计 {human(expect)}")

            got = have
            last = time.time()
            with open(DEST, mode) as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    got += len(chunk)
                    if time.time() - last > 5:
                        sp = got / max(1e-9, time.time() - t0)
                        pct = 100 * got / expect if expect else 0
                        log(f"进度 {human(got)} / {human(expect) if expect else '?'}"
                            f"  ({pct:.1f}%)  {human(sp)}/s")
                        last = time.time()
        log(f"下载完成: {human(DEST.stat().st_size)}，用时 {time.time()-t0:.0f}s")

    size = DEST.stat().st_size
    log("计算 sha256 ...")
    digest = sha256_file(DEST)
    log(f"sha256 = {digest}")

    record = {
        "dataset_id": "zyznull/dureader-retrieval-ranking (HF 镜像 hf-mirror.com)",
        "role": "dev 候选来源（从 train 切 dev）",
        "file": DEST.name,
        "size": size,
        "sha256": digest,
        "source_url": URL,
        "downloaded_at": datetime.now().isoformat(timespec="seconds"),
        "note": "dev.jsonl.gz 已核实为官方测试集（与 C-MTEB 2,000 query 同一批），"
                "不可用作 dev；dev 从本 train 文件切分。",
    }
    MANIFEST.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"manifest -> {MANIFEST.relative_to(ROOT)}")
    return 0


def check() -> int:
    print(f"目标文件: {DEST}")
    print(f"存在: {DEST.exists()}")
    if DEST.exists():
        print(f"大小: {human(DEST.stat().st_size)}")
    print(f"远端大小: {human(remote_size()) if remote_size() else '未知'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 DuReader-retrieval 训练集")
    ap.add_argument("--check", action="store_true", help="只检查不下载")
    args = ap.parse_args()
    return check() if args.check else download()


if __name__ == "__main__":
    sys.exit(main())