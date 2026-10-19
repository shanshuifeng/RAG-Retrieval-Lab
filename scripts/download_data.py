"""下载数据集 —— RAG-Retrieval-Lab (M1)

用法:
    .venv\\Scripts\\python.exe scripts\\download_data.py               # 默认下载 DuRetrieval
    .venv\\Scripts\\python.exe scripts\\download_data.py --dataset t2  # 下载 T2Retrieval
    .venv\\Scripts\\python.exe scripts\\download_data.py --dry-run     # 只检查不下载

⭐ 设计要点:
    · 走 hf-mirror.com 镜像（国内可达，已实测）
    · 支持断点续传（大文件）
    · 自动查找 qrels（相关性标注，可能在独立 -qrels 仓库）
    · 校验文件大小
    · 全部元数据落盘，便于复现
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ────────────────────────────────────────────────────────────
# 配置
# ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# ⭐ 已实测可用的数据集（2026-10-17 探测）
DATASETS = {
    "dureval": {
        "name": "C-MTEB/DuRetrieval",
        "desc": "DuReader-retrieval 的 C-MTEB 版本（中文段落检索）",
        "qrels_candidates": [
            "C-MTEB/DuRetrieval-qrels",
            "mteb/DuRetrieval-qrels",
        ],
        "expected_mb": 61.5,
    },
    "t2": {
        "name": "C-MTEB/T2Retrieval",
        "desc": "T2Ranking（大规模中文段落排序）",
        "qrels_candidates": [
            "C-MTEB/T2Retrieval-qrels",
            "mteb/T2Retrieval-qrels",
        ],
        "expected_mb": 150.3,
    },
}

S = requests.Session()
S.headers.update({"User-Agent": "RAG-Retrieval-Lab/0.1"})


def hr(c: str = "=", w: int = 72) -> None:
    print(c * w)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ────────────────────────────────────────────────────────────
# 下载
# ────────────────────────────────────────────────────────────
def download_file(url: str, dest: Path, expected_size: int | None = None,
                  retries: int = 3) -> bool:
    """下载单个文件，支持断点续传。"""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 已存在且大小匹配 → 跳过
    if dest.exists():
        if expected_size and dest.stat().st_size == expected_size:
            print(f"    [跳过] 已存在且大小正确: {dest.name}")
            return True
        print(f"    [续传] 已存在部分文件 ({human(dest.stat().st_size)})")

    for attempt in range(1, retries + 1):
        try:
            pos = dest.stat().st_size if dest.exists() else 0
            headers = {"Range": f"bytes={pos}-"} if pos else {}

            with S.get(url, stream=True, timeout=120, headers=headers,
                       allow_redirects=True) as r:
                if r.status_code == 416:           # 已下载完整
                    print(f"    [完成] 文件已完整: {dest.name}")
                    return True
                if r.status_code >= 400:
                    raise RuntimeError(f"HTTP {r.status_code}")

                # 服务器若不支持 range，会返回 200 并从头开始
                if pos and r.status_code == 200:
                    print("    [注意] 服务器不支持断点续传，从头下载")
                    pos = 0

                total = int(r.headers.get("Content-Length", 0)) + pos
                mode = "ab" if pos else "wb"

                with open(dest, mode) as f, tqdm(
                    total=total or None,
                    initial=pos,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"      {dest.name[:32]}",
                    ncols=80,
                    leave=False,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

            # 校验
            actual = dest.stat().st_size
            if expected_size and actual != expected_size:
                raise RuntimeError(
                    f"大小不符: 期望 {expected_size}, 实际 {actual}"
                )
            print(f"    [OK] {dest.name}  ({human(actual)})")
            return True

        except Exception as e:  # noqa: BLE001
            print(f"    [!] 第 {attempt}/{retries} 次失败: {type(e).__name__}: {e}")
            if attempt < retries:
                wait = 2 ** attempt
                print(f"        等待 {wait}s 后重试…")
                time.sleep(wait)

    return False


def list_files(ds_id: str) -> list[dict]:
    """列出数据集仓库的文件。"""
    r = S.get(f"{HF_MIRROR}/api/datasets/{ds_id}", timeout=30)
    if not r.ok:
        return []
    return r.json().get("siblings") or []


def file_size(ds_id: str, fname: str) -> int | None:
    url = f"{HF_MIRROR}/datasets/{ds_id}/resolve/main/{fname}"
    try:
        r = S.head(url, timeout=30, allow_redirects=True,
                   headers={"Accept-Encoding": "identity"})
        if r.status_code < 400:
            n = r.headers.get("Content-Length")
            return int(n) if n and n.isdigit() else None
    except Exception:  # noqa: BLE001
        pass
    return None


def download_dataset(key: str, dry_run: bool = False) -> dict:
    """下载一个数据集的全部文件。"""
    cfg = DATASETS[key]
    ds_id = cfg["name"]

    hr()
    print(f"  下载: {ds_id}")
    print(f"  说明: {cfg['desc']}")
    hr()

    files = list_files(ds_id)
    if not files:
        print(f"  [X] 无法获取文件列表（{ds_id}）")
        return {"ok": False, "error": "无法获取文件列表"}

    out_dir = RAW / key
    manifest = {
        "dataset_key": key,
        "dataset_id": ds_id,
        "desc": cfg["desc"],
        "mirror": HF_MIRROR,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": [],
        "qrels": [],
    }

    print(f"\n  目标目录: {out_dir}")
    print(f"  文件数  : {len(files)}\n")

    n_ok = 0
    for s in files:
        fname = s.get("rfilename")
        if not fname or fname in (".gitattributes",):
            continue

        url = f"{HF_MIRROR}/datasets/{ds_id}/resolve/main/{fname}"
        size = file_size(ds_id, fname)

        if dry_run:
            print(f"    [DRY] {fname:50s} {human(size) if size else '?'}")
            n_ok += 1
            continue

        ok = download_file(url, out_dir / fname, size)
        if ok:
            n_ok += 1
            p = out_dir / fname
            manifest["files"].append({
                "name": fname,
                "size": p.stat().st_size,
                "sha256": sha256_file(p) if p.stat().st_size < 200 * 1024 * 1024 else None,
            })

    # ⭐ 找 qrels（相关性标注）
    print("\n  ── 查找 qrels（相关性标注）──")
    for qid in cfg["qrels_candidates"]:
        try:
            r = S.get(f"{HF_MIRROR}/api/datasets/{qid}", timeout=25)
            if not r.ok:
                print(f"    [--] {qid}: HTTP {r.status_code}")
                continue
            qfiles = r.json().get("siblings") or []
            print(f"    [OK] 找到 {qid}（{len(qfiles)} 个文件）")

            qdir = out_dir / "qrels"
            for qs in qfiles:
                qf = qs.get("rfilename")
                if not qf or qf in (".gitattributes", "README.md"):
                    continue
                qurl = f"{HF_MIRROR}/datasets/{qid}/resolve/main/{qf}"
                qsize = file_size(qid, qf)
                if dry_run:
                    print(f"      [DRY] {qf:46s} {human(qsize) if qsize else '?'}")
                    continue
                if download_file(qurl, qdir / qf, qsize):
                    manifest["qrels"].append({"from": qid, "name": qf,
                                              "size": (qdir / qf).stat().st_size})
            if manifest["qrels"]:
                break
        except Exception as e:  # noqa: BLE001
            print(f"    [X] {qid}: {type(e).__name__}")

    if not dry_run:
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n  ⭐ 清单已保存: {out_dir / 'manifest.json'}")

    return {"ok": n_ok > 0, "manifest": manifest, "out_dir": str(out_dir)}


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="下载中文检索数据集")
    ap.add_argument("--dataset", default="dureval", choices=list(DATASETS),
                    help="要下载的数据集（默认 dureval = C-MTEB/DuRetrieval）")
    ap.add_argument("--dry-run", action="store_true", help="只检查不下载")
    ap.add_argument("--all", action="store_true", help="下载全部数据集")
    args = ap.parse_args()

    hr("#")
    print("#  RAG-Retrieval-Lab · 数据集下载 (M1)")
    hr("#")
    print(f"  镜像   : {HF_MIRROR}")
    print(f"  目标   : {RAW}")
    print(f"  模式   : {'DRY-RUN（只检查）' if args.dry_run else '实际下载'}")

    keys = list(DATASETS) if args.all else [args.dataset]

    results = {}
    for k in keys:
        results[k] = download_dataset(k, dry_run=args.dry_run)

    hr("#")
    print("  下载结果汇总")
    hr("#")
    all_ok = True
    for k, r in results.items():
        status = "✅ 成功" if r.get("ok") else "❌ 失败"
        print(f"  {k:12s} {status}   {r.get('out_dir', '')}")
        if not r.get("ok"):
            all_ok = False

    hr("#")
    if all_ok and not args.dry_run:
        print("\n  ⭐ 下一步：检查数据内容")
        print(f"     .venv\\Scripts\\python.exe scripts\\inspect_data.py")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
