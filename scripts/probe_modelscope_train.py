"""探查 ModelScope 训练集（用于切分 dev）—— 只读探测，不下载

目的：
  现有 C-MTEB/DuRetrieval 只有 corpus(100,001) + queries(2,000)，
  没有 train split，无法按 docs/04 的方案切 dev。
  本脚本核实 ModelScope 训练集是否可用、文件清单与体积、
  以及它到底是「query-passage-label 三元组」还是「独立语料库」。

用法：
  .venv\\Scripts\\python.exe scripts\\probe_modelscope_train.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent

# ⭐ 候选数据集（docs/04 记录的 + 常见镜像）
CANDIDATES = [
    ("modelscope", "zyznull/dureader-retrieval-ranking"),
    ("modelscope", "zyznull/dureader-retrieval-corpus"),
    ("hf", "zyznull/dureader-retrieval-ranking"),
    ("hf", "zyznull/dureader-retrieval-corpus"),
]

MS_API = "https://www.modelscope.cn/api/v1"
HF_MIRROR = "https://hf-mirror.com"

S = requests.Session()
S.headers.update({"User-Agent": "RAG-Retrieval-Lab/0.1"})


def hr(c: str = "=", w: int = 74) -> None:
    print(c * w)


def human(n: float | None) -> str:
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def ms_dataset_info(ds_id: str) -> dict | None:
    """ModelScope 数据集元信息。"""
    url = f"{MS_API}/datasets/{ds_id}"
    try:
        r = S.get(url, timeout=30)
    except Exception as e:  # noqa: BLE001
        print(f"  [ERR] {url} -> {type(e).__name__}: {e}")
        return None
    print(f"  GET {url} -> HTTP {r.status_code}")
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:  # noqa: BLE001
        return None


def ms_file_list(ds_id: str) -> list[dict]:
    """ModelScope 数据集文件树（递归）。"""
    url = f"{MS_API}/datasets/{ds_id}/repo/files"
    params = {"Revision": "master", "Recursive": "true"}
    try:
        r = S.get(url, params=params, timeout=30)
    except Exception as e:  # noqa: BLE001
        print(f"  [ERR] {url} -> {type(e).__name__}: {e}")
        return []
    print(f"  GET {url} -> HTTP {r.status_code}")
    if r.status_code != 200:
        return []
    try:
        data = r.json().get("Data", {}) or {}
    except Exception:  # noqa: BLE001
        return []
    files = data.get("Files") or []
    out = []
    for f in files:
        out.append(
            {
                "path": f.get("Path"),
                "size": f.get("Size"),
                "type": f.get("Type"),
            }
        )
    return out


def hf_file_list(ds_id: str) -> list[dict]:
    """HuggingFace 镜像文件树。"""
    url = f"{HF_MIRROR}/api/datasets/{ds_id}/tree/main"
    try:
        r = S.get(url, params={"recursive": "true"}, timeout=30)
    except Exception as e:  # noqa: BLE001
        print(f"  [ERR] {url} -> {type(e).__name__}: {e}")
        return []
    print(f"  GET {url} -> HTTP {r.status_code}")
    if r.status_code != 200:
        return []
    try:
        return [
            {"path": it.get("path"), "size": it.get("size"), "type": it.get("type")}
            for it in r.json()
        ]
    except Exception:  # noqa: BLE001
        return []


def show_files(files: list[dict], limit: int = 40) -> None:
    if not files:
        print("  (无文件清单)")
        return
    total = 0
    for f in files:
        if f.get("type") == "tree":
            continue
        total += f.get("size") or 0
    print(f"  共 {len(files)} 项，合计 {human(total)}")
    for f in files[:limit]:
        mark = "/" if f.get("type") == "tree" else " "
        print(f"    {mark} {f.get('path'):<58} {human(f.get('size'))}")
    if len(files) > limit:
        print(f"    ... 其余 {len(files) - limit} 项省略")


def check_disk() -> None:
    hr()
    print("  磁盘空间")
    hr()
    usage = shutil.disk_usage(ROOT)
    print(f"  可用 {human(usage.free)} / 总 {human(usage.total)}   ({ROOT})")


def main() -> int:
    hr("#")
    print("  ModelScope 训练集探查（只读）")
    hr("#")

    for kind, ds_id in CANDIDATES:
        hr()
        print(f"  【{kind}】{ds_id}")
        hr()
        if kind == "modelscope":
            info = ms_dataset_info(ds_id)
            if info:
                d = info.get("Data", {}) or {}
                for key in ("Name", "ChineseName", "Description", "Downloads", "License"):
                    if d.get(key):
                        val = str(d[key]).replace("\n", " ")[:160]
                        print(f"    {key}: {val}")
                print(f"    Raw 顶层键: {list(info.keys())}")
            files = ms_file_list(ds_id)
            show_files(files)
        else:
            files = hf_file_list(ds_id)
            show_files(files)

    check_disk()
    hr("#")
    print("  结论请人工判读：训练集是三元组还是独立语料库？体积是否可接受？")
    hr("#")
    return 0


if __name__ == "__main__":
    sys.exit(main())