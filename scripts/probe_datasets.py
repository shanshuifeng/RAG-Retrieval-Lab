"""探查数据源可用性 —— 只读探测，不下载

用法:
    .venv\\Scripts\\python.exe scripts\\probe_datasets.py
"""

from __future__ import annotations

import json
import sys
import time

import requests

HF_MIRROR = "https://hf-mirror.com"
HF_API = f"{HF_MIRROR}/api/datasets"
MS_API = "https://www.modelscope.cn/api/v1/dolphin/datasets"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "RAG-Retrieval-Lab/0.1 (probe)"})


def hr(c: str = "=", w: int = 72) -> None:
    print(c * w)


def probe_hf_search(queries: list[str]) -> list[dict]:
    """在 HF 镜像上搜索数据集。"""
    hr()
    print("  HuggingFace 镜像 (hf-mirror.com) 数据集搜索")
    hr()

    found: dict[str, dict] = {}
    for q in queries:
        try:
            r = SESSION.get(HF_API, params={"search": q, "limit": 20}, timeout=30)
            if not r.ok:
                print(f"  [!] 搜索 '{q}' 返回 HTTP {r.status_code}")
                continue
            items = r.json()
            print(f"\n  ── 搜索 '{q}' → {len(items)} 个结果 ──")
            for it in items[:8]:
                pid = it.get("id", "?")
                dl = it.get("downloads", 0)
                likes = it.get("likes", 0)
                print(f"      {pid:48s} dl={dl:<10} likes={likes}")
                found.setdefault(pid, it)
        except Exception as e:  # noqa: BLE001
            print(f"  [X] 搜索 '{q}' 异常: {type(e).__name__}: {e}")

    return list(found.values())


def probe_dataset_files(ds_id: str) -> dict | None:
    """查看某个 HF 数据集的文件清单与大小。"""
    try:
        url = f"{HF_API}/{ds_id}"
        r = SESSION.get(url, timeout=30)
        if not r.ok:
            print(f"  [X] {ds_id}: HTTP {r.status_code}")
            return None

        info = r.json()
        siblings = info.get("siblings") or []
        print(f"\n  ── {ds_id} ──")
        print(f"      文件数: {len(siblings)}")
        for s in siblings[:20]:
            print(f"        {s.get('rfilename', '?')}")

        # 尝试拿下载量/描述
        card = info.get("cardData") or {}
        print(f"      描述: {str(card.get('pretty_name') or card.get('license') or '-')[:80]}")
        return info
    except Exception as e:  # noqa: BLE001
        print(f"  [X] 查 {ds_id} 异常: {type(e).__name__}: {e}")
        return None


def probe_specific_candidates() -> None:
    """探测已知的几个候选数据集 ID 是否真实存在。"""
    hr()
    print("  候选数据集 ID 探测")
    hr()

    candidates = [
        # DuReader 相关
        "zyznull/dureader-retrieval-ranking",
        "zyznull/dureader-retrieval-corpus",
        "C-MTEB/DuRetrieval",
        "C-MTEB/T2Retrieval",
        "C-MTEB/MMarcoRetrieval",
        # 常见中文检索数据集
        "mteb/T2Retrieval",
        "mteb/DuRetrieval",
        "shibing624/nli_zh",
    ]

    for cid in candidates:
        try:
            r = SESSION.get(f"{HF_API}/{cid}", timeout=25)
            if r.ok:
                info = r.json()
                n_files = len(info.get("siblings") or [])
                dl = info.get("downloads", "?")
                print(f"  [OK]  {cid:42s} files={n_files:<4} downloads={dl}")
            else:
                print(f"  [--]  {cid:42s} HTTP {r.status_code}（不存在或无权限）")
        except Exception as e:  # noqa: BLE001
            print(f"  [X]   {cid:42s} {type(e).__name__}: {str(e)[:50]}")
        time.sleep(0.3)


def probe_modelscope() -> None:
    """探测 ModelScope 数据集。"""
    hr()
    print("  ModelScope 数据集探测")
    hr()

    ids = [
        "zyznull/dureader-retrieval-ranking",
        "iic/nlp_dureader_retrieval",
        "AI-ModelScope/duReader_robust",
    ]

    for did in ids:
        try:
            # ModelScope 数据集详情接口
            url = f"https://www.modelscope.cn/api/v1/datasets/{did}"
            r = SESSION.get(url, timeout=25)
            if r.ok:
                j = r.json()
                data = j.get("Data") or {}
                name = data.get("Name") or did
                print(f"  [OK]  {did}")
                print(f"        Name={name}")
                desc = (data.get("Description") or "")[:100]
                if desc:
                    print(f"        Desc={desc}")
            else:
                print(f"  [--]  {did:44s} HTTP {r.status_code}")
        except Exception as e:  # noqa: BLE001
            print(f"  [X]   {did:44s} {type(e).__name__}: {str(e)[:50]}")
        time.sleep(0.3)


def probe_disk() -> None:
    """检查磁盘余量。"""
    hr()
    print("  磁盘与目录")
    hr()
    import shutil
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    usage = shutil.disk_usage(str(root))
    print(f"  项目路径 : {root}")
    print(f"  可用空间 : {usage.free / 1024**3:.1f} GB")
    print(f"  data/raw : {'存在' if (root / 'data' / 'raw').exists() else '不存在'}")


def main() -> int:
    hr("#")
    print("#  数据源可用性探测（只读，不下载）")
    hr("#")

    probe_disk()
    probe_specific_candidates()
    probe_modelscope()
    items = probe_hf_search(["dureader", "DuRetrieval", "T2Retrieval", "chinese retrieval"])

    if items:
        print("\n  ⭐ 建议进一步查看：")
        for it in sorted(items, key=lambda x: -(x.get("downloads") or 0))[:5]:
            print(f"     {it.get('id')}")

    hr("#")
    print("  探测完成 —— 以上结果用于决定实际的下载方案")
    hr("#")
    return 0


if __name__ == "__main__":
    sys.exit(main())
