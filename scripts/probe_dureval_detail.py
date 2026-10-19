"""探查 C-MTEB/DuRetrieval 的具体文件与体积

用法:
    .venv\\Scripts\\python.exe scripts\\probe_dureval_detail.py
"""

from __future__ import annotations

import sys

import requests

HF_MIRROR = "https://hf-mirror.com"
S = requests.Session()
S.headers.update({"User-Agent": "RAG-Retrieval-Lab/0.1"})


def hr(c: str = "=", w: int = 72) -> None:
    print(c * w)


def head_size(url: str) -> tuple[int | None, str]:
    """用 HEAD 拿文件大小。。"""
    try:
        r = S.head(url, timeout=30, allow_redirects=True, headers={"Accept-Encoding": "identity"})
        if r.status_code < 400:
            size = r.headers.get("Content-Length") or r.headers.get("x-linked-size")
            return (int(size) if size and str(size).isdigit() else None,
                    r.headers.get("Content-Type", "-"))
        return (None, f"HTTP {r.status_code}")
    except Exception as e:  # noqa: BLE001
        return (None, f"{type(e).__name__}")


def human(n: int | None) -> str:
    if n is None:
        return "?"
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def inspect(ds_id: str) -> None:
    hr()
    print(f"  数据集: {ds_id}")
    hr()

    # 1) 元数据 / 文件清单
    try:
        r = S.get(f"{HF_MIRROR}/api/datasets/{ds_id}", timeout=30)
        if not r.ok:
            print(f"  [X] HTTP {r.status_code}")
            return
        info = r.json()
    except Exception as e:  # noqa: BLE001
        print(f"  [X] {type(e).__name__}: {e}")
        return

    siblings = info.get("siblings") or []
    print(f"\n  文件数: {len(siblings)}")
    print(f"\n  {'文件':52s} {'大小':>12s}")
    print("  " + "-" * 68)

    total = 0
    for s in siblings:
        fname = s.get("rfilename", "?")
        # 尝试 resolve URL 的 HEAD
        url = f"{HF_MIRROR}/datasets/{ds_id}/resolve/main/{fname}"
        size, ctype = head_size(url)
        if size:
            total += size
        print(f"  {fname:52s} {human(size):>12s}")

    print("  " + "-" * 68)
    print(f"  合计约: {human(total)}")

    # 2) cardData / splits 信息
    card = info.get("cardData") or {}
    print(f"\n  license     : {card.get('license', '-')}")
    print(f"  pretty_name : {card.get('pretty_name', '-')}")
    cfg = card.get("configs")
    if cfg:
        print(f"  configs     : {str(cfg)[:200]}")
    langs = card.get("language")
    if langs:
        print(f"  language    : {langs}")


def main() -> int:
    hr("#")
    print("#  探查 C-MTEB 中文检索数据集详情")
    hr("#")

    for ds in ["C-MTEB/DuRetrieval", "C-MTEB/T2Retrieval"]:
        inspect(ds)

    hr("#")
    return 0


if __name__ == "__main__":
    sys.exit(main())
