"""多连接并行下载器（位置写入 + 断点续传）

背景（实测 2026-10-17）：
  · hf-mirror 单连接被限速 ~325 KB/s（3.36 GB 要 3 小时）
  · 8 并发实测 ~1.9 MB/s，16 并发更快
  · ModelScope 的 zyznull/dureader-retrieval-ranking 仓库里【没有】jsonl 数据文件
    （只有 loader 代码），拿不到国内直连大文件

做法：
  · 已下载的前缀（train.jsonl.gz 现有内容）原地保留，先 truncate 到远端总长
  · 剩余区间按 chunk 切分，多线程各自 Range 下载，用独立文件句柄 seek 定位写入
    （Windows 无 os.pwrite，故每线程独立 "r+b" 句柄）
  · chunk 先在内存中完整收齐再一次性写入，写完后才记入 state → 中途被杀不会留下
    「半截却算完成」的 chunk
  · state 文件记录已完成 chunk，重启自动跳过

用法：
  .venv\\Scripts\\python.exe scripts\\download_parallel.py
  .venv\\Scripts\\python.exe scripts\\download_parallel.py --verify
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "dureader-ranking"
DEST = OUT_DIR / "train.jsonl.gz"
STATE = OUT_DIR / "download_state.json"
LOG = ROOT / "logs" / "download_train_parallel.log"

URL = ("https://hf-mirror.com/datasets/zyznull/dureader-retrieval-ranking"
       "/resolve/main/train.jsonl.gz")

CHUNK_MB = 2
THREADS = 12
UA = "RAG-Retrieval-Lab/0.1"

# ⭐ 实测结论（2026-10-18 00:1x）：
#   数据实际托管在 cas-bridge.xethub.hf.co，长连接会被【渐进限速】——
#   单连接 3 分钟掉到 ~300 KB/s，16 线程 64MB 分块合计仅 ~470 KB/s；
#   改为「每块全新连接 + 小分块」后实测 6×2MB=2172 KB/s、12×2MB=2604 KB/s、
#   12×1MB=2724 KB/s，提速 ~5 倍。故默认 2MB 分块 + 12 线程 + 每块新连接。
TIMEOUT = (15, 60)

_tls = threading.local()
# ⭐ 必须用 RLock：worker 持锁时调用 save_state()，后者也要拿锁
#    （普通 Lock 不可重入 → 2026-10-18 00:38 实测在第 25 块完成时全体死锁）
_lock = threading.RLock()
_done_bytes = 0


def session() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": UA})
        _tls.s = s
    return s


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def log(msg: str, quiet: bool = False) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    if not quiet:
        print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def remote_total() -> int:
    r = session().get(URL, headers={"Range": "bytes=0-0"}, timeout=60, stream=True)
    if r.status_code == 206:
        cr = r.headers.get("Content-Range", "")
        r.close()
        return int(cr.split("/")[-1])
    r.close()
    h = session().head(URL, timeout=60, allow_redirects=True)
    return int(h.headers["Content-Length"])


def valid_prefix_len(path: Path, cap: int) -> int:
    """返回【可证明有效】的 gzip 前缀字节数。

    ⭐ 不能用「文件大小」当前缀：中断时文件可能已被 truncate 预分配到全长，
       尾部是 0 填充。也不能只用 zlib 探测：截断的 deflate 流后跟零字节时
       zlib 可能静默吞掉而不报错（2026-10-18 00:31 实测翻车）。

    可靠做法：把已写 chunk 的 (offset, sha256) 记进 state，重启时逐块
    重算 sha256，只有校验通过的块才算「已完成」。本函数现在只处理
    「从未写过任何块」的干净文件：大小 = 0 → 0；否则按旧语义探测（已弃用路径）。
    """
    if not path.exists() or path.stat().st_size == 0:
        return 0
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    consumed = 0
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                if consumed + len(b) > cap:
                    b = b[: cap - consumed]
                    if not b:
                        break
                d.decompress(b)
                consumed += len(b)
    except zlib.error:
        consumed = max(0, (consumed // (1 << 20)) - 1) * (1 << 20)
    return min(consumed, cap)


def load_state(total: int, chunk_mb: int) -> dict:
    """读取状态；分块大小或 URL 变了就重开（chunk 哈希表结构随之失效）。"""
    prev = {}
    if STATE.exists():
        try:
            prev = json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev = {}
    if prev.get("url") == URL and prev.get("total") == total \
            and prev.get("chunk_mb") == chunk_mb:
        return prev
    return {"url": URL, "total": total, "chunk_mb": chunk_mb,
            "prefix_len": 0, "chunks": {}}


def verify_chunks(st: dict) -> dict:
    """⭐ 逐块校验 state 里已完成的 chunk（sha256）。

    只有哈希匹配的块才保留——这是对抗「半写块 / 零填充」的唯一可靠手段
    （2026-10-18 00:31 事故：zlib 探测被零填充骗过，进入校验才 EOFError）。
    返回更新后的 state。
    """
    chunks_info = st.get("chunks", {})
    if not chunks_info:
        return st
    csz = (st.get("chunk_mb") or CHUNK_MB) << 20
    keep = {}
    total_n = len(chunks_info)
    log(f"校验 {total_n} 个已记录 chunk 的 sha256 ...")
    with open(DEST, "rb") as f:
        for idx_s, meta in chunks_info.items():
            start = int(meta["start"])
            end = int(meta["end"])
            want = meta["sha256"]
            if end - start + 1 != min(csz, st["total"] - start):
                continue  # 长度不符（旧分块方案），丢弃
            f.seek(start)
            h = hashlib.sha256(f.read(end - start + 1)).hexdigest()
            if h == want:
                keep[idx_s] = meta
    dropped = total_n - len(keep)
    st["chunks"] = keep
    if dropped:
        log(f"  丢弃损坏/可疑 chunk {dropped} 个，保留 {len(keep)} 个")
    else:
        log(f"  全部 {len(keep)} 个 chunk 校验通过")
    return st


def save_state(st: dict) -> None:
    with _lock:
        STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


def grab(idx: int, start: int, end: int) -> tuple[int, str]:
    """下载 [start,end] → 定位写入 → 返回 (字节数, sha256)。

    ⭐ 每个 chunk 用【全新连接】（Connection: close，不复用 Session），
       以规避 cas-bridge 对长连接的渐进限速。
    ⭐ 先在内存收齐并校验长度，再一次性落盘 + 算哈希；state 只记
       「哈希确认过」的块，中途被杀不会产生半截却标记完成的块。
    """
    need = end - start + 1
    buf = bytearray()
    for attempt in range(5):
        try:
            buf.clear()
            headers = {"User-Agent": UA, "Connection": "close",
                       "Range": f"bytes={start}-{end}"}
            with requests.get(URL, headers=headers, stream=True,
                              timeout=TIMEOUT) as r:
                if r.status_code != 206:
                    raise RuntimeError(f"HTTP {r.status_code}")
                for c in r.iter_content(1 << 20):
                    buf.extend(c)
            if len(buf) != need:
                raise RuntimeError(f"长度不符 {len(buf)} != {need}")
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            log(f"  chunk {idx} 第 {attempt+1} 次失败（{type(e).__name__}: {e}），重试")
            time.sleep(1 + attempt)
    digest = hashlib.sha256(buf).hexdigest()
    with open(DEST, "r+b") as f:
        f.seek(start)
        f.write(buf)
    return len(buf), digest


def download() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=" * 66)
    log("  并行下载 train.jsonl.gz（2MB 分块 / 每块新连接 / sha256 记账）")
    log("=" * 66)

    total = remote_total()
    log(f"远端总大小: {human(total)}  ({total:,} bytes)")

    st = load_state(total, CHUNK_MB)

    if not DEST.exists():
        with open(DEST, "wb") as f:
            f.truncate(total)
    elif DEST.stat().st_size != total:
        log(f"本地文件 {human(DEST.stat().st_size)} ≠ 远端 {human(total)}，重建预分配")
        with open(DEST, "wb") as f:
            f.truncate(total)

    st = verify_chunks(st)
    done_hash = st["chunks"]

    csz = CHUNK_MB << 20
    chunks = []
    pos = 0
    idx = 0
    while pos < total:
        end = min(pos + csz - 1, total - 1)
        chunks.append((idx, pos, end))
        pos = end + 1
        idx += 1
    todo = [c for c in chunks if str(c[0]) not in done_hash]
    base = sum(c[2] - c[1] + 1 for c in chunks if str(c[0]) in done_hash)
    log(f"chunk 总数 {len(chunks)}，哈希确认 {len(done_hash)}（{human(base)}），"
        f"待下 {len(todo)}（每块 {CHUNK_MB} MB，并发 {THREADS}）")

    t0 = time.time()
    global _done_bytes
    _done_bytes = base

    def report(final: bool = False) -> None:
        el = max(1e-6, time.time() - t0)
        sp = (_done_bytes - base) / el
        pct = 100.0 * _done_bytes / total
        eta = (total - _done_bytes) / sp if sp > 0 else float("inf")
        log(f"进度 {human(_done_bytes)}/{human(total)} ({pct:.1f}%)"
            f"  速度 {human(sp)}/s  ETA {eta/60:.1f} 分钟"
            + ("  ✅ 完成" if final else ""))

    def worker(c: tuple[int, int, int]) -> int:
        global _done_bytes
        i, s, e = c
        n, digest = grab(i, s, e)
        with _lock:
            _done_bytes += n
            st["chunks"][str(i)] = {"start": s, "end": e, "sha256": digest}
            if len(st["chunks"]) % 25 == 0:
                save_state(st)
        return n

    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futs = [ex.submit(worker, c) for c in todo]
        while True:
            pending = [f for f in futs if not f.done()]
            report()
            if not pending:
                break
            time.sleep(5)
        for f in futs:
            f.result()  # 抛出异常

    save_state(st)
    report(final=True)
    size = DEST.stat().st_size
    if size != total:
        log(f"⚠️ 最终大小 {size:,} != 远端 {total:,}")
        return 1
    log("✅ 字节数一致，进入 gzip 完整性校验 ...")
    return verify()


def verify() -> int:
    log("=" * 66)
    log("  gzip 完整性 + 行数校验（顺带就是 dev 切分的第一遍统计）")
    log("=" * 66)
    n = 0
    bad = 0
    qids = set()
    with gzip.open(DEST, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            try:
                j = json.loads(line)
                qids.add(str(j.get("query_id")))
            except Exception:  # noqa: BLE001
                bad += 1
            if n % 10000 == 0:
                log(f"  已读 {n:,} 行 ...", quiet=True)
    log(f"✅ gzip 解压无损：{n:,} 行，JSON 解析失败 {bad}，唯一 query_id {len(qids):,}")
    if bad:
        log("⚠️ 存在无法解析的行，需人工检查")
    return 0 if not bad else 1


def main() -> int:
    global THREADS, CHUNK_MB
    ap = argparse.ArgumentParser(description="并行下载 + 校验")
    ap.add_argument("--verify", action="store_true", help="只做完整性校验")
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--chunk-mb", type=int, default=CHUNK_MB)
    args = ap.parse_args()
    THREADS, CHUNK_MB = args.threads, args.chunk_mb
    return verify() if args.verify else download()


if __name__ == "__main__":
    sys.exit(main())