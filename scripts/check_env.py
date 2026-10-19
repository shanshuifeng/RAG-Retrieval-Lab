"""环境检查脚本 —— RAG-Retrieval-Lab

用法:
    python scripts/check_env.py

⭐ 目的：一次性确认环境是否具备跑实验的条件，并明确告知"下一步该做什么"。

⚠️ 注意：本脚本【只读】检查，不安装任何东西。
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ────────────────────────────────────────────────────────────
# 项目根目录（本文件在 scripts/ 下，向上一级）
# ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ────────────────────────────────────────────────────────────
# 依赖清单：三档，对应安装策略的三个阶段
# ────────────────────────────────────────────────────────────
# 元组格式：(import 名, pip 包名, 说明)
# ⚠️ import 名与 pip 包名常不一致（如 yaml / PyYAML），必须分开记录，
#    否则给出错误的安装命令。
DEPS = {
    "阶段1 · 核心（必需）": [
        ("numpy", "numpy", "数值计算"),
        ("pandas", "pandas", "数据处理"),
        ("scipy", "scipy", "显著性检验"),
        ("yaml", "PyYAML", "配置文件"),
        ("tqdm", "tqdm", "进度条"),
        ("requests", "requests", "HTTP 下载"),
    ],
    "阶段2 · 检索（必需）": [
        ("jieba", "jieba", "中文分词"),
        ("rank_bm25", "rank-bm25", "BM25 稀疏检索"),
        ("torch", "torch", "深度学习后端"),
        ("transformers", "transformers", "模型加载"),
        ("sentence_transformers", "sentence-transformers", "句向量编码"),
        ("faiss", "faiss-cpu", "向量检索"),
        ("datasets", "datasets", "数据集加载"),
    ],
    "阶段3 · 可选（加分项）": [
        ("matplotlib", "matplotlib", "绘图"),
        ("openai", "openai", "LLM 生成（需 API Key）"),
    ],
}

# import 名 -> pip 包名 的映射（用于生成安装命令）
IMPORT_TO_PIP = {
    imp: pip for items in DEPS.values() for imp, pip, _ in items
}

# 目录清单
REQUIRED_DIRS = [
    "docs", "configs", "data", "data/raw", "data/processed", "data/index",
    "scripts", "src", "src/data", "src/retrieval", "src/rerank",
    "src/eval", "src/pipeline", "src/utils",
    "notebooks", "results", "results/figures", "logs",
]


# ────────────────────────────────────────────────────────────
# 输出工具
# ────────────────────────────────────────────────────────────
def hr(char: str = "=", width: int = 68) -> None:
    print(char * width)


def title(text: str) -> None:
    print()
    hr()
    print(f"  {text}")
    hr()


def ok(text: str) -> None:
    print(f"  [OK]   {text}")


def miss(text: str) -> None:
    print(f"  [MISS] {text}")


def warn(text: str) -> None:
    print(f"  [WARN] {text}")


def info(text: str) -> None:
    print(f"         {text}")


# ────────────────────────────────────────────────────────────
# 检查项
# ────────────────────────────────────────────────────────────
def check_python() -> bool:
    title("1. Python 版本")
    ver = sys.version_info
    print(f"  Python {ver.major}.{ver.minor}.{ver.micro}")
    print(f"  路径   {sys.executable}")

    if ver >= (3, 10):
        ok("版本满足要求（>= 3.10）")
        return True
    warn(f"版本偏低，建议 >= 3.10")
    return False


def check_deps() -> tuple[list[str], list[str]]:
    title("2. 依赖库")
    all_ok, all_missing = [], []

    for stage, items in DEPS.items():
        print(f"\n  ── {stage} ──")
        for mod_name, pip_name, desc in items:
            try:
                mod = importlib.import_module(mod_name)
                ver = getattr(mod, "__version__", "?")
                ok(f"{mod_name:22s} {str(ver):16s} {desc}")
                all_ok.append(mod_name)
            except ImportError:
                miss(f"{mod_name:22s} {'':16s} {desc}")
                all_missing.append(mod_name)

    return all_ok, all_missing


def check_gpu() -> bool:
    title("3. GPU（可选）")

    # 先看 nvidia-smi
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=20,
            )
            if out.returncode == 0 and out.stdout.strip():
                for line in out.stdout.strip().splitlines():
                    ok(f"GPU: {line.strip()}")
            else:
                warn("nvidia-smi 无输出")
        except Exception as e:  # noqa: BLE001
            warn(f"nvidia-smi 调用失败: {e}")
    else:
        warn("未找到 nvidia-smi（可能无独显，或用 CPU 跑）")

    # 再看 torch 是否能用 CUDA
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            ok(f"torch.cuda 可用，设备: {torch.cuda.get_device_name(0)}")
            ok(f"torch 版本: {torch.__version__}, CUDA: {torch.version.cuda}")
            return True
        warn("torch.cuda 不可用 → 将使用 CPU（慢但可行）")
        info(f"torch 版本: {torch.__version__}")
        return False
    except ImportError:
        warn("torch 未安装，无法判断 CUDA 可用性")
        return False


def check_hardware() -> None:
    title("4. 硬件资源")

    # 内存
    try:
        if platform.system() == "Windows":
            import ctypes  # noqa: PLC0415

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_gb = stat.ullTotalPhys / (1024 ** 3)
            avail_gb = stat.ullAvailPhys / (1024 ** 3)
            ok(f"内存 总 {total_gb:.1f} GB / 可用 {avail_gb:.1f} GB")
            if total_gb < 16:
                warn("内存 < 16GB，处理百万级语料可能吃力")
        else:
            info("非 Windows，跳过内存检查")
    except Exception as e:  # noqa: BLE001
        warn(f"内存检查失败: {e}")

    # 磁盘
    try:
        usage = shutil.disk_usage(str(ROOT))
        free_gb = usage.free / (1024 ** 3)
        ok(f"磁盘（项目所在盘）可用 {free_gb:.1f} GB")
        if free_gb < 50:
            warn("可用磁盘 < 50GB，全量语料 + 索引可能不够")
    except Exception as e:  # noqa: BLE001
        warn(f"磁盘检查失败: {e}")


def check_dirs() -> bool:
    title("5. 目录结构")
    missing = [d for d in REQUIRED_DIRS if not (ROOT / d).exists()]

    if not missing:
        ok(f"全部 {len(REQUIRED_DIRS)} 个目录就绪")
        return True

    for d in missing:
        miss(d)
    info("可用以下命令创建：")
    for d in missing:
        info(f"    mkdir \"{ROOT / d}\"")
    return False


def check_docs() -> bool:
    title("6. 文档")
    expected = [
        "01-项目章程.md", "02-需求规格说明书.md", "03-技术方案设计.md",
        "04-数据集规范.md", "05-评测协议.md", "06-实验设计.md",
        "07-风险与局限声明.md", "08-环境与技术栈.md", "09-实施计划.md",
    ]
    found = 0
    for name in expected:
        p = ROOT / "docs" / name
        if p.exists():
            found += 1
        else:
            miss(f"docs/{name}")

    if found == len(expected):
        ok(f"全部 {len(expected)} 份文档就绪")
        return True
    warn(f"缺少 {len(expected) - found} 份文档")
    return False


def check_network() -> None:
    title("7. 网络可达性（模型/数据下载）")
    targets = [
        ("HuggingFace 镜像", "https://hf-mirror.com"),
        ("ModelScope", "https://www.modelscope.cn"),
        ("PyPI", "https://pypi.org"),
    ]
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        warn("requests 未安装，跳过网络检查")
        return

    for name, url in targets:
        try:
            r = requests.head(url, timeout=8, allow_redirects=True)
            if r.status_code < 400:
                ok(f"{name:18s} 可达 (HTTP {r.status_code})")
            else:
                warn(f"{name:18s} 返回 HTTP {r.status_code}")
        except Exception:  # noqa: BLE001
            warn(f"{name:18s} 不可达（可能需要代理）")


def check_write_permission() -> bool:
    title("8. 写权限")
    probe = ROOT / "logs" / ".write_probe"
    try:
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("probe", encoding="utf-8")
        probe.unlink()
        ok("项目目录可写")
        return True
    except Exception as e:  # noqa: BLE001
        miss(f"项目目录不可写: {e}")
        return False


# ────────────────────────────────────────────────────────────
# 汇总与建议
# ────────────────────────────────────────────────────────────
def summarize(missing: list[str], gpu_ok: bool, dirs_ok: bool) -> int:
    title("检查结果汇总")

    stage1 = {"numpy", "pandas", "scipy", "yaml", "tqdm", "requests"}
    stage2 = {"jieba", "rank_bm25", "torch", "transformers",
              "sentence_transformers", "faiss", "datasets"}

    miss1 = sorted(stage1 & set(missing))
    miss2 = sorted(stage2 & set(missing))

    print(f"\n  缺失依赖总数 : {len(missing)}")
    if miss1:
        print(f"  阶段1 缺失   : {', '.join(miss1)}")
    if miss2:
        print(f"  阶段2 缺失   : {', '.join(miss2)}")

    print()
    hr("-")

    # ⚠️ 生成安装命令时，必须用【pip 包名】而不是 import 名
    #    （如 import 是 yaml，pip 装的是 PyYAML）
    pip1 = " ".join(IMPORT_TO_PIP.get(m, m) for m in miss1)
    pip2 = " ".join(IMPORT_TO_PIP.get(m, m) for m in miss2)

    if miss1:
        print("\n  ⚠️ 阶段1 依赖不全 —— 无法跑 BM25 基线\n")
        print("  ⭐ 下一步（复制执行）：")
        print(f"      pip install {pip1}")
        print("\n      说明：阶段1 是纯 CPU 依赖，装完即可跑通")
        print("            E1 基线的 BM25 部分。")
        print("            ⭐ 强烈建议先走这一步，验证评测代码正确后再装阶段2。")
        return 1

    if miss2:
        print("\n  🟡 阶段1 已就绪，可以跑 BM25 基线了\n")
        print("  ⭐ 下一步（二选一）：")
        print("     ① 先跑 BM25 基线（推荐）：")
        print("          python -m src.pipeline.build_index --config configs/e1_baseline.yaml")
        print("     ② 补装向量检索依赖：")
        print(f"          pip install {pip2}")
        print("\n      ⚠️ Windows 若 faiss 装不上，见 docs/08 第四节的备选方案。")
        return 0

    print("\n  ✅ 全部依赖就绪！\n")
    if not dirs_ok:
        print("  ⚠️ 但目录结构不完整，先补建目录。")
        return 1
    print("  ⭐ 下一步：")
    print("      python scripts/check_env.py --data   # 检查数据是否就位")
    print("      或直接开始 M1：下载数据集")
    if not gpu_ok:
        print("\n  ℹ️ 提示：CUDA 不可用，实验将用 CPU 跑。")
        print("     CPU 也能完成全部实验，只是慢一些（约 3-5 倍）。")
    return 0


def main() -> int:
    print()
    hr("#")
    print("#  RAG-Retrieval-Lab · 环境检查")
    hr("#")
    print(f"\n  项目根目录：{ROOT}")
    print(f"  操作系统  ：{platform.system()} {platform.release()}")

    check_python()
    all_ok, missing = check_deps()
    gpu_ok = check_gpu()
    check_hardware()
    dirs_ok = check_dirs()
    check_docs()
    check_network()
    check_network_ok = True  # noqa: F841  (仅作提示，不影响退出码)
    check_write_permission()

    code = summarize(missing, gpu_ok, dirs_ok)

    print()
    hr("#")
    return code


if __name__ == "__main__":
    sys.exit(main())
