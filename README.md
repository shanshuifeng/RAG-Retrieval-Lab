# RAG-Retrieval-Lab · 中文检索增强生成 实验台

> ⭐ **量化 RAG 每个环节的真实贡献** —— 不是"跑通 demo"，而是回答"哪一层值多少钱"。

---

## 这个项目是什么

一个**可复现的中文检索实验台**，系统对比：

| 实验 | 问题 | 主指标 |
|---|---|---|
| **E1** | 稠密向量检索比 BM25 好多少？ | Recall@10, nDCG@10 |
| **E2** | 混合检索（RRF）有提升吗？ | nDCG@10 |
| **E3** | ⭐ 加重排序值不值？ | nDCG@10 + 成本 |
| **E4** | 文档怎么切最好？ | Recall@10 |

**数据集**：[DuReader-retrieval](https://aclanthology.org/2022.emnlp-main.357/)（百度，97K queries / 890 万段落）

---

## 为什么值得做

| 依据 | 来源 |
|---|---|
| AI 应用开发工程师需求占 **53.9%**（最集中） | [前程无忧《2027届校招AI人才市场洞察》](https://m.jiemian.com/article/14983498.html) |
| 大厂 JD 要求 **RAG、Memory、Tool Use 的工程化落地** | [阿里 27 届算法岗 JD](https://campus-talent.alibaba.com/campus/position/199907740040) |
| 企业最看重**数学与统计基础（58.9%）** | 前程无忧同上 |

> 🎯 **与课程知识的衔接**：
> - **召回 + 排序两阶段** → Course 3 第 123 P
> - **双塔/双编码器** → Course 3 第 122 P
> - **评测方法论**（精确率/召回率/偏斜数据）→ Course 2 第 86–87 P、Course 3 第 111 P

---

## 快速开始

### 第 1 步：检查环境

```bash
python scripts/check_env.py
```

⚠️ 脚本会**直接给出你该执行的下一步命令**，照着做即可。

### 第 2 步：装最小依赖（纯 CPU，先跑通 BM25）

```bash
pip install requests tqdm PyYAML jieba rank-bm25
```

> ⭐ **为什么要分阶段装**：
> 先用 BM25 跑通**评测代码**并对照文献验证正确性，
> 确认无误后再装向量检索依赖。
> 这样即使后续 GPU 环境出问题，也已经有可用产出。

### 第 3 步：装向量检索依赖

```bash
pip install numpy pandas scipy matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu  # 或 cu121
pip install transformers sentence-transformers faiss-cpu datasets
```

⚠️ Windows 若 `faiss-cpu` 装不上，见 [`docs/08-环境与技术栈.md`](docs/08-环境与技术栈.md) 第四节的三个备选方案。

### 第 4 步：下载数据

```bash
python scripts/download_data.py
```

### 第 5 步：跑实验

```bash
python -m src.pipeline.run_experiment --config configs/e1_baseline.yaml
```

---

## 目录结构

```
RAG-Retrieval-Lab/
├── docs/           # ⭐ 9 份设计文档（先读 05 和 06）
├── configs/        # YAML 配置（改数字就能跑新实验）
├── data/           # 数据（不入 git）
├── src/
│   ├── data/       # 下载、清洗、分块
│   ├── retrieval/  # BM25 / 稠密 / 混合融合
│   ├── rerank/     # Cross-encoder 重排序
│   ├── eval/       # ⭐ 指标计算（有单元测试）
│   └── pipeline/   # 实验编排
├── scripts/        # check_env.py / download_data.py
├── tests/          # 单元测试
└── results/        # ⭐ 实验结果（入 git）
```

---

## ⭐ 必读文档（按顺序）

| 顺序 | 文档 | 为什么重要 |
|---|---|---|
| **1** | [`docs/05-评测协议.md`](docs/05-评测协议.md) | ⭐⭐ **最重要**——指标算错则全盘皆错 |
| **2** | [`docs/06-实验设计.md`](docs/06-实验设计.md) | ⭐ E1–E4 的完整方案 |
| **3** | [`docs/01-项目章程.md`](docs/01-项目章程.md) | 目标、范围边界、成功标准 |
| **4** | [`docs/04-数据集规范.md`](docs/04-数据集规范.md) | 数据格式与分阶段策略 |
| 5 | [`docs/03-技术方案设计.md`](docs/03-技术方案设计.md) | 架构与选型理由 |
| 6 | [`docs/08-环境与技术栈.md`](docs/08-环境与技术栈.md) | 环境坑与应对 |
| 7 | [`docs/07-风险与局限声明.md`](docs/07-风险与局限声明.md) | ⭐ 结论的有效边界 |
| 8 | [`docs/09-实施计划.md`](docs/09-实施计划.md) | 里程碑与降级策略 |
| 9 | [`docs/02-需求规格说明书.md`](docs/02-需求规格说明书.md) | 功能与非功能需求 |

---

## ⭐ 本项目的五条纪律

```
① 一次只改一个变量        否则无法归因
② 测试集只用一次          调参一律在 dev
③ 必须报告基线（BM25）     否则"提升"毫无意义
④ 必须报告成本             "提升2%但慢10倍"是重要结论
⑤ 提升必须做显著性检验     不把噪声当提升
```

---

## 硬件要求

| 项 | 最低 | 推荐 | 本机实测 |
|---|---|---|---|
| 内存 | 16 GB | 32 GB | 31.7 GB ✅ |
| 显存 | 不需要（CPU 可跑） | 8 GB | RTX 4060 8GB ✅ |
| 磁盘 | 50 GB | 100 GB | 695 GB ✅ |

> ⭐ **无独显也能完成全部实验**（CPU 慢 3–5 倍）。

---

## ⚠️ 局限声明（摘要）

| 局限 | 说明 |
|---|---|
| **规模** | 主实验用 10 万–100 万段落，生产级是千万+ ⚠️ |
| **领域** | 数据集源自通用网页搜索，**不适用于垂直领域** |
| **范围** | 评测的是**检索质量**，不是端到端 RAG 效果 |
| **方法** | 未覆盖多向量检索、领域微调、检索-生成联合优化 |

**完整声明**见 [`docs/07-风险与局限声明.md`](docs/07-风险与局限声明.md)。

---

## 引用

```bibtex
@inproceedings{qiu2022dureader,
  title={DuReader-Retrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine},
  author={Qiu, Yifu and others},
  booktitle={EMNLP},
  year={2022}
}
```

模型：`BAAI/bge-small-zh-v1.5`、`BAAI/bge-reranker-base`

---

## 项目状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| **M0** 环境搭建 + 文档 | ✅ 完成 | venv 验证 34/34 通过 |
| **M1** 数据准备 | ✅ 完成 | dev 8,640 / test 2,000 query；质检全绿；见 [DATACARD](data/DATACARD.md) |
| **M1.5** 指标实现验证 | ✅ 完成 | 单测 108/108；真实数据 oracle/random/anti 交叉验证通过 |
| **M2** E1 基线 | ✅ 完成 | 稠密 ≫ BM25（nDCG@10 +38.2%~+53.4%）；见 [docs/11](docs/11-E1实验报告草稿.md) |
| **M3** E2 混合检索 | ✅ 完成 | ⭐ **负面结果**：RRF 比纯 D2 差 -13%；见 [docs/13](docs/13-E2混合检索实验报告.md) |
| **M4** E3 重排序 | ✅ 完成 | K=20 甜点；+2.67%（dev）；见 [docs/14](docs/14-E3重排序实验报告.md) |
| **M4.5** test 最终评测 | ✅ 完成 | ⭐ 已锁定；三结论两复现一反转；见 [docs/15](docs/15-test最终评测报告.md) |
| **M5** E4 分块消融 | ⏳ 待做 | 可降级项 |
| **M6** 实验报告 | ⏳ 待做 | 汇总 E1–E3 + test |

> ⭐ **最终结论（test 集，已锁定）**：
> ① 稠密检索显著优于 BM25（nDCG@10 **+55.1%**，效应量 d=1.035 大）——复现且更强；
> ② 混合检索是负优化（-12.8%）——复现；
> ③ 重排在 dev 显著（+2.67%）但 test 上**不显著**（+0.16%，p=0.625）——**未复现**。

### ⚠️ 数据划分的一个重要说明

本项目 **dev 与 test 来自不同来源**，绝对指标不可跨集比较：

| 集合 | 来源 | query 数 | 每 query 平均相关数 |
|---|---|---|---|
| **dev** | `zyznull/dureader-retrieval-ranking` train 切片 | 8,640 | 2.60 |
| **test** | `C-MTEB/DuRetrieval` 官方测试集 | 2,000 | 4.92 |

> 原因：官方 `dev.jsonl.gz` 经逐条比对实为**测试集**（与 C-MTEB 2,000 query 文本交集 100%），
> 直接使用等于在测试集上调参。详见 [data/DATACARD.md](data/DATACARD.md) 第四节问题 6。

---

## 许可与用途

⚠️ **本项目为学习性实验，仅用于学术研究。**
数据集与模型的使用须遵守其原始许可条款。
