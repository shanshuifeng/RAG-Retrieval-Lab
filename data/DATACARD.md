# 数据卡片 · DATACARD

> **状态**：✅ **已填写（2026-10-19）** —— dev / test 两套数据均已就绪并通过质检
> **数据版本**：data-split-v1（[`data/processed/dev/`](processed/dev)、[`data/processed/test/`](processed/test)）
> **质检报告**：[`data/processed/quality_report.json`](processed/quality_report.json)

---

## 一、基本信息

| 项 | 内容 |
|---|---|
| **数据集名称** | DuReader-retrieval（C-MTEB 版 / 官方 ranking 版） |
| **来源** | 百度 |
| **论文** | [Qiu et al., EMNLP 2022](https://aclanthology.org/2022.emnlp-main.357/) |
| **GitHub** | https://github.com/baidu/DuReader |
| **test 获取渠道** | HuggingFace 镜像 `C-MTEB/DuRetrieval` + `C-MTEB/DuRetrieval-qrels` |
| **dev 获取渠道** | HuggingFace 镜像 `zyznull/dureader-retrieval-ranking` 的 `train.jsonl.gz` |
| **下载日期** | 2026-10-17（test）、2026-10-18（dev 原料） |
| **下载镜像** | `https://hf-mirror.com` |

### ⭐ 数据版本与哈希（复现必需）

| 文件 | 大小 | sha256 |
|---|---|---|
| `train.jsonl.gz`（dev 原料） | 3,612,498,305 B | `785da278a2b4323262c2a51dc400d02429efe75a27a0568af1439c8de9dd2115` |
| `corpus-*.parquet`（test 语料） | 64,412,709 B | `d4b4eb51b63549ef0851a15fc63c2a61b703dce95e3727b535a08f7ba1d14424` |
| `queries-*.parquet`（test 查询） | 118,461 B | `62ac55e764bffd4ffceb0aa51e7a536a0e5932f23c8606566906db6a9efb4b94` |
| `dev-*.parquet`（test qrels） | 420,443 B | 见 [`data/raw/dureval/manifest.json`](raw/dureval/manifest.json) |

---

## 二、⭐ 规模统计（实测，2026-10-19）

### 2.1 语料库

| 项 | dev | test | 说明 |
|---|---|---|---|
| **段落总数** | 100,000 | 100,001 | ⭐ 刻意同规模，便于横向比较 |
| 唯一 id 数 | 100,000 | 100,001 | 无重复 id |
| 文本完全重复 | 0 段 | 0 段 | 已核验 |
| **平均长度（字）** | 357.31 | 331.32 | |
| **中位长度（字）** | 283.0 | 271 | |
| **P95 长度（字）** | 797 | 736 | |
| 最长段落（字） | 87,586 | 60,975 | ⚠️ 见第四节问题 1 |
| < 20 字段落 | 413 | 417 | ⚠️ 见第四节问题 1 |
| \> 1000 字段落 | 3,355 | 3,018 | ⚠️ 见第四节问题 1 |
| \> 5000 字段落 | 70 | 124 | ⚠️ 见第四节问题 1 |
| 含 HTML 标签 | 196 | 182 | 仅记录，基线阶段不删（见第四节） |
| 含连续空白/换行 | 5,159 | 3,160 | 同上 |

### 2.2 查询集

| 项 | dev | test | 说明 |
|---|---|---|---|
| **query 数** | **8,640** | **2,000** | dev = train 的 10.0% |
| 唯一 qid | 8,640 | 2,000 | |
| 空 query | 0 | 0 | |
| 重复 query 文本 | 0 | 0 | |
| 平均长度（字） | 9.52 | 9.29 | |
| 最长 query（字） | 39 | 55 | |

### 2.3 ⭐ 相关标注分布（关键）

| 项 | dev | test | 说明 |
|---|---|---|---|
| **平均相关段落数/query** | **2.60** | **4.92** | ⚠️ **两者不同，绝对值不可直接比较** |
| 中位数 | 2.0 | 5.0 | |
| 最大值 | 65 | 31 | |
| P95 | 8 | 11 | |
| **相关段落数为 0 的 query** | **0** | **0** | ✅ 无需过滤 |
| **有效 query 数** | **8,640** | **2,000** | = 总数（无过滤） |
| qrels 行数 | 22,470 | 9,839 | |
| 悬空引用（qrels 指向不存在的段落） | **0** | **0** | ✅ 已核验 |

> ⚠️ **必须声明的非对称性**：
> | 维度 | dev | test | 影响 |
> |---|---|---|---|
> | 每 query 正例数 | 2.60 | 4.92 | 同样召回 2 个时，dev 的 Recall 更高 |
> | query 数 | 8,640 | 2,000 | dev 统计效力更强 |
> | 语料来源 | train 正例 + 负例采样 | 官方语料库 | dev 正例保证在库内 |
>
> ⭐ **结论：dev 只用于调参与方法横向比较；dev 与 test 的【绝对指标值】不可直接对比。**
> 报告中须同时给出两套数字并注明此差异。

---

## 三、数据格式

### 3.1 语料库（`corpus.jsonl`）

```jsonl
{"id": "p_xxx", "text": "..."}
```

### 3.2 查询（`queries.jsonl`）

```jsonl
{"qid": "q_xxx", "query": "...", "relevant_ids": ["p_a", "p_b"]}
```

### 3.3 qrels（`qrels.tsv`）

```
# qid<TAB>pid<TAB>score
edb58f525bd14724d6f490722fa8a657	2c4fe63d3378ac39907b6b2648eb40c5	1
```

> 统一格式保证 **dev 与 test 的读取代码完全相同**，避免「换数据集就改代码」导致的口径漂移。

---

## 四、⚠️ 已知问题与处理决定

| # | 问题 | 实测数据 | 处理方式 |
|---|---|---|---|
| **1** | 段落长度差异极大（1 字 ~ 87,586 字） | dev 最长 87,586 字；<20 字 413 段；>1000 字 3,355 段 | ⭐ **基线阶段（E1）不删除、不截断** —— [docs/04](../docs/04-数据集规范.md) §4.2 规定用「策略 A：原生 passage」以与论文可比。**E4 分块实验才系统对比截断/切分的影响**，届时用同一份数据做对照。 |
| **2** | 少量段落含 HTML 标签 | dev 196 段 / test 182 段（约 0.2%） | ⭐ **不删除**（同上，与论文口径一致）。**记录在案**，若指标异常可做「清洗 vs 不清洗」的敏感性分析。 |
| **3** | test 是 C-MTEB 采样版，非 890 万全量 | 100,001 段 | ⭐ **不与论文全量指标直接比**，只比趋势与量级。报告中声明。 |
| **4** | dev 由 train 切出，正例保证在库内 | — | ⭐ 这会使 dev 的召回绝对值偏高。**故 dev 只用于调参与方法横比**，绝对值以 test 为准。 |
| **5** | dev 与 test 每 query 正例数不同（2.60 vs 4.92） | — | ⭐ 报告中**必须同时披露**，禁止跨集比较绝对指标。 |
| **6** | 官方 `dev.jsonl.gz` 实为测试集 | 与 C-MTEB 2,000 query 文本交集 **100%** | ⭐ **已确认并规避**：未使用该文件作 dev，见 [`data/raw/dureader-ranking/compare_cmteb_dev.json`](raw/dureader-ranking/compare_cmteb_dev.json) |

---

## 五、⭐ 数据划分（已锁定，勿改）

| 集合 | 来源 | 数量 | 用途 |
|---|---|---|---|
| **train** | 官方 train.jsonl.gz | 86,395 query | ⭐ 仅作为 dev 候选池，本项目**不训练模型** |
| **dev** | 从 train 按「每 10 条取 1 条」切分 | 8,640 query / 100,000 段 | ⭐ **调超参数** |
| **test** | C-MTEB/DuRetrieval 官方 | 2,000 query / 100,001 段 | ⭐ **只跑一次，用于报告** |

### 切分规则（可复现）

```
dev query 选取：train.jsonl 按文件顺序，索引 % 10 == 0 的记录（共 8,640 条）
dev qrels     ：该 8,640 条 query 的全部 positive_passages（22,470 行）
dev 语料      ：22,280 条正例（全收）
              + 从 3,365,607 条负例池中随机抽 77,720 条（seed=42）
              = 100,000 段
```

```
⚠️ 铁律：test 集一旦锁定，不得用于任何调参
         所有调参在 dev 上完成
```

### ⭐ dev 与 test 的 query 是否重叠？

| 检查 | 结果 |
|---|---|
| dev 8,640 query 与 test 2,000 query 文本交集 | **0**（来源不同：train vs 官方测试集） |

---

## 六、⭐ 数据质检结果（2026-10-19）

| 检查项 | dev | test |
|---|---|---|
| 语料 id 唯一 | ✅ | ✅ |
| 文本无重复 | ✅ | ✅ |
| qrels 全部指向存在的段落 | ✅ 0 悬空 | ✅ 0 悬空 |
| 零相关 query | ✅ 0 | ✅ 0 |
| query 非空 | ✅ 0 空 | ✅ 0 空 |
| query 文本无重复 | ✅ | ✅ |
| 格式校验通过 | ✅ JSONL/TSV | ✅ JSONL/TSV |

完整报告：[`data/processed/quality_report.json`](processed/quality_report.json)

---

## 七、⭐ 指标实现验证（M2.3 硬门槛）

指标代码：[`src/eval/metrics.py`](../src/eval/metrics.py) ｜ 单测：[`tests/test_metrics.py`](../tests/test_metrics.py)

| 验证 | 结果 |
|---|---|
| 单元测试 | ✅ **29/29 通过**（含 [docs/05](../docs/05-评测协议.md) 全部人工算例） |
| 真实数据 oracle 排序 nDCG@100 | ✅ **1.000000** |
| 真实数据 oracle Recall@100 | ✅ **1.000000** |
| 随机排序 nDCG@100 | ✅ 0.000089（符合 100/100,000 的随机命中率量级） |
| 反排序（正例压到最后）Recall@100 | ✅ 1.0000，但 nDCG@100 仅 0.1869 |
| ⭐ 该对照证明 | **指标能区分"召回"与"排序质量"** —— 正是 E3 重排序实验的理论基础 |
| DCG ≤ IDCG 断言 | ✅ 13,500 个指标值无一违反 |

完整报告：[`data/processed/metrics_verify_dev.json`](processed/metrics_verify_dev.json)

---

## 八、许可与引用

| 项 | 内容 |
|---|---|
| **许可** | Apache License 2.0（ranking 版）/ 学术研究使用 |
| **商用** | ❌ 不允许 |
| **引用要求** | ⭐ 必须引用原论文 |

```bibtex
@inproceedings{qiu2022dureader,
  title={DuReader-Retrieval: A Large-scale Chinese Benchmark for Passage Retrieval from Web Search Engine},
  author={Qiu, Yifu and others},
  booktitle={EMNLP},
  year={2022}
}
```

---

## 九、复现步骤

```powershell
# 0. 环境（Python 3.12 + 已装依赖）
.venv\Scripts\python.exe scripts\verify_venv.py

# 1. 下载 test（C-MTEB）
.venv\Scripts\python.exe scripts\download_data.py --dataset dureval

# 2. 下载 dev 原料（train.jsonl.gz，3.36GB；支持断点续传）
.venv\Scripts\python.exe scripts\download_parallel.py

# 3. test 转统一格式
.venv\Scripts\python.exe scripts\convert_cmteb.py

# 4. 构建 dev（≈2 分钟）
.venv\Scripts\python.exe scripts\build_dev.py

# 5. 质量核查（生成 quality_report.json）
.venv\Scripts\python.exe scripts\data_quality.py

# 6. 指标验证（M2.3 硬门槛）
.venv\Scripts\python.exe -m unittest tests.test_metrics -v
.venv\Scripts\python.exe scripts\verify_metrics_real.py --split dev --n 500
```

---

## 十、检查清单（M1 完成标准）

```
☑ 语料库下载完整
☑ 查询集下载完整（含标注）
☑ 格式转换完成并通过校验
☑ 本文档第二、五节的数字已全部填上实测值
☑ 检查过 relevant_ids 全部指向存在的段落（0 悬空）
☑ 记录了过滤掉的 query 数量（0 条）
☑ dev 集已切分并保存为文件（8,640 query / 100,000 段）
☑ ⭐ test 集已锁定，未用于任何调参
☑ 数据版本/哈希已记录
```

**⭐ M1 已完成。**

---

**⭐ 一句话**：**dev/test 双集就绪、质检全绿、指标验证通过——可以进入 M2（E1 基线实验）。**