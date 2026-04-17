# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

心脏毒性预测数据集增强工具。通过扩展DIQTA训练集的化学空间邻域，结合多源证据检索（ChEMBL、PubMed、ClinicalTrials.gov），为每个分子构建包含数值证据、临床状态证据、文献上下文证据的证据包（Evidence Bundle）。

## 常用命令

```bash
# 运行Pipeline（5个样本测试）
python scripts/run_pipeline.py --sample 5

# 运行全部数据
python scripts/run_pipeline.py --sample 0

# 指定DIQTA数据文件
python scripts/run_pipeline.py --sample 5 --diqta data/input/diqta.csv

# 运行单元测试
pytest tests/ -v

# 运行单个测试文件
pytest tests/test_m1_similarity.py -v
```

## 项目架构

### 模块划分（M1-M6）

```
src/
├── m1_similarity.py       # M1: 相似性检索器 - 从ChEMBL检索Tanimoto≥0.8的相似分子
├── m2_physchem.py         # M2: 理化性质计算器 - ChEMBL优先，RDKit兜底
├── m3a_bioactivity.py     # M3-A: 靶点活性检索 - hERG/Nav1.5/Cav1.2
├── m3b_clinical.py        # M3-B: 临床状态检索 - max_phase/撤市信息/黑框警告
├── m3c_literature.py      # M3-C: 文献检索 - PubMed心脏毒性相关文献
├── m4_assembler.py        # M4: 证据包组装器 - 组装统一JSON格式
├── m5_sufficiency.py      # M5: 证据充分性判定 - V1临时规则判定标签
├── m6_conflict.py         # M6: 冲突检测器 - 检测A类vs B类冲突
├── schemas.py             # Pydantic数据模型定义
└── utils/
    ├── chembl_client.py       # ChEMBL API客户端
    ├── pubmed_client.py       # PubMed E-utilities API客户端
    ├── clinicaltrials_client.py # ClinicalTrials.gov API客户端
    └── cache.py               # 文件缓存层
```

### 数据处理双路径

- **路径A (diqta_original)**: DIQTA原始分子，标签已知（diqta_ground_truth）
- **路径B (similarity_expanded)**: 相似性扩展分子，标签由证据驱动（evidence_derived）

### 核心配置文件

- `config/targets.yaml`: 靶点配置（hERG CHEMBL240, Nav1.5 CHEMBL1971, Cav1.2 CHEMBL1940）
- `config/keywords.yaml`: 心脏毒性关键词
- `config/pipeline.yaml`: Pipeline参数配置

### 输出格式

证据包JSON包含：molecule_id, smiles, source_path, label, physchem_properties, evidence_A_bioactivity, evidence_B_clinical, evidence_C_literature, metadata

## 依赖环境

Python环境位于: `/AIRvePFS/dair/conda_envs/cad_new`

核心依赖: chembl-webresource-client, rdkit, pkasolver, biopython, requests, pandas, pydantic, tqdm, loguru

## 注意事项

- 所有API调用经过缓存层（data/cache/）
- 速率限制：ChEMBL/PubMed/ClinicalTrials API调用间隔0.5秒
- 相似性检索阈值默认0.8（Tanimoto，基于Morgan FP, radius=2, nBits=2048）
- 标签判定规则：withdrawn_reason含QT/TdP → torsadogenic；hERG IC50<10μM → torsadogenic；否则undetermined