# 心脏毒性预测数据集增强

通过扩展DIQTA训练集的化学空间邻域，结合多源证据检索，为每个分子构建包含数值证据、临床状态证据、文献上下文证据的证据包（Evidence Bundle），最终形成可支撑下游大模型生成"结构→靶点→效应"推理链的高质量数据集。

## 项目结构

```
cardiotox_dataset_augmentation/
├── README.md
├── requirements.txt
├── config/
│   ├── targets.yaml              # 靶点配置
│   ├── keywords.yaml             # 心脏毒性关键词
│   └── pipeline.yaml             # Pipeline配置
├── data/
│   ├── input/                    # 输入数据
│   ├── cache/                    # API响应缓存
│   ├── raw/                      # 各模块原始输出
│   └── output/                   # 最终输出
├── src/
│   ├── schemas.py                # Pydantic模型定义
│   ├── m1_similarity.py          # 相似性检索器
│   ├── m2_physchem.py            # 理化性质计算器
│   ├── m3a_bioactivity.py        # 靶点活性检索器
│   ├── m3b_clinical.py           # 临床状态检索器
│   ├── m3c_literature.py         # 文献检索器
│   ├── m4_assembler.py           # 证据包组装器
│   ├── m5_sufficiency.py         # 证据充分性判定器
│   ├── m6_conflict.py            # 冲突检测器
│   └── utils/
│       ├── chembl_client.py       # ChEMBL API
│       ├── pubmed_client.py       # PubMed API
│       ├── clinicaltrials_client.py  # ClinicalTrials API
│       └── cache.py               # 缓存层
├── scripts/
│   └── run_pipeline.py           # 主流程入口
└── tests/                        # 单元测试
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 运行完整流程

```bash
python scripts/run_pipeline.py --sample 5
```

参数说明：
- `--sample N`: 运行N个样本测试（0表示处理全部）
- `--diqta PATH`: DIQTA数据文件路径
- `--output PATH`: 输出文件路径
- `--config PATH`: 配置文件路径

### DIQTA 表中药名与 ChEMBL 对齐（宜暂时替换为 pref\_name）

通过名称解析 ChEMBL ID 时，应以 **ChEMBL `pref_name`** 与 API 对齐。下列 **6** 条在 DIQTA 表中的**原药物名字符串**与 ChEMBL 首选名不一致或含制剂/商标说明；**建议**在 Excel/导入列中**暂时将 `name`（或用于查 ChEMBL 的字段）替换为右侧 `pref_name`**，以便稳定命中同一母体/登记名。

| 表行 | 原药物名（片段） | 建议替换为 pref\_name | 解析示例 molecule\_chembl\_id |
|------|------------------|----------------------|------------------------------|
| 3 | `Thioridazine (Melleril)`（含 NBSP/商品名） | `THIORIDAZINE` | CHEMBL479 |
| 224 | `TEMSIROLIMUS INJECTION` | `TEMSIROLIMUS` | CHEMBL1201182 |
| 286 | `LEXISCAN(R) (REGADENOSON)` | `REGADENOSON ANHYDROUS` | CHEMBL317052 |
| 287 | `ADENOSINE INJECTION` | `ADENOSINE` | CHEMBL477 |
| 315 | `LORCASERIN HYDROCHLORIDE HEMIHYDRATE` | `LORCASERIN` | CHEMBL360328 |
| 326 | `ATROPINE INJECTION, 2 MG; ATROPINE SULFATE ...`（长制剂串） | `ATROPINE` | CHEMBL9751 |

> 说明：行号为 **Excel/表内数据行**（首行为表头时，与「行号」一致）。替换仅用于 **名称→ChEMBL ID** 解析；**SMILES** 仍以表内结构为准，不必改写。

### 配置

编辑 `config/` 目录下的配置文件：
- `targets.yaml`: 配置目标靶点
- `keywords.yaml`: 配置关键词
- `pipeline.yaml`: 配置Pipeline参数

## 模块说明

### M1: 相似性检索器
通过ChEMBL API检索与DIQTA分子相似的分子（Tanimoto ≥ 0.8）。

### M2: 理化性质计算器
计算分子量、logP、TPSA、pKa等理化性质。

### M3-A: 靶点活性检索器
检索hERG、Nav1.5、Cav1.2等靶点的活性数据。

### M3-B: 临床状态检索器
检索临床试验、撤市信息、黑框警告等。

### M3-C: 文献检索器
从PubMed检索心脏毒性相关文献。

### M4: 证据包组装器
将所有证据组装成统一的JSON格式。

### M5: 证据充分性判定器
判定证据是否足以赋予标签。

### M6: 冲突检测器
检测A类（体外）vs B类（临床）冲突。

## 输出格式

每个证据包包含：
```json
{
  "molecule_id": "CHEMBL123456",
  "smiles": "...",
  "source_path": "A_diqta_original | B_similarity_expanded",
  "label": {"value": "torsadogenic", "source": "diqta_ground_truth", "confidence": "high"},
  "physchem_properties": {...},
  "evidence_A_bioactivity": {...},
  "evidence_B_clinical": {...},
  "evidence_C_literature": {...},
  "metadata": {...}
}
```

## 依赖

- Python >= 3.10
- chembl_webresource_client
- rdkit
- pkasolver
- biopython
- requests
- pandas
- pydantic
- tqdm
- loguru

## License

MIT