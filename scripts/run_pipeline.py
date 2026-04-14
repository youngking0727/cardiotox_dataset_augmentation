"""主流程入口脚本"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

import yaml
from tqdm import tqdm
from loguru import logger

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.chembl_client import ChEMBLClient
from utils.pubmed_client import PubMedClient
from utils.clinicaltrials_client import ClinicalTrialsClient
from utils.cache import get_global_cache

from m1_similarity import SimilarityRetriever
from m2_physchem import PhysChemCalculator
from m3a_bioactivity import BioactivityRetriever
from m3b_clinical import ClinicalStatusRetriever
from m3c_literature import LiteratureRetriever
from m4_assembler import EvidenceBundleAssembler
from m5_sufficiency import EvidenceSufficiencyJudge
from m6_conflict import ConflictDetector


def setup_logging(log_file: Optional[str] = None):
    """设置日志"""
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level="DEBUG", format="{time} | {level} | {name}:{line} - {message}")


def load_config(config_file: str = "config/pipeline.yaml") -> Dict[str, Any]:
    """加载配置文件"""
    config_path = Path(__file__).parent.parent / config_file
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_diqta_data(diqta_file: str) -> List[Dict[str, Any]]:
    """加载DIQTA数据"""
    import pandas as pd

    diqta_path = Path(__file__).parent.parent / diqta_file
    if not diqta_path.exists():
        logger.warning(f"DIQTA数据文件不存在: {diqta_path}")
        return []

    try:
        # 尝试读取Excel文件
        df = pd.read_excel(diqta_path)
    except Exception as e:
        logger.warning(f"读取DIQTA文件失败: {e}")
        return []

    # 转换为字典列表
    records = []
    for _, row in df.iterrows():
        record = {
            "smiles": row.get("SMILES", ""),
            "chembl_id": row.get("ChEMBL_ID", ""),
            "label": row.get("Label", ""),
            "name": row.get("Name", "")
        }
        if record["smiles"]:
            records.append(record)

    return records


def process_path_a(molecule: Dict[str, Any],
                  assembler: EvidenceBundleAssembler,
                  physchem_calc: PhysChemCalculator,
                  bioactivity_retriever: BioactivityRetriever,
                  clinical_retriever: ClinicalStatusRetriever,
                  literature_retriever: LiteratureRetriever) -> Optional[Dict]:
    """
    处理路径A：DIQTA原始分子（标签已知）

    Args:
        molecule: 分子信息
        assembler: 证据包组装器
        physchem_calc: 理化性质计算器
        bioactivity_retriever: 靶点活性检索器
        clinical_retriever: 临床状态检索器
        literature_retriever: 文献检索器

    Returns:
        证据包字典
    """
    chembl_id = molecule.get("chembl_id")
    smiles = molecule.get("smiles")
    label = molecule.get("label", "")

    if not chembl_id or not smiles:
        logger.warning(f"跳过无效分子: {molecule}")
        return None

    logger.info(f"处理路径A分子: {chembl_id}")

    # 获取理化性质
    physchem_props = physchem_calc.get_full_properties(chembl_id=chembl_id, smiles=smiles)

    # 获取靶点活性
    bioactivity = bioactivity_retriever.retrieve(chembl_id)

    # 获取临床状态
    drug_name = molecule.get("name", molecule.get("pref_name", chembl_id))
    clinical = clinical_retriever.retrieve(chembl_id, drug_name=drug_name)

    # 获取文献
    literature = literature_retriever.retrieve(chembl_id, drug_name)

    # 组装证据包
    bundle = assembler.assemble(
        chembl_id=chembl_id,
        smiles=smiles,
        source_path="A_diqta_original",
        label_value=label if label in ["torsadogenic", "non-torsadogenic"] else "undetermined",
        label_source="diqta_ground_truth",
        label_confidence="high",
        physchem_properties=physchem_props,
        bioactivity_evidence=bioactivity,
        clinical_evidence=clinical,
        literature_evidence=literature
    )

    return json.loads(bundle.model_dump_json())


def process_path_b(similar_molecule: Dict[str, Any],
                  parent_chembl_id: str,
                  tanimoto: float,
                  assembler: EvidenceBundleAssembler,
                  physchem_calc: PhysChemCalculator,
                  bioactivity_retriever: BioactivityRetriever,
                  clinical_retriever: ClinicalStatusRetriever,
                  literature_retriever: LiteratureRetriever,
                  sufficiency_judge: EvidenceSufficiencyJudge,
                  conflict_detector: ConflictDetector) -> Optional[Dict]:
    """
    处理路径B：相似性扩展分子（标签未知）

    Args:
        similar_molecule: 相似分子信息
        parent_chembl_id: 父分子ChEMBL ID
        tanimoto: Tanimoto相似度
        assembler: 证据包组装器
        physchem_calc: 理化性质计算器
        bioactivity_retriever: 靶点活性检索器
        clinical_retriever: 临床状态检索器
        literature_retriever: 文献检索器
        sufficiency_judge: 证据充分性判定器
        conflict_detector: 冲突检测器

    Returns:
        证据包字典，如果不满足条件则返回None
    """
    chembl_id = similar_molecule.get("chembl_id")
    smiles = similar_molecule.get("smiles")

    if not chembl_id or not smiles:
        return None

    # 如果已在DIQTA中，跳过
    if similar_molecule.get("already_in_diqta"):
        logger.debug(f"跳过已存在于DIQTA的分子: {chembl_id}")
        return None

    logger.info(f"处理路径B分子: {chembl_id}")

    # 获取理化性质
    physchem_props = physchem_calc.get_full_properties(chembl_id=chembl_id, smiles=smiles)

    # 获取靶点活性
    bioactivity = bioactivity_retriever.retrieve(chembl_id)

    # 获取临床状态（需要先获取分子名称）
    # TODO: 获取分子名称
    clinical = clinical_retriever.retrieve(chembl_id)

    # 获取文献
    literature = literature_retriever.retrieve(chembl_id, chembl_id)

    # 计算证据密度
    evidence_density = sufficiency_judge.compute_evidence_density(
        bioactivity, clinical, literature
    )

    # 判定标签
    label_value, label_confidence = sufficiency_judge.judge_label(bioactivity, clinical)

    # 如果无法确定标签，丢弃
    if label_value is None:
        logger.info(f"证据不足，丢弃分子: {chembl_id}")
        return None

    # 检测冲突
    conflict_result = conflict_detector.detect(bioactivity, clinical, literature)
    conflicts = None
    if conflict_result.has_conflict:
        conflicts = {
            "conflict_A_vs_B": conflict_result.conflict_A_vs_B,
            "conflict_within_C": conflict_result.conflict_within_C
        }

    # 组装证据包
    bundle = assembler.assemble(
        chembl_id=chembl_id,
        smiles=smiles,
        source_path="B_similarity_expanded",
        label_value=label_value,
        label_source="evidence_derived",
        label_confidence=label_confidence,
        parent_diqta_molecules=[parent_chembl_id],
        tanimoto_to_parents=[tanimoto],
        physchem_properties=physchem_props,
        bioactivity_evidence=bioactivity,
        clinical_evidence=clinical,
        literature_evidence=literature,
        evidence_density=evidence_density,
        conflicts=conflicts
    )

    return json.loads(bundle.model_dump_json())


def run_pipeline(diqta_file: str = "data/input/diqta.csv",
                output_file: str = "data/output/evidence_bundles.jsonl",
                config_file: str = "config/pipeline.yaml",
                max_molecules: Optional[int] = None,
                sample_size: int = 5):
    """
    运行完整的数据处理流程

    Args:
        diqta_file: DIQTA数据文件路径
        output_file: 输出文件路径
        config_file: 配置文件路径
        max_molecules: 最大处理分子数（用于测试）
        sample_size: 样本数量（用于快速测试）
    """
    # 加载配置
    config = load_config(config_file)
    setup_logging(config.get("logging", {}).get("file"))

    logger.info("=" * 60)
    logger.info("开始心脏毒性数据集增强流程")
    logger.info("=" * 60)

    # 初始化组件
    cache_dir = Path(__file__).parent.parent / config["data"]["cache"]["directory"]
    get_global_cache(cache_dir)

    chembl_client = ChEMBLClient()
    pubmed_client = PubMedClient()
    clinicaltrials_client = ClinicalTrialsClient()

    similarity_retriever = SimilarityRetriever(chembl_client)
    physchem_calc = PhysChemCalculator(chembl_client)
    bioactivity_retriever = BioactivityRetriever(chembl_client)
    clinical_retriever = ClinicalStatusRetriever(chembl_client, clinicaltrials_client)
    literature_retriever = LiteratureRetriever(chembl_client, pubmed_client)
    assembler = EvidenceBundleAssembler(chembl_client)
    sufficiency_judge = EvidenceSufficiencyJudge(clinical_retriever, literature_retriever)
    conflict_detector = ConflictDetector()

    # 加载DIQTA数据
    diqta_molecules = load_diqta_data(diqta_file)
    logger.info(f"加载了 {len(diqta_molecules)} 个DIQTA分子")

    if not diqta_molecules:
        logger.error("没有找到DIQTA数据，请检查数据文件")
        return

    # 用于测试：只处理少量样本
    if sample_size > 0:
        diqta_molecules = diqta_molecules[:sample_size]
        logger.info(f"测试模式：只处理 {len(diqta_molecules)} 个样本")

    # DIQTA分子SMILES集合（用于去重）
    diqta_smiles = {m["smiles"] for m in diqta_molecules}

    # 输出路径
    output_path = Path(__file__).parent.parent / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 处理路径A：DIQTA原始分子
    logger.info("-" * 40)
    logger.info("处理路径A: DIQTA原始分子")
    logger.info("-" * 40)

    results = []
    discarded = []

    for i, molecule in enumerate(tqdm(diqta_molecules, desc="路径A处理")):
        try:
            bundle = process_path_a(
                molecule=molecule,
                assembler=assembler,
                physchem_calc=physchem_calc,
                bioactivity_retriever=bioactivity_retriever,
                clinical_retriever=clinical_retriever,
                literature_retriever=literature_retriever
            )

            if bundle:
                results.append(bundle)
            else:
                discarded.append({
                    "chembl_id": molecule.get("chembl_id"),
                    "reason": "failed_to_process"
                })

        except Exception as e:
            logger.error(f"处理分子失败: {molecule.get('chembl_id')}, 错误: {e}")
            discarded.append({
                "chembl_id": molecule.get("chembl_id"),
                "reason": str(e)
            })

    # 处理路径B：相似性扩展分子
    logger.info("-" * 40)
    logger.info("处理路径B: 相似性扩展分子")
    logger.info("-" * 40)

    for molecule in tqdm(diqta_molecules[:sample_size], desc="路径B处理"):
        try:
            chembl_id = molecule.get("chembl_id")
            smiles = molecule.get("smiles")

            if not chembl_id or not smiles:
                continue

            # 相似性检索
            similarity_result = similarity_retriever.retrieve(
                query_smiles=smiles,
                query_chembl_id=chembl_id,
                diqta_smiles=diqta_smiles
            )

            # 处理每个相似分子
            for similar in similarity_result.similar_molecules:
                try:
                    similar_data = {
                        "chembl_id": similar.chembl_id,
                        "smiles": similar.smiles,
                        "already_in_diqta": similar.already_in_diqta
                    }

                    bundle = process_path_b(
                        similar_molecule=similar_data,
                        parent_chembl_id=chembl_id,
                        tanimoto=similar.tanimoto,
                        assembler=assembler,
                        physchem_calc=physchem_calc,
                        bioactivity_retriever=bioactivity_retriever,
                        clinical_retriever=clinical_retriever,
                        literature_retriever=literature_retriever,
                        sufficiency_judge=sufficiency_judge,
                        conflict_detector=conflict_detector
                    )

                    if bundle:
                        results.append(bundle)
                    else:
                        discarded.append({
                            "chembl_id": similar.chembl_id,
                            "reason": "insufficient_evidence"
                        })

                except Exception as e:
                    logger.warning(f"处理相似分子失败: {similar.chembl_id}, 错误: {e}")

        except Exception as e:
            logger.error(f"相似性检索失败: {molecule.get('chembl_id')}, 错误: {e}")

    # 保存结果
    logger.info("-" * 40)
    logger.info("保存结果")
    logger.info("-" * 40)

    with open(output_path, 'w', encoding='utf-8') as f:
        for bundle in results:
            f.write(json.dumps(bundle, ensure_ascii=False) + '\n')

    logger.info(f"成功保存 {len(results)} 个证据包到 {output_path}")

    # 保存丢弃的分子
    if discarded:
        discarded_path = output_path.parent / "discarded.jsonl"
        with open(discarded_path, 'w', encoding='utf-8') as f:
            for item in discarded:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"保存了 {len(discarded)} 个丢弃分子到 {discarded_path}")

    logger.info("=" * 60)
    logger.info("流程完成!")
    logger.info(f"总计处理: {len(results)} 个成功, {len(discarded)} 个丢弃")
    logger.info("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="心脏毒性数据集增强流程")
    parser.add_argument("--diqta", type=str, default="data/input/diqta.csv",
                       help="DIQTA数据文件路径")
    parser.add_argument("--output", type=str, default="data/output/evidence_bundles.jsonl",
                       help="输出文件路径")
    parser.add_argument("--config", type=str, default="config/pipeline.yaml",
                       help="配置文件路径")
    parser.add_argument("--sample", type=int, default=5,
                       help="样本数量（0表示处理全部）")
    parser.add_argument("--max", type=int, default=None,
                       help="最大处理分子数")

    args = parser.parse_args()

    run_pipeline(
        diqta_file=args.diqta,
        output_file=args.output,
        config_file=args.config,
        max_molecules=args.max,
        sample_size=args.sample
    )


if __name__ == "__main__":
    main()