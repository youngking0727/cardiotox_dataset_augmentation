"""主流程入口脚本"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

import yaml
from tqdm import tqdm
from loguru import logger

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.chembl_client import ChEMBLClient
from utils.pubmed_client import PubMedClient, apply_ncbi_settings_from_config
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
        data = yaml.safe_load(f) or {}
    # pipeline.yaml 将选项放在顶层 pipeline: 下；扁平文件则直接使用
    return data.get("pipeline", data)


def load_diqta_data(diqta_file: str) -> List[Dict[str, Any]]:
    """加载 DIQTA 表（Excel/CSV），列名兼容常见导出。"""
    import pandas as pd

    def cell(row: Any, *keys: str) -> str:
        for k in keys:
            if k not in row.index:
                continue
            v = row[k]
            if pd.isna(v):
                continue
            s = str(v).strip()
            if s:
                return s
        return ""

    diqta_path = Path(__file__).parent.parent / diqta_file
    if not diqta_path.exists():
        logger.warning(f"DIQTA数据文件不存在: {diqta_path}")
        return []

    try:
        suf = diqta_path.suffix.lower()
        if suf == ".csv":
            df = pd.read_csv(diqta_path)
        elif suf == ".tsv":
            df = pd.read_csv(diqta_path, sep="\t")
        else:
            df = pd.read_excel(diqta_path)
    except Exception as e:
        logger.warning(f"读取DIQTA文件失败: {e}")
        return []

    records = []
    for _, row in df.iterrows():
        smiles = cell(row, "NEW SMILES", "SMILES", "smiles")
        chembl_id = cell(row, "ChEMBL_ID", "CHEMBL_ID", "chembl_id")
        name = cell(row, "name", "Name")
        label_raw = row.get("label", row.get("Label", ""))
        if pd.isna(label_raw):
            label = ""
        else:
            label = str(label_raw).strip()

        if not smiles:
            continue

        rec: Dict[str, Any] = {
            "smiles": smiles,
            "chembl_id": chembl_id,
            "label": label,
            "name": name,
        }
        if "Pubchem_ID" in row.index and not pd.isna(row["Pubchem_ID"]):
            rec["pubchem_id"] = row["Pubchem_ID"]

        records.append(rec)

    return records


def _label_matches_filter(m: Dict[str, Any], target: str) -> bool:
    """与 CLI --label 比较；兼容 Excel 中数值 0/1、0.0/1.0 及字符串。"""
    import pandas as pd

    raw = m.get("label", "")
    try:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return False
    except Exception:
        if raw is None:
            return False
    s_raw = str(raw).strip()
    if not s_raw:
        return False
    t = str(target).strip()
    if not t:
        return False
    try:
        return float(s_raw) == float(t)
    except (TypeError, ValueError):
        pass
    return s_raw.lower() == t.lower()


def filter_diqta_by_label(
    molecules: List[Dict[str, Any]], label_filter: Optional[str]
) -> List[Dict[str, Any]]:
    if not label_filter:
        return molecules
    return [m for m in molecules if _label_matches_filter(m, label_filter)]


def enrich_diqta_chembl_ids(
    molecules: List[Dict[str, Any]], chembl_client: ChEMBLClient
) -> List[Dict[str, Any]]:
    """若行内无 ChEMBL ID，则根据 pref_name / 药品名（Excel name 等）在 ChEMBL 中解析。"""
    out: List[Dict[str, Any]] = []
    for m in molecules:
        cid = (m.get("chembl_id") or "").strip()
        if cid.upper().startswith("CHEMBL"):
            out.append(m)
            continue
        pref = (m.get("pref_name") or m.get("name") or "").strip()
        smiles = (m.get("smiles") or "").strip()
        if not pref and not smiles:
            logger.warning(
                f"跳过无 ChEMBL ID 且无 pref_name/name/SMILES 的记录: {m}"
            )
            continue
        # TODO: 这一步就是希望通过商品名获取chembl_id，但现在逻辑有问题
        info, mol_ok = chembl_client.get_drug_by_name(pref, smiles=smiles or None)
        if not mol_ok:
            logger.warning(
                f"molecule.json 请求失败，跳过解析 ChEMBL ID（pref_name 前80字符）: {pref[:80]}"
            )
            continue
        if not info:
            logger.warning(f"ChEMBL 未匹配 pref_name（未命中）: {pref[:80]}...")
            continue
        mid = info.get("molecule_chembl_id")
        if not mid:
            logger.warning(f"ChEMBL 返回无 molecule_chembl_id: {pref[:80]}...")
            continue
        m2 = dict(m)
        m2["chembl_id"] = mid
        out.append(m2)
    return out


def enrich_drug_names_from_chembl(
    molecules: List[Dict[str, Any]], chembl_client: ChEMBLClient
) -> None:
    """
    为每条 DIQTA 记录写入 drug_name：优先 ChEMBL pref_name，否则 Excel name，否则 chembl_id。
    供路径 B 临床/文献检索与 DIQTA_Chembl.json 使用。
    """
    for m in molecules:
        cid = (m.get("chembl_id") or "").strip()
        excel_name = (m.get("name") or "").strip()
        if not cid:
            m["drug_name"] = excel_name
            continue
        mol, ok = chembl_client.get_molecule_by_chembl_id(cid)
        if ok and mol:
            pref = (mol.get("pref_name") or "").strip()
            m["drug_name"] = pref or excel_name or cid
            m["chembl_pref_name"] = pref or None
        else:
            m["drug_name"] = excel_name or cid
            m["chembl_pref_name"] = None


def save_diqta_chembl_json(
    path: Path,
    molecules: List[Dict[str, Any]],
    diqta_source: str,
) -> None:
    """路径 A 完成后写出父分子 ChEMBL 对齐信息与 drug_name 真值。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_diqta": diqta_source,
        "records": [
            {
                "chembl_id": m.get("chembl_id"),
                "smiles": m.get("smiles"),
                "label": m.get("label", ""),
                "excel_name": m.get("name", ""),
                "drug_name": m.get("drug_name", ""),
                "chembl_pref_name": m.get("chembl_pref_name"),
            }
            for m in molecules
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存 DIQTA↔ChEMBL 父分子表（含 drug_name）: {path}")


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

    # 获取靶点活性（retrieve 只返回 dict，勿对 dict 解包）
    bioactivity = bioactivity_retriever.retrieve(chembl_id)

    # 获取临床状态 / 文献（路径 A：不传 clinicaltrials_query_name / pubmed_query_name，
    # M3B/M3C 使用 DIQTA drug_name 与 ChEMBL pref_name 的一致性校验或 pref 回退）
    drug_name = molecule.get("name", molecule.get("pref_name", chembl_id))
    clinical = clinical_retriever.retrieve(chembl_id, drug_name=drug_name)
    literature = literature_retriever.retrieve(chembl_id, drug_name=drug_name)

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


def process_path_b(
    similar_molecule: Dict[str, Any],
    parent_chembl_id: str,
    tanimoto: float,
    assembler: EvidenceBundleAssembler,
    physchem_calc: PhysChemCalculator,
    bioactivity_retriever: BioactivityRetriever,
    clinical_retriever: ClinicalStatusRetriever,
    literature_retriever: LiteratureRetriever,
    sufficiency_judge: EvidenceSufficiencyJudge,
    conflict_detector: ConflictDetector,
    chembl_client: ChEMBLClient,
    parent_drug_name: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    处理路径B：相似性扩展分子（标签未知）

    Returns:
        (证据包 dict 或 None, screening_status)
        screening_status: kept | evidence_insufficient | retrieval_incomplete | skipped
    """
    chembl_id = similar_molecule.get("chembl_id")
    smiles = similar_molecule.get("smiles")

    if not chembl_id or not smiles:
        return None, "skipped"

    if similar_molecule.get("already_in_diqta"):
        logger.debug(f"跳过已存在于DIQTA的分子: {chembl_id}")
        return None, "skipped"

    logger.info(f"处理路径B分子: {chembl_id}")

    _mol, mol_ok = chembl_client.get_molecule_by_chembl_id(chembl_id)
    # 相似度检索已完成；理化补全（ChEMBL → RDKit）
    physchem_props = physchem_calc.get_full_properties(chembl_id=chembl_id, smiles=smiles)
    bioactivity = bioactivity_retriever.retrieve(chembl_id)
    bio_ok = True
    # 路径 B：相似分子不是「同一药品」，ClinicalTrials/PubMed 以父药（DIQTA）为锚；
    # M3A 仍按子 chembl_id 拉活性；M3B drug 记录仍按子 chembl_id。
    child_pref = ""
    if _mol and isinstance(_mol, dict):
        child_pref = (str(_mol.get("pref_name") or "")).strip()
    parent_dn = (parent_drug_name or "").strip()
    fallback_name = parent_dn or child_pref or chembl_id

    ct_list: List[str] = []
    for nm in (child_pref, parent_dn):
        s = (nm or "").strip()
        if s and s not in ct_list:
            ct_list.append(s)

    clinical = clinical_retriever.retrieve(
        chembl_id,
        drug_name=fallback_name,
        path_b_clinicaltrials_names=ct_list if ct_list else None,
        clinicaltrials_query_name=None if ct_list else (parent_dn or None),
    )
    literature = literature_retriever.retrieve(
        chembl_id,
        fallback_name,
        max_pubmed=40,
        path_b=True,
        child_pref_name=child_pref,
        parent_drug_name=parent_dn,
        pubmed_query_name=parent_dn or None,
    )

    retrieval_incomplete = (
        not mol_ok
        or not bio_ok
        or (clinical.drug_info_status == "request_failed")
    )

    if retrieval_incomplete:
        logger.warning(f"取数不完整，暂不判定证据不足: {chembl_id}")
        return None, "retrieval_incomplete"

    retention = sufficiency_judge.judge_retention_for_path_b(
        bioactivity, clinical, literature
    )
    if not retention.should_keep:
        logger.info(f"Path B 证据不足（无 priority_1/2/3），丢弃: {chembl_id}")
        return None, "evidence_insufficient"

    evidence_density = sufficiency_judge.compute_evidence_density(
        bioactivity, clinical, literature
    )

    label_value, label_confidence = sufficiency_judge.judge_label(bioactivity, clinical)
    label_source = "evidence_derived"
    if label_value is None:
        label_value = "undetermined"
        label_confidence = None
        label_source = "evidence_retained_but_not_final_labeled"

    conflict_result = conflict_detector.detect(bioactivity, clinical, literature)
    conflicts = None
    if conflict_result.has_conflict:
        conflicts = {
            "conflict_A_vs_B": conflict_result.conflict_A_vs_B,
            "conflict_within_C": conflict_result.conflict_within_C
        }

    bundle = assembler.assemble(
        chembl_id=chembl_id,
        smiles=smiles,
        source_path="B_similarity_expanded",
        label_value=label_value,
        label_source=label_source,
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

    bundle_dict = json.loads(bundle.model_dump_json())
    meta = bundle_dict.setdefault("metadata", {})
    meta["screening_status"] = "kept"
    meta["retained_by_path_b_rule"] = True
    meta["evidence_priority"] = retention.evidence_priority
    meta["retention_reason_codes"] = retention.reason_codes
    meta["retention_notes"] = retention.notes
    if parent_drug_name:
        meta["parent_drug_name"] = parent_drug_name
    meta["top_literature_bucket"] = literature.top_relevance_bucket
    meta["priority_1_article_count"] = literature.priority_1_article_count
    meta["priority_2_article_count"] = literature.priority_2_article_count
    meta["priority_3_article_count"] = literature.priority_3_article_count

    has_herg = bool(
        bioactivity.get("hERG") and bioactivity["hERG"].measurements
    )
    logger.info(
        "PATH_B_DECISION | chembl_id={} | parent={} | keep=True | priority={} | label={} | "
        "reasons={} | has_hERG_measurement={} | p1={} p2={} p3={} | direct_qt_clinical_hit={} | status=kept",
        chembl_id,
        parent_chembl_id,
        retention.evidence_priority,
        label_value,
        retention.reason_codes,
        has_herg,
        literature.priority_1_article_count,
        literature.priority_2_article_count,
        literature.priority_3_article_count,
        getattr(clinical, "direct_qt_clinical_hit", False),
    )
    return bundle_dict, "kept"


def run_pipeline(diqta_file: str = "data/DIQTA阴性样本为主划分.xlsx",
                output_file: str = "data/output/evidence_bundles.jsonl",
                config_file: str = "config/pipeline.yaml",
                max_molecules: Optional[int] = None,
                sample_size: int = 5,
                diqta_chembl_json: str = "data/output/DIQTA_Chembl.json",
                path_a_only: bool = False,
                path_b_only: bool = False,
                label_filter: Optional[str] = None):
    """
    运行完整的数据处理流程

    Args:
        diqta_file: DIQTA数据文件路径
        output_file: 输出文件路径
        config_file: 配置文件路径
        max_molecules: 最大处理分子数（用于测试）
        sample_size: 样本数量（用于快速测试）
        diqta_chembl_json: 路径 A 结束后写入的父分子表（含 drug_name），默认与 evidence 同目录
        path_a_only: 仅跑路径 A（DIQTA 原始分子证据包）
        path_b_only: 仅跑路径 B（相似扩展）；仍加载 DIQTA、解析 ChEMBL ID，并做父分子 drug_name 补全后跑 M1→相似分子流水线
        label_filter: 若设置（如 '0'、'1'），仅保留 label 列匹配的分子；在 sample_size 截断之前应用
    """
    # 加载配置
    config = load_config(config_file)
    apply_ncbi_settings_from_config(config)
    setup_logging(config.get("logging", {}).get("file"))

    if path_a_only and path_b_only:
        logger.error("不能同时指定 path_a_only 与 path_b_only")
        return

    logger.info("=" * 60)
    logger.info("开始心脏毒性数据集增强流程")
    if path_a_only:
        logger.info("模式: 仅路径 A（DIQTA 原始）")
    elif path_b_only:
        logger.info("模式: 仅路径 B（相似扩展，跳过路径 A 组装）")
    logger.info("=" * 60)

    # 初始化组件
    cache_dir = Path(__file__).parent.parent / config["data"]["cache"]["directory"]
    get_global_cache(cache_dir)

    chembl_client = ChEMBLClient()
    pub_cfg = (config.get("api") or {}).get("pubmed") or {}
    pubmed_client = PubMedClient(
        rate_limit=float(pub_cfg.get("rate_limit", 0.5)),
        ncbi_email=pub_cfg.get("ncbi_email"),
        ncbi_api_key=pub_cfg.get("ncbi_api_key"),
        ncbi_tool=pub_cfg.get("ncbi_tool"),
    )
    clinicaltrials_client = ClinicalTrialsClient()

    sim_cfg = (config.get("similarity") or {})
    similarity_retriever = SimilarityRetriever(
        chembl_client,
        threshold=float(sim_cfg.get("threshold", 0.8)),
        max_results=int(sim_cfg.get("max_results", 100)),
    )
    logger.info(
        f"路径B 相似性扩展: threshold={similarity_retriever.threshold}, "
        f"max_results={similarity_retriever.max_results}（与 chembl_excel_full_enrichment 一致："
        "多组分 SMILES 取主片段再查 /similarity；结果去重并排除父分子）"
    )
    physchem_calc = PhysChemCalculator(chembl_client)
    bioactivity_retriever = BioactivityRetriever(chembl_client)
    clinical_retriever = ClinicalStatusRetriever(chembl_client, clinicaltrials_client)
    literature_retriever = LiteratureRetriever(chembl_client, pubmed_client)
    assembler = EvidenceBundleAssembler(chembl_client)
    sufficiency_judge = EvidenceSufficiencyJudge(literature_retriever)
    conflict_detector = ConflictDetector()

    # 加载DIQTA数据
    diqta_molecules = load_diqta_data(diqta_file)
    logger.info(f"加载了 {len(diqta_molecules)} 个DIQTA分子")

    if not diqta_molecules:
        logger.error("没有找到DIQTA数据，请检查数据文件")
        return

    if label_filter is not None and str(label_filter).strip() != "":
        before = len(diqta_molecules)
        diqta_molecules = filter_diqta_by_label(diqta_molecules, str(label_filter).strip())
        logger.info(f"按 label={label_filter!r} 筛选: {before} -> {len(diqta_molecules)} 条")
        if not diqta_molecules:
            logger.error("label 筛选后无数据，请检查 --label 与表中 label 列是否一致")
            return

    # 用于测试：只处理少量样本（在 label 筛选之后）
    if sample_size > 0:
        diqta_molecules = diqta_molecules[:sample_size]
        logger.info(f"测试模式：只处理 {len(diqta_molecules)} 个样本")

    # TODO: 这一步只是把chembl_id查询出来了
    diqta_molecules = enrich_diqta_chembl_ids(diqta_molecules, chembl_client)
    logger.info(f"解析 ChEMBL ID 后剩余 {len(diqta_molecules)} 个分子")
    if not diqta_molecules:
        logger.error(
            "没有可处理的分子（缺少 ChEMBL ID 且 pref_name/name 无法在 ChEMBL 中匹配）"
        )
        return

    # DIQTA分子SMILES集合（用于去重）
    diqta_smiles = {m["smiles"] for m in diqta_molecules}

    # 输出路径
    output_path = Path(__file__).parent.parent / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    discarded = []
    retry_candidates: List[Dict[str, Any]] = []

    if not path_b_only:
        logger.info("-" * 40)
        logger.info("处理路径A: DIQTA原始分子")
        logger.info("-" * 40)

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
    else:
        logger.info("path_b_only=True：跳过路径 A（不生成 DIQTA 原始证据包）")

    # 父分子 drug_name（ChEMBL pref_name 优先）并落盘；路径 B 依赖 molecule['drug_name']
    enrich_drug_names_from_chembl(diqta_molecules, chembl_client)
    diqta_chembl_path = Path(__file__).parent.parent / diqta_chembl_json
    save_diqta_chembl_json(diqta_chembl_path, diqta_molecules, diqta_file)

    if path_a_only:
        logger.info("path_a_only=True：跳过路径 B，直接保存结果")
        # 仅路径 A 时 results 只有 A 的包；进入统一写盘逻辑
    else:
        # 处理路径B：相似性扩展分子（相似度 → 理化补全 在 process_path_b 内顺序执行）
        logger.info("-" * 40)
        logger.info("处理路径B: 相似性扩展分子")
        logger.info("-" * 40)

        for molecule in tqdm(diqta_molecules, desc="路径B处理"):
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

                if not similarity_result.query_similarity_ok:
                    logger.warning(
                        f"取数不完整，暂不判定证据不足: {chembl_id} (similarity.json)"
                    )
                    retry_candidates.append({
                        "chembl_id": chembl_id,
                        "context": "path_b_parent_similarity",
                        "screening_status": "retrieval_incomplete",
                        "reason": "similarity.json request failed",
                    })
                    continue

                # 处理每个相似分子
                for similar in similarity_result.similar_molecules:
                    try:
                        similar_data = {
                            "chembl_id": similar.chembl_id,
                            "smiles": similar.smiles,
                            "already_in_diqta": similar.already_in_diqta
                        }

                        bundle, pb_status = process_path_b(
                            similar_molecule=similar_data,
                            parent_chembl_id=chembl_id,
                            tanimoto=similar.tanimoto,
                            assembler=assembler,
                            physchem_calc=physchem_calc,
                            bioactivity_retriever=bioactivity_retriever,
                            clinical_retriever=clinical_retriever,
                            literature_retriever=literature_retriever,
                            sufficiency_judge=sufficiency_judge,
                            conflict_detector=conflict_detector,
                            chembl_client=chembl_client,
                            parent_drug_name=molecule.get("drug_name"),
                        )

                        if pb_status == "kept" and bundle:
                            results.append(bundle)
                        elif pb_status == "retrieval_incomplete":
                            retry_candidates.append({
                                "chembl_id": similar.chembl_id,
                                "parent_chembl_id": chembl_id,
                                "screening_status": "retrieval_incomplete",
                                "reason": "chmbl_retrieval_incomplete",
                            })
                        elif pb_status == "evidence_insufficient":
                            discarded.append({
                                "chembl_id": similar.chembl_id,
                                "parent_chembl_id": chembl_id,
                                "reason": "evidence_insufficient",
                                "screening_status": "evidence_insufficient",
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

    # 保存丢弃的分子（真证据不足等）
    if discarded:
        discarded_path = output_path.parent / "discarded.jsonl"
        with open(discarded_path, 'w', encoding='utf-8') as f:
            for item in discarded:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"保存了 {len(discarded)} 条丢弃记录到 {discarded_path}")

    if retry_candidates:
        retry_path = output_path.parent / "retry_candidates.jsonl"
        with open(retry_path, 'w', encoding='utf-8') as f:
            for item in retry_candidates:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(
            f"保存了 {len(retry_candidates)} 条取数不完整待重试记录到 {retry_path}"
        )

    logger.info("=" * 60)
    logger.info("流程完成!")
    logger.info(
        f"总计: {len(results)} 个证据包, {len(discarded)} 条丢弃, "
        f"{len(retry_candidates)} 条取数不完整（retry_candidates）"
    )
    logger.info("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="心脏毒性数据集增强流程")
    parser.add_argument("--diqta", type=str, default="data/DIQTA阴性样本为主划分.xlsx",
                       help="DIQTA数据文件路径")
    parser.add_argument("--output", type=str, default="data/output/evidence_bundles.jsonl",
                       help="输出文件路径")
    parser.add_argument("--config", type=str, default="config/pipeline.yaml",
                       help="配置文件路径")
    parser.add_argument("--sample", type=int, default=5,
                       help="样本数量（0表示处理全部）")
    parser.add_argument("--max", type=int, default=None,
                       help="最大处理分子数")
    parser.add_argument(
        "--diqta-chembl-json",
        type=str,
        default="data/output/DIQTA_Chembl.json",
        help="路径 A 完成后保存的 DIQTA↔ChEMBL 父分子表（含 drug_name）",
    )
    parser.add_argument(
        "--path-a-only",
        action="store_true",
        help="仅处理路径 A（DIQTA 原始分子证据包）",
    )
    parser.add_argument(
        "--path-b-only",
        action="store_true",
        help="仅处理路径 B（相似扩展）；不组装路径 A 证据包，仍加载 DIQTA 并补全父分子 drug_name",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        metavar="VALUE",
        help="仅保留 label 列等于该值的行（如 0、1）；在 --sample 截断之前筛选，便于只跑某类标签的前 N 条",
    )

    args = parser.parse_args()

    run_pipeline(
        diqta_file=args.diqta,
        output_file=args.output,
        config_file=args.config,
        max_molecules=args.max,
        sample_size=args.sample,
        diqta_chembl_json=args.diqta_chembl_json,
        path_a_only=args.path_a_only,
        path_b_only=args.path_b_only,
        label_filter=args.label,
    )


if __name__ == "__main__":
    main()