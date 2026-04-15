"""M4: 证据包组装器模块"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from schemas import (
    EvidenceBundle, LabelInfo, PhysChemProperties,
    TargetBioactivity, ClinicalEvidence, LiteratureEvidence,
    EvidenceDensity
)
from utils.chembl_client import ChEMBLClient

logger = logging.getLogger(__name__)


# 当前版本信息
PIPELINE_VERSION = "1.0"
CHEMBL_VERSION = "34"  # 需要根据实际情况更新


class EvidenceBundleAssembler:
    """证据包组装器"""

    def __init__(self, chembl_client: Optional[ChEMBLClient] = None):
        """
        初始化证据包组装器

        Args:
            chembl_client: ChEMBL客户端
        """
        self.chembl_client = chembl_client or ChEMBLClient()

    def assemble(self,
                chembl_id: str,
                smiles: str,
                source_path: str,
                label_value: Optional[str] = None,
                label_source: str = "undetermined",
                label_confidence: Optional[str] = None,
                parent_diqta_molecules: Optional[List[str]] = None,
                tanimoto_to_parents: Optional[List[float]] = None,
                physchem_properties: Optional[PhysChemProperties] = None,
                bioactivity_evidence: Optional[Dict[str, TargetBioactivity]] = None,
                clinical_evidence: Optional[ClinicalEvidence] = None,
                literature_evidence: Optional[LiteratureEvidence] = None,
                evidence_density: Optional[EvidenceDensity] = None,
                conflicts: Optional[Dict[str, str]] = None) -> EvidenceBundle:
        """
        组装证据包

        Args:
            chembl_id: ChEMBL ID
            smiles: SMILES
            source_path: 来源路径（A_diqta_original/B_similarity_expanded）
            label_value: 标签值（torsadogenic/non-torsadogenic/undetermined）
            label_source: 标签来源（diqta_ground_truth/evidence_derived/undetermined）
            label_confidence: 标签置信度（high/medium/low）
            parent_diqta_molecules: 父DIQTA分子列表
            tanimoto_to_parents: 与父分子的Tanimoto相似度
            physchem_properties: 理化性质
            bioactivity_evidence: 生物活性证据
            clinical_evidence: 临床证据
            literature_evidence: 文献证据
            evidence_density: 证据密度
            conflicts: 冲突信息

        Returns:
            完整的证据包
        """
        logger.info(f"组装证据包: {chembl_id}")

        # 获取分子基本信息
        molecule_info = self._get_molecule_info(chembl_id)

        # 创建标签信息
        label = LabelInfo(
            value=label_value,
            source=label_source,
            confidence=label_confidence
        )

        # 构建元数据
        metadata = {
            "retrieval_timestamp": datetime.now().isoformat(),
            "chembl_version": CHEMBL_VERSION,
            "pipeline_version": PIPELINE_VERSION
        }

        # 如果有冲突，添加到元数据
        if conflicts:
            metadata["conflicts"] = conflicts

        # 组装证据包
        bundle = EvidenceBundle(
            molecule_id=chembl_id,
            smiles=smiles,
            inchi_key=molecule_info.get("inchi_key"),
            pref_name=molecule_info.get("pref_name"),
            synonyms=molecule_info.get("synonyms", []),
            source_path=source_path,
            parent_diqta_molecules=parent_diqta_molecules or [],
            tanimoto_to_parents=tanimoto_to_parents or [],
            label=label,
            physchem_properties=physchem_properties or self._empty_physchem(),
            evidence_A_bioactivity=bioactivity_evidence or {},
            evidence_B_clinical=clinical_evidence or self._empty_clinical(),
            evidence_C_literature=literature_evidence or self._empty_literature(),
            evidence_density=evidence_density,
            conflicts=conflicts,
            metadata=metadata
        )

        return bundle

    def _get_molecule_info(self, chembl_id: str) -> Dict[str, Any]:
        """
        获取分子基本信息

        Args:
            chembl_id: ChEMBL ID

        Returns:
            分子信息字典
        """
        try:
            molecule, _ok = self.chembl_client.get_molecule_by_chembl_id(chembl_id)
            if not molecule or not isinstance(molecule, dict):
                return {}

            structures = molecule.get("molecule_structures", {})
            synonyms = molecule.get("molecule_synonyms", [])

            return {
                "inchi_key": structures.get("standard_inchi_key") if structures else None,
                "pref_name": molecule.get("pref_name"),
                "synonyms": [s.get("synonym") for s in synonyms if s.get("synonym")] if synonyms else []
            }

        except Exception as e:
            logger.warning(f"获取分子信息失败: {chembl_id}, 错误: {e}")
            return {}

    def _empty_physchem(self) -> PhysChemProperties:
        """返回空的理化性质对象"""
        from schemas import SourceInfo
        return PhysChemProperties(
            mw=SourceInfo(value=0, source="none"),
            logp=SourceInfo(value=0, source="none"),
            logd_7_4=None,
            tpsa=SourceInfo(value=0, source="none"),
            hbd=SourceInfo(value=0, source="none"),
            hba=SourceInfo(value=0, source="none"),
            rotatable_bonds=SourceInfo(value=0, source="none"),
            aromatic_rings=SourceInfo(value=0, source="none"),
            heavy_atoms=SourceInfo(value=0, source="none"),
            basic_pka=None,
            acidic_pka=None,
            ro5_violations=SourceInfo(value=0, source="none"),
            qed=SourceInfo(value=0, source="none"),
            rdkit_descriptors={},
            engineered_features={},
        )

    def _empty_clinical(self) -> ClinicalEvidence:
        """返回空的临床证据对象"""
        from schemas import WithdrawalInfo, ExternalDBFlags
        return ClinicalEvidence(
            max_phase=None,
            approved=None,
            withdrawn=WithdrawalInfo(flag=False),
            black_box_warning=None,
            clinical_trials=[],
            external_db_flags={
                "cardiotox": ExternalDBFlags(present=False),
                "etox": ExternalDBFlags(present=False),
                "dilirank": ExternalDBFlags(present=False)
            },
            fda_label_warnings=[]
        )

    def _empty_literature(self) -> LiteratureEvidence:
        """返回空的文献证据对象"""
        return LiteratureEvidence(
            pubmed_articles=[],
            patents=[]
        )


def create_evidence_bundle_assembler() -> EvidenceBundleAssembler:
    """创建证据包组装器的工厂函数"""
    return EvidenceBundleAssembler()