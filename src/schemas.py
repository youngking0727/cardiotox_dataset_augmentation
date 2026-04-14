"""Pydantic Schema定义"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


# 基础模型
class SourceInfo(BaseModel):
    """来源信息"""
    value: Union[float, int, str, bool]
    source: str = Field(..., description="数据来源: chembl/rdkit/pkasolver/calculated/clinicaltrials/pubmed/patent")


# M1: 相似性检索器输出
class SimilarMolecule(BaseModel):
    """相似分子信息"""
    chembl_id: str
    smiles: str
    tanimoto: float = Field(..., ge=0, le=1)
    already_in_diqta: bool


class SimilarityResult(BaseModel):
    """相似性检索结果"""
    query_smiles: str
    query_chembl_id: Optional[str] = None
    similar_molecules: List[SimilarMolecule]


# M2: 理化性质
class PhysChemProperties(BaseModel):
    """理化性质集合"""
    mw: SourceInfo = Field(..., description="分子量")
    logp: SourceInfo = Field(..., description="logP")
    logd_7_4: Optional[SourceInfo] = Field(None, description="logD at pH 7.4")
    tpsa: SourceInfo = Field(..., description="TPSA")
    hbd: SourceInfo = Field(..., description="氢键供体数")
    hba: SourceInfo = Field(..., description="氢键受体数")
    rotatable_bonds: SourceInfo = Field(..., description="可旋转键数")
    aromatic_rings: SourceInfo = Field(..., description="芳香环数")
    heavy_atoms: SourceInfo = Field(..., description="重原子数")
    basic_pka: Optional[SourceInfo] = Field(None, description="碱性中心pKa")
    acidic_pka: Optional[SourceInfo] = Field(None, description="酸性中心pKa")
    ro5_violations: SourceInfo = Field(..., description="Lipinski规则违反数")
    qed: SourceInfo = Field(..., description="QED类药性评分")


# M3-A: 靶点活性
class BioactivityMeasurement(BaseModel):
    """单个生物活性测量值"""
    type: str = Field(..., description="IC50/Ki/Kd")
    value: float
    units: str
    pchembl: Optional[float] = None
    assay_type: Optional[str] = Field(None, description="B=Binding, F=Functional")
    assay_description: Optional[str] = None
    document_chembl_id: Optional[str] = None
    confidence_score: Optional[int] = Field(None, ge=0, le=9)


class TargetBioactivity(BaseModel):
    """靶点活性数据"""
    target: str
    target_chembl_id: str
    measurements: List[BioactivityMeasurement]


# M3-B: 临床状态
class WithdrawalInfo(BaseModel):
    """撤市信息"""
    flag: bool
    year: Optional[int] = None
    country: Optional[str] = None
    reason: Optional[str] = None


class ClinicalTrialInfo(BaseModel):
    """临床试验信息"""
    nct_id: str
    title: str
    status: str
    qt_related: bool
    summary: Optional[str] = None


class ExternalDBFlags(BaseModel):
    """外部数据库标记"""
    present: bool
    risk_level: Optional[str] = None
    severity: Optional[str] = None
    evidence: Optional[str] = None


class ClinicalEvidence(BaseModel):
    """临床证据"""
    max_phase: Optional[int] = Field(None, ge=0, le=4)
    approved: Optional[bool] = None
    withdrawn: WithdrawalInfo
    black_box_warning: Optional[bool] = None
    clinical_trials: List[ClinicalTrialInfo]
    external_db_flags: Dict[str, ExternalDBFlags]
    fda_label_warnings: List[str]


# M3-C: 文献证据
class PubMedArticle(BaseModel):
    """PubMed文章"""
    pmid: str
    title: str
    abstract: Optional[str] = None
    mesh_terms: List[str]
    is_review: bool
    relevance_keywords_hit: List[str]
    molecule_mentioned: bool
    publication_year: Optional[int] = None


class PatentInfo(BaseModel):
    """专利信息"""
    patent_id: str
    title: str
    abstract_excerpt: Optional[str] = None
    cardiotox_related: bool


class LiteratureEvidence(BaseModel):
    """文献证据"""
    pubmed_articles: List[PubMedArticle]
    patents: List[PatentInfo]


# M4: 证据包
class LabelInfo(BaseModel):
    """标签信息"""
    value: Optional[str] = Field(None, description="torsadogenic/non-torsadogenic")
    source: str = Field(..., description="diqta_ground_truth/evidence_derived/undetermined")
    confidence: Optional[str] = Field(None, description="high/medium/low")


class EvidenceDensity(BaseModel):
    """证据密度指标"""
    has_herg_measurement: bool
    has_clinical_phase_info: bool
    has_withdrawal_info: bool
    external_db_hit_count: int
    pubmed_article_count: int
    pubmed_cardiotox_relevant_count: int
    patent_count: int
    total_score: int


class ConflictInfo(BaseModel):
    """冲突信息"""
    conflict_A_vs_B: Optional[str] = None
    conflict_within_C: Optional[str] = None


class EvidenceBundle(BaseModel):
    """完整的证据包"""
    molecule_id: str = Field(..., description="ChEMBL ID")
    smiles: str
    inchi_key: Optional[str] = None
    pref_name: Optional[str] = None
    synonyms: List[str] = Field(default_factory=list)
    source_path: str = Field(..., description="A_diqta_original/B_similarity_expanded")
    parent_diqta_molecules: List[str] = Field(default_factory=list)
    tanimoto_to_parents: List[float] = Field(default_factory=list)

    label: LabelInfo

    physchem_properties: PhysChemProperties

    evidence_A_bioactivity: Dict[str, TargetBioactivity]
    evidence_B_clinical: ClinicalEvidence
    evidence_C_literature: LiteratureEvidence

    evidence_density: Optional[EvidenceDensity] = None
    conflicts: Optional[ConflictInfo] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)


# 配置模型
class TargetConfig(BaseModel):
    """靶点配置"""
    name: str
    chembl_id: str
    priority: int


class KeywordsConfig(BaseModel):
    """关键词配置"""
    cardiotox_keywords: List[str]
    qt_keywords: List[str]
    arrhythmia_keywords: List[str]
    review_mesh_terms: List[str]


class PipelineConfig(BaseModel):
    """Pipeline配置"""
    tanimoto_threshold: float = Field(default=0.8, ge=0, le=1)
    herg_threshold_um: float = Field(default=10.0)
    evidence_sufficiency_score_threshold: int = Field(default=5)
    cache_ttl_hours: int = Field(default=24)
    max_retries: int = Field(default=3)
    request_delay_seconds: float = Field(default=0.1)


# 输入数据模型
class DIQTAMolecule(BaseModel):
    """DIQTA输入分子"""
    chembl_id: str
    smiles: str
    label: str = Field(..., description="torsadogenic/non-torsadogenic")
    inchi_key: Optional[str] = None
    pref_name: Optional[str] = None


class ProcessingResult(BaseModel):
    """处理结果"""
    success: bool
    molecule_id: str
    evidence_bundle: Optional[EvidenceBundle] = None
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
