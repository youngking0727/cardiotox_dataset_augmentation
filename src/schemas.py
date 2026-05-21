"""Pydantic Schema定义"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union, Literal
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
    query_similarity_ok: bool = Field(
        True,
        description="False 表示 similarity.json 请求失败（非空列表与未命中）",
    )


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
    rdkit_descriptors: Dict[str, Any] = Field(
        default_factory=dict,
        description="RDKit 计算描述符全集（与 config/Features.json 对齐）",
    )
    engineered_features: Dict[str, Any] = Field(
        default_factory=dict,
        description="规则/SMARTS/hERG-like 等工程特征（与 config/Features.json 对齐）",
    )


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
    normalized_type: Optional[str] = Field(None, description="归一化后的测量类型")
    evidence_bucket: Optional[str] = Field(
        None,
        description="direct_qt / mechanistic_herg_ikr / secondary_pharmacology / unknown",
    )
    mechanistic_context_hit: bool = False
    direct_qt_context_hit: bool = False
    secondary_pharmacology_context_hit: bool = False


class TargetBioactivity(BaseModel):
    """靶点活性数据"""
    target: str
    target_chembl_id: str
    measurements: List[BioactivityMeasurement]
    supplemental_retention_rows: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="无法数值化但参与 Path B 保留判断的原始 activity 摘要",
    )


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
    qt_related_title: Optional[bool] = None  # 标题/摘要是否 QT 相关
    qt_related_outcome: Optional[bool] = None  # 主要终点是否 QT 相关
    qt_outcome_measure: Optional[str] = None   # QT 相关的主要终点描述
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
    approved: Optional[bool] = Field(
        None,
        description=(
            "是否与 max_phase 一致推导：max_phase==4 时为 True，0–3 为 False；"
            "无 drug 记录时为 None。勿与外部库占位字段混读。"
        ),
    )
    withdrawn: WithdrawalInfo
    black_box_warning: Optional[bool] = None
    clinical_trials: List[ClinicalTrialInfo]
    external_db_flags: Dict[str, ExternalDBFlags] = Field(
        ...,
        description=(
            "外部专项库（CardioTox/eTox/DILIrank 等）占位结构；全为 present=false 通常表示"
            "尚未接入或尚未映射，不代表「无心脏毒性风险」。"
        ),
    )
    fda_label_warnings: List[str] = Field(
        default_factory=list,
        description="FDA/标签警告摘录；空列表多因未接 Orange Book/标签 API，不代表无黑框或无心律风险。",
    )
    drug_info_status: Optional[str] = Field(
        None,
        description=(
            "ChEMBL drug.json 拉取状态：ok | not_found | request_failed；"
            "null 表示未写入状态位，不等同于「无临床信息」。"
        ),
    )
    direct_qt_clinical_hit: bool = Field(
        False,
        description="撤市理由/试验标题摘要/FDA 警告等是否含直接 QT/心律失常语义（Path B 保留与 M5 共用）",
    )


# M3-C: 文献证据
class EvidenceContext(BaseModel):
    """服务返回的 QT/hERG 证据段落"""
    source: str = Field("", description="fulltext / abstract")
    section: str = Field("", description="Introduction / Results / Discussion 等")
    matched_term: str = Field("", description="匹配到的心脏毒性关键词")
    evidence_type: str = Field(
        "",
        description="clinical_or_direct_qt_evidence / mechanistic_herg_ikr_evidence / secondary_pharmacology_evidence",
    )
    context: str = Field("", description="原文片段")


class PubMedArticle(BaseModel):
    """PubMed文章"""
    pmid: str
    title: str
    abstract: Optional[str] = None
    mesh_terms: List[str] = Field(default_factory=list)
    is_review: bool = False
    relevance_keywords_hit: List[str] = Field(default_factory=list)
    molecule_mentioned: bool = True
    publication_year: Optional[int] = None
    relevance_bucket: Optional[str] = Field(
        None,
        description="priority_1_direct_qt | priority_2_mechanistic_herg_ikr | priority_3_secondary_pharmacology | irrelevant",
    )
    relevance_reason_codes: List[str] = Field(default_factory=list)
    contexts: List[EvidenceContext] = Field(
        default_factory=list,
        description="服务返回的 QT/hERG 证据段落",
    )
    source: str = Field(
        "",
        description="chembl_known / pubmed_search 等",
    )
    doi: Optional[str] = None


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
    priority_1_article_count: int = 0
    priority_2_article_count: int = 0
    priority_3_article_count: int = 0
    top_relevance_bucket: Optional[str] = None
    top_relevance_reason_codes: List[str] = Field(default_factory=list)
    chembl_enrichment: Optional[Dict[str, Any]] = Field(
        None,
        description="服务返回的 ChEMBL enrichment（name_variants、known_pubmed_ids、herg_activities），追溯证据怎么查出来的",
    )


# M4: 证据包
class LabelInfo(BaseModel):
    """标签信息"""
    value: Optional[str] = Field(None, description="torsadogenic/non-torsadogenic")
    source: str = Field(..., description="diqta_ground_truth/evidence_derived/undetermined")
    confidence: Optional[str] = Field(None, description="high/medium/low")


EvidenceSufficiencyTier = Literal["insufficient", "usable", "strong"]


class EvidenceDensity(BaseModel):
    """证据密度指标（第一层：充分性；total_score 映射 sufficiency_tier）"""

    has_herg_measurement: bool
    has_clinical_phase_info: bool
    has_withdrawal_info: bool
    external_db_hit_count: int
    pubmed_article_count: int
    pubmed_cardiotox_relevant_count: int
    patent_count: int
    total_score: int
    sufficiency_tier: EvidenceSufficiencyTier = Field(
        ...,
        description="insufficient：难继续推理；usable：可入库/可推理；strong：多维度命中",
    )


class RetentionJudgmentDetail(BaseModel):
    """Path B：是否保留进入扩充集（与标签判定解耦）"""

    should_keep: bool
    evidence_priority: Optional[str] = Field(
        None,
        description="priority_1_direct_qt | priority_2_mechanistic_herg_ikr | priority_3_secondary_pharmacology",
    )
    reason_codes: List[str] = Field(default_factory=list)
    notes: str = ""


class LabelJudgmentDetail(BaseModel):
    """第二层：标签判定输出（与充分性分层解耦；confidence 语义见 notes）"""

    label: Optional[str] = Field(
        None,
        description="torsadogenic / non-torsadogenic；None 表示 undetermined",
    )
    confidence: Optional[str] = Field(
        None,
        description="low / medium / high；机制提示与临床 TdP 结论区分见 notes",
    )
    rationale_codes: List[str] = Field(
        default_factory=list,
        description="命中的规则/信号编码，便于审计",
    )
    notes: str = Field(
        "",
        description="为何给出该置信度、与真实世界标签的差异提示",
    )


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
