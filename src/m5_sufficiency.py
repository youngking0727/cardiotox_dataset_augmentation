"""M5: 证据充分性判定器模块"""

import logging
from typing import Optional, Dict, Any, Tuple

from ..schemas import (
    EvidenceDensity, ClinicalEvidence, LiteratureEvidence,
    TargetBioactivity, LabelInfo
)
from .m3b_clinical import ClinicalStatusRetriever
from .m3c_literature import LiteratureRetriever

logger = logging.getLogger(__name__)


class EvidenceSufficiencyJudge:
    """证据充分性判定器"""

    # V1临时规则配置
    HERG_IC50_THRESHOLD_TORSADOGENIC = 10.0  # uM
    HERG_IC50_THRESHOLD_STRONG = 1.0  # uM

    def __init__(self, clinical_retriever: Optional[ClinicalStatusRetriever] = None,
                 literature_retriever: Optional[LiteratureRetriever] = None):
        """
        初始化证据充分性判定器

        Args:
            clinical_retriever: 临床状态检索器
            literature_retriever: 文献检索器
        """
        self.clinical_retriever = clinical_retriever or ClinicalStatusRetriever()
        self.literature_retriever = literature_retriever or LiteratureRetriever()

    def compute_evidence_density(self,
                                 bioactivity_evidence: Dict[str, TargetBioactivity],
                                 clinical_evidence: ClinicalEvidence,
                                 literature_evidence: LiteratureEvidence) -> EvidenceDensity:
        """
        计算证据密度指标

        Args:
            bioactivity_evidence: 生物活性证据
            clinical_evidence: 临床证据
            literature_evidence: 文献证据

        Returns:
            证据密度对象
        """
        # 是否有hERG测量
        has_herg_measurement = "hERG" in bioactivity_evidence and bool(
            bioactivity_evidence["hERG"].measurements
        )

        # 是否有临床阶段信息
        has_clinical_phase_info = clinical_evidence.max_phase is not None

        # 是否有撤市信息
        has_withdrawal_info = clinical_evidence.withdrawn.flag

        # 外部数据库命中数
        external_db_hit_count = sum(
            1 for db_flag in clinical_evidence.external_db_flags.values()
            if db_flag.present
        )

        # PubMed文章数
        pubmed_article_count = len(literature_evidence.pubmed_articles)

        # 心脏毒性相关PubMed文章数
        pubmed_cardiotox_relevant_count = self.literature_retriever.get_cardiotox_relevant_count(
            literature_evidence.pubmed_articles
        )

        # 专利数
        patent_count = len(literature_evidence.patents)

        # 总分（简单加权）
        total_score = sum([
            has_herg_measurement,
            has_clinical_phase_info,
            has_withdrawal_info,
            external_db_hit_count > 0,
            pubmed_cardiotox_relevant_count >= 3,
            patent_count > 0
        ])

        return EvidenceDensity(
            has_herg_measurement=has_herg_measurement,
            has_clinical_phase_info=has_clinical_phase_info,
            has_withdrawal_info=has_withdrawal_info,
            external_db_hit_count=external_db_hit_count,
            pubmed_article_count=pubmed_article_count,
            pubmed_cardiotox_relevant_count=pubmed_cardiotox_relevant_count,
            patent_count=patent_count,
            total_score=total_score
        )

    def judge_label(self,
                   bioactivity_evidence: Dict[str, TargetBioactivity],
                   clinical_evidence: ClinicalEvidence) -> Tuple[Optional[str], Optional[str]]:
        """
        判定标签（V1临时规则）

        规则：
        1. 若有明确的withdrawn_reason提及QT/TdP → torsadogenic, high
        2. 若有hERG IC50 < 10μM 且有PubMed文献支持 → torsadogenic, medium
        3. 若有CardioTox明确记录 → 采纳其标签, high
        4. 否则 → undetermined

        Args:
            bioactivity_evidence: 生物活性证据
            clinical_evidence: 临床证据

        Returns:
            (标签值, 置信度)
        """
        # 规则1: 检查是否因QT/TdP撤市
        if clinical_evidence.withdrawn.flag:
            reason = clinical_evidence.withdrawn.reason or ""
            reason_lower = reason.lower()
            qt_keywords = ["qt", "torsades", "torsade", "arrhythmia"]

            if any(kw in reason_lower for kw in qt_keywords):
                return "torsadogenic", "high"

        # 规则2: 检查hERG活性
        herg_data = bioactivity_evidence.get("hERG")
        if herg_data and herg_data.measurements:
            # 取最有效的IC50
            best_ic50 = herg_data.measurements[0].value

            if best_ic50 < self.HERG_IC50_THRESHOLD_TORSADOGENIC:
                return "torsadogenic", "medium"

        # 规则3: 检查外部数据库（CardioTox）
        cardiotox_flag = clinical_evidence.external_db_flags.get("cardiotox")
        if cardiotox_flag and cardiotox_flag.present:
            risk_level = cardiotox_flag.risk_level or ""
            if "high" in risk_level.lower():
                return "torsadogenic", "high"
            elif "medium" in risk_level.lower() or "low" in risk_level.lower():
                return "non-torsadogenic", "high"

        # 默认：无法确定
        return None, None


def create_evidence_sufficiency_judge() -> EvidenceSufficiencyJudge:
    """创建证据充分性判定器的工厂函数"""
    return EvidenceSufficiencyJudge()