"""M6: 冲突检测器模块"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from schemas import (
    TargetBioactivity, ClinicalEvidence, LiteratureEvidence,
    PubMedArticle
)

logger = logging.getLogger(__name__)


@dataclass
class ConflictResult:
    """冲突检测结果"""
    has_conflict: bool
    conflict_A_vs_B: Optional[str] = None
    conflict_within_C: Optional[str] = None
    conflict_details: Optional[Dict[str, Any]] = None


class ConflictDetector:
    """冲突检测器

    当前版本只做检测与标记，不做裁决。
    """

    # hERG高活性阈值
    HERG_HIGH_ACTIVITY_UM = 1.0  # uM

    def __init__(self):
        """初始化冲突检测器"""
        pass

    def detect(self,
              bioactivity_evidence: Dict[str, TargetBioactivity],
              clinical_evidence: ClinicalEvidence,
              literature_evidence: LiteratureEvidence) -> ConflictResult:
        """
        检测证据冲突

        Args:
            bioactivity_evidence: 生物活性证据（A类）
            clinical_evidence: 临床证据（B类）
            literature_evidence: 文献证据（C类）

        Returns:
            冲突检测结果
        """
        conflicts = []

        # 检测A类vs B类冲突
        a_vs_b_conflict = self._detect_A_vs_B_conflict(
            bioactivity_evidence, clinical_evidence
        )
        if a_vs_b_conflict:
            conflicts.append(a_vs_b_conflict)

        # 检测C类内部冲突
        c_conflict = self._detect_C_conflict(literature_evidence)
        if c_conflict:
            conflicts.append(c_conflict)

        if conflicts:
            return ConflictResult(
                has_conflict=True,
                conflict_A_vs_B=conflicts[0] if a_vs_b_conflict else None,
                conflict_within_C=conflicts[-1] if c_conflict else None,
                conflict_details={"conflicts": conflicts}
            )

        return ConflictResult(has_conflict=False)

    def _detect_A_vs_B_conflict(self,
                                bioactivity_evidence: Dict[str, TargetBioactivity],
                                clinical_evidence: ClinicalEvidence) -> Optional[str]:
        """
        检测A类（体外）vs B类（临床）冲突

        场景：hERG IC50 < 1μM（高体外毒性）但无临床警告

        Args:
            bioactivity_evidence: 生物活性证据
            clinical_evidence: 临床证据

        Returns:
            冲突描述，如果没有冲突返回None
        """
        # 检查hERG高活性
        herg_data = bioactivity_evidence.get("hERG")
        if not herg_data or not herg_data.measurements:
            return None

        best_ic50 = herg_data.measurements[0].value

        # 只有当hERG IC50 < 1μM时才可能存在冲突
        if best_ic50 >= self.HERG_HIGH_ACTIVITY_UM:
            return None

        # 检查临床是否有警告
        has_clinical_warning = (
            clinical_evidence.withdrawn.flag or
            clinical_evidence.black_box_warning or
            bool(clinical_evidence.fda_label_warnings)
        )

        # 如果有高体外毒性但无临床警告，标记为冲突
        if not has_clinical_warning:
            return (
                f"hERG IC50 = {best_ic50:.2f} μM (high in vitro toxicity) "
                "but no clinical warning found"
            )

        return None

    def _detect_C_conflict(self,
                          literature_evidence: LiteratureEvidence) -> Optional[str]:
        """
        检测C类（文献）内部冲突

        场景：不同文献给出相反结论

        Args:
            literature_evidence: 文献证据

        Returns:
            冲突描述，如果没有冲突返回None
        """
        articles = literature_evidence.pubmed_articles

        if len(articles) < 2:
            return None

        # 简化检测：检查是否有综述文章（可能与其他文章冲突）
        has_review = any(a.is_review for a in articles)

        # 统计关键词分布
        cardiotox_articles = [
            a for a in articles
            if a.relevance_keywords_hit and a.molecule_mentioned and not a.is_review
        ]

        # 简单冲突检测：如果有综述且有实际研究文章，可能存在观点差异
        # 实际应用中需要更复杂的NLP分析

        if has_review and len(cardiotox_articles) > 0:
            # 这里可以做更复杂的冲突检测
            pass

        return None

    def get_conflict_summary(self, conflict_result: ConflictResult) -> str:
        """
        获取冲突摘要

        Args:
            conflict_result: 冲突检测结果

        Returns:
            冲突描述字符串
        """
        if not conflict_result.has_conflict:
            return "No conflicts detected"

        parts = []
        if conflict_result.conflict_A_vs_B:
            parts.append(f"A vs B: {conflict_result.conflict_A_vs_B}")
        if conflict_result.conflict_within_C:
            parts.append(f"C: {conflict_result.conflict_within_C}")

        return "; ".join(parts) if parts else "No conflicts detected"


def create_conflict_detector() -> ConflictDetector:
    """创建冲突检测器的工厂函数"""
    return ConflictDetector()