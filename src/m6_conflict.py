"""M6: 冲突检测器模块"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from schemas import TargetBioactivity, ClinicalEvidence, LiteratureEvidence

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

    def _detect_C_conflict(
        self, literature_evidence: LiteratureEvidence
    ) -> Optional[str]:
        """C 类文献内部对立结论：预留，当前未实现（需结构化对立或 NLP）。"""
        del literature_evidence
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