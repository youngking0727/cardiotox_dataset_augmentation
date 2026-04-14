"""单元测试：M6 冲突检测器"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from m6_conflict import ConflictDetector, ConflictResult
from schemas import (
    TargetBioactivity, BioactivityMeasurement,
    ClinicalEvidence, WithdrawalInfo, ExternalDBFlags,
    LiteratureEvidence, PubMedArticle
)


class TestConflictDetector:
    """测试冲突检测器"""

    def test_create_detector(self):
        """测试创建检测器"""
        detector = ConflictDetector()
        assert detector is not None

    def test_no_conflict(self):
        """测试无冲突情况"""
        detector = ConflictDetector()

        herg_data = TargetBioactivity(
            target="hERG",
            target_chembl_id="CHEMBL240",
            measurements=[
                BioactivityMeasurement(
                    type="IC50",
                    value=5.0,
                    units="uM"
                )
            ]
        )

        clinical = ClinicalEvidence(
            max_phase=4,
            approved=True,
            withdrawn=WithdrawalInfo(flag=False),
            black_box_warning=False,
            clinical_trials=[],
            external_db_flags={},
            fda_label_warnings=[]
        )

        literature = LiteratureEvidence(
            pubmed_articles=[],
            patents=[]
        )

        result = detector.detect(
            bioactivity_evidence={"hERG": herg_data},
            clinical_evidence=clinical,
            literature_evidence=literature
        )

        assert result.has_conflict is False

    def test_conflict_A_vs_B(self):
        """测试A类vs B类冲突"""
        detector = ConflictDetector()

        # hERG IC50 < 1uM（高活性）
        herg_data = TargetBioactivity(
            target="hERG",
            target_chembl_id="CHEMBL240",
            measurements=[
                BioactivityMeasurement(
                    type="IC50",
                    value=0.5,  # < 1uM，高活性
                    units="uM"
                )
            ]
        )

        # 但无临床警告
        clinical = ClinicalEvidence(
            max_phase=4,
            approved=True,
            withdrawn=WithdrawalInfo(flag=False),
            black_box_warning=False,
            clinical_trials=[],
            external_db_flags={},
            fda_label_warnings=[]
        )

        literature = LiteratureEvidence(
            pubmed_articles=[],
            patents=[]
        )

        result = detector.detect(
            bioactivity_evidence={"hERG": herg_data},
            clinical_evidence=clinical,
            literature_evidence=literature
        )

        assert result.has_conflict is True
        assert result.conflict_A_vs_B is not None

    def test_no_conflict_with_warning(self):
        """测试有临床警告时无冲突"""
        detector = ConflictDetector()

        # hERG IC50 < 1uM
        herg_data = TargetBioactivity(
            target="hERG",
            target_chembl_id="CHEMBL240",
            measurements=[
                BioactivityMeasurement(
                    type="IC50",
                    value=0.5,
                    units="uM"
                )
            ]
        )

        # 且有黑框警告
        clinical = ClinicalEvidence(
            max_phase=4,
            approved=True,
            withdrawn=WithdrawalInfo(flag=False),
            black_box_warning=True,  # 有警告
            clinical_trials=[],
            external_db_flags={},
            fda_label_warnings=[]
        )

        literature = LiteratureEvidence(
            pubmed_articles=[],
            patents=[]
        )

        result = detector.detect(
            bioactivity_evidence={"hERG": herg_data},
            clinical_evidence=clinical,
            literature_evidence=literature
        )

        assert result.has_conflict is False


class TestConflictResult:
    """测试冲突结果模型"""

    def test_conflict_result_default(self):
        """测试默认冲突结果"""
        result = ConflictResult(has_conflict=False)
        assert result.has_conflict is False
        assert result.conflict_A_vs_B is None
        assert result.conflict_within_C is None

    def test_conflict_result_with_conflicts(self):
        """测试有冲突的结果"""
        result = ConflictResult(
            has_conflict=True,
            conflict_A_vs_B="hERG high activity but no warning",
            conflict_within_C=None
        )
        assert result.has_conflict is True
        assert "hERG" in result.conflict_A_vs_B

    def test_get_conflict_summary_no_conflict(self):
        """测试无冲突摘要"""
        detector = ConflictDetector()
        result = ConflictResult(has_conflict=False)
        summary = detector.get_conflict_summary(result)
        assert summary == "No conflicts detected"

    def test_get_conflict_summary_with_conflict(self):
        """测试有冲突摘要"""
        detector = ConflictDetector()
        result = ConflictResult(
            has_conflict=True,
            conflict_A_vs_B="Test conflict"
        )
        summary = detector.get_conflict_summary(result)
        assert "Test conflict" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])