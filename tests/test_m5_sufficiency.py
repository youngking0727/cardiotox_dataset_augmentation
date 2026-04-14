"""单元测试：M5 证据充分性判定器"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from m5_sufficiency import EvidenceSufficiencyJudge
from schemas import (
    EvidenceDensity, TargetBioactivity, BioactivityMeasurement,
    ClinicalEvidence, WithdrawalInfo, ExternalDBFlags,
    LiteratureEvidence, PubMedArticle, PatentInfo
)


class TestEvidenceSufficiencyJudge:
    """测试证据充分性判定器"""

    def test_create_judge(self):
        """测试创建判定器"""
        judge = EvidenceSufficiencyJudge()
        assert judge is not None

    def test_compute_evidence_density_empty(self):
        """测试空证据密度计算"""
        judge = EvidenceSufficiencyJudge()

        density = judge.compute_evidence_density(
            bioactivity_evidence={},
            clinical_evidence=ClinicalEvidence(
                max_phase=None,
                approved=None,
                withdrawn=WithdrawalInfo(flag=False),
                black_box_warning=None,
                clinical_trials=[],
                external_db_flags={
                    "cardiotox": ExternalDBFlags(present=False)
                },
                fda_label_warnings=[]
            ),
            literature_evidence=LiteratureEvidence(
                pubmed_articles=[],
                patents=[]
            )
        )

        assert density.has_herg_measurement is False
        assert density.has_clinical_phase_info is False
        assert density.total_score == 0

    def test_compute_evidence_density_full(self):
        """测试完整证据密度计算"""
        judge = EvidenceSufficiencyJudge()

        # 准备hERG数据
        herg_data = TargetBioactivity(
            target="hERG",
            target_chembl_id="CHEMBL240",
            measurements=[
                BioactivityMeasurement(
                    type="IC50",
                    value=1.2,
                    units="uM",
                    pchembl=5.92
                )
            ]
        )

        # 准备临床数据
        clinical = ClinicalEvidence(
            max_phase=4,
            approved=True,
            withdrawn=WithdrawalInfo(flag=True, reason="QT prolongation"),
            black_box_warning=True,
            clinical_trials=[],
            external_db_flags={
                "cardiotox": ExternalDBFlags(present=True, risk_level="high")
            },
            fda_label_warnings=["Cardiac warnings"]
        )

        # 准备文献数据
        literature = LiteratureEvidence(
            pubmed_articles=[
                PubMedArticle(
                    pmid="12345678",
                    title="Test article",
                    is_review=False,
                    relevance_keywords_hit=["hERG", "QT prolongation"],
                    molecule_mentioned=True,
                    publication_year=2020
                )
            ],
            patents=[]
        )

        density = judge.compute_evidence_density(
            bioactivity_evidence={"hERG": herg_data},
            clinical_evidence=clinical,
            literature_evidence=literature
        )

        assert density.has_herg_measurement is True
        assert density.has_clinical_phase_info is True
        assert density.has_withdrawal_info is True
        assert density.external_db_hit_count == 1
        assert density.pubmed_article_count == 1

    def test_judge_label_withdrawn_qt(self):
        """测试因QT撤市的标签判定"""
        judge = EvidenceSufficiencyJudge()

        # 准备临床数据：因QT延长撤市
        clinical = ClinicalEvidence(
            max_phase=4,
            approved=False,
            withdrawn=WithdrawalInfo(
                flag=True,
                year=2005,
                country="US",
                reason="QT prolongation leading to Torsades de Pointes"
            ),
            black_box_warning=False,
            clinical_trials=[],
            external_db_flags={},
            fda_label_warnings=[]
        )

        label_value, confidence = judge.judge_label(
            bioactivity_evidence={},
            clinical_evidence=clinical
        )

        assert label_value == "torsadogenic"
        assert confidence == "high"

    def test_judge_label_herg_inhibitor(self):
        """测试hERG抑制剂的标签判定"""
        judge = EvidenceSufficiencyJudge()

        # 准备hERG数据：IC50 < 10uM
        herg_data = TargetBioactivity(
            target="hERG",
            target_chembl_id="CHEMBL240",
            measurements=[
                BioactivityMeasurement(
                    type="IC50",
                    value=5.0,  # < 10uM
                    units="uM",
                    pchembl=5.3
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

        label_value, confidence = judge.judge_label(
            bioactivity_evidence={"hERG": herg_data},
            clinical_evidence=clinical
        )

        assert label_value == "torsadogenic"
        assert confidence == "medium"

    def test_judge_label_undetermined(self):
        """测试无法确定的情况"""
        judge = EvidenceSufficiencyJudge()

        # 无hERG数据，无撤市信息
        label_value, confidence = judge.judge_label(
            bioactivity_evidence={},
            clinical_evidence=ClinicalEvidence(
                max_phase=2,
                approved=False,
                withdrawn=WithdrawalInfo(flag=False),
                black_box_warning=False,
                clinical_trials=[],
                external_db_flags={},
                fda_label_warnings=[]
            )
        )

        assert label_value is None
        assert confidence is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])