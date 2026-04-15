"""单元测试：M5 证据充分性判定器"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from m5_sufficiency import EvidenceSufficiencyJudge
from schemas import (
    TargetBioactivity,
    BioactivityMeasurement,
    ClinicalEvidence,
    WithdrawalInfo,
    ExternalDBFlags,
    LiteratureEvidence,
    PubMedArticle,
)


def _ext(
    *,
    cardiotox: ExternalDBFlags | None = None,
    etox: ExternalDBFlags | None = None,
    dilirank: ExternalDBFlags | None = None,
) -> dict:
    return {
        "cardiotox": cardiotox or ExternalDBFlags(present=False),
        "etox": etox or ExternalDBFlags(present=False),
        "dilirank": dilirank or ExternalDBFlags(present=False),
    }


class TestEvidenceSufficiencyJudge(unittest.TestCase):
    """测试证据充分性判定器"""

    def test_create_judge(self):
        judge = EvidenceSufficiencyJudge()
        self.assertIsNotNone(judge)

    def test_compute_evidence_density_empty(self):
        judge = EvidenceSufficiencyJudge()

        density = judge.compute_evidence_density(
            bioactivity_evidence={},
            clinical_evidence=ClinicalEvidence(
                max_phase=None,
                approved=None,
                withdrawn=WithdrawalInfo(flag=False),
                black_box_warning=None,
                clinical_trials=[],
                external_db_flags=_ext(),
                fda_label_warnings=[],
            ),
            literature_evidence=LiteratureEvidence(
                pubmed_articles=[],
                patents=[],
            ),
        )

        self.assertFalse(density.has_herg_measurement)
        self.assertFalse(density.has_clinical_phase_info)
        self.assertEqual(density.total_score, 0)
        self.assertEqual(density.sufficiency_tier, "insufficient")

    def test_compute_evidence_density_full(self):
        judge = EvidenceSufficiencyJudge()

        herg_data = TargetBioactivity(
            target="hERG",
            target_chembl_id="CHEMBL240",
            measurements=[
                BioactivityMeasurement(
                    type="IC50",
                    value=1.2,
                    units="uM",
                    pchembl=5.92,
                )
            ],
        )

        clinical = ClinicalEvidence(
            max_phase=4,
            approved=True,
            withdrawn=WithdrawalInfo(flag=True, reason="QT prolongation"),
            black_box_warning=True,
            clinical_trials=[],
            external_db_flags=_ext(
                cardiotox=ExternalDBFlags(present=True, risk_level="high")
            ),
            fda_label_warnings=["Cardiac warnings"],
        )

        literature = LiteratureEvidence(
            pubmed_articles=[
                PubMedArticle(
                    pmid="12345678",
                    title="Test article",
                    mesh_terms=[],
                    is_review=False,
                    relevance_keywords_hit=["hERG", "QT prolongation"],
                    molecule_mentioned=True,
                    publication_year=2020,
                )
            ],
            patents=[],
        )

        density = judge.compute_evidence_density(
            bioactivity_evidence={"hERG": herg_data},
            clinical_evidence=clinical,
            literature_evidence=literature,
        )

        self.assertTrue(density.has_herg_measurement)
        self.assertTrue(density.has_clinical_phase_info)
        self.assertTrue(density.has_withdrawal_info)
        self.assertEqual(density.external_db_hit_count, 1)
        self.assertEqual(density.pubmed_article_count, 1)
        self.assertEqual(density.total_score, 4)
        self.assertEqual(density.sufficiency_tier, "strong")

    def test_judge_label_withdrawn_qt_single_signal_medium(self):
        """仅撤市 QT 叙述：不上调 high。"""
        judge = EvidenceSufficiencyJudge()

        clinical = ClinicalEvidence(
            max_phase=4,
            approved=False,
            withdrawn=WithdrawalInfo(
                flag=True,
                year=2005,
                country="US",
                reason="QT prolongation leading to Torsades de Pointes",
            ),
            black_box_warning=False,
            clinical_trials=[],
            external_db_flags=_ext(),
            fda_label_warnings=[],
        )

        detail = judge.judge_label_detailed(
            bioactivity_evidence={},
            clinical_evidence=clinical,
        )

        self.assertEqual(detail.label, "torsadogenic")
        self.assertEqual(detail.confidence, "medium")

    def test_judge_label_withdrawal_plus_herg_high(self):
        judge = EvidenceSufficiencyJudge()
        herg_data = TargetBioactivity(
            target="hERG",
            target_chembl_id="CHEMBL240",
            measurements=[
                BioactivityMeasurement(
                    type="IC50",
                    value=0.5,
                    units="uM",
                    pchembl=6.0,
                )
            ],
        )
        clinical = ClinicalEvidence(
            max_phase=4,
            approved=True,
            withdrawn=WithdrawalInfo(
                flag=True,
                reason="QT prolongation",
            ),
            black_box_warning=False,
            clinical_trials=[],
            external_db_flags=_ext(),
            fda_label_warnings=[],
        )
        detail = judge.judge_label_detailed(
            bioactivity_evidence={"hERG": herg_data},
            clinical_evidence=clinical,
        )
        self.assertEqual(detail.label, "torsadogenic")
        self.assertEqual(detail.confidence, "high")

    def test_judge_label_herg_only_low(self):
        """仅 hERG 机制：low，不等同临床 TdP 标签。"""
        judge = EvidenceSufficiencyJudge()

        herg_data = TargetBioactivity(
            target="hERG",
            target_chembl_id="CHEMBL240",
            measurements=[
                BioactivityMeasurement(
                    type="IC50",
                    value=5.0,
                    units="uM",
                    pchembl=5.3,
                )
            ],
        )

        clinical = ClinicalEvidence(
            max_phase=4,
            approved=True,
            withdrawn=WithdrawalInfo(flag=False),
            black_box_warning=False,
            clinical_trials=[],
            external_db_flags=_ext(),
            fda_label_warnings=[],
        )

        detail = judge.judge_label_detailed(
            bioactivity_evidence={"hERG": herg_data},
            clinical_evidence=clinical,
        )

        self.assertEqual(detail.label, "torsadogenic")
        self.assertEqual(detail.confidence, "low")

    def test_judge_label_undetermined(self):
        judge = EvidenceSufficiencyJudge()

        detail = judge.judge_label_detailed(
            bioactivity_evidence={},
            clinical_evidence=ClinicalEvidence(
                max_phase=2,
                approved=False,
                withdrawn=WithdrawalInfo(flag=False),
                black_box_warning=False,
                clinical_trials=[],
                external_db_flags=_ext(),
                fda_label_warnings=[],
            ),
        )

        self.assertIsNone(detail.label)
        self.assertIsNone(detail.confidence)


if __name__ == "__main__":
    unittest.main()
