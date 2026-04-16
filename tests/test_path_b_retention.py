"""Path B 保留层 judge_retention_for_path_b 验收测试。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from m5_sufficiency import EvidenceSufficiencyJudge
from rules.cardiotox_evidence_rules import (
    EVIDENCE_PRIORITY_1,
    EVIDENCE_PRIORITY_2,
    EVIDENCE_PRIORITY_3,
)
from schemas import (
    TargetBioactivity,
    BioactivityMeasurement,
    ClinicalEvidence,
    WithdrawalInfo,
    ExternalDBFlags,
    LiteratureEvidence,
    PubMedArticle,
)


def _ext() -> dict:
    return {
        "cardiotox": ExternalDBFlags(present=False),
        "etox": ExternalDBFlags(present=False),
        "dilirank": ExternalDBFlags(present=False),
    }


def _clinical_empty() -> ClinicalEvidence:
    return ClinicalEvidence(
        max_phase=None,
        approved=None,
        withdrawn=WithdrawalInfo(flag=False),
        black_box_warning=None,
        clinical_trials=[],
        external_db_flags=_ext(),
        fda_label_warnings=[],
    )


def _lit_empty() -> LiteratureEvidence:
    return LiteratureEvidence(pubmed_articles=[], patents=[])


class TestPathBRetention(unittest.TestCase):
    def test_priority2_herg_mechanism_keeps(self):
        """hERG IC50 + 机制语境 → priority_2，应保留。"""
        judge = EvidenceSufficiencyJudge()
        m = BioactivityMeasurement(
            type="IC50",
            value=1.0,
            units="uM",
            mechanistic_context_hit=True,
            direct_qt_context_hit=False,
        )
        tb = TargetBioactivity(
            target="hERG",
            target_chembl_id="CHEMBL240",
            measurements=[m],
        )
        r = judge.judge_retention_for_path_b({"hERG": tb}, _clinical_empty(), _lit_empty())
        self.assertTrue(r.should_keep)
        self.assertEqual(r.evidence_priority, EVIDENCE_PRIORITY_2)

    def test_priority1_literature_fpd_terms(self):
        """仅文献命中直接 QT 术语 → priority_1。"""
        judge = EvidenceSufficiencyJudge()
        art = PubMedArticle(
            pmid="1",
            title="FPD and QTc prolongation in hiPSC-CM MEA assay with drug X",
            abstract="",
            mesh_terms=[],
            is_review=False,
            relevance_keywords_hit=[],
            molecule_mentioned=True,
            relevance_bucket=EVIDENCE_PRIORITY_1,
            relevance_reason_codes=["direct_qt_term:FPD"],
        )
        le = LiteratureEvidence(
            pubmed_articles=[art],
            patents=[],
            priority_1_article_count=1,
        )
        r = judge.judge_retention_for_path_b({}, _clinical_empty(), le)
        self.assertTrue(r.should_keep)
        self.assertEqual(r.evidence_priority, EVIDENCE_PRIORITY_1)

    def test_priority3_secondary_pharmacology(self):
        ac50_art = PubMedArticle(
            pmid="2",
            title="Secondary pharmacology and off-target ADR association with cardiotoxicity risk",
            abstract="",
            mesh_terms=[],
            is_review=False,
            relevance_keywords_hit=[],
            molecule_mentioned=True,
            relevance_bucket=EVIDENCE_PRIORITY_3,
            relevance_reason_codes=["secondary_pharm_plus_cardiac_bridge"],
        )
        le = LiteratureEvidence(
            pubmed_articles=[ac50_art],
            patents=[],
            priority_3_article_count=1,
        )
        r = EvidenceSufficiencyJudge().judge_retention_for_path_b(
            {}, _clinical_empty(), le
        )
        self.assertTrue(r.should_keep)
        self.assertEqual(r.evidence_priority, EVIDENCE_PRIORITY_3)

    def test_no_signal_drops(self):
        r = EvidenceSufficiencyJudge().judge_retention_for_path_b(
            {}, _clinical_empty(), _lit_empty()
        )
        self.assertFalse(r.should_keep)
        self.assertIsNone(r.evidence_priority)

    def test_retrieval_incomplete_is_out_of_scope(self):
        """
        retrieval_incomplete 由 process_path_b 在调用 retention 之前返回；
        此处仅文档化：judge_retention 不因「无标签」拒绝。
        """
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
