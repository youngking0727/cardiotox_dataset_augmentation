"""Tests for strict per-outcome ECG/QT classification."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.clinicaltrials_result_evidence import classify_outcome_text  # noqa: E402


def _t(text: str) -> str:
    return classify_outcome_text(text)["evidence_type"]


def test_standalone_qt_not_qt_specific():
    assert _t("QT") == "non_qt"
    assert _t("Q-T") == "non_qt"
    assert _t("Change in QT score") == "non_qt"


def test_qt_with_context_is_qt_specific():
    assert _t("QT interval change from baseline") == "qt_specific"
    assert _t("Q-T interval") == "qt_specific"
    assert _t("QT prolongation") == "qt_specific"


def test_qtc_direct():
    assert _t("QTc") == "qt_specific"
    assert _t("Q-Tc interval") == "qt_specific"
    assert _t("QTcF Fridericia corrected") == "qt_specific"


def test_qtc_with_pk_not_excluded():
    assert _t("Relationship between QTc and Cmax") == "qt_specific"
    assert _t("Exposure-QTc analysis with AUC") == "qt_specific"


def test_ecg_conduction():
    assert _t("QRS duration") == "ecg_conduction"
    assert _t("PR interval") == "ecg_conduction"
    assert _t("RR interval") == "ecg_conduction"


def test_ecg_broad_only():
    assert _t("12-lead ECG abnormality") == "ecg_broad"
    assert _t("Electrocardiogram findings") == "ecg_broad"


def test_non_qt_exclusions():
    assert _t("KLK5 Protease Activity") == "non_qt"
    assert _t("Food cravings score") == "non_qt"
    assert _t("RT-PCR viral load") == "non_qt"
    assert _t("qPCR assay") == "non_qt"


def test_torsade_qt_specific():
    result = classify_outcome_text("Incidence of torsades de pointes")
    assert result["evidence_type"] == "qt_specific"
    assert result["evidence_subtype"] in {"tdp", "tdp_related"}


if __name__ == "__main__":
    test_standalone_qt_not_qt_specific()
    test_qt_with_context_is_qt_specific()
    test_qtc_direct()
    test_qtc_with_pk_not_excluded()
    test_ecg_conduction()
    test_ecg_broad_only()
    test_non_qt_exclusions()
    test_torsade_qt_specific()
    print("all passed")
