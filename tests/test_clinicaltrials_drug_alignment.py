"""Unit tests for ClinicalTrials drug–trial alignment matching."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.clinicaltrials_drug_alignment import (  # noqa: E402
    assess_drug_trial_alignment,
    classify_evidence_tier,
    match_terms_in_text,
)


def _minimal_study(
    *,
    intervention_names: list[str],
    arm_labels: list[str] | None = None,
    results_labels: list[str] | None = None,
) -> dict:
    arm_labels = arm_labels or []
    results_labels = results_labels or []
    return {
        "protocolSection": {
            "identificationModule": {"briefTitle": "Test study"},
            "descriptionModule": {},
            "armsInterventionsModule": {
                "interventions": [
                    {"type": "DRUG", "name": name, "description": "", "armGroupLabels": []}
                    for name in intervention_names
                ],
                "armGroups": [
                    {"label": label, "description": "", "type": "EXPERIMENTAL"}
                    for label in arm_labels
                ],
            },
        },
        "resultsSection": {
            "outcomeMeasuresModule": {
                "outcomeMeasures": [
                    {
                        "groups": [
                            {"id": f"OG{i:03d}", "title": title, "description": ""}
                            for i, title in enumerate(results_labels, start=1)
                        ]
                    }
                ]
            }
        },
    }


def test_strong_term_matches_intervention_name_case_insensitive():
    cases = [
        ("Cetirizine", ["Levocetirizine", "Cetirizine"]),
        ("Pentazocine", ["Magnesium sulphate", "Pentazocine"]),
        ("Doxylamine", ["Doxylamine + Pyridoxine"]),
    ]
    for term, interventions in cases:
        assert match_terms_in_text([term.upper()], interventions[0]) or any(
            match_terms_in_text([term.upper()], iv) for iv in interventions
        )


def test_cetirizine_alignment_on_named_intervention():
    raw = _minimal_study(
        intervention_names=["Levocetirizine", "Cetirizine"],
        arm_labels=[
            "Levocetirizine oral solution 5 mg",
            "Cetirizine dry syrup 10 mg",
        ],
    )
    align = assess_drug_trial_alignment(
        {"_strong_match_terms": ["CETIRIZINE"]},
        raw,
        has_ecg_broad_protocol=True,
    )
    assert align["target_drug_in_intervention"] is True
    assert align["target_drug_in_arm_group"] is True
    assert align["drug_match_level"] == "strong"
    assert align["evidence_attribution_level"] == "direct"


def test_pentazocine_alignment_despite_rescue_wording_in_description():
    raw = _minimal_study(
        intervention_names=["Magnesium sulphate", "Pentazocine"],
        arm_labels=["Magnesium sulphate", "Pentazocine"],
        results_labels=["Magnesium Sulphate", "Pentazocine", "Pentazocine Only"],
    )
    raw["protocolSection"]["armsInterventionsModule"]["interventions"][1][
        "description"
    ] = (
        "rescue analgesia as needed with intramuscular pentazocine 30 mg"
    )
    align = assess_drug_trial_alignment(
        {"_strong_match_terms": ["PENTAZOCINE"]},
        raw,
        has_ecg_conduction_protocol=True,
    )
    assert align["target_drug_in_intervention"] is True
    assert align["target_drug_in_arm_group"] is True
    assert align["target_drug_in_results_group"] is True
    assert align["drug_match_level"] == "strong"
    assert align["target_drug_role"] == "main_intervention"


def test_doxylamine_combination_intervention_is_attributable():
    raw = _minimal_study(
        intervention_names=["Doxylamine + Pyridoxine"],
        arm_labels=["Doxylamine + Pyridoxine 10 mg"],
    )
    align = assess_drug_trial_alignment(
        {"_strong_match_terms": ["DOXYLAMINE"]},
        raw,
        has_qt_specific_protocol=True,
    )
    assert align["target_drug_in_intervention"] is True
    assert align["target_drug_in_arm_group"] is True
    assert align["drug_match_level"] in {"strong", "medium"}
    assert align["evidence_attribution_level"] in {"direct", "partial"}

    tier = classify_evidence_tier(
        align,
        qt_result_attribution={"qt_result_for_target_drug": False},
        protocol_qt_hit=True,
        results_qt_hit=False,
        protocol_outcomes={"has_qt_specific": True},
        title_classification={"evidence_type": "other"},
        results_section={"has_qt_results": False},
        qt_outcome_measure=(
            "Safety and Tolerability: 12-lead electrocardiogram (ECG) - "
            "corrected QT interval (QTc)"
        ),
    )
    assert tier == "combination_qt_evidence"


def test_cetirizine_ecg_only_protocol_is_not_primary_qt():
    raw = _minimal_study(
        intervention_names=["Levocetirizine", "Cetirizine"],
        arm_labels=[
            "Levocetirizine oral solution 5 mg",
            "Cetirizine dry syrup 10 mg",
        ],
    )
    align = assess_drug_trial_alignment(
        {"_strong_match_terms": ["CETIRIZINE"]},
        raw,
        has_ecg_broad_protocol=True,
    )
    tier = classify_evidence_tier(
        align,
        qt_result_attribution={"qt_result_for_target_drug": False},
        protocol_qt_hit=True,
        results_qt_hit=False,
        protocol_outcomes={"has_qt_specific": True, "has_ecg_broad": True},
        title_classification={"evidence_type": "other"},
        results_section={"has_qt_results": False},
        qt_outcome_measure="ECG",
    )
    assert tier == "ecg_broad_supportive_evidence"


def test_levodopa_does_not_match_spm962_intervention():
    raw = _minimal_study(intervention_names=["SPM 962"])
    align = assess_drug_trial_alignment(
        {"_strong_match_terms": ["LEVODOPA"]},
        raw,
        has_qt_specific_results=True,
    )
    assert align["target_drug_in_intervention"] is False
    assert align["drug_match_level"] == "false_positive"


def test_remifentanil_does_not_match_remimazolam():
    raw = _minimal_study(intervention_names=["Remimazolam"])
    align = assess_drug_trial_alignment(
        {"_strong_match_terms": ["REMIFENTANIL"]},
        raw,
        has_qt_specific_results=True,
    )
    assert align["target_drug_in_intervention"] is False
    tier = classify_evidence_tier(
        align,
        qt_result_attribution={"qt_result_for_target_drug": False},
        protocol_qt_hit=False,
        results_qt_hit=True,
        results_section={"has_qt_results": True},
    )
    assert tier == "false_positive_mapping"


def test_levosalbutamol_does_not_match_generic_bronchodilator_class():
    raw = _minimal_study(intervention_names=["Bronchodilator Agents"])
    align = assess_drug_trial_alignment(
        {"_strong_match_terms": ["LEVOSALBUTAMOL"]},
        raw,
        has_qt_specific_protocol=True,
    )
    assert align["target_drug_in_intervention"] is False
    assert align["drug_match_level"] == "false_positive"


if __name__ == "__main__":
    test_strong_term_matches_intervention_name_case_insensitive()
    test_cetirizine_alignment_on_named_intervention()
    test_pentazocine_alignment_despite_rescue_wording_in_description()
    test_doxylamine_combination_intervention_is_attributable()
    test_cetirizine_ecg_only_protocol_is_not_primary_qt()
    test_levodopa_does_not_match_spm962_intervention()
    test_remifentanil_does_not_match_remimazolam()
    test_levosalbutamol_does_not_match_generic_bronchodilator_class()
    print("all tests passed")
