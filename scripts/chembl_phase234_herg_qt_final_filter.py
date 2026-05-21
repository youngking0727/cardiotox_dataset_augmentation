#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final ChEMBL Phase II/III/Approved hERG/QT assay filter.

Purpose:
1. Input:
  -all_activity_records_classified.csv

2. Output:
   - human hERG/KCNH2 assay records
   - human/non-animal QT assay records
   - excluded animal QT records
   - excluded false-positive records
   - molecule-level summary
   - phase-level summary

Important definitions:
- hERG assay:
  Human KCNH2 / hERG / ERG / IKr activity record.
  Target CHEMBL240 with target_organism Homo sapiens is retained.
  CHO / HEK293 / COS-7 / mammalian cells are not removed if target is human KCNH2.

- QT assay:
  Clinical human QT/QTc/ECG/TQT records OR in vitro human-cell QT-like records
  such as hiPSC-CM / human cardiomyocyte / MEA / FPD / FPDc.
  Animal QT/APD/ECG records are removed from the final QT count.

- Removed false positives:
  QT-PCR / qPCR / RT-PCR / real-time PCR
  TDP-43 / TAR DNA-binding protein 43
  measured / mean cannot trigger MEA
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


HERG_TARGET_CHEMBL_ID = "CHEMBL240"

PHASE_NAME = {
    2.0: "Phase II",
    3.0: "Phase III",
    4.0: "Approved",
}


def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def norm_bool(x) -> bool:
    if pd.isna(x):
        return False
    return str(x).strip().lower() in {"true", "1", "yes", "y", "t"}


def regex_any(text: str, patterns: List[str], flags=re.IGNORECASE) -> bool:
    return any(re.search(p, text, flags=flags) for p in patterns)


def regex_hits(text: str, labeled_patterns: List[Tuple[str, str]], flags=re.IGNORECASE) -> List[str]:
    hits = []
    for label, pattern in labeled_patterns:
        if re.search(pattern, text, flags=flags):
            hits.append(label)
    return sorted(set(hits))


def make_text(row) -> str:
    cols = [
        "molecule_chembl_id",
        "molecule_pref_name",
        "pref_name",
        "activity_id",
        "assay_chembl_id",
        "target_chembl_id",
        "target_pref_name",
        "target_organism",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "assay_description",
        "activity_comment",
        "data_validity_comment",
        "document_chembl_id",
        "activity_evidence_type",
        "matched_herg_keywords",
        "matched_qt_keywords",
        "matched_generic_non_qt_keywords",
    ]
    return " ".join(norm_text(row.get(c, "")) for c in cols)


def require_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df


HERG_PATTERNS = [
    r"\bKCNH2\b",
    r"\bhERG\b",
    r"\bHERG\b",
    r"\bERG\s+channel\b",
    r"\bIKr\b",
    r"\bKv11\.1\b",
    r"\bK\+\s*channel\s+HERG\b",
]

REAL_QT_PATTERNS = [
    r"\bQTc\b",
    r"\bQT\s+interval\b",
    r"\bQTc\s+interval\b",
    r"\bcorrected\s+QT\b",
    r"\bQT\s+prolongation\b",
    r"\bQTc\s+prolongation\b",
    r"\bprolonged\s+QT\b",
    r"\blong\s+QT\b",
    r"\bTQT\b",
    r"\bthorough\s+QT\b",
    r"\belectrocardiogram\b",
    r"\bECG\b",
    r"\btorsades?\b",
    r"\btorsades?\s+de\s+pointes\b",
    r"(?<![A-Za-z0-9])TdP(?![A-Za-z0-9-])",
    r"\bAPD\b",
    r"\bAPD50\b",
    r"\bAPD90\b",
    r"\bAPD95\b",
    r"\baction\s+potential\s+duration\b",
    r"\bFPD\b",
    r"\bFPDc\b",
    r"\bfield\s+potential\s+duration\b",
    r"\bmicro[- ]?electrode\s+array\b",
    r"\bmulti[- ]electrode\s+array\b",
    r"\bmultielectrode\s+array\b",
    r"\bMEA\s+assay\b",
    r"\bMEA\s+platform\b",
    r"\bMEA\s+recording\b",
    r"\bMEA\s+system\b",
    r"\bMEA[- ]based\b",
]

REPOLARIZATION_PATTERNS = [
    r"\brepolarization\b",
    r"\brepolarisation\b",
]

CARDIAC_CONTEXT_PATTERNS = [
    r"\bcardiac\b",
    r"\bcardiomyocyte",
    r"\bmyocyte",
    r"\bventricular\b",
    r"\bPurkinje\b",
    r"\bQTc?\b",
    r"\bECG\b",
    r"\belectrocardiogram\b",
    r"\bAPD\d*\b",
    r"\bFPDc?\b",
    r"\bfield\s+potential\s+duration\b",
    r"\bMEA\b",
    r"\bmicro[- ]?electrode\s+array\b",
]

QT_FALSE_POSITIVE_PATTERNS = [
    r"\bQT[- ]?PCR\b",
    r"\bqPCR\b",
    r"\bRT[- ]?PCR\b",
    r"\breal[- ]time\s+PCR\b",
    r"\bquantitative\s+PCR\b",
    r"\bTDP[- ]?43\b",
    r"\bTDP43\b",
    r"\bTAR\s+DNA[- ]binding\s+protein\s+43\b",
]

NON_CARDIAC_QT_EXCLUSION_PATTERNS = [
    r"\bbeta[- ]?cell\b",
    r"\bbeta-TC3\b",
    r"\bKATP\b",
    r"\bKir6\.2\b",
    r"\btolbutamide\b",
    r"\bvoltage[- ]gated\s+sodium\b",
    r"\bsodium\s+channel\b",
    r"\bNaV\d",
    r"\bSCN\d",
    r"\bmean\s+graph\s+midpoint\b",
]

ANIMAL_PATTERNS_WITH_LABELS = [
    ("Canis familiaris", r"\bCanis\s+familiaris\b"),
    ("dog", r"\bdog\b"),
    ("canine", r"\bcanine\b"),
    ("beagle", r"\bbeagle\b"),
    ("mongrel dog", r"\bmongrel\s+dog\b"),
    ("guinea pig", r"\bguinea\s+pig\b"),
    ("Cavia porcellus", r"\bCavia\s+porcellus\b"),
    ("cynomolgus monkey", r"\bcynomolgus\s+monkey\b"),
    ("monkey", r"\bmonkey\b"),
    ("macaque", r"\bmacaque\b"),
    ("Macaca", r"\bMacaca\b"),
    ("mouse", r"\bmouse\b"),
    ("mice", r"\bmice\b"),
    ("Mus musculus", r"\bMus\s+musculus\b"),
    ("rat", r"\brat\b"),
    ("rats", r"\brats\b"),
    ("Rattus norvegicus", r"\bRattus\s+norvegicus\b"),
    ("rabbit", r"\brabbit\b"),
    ("Oryctolagus cuniculus", r"\bOryctolagus\s+cuniculus\b"),
    ("pig", r"\bpig\b"),
    ("porcine", r"\bporcine\b"),
    ("swine", r"\bswine\b"),
    ("hamster", r"\bhamster\b"),
    ("zebrafish", r"\bzebrafish\b"),
    ("Danio rerio", r"\bDanio\s+rerio\b"),
    ("Langendorff", r"\bLangendorff\b"),
    ("isolated heart", r"\bisolated\s+heart\b"),
    ("Purkinje fiber", r"\bPurkinje\s+fib(?:er|re)s?\b"),
    ("papillary muscle", r"\bpapillary\s+muscle\b"),
    ("ventricular wedge", r"\bventricular\s+wedge\b"),
    ("anesthetized", r"\banesthetized\b"),
    ("anaesthetized", r"\banaesthetized\b"),
    ("telemetry in dog", r"\btelemetry\s+in\s+dog\b"),
    ("telemetry in monkey", r"\btelemetry\s+in\s+monkey\b"),
]

HUMAN_CLINICAL_PATTERNS_WITH_LABELS = [
    ("healthy volunteer", r"\bhealthy\s+volunteers?\b"),
    ("subject", r"\bsubjects?\b"),
    ("patient", r"\bpatients?\b"),
    ("clinical trial", r"\bclinical\s+trial\b"),
    ("phase 1 clinical trial", r"\bphase[- ]?1\s+clinical\s+trial\b"),
    ("phase 2 clinical trial", r"\bphase[- ]?2\s+clinical\s+trial\b"),
    ("phase-1 clinical trial", r"\bphase[- ]?1\b.*\bclinical\b"),
    ("phase-2 clinical trial", r"\bphase[- ]?2\b.*\bclinical\b"),
    ("human assessed as", r"\bhuman\s+assessed\s+as\b"),
    ("in humans", r"\bin\s+humans\b"),
    ("in human subjects", r"\bin\s+human\s+subjects\b"),
    ("administered to humans", r"\badministered\s+to\s+humans?\b"),
    ("thorough QT", r"\bthorough\s+QT\b"),
    ("TQT", r"\bTQT\b"),
]

HUMAN_CELL_PATTERNS_WITH_LABELS = [
    ("hiPSC-CM", r"\bhiPSC[- ]CMs?\b"),
    ("iPSC cardiomyocyte", r"\biPSC.*cardiomyocyte"),
    ("iCell cardiomyocyte", r"\biCell.*cardiomyocyte"),
    ("human cardiomyocyte", r"\bhuman.*cardiomyocyte"),
    ("human cardiac myocyte", r"\bhuman.*cardiac.*myocyte"),
    ("microelectrode array", r"\bmicro[- ]?electrode\s+array\b"),
    ("multi-electrode array", r"\bmulti[- ]electrode\s+array\b"),
    ("multielectrode array", r"\bmultielectrode\s+array\b"),
    ("MEA assay", r"\bMEA\s+assay\b"),
    ("MEA platform", r"\bMEA\s+platform\b"),
    ("MEA recording", r"\bMEA\s+recording\b"),
    ("MEA system", r"\bMEA\s+system\b"),
    ("MEA-based", r"\bMEA[- ]based\b"),
    ("field potential duration", r"\bfield\s+potential\s+duration\b"),
    ("FPD", r"\bFPD\b"),
    ("FPDc", r"\bFPDc\b"),
]


def is_human_herg_record(row) -> bool:
    target_id = norm_text(row.get("target_chembl_id", ""))
    target_name = norm_text(row.get("target_pref_name", ""))
    target_org = norm_text(row.get("target_organism", ""))
    text = make_text(row)
    activity_type = norm_text(row.get("activity_evidence_type", "")).lower()

    if target_id == HERG_TARGET_CHEMBL_ID:
        if re.search(r"\bHomo\s+sapiens\b", target_org, flags=re.I):
            return True
        if re.search(r"\bhuman\b", text, flags=re.I):
            return True
        if not target_org.strip():
            return True
        return False

    if activity_type == "herg_assay":
        if re.search(r"\bhuman\b|\bHomo\s+sapiens\b", text, flags=re.I):
            return True

    if regex_any(target_name + " " + text, HERG_PATTERNS):
        if re.search(r"\bhuman\b|\bHomo\s+sapiens\b", text, flags=re.I):
            return True

    return False


def has_real_qt_signal(row) -> bool:
    text = make_text(row)

    if regex_any(text, QT_FALSE_POSITIVE_PATTERNS):
        real_non_pcr_qt = regex_any(text, [
            r"\bQTc\b",
            r"\bQT\s+interval\b",
            r"\bQTc\s+interval\b",
            r"\bcorrected\s+QT\b",
            r"\bQT\s+prolongation\b",
            r"\bQTc\s+prolongation\b",
            r"\blong\s+QT\b",
            r"\bTQT\b",
            r"\bthorough\s+QT\b",
            r"\belectrocardiogram\b",
            r"\bECG\b",
            r"\bAPD\d*\b",
            r"\bFPDc?\b",
            r"\bfield\s+potential\s+duration\b",
            r"\baction\s+potential\s+duration\b",
        ])
        if not real_non_pcr_qt:
            return False

    if regex_any(text, NON_CARDIAC_QT_EXCLUSION_PATTERNS):
        real_cardiac_endpoint = regex_any(text, [
            r"\bQTc\b",
            r"\bQT\s+interval\b",
            r"\bECG\b",
            r"\belectrocardiogram\b",
            r"\bAPD\d*\b",
            r"\bFPDc?\b",
            r"\bfield\s+potential\s+duration\b",
            r"\baction\s+potential\s+duration\b",
        ])
        if not real_cardiac_endpoint:
            return False

    if regex_any(text, REAL_QT_PATTERNS):
        return True

    if regex_any(text, REPOLARIZATION_PATTERNS) and regex_any(text, CARDIAC_CONTEXT_PATTERNS):
        return True

    return False


def classify_activity(row) -> str:
    text = make_text(row)

    if is_human_herg_record(row):
        return "herg_human_assay"

    target_id = norm_text(row.get("target_chembl_id", ""))
    activity_type = norm_text(row.get("activity_evidence_type", "")).lower()
    if target_id == HERG_TARGET_CHEMBL_ID or activity_type == "herg_assay":
        return "not_herg_or_qt"

    if not has_real_qt_signal(row):
        if regex_any(text, QT_FALSE_POSITIVE_PATTERNS):
            return "excluded_false_positive"
        return "not_herg_or_qt"

    animal_hits = regex_hits(text, ANIMAL_PATTERNS_WITH_LABELS)
    human_clinical_hits = regex_hits(text, HUMAN_CLINICAL_PATTERNS_WITH_LABELS)
    human_cell_hits = regex_hits(text, HUMAN_CELL_PATTERNS_WITH_LABELS)

    if animal_hits:
        return "qt_nonclinical_animal_assay"

    if human_clinical_hits:
        return "qt_clinical_human_assay"

    if human_cell_hits:
        # hiPSC-CM alone (e.g. cytotoxicity) is not enough; need electrophysiology endpoint.
        strong_cell_qt_endpoint = regex_any(text, [
            r"\bAPD\d*\b",
            r"\bFPDc?\b",
            r"\bfield\s+potential\s+duration\b",
            r"\baction\s+potential\s+duration\b",
            r"\bMEA\s+assay\b",
            r"\bMEA\s+platform\b",
            r"\bMEA\s+recording\b",
            r"\bMEA\s+system\b",
            r"\bMEA[- ]based\b",
            r"\bmicro[- ]?electrode\s+array\b",
            r"\bmulti[- ]electrode\s+array\b",
            r"\bmultielectrode\s+array\b",
        ])
        if strong_cell_qt_endpoint:
            return "qt_invitro_human_cell_assay"

    return "qt_uncertain_nonanimal_assay"


def add_classification_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["final_activity_type"] = df.apply(classify_activity, axis=1)

    animal_hits = []
    human_clinical_hits = []
    human_cell_hits = []
    false_positive_hits = []

    for _, row in df.iterrows():
        text = make_text(row)
        animal_hits.append("; ".join(regex_hits(text, ANIMAL_PATTERNS_WITH_LABELS)))
        human_clinical_hits.append("; ".join(regex_hits(text, HUMAN_CLINICAL_PATTERNS_WITH_LABELS)))
        human_cell_hits.append("; ".join(regex_hits(text, HUMAN_CELL_PATTERNS_WITH_LABELS)))
        false_positive_hits.append("; ".join(
            p for p in ["QT-PCR/qPCR/RT-PCR or TDP-43"]
            if regex_any(text, QT_FALSE_POSITIVE_PATTERNS)
        ))

    df["animal_context_hits_final"] = animal_hits
    df["human_clinical_context_hits_final"] = human_clinical_hits
    df["human_cell_context_hits_final"] = human_cell_hits
    df["false_positive_context_hits_final"] = false_positive_hits

    df["is_final_human_herg_record"] = df["final_activity_type"].eq("herg_human_assay")
    df["is_final_human_qt_record"] = df["final_activity_type"].isin([
        "qt_clinical_human_assay",
        "qt_invitro_human_cell_assay",
    ])
    df["is_final_human_herg_or_qt_record"] = (
        df["is_final_human_herg_record"] | df["is_final_human_qt_record"]
    )

    return df


def build_molecule_summary(records: pd.DataFrame, molecules: pd.DataFrame) -> pd.DataFrame:
    grouped_rows = []

    for mol_id, g in records.groupby("molecule_chembl_id", dropna=False):
        has_herg = g["final_activity_type"].eq("herg_human_assay").any()
        has_clinical_qt = g["final_activity_type"].eq("qt_clinical_human_assay").any()
        has_cell_qt = g["final_activity_type"].eq("qt_invitro_human_cell_assay").any()
        has_human_qt = has_clinical_qt or has_cell_qt
        has_animal_qt = g["final_activity_type"].eq("qt_nonclinical_animal_assay").any()
        has_uncertain_qt = g["final_activity_type"].eq("qt_uncertain_nonanimal_assay").any()

        grouped_rows.append({
            "molecule_chembl_id": mol_id,
            "has_human_herg_assay": has_herg,
            "has_clinical_human_qt_assay": has_clinical_qt,
            "has_invitro_human_cell_qt_assay": has_cell_qt,
            "has_human_qt_assay": has_human_qt,
            "has_nonclinical_animal_qt_assay_removed": has_animal_qt,
            "has_uncertain_nonanimal_qt_assay": has_uncertain_qt,
            "has_human_herg_or_human_qt_assay": has_herg or has_human_qt,
            "has_both_human_herg_and_human_qt_assay": has_herg and has_human_qt,
            "human_herg_record_count": int(g["final_activity_type"].eq("herg_human_assay").sum()),
            "clinical_human_qt_record_count": int(g["final_activity_type"].eq("qt_clinical_human_assay").sum()),
            "invitro_human_cell_qt_record_count": int(g["final_activity_type"].eq("qt_invitro_human_cell_assay").sum()),
            "human_qt_record_count": int(g["final_activity_type"].isin([
                "qt_clinical_human_assay",
                "qt_invitro_human_cell_assay",
            ]).sum()),
            "nonclinical_animal_qt_record_count_removed": int(
                g["final_activity_type"].eq("qt_nonclinical_animal_assay").sum()
            ),
            "uncertain_nonanimal_qt_record_count": int(
                g["final_activity_type"].eq("qt_uncertain_nonanimal_assay").sum()
            ),
        })

    hit_summary = pd.DataFrame(grouped_rows)

    molecules = molecules.copy()
    molecules = require_columns(
        molecules, ["molecule_chembl_id", "pref_name", "molecule_pref_name", "max_phase", "phase_name"]
    )

    if "molecule_pref_name" not in molecules.columns or molecules["molecule_pref_name"].eq("").all():
        molecules["molecule_pref_name"] = molecules["pref_name"]

    molecules["max_phase"] = pd.to_numeric(molecules["max_phase"], errors="coerce")
    molecules["phase_name"] = molecules.apply(
        lambda r: norm_text(r["phase_name"]) if norm_text(r["phase_name"]) else PHASE_NAME.get(r["max_phase"], ""),
        axis=1,
    )

    base = molecules[[
        "molecule_chembl_id",
        "molecule_pref_name",
        "max_phase",
        "phase_name",
    ]].drop_duplicates("molecule_chembl_id")

    out = base.merge(hit_summary, on="molecule_chembl_id", how="left")

    bool_cols = [c for c in out.columns if c.startswith("has_")]
    count_cols = [c for c in out.columns if c.endswith("_count") or c.endswith("_removed")]

    for c in bool_cols:
        out[c] = out[c].fillna(False).astype(bool)

    for c in count_cols:
        if c in out.columns and c not in bool_cols:
            out[c] = out[c].fillna(0).astype(int)

    return out


def build_overall_summary(mol: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("has_human_herg_assay", "Molecules with human hERG/KCNH2 assay"),
        ("has_clinical_human_qt_assay", "Molecules with clinical human QT/QTc/ECG assay"),
        ("has_invitro_human_cell_qt_assay", "Molecules with in vitro human-cell QT/FPD/MEA assay"),
        ("has_human_qt_assay", "Molecules with human/non-animal QT assay"),
        ("has_human_herg_or_human_qt_assay", "Molecules with human hERG or human QT assay"),
        ("has_both_human_herg_and_human_qt_assay", "Molecules with both human hERG and human QT assay"),
        ("has_nonclinical_animal_qt_assay_removed", "Molecules with animal QT assay removed"),
        ("has_uncertain_nonanimal_qt_assay", "Molecules with uncertain non-animal QT assay"),
    ]

    total = len(mol)
    rows = []
    for col, label in metrics:
        count = int(mol[col].fillna(False).astype(bool).sum()) if col in mol.columns else 0
        rows.append({
            "metric": label,
            "count": count,
            "percentage": round(count / total * 100, 4) if total else 0.0,
        })

    return pd.DataFrame(rows)


def build_phase_summary(mol: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("has_human_herg_assay", "molecules_with_human_herg_assay", "percentage_with_human_herg_assay"),
        ("has_clinical_human_qt_assay", "molecules_with_clinical_human_qt_assay", "percentage_with_clinical_human_qt_assay"),
        ("has_invitro_human_cell_qt_assay", "molecules_with_invitro_human_cell_qt_assay", "percentage_with_invitro_human_cell_qt_assay"),
        ("has_human_qt_assay", "molecules_with_human_qt_assay", "percentage_with_human_qt_assay"),
        ("has_human_herg_or_human_qt_assay", "molecules_with_human_herg_or_human_qt_assay", "percentage_with_human_herg_or_human_qt_assay"),
        ("has_both_human_herg_and_human_qt_assay", "molecules_with_both_human_herg_and_human_qt_assay", "percentage_with_both_human_herg_and_human_qt_assay"),
        ("has_nonclinical_animal_qt_assay_removed", "molecules_with_animal_qt_removed", "percentage_with_animal_qt_removed"),
        ("has_uncertain_nonanimal_qt_assay", "molecules_with_uncertain_nonanimal_qt_assay", "percentage_with_uncertain_nonanimal_qt_assay"),
    ]

    rows = []
    for phase_value in [2.0, 3.0, 4.0]:
        sub = mol[mol["max_phase"] == phase_value]
        total = len(sub)

        row = {
            "max_phase": int(phase_value),
            "phase_name": PHASE_NAME[phase_value],
            "total_molecules": total,
        }

        for src, count_col, pct_col in metrics:
            count = int(sub[src].fillna(False).astype(bool).sum()) if src in sub.columns else 0
            row[count_col] = count
            row[pct_col] = round(count / total * 100, 4) if total else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def load_activity_records(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    return require_columns(df, [
        "molecule_chembl_id",
        "molecule_pref_name",
        "target_chembl_id",
        "target_pref_name",
        "target_organism",
        "standard_type",
        "assay_description",
        "activity_evidence_type",
        "matched_herg_keywords",
        "matched_qt_keywords",
        "matched_generic_non_qt_keywords",
    ])


def load_molecules(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    df = require_columns(df, ["molecule_chembl_id", "pref_name", "molecule_pref_name", "max_phase", "phase_name"])

    if df["molecule_pref_name"].eq("").all():
        df["molecule_pref_name"] = df["pref_name"]

    df["max_phase"] = pd.to_numeric(df["max_phase"], errors="coerce")
    df = df[df["max_phase"].isin([2.0, 3.0, 4.0])].copy()

    df["phase_name"] = df.apply(
        lambda r: norm_text(r["phase_name"]) if norm_text(r["phase_name"]) else PHASE_NAME.get(r["max_phase"], ""),
        axis=1,
    )

    return df


def maybe_prefilter_candidate_records(df: pd.DataFrame) -> pd.DataFrame:
    candidate_by_existing_flag = pd.Series(False, index=df.index)
    if "is_herg_or_qt_assay" in df.columns:
        candidate_by_existing_flag = df["is_herg_or_qt_assay"].apply(norm_bool)

    text_series = df.apply(make_text, axis=1)

    candidate_by_pattern = (
        df["target_chembl_id"].eq(HERG_TARGET_CHEMBL_ID)
        | text_series.apply(lambda x: regex_any(x, HERG_PATTERNS))
        | text_series.apply(lambda x: regex_any(x, REAL_QT_PATTERNS))
        | text_series.apply(
            lambda x: regex_any(x, REPOLARIZATION_PATTERNS) and regex_any(x, CARDIAC_CONTEXT_PATTERNS)
        )
    )

    return df[candidate_by_existing_flag | candidate_by_pattern].copy()


def run_validation_tests() -> None:
    cases = [
        {
            "name": "cytotoxicity + hiPSC-CM no strong endpoint",
            "row": {
                "assay_description": "Cytotoxicity measured in hiPSC-CM cells by cell viability assay",
                "target_organism": "Homo sapiens",
            },
            "expected": "not_herg_or_qt",
        },
        {
            "name": "cytotoxicity + APD90",
            "row": {
                "assay_description": "Cytotoxicity and APD90 prolongation measured in hiPSC-CM",
                "target_organism": "Homo sapiens",
            },
            "expected": "qt_invitro_human_cell_assay",
        },
        {
            "name": "TDP-43 is not TdP",
            "row": {
                "target_pref_name": "TAR DNA-binding protein 43",
                "assay_description": "qHTS of TDP-43 inhibitors",
            },
            "expected": "excluded_false_positive",
        },
        {
            "name": "measured is not MEA",
            "row": {
                "assay_description": "Compound measured after 24 hrs by fluorescence assay",
            },
            "expected": "not_herg_or_qt",
        },
        {
            "name": "MEA assay + FPD",
            "row": {
                "assay_description": "MEA assay measured field potential duration in hiPSC-derived cardiomyocytes",
                "target_organism": "Homo sapiens",
            },
            "expected": "qt_invitro_human_cell_assay",
        },
        {
            "name": "animal QT removed",
            "row": {
                "assay_description": "Cardiovascular activity in mongrel dog assessed as change in QTc intervals",
                "target_organism": "Canis familiaris",
            },
            "expected": "qt_nonclinical_animal_assay",
        },
        {
            "name": "human hERG in CHO retained",
            "row": {
                "target_chembl_id": "CHEMBL240",
                "target_pref_name": "Voltage-gated inwardly rectifying potassium channel KCNH2",
                "target_organism": "Homo sapiens",
                "assay_description": "Inhibition of human ERG expressed in CHO cells by whole cell patch clamp technique",
            },
            "expected": "herg_human_assay",
        },
        {
            "name": "QT-PCR rejected",
            "row": {
                "assay_description": "Gene expression measured by QT-PCR",
            },
            "expected": "excluded_false_positive",
        },
        {
            "name": "clinical human QT",
            "row": {
                "target_organism": "Homo sapiens",
                "assay_description": "Cardiotoxicity in healthy human assessed as change in corrected QT interval by ECG",
            },
            "expected": "qt_clinical_human_assay",
        },
    ]

    for case in cases:
        row = pd.Series(case["row"])
        got = classify_activity(row)
        assert got == case["expected"], f"{case['name']} failed: expected {case['expected']}, got {got}"

    print("Validation tests passed.")


def run_pipeline(activity_path: Path, molecules_path: Path, outdir: Path, run_tests: bool = True) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    if run_tests:
        run_validation_tests()

    print("\nLoading input files...")
    activities = load_activity_records(activity_path)
    molecules = load_molecules(molecules_path)

    phase234_ids = set(molecules["molecule_chembl_id"].dropna().astype(str))
    activities = activities[activities["molecule_chembl_id"].astype(str).isin(phase234_ids)].copy()

    print(f"Phase II/III/Approved molecules: {len(molecules)}")
    print(f"Activity records after molecule filter: {len(activities)}")

    candidate = maybe_prefilter_candidate_records(activities)
    print(f"Candidate hERG/QT records before final classification: {len(candidate)}")

    classified = add_classification_columns(candidate)

    all_reclassified = outdir / "phase2_3_4_all_candidate_records_reclassified.csv"
    classified.to_csv(all_reclassified, index=False, encoding="utf-8-sig")

    final_records = classified[classified["final_activity_type"].isin([
        "herg_human_assay",
        "qt_clinical_human_assay",
        "qt_invitro_human_cell_assay",
    ])].copy()
    final_records_path = outdir / "phase2_3_4_final_human_herg_qt_records.csv"
    final_records.to_csv(final_records_path, index=False, encoding="utf-8-sig")

    human_herg_path = outdir / "phase2_3_4_human_herg_records.csv"
    classified[classified["final_activity_type"].eq("herg_human_assay")].to_csv(
        human_herg_path, index=False, encoding="utf-8-sig"
    )

    clinical_qt_path = outdir / "phase2_3_4_clinical_human_qt_records.csv"
    classified[classified["final_activity_type"].eq("qt_clinical_human_assay")].to_csv(
        clinical_qt_path, index=False, encoding="utf-8-sig"
    )

    human_cell_qt_path = outdir / "phase2_3_4_invitro_human_cell_qt_records.csv"
    classified[classified["final_activity_type"].eq("qt_invitro_human_cell_assay")].to_csv(
        human_cell_qt_path, index=False, encoding="utf-8-sig"
    )

    animal_removed_path = outdir / "phase2_3_4_nonclinical_animal_qt_removed_records.csv"
    classified[classified["final_activity_type"].eq("qt_nonclinical_animal_assay")].to_csv(
        animal_removed_path, index=False, encoding="utf-8-sig"
    )

    uncertain_qt_path = outdir / "phase2_3_4_uncertain_nonanimal_qt_records_for_review.csv"
    classified[classified["final_activity_type"].eq("qt_uncertain_nonanimal_assay")].to_csv(
        uncertain_qt_path, index=False, encoding="utf-8-sig"
    )

    false_positive_path = outdir / "phase2_3_4_excluded_false_positive_records.csv"
    classified[classified["final_activity_type"].eq("excluded_false_positive")].to_csv(
        false_positive_path, index=False, encoding="utf-8-sig"
    )

    record_type_counts = (
        classified["final_activity_type"]
        .value_counts(dropna=False)
        .rename_axis("final_activity_type")
        .reset_index(name="record_count")
    )
    record_type_counts_path = outdir / "phase2_3_4_record_type_counts.csv"
    record_type_counts.to_csv(record_type_counts_path, index=False, encoding="utf-8-sig")

    molecule_summary = build_molecule_summary(classified, molecules)
    molecule_summary_path = outdir / "phase2_3_4_final_molecule_level_summary.csv"
    molecule_summary.to_csv(molecule_summary_path, index=False, encoding="utf-8-sig")

    overall_summary = build_overall_summary(molecule_summary)
    overall_summary_path = outdir / "phase2_3_4_final_overall_summary.csv"
    overall_summary.to_csv(overall_summary_path, index=False, encoding="utf-8-sig")

    phase_summary = build_phase_summary(molecule_summary)
    phase_summary_path = outdir / "phase2_3_4_final_phase_level_summary.csv"
    phase_summary.to_csv(phase_summary_path, index=False, encoding="utf-8-sig")

    print("\n========== Record-level counts ==========")
    print(record_type_counts.to_string(index=False))

    print("\n========== Overall molecule-level summary ==========")
    print(overall_summary.to_string(index=False))

    print("\n========== Phase-level summary ==========")
    print(phase_summary.to_string(index=False))

    print("\n========== Key Check ==========")
    print(f"Total Phase II/III/Approved molecules: {len(molecule_summary)}")
    print(f"Molecules with human hERG assay: {int(molecule_summary['has_human_herg_assay'].sum())}")
    print(f"Molecules with human/non-animal QT assay: {int(molecule_summary['has_human_qt_assay'].sum())}")
    print(f"Molecules with both human hERG and human QT assay: {int(molecule_summary['has_both_human_herg_and_human_qt_assay'].sum())}")
    print(f"Molecules with human hERG or human QT assay: {int(molecule_summary['has_human_herg_or_human_qt_assay'].sum())}")
    print(f"Animal QT records removed: {int(classified['final_activity_type'].eq('qt_nonclinical_animal_assay').sum())}")
    print(f"False-positive records removed: {int(classified['final_activity_type'].eq('excluded_false_positive').sum())}")
    print(f"Uncertain non-animal QT records for review: {int(classified['final_activity_type'].eq('qt_uncertain_nonanimal_assay').sum())}")

    print("\n========== Saved files ==========")
    for p in [
        all_reclassified,
        final_records_path,
        human_herg_path,
        clinical_qt_path,
        human_cell_qt_path,
        animal_removed_path,
        uncertain_qt_path,
        false_positive_path,
        record_type_counts_path,
        molecule_summary_path,
        overall_summary_path,
        phase_summary_path,
    ]:
        print(f"- {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activities",
        default="data/output/all_activity_records_classified.csv",
        help="Input classified activity CSV",
    )
    parser.add_argument(
        "--molecules",
        default="data/output/phase2_3_4_molecules.csv",
        help="Phase II/III/Approved molecule CSV",
    )
    parser.add_argument(
        "--outdir",
        default="data/output",
        help="Output directory",
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip built-in validation tests",
    )

    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]

    activity_path = Path(args.activities)
    if not activity_path.is_absolute():
        activity_path = project_root / activity_path

    molecules_path = Path(args.molecules)
    if not molecules_path.is_absolute():
        molecules_path = project_root / molecules_path

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = project_root / outdir

    run_pipeline(
        activity_path=activity_path,
        molecules_path=molecules_path,
        outdir=outdir,
        run_tests=not args.no_tests,
    )


if __name__ == "__main__":
    main()
