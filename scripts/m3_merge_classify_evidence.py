#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 M3A + M3B 预处理结果，按 A–F 证据类重新分类并输出分文件 + 总表。

输入（默认）:
  data/output/M3/M3A/M3A_filter_records.csv
  data/output/M3/M3B/M3B_qt_phenotype_preprocess.csv

输出（默认 data/output/M3/M3_merged/）:
  M3_merged_all_preprocessed.csv       — 合并总表（含 final_evidence_class）
  M3_class_summary.csv
  M3_class_herg_human_assay.csv          — A
  M3_class_qt_clinical_human_assay_strong.csv — B
  M3_class_qt_clinical_human_ecg_broad.csv    — C
  M3_class_qt_invitro_human_cell_assay.csv    — D
  M3_class_qt_nonclinical_animal_assay.csv    — E
  M3_class_excluded.csv                  — F

用法:
  python scripts/m3_merge_classify_evidence.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_M3A = PROJECT_ROOT / "data/output/M3/M3A/M3A_filter_records.csv"
DEFAULT_M3B = PROJECT_ROOT / "data/output/M3/M3B/M3B_qt_phenotype_preprocess.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "data/output/M3/M3_merged"

HERG_TARGET_CHEMBL_ID = "CHEMBL240"

CLASS_HERG = "herg_human_assay"
CLASS_QT_STRONG = "qt_clinical_human_assay_strong"
CLASS_ECG_BROAD = "qt_clinical_human_ecg_broad"
CLASS_INVITRO_CELL = "qt_invitro_human_cell_assay"
CLASS_ANIMAL = "qt_nonclinical_animal_assay"
CLASS_EXCLUDED = "excluded"
CLASS_UNCLASSIFIED = "unclassified"

# --- A: hERG 机制 ---
HERG_TEXT_PATTERNS = [
    r"\bhERG\b",
    r"\bHERG\b",
    r"\bKCNH2\b",
    r"\bIKr\b",
    r"\bKv11\.1\b",
]

# --- B: 临床 QT 强证据（仅 assay_description）---
QT_STRONG_PATTERNS = [
    r"\bQTc\b",
    r"\bQT\s+interval\b",
    r"\bcorrected\s+QT\b",
    r"\bQT\s+prolongation\b",
    r"\bQTc\s+prolongation\b",
    r"\bprolonged\s+QT\b",
    r"(?<![A-Za-z0-9])QT(?![A-Za-z0-9-])",  # QT 词，非 QT-PCR
]

HUMAN_CLINICAL_PATTERNS = [
    r"\bHomo\s+sapiens\b",
    r"\bhealthy\s+human\b",
    r"\bhealthy\s+volunteers?\b",
    r"\bhuman\b",
    r"\bpatients?\b",
    r"\bclinical\s+trial\b",
    r"\bhuman\s+assessed\s+as\b",
    r"\bin\s+humans\b",
    r"\bin\s+human\s+subjects\b",
]

# --- C: 临床 ECG 广义（assay 有 ECG 无 QT）---
ECG_BROAD_PATTERNS = [
    r"\b12[- ]lead\s+ECG\b",
    r"\bECG\s+profile\b",
    r"\bECG\s+parameters\b",
    r"\bECG\b",
    r"\belectrocardiogram\b",
]

# --- D: 人源细胞电生理 ---
HUMAN_CELL_PATTERNS = [
    r"\bhiPSC[- ]CMs?\b",
    r"\bhuman\s+iPSC[- ]CMs?\b",
    r"\bhuman\s+cardiomyocyte",
    r"\bhuman\s+cardiac\s+myocyte",
    r"\bMEA\b",
    r"\bmicro[- ]?electrode\s+array\b",
    r"\bmulti[- ]electrode\s+array\b",
]

CELL_EP_PATTERNS = [
    r"\bFPDc?\b",
    r"\bAPD90\b",
    r"\bAPD95\b",
    r"\bAPD\d*\b",
    r"\baction\s+potential\s+duration\b",
    r"\bfield\s+potential\s+duration\b",
    r"\brepolarization\b",
    r"\brepolarisation\b",
]

# --- E: 动物 ---
ANIMAL_PATTERNS: List[Tuple[str, str]] = [
    ("dog", r"\bdog\b"),
    ("canine", r"\bcanine\b"),
    ("beagle", r"\bbeagle\b"),
    ("guinea pig", r"\bguinea\s+pig\b"),
    ("rabbit", r"\brabbit\b"),
    ("monkey", r"\bmonkey\b"),
    ("macaque", r"\bmacaque\b"),
    ("mouse", r"\bmouse\b"),
    ("mice", r"\bmice\b"),
    ("rat", r"\brat\b"),
    ("rats", r"\brats\b"),
    ("Purkinje fiber", r"\bPurkinje\s+fib(?:er|re)s?\b"),
    ("Langendorff", r"\bLangendorff\b"),
    ("Mus musculus", r"\bMus\s+musculus\b"),
    ("Rattus norvegicus", r"\bRattus\s+norvegicus\b"),
]

# --- F: 排除 ---
TDP_FALSE_POSITIVE = [
    r"\bTDP[- ]?43\b",
    r"\bTDP43\b",
    r"\bTDP1\b",
    r"\bTAR\s+DNA[- ]binding\s+protein\s+43\b",
]

CYTOTOX_ALONE_MARKERS = [
    r"\bcytotoxicity\b",
    r"\bcell\s+viability\b",
    r"\bviability\s+assay\b",
]

BEATING_ALONE_MARKERS = [
    r"\bcell\s+beating\b",
    r"\bbeating\s+rate\b",
    r"\bbeating\s+frequency\b",
]


def _norm(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _regex_any(text: str, patterns: List[str], flags: int = re.IGNORECASE) -> bool:
    return any(re.search(p, text, flags=flags) for p in patterns)


def _regex_hits(text: str, labeled: List[Tuple[str, str]]) -> List[str]:
    hits = []
    for label, pat in labeled:
        if re.search(pat, text, flags=re.I):
            hits.append(label)
    return sorted(set(hits))


def _assay_text(row: pd.Series) -> str:
    return _norm(row.get("assay_description"))


def _target_assay_text(row: pd.Series) -> str:
    return " ".join(
        [
            _norm(row.get("target_chembl_id")),
            _norm(row.get("target_pref_name")),
            _norm(row.get("target_organism")),
            _assay_text(row),
            _norm(row.get("activity_comment")),
            _norm(row.get("standard_type")),
        ]
    )


def _full_context_text(row: pd.Series) -> str:
    return " ".join(
        [
            _target_assay_text(row),
            _norm(row.get("molecule_pref_name")),
        ]
    )


def _has_qt_in_assay_description(row: pd.Series) -> bool:
    ad = _assay_text(row)
    if not ad:
        return False
    if _regex_any(ad, QT_STRONG_PATTERNS):
        return True
    if re.search(r"\bQTc\b", ad, flags=re.I):
        return True
    return False


def _has_human_clinical_context(text: str) -> bool:
    return _regex_any(text, HUMAN_CLINICAL_PATTERNS)


def _has_strong_ep_endpoint(text: str) -> bool:
    return _regex_any(text, CELL_EP_PATTERNS) or _regex_any(text, QT_STRONG_PATTERNS)


def classify_row(row: pd.Series) -> Tuple[str, str]:
    """返回 (final_evidence_class, class_reason_codes)。"""
    assay = _assay_text(row)
    ctx = _full_context_text(row)
    target_id = _norm(row.get("target_chembl_id")).upper()
    reasons: List[str] = []

    # F: 排除
    if _regex_any(ctx, TDP_FALSE_POSITIVE) and not re.search(
        r"(?<![A-Za-z0-9])TdP(?![A-Za-z0-9-])", ctx, flags=re.I
    ):
        return CLASS_EXCLUDED, "tdp43_not_tdp"

    if _regex_any(ctx, CYTOTOX_ALONE_MARKERS) and not _has_strong_ep_endpoint(ctx):
        return CLASS_EXCLUDED, "cytotoxicity_without_qt_ep"

    if _regex_any(ctx, BEATING_ALONE_MARKERS) and not _has_strong_ep_endpoint(ctx):
        return CLASS_EXCLUDED, "cell_beating_without_ep_endpoint"

    if re.search(r"\bECG\s+analysis\b", assay, flags=re.I) and not _has_qt_in_assay_description(row):
        if _regex_any(assay, ECG_BROAD_PATTERNS) and _has_human_clinical_context(assay):
            reasons.append("ecg_analysis_to_broad_only")

    # E: 动物（不进入 human 主统计）
    animal_hits = _regex_hits(ctx, ANIMAL_PATTERNS)
    if animal_hits:
        return CLASS_ANIMAL, "animal:" + ";".join(animal_hits)

    # A: hERG 机制
    if target_id == HERG_TARGET_CHEMBL_ID:
        return CLASS_HERG, "target_chembl_id=CHEMBL240"
    if _regex_any(_target_assay_text(row), HERG_TEXT_PATTERNS):
        return CLASS_HERG, "herg_text_in_target_or_assay"

    # B: 临床 QT 强（assay_description 含 QT + human）
    if _has_qt_in_assay_description(row) and _has_human_clinical_context(assay):
        return CLASS_QT_STRONG, "qt_terms_and_human_in_assay_description"

    # C: 临床 ECG 广义（有 ECG + human，assay 无 QT）
    if (
        _regex_any(assay, ECG_BROAD_PATTERNS)
        and _has_human_clinical_context(assay)
        and not _has_qt_in_assay_description(row)
    ):
        reason = "ecg_and_human_no_qt_in_assay"
        if reasons:
            reason = reason + ";" + ";".join(reasons)
        return CLASS_ECG_BROAD, reason

    # D: 人源细胞 QT-like
    if _regex_any(ctx, HUMAN_CELL_PATTERNS) and _regex_any(ctx, CELL_EP_PATTERNS):
        return CLASS_INVITRO_CELL, "human_cell_and_ep_endpoint"

    return CLASS_UNCLASSIFIED, "no_rule_matched"


def load_and_merge(m3a_path: Path, m3b_path: Path) -> pd.DataFrame:
    m3a = pd.read_csv(m3a_path, dtype=str, encoding="utf-8-sig").fillna("")
    m3b = pd.read_csv(m3b_path, dtype=str, encoding="utf-8-sig").fillna("")

    m3a["m3_preprocess_source"] = "m3a"
    m3b["m3_preprocess_source"] = "m3b"

    if "activity_id" not in m3a.columns or "activity_id" not in m3b.columns:
        raise ValueError("activity_id column required in M3A/M3B inputs")

    merged = pd.concat([m3a, m3b], ignore_index=True)
    merged["_source_dup"] = merged.duplicated(subset=["activity_id"], keep=False)
    merged = merged.drop_duplicates(subset=["activity_id"], keep="first")

    # 标记来源：若 activity 在两边都有，标 both
    a_ids = set(m3a["activity_id"].astype(str))
    b_ids = set(m3b["activity_id"].astype(str))
    both = a_ids & b_ids

    def _src(aid: str, row_src: str) -> str:
        if aid in both:
            return "m3a+m3b"
        return row_src

    merged["m3_preprocess_source"] = merged.apply(
        lambda r: _src(str(r["activity_id"]), str(r["m3_preprocess_source"])),
        axis=1,
    )
    merged["m3a_in_preprocess"] = merged["activity_id"].astype(str).isin(a_ids)
    merged["m3b_in_preprocess"] = merged["activity_id"].astype(str).isin(b_ids)

    return merged


def run(m3a_path: Path, m3b_path: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"M3A input: {m3a_path}")
    print(f"M3B input: {m3b_path}")
    merged = load_and_merge(m3a_path, m3b_path)
    print(f"Merged unique activities: {len(merged)} (M3A rows={pd.read_csv(m3a_path).shape[0]}, M3B rows={pd.read_csv(m3b_path).shape[0]})")

    classes: List[str] = []
    reasons: List[str] = []
    for _, row in merged.iterrows():
        c, r = classify_row(row)
        classes.append(c)
        reasons.append(r)

    merged["final_evidence_class"] = classes
    merged["class_reason_codes"] = reasons

    master_path = outdir / "M3_merged_all_preprocessed.csv"
    merged.to_csv(master_path, index=False, encoding="utf-8-sig")

    class_files = {
        CLASS_HERG: "M3_class_herg_human_assay.csv",
        CLASS_QT_STRONG: "M3_class_qt_clinical_human_assay_strong.csv",
        CLASS_ECG_BROAD: "M3_class_qt_clinical_human_ecg_broad.csv",
        CLASS_INVITRO_CELL: "M3_class_qt_invitro_human_cell_assay.csv",
        CLASS_ANIMAL: "M3_class_qt_nonclinical_animal_assay.csv",
        CLASS_EXCLUDED: "M3_class_excluded.csv",
    }

    summary_rows: List[Dict[str, Any]] = [
        {"metric": "merged_unique_activities", "count": len(merged)},
        {
            "metric": "unique_molecules",
            "count": int(merged["molecule_chembl_id"].nunique()),
        },
    ]

    for cls, fname in class_files.items():
        sub = merged[merged["final_evidence_class"] == cls]
        path = outdir / fname
        sub.to_csv(path, index=False, encoding="utf-8-sig")
        summary_rows.append({"metric": f"class_{cls}_records", "count": len(sub)})
        summary_rows.append({
            "metric": f"class_{cls}_molecules",
            "count": int(sub["molecule_chembl_id"].nunique()) if len(sub) else 0,
        })
        print(f"  {cls}: {len(sub)} -> {path}")

    uncl = merged[merged["final_evidence_class"] == CLASS_UNCLASSIFIED]
    if len(uncl):
        uncl_path = outdir / "M3_class_unclassified.csv"
        uncl.to_csv(uncl_path, index=False, encoding="utf-8-sig")
        summary_rows.append({"metric": "class_unclassified_records", "count": len(uncl)})
        print(f"  {CLASS_UNCLASSIFIED}: {len(uncl)} -> {uncl_path}")

    human_main = merged[
        merged["final_evidence_class"].isin([
            CLASS_HERG,
            CLASS_QT_STRONG,
            CLASS_ECG_BROAD,
            CLASS_INVITRO_CELL,
        ])
    ]
    human_main_path = outdir / "M3_class_human_main_evidence.csv"
    human_main.to_csv(human_main_path, index=False, encoding="utf-8-sig")
    summary_rows.append({"metric": "human_main_evidence_records", "count": len(human_main)})
    summary_rows.append({
        "metric": "human_main_evidence_molecules",
        "count": int(human_main["molecule_chembl_id"].nunique()) if len(human_main) else 0,
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = outdir / "M3_class_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved master: {master_path}")
    print(f"Saved human main (A+B+C+D): {human_main_path}")
    print(f"Saved summary: {summary_path}")
    print(f"\n{summary_df.to_string(index=False)}")


def phase234_abcd_distribution(
    merged: pd.DataFrame,
    molecules_path: Path,
    outdir: Path,
) -> None:
    """A–D 类在 Phase II/III/Approved 15,138 分子上的分布。"""
    molecules = pd.read_csv(molecules_path, dtype=str, encoding="utf-8-sig").fillna("")
    molecules["max_phase"] = pd.to_numeric(molecules["max_phase"], errors="coerce")
    phase234_ids = set(molecules["molecule_chembl_id"].astype(str))
    n_total = len(phase234_ids)

    mp = merged[merged["molecule_chembl_id"].astype(str).isin(phase234_ids)].copy()
    classes = {
        "A_herg_human_assay": CLASS_HERG,
        "B_qt_clinical_human_assay_strong": CLASS_QT_STRONG,
        "C_qt_clinical_human_ecg_broad": CLASS_ECG_BROAD,
        "D_qt_invitro_human_cell_assay": CLASS_INVITRO_CELL,
    }

    mol_flags = {mid: {k: False for k in classes} for mid in phase234_ids}
    for _, row in mp.iterrows():
        mid = str(row["molecule_chembl_id"])
        cls = str(row["final_evidence_class"])
        for label, c in classes.items():
            if cls == c:
                mol_flags[mid][label] = True

    mol_df = molecules[["molecule_chembl_id", "pref_name", "max_phase", "phase_name"]].copy()
    for label in classes:
        mol_df[f"has_{label}"] = mol_df["molecule_chembl_id"].map(
            lambda x, lb=label: mol_flags.get(str(x), {}).get(lb, False)
        )
    mol_df["has_any_ABCD"] = mol_df[[f"has_{k}" for k in classes]].any(axis=1)

    rows = [{"metric": "phase234_total_molecules", "count": n_total, "percentage": 100.0}]
    for label, cls in classes.items():
        sub = mp[mp["final_evidence_class"] == cls]
        mols = int(sub["molecule_chembl_id"].nunique())
        rows.append({"metric": f"{label}_records", "count": len(sub), "percentage": round(len(sub) / len(mp) * 100, 4) if len(mp) else 0})
        rows.append({"metric": f"{label}_molecules", "count": mols, "percentage": round(mols / n_total * 100, 4)})
    rows.append({
        "metric": "ABCD_union_molecules",
        "count": int(mol_df["has_any_ABCD"].sum()),
        "percentage": round(mol_df["has_any_ABCD"].sum() / n_total * 100, 4),
    })

    overall = pd.DataFrame(rows)
    overall.to_csv(outdir / "M3_ABCD_phase234_overall_summary.csv", index=False, encoding="utf-8-sig")
    mol_df.to_csv(outdir / "M3_ABCD_phase234_molecule_level_flags.csv", index=False, encoding="utf-8-sig")

    phase_rows = []
    for phase in [2.0, 3.0, 4.0]:
        sub_m = molecules[molecules["max_phase"] == phase]
        n = len(sub_m)
        pids = set(sub_m["molecule_chembl_id"].astype(str))
        pname = {2.0: "Phase II", 3.0: "Phase III", 4.0: "Approved"}[phase]
        phase_rows.append({"max_phase": int(phase), "phase_name": pname, "metric": "total_molecules", "count": n, "percentage": 100.0})
        for label, cls in classes.items():
            mc = int((mol_df["molecule_chembl_id"].astype(str).isin(pids) & mol_df[f"has_{label}"]).sum())
            rc = len(mp[(mp["final_evidence_class"] == cls) & (mp["molecule_chembl_id"].astype(str).isin(pids))])
            phase_rows.append({"max_phase": int(phase), "phase_name": pname, "metric": f"{label}_molecules", "count": mc, "percentage": round(mc / n * 100, 4) if n else 0})
            phase_rows.append({"max_phase": int(phase), "phase_name": pname, "metric": f"{label}_records", "count": rc, "percentage": None})
        union_c = int((mol_df["molecule_chembl_id"].astype(str).isin(pids) & mol_df["has_any_ABCD"]).sum())
        phase_rows.append({"max_phase": int(phase), "phase_name": pname, "metric": "ABCD_union_molecules", "count": union_c, "percentage": round(union_c / n * 100, 4) if n else 0})

    pd.DataFrame(phase_rows).to_csv(outdir / "M3_ABCD_phase234_phase_level_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\nPhase234 A–D distribution saved under {outdir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge M3A/M3B preprocess and classify A–F")
    parser.add_argument("--m3a", type=str, default=str(DEFAULT_M3A))
    parser.add_argument("--m3b", type=str, default=str(DEFAULT_M3B))
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    parser.add_argument(
        "--molecules",
        type=str,
        default=str(PROJECT_ROOT / "data/output/phase2_3_4_molecules.csv"),
    )
    parser.add_argument(
        "--phase234-only",
        action="store_true",
        help="Only compute A–D distribution on existing merged master CSV",
    )
    args = parser.parse_args()
    outdir = Path(args.outdir)
    if args.phase234_only:
        merged = pd.read_csv(outdir / "M3_merged_all_preprocessed.csv", dtype=str, encoding="utf-8-sig").fillna("")
        phase234_abcd_distribution(merged, Path(args.molecules), outdir)
        return
    run(Path(args.m3a), Path(args.m3b), outdir)
    merged = pd.read_csv(outdir / "M3_merged_all_preprocessed.csv", dtype=str, encoding="utf-8-sig").fillna("")
    phase234_abcd_distribution(merged, Path(args.molecules), outdir)


if __name__ == "__main__":
    main()
