#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 all_activity_records_classified.csv 应用 M3-A / M3-B 过滤（不修改 m3a/m3b/m3c 源码，不使用 M3-C）。

逻辑来源（仅 import 调用）：
- M3-A: classify_activity_evidence_dict + BioactivityRetriever._parse_activity / supplemental 保留规则
- M3-B 两阶段：
  1) 预处理：assay_description + activity_comment + standard_type 命中 QT 表型 → M3B_qt_phenotype_preprocess.csv
  2) 严格筛选：在临床/体外/动物等子类型上再要求「临床人类 QT」→ M3B_filter_records.csv

输出（默认 data/output/M3/）:
  M3_filter_result.csv, M3_filter_summary.csv
  M3A/, M3B/ 及 M3B/M3B_qt_phenotype_preprocess.csv（供检查严格筛选是否过严）

用法（项目根目录）:
  python scripts/m3_filter_classified_activities.py
  python scripts/m3_filter_classified_activities.py --extract-m3ab-only
  python scripts/m3_merge_classify_evidence.py   # 合并 M3A/M3B 预处理并按 A–F 分类
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from m3a_bioactivity_retriever import DEFAULT_TARGETS, BioactivityRetriever  # noqa: E402
from rules.cardiotox_evidence_rules import classify_activity_evidence_dict  # noqa: E402

DEFAULT_INPUT = PROJECT_ROOT / "data/output/all_activity_records_classified.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "data/output/M3"
DEFAULT_OUTPUT = DEFAULT_OUTDIR / "M3_filter_result.csv"
DEFAULT_SUMMARY = DEFAULT_OUTDIR / "M3_filter_summary.csv"
DEFAULT_M3AB_OUTPUT = DEFAULT_OUTDIR / "M3_filter_m3a_m3b_deduped.csv"
DEFAULT_M3AB_SUMMARY = DEFAULT_OUTDIR / "M3_filter_m3a_m3b_summary.csv"
DEFAULT_M3A_DIR = DEFAULT_OUTDIR / "M3A"
DEFAULT_M3B_DIR = DEFAULT_OUTDIR / "M3B"
M3A_RECORDS_NAME = "M3A_filter_records.csv"
M3A_SUMMARY_NAME = "M3A_filter_summary.csv"
M3B_RECORDS_NAME = "M3B_filter_records.csv"
M3B_SUMMARY_NAME = "M3B_filter_summary.csv"
M3B_PREPROCESS_NAME = "M3B_qt_phenotype_preprocess.csv"
M3B_PREPROCESS_SUMMARY_NAME = "M3B_preprocess_summary.csv"

HERG_TARGET_CHEMBL_ID = "CHEMBL240"

DEDUPE_SUBSET = [
    "molecule_chembl_id",
    "assay_chembl_id",
    "standard_type",
    "standard_value",
    "standard_units",
    "document_chembl_id",
]

M3A_TARGET_MAP = {t.chembl_id: t.name for t in DEFAULT_TARGETS}

# --- M3-B: 临床人类 QT（仅 activity 描述文本，与 phase234 临床 QT 逻辑一致）---

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

STRONG_CELL_QT_ENDPOINT_PATTERNS = [
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
]

HUMAN_CELL_PATTERNS_WITH_LABELS: List[Tuple[str, str]] = [
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
    ("field potential duration", r"\bfield\s+potential\s+duration\b"),
    ("FPD", r"\bFPD\b"),
    ("FPDc", r"\bFPDc\b"),
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
    r"\bQTc?\b",
    r"\bECG\b",
    r"\belectrocardiogram\b",
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

ANIMAL_PATTERNS_WITH_LABELS: List[Tuple[str, str]] = [
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

# 不用单独 \bsubjects?\b，避免非临床字段误命中
M3B_HUMAN_CLINICAL_PATTERNS_WITH_LABELS: List[Tuple[str, str]] = [
    ("healthy volunteer", r"\bhealthy\s+volunteers?\b"),
    ("healthy human", r"\bhealthy\s+human\b"),
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
    ("TQT study", r"\bTQT\b"),
]

NON_HUMAN_ORGANISM_PATTERNS = [
    r"\bMus\s+musculus\b",
    r"\bRattus\s+norvegicus\b",
    r"\bCanis\s+familiaris\b",
    r"\bCavia\s+porcellus\b",
    r"\bMacaca\b",
    r"\bDanio\s+rerio\b",
    r"\bOryctolagus\s+cuniculus\b",
]


def _norm_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _regex_any(text: str, patterns: List[str], flags: int = re.IGNORECASE) -> bool:
    return any(re.search(p, text, flags=flags) for p in patterns)


def _regex_hits(text: str, labeled: List[Tuple[str, str]], flags: int = re.IGNORECASE) -> List[str]:
    hits = []
    for label, pattern in labeled:
        if re.search(pattern, text, flags=flags):
            hits.append(label)
    return sorted(set(hits))


def _m3b_phenotype_text(row: pd.Series) -> str:
    """M3-B 表型检索：assay 描述 + standard_type（不用 target_pref_name / matched_keywords）。"""
    return " ".join(
        [
            _norm_text(row.get("assay_description")),
            _norm_text(row.get("activity_comment")),
            _norm_text(row.get("standard_type")),
        ]
    )


def _m3b_has_real_qt_signal(text: str) -> bool:
    if _regex_any(text, QT_FALSE_POSITIVE_PATTERNS):
        if not _regex_any(text, REAL_QT_PATTERNS):
            return False

    if _regex_any(text, NON_CARDIAC_QT_EXCLUSION_PATTERNS):
        if not _regex_any(
            text,
            REAL_QT_PATTERNS
            + [r"\bECG\b", r"\belectrocardiogram\b"],
        ):
            return False

    if _regex_any(text, REAL_QT_PATTERNS):
        return True

    if _regex_any(text, REPOLARIZATION_PATTERNS) and _regex_any(text, CARDIAC_CONTEXT_PATTERNS):
        return True

    return False


def _m3b_target_organism_blocks(row: pd.Series) -> bool:
    org = _norm_text(row.get("target_organism"))
    if not org:
        return False
    if _regex_any(org, NON_HUMAN_ORGANISM_PATTERNS):
        return True
    if re.search(r"\bHomo\s+sapiens\b", org, flags=re.I):
        return False
    if re.search(r"\bhuman\b", org, flags=re.I):
        return False
    # 明确非空且非人源物种名
    if org and not re.search(r"\bHomo\s+sapiens\b|\bhuman\b", org, flags=re.I):
        return True
    return False


def _classify_m3b_phenotype_subtype(
    text: str,
    animal_hits: List[str],
    clinical_hits: List[str],
    cell_hits: List[str],
) -> str:
    """预处理子类型（对齐 phase234 B 线分桶，便于对照严格筛选掉了哪些）。"""
    if animal_hits:
        return "qt_animal"
    if clinical_hits:
        return "qt_clinical_human"
    if cell_hits and _regex_any(text, STRONG_CELL_QT_ENDPOINT_PATTERNS):
        return "qt_invitro_human_cell"
    return "qt_uncertain"


def evaluate_m3b_phenotype(row: pd.Series) -> Dict[str, Any]:
    """
    阶段 1：QT 表型候选（assay+standard_type 含真实 QT/APD/FPD/MEA 等表型词）。
    阶段 2：m3b_pass 仅保留 qt_clinical_human 且通过人群/物种检查。
    """
    text = _m3b_phenotype_text(row)
    animal_hits = _regex_hits(text, ANIMAL_PATTERNS_WITH_LABELS)
    clinical_hits = _regex_hits(text, M3B_HUMAN_CLINICAL_PATTERNS_WITH_LABELS)
    cell_hits = _regex_hits(text, HUMAN_CELL_PATTERNS_WITH_LABELS)

    strict_pass = False
    strict_fail = ""
    candidate = False
    subtype = "not_qt"

    if not text.strip():
        strict_fail = "empty_phenotype_text"
    elif _regex_any(text, QT_FALSE_POSITIVE_PATTERNS) and not _m3b_has_real_qt_signal(text):
        strict_fail = "false_positive_qt"
        subtype = "excluded_false_positive"
    elif not _m3b_has_real_qt_signal(text):
        strict_fail = "no_qt_phenotype_signal"
    else:
        candidate = True
        subtype = _classify_m3b_phenotype_subtype(text, animal_hits, clinical_hits, cell_hits)
        if subtype != "qt_clinical_human":
            strict_fail = f"strict_requires_clinical_human_got_{subtype}"
        elif animal_hits:
            strict_fail = "animal_context_in_assay"
        elif _m3b_target_organism_blocks(row):
            strict_fail = "non_human_target_organism"
        elif not clinical_hits:
            strict_fail = "no_human_clinical_context"
        else:
            strict_pass = True

    return {
        "m3b_qt_phenotype_candidate": candidate,
        "m3b_phenotype_subtype": subtype,
        "m3b_human_clinical_hits": "; ".join(clinical_hits),
        "m3b_human_cell_hits": "; ".join(cell_hits),
        "m3b_animal_context_hits": "; ".join(animal_hits),
        "m3b_clinical_direct_qt_hit": strict_pass,
        "m3b_pass": strict_pass,
        "m3b_strict_fail_reason": strict_fail,
    }


def _row_to_activity_dict(row: pd.Series) -> Dict[str, Any]:
    return {k: ("" if pd.isna(v) else v) for k, v in row.items()}


def _target_name_for(row: pd.Series) -> str:
    tid = str(row.get("target_chembl_id") or "").strip().upper()
    if tid in M3A_TARGET_MAP:
        return M3A_TARGET_MAP[tid]
    pref = str(row.get("target_pref_name") or "").strip()
    return pref or tid


def apply_m3a(row: pd.Series, retriever: BioactivityRetriever) -> Dict[str, Any]:
    """复用 M3-A：classify_activity_evidence_dict + _parse_activity + supplemental 规则。"""
    act = _row_to_activity_dict(row)
    target_id = str(act.get("target_chembl_id") or "").strip()
    in_default_targets = target_id in M3A_TARGET_MAP

    empty_m3a = {
        "m3a_in_default_targets": False,
        "m3a_normalized_type": "",
        "m3a_evidence_bucket": "",
        "m3a_direct_qt_context_hit": False,
        "m3a_mechanistic_context_hit": False,
        "m3a_secondary_pharmacology_context_hit": False,
        "m3a_priority_guess": "",
        "m3a_has_parsed_measurement": False,
        "m3a_supplemental_retained": False,
        "m3a_parse_skip_reason": "target_not_in_m3a_default_map",
        "m3a_pass": False,
    }
    if not in_default_targets:
        return empty_m3a

    target_name = _target_name_for(row)
    cls_info = classify_activity_evidence_dict(act, target_name, target_id)
    measurement, skip_reason = retriever._parse_activity(act, cls_info)

    has_measurement = measurement is not None
    supplemental = bool(
        not has_measurement
        and (cls_info.get("priority_guess") or cls_info.get("direct_qt_context_hit"))
    )
    m3a_pass = has_measurement or supplemental

    return {
        "m3a_in_default_targets": True,
        "m3a_normalized_type": cls_info.get("normalized_type"),
        "m3a_evidence_bucket": cls_info.get("evidence_bucket"),
        "m3a_direct_qt_context_hit": bool(cls_info.get("direct_qt_context_hit")),
        "m3a_mechanistic_context_hit": bool(cls_info.get("mechanistic_context_hit")),
        "m3a_secondary_pharmacology_context_hit": bool(
            cls_info.get("secondary_pharmacology_context_hit")
        ),
        "m3a_priority_guess": cls_info.get("priority_guess") or "",
        "m3a_has_parsed_measurement": has_measurement,
        "m3a_supplemental_retained": supplemental,
        "m3a_parse_skip_reason": skip_reason or "",
        "m3a_pass": bool(m3a_pass),
    }


def apply_m3b(row: pd.Series) -> Dict[str, Any]:
    """M3-B：先标 QT 表型候选（预处理），再判严格临床人类 QT。"""
    return evaluate_m3b_phenotype(row)


def process_row(row: pd.Series, retriever: BioactivityRetriever) -> Dict[str, Any]:
    m3a = apply_m3a(row, retriever)
    m3b = apply_m3b(row)

    layers: List[str] = []
    if m3a["m3a_pass"]:
        layers.append("m3a")
    if m3b["m3b_pass"]:
        layers.append("m3b")

    out = {**m3a, **m3b}
    out["m3_filter_pass"] = bool(layers)
    out["m3_filter_pass_layers"] = ";".join(layers)
    return out


def annotate_chunk(chunk: pd.DataFrame, retriever: BioactivityRetriever) -> pd.DataFrame:
    m3_cols: List[Dict[str, Any]] = []
    for _, row in chunk.iterrows():
        m3_cols.append(process_row(row, retriever))

    m3_df = pd.DataFrame(m3_cols)
    return pd.concat([chunk.reset_index(drop=True), m3_df], axis=1)


def process_chunk(chunk: pd.DataFrame, retriever: BioactivityRetriever) -> pd.DataFrame:
    merged = annotate_chunk(chunk, retriever)
    return merged[merged["m3_filter_pass"].astype(bool)]


def build_summary(
    total_scanned: int,
    total_kept: int,
    kept_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {"metric": "total_activity_records_scanned", "count": total_scanned},
        {"metric": "total_records_kept_m3_ab_filter", "count": total_kept},
        {
            "metric": "records_kept_pct",
            "count": round(total_kept / total_scanned * 100, 4) if total_scanned else 0.0,
        },
        {
            "metric": "records_pass_m3a_only",
            "count": int((kept_df["m3_filter_pass_layers"] == "m3a").sum()) if len(kept_df) else 0,
        },
        {
            "metric": "records_pass_m3b_only",
            "count": int((kept_df["m3_filter_pass_layers"] == "m3b").sum()) if len(kept_df) else 0,
        },
        {
            "metric": "records_pass_m3a_and_m3b",
            "count": int((kept_df["m3_filter_pass_layers"] == "m3a;m3b").sum()) if len(kept_df) else 0,
        },
    ]

    if len(kept_df):
        for col, label in [
            ("m3a_pass", "kept_m3a_pass"),
            ("m3b_pass", "kept_m3b_pass"),
        ]:
            rows.append({
                "metric": label,
                "count": int(kept_df[col].fillna(False).astype(bool).sum()),
            })
        if "m3a_evidence_bucket" in kept_df.columns:
            vc = kept_df["m3a_evidence_bucket"].value_counts()
            for k, v in vc.items():
                rows.append({"metric": f"kept_m3a_bucket_{k}", "count": int(v)})

        mol_kept = kept_df["molecule_chembl_id"].nunique()
        rows.append({"metric": "unique_molecules_kept", "count": int(mol_kept)})

    return pd.DataFrame(rows)


def build_m3b_preprocess_summary(
    total_scanned: int,
    preprocess_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = [
        {"metric": "total_activity_records_scanned", "count": total_scanned},
        {"metric": "m3b_qt_phenotype_candidates", "count": len(preprocess_df)},
        {
            "metric": "m3b_preprocess_pct_of_scanned",
            "count": round(len(preprocess_df) / total_scanned * 100, 4) if total_scanned else 0.0,
        },
        {
            "metric": "m3b_strict_pass_among_candidates",
            "count": int(_as_bool_series(preprocess_df["m3b_pass"]).sum()) if len(preprocess_df) else 0,
        },
        {
            "metric": "m3b_strict_pass_pct_of_candidates",
            "count": round(
                _as_bool_series(preprocess_df["m3b_pass"]).sum() / len(preprocess_df) * 100, 4
            )
            if len(preprocess_df)
            else 0.0,
        },
        {
            "metric": "m3b_dropped_by_strict_filter",
            "count": int((~_as_bool_series(preprocess_df["m3b_pass"])).sum()) if len(preprocess_df) else 0,
        },
    ]
    if len(preprocess_df):
        for col in ["m3b_phenotype_subtype", "m3b_strict_fail_reason"]:
            if col in preprocess_df.columns:
                for k, v in preprocess_df[col].value_counts().items():
                    if k:
                        rows.append({"metric": f"{col}_{k}", "count": int(v)})
        rows.append({
            "metric": "unique_molecules_in_preprocess",
            "count": int(preprocess_df["molecule_chembl_id"].nunique()),
        })
    return pd.DataFrame(rows)


def run(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    chunk_size: int,
    max_rows: Optional[int],
) -> None:
    retriever = BioactivityRetriever()
    outdir = output_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    m3b_dir = outdir / "M3B"
    m3b_dir.mkdir(parents=True, exist_ok=True)
    preprocess_path = m3b_dir / M3B_PREPROCESS_NAME
    preprocess_summary_path = m3b_dir / M3B_PREPROCESS_SUMMARY_NAME

    total_scanned = 0
    total_kept = 0
    total_preprocess = 0
    kept_parts: List[pd.DataFrame] = []
    preprocess_parts: List[pd.DataFrame] = []
    first_write = True
    first_preprocess_write = True

    reader = pd.read_csv(input_path, dtype=str, encoding="utf-8-sig", chunksize=chunk_size)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"M3-B preprocess: {preprocess_path}")
    print(f"M3-A targets: {M3A_TARGET_MAP}")
    print("M3-B: QT phenotype preprocess -> strict clinical human QT (no M3-C)")
    print(f"Chunk size: {chunk_size}")

    for chunk_idx, chunk in enumerate(reader, start=1):
        if max_rows is not None and total_scanned >= max_rows:
            break
        if max_rows is not None:
            remaining = max_rows - total_scanned
            chunk = chunk.head(remaining)

        chunk = chunk.fillna("")
        n = len(chunk)
        total_scanned += n

        annotated = annotate_chunk(chunk, retriever)
        pre_mask = _as_bool_series(annotated["m3b_qt_phenotype_candidate"])
        pre_chunk = annotated[pre_mask]
        p = len(pre_chunk)
        total_preprocess += p

        if p > 0:
            preprocess_parts.append(pre_chunk)
            pre_chunk.to_csv(
                preprocess_path,
                mode="w" if first_preprocess_write else "a",
                header=first_preprocess_write,
                index=False,
                encoding="utf-8-sig",
            )
            first_preprocess_write = False

        filtered = annotated[annotated["m3_filter_pass"].astype(bool)]
        k = len(filtered)
        total_kept += k

        if k > 0:
            kept_parts.append(filtered)
            filtered.to_csv(
                output_path,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
                encoding="utf-8-sig",
            )
            first_write = False

        print(
            f"Chunk {chunk_idx}: scanned={n} preprocess={p} kept={k} "
            f"(cumulative scanned={total_scanned} preprocess={total_preprocess} kept={total_kept})"
        )

    if not kept_parts and output_path.exists():
        output_path.unlink()
    if not preprocess_parts and preprocess_path.exists():
        preprocess_path.unlink()

    if not preprocess_parts:
        pd.DataFrame().to_csv(preprocess_path, index=False, encoding="utf-8-sig")
        preprocess_all = pd.DataFrame()
    else:
        preprocess_all = pd.concat(preprocess_parts, ignore_index=True)

    if not kept_parts:
        pd.DataFrame().to_csv(output_path, index=False, encoding="utf-8-sig")
        kept_all = pd.DataFrame()
    else:
        kept_all = pd.concat(kept_parts, ignore_index=True)

    summary_df = build_summary(total_scanned, total_kept, kept_all)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    preprocess_summary_df = build_m3b_preprocess_summary(total_scanned, preprocess_all)
    preprocess_summary_df.to_csv(preprocess_summary_path, index=False, encoding="utf-8-sig")

    print("\n========== M3 A/B Filter Done ==========")
    print(f"Scanned: {total_scanned}")
    print(f"M3-B QT phenotype candidates (preprocess): {total_preprocess}")
    print(f"Kept (M3-A or M3-B strict): {total_kept}")
    if total_scanned:
        print(f"Preprocess rate: {total_preprocess / total_scanned * 100:.4f}%")
        print(f"Keep rate: {total_kept / total_scanned * 100:.4f}%")
    print(f"\nSummary:\n{summary_df.to_string(index=False)}")
    print(f"\nM3-B preprocess summary:\n{preprocess_summary_df.to_string(index=False)}")
    print(f"\nSaved: {preprocess_path}")
    print(f"Saved: {preprocess_summary_path}")
    print(f"Saved: {output_path}")
    print(f"Saved: {summary_path}")

    if len(kept_all):
        export_split_by_layer(kept_all, outdir)
    if len(preprocess_all):
        strict_m3b = preprocess_all[_as_bool_series(preprocess_all["m3b_pass"])]
        strict_m3b.to_csv(m3b_dir / M3B_RECORDS_NAME, index=False, encoding="utf-8-sig")
        build_layer_summary(strict_m3b, "m3b").to_csv(
            m3b_dir / M3B_SUMMARY_NAME, index=False, encoding="utf-8-sig"
        )
        print(f"Updated strict M3-B: {m3b_dir / M3B_RECORDS_NAME} ({len(strict_m3b)} rows)")


def _as_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def build_layer_summary(df: pd.DataFrame, layer: str) -> pd.DataFrame:
    """单层的记录/分子汇总（用于 M3A、M3B 子目录）。"""
    rows: List[Dict[str, Any]] = [
        {"metric": f"total_{layer}_records", "count": len(df)},
        {"metric": f"unique_molecules_{layer}", "count": int(df["molecule_chembl_id"].nunique()) if len(df) else 0},
    ]
    if layer == "m3a" and len(df) and "m3a_evidence_bucket" in df.columns:
        for k, v in df["m3a_evidence_bucket"].value_counts().items():
            rows.append({"metric": f"m3a_bucket_{k}", "count": int(v)})
        rows.append({
            "metric": "m3a_with_parsed_measurement",
            "count": int(_as_bool_series(df["m3a_has_parsed_measurement"]).sum()),
        })
        rows.append({
            "metric": "m3a_supplemental_retained",
            "count": int(_as_bool_series(df["m3a_supplemental_retained"]).sum()),
        })
    if layer == "m3b" and len(df):
        for col, prefix in [
            ("m3b_strict_fail_reason", "m3b_strict_fail"),
            ("m3b_phenotype_subtype", "m3b_subtype"),
        ]:
            if col in df.columns:
                for k, v in df[col].value_counts().items():
                    if k:
                        rows.append({"metric": f"{prefix}_{k}", "count": int(v)})
        if "m3b_human_clinical_hits" in df.columns:
            top_hits = df["m3b_human_clinical_hits"].value_counts().head(10)
            for k, v in top_hits.items():
                if k:
                    rows.append({"metric": f"m3b_hit_pattern_{k}", "count": int(v)})
    return pd.DataFrame(rows)


def export_split_by_layer(df: pd.DataFrame, outdir: Path) -> None:
    """将合并结果拆到 M3A/、M3B/ 子目录，便于分别检查。"""
    m3a_dir = outdir / "M3A"
    m3b_dir = outdir / "M3B"
    m3a_dir.mkdir(parents=True, exist_ok=True)
    m3b_dir.mkdir(parents=True, exist_ok=True)

    m3a_hit = _as_bool_series(df["m3a_pass"]) if "m3a_pass" in df.columns else pd.Series(False, index=df.index)
    m3b_hit = _as_bool_series(df["m3b_pass"]) if "m3b_pass" in df.columns else pd.Series(False, index=df.index)

    m3a_df = df[m3a_hit].copy()
    m3b_df = df[m3b_hit].copy()

    m3a_records_path = m3a_dir / M3A_RECORDS_NAME
    m3a_summary_path = m3a_dir / M3A_SUMMARY_NAME
    m3b_records_path = m3b_dir / M3B_RECORDS_NAME
    m3b_summary_path = m3b_dir / M3B_SUMMARY_NAME

    m3a_df.to_csv(m3a_records_path, index=False, encoding="utf-8-sig")
    build_layer_summary(m3a_df, "m3a").to_csv(m3a_summary_path, index=False, encoding="utf-8-sig")

    m3b_df.to_csv(m3b_records_path, index=False, encoding="utf-8-sig")
    build_layer_summary(m3b_df, "m3b").to_csv(m3b_summary_path, index=False, encoding="utf-8-sig")

    print("\n========== Split by layer ==========")
    print(f"M3-A: {len(m3a_df)} records -> {m3a_records_path}")
    print(f"M3-A summary -> {m3a_summary_path}")
    print(f"M3-B: {len(m3b_df)} records -> {m3b_records_path}")
    print(f"M3-B summary -> {m3b_summary_path}")


def extract_m3a_m3b_deduped(
    filter_result_path: Path,
    output_path: Path,
    summary_path: Path,
) -> pd.DataFrame:
    """从 M3_filter_result.csv 取出 M3-A / M3-B 通过记录并去重。"""
    df = pd.read_csv(filter_result_path, dtype=str, encoding="utf-8-sig").fillna("")

    m3a_hit = _as_bool_series(df["m3a_pass"]) if "m3a_pass" in df.columns else pd.Series(False, index=df.index)
    m3b_hit = _as_bool_series(df["m3b_pass"]) if "m3b_pass" in df.columns else pd.Series(False, index=df.index)
    sub = df[m3a_hit | m3b_hit].copy()

    before = len(sub)
    dup_activity = int(sub["activity_id"].duplicated().sum()) if "activity_id" in sub.columns and before else 0

    if "activity_id" in sub.columns:
        sub = sub.drop_duplicates(subset=["activity_id"], keep="first")

    dedupe_cols = [c for c in DEDUPE_SUBSET if c in sub.columns]
    before_assay_dedupe = len(sub)
    if dedupe_cols:
        sub = sub.drop_duplicates(subset=dedupe_cols, keep="first")

    def _layers(row: pd.Series) -> str:
        parts = []
        if _as_bool_series(pd.Series([row.get("m3a_pass", "")])).iloc[0]:
            parts.append("m3a")
        if _as_bool_series(pd.Series([row.get("m3b_pass", "")])).iloc[0]:
            parts.append("m3b")
        return ";".join(parts)

    sub["m3ab_source_layers"] = sub.apply(_layers, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary_rows = [
        {"metric": "rows_in_M3_filter_result", "count": len(df)},
        {"metric": "rows_m3a_or_m3b_before_dedupe", "count": before},
        {"metric": "duplicate_activity_id_before_dedupe", "count": dup_activity},
        {"metric": "rows_after_activity_id_dedupe", "count": before_assay_dedupe},
        {"metric": "rows_after_assay_key_dedupe", "count": len(sub)},
        {"metric": "rows_m3a_only", "count": int((sub["m3ab_source_layers"] == "m3a").sum())},
        {"metric": "rows_m3b_only", "count": int((sub["m3ab_source_layers"] == "m3b").sum())},
        {
            "metric": "rows_m3a_and_m3b",
            "count": int(
                (sub["m3ab_source_layers"].str.contains("m3a") & sub["m3ab_source_layers"].str.contains("m3b")).sum()
            )
            if len(sub)
            else 0,
        },
        {"metric": "unique_molecules", "count": int(sub["molecule_chembl_id"].nunique()) if len(sub) else 0},
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n========== M3-A / M3-B Extract & Dedupe ==========")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {output_path}")
    print(f"Saved: {summary_path}")

    export_split_by_layer(sub, output_path.parent)
    return sub


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply M3-A (targets) and M3-B (clinical human QT on assay text) filters"
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", type=str, default=str(DEFAULT_SUMMARY))
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--max-rows", type=int, default=None, help="Debug: limit rows scanned")
    parser.add_argument(
        "--extract-m3ab-only",
        action="store_true",
        help="Only extract M3-A/M3-B rows from existing M3_filter_result.csv and dedupe",
    )
    parser.add_argument("--m3ab-output", type=str, default=str(DEFAULT_M3AB_OUTPUT))
    parser.add_argument("--m3ab-summary", type=str, default=str(DEFAULT_M3AB_SUMMARY))
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Split existing --output CSV into data/output/M3/M3A and M3B folders",
    )
    args = parser.parse_args()

    if args.split_only:
        df = pd.read_csv(Path(args.output), dtype=str, encoding="utf-8-sig").fillna("")
        export_split_by_layer(df, Path(args.output).parent)
        return

    if args.extract_m3ab_only:
        extract_m3a_m3b_deduped(
            filter_result_path=Path(args.output),
            output_path=Path(args.m3ab_output),
            summary_path=Path(args.m3ab_summary),
        )
        return

    run(
        input_path=Path(args.input),
        output_path=Path(args.output),
        summary_path=Path(args.summary),
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
