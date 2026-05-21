#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 all_activity_records_classified.csv 应用 M3-A / M3-B / M3-C 的过滤逻辑（不修改 m3a/m3b/m3c 源码）。

逻辑来源（仅 import 调用）：
- M3-A: classify_activity_evidence_dict + BioactivityRetriever._parse_activity / supplemental 保留规则
- M3-B: clinical_text_has_direct_qt（与 ClinicalStatusRetriever._compute_direct_qt_clinical_hit 相同文本规则）
- M3-C: classify_literature_text_buckets（与 LiteratureRetriever 文献分桶相同）

输出：
- data/output/M3_filter_result.csv  — 通过任一层 M3 过滤的 activity 记录
- data/output/M3_filter_summary.csv   — 记录级 / 分子级汇总

用法（项目根目录）:
  python scripts/m3_filter_classified_activities.py
  python scripts/m3_filter_classified_activities.py --chunk-size 50000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from m3a_bioactivity_retriever import DEFAULT_TARGETS, BioactivityRetriever  # noqa: E402
from rules.cardiotox_evidence_rules import (  # noqa: E402
    EVIDENCE_PRIORITY_1,
    EVIDENCE_PRIORITY_2,
    EVIDENCE_PRIORITY_3,
    classify_activity_evidence_dict,
    classify_literature_text_buckets,
    clinical_text_has_direct_qt,
)

DEFAULT_INPUT = PROJECT_ROOT / "data/output/all_activity_records_classified.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/output/M3_filter_result.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "data/output/M3_filter_summary.csv"

M3A_TARGET_MAP = {t.chembl_id: t.name for t in DEFAULT_TARGETS}


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
    # 与 retrieve() 一致：有解析测量值，或 supplemental 文本保留
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
    """复用 M3-B 临床 QT 文本规则（activity 行上的 assay/comment 文本）。"""
    texts = [
        str(row.get("assay_description") or ""),
        str(row.get("activity_comment") or ""),
        str(row.get("target_pref_name") or ""),
        str(row.get("standard_type") or ""),
        str(row.get("matched_qt_keywords") or ""),
    ]
    hit = clinical_text_has_direct_qt(*texts)
    return {
        "m3b_clinical_direct_qt_hit": hit,
        "m3b_pass": bool(hit),
    }


def apply_m3c(row: pd.Series) -> Dict[str, Any]:
    """复用 M3-C 文献分桶：将 assay 文本视作 title+abstract。"""
    title = " ".join(
        filter(
            None,
            [
                str(row.get("molecule_pref_name") or ""),
                str(row.get("target_pref_name") or ""),
                str(row.get("standard_type") or ""),
            ],
        )
    )
    abstract = " ".join(
        filter(
            None,
            [
                str(row.get("assay_description") or ""),
                str(row.get("activity_comment") or ""),
                str(row.get("matched_herg_keywords") or ""),
                str(row.get("matched_qt_keywords") or ""),
                str(row.get("matched_generic_non_qt_keywords") or ""),
            ],
        )
    )
    bucket, reasons = classify_literature_text_buckets(title, abstract, mesh_terms=None)
    m3c_pass = bucket != "irrelevant"
    return {
        "m3c_literature_bucket": bucket,
        "m3c_literature_reason_codes": "; ".join(reasons),
        "m3c_pass": bool(m3c_pass),
    }


def process_row(row: pd.Series, retriever: BioactivityRetriever) -> Dict[str, Any]:
    m3a = apply_m3a(row, retriever)
    m3b = apply_m3b(row)
    m3c = apply_m3c(row)

    layers: List[str] = []
    if m3a["m3a_pass"]:
        layers.append("m3a")
    if m3b["m3b_pass"]:
        layers.append("m3b")
    if m3c["m3c_pass"]:
        layers.append("m3c")

    m3_filter_pass = bool(layers)

    out = {**m3a, **m3b, **m3c}
    out["m3_filter_pass"] = m3_filter_pass
    out["m3_filter_pass_layers"] = ";".join(layers)
    return out


def process_chunk(chunk: pd.DataFrame, retriever: BioactivityRetriever) -> pd.DataFrame:
    m3_cols: List[Dict[str, Any]] = []
    for _, row in chunk.iterrows():
        m3_cols.append(process_row(row, retriever))

    m3_df = pd.DataFrame(m3_cols)
    merged = pd.concat([chunk.reset_index(drop=True), m3_df], axis=1)
    return merged[merged["m3_filter_pass"].astype(bool)]


def build_summary(
    total_scanned: int,
    total_kept: int,
    kept_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {"metric": "total_activity_records_scanned", "count": total_scanned},
        {"metric": "total_records_kept_m3_filter", "count": total_kept},
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
            "metric": "records_pass_m3c_only",
            "count": int((kept_df["m3_filter_pass_layers"] == "m3c").sum()) if len(kept_df) else 0,
        },
    ]

    if len(kept_df):
        for col, label in [
            ("m3a_pass", "kept_m3a_pass"),
            ("m3b_pass", "kept_m3b_pass"),
            ("m3c_pass", "kept_m3c_pass"),
        ]:
            rows.append({
                "metric": label,
                "count": int(kept_df[col].fillna(False).astype(bool).sum()),
            })
        for b, label in [
            ("m3a_evidence_bucket", "kept_m3a_bucket"),
            ("m3c_literature_bucket", "kept_m3c_bucket"),
        ]:
            if b in kept_df.columns:
                vc = kept_df[b].value_counts()
                for k, v in vc.items():
                    rows.append({"metric": f"{label}_{k}", "count": int(v)})

        mol_kept = kept_df["molecule_chembl_id"].nunique()
        rows.append({"metric": "unique_molecules_kept", "count": int(mol_kept)})

    return pd.DataFrame(rows)


def run(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    chunk_size: int,
    max_rows: Optional[int],
) -> None:
    retriever = BioactivityRetriever()

    total_scanned = 0
    total_kept = 0
    kept_parts: List[pd.DataFrame] = []
    first_write = True

    reader = pd.read_csv(input_path, dtype=str, encoding="utf-8-sig", chunksize=chunk_size)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"M3-A targets: {M3A_TARGET_MAP}")
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

        filtered = process_chunk(chunk, retriever)
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
            f"Chunk {chunk_idx}: scanned={n} kept={k} "
            f"(cumulative scanned={total_scanned} kept={total_kept})"
        )

    if not kept_parts and output_path.exists():
        output_path.unlink()

    if not kept_parts:
        pd.DataFrame().to_csv(output_path, index=False, encoding="utf-8-sig")
        kept_all = pd.DataFrame()
    else:
        kept_all = pd.concat(kept_parts, ignore_index=True)

    summary_df = build_summary(total_scanned, total_kept, kept_all)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n========== M3 Filter Done ==========")
    print(f"Scanned: {total_scanned}")
    print(f"Kept: {total_kept}")
    if total_scanned:
        print(f"Keep rate: {total_kept / total_scanned * 100:.4f}%")
    print(f"\nSummary:\n{summary_df.to_string(index=False)}")
    print(f"\nSaved: {output_path}")
    print(f"Saved: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply M3-A/B/C filters to classified ChEMBL activities")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary", type=str, default=str(DEFAULT_SUMMARY))
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--max-rows", type=int, default=None, help="Debug: limit rows scanned")
    args = parser.parse_args()

    run(
        input_path=Path(args.input),
        output_path=Path(args.output),
        summary_path=Path(args.summary),
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
