#!/usr/bin/env python3
"""批量测试 ClinicalTrialsClient：对 Chembl_phase234_Data.csv 中每个 pref_name 查 QT 相关试验。

用法（项目根目录）:
  python scripts/test_clinicaltrials_phase234.py
  python scripts/test_clinicaltrials_phase234.py --limit 50
  python scripts/test_clinicaltrials_phase234.py --rate-limit 0.3

输出:
  data/output/phase234_clinicaltrials_scan.csv              # 分子级汇总
  data/output/phase234_clinicaltrials_trials_detail.csv     # 逐条试验明细（含可信度 + results 摘要列）
  data/output/phase234_clinicaltrials_raw_study.jsonl       # get_study_by_nct_id 原始回传
  data/output/phase234_clinicaltrials_evidence.json         # 三层嵌套 JSON（分子 → 试验 → resultsSection）
  data/output/phase234_clinicaltrials_scan_summary.json     # 统计汇总（含 sponsor / results 分布）
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.clinicaltrials_client import ClinicalTrialsClient
from utils.clinicaltrials_drug_alignment import build_drug_name_set
from utils.chembl_drug_name_enrichment import ChemblDrugNameEnricher
from utils.chembl_client import ChEMBLClient

DEFAULT_INPUT          = _PROJECT_ROOT / "data/Chembl_phase234_Data.csv"
DEFAULT_OUTPUT         = _PROJECT_ROOT / "data/output/phase234_clinicaltrials_scan.csv"
DEFAULT_TRIALS_OUTPUT  = _PROJECT_ROOT / "data/output/phase234_clinicaltrials_trials_detail.csv"
DEFAULT_RAW_JSONL      = _PROJECT_ROOT / "data/output/phase234_clinicaltrials_raw_study.jsonl"
DEFAULT_EVIDENCE_JSON  = _PROJECT_ROOT / "data/output/phase234_clinicaltrials_evidence.json"
DEFAULT_SUMMARY        = _PROJECT_ROOT / "data/output/phase234_clinicaltrials_scan_summary.json"
DEFAULT_CHEMBL_NAME_CACHE = _PROJECT_ROOT / "data/cache/chembl_drug_name_set"


def _load_molecules(csv_path: Path, limit: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["pref_name"].astype(str).str.strip().str.len() > 0].copy()
    if limit > 0:
        df = df.head(limit)
    return df


def _flatten_credibility(cred: Dict[str, Any]) -> Dict[str, Any]:
    """把 research_institution_credibility dict 平铺为 CSV 列。"""
    return {
        "cred_level": cred.get("level", ""),
        "cred_sponsor_name": cred.get("sponsor_name", ""),
        "cred_sponsor_class": cred.get("sponsor_class", ""),
        "cred_responsible_party": cred.get("responsible_party", ""),
        "cred_collaborators": "|".join(cred.get("collaborators") or []),
        "cred_reason_codes": "|".join(cred.get("reason_codes") or []),
        "cred_summary": cred.get("summary", ""),
    }


def _flatten_results_section(rs: Dict[str, Any]) -> Dict[str, Any]:
    """
    把 clinical_results_section 的标量字段平铺为 CSV 列。
    qt_result_measures 等嵌套结构保留在 evidence JSON 中。
    """
    return {
        "results_has_posted_results":       rs.get("has_posted_results", False),
        "results_has_results_section":      rs.get("has_results_section", False),
        "results_has_outcome_measures":     rs.get("has_results_outcome_measures", False),
        "results_has_qt_results":           rs.get("has_qt_results", False),
        "results_has_ecg_conduction_results": rs.get("has_ecg_conduction_results", False),
        "results_has_ecg_broad_results":    rs.get("has_ecg_broad_results", False),
        "results_has_cardiac_ae_results":   rs.get("has_cardiac_ae_results", False),
        "results_qt_measure_count":         len(rs.get("qt_result_measures") or []),
        "results_ecg_conduction_measure_count": len(rs.get("ecg_conduction_result_measures") or []),
        "results_ecg_broad_measure_count":  len(rs.get("ecg_broad_result_measures") or []),
        "results_cardiac_ae_measure_count": len(rs.get("cardiac_ae_result_measures") or []),
        "results_has_adverse_events":       rs.get("has_adverse_events", False),
        "results_has_more_info":            rs.get("has_more_info", False),
        "results_summary":                  rs.get("result_summary", ""),
    }


def _flatten_alignment(align: Dict[str, Any], qt_attr: Dict[str, Any], evidence_tier: str) -> Dict[str, Any]:
    """平铺 drug_trial_alignment / qt_result_attribution 标量字段到 CSV。"""
    return {
        "align_target_in_intervention": align.get("target_drug_in_intervention", False),
        "align_target_in_arm_group":    align.get("target_drug_in_arm_group", False),
        "align_target_in_results_group": align.get("target_drug_in_results_group", False),
        "align_target_drug_role":       align.get("target_drug_role", ""),
        "align_drug_match_level":       align.get("drug_match_level", ""),
        "align_evidence_attribution":   align.get("evidence_attribution_level", ""),
        "align_matched_terms":          "|".join(align.get("matched_drug_terms") or []),
        "align_intervention_names":     "|".join(align.get("intervention_names") or []),
        "align_summary":                align.get("alignment_summary", ""),
        "qt_attr_for_target_drug":      qt_attr.get("qt_result_for_target_drug", False),
        "qt_attr_comparator_only":      qt_attr.get("qt_result_for_comparator_only", False),
        "qt_attr_target_group_ids":     "|".join(qt_attr.get("target_group_ids") or []),
        "qt_attr_summary":              qt_attr.get("summary", ""),
        "evidence_tier":                evidence_tier,
    }


def _build_trial_entry(enriched: Dict[str, Any]) -> Dict[str, Any]:
    """Level-2 trial 对象，内嵌 alignment / results / attribution。"""
    cred = enriched.get("research_institution_credibility") or {}
    return {
        "nct_id":             enriched.get("nct_id", ""),
        "status":             enriched.get("status", ""),
        "title":              enriched.get("title", ""),
        "summary":            enriched.get("summary", ""),
        "sponsor_name":       enriched.get("sponsor_name", ""),
        "sponsor_class":      enriched.get("sponsor_class", ""),
        "institution_credibility_level": enriched.get("institution_credibility_level", "unknown"),
        "qt_related":         enriched.get("qt_related"),
        "qt_related_title":   enriched.get("qt_related_title"),
        "qt_related_outcome": enriched.get("qt_related_outcome"),
        "qt_outcome_measure": enriched.get("qt_outcome_measure", ""),
        "search_branch":      enriched.get("search_branch", ""),
        "protocol_qt_hit":    enriched.get("protocol_qt_hit", False),
        "results_qt_hit":     enriched.get("results_qt_hit", False),
        "research_institution_credibility": cred,
        "drug_trial_alignment": enriched.get("drug_trial_alignment") or {},
        "clinical_results_section": enriched.get("clinical_results_section") or {},
        "qt_result_attribution": enriched.get("qt_result_attribution") or {},
        "evidence_tier": enriched.get("evidence_tier", "manual_review_required"),
    }


def _scan_one(
    client: ClinicalTrialsClient,
    enricher: ChemblDrugNameEnricher,
    row: pd.Series,
    max_results: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
        molecule_entry : Level-1 分子对象（含 trials 列表）
        trial_rows     : 逐条试验明细（含可信度 + results 摘要平铺列），写 CSV
        raw_studies    : get_study_by_nct_id 原始回传，写 raw_study JSONL
    """
    chembl_id = str(row.get("molecule_chembl_id", "")).strip()
    drug_name = str(row.get("pref_name", "")).strip()
    molecule_entry: Dict[str, Any] = {
        "molecule_chembl_id": chembl_id,
        "pref_name": drug_name,
        "max_phase": row.get("max_phase"),
        "phase_name": row.get("phase_name"),
        "drug_name_set": {},
        "recall_audit": {
            "related_match_terms": [],
            "weak_terms": [],
            "recall_audit_terms": [],
        },
        "qt_trials_count": 0,
        "qt_title_hits": 0,
        "qt_outcome_hits": 0,
        "qt_results_hits": 0,
        "recall_branch_hits": 0,
        "attributable_qt_hits": 0,
        "direct_evidence_hits": 0,
        "false_positive_hits": 0,
        "example_nct_ids": [],
        "status": "ok",
        "error": "",
        "trials": [],
    }
    trial_rows:  List[Dict[str, Any]] = []
    raw_studies: List[Dict[str, Any]] = []

    if not drug_name:
        molecule_entry["status"] = "skipped"
        molecule_entry["error"] = "empty pref_name"
        return molecule_entry, trial_rows, raw_studies

    # ── ChEMBL synonym enrichment ────────────────────────────────────────────
    # enrich() 对 chembl_id 调用 ChEMBL molecule API，提取 pref_name、
    # molecule_synonyms、molecule_hierarchy 及 parent molecule，
    # 并将名称分为 strong/related/weak 三层：
    #   strong_match_terms  → query.intr 检索 + direct evidence
    #   related_match_terms → recall_audit only，不进入 direct evidence
    #   weak_terms          → recall_audit only
    # API 失败时自动回退到 pref_name 单名 strong set。
    chembl_enriched = enricher.enrich(chembl_id, pref_name_fallback=drug_name)
    drug_name_set = build_drug_name_set(drug_name, enriched=chembl_enriched)

    molecule_entry["drug_name_set"] = {
        "molecule_chembl_id":        chembl_enriched.get("molecule_chembl_id", chembl_id),
        "pref_name":                 chembl_enriched.get("pref_name", drug_name),
        "chembl_pref_name":          chembl_enriched.get("chembl_pref_name", ""),
        "strong_match_terms":        chembl_enriched.get("strong_match_terms", []),
        "related_match_terms":       chembl_enriched.get("related_match_terms", []),
        "weak_terms":                chembl_enriched.get("weak_terms", []),
        "parent_molecule_chembl_id": chembl_enriched.get("parent_molecule_chembl_id", ""),
        "parent_pref_name":          chembl_enriched.get("parent_pref_name", ""),
        "enrichment_status":         chembl_enriched.get("enrichment_status", "ok"),
    }
    molecule_entry["recall_audit"] = {
        "related_match_terms": chembl_enriched.get("related_match_terms", []),
        "weak_terms":          chembl_enriched.get("weak_terms", []),
        "recall_audit_terms":  chembl_enriched.get("recall_audit_terms", []),
    }

    try:
        trials = client.search_qt_related_trials(
            drug_name, max_results=max_results, drug_name_set=drug_name_set
        )
        molecule_entry["qt_trials_count"] = len(trials)
        molecule_entry["qt_title_hits"]   = sum(1 for t in trials if t.get("qt_related_title"))
        molecule_entry["qt_outcome_hits"] = sum(1 for t in trials if t.get("qt_related_outcome"))
        molecule_entry["example_nct_ids"] = [
            t.get("nct_id", "") for t in trials[:3] if t.get("nct_id")
        ]

        for t in trials:
            nct_id = t.get("nct_id", "")
            cred   = t.get("research_institution_credibility") or {}

            raw: Dict[str, Any] = {}
            if nct_id:
                # recall branch 已在 search 阶段 prefetch 完整 study JSON
                raw = t.get("_prefetched_raw") or client.get_study_by_nct_id(nct_id) or {}
                raw_studies.append({
                    "molecule_chembl_id": chembl_id,
                    "pref_name":          drug_name,
                    "nct_id":             nct_id,
                    "raw_response":       raw,
                })

            enriched = client.enrich_trial_evidence(t, raw, drug_name_set)
            results_section = enriched.get("clinical_results_section") or {}
            alignment = enriched.get("drug_trial_alignment") or {}
            qt_attr = enriched.get("qt_result_attribution") or {}
            evidence_tier = enriched.get("evidence_tier", "manual_review_required")
            search_branch = enriched.get("search_branch", "")

            if search_branch == "results_recall":
                molecule_entry["recall_branch_hits"] += 1
            if results_section.get("has_qt_results"):
                molecule_entry["qt_results_hits"] += 1
            if alignment.get("evidence_attribution_level") in {"direct", "partial"}:
                molecule_entry["attributable_qt_hits"] += 1
            if evidence_tier.startswith("direct_qt"):
                molecule_entry["direct_evidence_hits"] += 1
            if evidence_tier in {"false_positive_mapping", "rescue_or_background_only"}:
                molecule_entry["false_positive_hits"] += 1

            trial_entry = _build_trial_entry(enriched)
            molecule_entry["trials"].append(trial_entry)

            detail_row: Dict[str, Any] = {
                "molecule_chembl_id":   chembl_id,
                "pref_name":            drug_name,
                "max_phase":            row.get("max_phase"),
                "phase_name":           row.get("phase_name"),
                "nct_id":               nct_id,
                "trial_status":         t.get("status", ""),
                "title":                t.get("title", ""),
                "summary":              t.get("summary", ""),
                "search_branch":        search_branch,
                "protocol_qt_hit":      enriched.get("protocol_qt_hit", False),
                "results_qt_hit":       enriched.get("results_qt_hit", False),
                "qt_related_title":     t.get("qt_related_title"),
                "qt_related_outcome":   t.get("qt_related_outcome"),
                "qt_outcome_measure":   t.get("qt_outcome_measure", ""),
            }
            detail_row.update(_flatten_credibility(cred))
            detail_row.update(_flatten_results_section(results_section))
            detail_row.update(_flatten_alignment(alignment, qt_attr, evidence_tier))
            trial_rows.append(detail_row)

    except Exception as e:
        molecule_entry["status"] = "error"
        molecule_entry["error"]  = str(e)

    return molecule_entry, trial_rows, raw_studies


def _flatten_molecule_for_csv(molecule: Dict[str, Any]) -> Dict[str, Any]:
    """分子级 CSV 行：example_nct_ids 用分号拼接。"""
    row = dict(molecule)
    row.pop("trials", None)
    row.pop("drug_name_set", None)
    row["example_nct_ids"] = ";".join(molecule.get("example_nct_ids") or [])
    return row


def _iter_trials(molecules: List[Dict[str, Any]]):
    for mol in molecules:
        for trial in mol.get("trials") or []:
            yield trial


def _iter_results_sections(molecules: List[Dict[str, Any]]):
    """从三层 JSON 中遍历所有 clinical_results_section。"""
    for trial in _iter_trials(molecules):
        yield trial.get("clinical_results_section") or {}


def _build_evidence_json(
    molecules: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    """组装三层嵌套 JSON：metadata → molecules → trials（含 alignment/results/attribution）。"""
    return {
        "metadata": summary,
        "molecules": molecules,
    }


def _summarize(
    molecules: List[Dict[str, Any]],
    trial_rows: List[Dict[str, Any]],
    *,
    input_csv: Path,
    limit: int,
) -> Dict[str, Any]:
    ok_rows    = [m for m in molecules if m["status"] == "ok"]
    with_trials = sum(1 for m in ok_rows if m["qt_trials_count"] > 0)
    errors     = sum(1 for m in molecules if m["status"] == "error")
    skipped    = sum(1 for m in molecules if m["status"] == "skipped")
    trial_counts = Counter(m["qt_trials_count"] for m in ok_rows)

    # ── sponsor 分布 ─────────────────────────────────────────────────────────
    sponsor_class_dist:    Counter = Counter()
    credibility_level_dist: Counter = Counter()
    sponsor_name_top:      Counter = Counter()
    for tr in trial_rows:
        cls_ = (tr.get("cred_sponsor_class") or "UNKNOWN").strip() or "UNKNOWN"
        lvl  = (tr.get("cred_level") or "unknown").strip() or "unknown"
        name = (tr.get("cred_sponsor_name") or "").strip()
        sponsor_class_dist[cls_] += 1
        credibility_level_dist[lvl] += 1
        if name:
            sponsor_name_top[name] += 1

    # ── resultsSection 分布 ──────────────────────────────────────────────────
    rs_list = list(_iter_results_sections(molecules))
    results_stats: Dict[str, Any] = {
        "trials_with_posted_results":        sum(1 for r in rs_list if r.get("has_posted_results")),
        "trials_with_results_section":       sum(1 for r in rs_list if r.get("has_results_section")),
        "trials_with_outcome_measures":      sum(1 for r in rs_list if r.get("has_results_outcome_measures")),
        "trials_with_qt_result_measures":    sum(1 for r in rs_list if r.get("has_qt_results")),
        "trials_with_adverse_events":        sum(1 for r in rs_list if r.get("has_adverse_events")),
        "trials_with_more_info":             sum(1 for r in rs_list if r.get("has_more_info")),
        "total_qt_result_measure_objects":   sum(
            len(r.get("qt_result_measures") or []) for r in rs_list
        ),
    }

    # ── alignment / evidence tier / recall branch 分布 ───────────────────────
    tier_dist: Counter = Counter()
    match_level_dist: Counter = Counter()
    attribution_dist: Counter = Counter()
    search_branch_dist: Counter = Counter()
    for trial in _iter_trials(molecules):
        tier_dist[(trial.get("evidence_tier") or "unknown")] += 1
        search_branch_dist[(trial.get("search_branch") or "unknown")] += 1
        align = trial.get("drug_trial_alignment") or {}
        match_level_dist[(align.get("drug_match_level") or "unknown")] += 1
        attribution_dist[(align.get("evidence_attribution_level") or "unknown")] += 1

    # ── 三类分组统计 ──────────────────────────────────────────────────────────
    _PRIMARY_TIERS = {"direct_qt_actual_result", "direct_qt_protocol_outcome",
                      "results_only_qt_actual_result"}
    _SUPPORT_TIERS = {"direct_ecg_conduction_evidence", "ecg_broad_supportive_evidence",
                      "comparator_qt_evidence", "combination_qt_evidence",
                      "combination_ecg_evidence", "comparator_ecg_evidence",
                      "direct_qt_context"}
    _EXCLUDE_TIERS = {"rescue_or_background_only", "manual_review_required",
                      "actual_result_needs_review", "combination_context_only",
                      "non_qt_cardiology_endpoint", "non_qt_excluded",
                      "false_positive_mapping", "recall_audit",
                      "direct_actual_result_review"}

    alignment_stats = {
        "evidence_tier_distribution": dict(tier_dist.most_common()),
        "search_branch_distribution": dict(search_branch_dist.most_common()),
        "drug_match_level_distribution": dict(match_level_dist.most_common()),
        "evidence_attribution_distribution": dict(attribution_dist.most_common()),
        # ── 主证据（进入 direct clinical QT evidence 统计）────────────────────
        "primary_qt_evidence": {
            t: tier_dist.get(t, 0) for t in sorted(_PRIMARY_TIERS)
        },
        "primary_qt_evidence_total": sum(tier_dist.get(t, 0) for t in _PRIMARY_TIERS),
        # ── 支持证据 ──────────────────────────────────────────────────────────
        "supportive_evidence": {
            t: tier_dist.get(t, 0) for t in sorted(_SUPPORT_TIERS)
        },
        "supportive_evidence_total": sum(tier_dist.get(t, 0) for t in _SUPPORT_TIERS),
        # ── 排除 / 复核 ──────────────────────────────────────────────────────
        "excluded_or_review": {
            t: tier_dist.get(t, 0) for t in sorted(_EXCLUDE_TIERS)
        },
        "excluded_or_review_total": sum(tier_dist.get(t, 0) for t in _EXCLUDE_TIERS),
        # ── recall branch ─────────────────────────────────────────────────────
        "trials_strict_protocol_qt": search_branch_dist.get("strict_protocol_qt", 0),
        "trials_results_recall_branch": search_branch_dist.get("results_recall", 0),
    }

    recall_branch_stats = {
        "molecules_with_recall_branch_hits": sum(
            1 for m in ok_rows if m.get("recall_branch_hits", 0) > 0
        ),
        "total_recall_branch_hits": sum(m.get("recall_branch_hits", 0) for m in ok_rows),
    }

    return {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "input_csv":      str(input_csv),
        "requested_limit": limit,
        "total_rows":     len(molecules),
        "ok_rows":        len(ok_rows),
        "error_rows":     errors,
        "skipped_rows":   skipped,
        "molecules_with_qt_trials": with_trials,
        "molecules_with_qt_trials_pct": (
            round(100.0 * with_trials / len(ok_rows), 2) if ok_rows else 0.0
        ),
        "total_qt_trial_entries":  sum(m["qt_trials_count"] for m in ok_rows),
        "qt_trials_count_distribution": {
            str(k): v for k, v in sorted(trial_counts.items())
        },
        # sponsor 分布
        "sponsor_class_distribution":    dict(sponsor_class_dist.most_common()),
        "credibility_level_distribution": dict(sorted(credibility_level_dist.items())),
        "top20_sponsor_names":           dict(sponsor_name_top.most_common(20)),
        # resultsSection 分布
        "results_section_stats": results_stats,
        # alignment / tier / recall branch 分布
        "alignment_stats": alignment_stats,
        "recall_branch_stats": recall_branch_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="对 Chembl_phase234_Data.csv 批量调用 ClinicalTrialsClient"
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="输入 CSV（默认 data/Chembl_phase234_Data.csv）",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="分子级汇总 CSV 输出路径",
    )
    parser.add_argument(
        "--trials-output", type=Path, default=DEFAULT_TRIALS_OUTPUT,
        help="逐条试验明细 CSV（含 title/summary/可信度/results 摘要）",
    )
    parser.add_argument(
        "--raw-jsonl", type=Path, default=DEFAULT_RAW_JSONL,
        help="get_study_by_nct_id 原始回传 JSONL 路径",
    )
    parser.add_argument(
        "--evidence-json", type=Path, default=DEFAULT_EVIDENCE_JSON,
        help="三层嵌套 JSON（分子 → 试验 → clinical_results_section）",
    )
    parser.add_argument(
        "--summary", type=Path, default=DEFAULT_SUMMARY,
        help="汇总 JSON 输出路径",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="只测前 N 条（0=全部）",
    )
    parser.add_argument(
        "--max-results", type=int, default=30,
        help="每个药名传给 search_qt_related_trials 的上限",
    )
    parser.add_argument(
        "--rate-limit", type=float, default=0.5,
        help="API 调用间隔（秒）",
    )
    args = parser.parse_args()

    input_csv     = args.input.resolve()
    output_csv    = args.output.resolve()
    trials_csv    = args.trials_output.resolve()
    raw_jsonl     = args.raw_jsonl.resolve()
    evidence_json = args.evidence_json.resolve()
    summary_json  = args.summary.resolve()

    if not input_csv.is_file():
        print(f"找不到输入文件: {input_csv}", file=sys.stderr)
        return 2

    df = _load_molecules(input_csv, args.limit)
    client = ClinicalTrialsClient(rate_limit=args.rate_limit)
    chembl_client = ChEMBLClient()
    enricher = ChemblDrugNameEnricher(
        chembl_client=chembl_client,
        cache_dir=DEFAULT_CHEMBL_NAME_CACHE,
    )

    molecules:   List[Dict[str, Any]] = []
    trial_rows:  List[Dict[str, Any]] = []
    raw_studies: List[Dict[str, Any]] = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="ClinicalTrials",
        unit="mol",
        file=sys.stderr,
    ):
        molecule_entry, details, raws = _scan_one(
            client, enricher, row, max_results=args.max_results
        )
        molecules.append(molecule_entry)
        trial_rows.extend(details)
        raw_studies.extend(raws)

    # ── 写输出 ───────────────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([_flatten_molecule_for_csv(m) for m in molecules]).to_csv(
        output_csv, index=False
    )
    pd.DataFrame(trial_rows).to_csv(trials_csv, index=False)

    with open(raw_jsonl, "w", encoding="utf-8") as f:
        for item in raw_studies:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = _summarize(
        molecules, trial_rows,
        input_csv=input_csv, limit=args.limit,
    )
    summary["output_csv"]              = str(output_csv)
    summary["trials_csv"]              = str(trials_csv)
    summary["raw_jsonl"]               = str(raw_jsonl)
    summary["evidence_json"]           = str(evidence_json)
    summary["raw_study_count"]         = len(raw_studies)
    summary["total_trial_detail_rows"] = len(trial_rows)
    summary["molecule_count"]          = len(molecules)

    evidence_payload = _build_evidence_json(molecules, summary)
    evidence_json.parent.mkdir(parents=True, exist_ok=True)
    evidence_json.write_text(
        json.dumps(evidence_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"分子汇总        -> {output_csv}")
    print(f"试验明细        -> {trials_csv}")
    print(f"原始回传        -> {raw_jsonl}")
    print(f"三层 evidence   -> {evidence_json}")
    print(f"统计汇总        -> {summary_json}")
    return 1 if summary["error_rows"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
