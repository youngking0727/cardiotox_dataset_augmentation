#!/usr/bin/env python3
"""单 case 跑 ClinicalTrials phase234 全流程，各阶段 JSON 落盘。

用法（项目根目录）:
  python scripts/test_clinicaltrials_one_case.py --chembl-id CHEMBL10272
  python scripts/test_clinicaltrials_one_case.py --pref-name BENZETIMIDE
  python scripts/test_clinicaltrials_one_case.py --offset 10 --phases 2

默认输出目录:
  data/output/ClinicalTrails/case/{chembl_id}_{pref_name}/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
_SCRIPTS = _PROJECT_ROOT / "scripts"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import test_clinicaltrials_phase234 as phase234  # noqa: E402

from utils.chembl_client import ChEMBLClient  # noqa: E402
from utils.chembl_drug_name_enrichment import ChemblDrugNameEnricher  # noqa: E402
from utils.clinicaltrials_client import ClinicalTrialsClient  # noqa: E402
from utils.clinicaltrials_drug_alignment import build_drug_name_set  # noqa: E402

DEFAULT_INPUT = phase234.DEFAULT_INPUT
DEFAULT_CASE_BASE = _PROJECT_ROOT / "data/output/ClinicalTrails/case"
DEFAULT_CHEMBL_NAME_CACHE = phase234.DEFAULT_CHEMBL_NAME_CACHE

_BRANCH_INTERVENTION = phase234._BRANCH_INTERVENTION
_BRANCH_FULL_TEXT = phase234._BRANCH_FULL_TEXT


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", (text or "").strip()).strip("_")
    return slug or "unknown"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _local_qt_signal_detail(
    client: ClinicalTrialsClient,
    protocol_study: Dict[str, Any],
    raw_study: Dict[str, Any],
) -> Dict[str, Any]:
    protocol_hit = phase234._protocol_local_qt_signal(protocol_study)
    results_section = client.parse_results_section(raw_study) if raw_study else {}
    results_hit = phase234._results_local_qt_signal(results_section)
    ae_hit = phase234._adverse_events_local_qt_signal(raw_study)
    passes = protocol_hit or results_hit or ae_hit
    return {
        "passes_local_qt_filter": passes,
        "protocol_qt_hit": protocol_hit,
        "results_qt_hit": results_hit,
        "adverse_events_qt_hit": ae_hit,
        "results_section_flags": {
            "has_posted_results": results_section.get("has_posted_results", False),
            "has_qt_results": results_section.get("has_qt_results", False),
            "has_ecg_conduction_results": results_section.get(
                "has_ecg_conduction_results", False
            ),
            "has_ecg_broad_results": results_section.get("has_ecg_broad_results", False),
            "has_cardiac_ae_results": results_section.get("has_cardiac_ae_results", False),
        },
    }


def _resolve_molecule_row(
    input_csv: Path,
    *,
    chembl_id: str | None,
    pref_name: str | None,
    offset: int,
    phases: List[str] | None,
) -> pd.Series:
    df = pd.read_csv(input_csv)
    df = phase234._apply_phase_filters(
        df, phases=phases, exclude_approved=False, max_phases=None
    )

    if chembl_id:
        matched = df[
            df["molecule_chembl_id"].astype(str).str.strip() == chembl_id.strip()
        ]
        if matched.empty:
            raise ValueError(f"未在 {input_csv} 中找到 molecule_chembl_id={chembl_id}")
        return matched.iloc[0]

    if pref_name:
        matched = df[df["pref_name"].astype(str).str.strip().upper() == pref_name.strip().upper()]
        if matched.empty:
            raise ValueError(f"未在 {input_csv} 中找到 pref_name={pref_name}")
        return matched.iloc[0]

    if offset > 0:
        df = df.iloc[offset:].reset_index(drop=True)
    if df.empty:
        raise ValueError("过滤后没有可处理的分子")
    return df.iloc[0]


def _save_branch_queries(
    client: ClinicalTrialsClient,
    query_terms: List[str],
    query_term_sources: Dict[str, str],
    *,
    max_results: int,
    phase_name: Any,
    max_phase: Any,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    """逐 term × 分支查询 CT.gov，保存原始 studies 回传。"""
    saved_rows: List[Dict[str, Any]] = []
    search_strategy = phase234._resolve_search_strategy(phase_name, max_phase)

    for term in query_terms:
        if not (term or "").strip():
            continue

        term_dir = out_dir / "terms" / _slug(term)
        term_dir.mkdir(parents=True, exist_ok=True)

        intr_url = phase234._build_ctgov_branch_query_url(
            term, max_results, _BRANCH_INTERVENTION
        )
        intr_studies = phase234._search_ctgov_branch(
            client, term, max_results, _BRANCH_INTERVENTION
        )
        intr_ncts, intr_titles, _ = phase234._summarize_branch_studies(client, intr_studies)
        _write_json(
            term_dir / "intr_search.json",
            {
                "query_term": term,
                "query_term_source": query_term_sources.get(term, ""),
                "branch": _BRANCH_INTERVENTION,
                "search_strategy": search_strategy,
                "query_url": intr_url,
                "study_count": len(intr_studies),
                "nct_ids": intr_ncts,
                "nct_titles": intr_titles,
                "studies": intr_studies,
            },
        )

        term_payload: Dict[str, Any] | None = None
        if phase234._should_run_term_search(search_strategy, len(intr_ncts)):
            term_url = phase234._build_ctgov_branch_query_url(
                term, max_results, _BRANCH_FULL_TEXT
            )
            term_studies = phase234._search_ctgov_branch(
                client, term, max_results, _BRANCH_FULL_TEXT
            )
            term_ncts, term_titles, _ = phase234._summarize_branch_studies(
                client, term_studies
            )
            term_payload = {
                "query_term": term,
                "query_term_source": query_term_sources.get(term, ""),
                "branch": _BRANCH_FULL_TEXT,
                "search_strategy": search_strategy,
                "query_url": term_url,
                "study_count": len(term_studies),
                "nct_ids": term_ncts,
                "nct_titles": term_titles,
                "studies": term_studies,
            }
            _write_json(term_dir / "full_text_search.json", term_payload)

        saved_rows.append(
            {
                "query_term": term,
                "query_term_source": query_term_sources.get(term, ""),
                "intr_hit_count": len(intr_ncts),
                "term_hit_count": len(term_payload["nct_ids"]) if term_payload else 0,
                "intr_query_url": intr_url,
                "full_text_query_url": (
                    term_payload["query_url"] if term_payload else ""
                ),
            }
        )

    return saved_rows


def run_one_case(
    row: pd.Series,
    *,
    out_dir: Path,
    max_results: int,
    rate_limit: float,
    include_nct_ids: List[str] | None = None,
) -> Dict[str, Any]:
    """跑单分子全流程，各阶段 JSON 写入 out_dir。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    chembl_id = str(row.get("molecule_chembl_id", "")).strip()
    raw_pref = row.get("pref_name", "")
    drug_name = "" if pd.isna(raw_pref) else str(raw_pref).strip()
    phase_name = row.get("phase_name")

    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_dir": str(out_dir),
        "molecule_chembl_id": chembl_id,
        "pref_name": drug_name,
        "stages": [],
    }

    def _stage(name: str, rel_path: str, summary: Dict[str, Any] | None = None) -> None:
        entry = {"stage": name, "path": rel_path}
        if summary:
            entry["summary"] = summary
        manifest["stages"].append(entry)

    _write_json(out_dir / "00_input_molecule.json", row.to_dict())
    _stage("input_molecule", "00_input_molecule.json")

    client = ClinicalTrialsClient(rate_limit=rate_limit)
    chembl_client = ChEMBLClient()
    enricher = ChemblDrugNameEnricher(
        chembl_client=chembl_client,
        cache_dir=DEFAULT_CHEMBL_NAME_CACHE,
    )

    chembl_enriched = enricher.enrich(chembl_id, pref_name_fallback=drug_name)
    chembl_molecule: Dict[str, Any] | None = None
    if chembl_id:
        chembl_molecule, _ = enricher.fetch_molecule(chembl_id)

    chembl_payload = {
        "chembl_enriched": chembl_enriched,
        "chembl_molecule": chembl_molecule,
        "raw_chembl_debug_fields": phase234._extract_raw_chembl_debug_fields(
            drug_name, chembl_molecule
        ),
    }
    _write_json(out_dir / "01_chembl_enrichment.json", chembl_payload)
    _stage(
        "chembl_enrichment",
        "01_chembl_enrichment.json",
        {
            "enrichment_status": chembl_enriched.get("enrichment_status", ""),
            "strong_match_terms": len(chembl_enriched.get("strong_match_terms") or []),
            "weak_terms": len(chembl_enriched.get("weak_terms") or []),
        },
    )

    query_terms, query_term_sources = phase234._build_query_terms(
        drug_name, chembl_molecule, chembl_enriched
    )
    query_payload = {
        "query_terms": query_terms,
        "query_term_sources": query_term_sources,
        "formatted_sources": phase234._format_query_term_sources(
            query_terms, query_term_sources
        ),
    }
    _write_json(out_dir / "02_query_terms.json", query_payload)
    _stage("query_terms", "02_query_terms.json", {"count": len(query_terms)})

    debug_row = phase234._make_query_debug_row(0, chembl_id, phase_name)
    debug_row["pref_name"] = drug_name
    debug_row.update(chembl_payload["raw_chembl_debug_fields"])
    debug_row["filtered_query_terms"] = phase234._pipe_join(query_terms)
    debug_row["query_terms_used"] = phase234._pipe_join(query_terms)
    debug_row["query_term_sources"] = query_payload["formatted_sources"]
    debug_row["query_terms_count"] = len(query_terms)

    molecule_entry: Dict[str, Any] = {
        "molecule_chembl_id": chembl_id,
        "pref_name": drug_name,
        "max_phase": row.get("max_phase"),
        "phase_name": phase_name,
        "drug_name_set": {},
        "recall_audit": {
            "related_match_terms": [],
            "weak_terms": [],
            "recall_audit_terms": [],
        },
        "qt_trials_count": 0,
        "drug_only_raw_count": 0,
        "qt_after_local_filter_count": 0,
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
    trial_rows: List[Dict[str, Any]] = []
    raw_studies: List[Dict[str, Any]] = []
    include_nct_ids = [str(x).strip().upper() for x in (include_nct_ids or []) if str(x).strip()]

    if not query_terms:
        molecule_entry["status"] = "skipped"
        molecule_entry["error"] = "no_searchable_identity"
        phase234._assign_query_debug_skip_reason(debug_row)
        _write_json(out_dir / "09_query_debug.json", debug_row)
        _write_json(out_dir / "10_molecule_entry.json", molecule_entry)
        _stage("query_debug", "09_query_debug.json", {"status": "skipped"})
        _stage("molecule_entry", "10_molecule_entry.json")
        _write_json(out_dir / "manifest.json", manifest)
        return {
            "molecule_entry": molecule_entry,
            "trial_rows": trial_rows,
            "summary": {"status": "skipped"},
            "out_dir": str(out_dir),
        }

    branch_query_dir = out_dir / "03_ctgov_branch_queries"
    saved_branch_rows = _save_branch_queries(
        client,
        query_terms,
        query_term_sources,
        max_results=max_results,
        phase_name=phase_name,
        max_phase=row.get("max_phase"),
        out_dir=branch_query_dir,
    )
    _stage(
        "ctgov_branch_queries",
        "03_ctgov_branch_queries/",
        {"term_count": len(saved_branch_rows)},
    )

    ctgov_probe = phase234._probe_ctgov_raw_hits(
        client,
        query_terms,
        max_results,
        query_term_sources=query_term_sources,
        row_idx=0,
        chembl_id=chembl_id,
        phase_name=phase_name,
        max_phase=row.get("max_phase"),
    )
    branch_debug_rows = ctgov_probe.pop("_branch_debug_rows")
    drug_only_candidates = ctgov_probe.pop("_drug_only_candidates")
    search_strategy = ctgov_probe.pop("_search_strategy", "")

    forced_nct_debug_rows: List[Dict[str, Any]] = []
    for forced_nct in include_nct_ids:
        if not re.match(r"^NCT\d{8}$", forced_nct):
            forced_nct_debug_rows.append({
                "nct_id": forced_nct,
                "status": "skipped_invalid_nct_id",
                "reason": "NCT ID must match NCT followed by 8 digits.",
            })
            continue
        if forced_nct in drug_only_candidates:
            forced_nct_debug_rows.append({
                "nct_id": forced_nct,
                "status": "already_in_recall",
            })
            continue
        forced_raw = client.get_study_by_nct_id(forced_nct) or {}
        forced_protocol = forced_raw.get("protocolSection") or forced_raw.get("protocol") or {}
        if not forced_protocol:
            forced_nct_debug_rows.append({
                "nct_id": forced_nct,
                "status": "fetch_failed_or_no_protocol",
            })
            continue
        title = str(
            ((forced_protocol.get("identificationModule") or {}).get("briefTitle"))
            or ((forced_protocol.get("identificationModule") or {}).get("officialTitle"))
            or ""
        )
        drug_only_candidates[forced_nct] = {
            "nct_id": forced_nct,
            "title": title,
            "protocol": forced_protocol,
            "selected_branch": "forced_nct_debug",
            "query_term": forced_nct,
            "alignment_fields_matched": ["forced_nct_debug"],
            "forced_debug_nct": True,
            "_prefetched_raw": forced_raw,
        }
        forced_nct_debug_rows.append({
            "nct_id": forced_nct,
            "status": "added",
            "title": title,
        })

    probe_summary = {
        **ctgov_probe,
        "selected_search_strategy": search_strategy,
        "branch_debug_rows": branch_debug_rows,
        "saved_branch_query_rows": saved_branch_rows,
        "forced_nct_debug_rows": forced_nct_debug_rows,
        "include_nct_ids": include_nct_ids,
    }
    _write_json(out_dir / "04_ctgov_probe_summary.json", probe_summary)
    _stage(
        "ctgov_probe_summary",
        "04_ctgov_probe_summary.json",
        {
            "drug_only_raw_hit_count": ctgov_probe.get("drug_only_raw_hit_count", 0),
            "broad_added_count": ctgov_probe.get("broad_added_count", 0),
            "forced_nct_count": len(forced_nct_debug_rows),
            "search_strategy": search_strategy,
        },
    )
    _write_json(out_dir / "04b_forced_nct_debug.json", forced_nct_debug_rows)
    _stage("forced_nct_debug", "04b_forced_nct_debug.json", {"count": len(forced_nct_debug_rows)})

    drug_only_payload = {
        nct: {
            "nct_id": nct,
            "title": cand.get("title", ""),
            "meta": {k: v for k, v in cand.items() if k not in {"protocol", "title"}},
            "protocol": cand.get("protocol"),
        }
        for nct, cand in drug_only_candidates.items()
    }
    _write_json(out_dir / "05_drug_only_candidates.json", drug_only_payload)
    _stage(
        "drug_only_candidates",
        "05_drug_only_candidates.json",
        {"count": len(drug_only_candidates)},
    )

    debug_row["selected_search_strategy"] = search_strategy
    debug_row.update(ctgov_probe)
    if branch_debug_rows:
        debug_row["intr_hit_count"] = max(
            int(r.get("intr_hit_count") or 0) for r in branch_debug_rows
        )
        debug_row["term_hit_count"] = sum(
            int(r.get("term_hit_count") or 0) for r in branch_debug_rows
        )

    local_filter_dir = out_dir / "06_local_qt_filter"
    raw_studies_dir = out_dir / "07_raw_studies"
    built_trials: List[Dict[str, Any]] = []

    for nct_id, cand in drug_only_candidates.items():
        protocol_study = cand["protocol"]
        raw_study = cand.get("_prefetched_raw") or client.get_study_by_nct_id(nct_id) or {}
        raw_studies.append(
            {
                "molecule_chembl_id": chembl_id,
                "pref_name": drug_name,
                "nct_id": nct_id,
                "stage": "drug_only_candidate",
                "raw_response": raw_study,
            }
        )
        _write_json(raw_studies_dir / f"{nct_id}_drug_only.json", raw_study)

        signal_detail = _local_qt_signal_detail(client, protocol_study, raw_study)
        _write_json(
            local_filter_dir / f"{nct_id}.json",
            {
                "nct_id": nct_id,
                "title": cand.get("title", ""),
                **signal_detail,
            },
        )

        if not signal_detail["passes_local_qt_filter"]:
            continue

        trial = phase234._build_trial_from_local_qt_candidate(
            client, protocol_study, raw_study
        )
        if not trial:
            continue

        selected_branch = cand.get("selected_branch", "")
        if selected_branch in {"broad_drug_qt_recall", "forced_nct_debug"}:
            trial["search_branch"] = selected_branch
            trial["_broad_recall_query"] = cand.get("query_term", "")
            trial["forced_debug_nct"] = bool(cand.get("forced_debug_nct", False))
            trial["alignment_fields_matched"] = cand.get("alignment_fields_matched", [])

        built_trials.append(trial)
        raw_studies.append(
            {
                "molecule_chembl_id": chembl_id,
                "pref_name": drug_name,
                "nct_id": nct_id,
                "stage": "qt_after_local_filter",
                "raw_response": raw_study,
            }
        )
        _write_json(raw_studies_dir / f"{nct_id}_qt_pass.json", raw_study)

    _stage(
        "local_qt_filter",
        "06_local_qt_filter/",
        {
            "drug_only_count": len(drug_only_candidates),
            "qt_pass_count": len(built_trials),
        },
    )
    _write_json(out_dir / "06b_qt_trials_built.json", built_trials)
    _stage("qt_trials_built", "06b_qt_trials_built.json", {"count": len(built_trials)})

    drug_name_set = build_drug_name_set(drug_name, enriched=chembl_enriched)
    alignment_terms = phase234._normalize_match_terms(list(query_terms))
    drug_name_set["strong_match_terms"] = alignment_terms
    drug_name_set["_strong_match_terms"] = alignment_terms

    molecule_entry["drug_name_set"] = {
        "molecule_chembl_id": chembl_enriched.get("molecule_chembl_id", chembl_id),
        "pref_name": chembl_enriched.get("pref_name", drug_name),
        "chembl_pref_name": chembl_enriched.get("chembl_pref_name", ""),
        "strong_match_terms": chembl_enriched.get("strong_match_terms", []),
        "related_match_terms": chembl_enriched.get("related_match_terms", []),
        "weak_terms": chembl_enriched.get("weak_terms", []),
        "parent_molecule_chembl_id": chembl_enriched.get("parent_molecule_chembl_id", ""),
        "parent_pref_name": chembl_enriched.get("parent_pref_name", ""),
        "enrichment_status": chembl_enriched.get("enrichment_status", "ok"),
    }
    molecule_entry["recall_audit"] = {
        "related_match_terms": chembl_enriched.get("related_match_terms", []),
        "weak_terms": chembl_enriched.get("weak_terms", []),
        "recall_audit_terms": chembl_enriched.get("recall_audit_terms", []),
    }
    molecule_entry["drug_only_raw_count"] = len(drug_only_candidates)
    molecule_entry["qt_after_local_filter_count"] = len(built_trials)
    molecule_entry["qt_trials_count"] = len(built_trials)
    debug_row["qt_after_local_filter_count"] = len(built_trials)
    debug_row["final_qt_hits"] = len(built_trials)
    debug_row["include_nct_ids"] = phase234._pipe_join(include_nct_ids)
    debug_row["forced_nct_added_count"] = sum(1 for r in forced_nct_debug_rows if r.get("status") == "added")

    enriched_dir = out_dir / "08_enriched_trials"
    for trial in built_trials:
        nct_id = trial.get("nct_id", "")
        raw = trial.get("_prefetched_raw") or {}
        enriched_trial = client.enrich_trial_evidence(trial, raw, drug_name_set)
        enriched_trial = phase234._apply_title_only_weak_override(enriched_trial)

        _write_json(enriched_dir / f"{nct_id}.json", enriched_trial)

        results_section = enriched_trial.get("clinical_results_section") or {}
        alignment = enriched_trial.get("drug_trial_alignment") or {}
        qt_attr = enriched_trial.get("qt_result_attribution") or {}
        evidence_tier = enriched_trial.get("evidence_tier", "manual_review_required")
        evidence_tier_reason = enriched_trial.get("evidence_tier_reason", "")
        search_branch = enriched_trial.get("search_branch", "")

        if search_branch == "results_recall":
            molecule_entry["recall_branch_hits"] += 1
        if results_section.get("has_qt_results"):
            molecule_entry["qt_results_hits"] += 1
        if alignment.get("evidence_attribution_level") in {"direct", "partial"}:
            molecule_entry["attributable_qt_hits"] += 1
        if str(evidence_tier).startswith("direct_qt"):
            molecule_entry["direct_evidence_hits"] += 1
        if evidence_tier in {"false_positive_mapping", "rescue_or_background_only"}:
            molecule_entry["false_positive_hits"] += 1

        trial_entry = phase234._build_trial_entry(enriched_trial)
        molecule_entry["trials"].append(trial_entry)

        detail_row: Dict[str, Any] = {
            "molecule_chembl_id": chembl_id,
            "pref_name": drug_name,
            "max_phase": row.get("max_phase"),
            "phase_name": phase_name,
            "nct_id": nct_id,
            "trial_status": trial.get("status", ""),
            "title": trial.get("title", ""),
            "summary": trial.get("summary", ""),
            "search_branch": search_branch,
            "broad_recall_query": enriched_trial.get("_broad_recall_query", trial.get("_broad_recall_query", "")),
            "forced_debug_nct": enriched_trial.get("forced_debug_nct", trial.get("forced_debug_nct", False)),
            "protocol_qt_hit": enriched_trial.get("protocol_qt_hit", False),
            "results_qt_hit": enriched_trial.get("results_qt_hit", False),
            "qt_related_title": trial.get("qt_related_title"),
            "qt_related_outcome": trial.get("qt_related_outcome"),
            "qt_outcome_measure": trial.get("qt_outcome_measure", ""),
            "error": "",
        }
        cred = enriched_trial.get("research_institution_credibility") or {}
        detail_row.update(phase234._flatten_credibility(cred))
        detail_row.update(phase234._flatten_results_section(results_section))
        detail_row.update(
            phase234._flatten_alignment(
                alignment,
                qt_attr,
                evidence_tier,
                evidence_tier_reason,
            )
        )
        trial_rows.append(detail_row)

    molecule_entry["qt_title_hits"] = sum(
        1 for t in molecule_entry["trials"] if t.get("qt_related_title")
    )
    molecule_entry["qt_outcome_hits"] = sum(
        1 for t in molecule_entry["trials"] if t.get("qt_related_outcome")
    )
    molecule_entry["example_nct_ids"] = [
        t.get("nct_id", "") for t in molecule_entry["trials"][:3] if t.get("nct_id")
    ]

    phase234._assign_query_debug_skip_reason(debug_row)
    _write_json(out_dir / "09_query_debug.json", debug_row)
    _write_json(out_dir / "10_molecule_entry.json", molecule_entry)
    _write_json(out_dir / "11_trial_details.json", trial_rows)

    summary = phase234._summarize(
        [molecule_entry],
        trial_rows,
        input_csv=DEFAULT_INPUT,
        limit=1,
        offset=0,
        phase_filter={"single_case": True},
    )
    summary["case_dir"] = str(out_dir)
    summary["raw_study_count"] = len(raw_studies)
    summary["include_nct_ids"] = include_nct_ids
    summary["forced_nct_debug_rows"] = forced_nct_debug_rows
    summary["broad_added_count"] = ctgov_probe.get("broad_added_count", 0)

    evidence_payload = phase234._build_evidence_json([molecule_entry], summary)
    _write_json(out_dir / "12_evidence.json", evidence_payload)
    _write_json(out_dir / "13_summary.json", summary)

    _stage("query_debug", "09_query_debug.json")
    _stage("molecule_entry", "10_molecule_entry.json", molecule_entry)
    _stage("trial_details", "11_trial_details.json", {"count": len(trial_rows)})
    _stage("evidence", "12_evidence.json")
    _stage("summary", "13_summary.json")
    _write_json(out_dir / "manifest.json", manifest)

    return {
        "molecule_entry": molecule_entry,
        "trial_rows": trial_rows,
        "summary": summary,
        "out_dir": str(out_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="单 case 跑 ClinicalTrials phase234 全流程并保存各阶段 JSON"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录（默认 data/output/ClinicalTrails/case/{chembl_id}_{pref_name}/）",
    )
    parser.add_argument("--chembl-id", type=str, default="", help="指定 molecule_chembl_id")
    parser.add_argument("--pref-name", type=str, default="", help="指定 pref_name")
    parser.add_argument("--offset", type=int, default=0, help="未指定 id/name 时跳过前 N 条")
    parser.add_argument(
        "--phases",
        nargs="+",
        metavar="PHASE",
        help="phase 过滤，同 test_clinicaltrials_phase234.py",
    )
    parser.add_argument("--max-results", type=int, default=30)
    parser.add_argument("--rate-limit", type=float, default=0.3)
    parser.add_argument(
        "--include-nct",
        nargs="+",
        default=[],
        help="调试用：额外强制纳入指定 NCT ID，仍走同一套 QT parser 和 attribution，不会自动升 Primary。",
    )

    args = parser.parse_args()

    try:
        phases_normalized = phase234._normalize_phase_names(args.phases or []) or None
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return 2

    if not args.chembl_id and not args.pref_name and args.offset < 0:
        print("offset 不能为负数", file=sys.stderr)
        return 2

    input_csv = args.input.resolve()
    if not input_csv.is_file():
        print(f"找不到输入文件: {input_csv}", file=sys.stderr)
        return 2

    try:
        row = _resolve_molecule_row(
            input_csv,
            chembl_id=args.chembl_id or None,
            pref_name=args.pref_name or None,
            offset=args.offset,
            phases=phases_normalized,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    chembl_id = str(row.get("molecule_chembl_id", "")).strip()
    pref_name = "" if pd.isna(row.get("pref_name")) else str(row.get("pref_name")).strip()
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (DEFAULT_CASE_BASE / f"{chembl_id}_{_slug(pref_name)}").resolve()
    )

    print(f"[case] {chembl_id} / {pref_name}", file=sys.stderr)
    print(f"[输出] {out_dir}", file=sys.stderr)

    result = run_one_case(
        row,
        out_dir=out_dir,
        max_results=args.max_results,
        rate_limit=args.rate_limit,
        include_nct_ids=args.include_nct,
    )

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"各阶段 JSON 已写入: {out_dir}")
    print(f"索引文件: {out_dir / 'manifest.json'}")
    return 0 if result["molecule_entry"].get("status") != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main())
