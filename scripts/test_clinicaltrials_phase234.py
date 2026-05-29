#!/usr/bin/env python3
"""批量测试 ClinicalTrialsClient：对 Chembl_phase234_Data.csv 中每个 pref_name 查 QT 相关试验。

用法（项目根目录）:
  python scripts/test_clinicaltrials_phase234.py
  python scripts/test_clinicaltrials_phase234.py --limit 50
  python scripts/test_clinicaltrials_phase234.py --offset 100 --limit 50
  python scripts/test_clinicaltrials_phase234.py --phases 2 3 --limit 20
  python scripts/test_clinicaltrials_phase234.py --phases "Phase II" "Phase III" --limit 20
  python scripts/test_clinicaltrials_phase234.py --exclude-approved --limit 30
  python scripts/test_clinicaltrials_phase234.py --max-phase 2 3 --limit 20
  python scripts/test_clinicaltrials_phase234.py --rate-limit 0.3

输出（默认根目录 data/output/ClinicalTrails/，按 --phases 写入 phase2 / phase3 / phase4）:
  .../phase2/phase234_clinicaltrials_scan.csv              # 分子级汇总
  .../phase2/phase234_clinicaltrials_trials_detail.csv     # 逐条试验明细
  .../phase2/phase234_clinicaltrials_raw_study.jsonl       # 原始回传
  .../phase2/phase234_clinicaltrials_evidence.json         # 三层嵌套 JSON
  .../phase2/phase234_clinicaltrials_scan_summary.json     # 统计汇总
  .../phase2/phase_ctgov_query_debug.csv                   # query 调试
  .../phase2/phase_ctgov_query_branch_debug.csv            # 分支查询调试
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from urllib.parse import urlencode
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
from utils.clinicaltrials_drug_alignment import (
    build_drug_name_set,
    extract_trial_drug_fields,
    match_terms_in_text,
    _normalize_match_terms,
)
from utils.clinicaltrials_result_evidence import (
    classify_outcome_text,
    classify_protocol_outcomes_module,
    pick_protocol_qt_outcome_measure,
)
from utils.chembl_drug_name_enrichment import ChemblDrugNameEnricher
from utils.chembl_client import ChEMBLClient

_LOCAL_QT_EVIDENCE_TYPES = frozenset({
    "qt_specific", "ecg_conduction", "ecg_broad", "cardiac_ae",
})


DEFAULT_INPUT = _PROJECT_ROOT / "data/Chembl_phase234_Data.csv"
DEFAULT_OUTPUT_BASE = _PROJECT_ROOT / "data/output/ClinicalTrails"
DEFAULT_CHEMBL_NAME_CACHE = _PROJECT_ROOT / "data/cache/chembl_drug_name_set"

_PHASE_NAME_TO_OUTPUT_DIR = {
    "Phase II": "phase2",
    "Phase III": "phase3",
    "Approved": "phase4",
}

_BRANCH_INTERVENTION = "intervention"
_BRANCH_FULL_TEXT = "full_text"
_BRANCH_BROAD_DRUG_QT_RECALL = "broad_drug_qt_recall"

_CONTROLLED_QT_RECALL_TERMS = (
    "QTc",
    "QTcB",
    "QTcF",
    "corrected QT",
    "QT interval",
    "Bazett",
    "Fridericia",
    "Fredericia",
)

_QUERY_STOPLIST = frozenset({
    "water", "sodium", "potassium", "chloride", "hydrochloride", "sulfate",
    "phosphate", "disodium", "monosodium", "hydrate", "anhydrous", "solution",
    "injection", "tablet", "capsule", "placebo", "vehicle", "saline", "dextrose",
})

_RESEARCH_CODE_VARIANT_RE = re.compile(r"^([A-Za-z]{2,4})(\d+)$")

_QUERY_TERM_SOURCES = frozenset({
    "pref_name",
    "synonym",
    "compound_record",
    "cross_reference",
    "research_code_variant",
})


_VALID_PHASE_NAMES = frozenset({"Phase II", "Phase III", "Approved"})

_PHASE_NAME_ALIASES = {
    "2": "Phase II",
    "ii": "Phase II",
    "phase2": "Phase II",
    "phase 2": "Phase II",
    "phase_2": "Phase II",
    "phaseii": "Phase II",
    "phase ii": "Phase II",
    "phase_ii": "Phase II",

    "3": "Phase III",
    "iii": "Phase III",
    "phase3": "Phase III",
    "phase 3": "Phase III",
    "phase_3": "Phase III",
    "phaseiii": "Phase III",
    "phase iii": "Phase III",
    "phase_iii": "Phase III",

    "4": "Approved",
    "approved": "Approved",
    "phase4": "Approved",
    "phase 4": "Approved",
    "phase_4": "Approved",
}


def _normalize_phase_names(raw: List[str]) -> List[str]:
    """解析 --phases：支持 Phase II / phase2 / 2 等别名。"""
    out: List[str] = []

    for item in raw:
        parts = [p.strip() for p in str(item).split(",") if p.strip()]

        for part in parts:
            key = part.lower().replace("-", " ").replace("_", " ").strip()
            key_compact = key.replace(" ", "")

            if key in _PHASE_NAME_ALIASES:
                out.append(_PHASE_NAME_ALIASES[key])
            elif key_compact in _PHASE_NAME_ALIASES:
                out.append(_PHASE_NAME_ALIASES[key_compact])
            elif part in _VALID_PHASE_NAMES:
                out.append(part)
            else:
                valid = ", ".join(sorted(_VALID_PHASE_NAMES))
                raise ValueError(
                    f"未知阶段 '{part}'，可用: {valid}；"
                    f"或别名 2/3/4, phase2, phase3, approved"
                )

    return list(dict.fromkeys(out))


def _resolve_phase_output_subdir(
    phases_normalized: List[str] | None,
    *,
    exclude_approved: bool,
) -> str:
    """根据 --phases 解析输出子目录：phase2 / phase3 / phase4。"""
    if phases_normalized:
        subdirs = [
            _PHASE_NAME_TO_OUTPUT_DIR[p]
            for p in phases_normalized
            if p in _PHASE_NAME_TO_OUTPUT_DIR
        ]
        subdirs = list(dict.fromkeys(subdirs))
        if len(subdirs) == 1:
            return subdirs[0]
        return "_".join(sorted(subdirs))

    if exclude_approved:
        return "phase2_phase3"
    return "all"


def _default_output_paths(phase_subdir: str) -> Dict[str, Path]:
    root = DEFAULT_OUTPUT_BASE / phase_subdir
    return {
        "output": root / "phase234_clinicaltrials_scan.csv",
        "trials_output": root / "phase234_clinicaltrials_trials_detail.csv",
        "raw_jsonl": root / "phase234_clinicaltrials_raw_study.jsonl",
        "evidence_json": root / "phase234_clinicaltrials_evidence.json",
        "summary": root / "phase234_clinicaltrials_scan_summary.json",
        "query_debug": root / "phase_ctgov_query_debug.csv",
        "branch_query_debug": root / "phase_ctgov_query_branch_debug.csv",
    }


def _expand_research_code_variants(
    term: str,
    parent_source: str,
) -> List[Tuple[str, str]]:
    """研发代号变体：BI409306 → (BI409306, parent), (BI 409306, variant), (BI-409306, variant)。"""
    raw = (term or "").strip()
    if not raw:
        return []

    compact = re.sub(r"[\s\-_]+", "", raw)
    m = _RESEARCH_CODE_VARIANT_RE.match(compact)
    if not m:
        return [(raw, parent_source)]

    prefix, num = m.group(1).upper(), m.group(2)
    base = f"{prefix}{num}"
    out: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for variant, source in (
        (base, parent_source),
        (f"{prefix} {num}", "research_code_variant"),
        (f"{prefix}-{num}", "research_code_variant"),
    ):
        key = variant.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((variant, source))
    return out


def _collect_sourced_name_candidates(
    pref_name: str,
    chembl_molecule: Dict[str, Any] | None,
    chembl_enriched: Dict[str, Any],
) -> List[Tuple[str, str]]:
    """从 pref_name / ChEMBL molecule / enrichment 收集 (名称, 来源)。"""
    out: List[Tuple[str, str]] = []

    def _push(raw: Any, source: str) -> None:
        text = str(raw or "").strip()
        if text and source in _QUERY_TERM_SOURCES:
            out.append((text, source))

    if pref_name:
        _push(pref_name, "pref_name")

    if chembl_molecule:
        mol_pref = str(chembl_molecule.get("pref_name") or "").strip()
        if mol_pref:
            _push(mol_pref, "pref_name")
        for entry in chembl_molecule.get("molecule_synonyms") or []:
            if not isinstance(entry, dict):
                continue
            _push(
                entry.get("molecule_synonym")
                or entry.get("synonyms")
                or entry.get("synonym"),
                "synonym",
            )
        for rec in chembl_molecule.get("compound_records") or []:
            if isinstance(rec, dict):
                _push(rec.get("compound_name"), "compound_record")
        for xref in chembl_molecule.get("cross_references") or []:
            if isinstance(xref, dict):
                _push(xref.get("xref_name"), "cross_reference")
                _push(xref.get("xref_id"), "cross_reference")

    for entry in chembl_enriched.get("molecule_synonyms") or []:
        if isinstance(entry, dict):
            _push(entry.get("name") or entry.get("molecule_synonym"), "synonym")

    return out


def _build_query_terms(
    pref_name: str,
    chembl_molecule: Dict[str, Any] | None,
    chembl_enriched: Dict[str, Any],
) -> Tuple[List[str], Dict[str, str]]:
    """构建 ClinicalTrials.gov 查询用词（含研发代号变体，过滤 stoplist）。"""
    seen: set[str] = set()
    terms: List[str] = []
    sources: Dict[str, str] = {}

    for name, parent_source in _collect_sourced_name_candidates(
        pref_name, chembl_molecule, chembl_enriched
    ):
        for variant, source in _expand_research_code_variants(name, parent_source):
            key = variant.strip().lower()
            if not key or key in _QUERY_STOPLIST:
                continue
            if key in seen:
                continue
            seen.add(key)
            term = variant.strip()
            terms.append(term)
            sources[term] = source

    return terms, sources


def _format_query_term_sources(
    terms: List[str],
    sources: Dict[str, str],
) -> str:
    return _pipe_join([f"{t}:{sources.get(t, '')}" for t in terms])


def _pipe_join(items: List[str]) -> str:
    return "|".join(items)


def _extract_raw_chembl_debug_fields(
    csv_pref_name: str,
    chembl_molecule: Dict[str, Any] | None,
) -> Dict[str, str]:
    """序列化 ChEMBL 原始名称来源，供 debug CSV 诊断。"""
    synonyms: List[str] = []
    compound_records: List[str] = []
    cross_refs: List[str] = []
    chembl_pref = ""

    if chembl_molecule:
        chembl_pref = str(chembl_molecule.get("pref_name") or "").strip()
        for entry in chembl_molecule.get("molecule_synonyms") or []:
            if not isinstance(entry, dict):
                continue
            syn = (
                entry.get("molecule_synonym")
                or entry.get("synonyms")
                or entry.get("synonym")
                or ""
            )
            syn = str(syn).strip()
            if syn:
                synonyms.append(syn)
        for rec in chembl_molecule.get("compound_records") or []:
            if isinstance(rec, dict):
                cn = str(rec.get("compound_name") or "").strip()
                if cn:
                    compound_records.append(cn)
        for xref in chembl_molecule.get("cross_references") or []:
            if not isinstance(xref, dict):
                continue
            xref_name = str(xref.get("xref_name") or "").strip()
            xref_id = str(xref.get("xref_id") or "").strip()
            if xref_name and xref_id:
                cross_refs.append(f"{xref_name}:{xref_id}")
            elif xref_name:
                cross_refs.append(xref_name)
            elif xref_id:
                cross_refs.append(xref_id)

    return {
        "raw_pref_name": chembl_pref or csv_pref_name,
        "raw_synonyms": _pipe_join(synonyms),
        "raw_compound_records": _pipe_join(compound_records),
        "raw_cross_references": _pipe_join(cross_refs),
    }


def _full_text_query_term(term: str) -> str:
    """Branch B: query.term 使用引号包裹，匹配 title/summary/description 等全文。"""
    escaped = term.replace('"', '\\"')
    return f'"{escaped}"'


def _build_ctgov_branch_query_url(
    term: str,
    max_results: int,
    branch_name: str,
) -> str:
    fields = ",".join(ClinicalTrialsClient._DEFAULT_SEARCH_FIELDS)
    if branch_name == _BRANCH_INTERVENTION:
        query_field, query_value = "query.intr", term
    elif branch_name == _BRANCH_FULL_TEXT:
        query_field, query_value = "query.term", _full_text_query_term(term)
    elif branch_name == _BRANCH_BROAD_DRUG_QT_RECALL:
        query_field, query_value = "query.term", term
    else:
        raise ValueError(f"未知 CT.gov 查询分支: {branch_name}")

    params = {
        query_field: query_value,
        "pageSize": max_results,
        "fields": fields,
    }
    return f"{ClinicalTrialsClient.API_BASE_URL}/studies?{urlencode(params)}"


def _build_ctgov_query_url(term: str, max_results: int) -> str:
    """兼容旧调用：默认 intervention 分支。"""
    return _build_ctgov_branch_query_url(term, max_results, _BRANCH_INTERVENTION)


def _extract_study_nct_title(
    client: ClinicalTrialsClient,
    study: Dict[str, Any],
) -> Tuple[str, str]:
    basics = client._parse_protocol_study_basics(study)
    if basics:
        return basics.get("nct_id", ""), basics.get("title", "")
    identification = study.get("identificationModule") or {}
    return (
        str(identification.get("nctId") or "").strip(),
        str(identification.get("briefTitle") or "").strip(),
    )


def _search_ctgov_branch(
    client: ClinicalTrialsClient,
    term: str,
    max_results: int,
    branch_name: str,
) -> List[Dict[str, Any]]:
    if branch_name == _BRANCH_INTERVENTION:
        return client.search_studies_by_intervention(term, max_results=max_results)
    if branch_name == _BRANCH_FULL_TEXT:
        return client.search_studies(
            _full_text_query_term(term),
            max_results=max_results,
            query_field="query.term",
        )
    if branch_name == _BRANCH_BROAD_DRUG_QT_RECALL:
        return client.search_studies(
            term,
            max_results=max_results,
            query_field="query.term",
        )
    raise ValueError(f"未知 CT.gov 查询分支: {branch_name}")


def _summarize_branch_studies(
    client: ClinicalTrialsClient,
    studies: List[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, str], Dict[str, Dict[str, Any]]]:
    nct_order: List[str] = []
    nct_to_title: Dict[str, str] = {}
    nct_to_protocol: Dict[str, Dict[str, Any]] = {}

    for study in studies:
        nct_id, title = _extract_study_nct_title(client, study)
        if not nct_id or nct_id in nct_to_title:
            continue
        nct_to_title[nct_id] = title
        nct_to_protocol[nct_id] = study
        nct_order.append(nct_id)

    return nct_order, nct_to_title, nct_to_protocol


def _strict_term_drug_alignment(
    protocol_study: Dict[str, Any],
    raw_study: Dict[str, Any] | None,
    alignment_terms: List[str],
) -> Dict[str, Any]:
    """query.term 召回 study 的严格 drug alignment（仅指定 intervention/arm/results 字段）。"""
    raw = raw_study if raw_study else {"protocolSection": protocol_study}
    fields = extract_trial_drug_fields(raw)
    matched_fields: List[str] = []

    protocol = protocol_study or raw.get("protocolSection") or {}
    arms_mod = protocol.get("armsInterventionsModule") or {}

    for iv in arms_mod.get("interventions") or []:
        if not isinstance(iv, dict):
            continue
        if match_terms_in_text(alignment_terms, str(iv.get("name") or "")):
            matched_fields.append("interventions.name")
        for other in iv.get("otherNames") or []:
            if match_terms_in_text(alignment_terms, str(other or "")):
                matched_fields.append("interventions.otherNames")

    for ag in arms_mod.get("armGroups") or []:
        if not isinstance(ag, dict):
            continue
        if match_terms_in_text(alignment_terms, str(ag.get("label") or "")):
            matched_fields.append("armGroups.label")
        if match_terms_in_text(alignment_terms, str(ag.get("description") or "")):
            matched_fields.append("armGroups.description")

    for grp in fields.get("results_groups") or []:
        if match_terms_in_text(alignment_terms, str(grp.get("title") or "")):
            matched_fields.append("resultsGroups.title")
        if match_terms_in_text(alignment_terms, str(grp.get("description") or "")):
            matched_fields.append("resultsGroups.description")

    matched_fields = list(dict.fromkeys(matched_fields))
    title_summary = f"{fields.get('title', '')} {fields.get('summary', '')}".strip()
    title_only = bool(match_terms_in_text(alignment_terms, title_summary)) and not matched_fields

    return {
        "pass": bool(matched_fields),
        "fields_matched": matched_fields,
        "title_only": title_only,
    }


def _apply_title_only_weak_override(enriched: Dict[str, Any]) -> Dict[str, Any]:
    """title/summary 命中但 intervention/arm/results 未命中 → weak，不进 primary。"""
    align = dict(enriched.get("drug_trial_alignment") or {})
    if align.get("drug_match_level") != "weak":
        return enriched
    if align.get("target_drug_in_intervention") or align.get("target_drug_in_arm_group"):
        return enriched
    if align.get("target_drug_in_results_group"):
        return enriched

    align["evidence_attribution_level"] = "not_attributable"
    enriched = dict(enriched)
    enriched["drug_trial_alignment"] = align
    tier = str(enriched.get("evidence_tier") or "")
    if tier.startswith("direct_qt"):
        enriched["evidence_tier"] = "manual_review_required"
    return enriched


def _is_phase4_intr_only(phase_name: Any, max_phase: Any = None) -> bool:
    phase = str(phase_name or "").strip()
    if phase in {"Approved", "Phase IV"}:
        return True
    if max_phase is None or (isinstance(max_phase, float) and pd.isna(max_phase)):
        return False
    try:
        return float(max_phase) == 4.0
    except (TypeError, ValueError):
        return False


def _resolve_search_strategy(phase_name: Any, max_phase: Any = None) -> str:
    """按分子 phase 决定 CT.gov 查询策略。"""
    if _is_phase4_intr_only(phase_name, max_phase):
        return "phase4_intr_only"
    phase = str(phase_name or "").strip()
    if phase == "Phase III":
        return "intr_primary_term_supplement"
    if phase == "Phase II":
        return "intr_and_term"
    return "intr_primary_term_supplement"


def _should_run_term_search(strategy: str, intr_hit_count: int) -> bool:
    if strategy == "phase4_intr_only":
        return False
    return strategy in {"intr_primary_term_supplement", "intr_and_term"}


def _add_intr_candidates(
    intr_ncts: List[str],
    intr_titles: Dict[str, str],
    intr_protocols: Dict[str, Dict[str, Any]],
    *,
    term: str,
    nct_to_title: Dict[str, str],
    nct_to_protocol: Dict[str, Dict[str, Any]],
    nct_to_meta: Dict[str, Dict[str, Any]],
    nct_order: List[str],
    kept_for_term: List[str],
) -> None:
    for nct_id in intr_ncts:
        kept_for_term.append(nct_id)
        if nct_id in nct_to_title:
            continue
        nct_to_title[nct_id] = intr_titles[nct_id]
        nct_to_protocol[nct_id] = intr_protocols[nct_id]
        nct_to_meta[nct_id] = {
            "selected_branch": _BRANCH_INTERVENTION,
            "query_term": term,
            "alignment_fields_matched": ["query.intr"],
        }
        nct_order.append(nct_id)


def _add_term_candidates(
    client: ClinicalTrialsClient,
    term_ncts: List[str],
    term_titles: Dict[str, str],
    term_protocols: Dict[str, Dict[str, Any]],
    *,
    term: str,
    alignment_terms: List[str],
    nct_to_title: Dict[str, str],
    nct_to_protocol: Dict[str, Dict[str, Any]],
    nct_to_meta: Dict[str, Dict[str, Any]],
    nct_order: List[str],
    kept_for_term: List[str],
    all_fields_matched: List[str],
) -> int:
    """严格 alignment 后合并 term 分支候选；返回新保留 study 数。"""
    added = 0
    for nct_id in term_ncts:
        if nct_id in nct_to_title:
            continue
        protocol_study = term_protocols[nct_id]
        raw_study = client.get_study_by_nct_id(nct_id) or {
            "protocolSection": protocol_study
        }
        align = _strict_term_drug_alignment(protocol_study, raw_study, alignment_terms)
        if align["title_only"] or not align["pass"]:
            continue

        kept_for_term.append(nct_id)
        all_fields_matched.extend(align["fields_matched"])
        nct_to_title[nct_id] = term_titles[nct_id]
        nct_to_protocol[nct_id] = protocol_study
        nct_to_meta[nct_id] = {
            "selected_branch": _BRANCH_FULL_TEXT,
            "query_term": term,
            "alignment_fields_matched": align["fields_matched"],
        }
        nct_order.append(nct_id)
        added += 1
    return added


def _resolve_selected_branch(
    *,
    intr_kept: int,
    term_added: int,
    term_hit_count: int,
) -> str:
    if intr_kept > 0 and term_added > 0:
        return "both"
    if intr_kept > 0:
        return _BRANCH_INTERVENTION
    if term_added > 0:
        return _BRANCH_FULL_TEXT
    if term_hit_count > 0:
        return "full_text_rejected"
    return "none"


def _make_term_query_debug_row(
    *,
    row_idx: int,
    chembl_id: str,
    phase_name: Any,
    selected_search_strategy: str,
    query_term: str,
    query_term_source: str,
    intr_hit_count: int,
    term_hit_count: int,
    selected_branch: str,
    alignment_pass: bool,
    alignment_fields_matched: str,
    query_urls: List[str],
    kept_nct_ids: List[str],
    nct_to_title: Dict[str, str],
) -> Dict[str, Any]:
    first5 = kept_nct_ids[:5]
    return {
        "row_idx": row_idx,
        "molecule_chembl_id": chembl_id,
        "phase_name": phase_name,
        "selected_search_strategy": selected_search_strategy,
        "query_term": query_term,
        "query_term_source": query_term_source,
        "intr_hit_count": intr_hit_count,
        "term_hit_count": term_hit_count,
        "selected_branch": selected_branch,
        "alignment_pass": alignment_pass,
        "alignment_fields_matched": alignment_fields_matched,
        "query_url": _pipe_join(query_urls),
        "raw_hit_count": len(kept_nct_ids),
        "first_5_nct_ids": _pipe_join(first5),
        "first_5_titles": _pipe_join([nct_to_title.get(n, "") for n in first5]),
    }



def _controlled_broad_query_terms(query_terms: List[str]) -> List[str]:
    """Build conservative drug+QT full-text queries."""
    out: List[str] = []
    seen: set[str] = set()
    drug_terms: List[str] = []
    for term in query_terms:
        t = str(term or "").strip()
        if len(t) < 3:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        drug_terms.append(t)
        if len(drug_terms) >= 8:
            break
    for drug_term in drug_terms:
        escaped = drug_term.replace('"', '\\"')
        for qt_term in _CONTROLLED_QT_RECALL_TERMS:
            out.append(f'"{escaped}" "{qt_term}"')
    return out


def _add_broad_drug_qt_candidates(
    client: ClinicalTrialsClient,
    *,
    query_terms: List[str],
    max_results: int,
    nct_to_title: Dict[str, str],
    nct_to_protocol: Dict[str, Dict[str, Any]],
    nct_to_meta: Dict[str, Dict[str, Any]],
    nct_order: List[str],
    urls: List[str],
) -> Dict[str, Any]:
    """
    Add controlled broad drug+QT candidates.

    Candidate is retained only when full-study parsing finds QT-specific
    protocol/results evidence. This branch is recall-only and does not imply Primary.
    """
    added = 0
    raw_hit_count = 0
    first_added: List[str] = []
    for broad_query in _controlled_broad_query_terms(query_terms):
        url = _build_ctgov_branch_query_url(broad_query, max_results, _BRANCH_BROAD_DRUG_QT_RECALL)
        urls.append(url)
        studies = _search_ctgov_branch(client, broad_query, max_results, _BRANCH_BROAD_DRUG_QT_RECALL)
        raw_hit_count += len(studies)
        ncts, titles, protocols = _summarize_branch_studies(client, studies)
        for nct_id in ncts:
            if nct_id in nct_to_title:
                continue
            protocol = protocols[nct_id]
            raw = client.get_study_by_nct_id(nct_id) or {"protocolSection": protocol}
            results_section = client.parse_results_section(raw)
            protocol_stats = classify_protocol_outcomes_module(protocol.get("outcomesModule") or {})
            title_summary = " ".join([
                str((protocol.get("identificationModule") or {}).get("briefTitle") or ""),
                str((protocol.get("descriptionModule") or {}).get("briefSummary") or ""),
            ])
            title_cls = classify_outcome_text(title_summary)
            has_qt_specific = bool(
                results_section.get("has_qt_results")
                or protocol_stats.get("has_qt_specific")
                or title_cls.get("evidence_type") == "qt_specific"
            )
            if not has_qt_specific:
                continue
            nct_to_title[nct_id] = titles[nct_id]
            nct_to_protocol[nct_id] = protocol
            nct_to_meta[nct_id] = {
                "selected_branch": _BRANCH_BROAD_DRUG_QT_RECALL,
                "query_term": broad_query,
                "alignment_fields_matched": ["query.term:drug+qt"],
                "_prefetched_raw": raw,
            }
            nct_order.append(nct_id)
            first_added.append(nct_id)
            added += 1
    return {"broad_raw_hit_count": raw_hit_count, "broad_added_count": added, "broad_first_5_nct_ids": _pipe_join(first_added[:5])}

def _probe_ctgov_raw_hits(
    client: ClinicalTrialsClient,
    query_terms: List[str],
    max_results: int,
    *,
    query_term_sources: Dict[str, str] | None = None,
    row_idx: int = 0,
    chembl_id: str = "",
    phase_name: Any = "",
    max_phase: Any = None,
) -> Dict[str, Any]:
    """Layer-1: 按 phase 策略组合 query.intr / query.term。"""
    urls: List[str] = []
    branch_debug_rows: List[Dict[str, Any]] = []
    nct_to_title: Dict[str, str] = {}
    nct_to_protocol: Dict[str, Dict[str, Any]] = {}
    nct_to_meta: Dict[str, Dict[str, Any]] = {}
    nct_order: List[str] = []
    term_sources = query_term_sources or {}
    search_strategy = _resolve_search_strategy(phase_name, max_phase)

    for term in query_terms:
        if not (term or "").strip():
            continue

        term_urls: List[str] = []
        intr_url = _build_ctgov_branch_query_url(term, max_results, _BRANCH_INTERVENTION)
        term_urls.append(intr_url)
        urls.append(intr_url)

        intr_studies = _search_ctgov_branch(client, term, max_results, _BRANCH_INTERVENTION)
        intr_ncts, intr_titles, intr_protocols = _summarize_branch_studies(
            client, intr_studies
        )
        intr_hit_count = len(intr_ncts)

        kept_for_term: List[str] = []
        all_fields_matched: List[str] = []
        term_titles: Dict[str, str] = {}
        term_hit_count = 0
        term_added = 0
        intr_kept_before = len(nct_order)

        _add_intr_candidates(
            intr_ncts,
            intr_titles,
            intr_protocols,
            term=term,
            nct_to_title=nct_to_title,
            nct_to_protocol=nct_to_protocol,
            nct_to_meta=nct_to_meta,
            nct_order=nct_order,
            kept_for_term=kept_for_term,
        )
        intr_kept = len(nct_order) - intr_kept_before

        if _should_run_term_search(search_strategy, intr_hit_count):
            term_url = _build_ctgov_branch_query_url(term, max_results, _BRANCH_FULL_TEXT)
            term_urls.append(term_url)
            urls.append(term_url)

            term_studies = _search_ctgov_branch(client, term, max_results, _BRANCH_FULL_TEXT)
            term_ncts, term_titles, term_protocols = _summarize_branch_studies(
                client, term_studies
            )
            term_hit_count = len(term_ncts)
            term_added = _add_term_candidates(
                client,
                term_ncts,
                term_titles,
                term_protocols,
                term=term,
                alignment_terms=query_terms,
                nct_to_title=nct_to_title,
                nct_to_protocol=nct_to_protocol,
                nct_to_meta=nct_to_meta,
                nct_order=nct_order,
                kept_for_term=kept_for_term,
                all_fields_matched=all_fields_matched,
            )

        selected_branch = _resolve_selected_branch(
            intr_kept=intr_kept,
            term_added=term_added,
            term_hit_count=term_hit_count,
        )

        branch_debug_rows.append(
            _make_term_query_debug_row(
                row_idx=row_idx,
                chembl_id=chembl_id,
                phase_name=phase_name,
                selected_search_strategy=search_strategy,
                query_term=term,
                query_term_source=term_sources.get(term, ""),
                intr_hit_count=intr_hit_count,
                term_hit_count=term_hit_count,
                selected_branch=selected_branch,
                alignment_pass=bool(kept_for_term),
                alignment_fields_matched=_pipe_join(list(dict.fromkeys(all_fields_matched))),
                query_urls=term_urls,
                kept_nct_ids=kept_for_term,
                nct_to_title={**intr_titles, **term_titles},
            )
        )

    # Controlled broad drug+QT recall: strict enough for candidate generation,
    # not sufficient for Primary evidence. This is enabled for all phases but
    # requires QT-specific protocol/results evidence after full-study fetch.
    broad_stats = _add_broad_drug_qt_candidates(
        client,
        query_terms=query_terms,
        max_results=max_results,
        nct_to_title=nct_to_title,
        nct_to_protocol=nct_to_protocol,
        nct_to_meta=nct_to_meta,
        nct_order=nct_order,
        urls=urls,
    )

    first5 = nct_order[:5]
    count = len(nct_order)
    first5_titles = _pipe_join([nct_to_title[n] for n in first5])
    return {
        "ctgov_query_urls": _pipe_join(urls),
        "ctgov_raw_response_count": count,
        "ctgov_first_5_nct_ids": _pipe_join(first5),
        "ctgov_first_5_titles": first5_titles,
        "drug_only_raw_hit_count": count,
        "drug_only_first_5_nct_ids": _pipe_join(first5),
        "drug_only_first_5_titles": first5_titles,
        "broad_raw_hit_count": broad_stats.get("broad_raw_hit_count", 0),
        "broad_added_count": broad_stats.get("broad_added_count", 0),
        "broad_first_5_nct_ids": broad_stats.get("broad_first_5_nct_ids", ""),
        "_drug_only_candidates": {
            nct: {
                "protocol": nct_to_protocol[nct],
                "title": nct_to_title[nct],
                **nct_to_meta.get(nct, {}),
            }
            for nct in nct_order
        },
        "_branch_debug_rows": branch_debug_rows,
        "_search_strategy": search_strategy,
    }


def _protocol_local_qt_signal(protocol_study: Dict[str, Any]) -> bool:
    identification = protocol_study.get("identificationModule") or {}
    description = protocol_study.get("descriptionModule") or {}
    title = str(identification.get("briefTitle") or "").strip()
    summary = str(description.get("briefSummary") or "").strip()
    title_cls = classify_outcome_text(f"{title} {summary}".strip())
    if title_cls.get("evidence_type") in _LOCAL_QT_EVIDENCE_TYPES:
        return True
    protocol_outcomes = classify_protocol_outcomes_module(
        protocol_study.get("outcomesModule") or {}
    )
    return bool(
        protocol_outcomes.get("has_qt_specific")
        or protocol_outcomes.get("has_ecg_conduction")
        or protocol_outcomes.get("has_ecg_broad")
        or protocol_outcomes.get("has_cardiac_ae")
    )


def _results_local_qt_signal(results_section: Dict[str, Any]) -> bool:
    return bool(
        results_section.get("has_qt_results")
        or results_section.get("has_ecg_conduction_results")
        or results_section.get("has_ecg_broad_results")
        or results_section.get("has_cardiac_ae_results")
    )


def _adverse_events_local_qt_signal(raw_study: Dict[str, Any]) -> bool:
    ae_mod = (raw_study.get("resultsSection") or {}).get("adverseEventsModule") or {}
    if not ae_mod:
        return False
    texts: List[str] = []
    for section in ("seriousEvents", "otherEvents"):
        for group in ae_mod.get(section) or []:
            if not isinstance(group, dict):
                continue
            for ev in group.get("events") or []:
                if not isinstance(ev, dict):
                    continue
                for key in ("term", "organSystem", "sourceVocabulary"):
                    val = str(ev.get(key) or "").strip()
                    if val:
                        texts.append(val)
    for text in texts:
        if classify_outcome_text(text).get("evidence_type") in _LOCAL_QT_EVIDENCE_TYPES:
            return True
    return False


def _local_qt_signal(
    client: ClinicalTrialsClient,
    protocol_study: Dict[str, Any],
    raw_study: Dict[str, Any],
) -> bool:
    """Layer-2: 对 drug-only candidate 做本地 QT 筛选（protocol + results + AE）。"""
    if _protocol_local_qt_signal(protocol_study):
        return True
    results_section = client.parse_results_section(raw_study)
    if _results_local_qt_signal(results_section):
        return True
    return _adverse_events_local_qt_signal(raw_study)


def _pick_qt_outcome_measure(protocol_outcomes: Dict[str, Any]) -> str:
    return pick_protocol_qt_outcome_measure(protocol_outcomes)


def _build_trial_from_local_qt_candidate(
    client: ClinicalTrialsClient,
    protocol_study: Dict[str, Any],
    raw_study: Dict[str, Any],
) -> Dict[str, Any] | None:
    """将本地 QT 命中 trial 转为 enrich_trial_evidence 可消费的 dict（复用 client 解析）。"""
    strict = client._parse_protocol_study(protocol_study)
    if strict:
        if raw_study and not strict.get("_prefetched_raw"):
            strict["_prefetched_raw"] = raw_study
        return strict

    basics = client._parse_protocol_study_basics(protocol_study)
    if not basics:
        return None

    recall = client._build_results_recall_trial(basics, raw_study)
    if recall:
        return recall

    protocol_outcomes = classify_protocol_outcomes_module(
        protocol_study.get("outcomesModule") or {}
    )
    title_cls = classify_outcome_text(
        f"{basics.get('title', '')} {basics.get('summary', '')}".strip()
    )
    results_section = client.parse_results_section(raw_study)
    protocol_qt_hit = _protocol_local_qt_signal(protocol_study)
    results_qt_hit = _results_local_qt_signal(results_section) or _adverse_events_local_qt_signal(
        raw_study
    )

    return {
        **basics,
        "qt_related": True,
        "qt_related_title": title_cls.get("evidence_type") == "qt_specific",
        "qt_related_outcome": protocol_outcomes.get("has_qt_specific", False),
        "qt_outcome_measure": _pick_qt_outcome_measure(protocol_outcomes),
        "search_branch": "strict_protocol_qt" if protocol_qt_hit else "results_recall",
        "protocol_qt_hit": protocol_qt_hit,
        "results_qt_hit": results_qt_hit,
        "protocol_outcomes": protocol_outcomes,
        "title_outcome_classification": title_cls,
        "_prefetched_raw": raw_study,
        "_clinical_results_section": results_section,
    }


def _drug_only_then_local_qt_trials(
    client: ClinicalTrialsClient,
    drug_only_candidates: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Layer-1 drug-only 候选 → Layer-2 本地 QT 筛选 → trials 列表。
    """
    qt_trials: List[Dict[str, Any]] = []

    for nct_id, cand in drug_only_candidates.items():
        protocol_study = cand["protocol"]
        raw_study = cand.get("_prefetched_raw") or client.get_study_by_nct_id(nct_id) or {}
        if not _local_qt_signal(client, protocol_study, raw_study):
            continue

        trial = _build_trial_from_local_qt_candidate(client, protocol_study, raw_study)
        if trial:
            if cand.get("selected_branch") == _BRANCH_BROAD_DRUG_QT_RECALL:
                trial["search_branch"] = _BRANCH_BROAD_DRUG_QT_RECALL
                trial["_broad_recall_query"] = cand.get("query_term", "")
            qt_trials.append(trial)

    return qt_trials


def _make_query_debug_row(
    row_idx: int,
    chembl_id: str,
    phase_name: Any,
) -> Dict[str, Any]:
    return {
        "row_idx": row_idx,
        "molecule_chembl_id": chembl_id,
        "pref_name": "",
        "phase_name": phase_name,
        "selected_search_strategy": "",
        "query_terms_used": "",
        "intr_hit_count": 0,
        "term_hit_count": 0,
        "selected_branch": "",
        "raw_pref_name": "",
        "raw_synonyms": "",
        "raw_compound_records": "",
        "raw_cross_references": "",
        "filtered_query_terms": "",
        "query_term_sources": "",
        "query_terms_count": 0,
        "query_terms_used": "",
        "skip_reason": "",
        "ctgov_query_urls": "",
        "ctgov_raw_response_count": 0,
        "ctgov_first_5_nct_ids": "",
        "ctgov_first_5_titles": "",
        "drug_only_raw_hit_count": 0,
        "drug_only_first_5_nct_ids": "",
        "drug_only_first_5_titles": "",
        "qt_after_local_filter_count": 0,
        "final_qt_hits": 0,
        "broad_raw_hit_count": 0,
        "broad_added_count": 0,
        "broad_first_5_nct_ids": "",
    }


def _assign_query_debug_skip_reason(debug_row: Dict[str, Any]) -> None:
    if not debug_row.get("query_terms_used"):
        debug_row["skip_reason"] = "no_searchable_identity"
    elif int(debug_row.get("drug_only_raw_hit_count") or 0) == 0:
        debug_row["skip_reason"] = "ctgov_no_raw_hits"
    elif int(debug_row.get("qt_after_local_filter_count") or 0) == 0:
        debug_row["skip_reason"] = "raw_hits_but_no_qt"
    else:
        debug_row["skip_reason"] = ""


def _apply_phase_filters(
    df: pd.DataFrame,
    *,
    phases: List[str] | None,
    exclude_approved: bool,
    max_phases: List[float] | None,
) -> pd.DataFrame:
    """按研发阶段筛选；在 offset / limit 之前执行。"""
    out = df.copy()

    if phases:
        allowed = set(_normalize_phase_names(phases))
        out = out[out["phase_name"].isin(allowed)]

    if exclude_approved:
        out = out[out["phase_name"] != "Approved"]

    if max_phases:
        allowed_mp = {float(p) for p in max_phases}
        out = out[out["max_phase"].astype(float).isin(allowed_mp)]

    return out.reset_index(drop=True)


def _load_molecules(
    csv_path: Path,
    limit: int,
    offset: int = 0,
    *,
    phases: List[str] | None = None,
    exclude_approved: bool = False,
    max_phases: List[float] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df = _apply_phase_filters(
        df,
        phases=phases,
        exclude_approved=exclude_approved,
        max_phases=max_phases,
    )

    if offset > 0:
        df = df.iloc[offset:].reset_index(drop=True)

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
        "results_has_posted_results": rs.get("has_posted_results", False),
        "results_has_results_section": rs.get("has_results_section", False),
        "results_has_outcome_measures": rs.get("has_results_outcome_measures", False),
        "results_has_qt_results": rs.get("has_qt_results", False),
        "results_has_ecg_conduction_results": rs.get("has_ecg_conduction_results", False),
        "results_has_ecg_broad_results": rs.get("has_ecg_broad_results", False),
        "results_has_cardiac_ae_results": rs.get("has_cardiac_ae_results", False),
        "results_qt_measure_count": len(rs.get("qt_result_measures") or []),
        "results_ecg_conduction_measure_count": len(
            rs.get("ecg_conduction_result_measures") or []
        ),
        "results_ecg_broad_measure_count": len(
            rs.get("ecg_broad_result_measures") or []
        ),
        "results_cardiac_ae_measure_count": len(
            rs.get("cardiac_ae_result_measures") or []
        ),
        "results_has_adverse_events": rs.get("has_adverse_events", False),
        "results_has_more_info": rs.get("has_more_info", False),
        "results_summary": rs.get("result_summary", ""),
    }


def _flatten_alignment(
    align: Dict[str, Any],
    qt_attr: Dict[str, Any],
    evidence_tier: str,
    evidence_tier_reason: str = "",
) -> Dict[str, Any]:
    """平铺 drug_trial_alignment / qt_result_attribution 标量字段到 CSV。"""
    return {
        "align_target_in_intervention": align.get("target_drug_in_intervention", False),
        "align_target_in_arm_group": align.get("target_drug_in_arm_group", False),
        "align_target_in_results_group": align.get("target_drug_in_results_group", False),
        "align_target_drug_role": align.get("target_drug_role", ""),
        "align_drug_match_level": align.get("drug_match_level", ""),
        "align_evidence_attribution": align.get("evidence_attribution_level", ""),
        "align_matched_terms": "|".join(align.get("matched_drug_terms") or []),
        "align_intervention_names": "|".join(align.get("intervention_names") or []),
        "align_summary": align.get("alignment_summary", ""),
        "qt_attr_for_target_drug": qt_attr.get("qt_result_for_target_drug", False),
        "qt_attr_comparator_only": qt_attr.get("qt_result_for_comparator_only", False),
        "qt_attr_target_group_ids": "|".join(qt_attr.get("target_group_ids") or []),
        "qt_attr_summary": qt_attr.get("summary", ""),
        "evidence_tier": evidence_tier,
        "evidence_tier_reason": evidence_tier_reason,
    }


def _build_trial_entry(enriched: Dict[str, Any]) -> Dict[str, Any]:
    """Level-2 trial 对象，内嵌 alignment / results / attribution。"""
    cred = enriched.get("research_institution_credibility") or {}

    return {
        "nct_id": enriched.get("nct_id", ""),
        "status": enriched.get("status", ""),
        "title": enriched.get("title", ""),
        "summary": enriched.get("summary", ""),
        "sponsor_name": enriched.get("sponsor_name", ""),
        "sponsor_class": enriched.get("sponsor_class", ""),
        "institution_credibility_level": enriched.get(
            "institution_credibility_level", "unknown"
        ),
        "qt_related": enriched.get("qt_related"),
        "qt_related_title": enriched.get("qt_related_title"),
        "qt_related_outcome": enriched.get("qt_related_outcome"),
        "qt_outcome_measure": enriched.get("qt_outcome_measure", ""),
        "search_branch": enriched.get("search_branch", ""),
        "protocol_qt_hit": enriched.get("protocol_qt_hit", False),
        "results_qt_hit": enriched.get("results_qt_hit", False),
        "research_institution_credibility": cred,
        "drug_trial_alignment": enriched.get("drug_trial_alignment") or {},
        "clinical_results_section": enriched.get("clinical_results_section") or {},
        "qt_result_attribution": enriched.get("qt_result_attribution") or {},
        "evidence_tier": enriched.get("evidence_tier", "manual_review_required"),
        "evidence_tier_reason": enriched.get("evidence_tier_reason", ""),
    }


def _scan_one(
    client: ClinicalTrialsClient,
    enricher: ChemblDrugNameEnricher,
    row: pd.Series,
    max_results: int,
    row_idx: int,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    """
    Returns:
        molecule_entry : Level-1 分子对象（含 trials 列表）
        trial_rows     : 逐条试验明细，写 CSV
        raw_studies    : get_study_by_nct_id 原始回传，写 raw_study JSONL
    """
    chembl_id = str(row.get("molecule_chembl_id", "")).strip()
    raw_pref = row.get("pref_name", "")
    drug_name = "" if pd.isna(raw_pref) else str(raw_pref).strip()

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
    branch_debug_rows: List[Dict[str, Any]] = []
    phase_name = row.get("phase_name")

    debug_row = _make_query_debug_row(row_idx, chembl_id, phase_name)
    debug_row["pref_name"] = drug_name

    chembl_enriched = enricher.enrich(chembl_id, pref_name_fallback=drug_name)
    chembl_molecule: Dict[str, Any] | None = None
    if chembl_id:
        chembl_molecule, _ = enricher.fetch_molecule(chembl_id)

    debug_row.update(_extract_raw_chembl_debug_fields(drug_name, chembl_molecule))

    query_terms, query_term_sources = _build_query_terms(
        drug_name, chembl_molecule, chembl_enriched
    )
    terms_joined = _pipe_join(query_terms)
    debug_row["filtered_query_terms"] = terms_joined
    debug_row["query_terms_used"] = terms_joined
    debug_row["query_term_sources"] = _format_query_term_sources(
        query_terms, query_term_sources
    )
    debug_row["query_terms_count"] = len(query_terms)

    if not query_terms:
        molecule_entry["status"] = "skipped"
        molecule_entry["error"] = "no_searchable_identity"
        _assign_query_debug_skip_reason(debug_row)
        return molecule_entry, trial_rows, raw_studies, debug_row, branch_debug_rows

    ctgov_probe = _probe_ctgov_raw_hits(
        client,
        query_terms,
        max_results,
        query_term_sources=query_term_sources,
        row_idx=row_idx,
        chembl_id=chembl_id,
        phase_name=phase_name,
        max_phase=row.get("max_phase"),
    )
    branch_debug_rows = ctgov_probe.pop("_branch_debug_rows")
    drug_only_candidates = ctgov_probe.pop("_drug_only_candidates")
    search_strategy = ctgov_probe.pop("_search_strategy", "")
    debug_row["selected_search_strategy"] = search_strategy
    debug_row.update(ctgov_probe)

    if search_strategy == "phase4_intr_only":
        debug_row["term_hit_count"] = 0
        debug_row["intr_hit_count"] = int(debug_row.get("drug_only_raw_hit_count") or 0)
        debug_row["selected_branch"] = (
            _BRANCH_INTERVENTION if debug_row["intr_hit_count"] > 0 else "none"
        )
    elif branch_debug_rows:
        debug_row["intr_hit_count"] = max(
            int(r.get("intr_hit_count") or 0) for r in branch_debug_rows
        )
        debug_row["term_hit_count"] = sum(
            int(r.get("term_hit_count") or 0) for r in branch_debug_rows
        )
        branches = {str(r.get("selected_branch") or "") for r in branch_debug_rows}
        if "both" in branches:
            debug_row["selected_branch"] = "both"
        elif _BRANCH_FULL_TEXT in branches:
            debug_row["selected_branch"] = _BRANCH_FULL_TEXT
        elif _BRANCH_INTERVENTION in branches:
            debug_row["selected_branch"] = _BRANCH_INTERVENTION
        else:
            debug_row["selected_branch"] = "none"

    drug_name_set = build_drug_name_set(drug_name, enriched=chembl_enriched)
    alignment_terms = _normalize_match_terms(list(query_terms))
    drug_name_set["strong_match_terms"] = alignment_terms
    drug_name_set["_strong_match_terms"] = alignment_terms

    molecule_entry["drug_name_set"] = {
        "molecule_chembl_id": chembl_enriched.get("molecule_chembl_id", chembl_id),
        "pref_name": chembl_enriched.get("pref_name", drug_name),
        "chembl_pref_name": chembl_enriched.get("chembl_pref_name", ""),
        "strong_match_terms": chembl_enriched.get("strong_match_terms", []),
        "related_match_terms": chembl_enriched.get("related_match_terms", []),
        "weak_terms": chembl_enriched.get("weak_terms", []),
        "parent_molecule_chembl_id": chembl_enriched.get(
            "parent_molecule_chembl_id", ""
        ),
        "parent_pref_name": chembl_enriched.get("parent_pref_name", ""),
        "enrichment_status": chembl_enriched.get("enrichment_status", "ok"),
    }

    molecule_entry["recall_audit"] = {
        "related_match_terms": chembl_enriched.get("related_match_terms", []),
        "weak_terms": chembl_enriched.get("weak_terms", []),
        "recall_audit_terms": chembl_enriched.get("recall_audit_terms", []),
    }

    molecule_entry["drug_only_raw_count"] = len(drug_only_candidates)
    for nct_id, cand in drug_only_candidates.items():
        raw_studies.append(
            {
                "molecule_chembl_id": chembl_id,
                "pref_name": drug_name,
                "nct_id": nct_id,
                "stage": "drug_only_candidate",
                "raw_response": {"protocolSection": cand["protocol"]},
            }
        )

    try:
        trials = _drug_only_then_local_qt_trials(
            client,
            drug_only_candidates,
        )

        molecule_entry["qt_after_local_filter_count"] = len(trials)
        molecule_entry["qt_trials_count"] = len(trials)
        debug_row["qt_after_local_filter_count"] = len(trials)
        debug_row["final_qt_hits"] = len(trials)

        molecule_entry["qt_title_hits"] = sum(
            1 for t in trials if t.get("qt_related_title")
        )
        molecule_entry["qt_outcome_hits"] = sum(
            1 for t in trials if t.get("qt_related_outcome")
        )
        molecule_entry["example_nct_ids"] = [
            t.get("nct_id", "") for t in trials[:3] if t.get("nct_id")
        ]

        for t in trials:
            nct_id = t.get("nct_id", "")
            cred = t.get("research_institution_credibility") or {}

            try:
                raw: Dict[str, Any] = {}
                if nct_id:
                    raw = (
                        t.get("_prefetched_raw")
                        or client.get_study_by_nct_id(nct_id)
                        or {}
                    )
                    raw_studies.append(
                        {
                            "molecule_chembl_id": chembl_id,
                            "pref_name": drug_name,
                            "nct_id": nct_id,
                            "stage": "qt_after_local_filter",
                            "raw_response": raw,
                        }
                    )

                enriched_trial = client.enrich_trial_evidence(t, raw, drug_name_set)
                enriched_trial = _apply_title_only_weak_override(enriched_trial)
                results_section = enriched_trial.get("clinical_results_section") or {}
                alignment = enriched_trial.get("drug_trial_alignment") or {}
                qt_attr = enriched_trial.get("qt_result_attribution") or {}
                evidence_tier = enriched_trial.get(
                    "evidence_tier", "manual_review_required"
                )
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

                if evidence_tier in {
                    "false_positive_mapping",
                    "rescue_or_background_only",
                }:
                    molecule_entry["false_positive_hits"] += 1

                trial_entry = _build_trial_entry(enriched_trial)
                molecule_entry["trials"].append(trial_entry)

                detail_row: Dict[str, Any] = {
                    "molecule_chembl_id": chembl_id,
                    "pref_name": drug_name,
                    "max_phase": row.get("max_phase"),
                    "phase_name": row.get("phase_name"),
                    "nct_id": nct_id,
                    "trial_status": t.get("status", ""),
                    "title": t.get("title", ""),
                    "summary": t.get("summary", ""),
                    "search_branch": search_branch,
                    "protocol_qt_hit": enriched_trial.get("protocol_qt_hit", False),
                    "results_qt_hit": enriched_trial.get("results_qt_hit", False),
                    "qt_related_title": t.get("qt_related_title"),
                    "qt_related_outcome": t.get("qt_related_outcome"),
                    "qt_outcome_measure": t.get("qt_outcome_measure", ""),
                    "error": "",
                }

                detail_row.update(_flatten_credibility(cred))
                detail_row.update(_flatten_results_section(results_section))
                detail_row.update(
                    _flatten_alignment(
                        alignment,
                        qt_attr,
                        evidence_tier,
                        evidence_tier_reason,
                    )
                )

                trial_rows.append(detail_row)
            except Exception as trial_err:
                trial_entry = {
                    "nct_id": nct_id,
                    "status": t.get("status", ""),
                    "title": t.get("title", ""),
                    "summary": t.get("summary", ""),
                    "error": str(trial_err),
                }
                molecule_entry["trials"].append(trial_entry)

                trial_rows.append(
                    {
                        "molecule_chembl_id": chembl_id,
                        "pref_name": drug_name,
                        "max_phase": row.get("max_phase"),
                        "phase_name": row.get("phase_name"),
                        "nct_id": nct_id,
                        "trial_status": t.get("status", ""),
                        "title": t.get("title", ""),
                        "summary": t.get("summary", ""),
                        "error": str(trial_err),
                    }
                )

    except Exception as e:
        molecule_entry["status"] = "error"
        molecule_entry["error"] = str(e)

    _assign_query_debug_skip_reason(debug_row)
    return molecule_entry, trial_rows, raw_studies, debug_row, branch_debug_rows


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
    """组装三层嵌套 JSON：metadata → molecules → trials。"""
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
    offset: int = 0,
    phase_filter: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    ok_rows = [m for m in molecules if m["status"] == "ok"]
    with_trials = sum(1 for m in ok_rows if m["qt_trials_count"] > 0)
    errors = sum(1 for m in molecules if m["status"] == "error")
    skipped = sum(1 for m in molecules if m["status"] == "skipped")
    trial_counts = Counter(m["qt_trials_count"] for m in ok_rows)

    sponsor_class_dist: Counter = Counter()
    credibility_level_dist: Counter = Counter()
    sponsor_name_top: Counter = Counter()

    for tr in trial_rows:
        cls_ = (tr.get("cred_sponsor_class") or "UNKNOWN").strip() or "UNKNOWN"
        lvl = (tr.get("cred_level") or "unknown").strip() or "unknown"
        name = (tr.get("cred_sponsor_name") or "").strip()

        sponsor_class_dist[cls_] += 1
        credibility_level_dist[lvl] += 1

        if name:
            sponsor_name_top[name] += 1

    rs_list = list(_iter_results_sections(molecules))

    results_stats: Dict[str, Any] = {
        "trials_with_posted_results": sum(
            1 for r in rs_list if r.get("has_posted_results")
        ),
        "trials_with_results_section": sum(
            1 for r in rs_list if r.get("has_results_section")
        ),
        "trials_with_outcome_measures": sum(
            1 for r in rs_list if r.get("has_results_outcome_measures")
        ),
        "trials_with_qt_result_measures": sum(
            1 for r in rs_list if r.get("has_qt_results")
        ),
        "trials_with_adverse_events": sum(
            1 for r in rs_list if r.get("has_adverse_events")
        ),
        "trials_with_more_info": sum(
            1 for r in rs_list if r.get("has_more_info")
        ),
        "total_qt_result_measure_objects": sum(
            len(r.get("qt_result_measures") or []) for r in rs_list
        ),
    }

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

    _PRIMARY_TIERS = {
        "direct_qt_actual_result",
        "direct_qt_protocol_outcome",
        "results_only_qt_actual_result",
    }

    _SUPPORT_TIERS = {
        "direct_ecg_conduction_evidence",
        "ecg_broad_supportive_evidence",
        "comparator_qt_evidence",
        "combination_qt_evidence",
        "combination_ecg_evidence",
        "comparator_ecg_evidence",
        "direct_qt_context",
    }

    _EXCLUDE_TIERS = {
        "rescue_or_background_only",
        "manual_review_required",
        "actual_result_needs_review",
        "combination_context_only",
        "non_qt_cardiology_endpoint",
        "non_qt_excluded",
        "false_positive_mapping",
        "recall_audit",
        "direct_actual_result_review",
    }

    alignment_stats = {
        "evidence_tier_distribution": dict(tier_dist.most_common()),
        "search_branch_distribution": dict(search_branch_dist.most_common()),
        "drug_match_level_distribution": dict(match_level_dist.most_common()),
        "evidence_attribution_distribution": dict(attribution_dist.most_common()),
        "primary_qt_evidence": {
            t: tier_dist.get(t, 0) for t in sorted(_PRIMARY_TIERS)
        },
        "primary_qt_evidence_total": sum(
            tier_dist.get(t, 0) for t in _PRIMARY_TIERS
        ),
        "supportive_evidence": {
            t: tier_dist.get(t, 0) for t in sorted(_SUPPORT_TIERS)
        },
        "supportive_evidence_total": sum(
            tier_dist.get(t, 0) for t in _SUPPORT_TIERS
        ),
        "excluded_or_review": {
            t: tier_dist.get(t, 0) for t in sorted(_EXCLUDE_TIERS)
        },
        "excluded_or_review_total": sum(
            tier_dist.get(t, 0) for t in _EXCLUDE_TIERS
        ),
        "trials_strict_protocol_qt": search_branch_dist.get("strict_protocol_qt", 0),
        "trials_results_recall_branch": search_branch_dist.get("results_recall", 0),
    }

    recall_branch_stats = {
        "molecules_with_recall_branch_hits": sum(
            1 for m in ok_rows if m.get("recall_branch_hits", 0) > 0
        ),
        "total_recall_branch_hits": sum(
            m.get("recall_branch_hits", 0) for m in ok_rows
        ),
    }

    phase_dist = Counter(
        str(m.get("phase_name") or "unknown") for m in molecules
    )

    scanned = [m for m in molecules if m.get("status") != "skipped"]
    drug_only_counts = [int(m.get("drug_only_raw_count") or 0) for m in scanned]
    qt_local_counts = [int(m.get("qt_after_local_filter_count") or 0) for m in scanned]

    drug_only_recall_stats = {
        "molecules_with_any_ctgov_studies": sum(1 for c in drug_only_counts if c > 0),
        "total_drug_only_raw_studies": sum(drug_only_counts),
        "molecules_with_drug_only_hits": sum(1 for c in drug_only_counts if c > 0),
        "molecules_with_qt_after_local_filter": sum(1 for c in qt_local_counts if c > 0),
        "total_qt_after_local_filter": sum(qt_local_counts),
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_csv),
        "requested_limit": limit,
        "requested_offset": offset,
        "phase_filter": phase_filter or {},
        "processed_phase_name_distribution": dict(phase_dist.most_common()),
        "total_rows": len(molecules),
        "ok_rows": len(ok_rows),
        "error_rows": errors,
        "skipped_rows": skipped,
        "molecules_with_qt_trials": with_trials,
        "molecules_with_qt_trials_pct": (
            round(100.0 * with_trials / len(ok_rows), 2) if ok_rows else 0.0
        ),
        "total_qt_trial_entries": sum(m["qt_trials_count"] for m in ok_rows),
        "qt_trials_count_distribution": {
            str(k): v for k, v in sorted(trial_counts.items())
        },
        "sponsor_class_distribution": dict(sponsor_class_dist.most_common()),
        "credibility_level_distribution": dict(sorted(credibility_level_dist.items())),
        "top20_sponsor_names": dict(sponsor_name_top.most_common(20)),
        "results_section_stats": results_stats,
        "alignment_stats": alignment_stats,
        "recall_branch_stats": recall_branch_stats,
        "drug_only_recall_stats": drug_only_recall_stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="对 Chembl_phase234_Data.csv 批量调用 ClinicalTrialsClient"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="输入 CSV（默认 data/Chembl_phase234_Data.csv）",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="分子级汇总 CSV 输出路径（默认按 phase 写入 ClinicalTrails/phaseX/）",
    )

    parser.add_argument(
        "--trials-output",
        type=Path,
        default=None,
        help="逐条试验明细 CSV 输出路径",
    )

    parser.add_argument(
        "--raw-jsonl",
        type=Path,
        default=None,
        help="get_study_by_nct_id 原始回传 JSONL 路径",
    )

    parser.add_argument(
        "--evidence-json",
        type=Path,
        default=None,
        help="三层嵌套 JSON 输出路径",
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="汇总 JSON 输出路径",
    )

    parser.add_argument(
        "--query-debug",
        type=Path,
        default=None,
        help="query_terms 调试 CSV 输出路径",
    )

    parser.add_argument(
        "--branch-query-debug",
        type=Path,
        default=None,
        help="每 term × 分支 CT.gov 查询调试 CSV 输出路径",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只处理 N 条；0 表示全部",
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="跳过前 N 条有效 pref_name 记录后开始处理；默认 0",
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=30,
        help="每个药名传给 search_qt_related_trials 的最大试验数",
    )

    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="ClinicalTrials API 调用间隔，单位秒",
    )

    parser.add_argument(
        "--phases",
        nargs="+",
        metavar="PHASE",
        help=(
            "只处理指定 phase_name，可多选。"
            "接受 Phase II / Phase III / Approved，"
            "也接受 2 / 3 / 4 / phase2 / phase3 / approved。"
            "例：--phases 2 3"
        ),
    )

    parser.add_argument(
        "--exclude-approved",
        action="store_true",
        help="排除 Approved 分子，只处理非 Approved 分子",
    )

    parser.add_argument(
        "--max-phase",
        nargs="+",
        type=float,
        metavar="N",
        dest="max_phase_vals",
        help="按 max_phase 数值筛选，可多值。例：--max-phase 2 3",
    )

    args = parser.parse_args()

    input_csv = args.input.resolve()

    if not input_csv.is_file():
        print(f"找不到输入文件: {input_csv}", file=sys.stderr)
        return 2

    try:
        phases_normalized = _normalize_phase_names(args.phases or []) or None
    except ValueError as exc:
        print(f"参数错误: {exc}", file=sys.stderr)
        return 2

    phase_subdir = _resolve_phase_output_subdir(
        phases_normalized,
        exclude_approved=args.exclude_approved,
    )
    default_paths = _default_output_paths(phase_subdir)

    output_csv = (args.output or default_paths["output"]).resolve()
    trials_csv = (args.trials_output or default_paths["trials_output"]).resolve()
    raw_jsonl = (args.raw_jsonl or default_paths["raw_jsonl"]).resolve()
    evidence_json = (args.evidence_json or default_paths["evidence_json"]).resolve()
    summary_json = (args.summary or default_paths["summary"]).resolve()
    query_debug_csv = (args.query_debug or default_paths["query_debug"]).resolve()
    branch_query_debug_csv = (
        args.branch_query_debug or default_paths["branch_query_debug"]
    ).resolve()

    print(
        f"[输出目录] {DEFAULT_OUTPUT_BASE / phase_subdir}",
        file=sys.stderr,
    )

    phase_filter_meta: Dict[str, Any] = {
        "phases": phases_normalized,
        "exclude_approved": args.exclude_approved,
        "max_phase_vals": args.max_phase_vals,
        "output_phase_subdir": phase_subdir,
        "output_base_dir": str(DEFAULT_OUTPUT_BASE),
    }

    df = _load_molecules(
        input_csv,
        args.limit,
        offset=args.offset,
        phases=phases_normalized,
        exclude_approved=args.exclude_approved,
        max_phases=args.max_phase_vals,
    )

    print(
        f"[过滤后] 共 {len(df)} 条分子；"
        f"phase 分布: {dict(df['phase_name'].value_counts())}",
        file=sys.stderr,
    )

    client = ClinicalTrialsClient(rate_limit=args.rate_limit)
    chembl_client = ChEMBLClient()
    enricher = ChemblDrugNameEnricher(
        chembl_client=chembl_client,
        cache_dir=DEFAULT_CHEMBL_NAME_CACHE,
    )

    molecules: List[Dict[str, Any]] = []
    trial_rows: List[Dict[str, Any]] = []
    raw_studies: List[Dict[str, Any]] = []
    query_debug_rows: List[Dict[str, Any]] = []
    branch_query_debug_rows: List[Dict[str, Any]] = []

    for row_idx, (_, row) in enumerate(
        tqdm(
            df.iterrows(),
            total=len(df),
            desc="ClinicalTrials",
            unit="mol",
            file=sys.stderr,
        )
    ):
        molecule_entry, details, raws, debug_row, branch_rows = _scan_one(
            client,
            enricher,
            row,
            max_results=args.max_results,
            row_idx=row_idx,
        )

        molecules.append(molecule_entry)
        trial_rows.extend(details)
        raw_studies.extend(raws)
        query_debug_rows.append(debug_row)
        branch_query_debug_rows.extend(branch_rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    trials_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
    evidence_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([_flatten_molecule_for_csv(m) for m in molecules]).to_csv(
        output_csv,
        index=False,
    )

    pd.DataFrame(trial_rows).to_csv(
        trials_csv,
        index=False,
    )

    query_debug_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(query_debug_rows).to_csv(
        query_debug_csv,
        index=False,
    )

    branch_query_debug_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(branch_query_debug_rows).to_csv(
        branch_query_debug_csv,
        index=False,
    )

    with open(raw_jsonl, "w", encoding="utf-8") as f:
        for item in raw_studies:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = _summarize(
        molecules,
        trial_rows,
        input_csv=input_csv,
        limit=args.limit,
        offset=args.offset,
        phase_filter=phase_filter_meta,
    )

    summary["output_csv"] = str(output_csv)
    summary["trials_csv"] = str(trials_csv)
    summary["query_debug_csv"] = str(query_debug_csv)
    summary["branch_query_debug_csv"] = str(branch_query_debug_csv)
    summary["total_branch_query_rows"] = len(branch_query_debug_rows)
    summary["raw_jsonl"] = str(raw_jsonl)
    summary["evidence_json"] = str(evidence_json)
    summary["raw_study_count"] = len(raw_studies)
    summary["total_trial_detail_rows"] = len(trial_rows)
    summary["molecule_count"] = len(molecules)

    evidence_payload = _build_evidence_json(molecules, summary)

    evidence_json.write_text(
        json.dumps(evidence_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"分子汇总        -> {output_csv}")
    print(f"试验明细        -> {trials_csv}")
    print(f"query debug     -> {query_debug_csv}")
    print(f"branch debug    -> {branch_query_debug_csv}")
    print(f"原始回传        -> {raw_jsonl}")
    print(f"三层 evidence   -> {evidence_json}")
    print(f"统计汇总        -> {summary_json}")

    return 1 if summary["error_rows"] else 0


if __name__ == "__main__":
    raise SystemExit(main())