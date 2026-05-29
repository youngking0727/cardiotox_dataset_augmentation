"""ClinicalTrials.gov drug–trial alignment and QT result attribution."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from utils.clinicaltrials_result_evidence import is_qt_specific_outcome_text
from utils.chembl_drug_name_enrichment import (
    build_drug_name_set_fallback,
    normalize_drug_term,
    strip_salt_form,
)

_RESCUE_RE = re.compile(
    r"\b(?:rescue|as needed|prn|breakthrough|for relief|available for rescue)\b",
    re.IGNORECASE,
)
_BACKGROUND_RE = re.compile(
    r"\b(?:background|concomitant|allowed medication|prior medication|"
    r"stable throughout|maintained throughout|continued throughout)\b",
    re.IGNORECASE,
)
_COMPARATOR_RE = re.compile(
    r"\b(?:comparator|active control|active comparator|compared to|compared with|"
    r"versus|vs\.?|head-to-head|control arm)\b",
    re.IGNORECASE,
)
_POSITIVE_CONTROL_RE = re.compile(
    r"\b(?:moxifloxacin|positive control|positive-control)\b",
    re.IGNORECASE,
)
_COMBINATION_RE = re.compile(
    r"\b(?:combination|combined with|co-administered|coadministered|plus|\+)\b",
    re.IGNORECASE,
)
_NON_DRUG_INTERVENTION_TYPES = {"OTHER", "PROCEDURE", "BEHAVIORAL", "DEVICE", "DIETARY SUPPLEMENT"}


_COMBO_SPLIT_RE = re.compile(r"\s*(?:\+|/|,|\band\b|\bwith\b)\s*", re.IGNORECASE)
_FORMULATION_SUFFIX_RE = re.compile(
    r"\s+(?:"
    r"oral\s+solution|dry\s+syrup|tablets?|capsules?|injection|solution|"
    r"suspension|cream|gel|patch|powder|drops|spray|"
    r"\d+\s*(?:mg|mcg|g|ml|iu|unit|units)(?:\s|$)"
    r").*$",
    re.IGNORECASE,
)
_GENERIC_CLASS_ONLY_RE = re.compile(
    r"^(?:bronchodilator agents?|antihistamines?|placebo|standard therapy|"
    r"active comparator|vehicle control|sham)$",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _normalize_match_term(term: str) -> str:
    return _norm(strip_salt_form(term) or normalize_drug_term(term))


def _normalize_match_terms(terms: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for term in terms or []:
        norm = _normalize_match_term(term)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _field_components(field_text: str) -> List[str]:
    """Split intervention/arm/results labels into normalized drug phrase components."""
    norm = _norm(field_text)
    if not norm:
        return []
    components: List[str] = []
    for part in _COMBO_SPLIT_RE.split(norm):
        part = part.strip()
        if not part:
            continue
        components.append(part)
        base = _FORMULATION_SUFFIX_RE.sub("", part).strip()
        if base and base not in components:
            components.append(base)
    return components


def _term_matches_drug_field(term: str, field_text: str) -> bool:
    """
    Normalized exact token/phrase match in a drug field label.

    Supports combination strings (A + B, A / B, A, B), formulation suffixes
    (e.g. 'cetirizine dry syrup 10 mg'), and case-insensitive matching.
    """
    norm_term = _normalize_match_term(term)
    if not norm_term or not field_text:
        return False
    if _GENERIC_CLASS_ONLY_RE.match(_norm(field_text)):
        return False
    for comp in _field_components(field_text):
        if comp == norm_term:
            return True
        if comp.startswith(norm_term + " "):
            return True
        if _term_in_text(norm_term, comp):
            return True
    return _term_in_text(norm_term, field_text)


def _tokenize_drug(name: str) -> str:
    return strip_salt_form(name) or normalize_drug_term(name)


def build_drug_name_set(
    pref_name: str,
    *,
    enriched: Optional[Dict[str, Any]] = None,
    synonyms: Optional[List[str]] = None,
    parent_names: Optional[List[str]] = None,
    brand_names: Optional[List[str]] = None,
    exclude_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    构建 drug_name_set。

    优先使用 ChEMBL enrichment 结果；否则回退到 pref_name 最小集。
    """
    if enriched and enriched.get("_strong_match_terms"):
        return enriched

    payload = build_drug_name_set_fallback(pref_name)
    if synonyms:
        for s in synonyms:
            term = _tokenize_drug(s)
            if term and term not in payload["_strong_match_terms"]:
                payload["strong_match_terms"].append(term)
                payload["_strong_match_terms"].append(term)
    if parent_names:
        for p in parent_names:
            term = _tokenize_drug(p)
            if term and term not in payload["_related_match_terms"]:
                payload["related_match_terms"].append(term)
                payload["_related_match_terms"].append(term)
                payload["recall_audit_terms"] = sorted(
                    set(payload.get("recall_audit_terms") or []) | {term}
                )
    if exclude_names:
        payload["exclude_names"] = list(exclude_names)
    payload["_all_terms"] = sorted(
        set(payload.get("_strong_match_terms") or [])
        | set(payload.get("_related_match_terms") or [])
        | set(payload.get("_weak_terms") or [])
    )
    return payload


def _term_in_text(term: str, text: str) -> bool:
    if not term or not text:
        return False
    tl = _norm(text)
    tn = _normalize_match_term(term)
    if not tn:
        return False
    # word-boundary match; allow hyphen/space variants
    pattern = re.escape(tn).replace(r"\ ", r"[\s-]?")
    return bool(re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", tl))


def match_terms_in_text(terms: List[str], text: str) -> List[str]:
    norm_terms = _normalize_match_terms(terms)
    return [t for t in norm_terms if _term_matches_drug_field(t, text)]


def _match_terms_in_text(terms: List[str], text: str) -> List[str]:
    return match_terms_in_text(terms, text)


def extract_trial_drug_fields(raw_study: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract intervention / arm / results group fields from raw study JSON."""
    empty = {
        "interventions": [],
        "intervention_names": [],
        "arm_groups": [],
        "arm_group_labels": [],
        "results_groups": [],
        "results_group_labels": [],
        "title": "",
        "summary": "",
    }
    if not raw_study or not isinstance(raw_study, dict):
        return empty

    protocol = raw_study.get("protocolSection") or {}
    ident = protocol.get("identificationModule") or {}
    desc = protocol.get("descriptionModule") or {}
    arms_mod = protocol.get("armsInterventionsModule") or {}

    interventions: List[Dict[str, Any]] = []
    intervention_names: List[str] = []
    for iv in arms_mod.get("interventions") or []:
        if not isinstance(iv, dict):
            continue
        name = (iv.get("name") or "").strip()
        entry = {
            "name": name,
            "type": (iv.get("type") or "").strip(),
            "description": (iv.get("description") or "").strip(),
            "arm_group_labels": list(iv.get("armGroupLabels") or []),
        }
        interventions.append(entry)
        if name:
            intervention_names.append(name)

    arm_groups: List[Dict[str, Any]] = []
    arm_group_labels: List[str] = []
    for ag in arms_mod.get("armGroups") or []:
        if not isinstance(ag, dict):
            continue
        label = (ag.get("label") or "").strip()
        entry = {
            "label": label,
            "type": (ag.get("type") or "").strip(),
            "description": (ag.get("description") or "").strip(),
        }
        arm_groups.append(entry)
        if label:
            arm_group_labels.append(label)

    results_groups: List[Dict[str, Any]] = []
    results_group_labels: List[str] = []
    results_section = raw_study.get("resultsSection") or {}
    om_module = results_section.get("outcomeMeasuresModule") or {}
    seen_ids: Set[str] = set()
    for om in om_module.get("outcomeMeasures") or []:
        if not isinstance(om, dict):
            continue
        for grp in om.get("groups") or []:
            if not isinstance(grp, dict):
                continue
            gid = (grp.get("id") or "").strip()
            if gid and gid in seen_ids:
                continue
            if gid:
                seen_ids.add(gid)
            title = (grp.get("title") or "").strip()
            desc_g = (grp.get("description") or "").strip()
            entry = {"id": gid, "title": title, "description": desc_g}
            results_groups.append(entry)
            if title:
                results_group_labels.append(title)

    title = (ident.get("briefTitle") or ident.get("officialTitle") or "").strip()
    summary = (desc.get("briefSummary") or desc.get("detailedDescription") or "").strip()

    return {
        "interventions": interventions,
        "intervention_names": intervention_names,
        "arm_groups": arm_groups,
        "arm_group_labels": arm_group_labels,
        "results_groups": results_groups,
        "results_group_labels": results_group_labels,
        "title": title,
        "summary": summary,
    }


def _classify_context_role(text: str) -> Optional[str]:
    if _RESCUE_RE.search(text):
        return "rescue_medication"
    if _BACKGROUND_RE.search(text):
        return "background_medication"
    if _COMPARATOR_RE.search(text):
        return "active_comparator"
    if _POSITIVE_CONTROL_RE.search(text):
        return "active_comparator"
    if _COMBINATION_RE.search(text):
        return "combination_component"
    return None


def _intervention_drug_names(interventions: List[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for iv in interventions:
        iv_type = (iv.get("type") or "").upper()
        if iv_type in _NON_DRUG_INTERVENTION_TYPES:
            continue
        name = (iv.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def _intervention_name_matches(strong_terms: List[str], intervention: Dict[str, Any]) -> bool:
    name = (intervention.get("name") or "").strip()
    if name and any(_term_matches_drug_field(t, name) for t in strong_terms):
        return True
    for other in intervention.get("otherNames") or []:
        other_name = str(other or "").strip()
        if other_name and any(_term_matches_drug_field(t, other_name) for t in strong_terms):
            return True
    return False


def _is_combination_intervention_name(name: str) -> bool:
    if not name:
        return False
    if _COMBINATION_RE.search(name):
        return True
    return bool(_COMBO_SPLIT_RE.search(name))


def _resolve_target_drug_role(
    *,
    strong_terms: List[str],
    fields: Dict[str, Any],
    interventions: List[Dict[str, Any]],
    intervention_texts: List[str],
    arm_texts: List[str],
    matched_strong: Set[str],
    matched_related: Set[str],
    matched_in_title: List[str],
    target_in_intervention: bool,
    target_in_arm: bool,
    intervention_drug_names: List[str],
) -> str:
    """Assign target_drug_role; intervention name match overrides description-only rescue cues."""
    named_interventions = [
        iv for iv in interventions if _intervention_name_matches(strong_terms, iv)
    ]

    if named_interventions:
        combo_names = [
            (iv.get("name") or "").strip()
            for iv in named_interventions
            if _is_combination_intervention_name(iv.get("name") or "")
        ]
        if combo_names:
            return "combination_component"
        if len(intervention_drug_names) > 1:
            return "main_intervention"
        return "main_intervention"

    role_candidates: List[str] = []
    for text in intervention_texts + arm_texts:
        if not match_terms_in_text(strong_terms, text):
            continue
        ctx_role = _classify_context_role(text)
        if ctx_role:
            role_candidates.append(ctx_role)

    if matched_strong and "rescue_medication" in role_candidates:
        return "rescue_medication"
    if matched_strong and "background_medication" in role_candidates:
        return "background_medication"
    if matched_strong and "active_comparator" in role_candidates:
        return "active_comparator"
    if matched_strong and "combination_component" in role_candidates:
        return "combination_component"
    if matched_related and not matched_strong:
        return "related_drug_only"
    if matched_strong and (target_in_intervention or target_in_arm):
        return "main_intervention"
    if matched_in_title and not matched_strong:
        return "unclear"
    other_drug_interventions = [
        n for n in intervention_drug_names
        if not match_terms_in_text(strong_terms, n)
    ]
    if other_drug_interventions and not matched_strong:
        return "no_direct_match"
    return "unclear"


def assess_drug_trial_alignment(
    drug_name_set: Dict[str, Any],
    raw_study: Optional[Dict[str, Any]],
    *,
    has_qt_specific_protocol: bool = False,
    has_ecg_conduction_protocol: bool = False,
    has_ecg_broad_protocol: bool = False,
    has_qt_specific_results: bool = False,
    has_ecg_conduction_results: bool = False,
    has_ecg_broad_results: bool = False,
) -> Dict[str, Any]:
    """Assess whether a trial's QT evidence can be attributed to the target drug."""
    fields = extract_trial_drug_fields(raw_study)
    strong_terms = _normalize_match_terms(drug_name_set.get("_strong_match_terms") or [])
    related_terms = _normalize_match_terms(drug_name_set.get("_related_match_terms") or [])
    all_terms = _normalize_match_terms(drug_name_set.get("_all_terms") or [])

    interventions = fields["interventions"]
    intervention_texts = [
        " ".join([
            iv.get("name", ""),
            iv.get("description", ""),
            " ".join(iv.get("arm_group_labels") or []),
        ])
        for iv in interventions
    ]
    arm_texts = [
        " ".join([ag.get("label", ""), ag.get("description", "")])
        for ag in fields["arm_groups"]
    ]
    results_group_texts = [
        " ".join([g.get("title", ""), g.get("description", "")])
        for g in fields["results_groups"]
    ]
    title_summary = " ".join([fields["title"], fields["summary"]])

    matched_strong: Set[str] = set()
    matched_related: Set[str] = set()

    def _collect(terms: List[str], texts: List[str], bucket: Set[str]) -> None:
        for text in texts:
            for term in terms:
                if _term_matches_drug_field(term, text):
                    bucket.add(term)

    _collect(strong_terms, intervention_texts, matched_strong)
    _collect(strong_terms, arm_texts, matched_strong)
    _collect(strong_terms, results_group_texts, matched_strong)
    _collect(related_terms, intervention_texts + arm_texts + results_group_texts, matched_related)

    matched_in_title = match_terms_in_text(all_terms, title_summary)
    matched_related -= matched_strong

    target_in_intervention = any(
        _intervention_name_matches(strong_terms, iv) for iv in interventions
    )
    target_in_arm = any(
        match_terms_in_text(strong_terms, text) for text in arm_texts
    )
    target_in_results_group = any(
        match_terms_in_text(strong_terms, text) for text in results_group_texts
    )
    target_mentioned = bool(
        matched_in_title or matched_strong or matched_related or target_in_intervention
    )

    intervention_drug_names = _intervention_drug_names(interventions)

    target_drug_role = _resolve_target_drug_role(
        strong_terms=strong_terms,
        fields=fields,
        interventions=interventions,
        intervention_texts=intervention_texts,
        arm_texts=arm_texts,
        matched_strong=matched_strong,
        matched_related=matched_related,
        matched_in_title=matched_in_title,
        target_in_intervention=target_in_intervention,
        target_in_arm=target_in_arm,
        intervention_drug_names=intervention_drug_names,
    )

    drug_match_level = "unclear"
    if target_drug_role in {"rescue_medication", "background_medication", "covariate_only"}:
        drug_match_level = "weak"
    elif target_drug_role == "related_drug_only":
        drug_match_level = "false_positive"
    elif target_drug_role == "no_direct_match":
        drug_match_level = "false_positive"
    elif target_drug_role == "combination_component" and (
        target_in_intervention or target_in_arm
    ):
        drug_match_level = "medium"
    elif target_drug_role in {"active_comparator", "combination_component"}:
        drug_match_level = "medium"
    elif target_drug_role == "main_intervention":
        drug_match_level = "strong"
    elif target_in_intervention or target_in_arm:
        drug_match_level = "strong"
    elif matched_in_title and not matched_strong:
        drug_match_level = "weak"

    has_ep_signal = (
        has_qt_specific_protocol
        or has_ecg_conduction_protocol
        or has_ecg_broad_protocol
        or has_qt_specific_results
        or has_ecg_conduction_results
        or has_ecg_broad_results
    )
    evidence_attribution_level = "unclear"
    if drug_match_level == "strong" and has_ep_signal:
        evidence_attribution_level = "direct"
    elif drug_match_level == "medium" and has_ep_signal:
        evidence_attribution_level = "partial"
    elif drug_match_level == "weak":
        evidence_attribution_level = "weak" if has_ep_signal else "not_attributable"
    elif drug_match_level == "false_positive":
        evidence_attribution_level = "not_attributable"
    elif has_ep_signal and (target_in_intervention or target_in_arm):
        evidence_attribution_level = "direct"
    elif has_ep_signal:
        evidence_attribution_level = "unclear"

    alignment_summary = _build_alignment_summary(
        drug_name_set.get("pref_name", ""),
        target_drug_role,
        drug_match_level,
        evidence_attribution_level,
        fields,
        sorted(matched_strong | matched_related),
    )

    return {
        "target_drug_mentioned": target_mentioned,
        "target_drug_in_intervention": target_in_intervention,
        "target_drug_in_arm_group": target_in_arm,
        "target_drug_in_results_group": target_in_results_group,
        "target_drug_role": target_drug_role,
        "drug_match_level": drug_match_level,
        "evidence_attribution_level": evidence_attribution_level,
        "matched_drug_terms": sorted(matched_strong | matched_related | set(matched_in_title)),
        "intervention_names": fields["intervention_names"],
        "arm_group_labels": fields["arm_group_labels"],
        "results_group_labels": fields["results_group_labels"],
        "alignment_summary": alignment_summary,
    }


def _build_alignment_summary(
    pref_name: str,
    role: str,
    match_level: str,
    attribution_level: str,
    fields: Dict[str, Any],
    matched_terms: List[str],
) -> str:
    iv = ", ".join(fields.get("intervention_names") or []) or "none listed"
    if match_level == "false_positive":
        return (
            f"Trial interventions ({iv}) do not directly match target drug {pref_name}; "
            f"QT signal likely belongs to other study drug(s) or related-name noise."
        )
    if role == "rescue_medication":
        return (
            f"{pref_name} appears only as rescue/background medication; "
            f"QT evidence is not directly attributable."
        )
    if role == "combination_component":
        return (
            f"{pref_name} is a combination-component in this trial (interventions: {iv}); "
            f"QT results may reflect combined therapy."
        )
    if role == "active_comparator":
        return (
            f"{pref_name} appears as active comparator; QT evidence is partial attribution only."
        )
    if match_level == "strong":
        return (
            f"Strong alignment: {pref_name} matched in intervention/arm "
            f"({', '.join(matched_terms) or 'n/a'})."
        )
    if match_level == "weak":
        return (
            f"Weak alignment: {pref_name} mentioned in title/summary/context only; "
            f"interventions are {iv}."
        )
    return f"Unclear alignment for {pref_name}; interventions are {iv}."


def assess_qt_result_attribution(
    drug_name_set: Dict[str, Any],
    qt_result_measures: List[Dict[str, Any]],
    raw_study: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Arm-level attribution for QT resultsSection outcomes."""
    empty = {
        "has_qt_results": False,
        "qt_result_groups": [],
        "target_group_ids": [],
        "non_target_group_ids": [],
        "combination_group_ids": [],
        "qt_result_for_target_drug": False,
        "qt_result_for_comparator_only": False,
        "summary": "No QT/QTc result outcome measures to attribute.",
    }
    if not qt_result_measures:
        return empty

    strong_terms = _normalize_match_terms(drug_name_set.get("_strong_match_terms") or [])
    related_terms = _normalize_match_terms(drug_name_set.get("_related_match_terms") or [])
    fields = extract_trial_drug_fields(raw_study)

    group_map: Dict[str, Dict[str, str]] = {}
    for om in qt_result_measures:
        for grp in om.get("groups") or []:
            if not isinstance(grp, dict):
                continue
            gid = (grp.get("id") or "").strip()
            if not gid:
                continue
            text = " ".join([
                grp.get("title") or "",
                grp.get("description") or "",
            ])
            group_map[gid] = {
                "title": grp.get("title") or "",
                "description": grp.get("description") or "",
                "text": text,
            }

    qt_result_groups: List[Dict[str, Any]] = []
    target_group_ids: Set[str] = set()
    non_target_group_ids: Set[str] = set()
    combination_group_ids: Set[str] = set()
    has_comparator_group = False
    has_overall_only = False

    for gid, info in group_map.items():
        text = info["text"]
        strong_hits = _match_terms_in_text(strong_terms, text)
        related_hits = _match_terms_in_text(related_terms, text)
        is_positive_control = bool(_POSITIVE_CONTROL_RE.search(text))
        is_comparator = bool(_COMPARATOR_RE.search(text)) or is_positive_control
        is_combination_group = _is_combination_intervention_name(text)

        if strong_hits and is_combination_group:
            attribution = "combination_arm"
            combination_group_ids.add(gid)
        elif strong_hits:
            attribution = "target_drug"
            target_group_ids.add(gid)
        elif related_hits and not strong_hits:
            attribution = "related_drug"
            non_target_group_ids.add(gid)
        elif is_comparator or _intervention_names_in_text(fields["intervention_names"], text, strong_terms):
            attribution = "comparator_or_other_drug"
            non_target_group_ids.add(gid)
            if is_comparator:
                has_comparator_group = True
        elif _norm(info["title"]) in {"overall", "total", "all participants", "all subjects"}:
            attribution = "overall_unspecified"
            has_overall_only = True
        else:
            attribution = "unclear"

        qt_result_groups.append({
            "group_id": gid,
            "title": info["title"],
            "description": info["description"],
            "attribution": attribution,
            "matched_terms": strong_hits or related_hits,
        })

    qt_result_for_target = bool(target_group_ids)
    qt_result_for_comparator_only = (
        bool(non_target_group_ids) and not target_group_ids and not has_overall_only
    )

    if qt_result_for_target and non_target_group_ids:
        summary = (
            "QT results exist for both target-drug and non-target/comparator groups; "
            "use group-level attribution."
        )
    elif qt_result_for_target:
        summary = "QT/QTc results are reported for treatment group(s) containing the target drug."
    elif combination_group_ids:
        summary = (
            "QT/QTc results appear only for combination arm group(s); "
            "not attributable to target drug alone."
        )
    elif qt_result_for_comparator_only:
        summary = (
            "QT/QTc results appear only for comparator/other-drug groups; "
            "not attributable to target drug."
        )
    elif has_overall_only:
        summary = "QT/QTc results use overall/unspecified groups; arm attribution is uncertain."
    else:
        summary = "QT/QTc results present but group-level drug attribution is unclear."

    return {
        "has_qt_results": True,
        "qt_result_groups": qt_result_groups,
        "target_group_ids": sorted(target_group_ids),
        "non_target_group_ids": sorted(non_target_group_ids),
        "combination_group_ids": sorted(combination_group_ids),
        "qt_result_for_target_drug": qt_result_for_target,
        "qt_result_for_comparator_only": qt_result_for_comparator_only,
        "summary": summary,
    }


def _intervention_names_in_text(
    intervention_names: List[str],
    text: str,
    strong_terms: List[str],
) -> bool:
    for name in intervention_names:
        if _match_terms_in_text(strong_terms, name):
            continue  # target drug intervention
        if _term_in_text(_tokenize_drug(name), text):
            return True
    return False


# Roles that block Primary QT tiers (rescue / background / concomitant / comparator).
_EXCLUDED_PRIMARY_ROLES = frozenset({
    "rescue_medication",
    "background_medication",
    "active_comparator",
    "covariate_only",
})
_COMBINATION_ROLES = frozenset({
    "combination_component",
    "combination_drug",
})


def _has_strict_qt_protocol_endpoint(
    protocol_outcomes: Dict[str, Any],
    qt_outcome_measure: str,
    title_classification: Dict[str, Any],
) -> bool:
    """Primary protocol QT requires explicit QT/QTc wording on endpoint labels."""
    if is_qt_specific_outcome_text(qt_outcome_measure or ""):
        return True

    for om in protocol_outcomes.get("qt_specific_outcomes") or []:
        label = " ".join(
            part for part in (om.get("title") or "", om.get("measure") or "") if part
        ).strip()
        if is_qt_specific_outcome_text(label):
            return True

    title_text = title_classification.get("raw_text") or ""
    return is_qt_specific_outcome_text(title_text)


def _has_target_qt_result_attribution(
    alignment: Dict[str, Any],
    qt_result_attribution: Dict[str, Any],
) -> bool:
    return bool(
        qt_result_attribution.get("qt_result_for_target_drug")
        and (
            alignment.get("target_drug_in_results_group")
            or qt_result_attribution.get("target_group_ids")
        )
    )


def _is_combination_role(role: str) -> bool:
    return role in _COMBINATION_ROLES


def classify_evidence_tier(
    alignment: Dict[str, Any],
    *,
    qt_result_attribution: Dict[str, Any],
    protocol_qt_hit: bool = True,
    results_qt_hit: bool = False,
    search_branch: str = "",
    protocol_outcomes: Optional[Dict[str, Any]] = None,
    title_classification: Optional[Dict[str, Any]] = None,
    results_section: Optional[Dict[str, Any]] = None,
    qt_outcome_measure: str = "",
    tier_reason: Optional[Dict[str, str]] = None,
) -> str:
    """
    Evidence tier assignment with strict Primary QT gating.

    Primary QT actual results require explicit results-group attribution.
    Primary QT protocol outcomes require explicit QT/QTc endpoint wording.
    Combination components are routed to combination_qt_evidence, not direct primary.
    """
    role = alignment.get("target_drug_role", "unclear")
    match_level = alignment.get("drug_match_level", "unclear")

    protocol_outcomes = protocol_outcomes or {}
    title_classification = title_classification or {}
    results_section = results_section or {}

    def _set_reason(reason: str) -> None:
        if tier_reason is not None:
            tier_reason["reason"] = reason

    has_qt_protocol = protocol_outcomes.get("has_qt_specific", False)
    has_qt_title = title_classification.get("evidence_type") == "qt_specific"
    has_qt_results = bool(results_section.get("has_qt_results"))

    has_cond_protocol = protocol_outcomes.get("has_ecg_conduction", False)
    has_cond_results = bool(results_section.get("has_ecg_conduction_results"))

    has_broad_protocol = protocol_outcomes.get("has_ecg_broad", False)
    has_broad_results = bool(results_section.get("has_ecg_broad_results"))

    has_non_qt_cardiology = (
        protocol_outcomes.get("has_non_qt_cardiology", False)
        or bool(results_section.get("has_non_qt_cardiology_results"))
    )

    target_in_intervention = bool(alignment.get("target_drug_in_intervention"))
    target_in_arm = bool(alignment.get("target_drug_in_arm_group"))
    qt_for_target = _has_target_qt_result_attribution(alignment, qt_result_attribution)
    qt_comparator_only = bool(qt_result_attribution.get("qt_result_for_comparator_only"))
    strict_qt_protocol = _has_strict_qt_protocol_endpoint(
        protocol_outcomes,
        qt_outcome_measure,
        title_classification,
    )

    # 1. rescue / background / concomitant only
    if role in {"rescue_medication", "background_medication", "covariate_only"}:
        _set_reason("Target drug appears only as rescue/background/concomitant medication.")
        return "rescue_or_background_only"

    # 2. comparator-only actual QT result
    if has_qt_results and qt_comparator_only and not qt_for_target:
        _set_reason("QT/QTc actual results belong to comparator/other-drug groups only.")
        return "comparator_qt_evidence"

    # 3. direct QT actual result
    if (
        has_qt_results
        and qt_for_target
        and not qt_comparator_only
        and role not in _EXCLUDED_PRIMARY_ROLES
        and not _is_combination_role(role)
    ):
        _set_reason("QT/QTc actual result attributed to target-drug results group.")
        return "direct_qt_actual_result"

    # 4. results-only QT actual result
    if (
        search_branch == "results_recall"
        and has_qt_results
        and results_qt_hit
        and qt_for_target
        and not qt_comparator_only
        and role not in _EXCLUDED_PRIMARY_ROLES
        and not _is_combination_role(role)
    ):
        _set_reason("Results-recall branch with target-drug QT/QTc actual result attribution.")
        return "results_only_qt_actual_result"

    # 5. combination QT evidence
    if _is_combination_role(role) and (
        (has_qt_results and (qt_for_target or qt_result_attribution.get("combination_group_ids")))
        or (protocol_qt_hit and strict_qt_protocol)
        or has_qt_protocol
    ):
        _set_reason(
            "QT-specific endpoint/result but target drug is a combination component; "
            "classified as combination_qt_evidence rather than direct primary."
        )
        return "combination_qt_evidence"

    # 6. direct QT protocol outcome
    if (
        protocol_qt_hit
        and strict_qt_protocol
        and (target_in_intervention or target_in_arm)
        and role not in _EXCLUDED_PRIMARY_ROLES
        and not _is_combination_role(role)
    ):
        _set_reason("Explicit QT/QTc protocol endpoint with target drug in intervention/arm.")
        return "direct_qt_protocol_outcome"

    # 7. actual QT result needs review
    if (
        has_qt_results
        and not qt_for_target
        and not qt_comparator_only
        and match_level != "false_positive"
    ):
        _set_reason("QT/QTc actual results present but target-drug group attribution is unclear.")
        return "actual_result_needs_review"

    # 8. ECG conduction supportive
    if (has_cond_protocol or has_cond_results) and match_level in {"strong", "medium", "weak"}:
        if role == "active_comparator":
            _set_reason("ECG conduction endpoint for active comparator.")
            return "comparator_ecg_evidence"
        if _is_combination_role(role):
            _set_reason("ECG conduction endpoint for combination component.")
            return "combination_ecg_evidence"
        _set_reason("ECG conduction endpoint without QT-specific primary evidence.")
        return "direct_ecg_conduction_evidence"

    # 9. broad ECG supportive
    if has_broad_protocol or has_broad_results or (
        protocol_qt_hit and not strict_qt_protocol and (has_qt_protocol or has_qt_title)
    ):
        if _is_combination_role(role) and not has_qt_results and not strict_qt_protocol:
            _set_reason("Combination context without QT-specific endpoint.")
            return "combination_context_only"
        if role == "active_comparator":
            _set_reason("Broad ECG endpoint for active comparator.")
            return "comparator_ecg_evidence"
        if _is_combination_role(role):
            _set_reason("Broad ECG endpoint for combination component.")
            return "combination_ecg_evidence"
        _set_reason("Broad ECG endpoint without QT-specific primary evidence.")
        return "ecg_broad_supportive_evidence"

    # 10. false positive mapping
    if (
        match_level == "false_positive"
        and not target_in_intervention
        and not target_in_arm
        and not alignment.get("target_drug_in_results_group")
        and not qt_for_target
    ):
        _set_reason("No structural drug match and no attributable QT result.")
        return "false_positive_mapping"

    # 11. non-QT excluded / manual review
    if has_non_qt_cardiology and not has_qt_results and not strict_qt_protocol:
        _set_reason("Non-QT cardiology endpoint without QT-specific evidence.")
        return "non_qt_cardiology_endpoint"

    if (
        not has_qt_results
        and not strict_qt_protocol
        and not has_cond_protocol
        and not has_cond_results
        and not has_broad_protocol
        and not has_broad_results
    ):
        _set_reason("No QT/ECG electrophysiology evidence matched.")
        return "non_qt_excluded"

    if has_qt_title and match_level in {"strong", "medium", "weak"}:
        _set_reason("QT-related title context without attributable endpoint/result.")
        return "direct_qt_context"

    _set_reason("Evidence present but tier assignment remains uncertain.")
    return "manual_review_required"
