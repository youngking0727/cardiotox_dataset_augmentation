"""Strict per-outcome ECG/QT evidence classification for ClinicalTrials.gov.

Evidence types:
- qt_specific: explicit QT/QTc/Q-T/Q-Tc/TQT/TdP evidence
- ecg_conduction: QRS/PR/RR/conduction interval evidence
- ecg_broad: ECG/EKG/electrocardiogram evidence without specific interval
- cardiac_ae: cardiac adverse event evidence
- non_qt: unrelated or explicitly non-cardiac/non-ECG endpoint

Important principle:
Trial-level query hit must not be inherited by every outcome.
Each protocol outcome and each resultsSection outcome must be classified independently.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


_UNICODE_DASHES = ("\u2013", "\u2014", "\u2212", "\u2010", "\u2011", "\u2012", "\u2015")
_SPACE_RE = re.compile(r"\s+")


# =============================================================================
# Normalization
# =============================================================================

def build_outcome_text(
    *,
    title: str = "",
    measure: str = "",
    description: str = "",
    time_frame: str = "",
    unit: str = "",
    classes: Optional[List[Any]] = None,
) -> str:
    """Build one searchable text block from an outcome measure."""
    parts = [title, measure, description, time_frame, unit]

    for entry in classes or []:
        if not isinstance(entry, dict):
            continue

        parts.append((entry.get("title") or "").strip())

        for cat in entry.get("categories") or []:
            if isinstance(cat, dict):
                parts.append((cat.get("title") or "").strip())

    return " ".join(p.strip() for p in parts if isinstance(p, str) and p.strip())


def normalize_outcome_text(raw: str) -> Tuple[str, str, str]:
    """Return raw, normalized, and compact text.

    normalized:
        lowercase, unicode dashes normalized, spaces compressed.

    compact:
        no spaces, useful for Q-T / Q - T / Q-Tc variants.
    """
    raw_text = (raw or "").strip()
    text = raw_text.lower()

    for ch in _UNICODE_DASHES:
        text = text.replace(ch, "-")

    normalized = _SPACE_RE.sub(" ", text).strip()

    compact = normalized
    compact = re.sub(r"q\s*-\s*t\s*c\b", "q-tc", compact)
    compact = re.sub(r"q\s*-\s*t\b", "q-t", compact)
    compact = re.sub(r"q\s+t\s+c\b", "qtc", compact)
    compact = re.sub(r"q\s+t\b", "qt", compact)
    compact = re.sub(r"\s+", "", compact)

    return raw_text, normalized, compact


# =============================================================================
# Keyword tables
# =============================================================================

# Strong QT/QTc/TdP phrases.
# Note: standalone "qt" or standalone "q-t" is intentionally NOT included.
_QT_SPECIFIC_PHRASES: Tuple[Tuple[str, str], ...] = (
    # TdP / long QT
    ("torsades de pointes", "tdp_related"),
    ("torsade de pointes", "tdp_related"),
    ("torsades", "tdp_related"),
    ("torsade", "tdp_related"),
    ("tdp", "tdp_related"),
    ("long qt syndrome", "long_qt"),
    ("long q-t syndrome", "long_qt"),
    ("long qt", "long_qt"),
    ("long q-t", "long_qt"),

    # TQT
    ("thorough qt", "tqt"),
    ("thorough q-t", "tqt"),
    ("tqt", "tqt"),

    # QTc / Q-Tc
    ("q-t interval corrected", "qtc"),
    ("qt interval corrected", "qtc"),
    ("corrected q-t", "qtc"),
    ("corrected qt", "qtc"),
    ("qt corrected", "qtc"),
    ("q-t corrected", "qtc"),
    ("qtc interval", "qtc"),
    ("q-tc interval", "qtc"),
    ("qtc prolongation", "qtc_prolongation"),
    ("q-tc prolongation", "qtc_prolongation"),
    ("prolonged qtc", "qtc_prolongation"),
    ("prolonged q-tc", "qtc_prolongation"),
    ("qtc prolonged", "qtc_prolongation"),
    ("q-tc prolonged", "qtc_prolongation"),
    ("change from baseline in qtc", "delta_qtc"),
    ("change from baseline in q-tc", "delta_qtc"),
    ("change in qtc", "delta_qtc"),
    ("change in q-tc", "delta_qtc"),
    ("delta qtc", "delta_qtc"),
    ("delta q-tc", "delta_qtc"),
    ("ddqtc", "delta_qtc"),
    ("dd-qtc", "delta_qtc"),
    ("dd qtc", "delta_qtc"),

    # QTc correction formula
    ("fridericia's correction", "fridericia"),
    ("fridericia correction", "fridericia"),
    ("fridericia corrected qt", "fridericia"),
    ("fridericia", "fridericia"),
    ("bazett's correction", "bazett"),
    ("bazett correction", "bazett"),
    ("bazett corrected qt", "bazett"),
    ("bazett", "bazett"),

    # QTcF / QTcB variants
    ("qtc-f", "qtcf"),
    ("q-tc-f", "qtcf"),
    ("q-tcf", "qtcf"),
    ("qtcf", "qtcf"),
    ("qtc-b", "qtcb"),
    ("q-tc-b", "qtcb"),
    ("q-tcb", "qtcb"),
    ("qtcb", "qtcb"),

    # compact QTc forms
    ("q-tc", "qtc"),
    ("qtc", "qtc"),

    # QT interval with explicit context
    ("qt interval", "qt_interval"),
    ("q-t interval", "qt_interval"),
    ("q t interval", "qt_interval"),
    ("qt duration", "qt_interval"),
    ("q-t duration", "qt_interval"),
    ("qt prolongation", "qt_prolongation"),
    ("q-t prolongation", "qt_prolongation"),
    ("prolonged qt", "qt_prolongation"),
    ("prolonged q-t", "qt_prolongation"),
    ("qt prolonged", "qt_prolongation"),
    ("q-t prolonged", "qt_prolongation"),
    ("change from baseline in qt", "delta_qt"),
    ("change from baseline in q-t", "delta_qt"),
    ("change in qt interval", "delta_qt"),
    ("change in q-t interval", "delta_qt"),
    ("delta qt", "delta_qt"),
    ("delta q-t", "delta_qt"),
)


_ECG_CONDUCTION_PHRASES: Tuple[Tuple[str, str], ...] = (
    # QRS — only interval/duration/conduction forms; NOT infarct-size forms
    ("q-r-s duration", "qrs"),
    ("q-r-s interval", "qrs"),
    ("qrs duration", "qrs"),
    ("qrs interval", "qrs"),
    ("qrs prolongation", "qrs"),
    ("prolonged qrs", "qrs"),

    # PR
    ("p-r duration", "pr"),
    ("p-r interval", "pr"),
    ("pr duration", "pr"),
    ("pr interval", "pr"),
    ("atrioventricular conduction", "pr"),
    ("av conduction", "pr"),

    # RR
    ("r-r duration", "rr"),
    ("r-r interval", "rr"),
    ("rr duration", "rr"),
    ("rr interval", "rr"),

    # General conduction (must be paired with conduction context, not infarct)
    ("ventricular conduction", "conduction"),
    ("cardiac conduction", "conduction"),
    ("conduction interval", "conduction"),
)

# Phrases that look like ECG/QRS but are non-interval cardiology endpoints.
# These are only applied when no true QT/QTc/conduction interval phrase is present.
_NON_QT_CARDIOLOGY_PHRASES: Tuple[Tuple[str, str], ...] = (
    ("qrs determined infarct size", "infarct_size"),
    ("qrs infarct size", "infarct_size"),
    ("myocardial infarct size", "infarct_size"),
    ("infarct size by ecg", "infarct_size"),
)


_ECG_BROAD_PHRASES: Tuple[Tuple[str, str], ...] = (
    ("12-lead ecg", "ecg_broad"),
    ("12 lead ecg", "ecg_broad"),
    ("12-lead ekg", "ecg_broad"),
    ("12 lead ekg", "ecg_broad"),
    ("electrocardiogram findings", "ecg_broad"),
    ("electrocardiographic", "ecg_broad"),
    ("electrocardiogram", "ecg_broad"),
    ("ecg abnormalities", "ecg_broad"),
    ("ecg abnormality", "ecg_broad"),
    ("ecg findings", "ecg_broad"),
    ("ecg safety", "ecg_broad"),
    ("ecg", "ecg_broad"),
    ("ekg", "ecg_broad"),
)


_CARDIAC_AE_PHRASES: Tuple[Tuple[str, str], ...] = (
    ("sudden cardiac death", "serious_cardiac_ae"),
    ("ventricular tachycardia", "arrhythmia_related"),
    ("ventricular arrhythmia", "arrhythmia_related"),
    ("cardiac arrhythmia", "arrhythmia_related"),
    ("atrial fibrillation", "arrhythmia_related"),
    ("cardiac arrest", "serious_cardiac_ae"),
    ("arrhythmia", "arrhythmia_related"),
    ("tachycardia", "arrhythmia_related"),
    ("bradycardia", "arrhythmia_related"),
    ("palpitations", "weak_cardiac_symptom"),
    ("syncope", "weak_cardiac_symptom"),
)


# These only exclude when no real QT/ECG/conduction/cardiac AE marker exists.
_NON_QT_PHRASES: Tuple[Tuple[str, str], ...] = (
    ("quantitative pcr", "pcr"),
    ("real-time pcr", "pcr"),
    ("rt-pcr", "pcr"),
    ("qt-pcr", "pcr"),
    ("qpcr", "pcr"),
    ("protease activity", "klk5"),
    ("kallikrein", "klk5"),
    ("klk5", "klk5"),
    ("food cravings", "food_craving"),
    ("food craving", "food_craving"),
    ("appetite", "appetite"),
    ("phone call", "administrative"),
    ("medical chart review", "administrative"),
    ("microbiological data", "non_cardiac"),
    ("biological data", "non_cardiac"),
    ("treatment success", "non_cardiac"),
    ("clinical cure", "non_cardiac"),
    ("fever recurrence", "non_cardiac"),
    ("hospital admission", "non_cardiac"),
    ("pharmacokinetic parameter only", "pk_only"),
    ("concentration only", "pk_only"),
    ("auc only", "pk_only"),
    ("cmax only", "pk_only"),
    ("tmax only", "pk_only"),
    ("biomarker", "biomarker"),
)


_EVIDENCE_PRIORITY = {
    "qt_specific": 5,
    "ecg_conduction": 4,
    "cardiac_ae": 3,
    "ecg_broad": 2,
    "non_qt_cardiology": 1,
    "non_qt": 0,
}


# =============================================================================
# Matching helpers
# =============================================================================

def _regex_escape_phrase(phrase: str) -> str:
    """Escape phrase and allow flexible whitespace around internal spaces."""
    escaped = re.escape(phrase)
    escaped = escaped.replace(r"\ ", r"\s+")
    return escaped


def _phrase_in_text(phrase: str, normalized: str, compact: str) -> bool:
    """Safer phrase matching.

    - For normal text, use token boundaries.
    - For compact Q-T/QTc forms, allow compact matching.
    """
    phrase_norm = phrase.lower().strip()
    if not phrase_norm:
        return False

    # compact variants for q-t/q-tc/qtc/qtcf/qtcb/ddqtc
    compact_phrase = re.sub(r"\s+", "", phrase_norm)
    compact_phrase = re.sub(r"q\s*-\s*t\s*c\b", "q-tc", compact_phrase)
    compact_phrase = re.sub(r"q\s*-\s*t\b", "q-t", compact_phrase)

    if compact_phrase in {"qtc", "q-tc", "qtcf", "qtcb", "qtc-f", "qtc-b", "q-tcf", "q-tcb", "ddqtc", "dd-qtc", "tdp", "tqt"}:
        if compact_phrase.replace("-", "") in compact.replace("-", ""):
            return True

    # token boundary regex on normalized text
    pattern = _regex_escape_phrase(phrase_norm)
    return bool(re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", normalized))


def _match_table(
    normalized: str,
    compact: str,
    table: Tuple[Tuple[str, str], ...],
) -> List[Tuple[str, str]]:
    hits: List[Tuple[str, str]] = []
    seen = set()

    for phrase, subtype in table:
        if _phrase_in_text(phrase, normalized, compact):
            key = (phrase, subtype)
            if key not in seen:
                hits.append(key)
                seen.add(key)

    return hits


def _has_real_ecg_or_qt_marker(normalized: str, compact: str) -> bool:
    """Return True if real QT/ECG/conduction/cardiac marker exists.

    This prevents non-QT exclusion terms such as AUC/Cmax from removing
    exposure-QTc outcomes.
    """
    if _match_table(normalized, compact, _QT_SPECIFIC_PHRASES):
        return True
    if _match_table(normalized, compact, _ECG_CONDUCTION_PHRASES):
        return True
    if _match_table(normalized, compact, _ECG_BROAD_PHRASES):
        return True
    if _match_table(normalized, compact, _CARDIAC_AE_PHRASES):
        return True
    return False


def _cardiac_safety_only(normalized: str) -> bool:
    """Detect broad cardiac safety without actual ECG/AE marker."""
    if not re.search(r"(?<![a-z0-9])cardiac\s+safety(?![a-z0-9])", normalized):
        return False

    has_ecg = bool(re.search(r"(?<![a-z0-9])(ecg|ekg|electrocardiogram|electrocardiographic)(?![a-z0-9])", normalized))
    has_ae = any(_phrase_in_text(p, normalized, normalized.replace(" ", "")) for p, _ in _CARDIAC_AE_PHRASES)
    has_qt = bool(_match_table(normalized, normalized.replace(" ", ""), _QT_SPECIFIC_PHRASES))
    has_cond = bool(_match_table(normalized, normalized.replace(" ", ""), _ECG_CONDUCTION_PHRASES))

    return not (has_ecg or has_ae or has_qt or has_cond)


# =============================================================================
# Main classifier
# =============================================================================

def classify_outcome_text(raw_text: str) -> Dict[str, Any]:
    """Classify one outcome text.

    Returns:
        Dict with raw_text, normalized_text, compact_text, evidence_type,
        evidence_subtype, matched_keywords, evidence_summary.
    """
    raw, normalized, compact = normalize_outcome_text(raw_text)

    qt_hits = _match_table(normalized, compact, _QT_SPECIFIC_PHRASES)
    conduction_hits = _match_table(normalized, compact, _ECG_CONDUCTION_PHRASES)
    ecg_broad_hits = _match_table(normalized, compact, _ECG_BROAD_PHRASES)
    cardiac_ae_hits = _match_table(normalized, compact, _CARDIAC_AE_PHRASES)
    non_qt_hits = _match_table(normalized, compact, _NON_QT_PHRASES)
    non_qt_cardiology_hits = _match_table(normalized, compact, _NON_QT_CARDIOLOGY_PHRASES)

    # Strong exclusion: non-QT if explicit non-cardiac/administrative terms
    # and no real ECG/QT/conduction/cardiac marker exists.
    if non_qt_hits and not _has_real_ecg_or_qt_marker(normalized, compact):
        kw = [p for p, _ in non_qt_hits]
        return _pack(raw, normalized, compact, "non_qt", non_qt_hits[0][1], kw,
                     f"Non-QT/ECG endpoint ({kw[0]}).")

    # Do not let generic "cardiac safety" alone become ECG evidence.
    if _cardiac_safety_only(normalized):
        return _pack(raw, normalized, compact, "non_qt", "cardiac_safety_broad",
                     ["cardiac safety"],
                     "Broad cardiac safety text without explicit ECG/QT/conduction/AE endpoint.")

    # Non-QT cardiology (e.g. infarct size, ECG changes) before conduction check
    if non_qt_cardiology_hits and not qt_hits and not conduction_hits:
        kw = [p for p, _ in non_qt_cardiology_hits]
        return _pack(raw, normalized, compact, "non_qt_cardiology",
                     non_qt_cardiology_hits[0][1], kw,
                     f"Non-QT cardiology endpoint ({kw[0]}); not an ECG interval endpoint.")

    # "qrs" alone without interval/duration/conduction context → non_qt_cardiology
    # (catches "QRS determined infarct size" even if the above didn't fire)
    if conduction_hits:
        clean_conduction = [
            h for h in conduction_hits
            if not any(nq_kw in normalized for nq_kw, _ in _NON_QT_CARDIOLOGY_PHRASES
                       if nq_kw in normalized)
        ]
        # If every conduction hit is contaminated by an infarct/non-qt phrase, downgrade
        if not clean_conduction and non_qt_cardiology_hits:
            kw = [p for p, _ in non_qt_cardiology_hits]
            return _pack(raw, normalized, compact, "non_qt_cardiology",
                         non_qt_cardiology_hits[0][1], kw,
                         f"QRS-like text but in non-QT cardiology context ({kw[0]}).")

    # Priority 1: QT/QTc/TdP
    if qt_hits:
        subtypes = {subtype for _, subtype in qt_hits}
        if "tdp_related" in subtypes:
            subtype = "tdp_related"
        elif "long_qt" in subtypes:
            subtype = "long_qt"
        elif any(s in subtypes for s in {"qtcf", "qtcb", "fridericia", "bazett"}):
            subtype = next(s for s in ("qtcf", "qtcb", "fridericia", "bazett") if s in subtypes)
        elif any(s in subtypes for s in {"qtc", "qtc_prolongation", "delta_qtc"}):
            subtype = next(s for s in ("qtc_prolongation", "delta_qtc", "qtc") if s in subtypes)
        else:
            subtype = qt_hits[0][1]
        kw = [p for p, _ in qt_hits]
        return _pack(raw, normalized, compact, "qt_specific", subtype, kw,
                     f"QT/QTc-specific endpoint ({kw[0]}).")

    # Priority 2: QRS/PR/RR/conduction (only strict interval forms)
    if conduction_hits:
        kw = [p for p, _ in conduction_hits]
        return _pack(raw, normalized, compact, "ecg_conduction", conduction_hits[0][1], kw,
                     f"ECG conduction endpoint ({kw[0]}).")

    # Priority 3: cardiac AE
    if cardiac_ae_hits:
        kw = [p for p, _ in cardiac_ae_hits]
        return _pack(raw, normalized, compact, "cardiac_ae", cardiac_ae_hits[0][1], kw,
                     f"Cardiac adverse event endpoint ({kw[0]}).")

    # Priority 4: broad ECG — only if no QT/QTc/QRS/PR/RR
    if ecg_broad_hits:
        kw = [p for p, _ in ecg_broad_hits]
        return _pack(raw, normalized, compact, "ecg_broad", "ecg_broad", kw,
                     f"Broad ECG endpoint without specific interval ({kw[0]}).")

    if non_qt_hits:
        kw = [p for p, _ in non_qt_hits]
        return _pack(raw, normalized, compact, "non_qt", non_qt_hits[0][1], kw,
                     f"Non-QT/ECG endpoint ({kw[0]}).")

    return _pack(raw, normalized, compact, "non_qt", "unrelated", [],
                 "No QT/ECG electrophysiology keywords matched.")


def _pack(
    raw: str,
    normalized: str,
    compact: str,
    evidence_type: str,
    subtype: str,
    keywords: List[str],
    summary: str,
) -> Dict[str, Any]:
    return {
        "raw_text": raw,
        "normalized_text": normalized,
        "compact_text": compact,
        "evidence_type": evidence_type,
        "evidence_subtype": subtype,
        "matched_keywords": keywords,
        "evidence_summary": summary,
    }


# =============================================================================
# Outcome dict classifiers
# =============================================================================

def _classify_outcome_dict(
    om: Dict[str, Any],
    *,
    source: str,
) -> Dict[str, Any]:
    title = (om.get("title") or om.get("measure") or "").strip()
    measure = (om.get("measure") or om.get("title") or "").strip()
    description = (om.get("description") or "").strip()
    time_frame = (om.get("timeFrame") or om.get("time_frame") or "").strip()
    unit = (om.get("unitOfMeasure") or om.get("unit") or "").strip()
    classes = om.get("classes") or []

    raw_text = build_outcome_text(
        title=title,
        measure=measure,
        description=description,
        time_frame=time_frame,
        unit=unit,
        classes=classes,
    )

    cls = classify_outcome_text(raw_text)

    return {
        "source": source,
        "title": title,
        "measure": measure,
        "description": description,
        "time_frame": time_frame,
        "unit": unit,
        "param_type": (om.get("paramType") or om.get("param_type") or "").strip(),
        "dispersion_type": (om.get("dispersionType") or om.get("dispersion_type") or "").strip(),
        "groups": om.get("groups") or [],
        "classes": classes,
        **cls,
    }


def _bucket_key(evidence_type: str) -> str:
    return {
        "qt_specific": "qt_specific_outcomes",
        "ecg_conduction": "ecg_conduction_outcomes",
        "ecg_broad": "ecg_broad_outcomes",
        "cardiac_ae": "cardiac_ae_outcomes",
        "non_qt_cardiology": "non_qt_cardiology_outcomes",
        "non_qt": "non_qt_outcomes",
    }.get(evidence_type, "non_qt_outcomes")


def _aggregate_classified(
    classified: List[Dict[str, Any]],
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "classified_outcomes": classified,
        "qt_specific_outcomes": [],
        "ecg_conduction_outcomes": [],
        "ecg_broad_outcomes": [],
        "cardiac_ae_outcomes": [],
        "non_qt_cardiology_outcomes": [],
        "non_qt_outcomes": [],
        "has_qt_specific": False,
        "has_ecg_conduction": False,
        "has_ecg_broad": False,
        "has_cardiac_ae": False,
        "has_non_qt_cardiology": False,
        "best_evidence_type": "non_qt",
    }

    for entry in classified:
        evidence_type = entry.get("evidence_type", "non_qt")
        stats[_bucket_key(evidence_type)].append(entry)

    stats["has_qt_specific"] = bool(stats["qt_specific_outcomes"])
    stats["has_ecg_conduction"] = bool(stats["ecg_conduction_outcomes"])
    stats["has_ecg_broad"] = bool(stats["ecg_broad_outcomes"])
    stats["has_cardiac_ae"] = bool(stats["cardiac_ae_outcomes"])
    stats["has_non_qt_cardiology"] = bool(stats["non_qt_cardiology_outcomes"])

    best = "non_qt"
    for entry in classified:
        evidence_type = entry.get("evidence_type", "non_qt")
        if _EVIDENCE_PRIORITY.get(evidence_type, 0) > _EVIDENCE_PRIORITY.get(best, 0):
            best = evidence_type

    stats["best_evidence_type"] = best
    return stats


# =============================================================================
# Public API
# =============================================================================

def classify_protocol_outcomes_module(outcomes_module: Dict[str, Any]) -> Dict[str, Any]:
    """Classify all protocolSection.outcomesModule outcomes independently."""
    if not outcomes_module:
        return _aggregate_classified([])

    all_outcomes: List[Dict[str, Any]] = []

    for key in ("primaryOutcomes", "secondaryOutcomes", "otherOutcomes"):
        for om in outcomes_module.get(key) or []:
            if isinstance(om, dict):
                all_outcomes.append(om)

    classified = [
        _classify_outcome_dict(om, source="protocol")
        for om in all_outcomes
    ]

    stats = _aggregate_classified(classified)

    # aliases for protocol consumers
    stats["has_protocol_qt_specific"] = stats["has_qt_specific"]
    stats["protocol_qt_specific_outcomes"] = stats["qt_specific_outcomes"]
    stats["has_protocol_ecg_conduction"] = stats["has_ecg_conduction"]
    stats["protocol_ecg_conduction_outcomes"] = stats["ecg_conduction_outcomes"]
    stats["has_protocol_ecg_broad"] = stats["has_ecg_broad"]
    stats["protocol_ecg_broad_outcomes"] = stats["ecg_broad_outcomes"]
    stats["has_protocol_cardiac_ae"] = stats["has_cardiac_ae"]
    stats["protocol_cardiac_ae_outcomes"] = stats["cardiac_ae_outcomes"]

    return stats


def classify_results_outcome_measures(
    outcome_measures: List[Any],
) -> Dict[str, Any]:
    """Classify all resultsSection outcome measures independently."""
    classified = [
        _classify_outcome_dict(om, source="results")
        for om in outcome_measures
        if isinstance(om, dict)
    ]

    stats = _aggregate_classified(classified)

    # aliases for resultsSection consumers
    stats["has_qt_results"] = stats["has_qt_specific"]
    stats["qt_result_measures"] = stats["qt_specific_outcomes"]

    stats["has_ecg_conduction_results"] = stats["has_ecg_conduction"]
    stats["ecg_conduction_result_measures"] = stats["ecg_conduction_outcomes"]

    stats["has_ecg_broad_results"] = stats["has_ecg_broad"]
    stats["ecg_broad_result_measures"] = stats["ecg_broad_outcomes"]

    stats["has_cardiac_ae_results"] = stats["has_cardiac_ae"]
    stats["cardiac_ae_result_measures"] = stats["cardiac_ae_outcomes"]

    stats["has_results_outcome_measures"] = bool(classified)

    return stats


def is_strict_branch_protocol_signal(protocol_stats: Dict[str, Any]) -> bool:
    """Strict search branch.

    Use only qt_specific or ecg_conduction protocol outcomes.
    ECG broad is intentionally excluded from strict branch to reduce false positives.
    """
    return bool(
        protocol_stats.get("has_qt_specific")
        or protocol_stats.get("has_ecg_conduction")
    )


def is_recall_branch_results_signal(results_stats: Dict[str, Any]) -> bool:
    """Recall branch.

    Use resultsSection qt_specific outcomes only.
    ECG broad and conduction results are not enough for results-only QT recall.
    """
    return bool(results_stats.get("has_qt_results"))


# =============================================================================
# Optional quick self-test
# =============================================================================

if __name__ == "__main__":
    examples = [
        "12-lead ECG - corrected QT interval (QTc)",
        "Change from baseline in Q-T interval",
        "QTcF using Fridericia correction",
        "QRS duration",
        "PR interval",
        "ECG findings",
        "Cardiac safety",
        "Kallikrein 5 KLK5 Protease Activity",
        "Food cravings score",
        "AUC and Cmax exposure-QTc analysis",
        "Torsades de pointes",
        "qPCR gene expression",
    ]

    for text in examples:
        print("\nTEXT:", text)
        print(classify_outcome_text(text))