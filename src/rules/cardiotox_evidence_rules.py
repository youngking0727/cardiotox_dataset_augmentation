"""从 config/evidence_rules.yaml 加载关键词与分层规则；供 M3A/M3B/M3C/M5 共用。"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

EVIDENCE_PRIORITY_1 = "priority_1_direct_qt"
EVIDENCE_PRIORITY_2 = "priority_2_mechanistic_herg_ikr"
EVIDENCE_PRIORITY_3 = "priority_3_secondary_pharmacology"

_PRIORITY_ORDER = {
    EVIDENCE_PRIORITY_1: 3,
    EVIDENCE_PRIORITY_2: 2,
    EVIDENCE_PRIORITY_3: 1,
}


def priority_rank(name: Optional[str]) -> int:
    if not name:
        return 0
    return _PRIORITY_ORDER.get(name, 0)


def _project_config_path() -> Path:
    here = Path(__file__).resolve()
    # src/rules/cardiotox_evidence_rules.py -> project root
    return here.parents[2] / "config" / "evidence_rules.yaml"


def _fallback_rules() -> Dict[str, Any]:
    """PyYAML 不可用或配置文件缺失时使用（与 config/evidence_rules.yaml 保持同步为理想状态）。"""
    return {
        "direct_qt_terms": [
            "QT", "QTc", "TQT", "long QT", "torsade", "torsades de pointes", "TdP",
            "proarrhythmia", "FPD", "FPDc", "APD", "field potential duration",
            "action potential duration", "hiPSC-CM", "MEA", "ventricular repolarization",
        ],
        "mechanistic_terms": [
            "hERG", "IKr", "KCNH2", "blocker", "blockade", "cardiac ion channel",
            "potassium channel", "long QT pharmacophore",
        ],
        "secondary_terms": [
            "secondary pharmacology", "safety pharmacology", "off-target", "ADR",
            "adverse drug reaction", "cardiotoxicity risk", "safety panel",
            "liability panel", "AC50",
        ],
        "direct_qt_assay_keywords": [
            "FPD", "QTc", "APD", "MEA", "hiPSC-CM", "field potential",
            "action potential", "repolarization",
        ],
        "mechanistic_measurement_types": [
            "IC50", "Ki", "Kd", "EC50", "AC50", "PIC50", "INHIBITION",
        ],
        "mechanism_context_keywords": [
            "hERG", "IKr", "KCNH2", "potassium channel", "ion channel",
            "long QT pharmacophore", "cardiac ion channel", "QT", "ventricular repolarization",
        ],
        "secondary_pharm_context_keywords": [
            "secondary pharmacology", "safety pharmacology", "off-target",
            "adverse", "ADR", "cardiotoxicity", "liability", "safety panel",
        ],
        "cardiac_risk_bridge_keywords": [
            "QT", "arrhythmia", "cardiac", "proarrhythmia", "torsade",
            "cardiotoxicity", "repolarization", "ion channel",
        ],
        "clinical_direct_qt_keywords": [
            "QT", "QT prolongation", "long QT", "torsade", "arrhythmia",
            "ventricular tachycardia", "proarrhythmia",
        ],
        "type_normalization_map": {
            "IC50": "IC50",
            "Ki": "Ki",
            "Kd": "Kd",
            "EC50": "EC50",
            "AC50": "AC50",
            "PIC50": "PIC50",
            "pIC50": "PIC50",
            "Inhibition": "Inhibition",
        },
    }


@lru_cache(maxsize=1)
def get_evidence_rules() -> Dict[str, Any]:
    path = _project_config_path()
    if path.is_file() and yaml is not None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data:
                    return data
        except Exception:
            pass
    elif path.is_file() and yaml is None:
        # 无 PyYAML：保持可导入，规则用 fallback
        pass
    return _fallback_rules()


def _lower_text(*parts: str) -> str:
    return " ".join(p for p in parts if p).lower()


def _any_term_in_text(text: str, terms: List[str]) -> List[str]:
    """返回在 text 中命中的词（子串匹配，大小写不敏感）。"""
    t = text.lower()
    hits: List[str] = []
    for term in terms:
        if not term:
            continue
        if term.lower() in t:
            hits.append(term)
    return hits


def normalize_standard_type(raw: Any, rules: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """将 ChEMBL standard_type 归一化为规范 token（大写比较）。"""
    if raw is None:
        return None
    key = str(raw).strip()
    if not key:
        return None
    r = rules or get_evidence_rules()
    mmap: Dict[str, str] = r.get("type_normalization_map") or {}
    # 精确键
    if key in mmap:
        return str(mmap[key]).strip()
    uk = key.upper().replace(" ", "")
    for a, b in mmap.items():
        if str(a).upper().replace(" ", "") == uk:
            return str(b).strip()
    for token in ("IC50", "KI", "KD", "EC50", "AC50", "PIC50", "INHIBITION"):
        if token in uk:
            return token
    # 允许已是规范形式
    return key


def clinical_text_has_direct_qt(*texts: str) -> bool:
    rules = get_evidence_rules()
    terms = list(rules.get("clinical_direct_qt_keywords") or [])
    blob = _lower_text(*texts)
    return bool(_any_term_in_text(blob, terms))


def classify_literature_text_buckets(
    title: str,
    abstract: str,
    mesh_terms: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    """
    单篇文献 relevance：返回 (bucket, reason_codes)。
    bucket: priority_1_direct_qt | priority_2_mechanistic_herg_ikr | priority_3_secondary_pharmacology | irrelevant
    """
    rules = get_evidence_rules()
    mesh_s = " ".join(mesh_terms or [])
    blob = _lower_text(title, abstract or "", mesh_s)
    reasons: List[str] = []

    d1 = list(rules.get("direct_qt_terms") or [])
    hits1 = _any_term_in_text(blob, d1)
    if hits1:
        reasons.append(f"direct_qt_term:{hits1[0]}")
        return EVIDENCE_PRIORITY_1, reasons

    mech = list(rules.get("mechanistic_terms") or [])
    hits2 = _any_term_in_text(blob, mech)
    if hits2:
        reasons.append(f"mechanistic_term:{hits2[0]}")
        return EVIDENCE_PRIORITY_2, reasons

    sec_kw = list(rules.get("secondary_terms") or [])
    bridge = list(rules.get("cardiac_risk_bridge_keywords") or [])
    pharm_ctx = list(rules.get("secondary_pharm_context_keywords") or [])
    has_sec = bool(_any_term_in_text(blob, sec_kw))
    has_bridge = bool(_any_term_in_text(blob, bridge))
    has_pharm = bool(_any_term_in_text(blob, pharm_ctx))
    # priority_3：需「次药理学/ADR 语境」且与心脏/QT 风险有可解释关联，避免泛泛毒理
    if has_sec and has_bridge and has_pharm:
        reasons.append("secondary_pharm_plus_cardiac_bridge")
        return EVIDENCE_PRIORITY_3, reasons
    if has_sec and has_bridge:
        reasons.append("secondary_terms_plus_cardiac_risk")
        return EVIDENCE_PRIORITY_3, reasons

    return "irrelevant", []


def classify_activity_evidence_dict(
    activity: Dict[str, Any],
    target_name: str,
    target_chembl_id: str,
) -> Dict[str, Any]:
    """
    对单条 ChEMBL activity dict 做证据桶分类。
    返回字段供写入 BioactivityMeasurement 或 supplemental row。
    """
    rules = get_evidence_rules()
    st_raw = activity.get("standard_type")
    norm = normalize_standard_type(st_raw, rules) or (str(st_raw).strip() if st_raw else "")
    assay = str(activity.get("assay_description") or "")
    atype = str(activity.get("assay_type") or "")
    blob = _lower_text(norm, assay, atype, target_name, target_chembl_id)

    direct_kw = list(rules.get("direct_qt_assay_keywords") or [])
    if _any_term_in_text(blob, direct_kw):
        return {
            "normalized_type": norm or None,
            "evidence_bucket": "direct_qt",
            "direct_qt_context_hit": True,
            "mechanistic_context_hit": False,
            "secondary_pharmacology_context_hit": False,
            "priority_guess": EVIDENCE_PRIORITY_1,
        }

    mech_types = {str(x).upper().replace(" ", "") for x in (rules.get("mechanistic_measurement_types") or [])}
    nt = (norm or "").upper().replace(" ", "")
    type_is_mech = nt in mech_types or any(m in nt for m in ("IC50", "KI", "KD", "PIC50", "INHIBITION"))

    mech_ctx = list(rules.get("mechanism_context_keywords") or [])
    has_mech_ctx = bool(_any_term_in_text(blob, mech_ctx))
    herg_like = "herg" in blob or "CHEMBL240" in (target_chembl_id or "").upper() or target_name.upper() == "HERG"

    if type_is_mech and has_mech_ctx and herg_like:
        return {
            "normalized_type": norm or None,
            "evidence_bucket": "mechanistic_herg_ikr",
            "direct_qt_context_hit": False,
            "mechanistic_context_hit": True,
            "secondary_pharmacology_context_hit": False,
            "priority_guess": EVIDENCE_PRIORITY_2,
        }

    sec_ctx = list(rules.get("secondary_pharm_context_keywords") or [])
    bridge = list(rules.get("cardiac_risk_bridge_keywords") or [])
    has_sec = bool(_any_term_in_text(blob, sec_ctx))
    has_bridge = bool(_any_term_in_text(blob, bridge))
    ac50_like = "AC50" in nt or "ac50" in assay.lower()
    if ac50_like and has_sec and has_bridge:
        return {
            "normalized_type": norm or None,
            "evidence_bucket": "secondary_pharmacology",
            "direct_qt_context_hit": False,
            "mechanistic_context_hit": False,
            "secondary_pharmacology_context_hit": True,
            "priority_guess": EVIDENCE_PRIORITY_3,
        }

    return {
        "normalized_type": norm or None,
        "evidence_bucket": "unknown",
        "direct_qt_context_hit": False,
        "mechanistic_context_hit": False,
        "secondary_pharmacology_context_hit": False,
        "priority_guess": None,
    }
