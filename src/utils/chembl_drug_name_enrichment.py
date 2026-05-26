"""ChEMBL molecule API enrichment for ClinicalTrials drug_name_set construction."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── synonym type buckets (ChEMBL syn_type) ────────────────────────────────────

_STRONG_SYN_TYPES = {
    "INN", "USAN", "BAN", "JAN", "DCF", "FDA", "EP", "BP", "OTHER",
}
_WEAK_SYN_TYPES = {
    "TRADE_NAME", "RESEARCH_CODE", "CODEN", "MERCK_INDEX", "SYSTEMATIC",
}
# parent / sibling / curated enantiomer pairs → related (recall_audit only)
_CURATED_RELATED_BY_PREF: Dict[str, List[str]] = {
    "LEVOSALBUTAMOL": ["salbutamol", "albuterol"],
    "SALBUTAMOL": ["levosalbutamol", "levalbuterol"],
}

_SALT_SUFFIXES = (
    " hydrochloride", " hcl", " mesylate", " maleate", " fumarate",
    " palmitate", " sodium", " potassium", " sulfate", " sulphate",
    " tartrate", " besylate", " dihydrate", " monohydrate", " acetate",
    " phosphate", " citrate", " tosylate", " succinate", " lactate",
)

_GENERIC_WEAK_TERMS = {
    "drug", "compound", "product", "agent", "medicine", "tablet",
    "capsule", "injection", "solution", "placebo", "standard",
    "control", "test", "study", "formulation", "mixture",
}

_PUNCT_RE = re.compile(r"[^\w\s/-]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_DIGIT_ONLY_RE = re.compile(r"^\d+$")
_RESEARCH_CODE_RE = re.compile(
    r"^(?:[A-Z]{1,4}[-_/]?\d{2,}|(?:GSK|PF|RO|BMS|AZ|NN|LY|MK|CP|SB)[- ]?\d+)$",
    re.IGNORECASE,
)

_MIN_STRONG_TERM_LEN = 4
_MIN_WEAK_TERM_LEN = 3


def normalize_drug_term(raw: str) -> str:
    """lower + 去标点 + 合并空白。"""
    text = (raw or "").strip().lower()
    text = text.replace("β", "beta").replace("α", "alpha")
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def strip_salt_form(term: str) -> str:
    n = normalize_drug_term(term)
    for suffix in _SALT_SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    return n


def _is_ambiguous_term(term: str, *, for_strong: bool) -> bool:
    if not term:
        return True
    if _DIGIT_ONLY_RE.match(term):
        return True
    if term in _GENERIC_WEAK_TERMS:
        return True
    min_len = _MIN_STRONG_TERM_LEN if for_strong else _MIN_WEAK_TERM_LEN
    if len(term) < min_len:
        return True
    if len(term.split()) == 1 and len(term) <= 2:
        return True
    if _RESEARCH_CODE_RE.match(term):
        return True
    return False


def _add_term(
    bucket: List[str],
    seen: Set[str],
    raw: str,
    *,
    for_strong: bool,
) -> None:
    for candidate in (normalize_drug_term(raw), strip_salt_form(raw)):
        if not candidate or candidate in seen:
            continue
        if _is_ambiguous_term(candidate, for_strong=for_strong):
            continue
        seen.add(candidate)
        bucket.append(candidate)


def _extract_synonym_entries(molecule: Dict[str, Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for entry in molecule.get("molecule_synonyms") or []:
        if not isinstance(entry, dict):
            continue
        name = (
            entry.get("molecule_synonym")
            or entry.get("synonyms")
            or entry.get("synonym")
            or ""
        ).strip()
        syn_type = (entry.get("syn_type") or entry.get("synonym_type") or "OTHER").strip().upper()
        if name:
            out.append((name, syn_type))
    return out


def _extract_hierarchy(molecule: Dict[str, Any]) -> Dict[str, str]:
    hier = molecule.get("molecule_hierarchy") or {}
    if not isinstance(hier, dict):
        return {}
    return {
        "molecule_chembl_id": (hier.get("molecule_chembl_id") or "").strip(),
        "parent_chembl_id": (hier.get("parent_chembl_id") or "").strip(),
    }


def _classify_synonym_to_bucket(syn_type: str) -> str:
    st = (syn_type or "OTHER").upper()
    if st in _STRONG_SYN_TYPES:
        return "strong"
    if st in _WEAK_SYN_TYPES:
        return "weak"
    return "weak"


def enrich_drug_name_set_from_molecule(
    molecule: Dict[str, Any],
    *,
    pref_name_fallback: str = "",
    parent_molecule: Optional[Dict[str, Any]] = None,
    curated_related: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    从 ChEMBL molecule JSON 构建 drug_name_set。

    strong_match_terms  → query.intr / direct evidence
    related_match_terms → recall_audit only
    weak_terms          → recall_audit only
    """
    chembl_id = (molecule.get("molecule_chembl_id") or "").strip()
    pref_name = (molecule.get("pref_name") or pref_name_fallback or "").strip()
    hierarchy = _extract_hierarchy(molecule)
    synonym_entries = _extract_synonym_entries(molecule)

    strong: List[str] = []
    related: List[str] = []
    weak: List[str] = []
    seen_strong: Set[str] = set()
    seen_related: Set[str] = set()
    seen_weak: Set[str] = set()

    # pref_name → strong
    _add_term(strong, seen_strong, pref_name, for_strong=True)

    # molecule_synonyms
    raw_synonyms: List[Dict[str, str]] = []
    for name, syn_type in synonym_entries:
        raw_synonyms.append({"name": name, "syn_type": syn_type})
        bucket = _classify_synonym_to_bucket(syn_type)
        if bucket == "strong":
            _add_term(strong, seen_strong, name, for_strong=True)
        else:
            _add_term(weak, seen_weak, name, for_strong=False)

    # parent molecule → related
    parent_pref_name = ""
    parent_chembl_id = hierarchy.get("parent_chembl_id", "")
    if parent_molecule and isinstance(parent_molecule, dict):
        parent_pref_name = (parent_molecule.get("pref_name") or "").strip()
        _add_term(related, seen_related, parent_pref_name, for_strong=False)
        for name, syn_type in _extract_synonym_entries(parent_molecule):
            if _classify_synonym_to_bucket(syn_type) == "strong":
                _add_term(related, seen_related, name, for_strong=False)

    # curated related (enantiomer / racemate pairs)
    for name in curated_related or []:
        _add_term(related, seen_related, name, for_strong=False)

    pref_key = pref_name.upper()
    for name in _CURATED_RELATED_BY_PREF.get(pref_key, []):
        _add_term(related, seen_related, name, for_strong=False)

    # strong terms should not appear in related/weak
    strong_set = set(strong)
    related = [t for t in related if t not in strong_set]
    weak = [t for t in weak if t not in strong_set and t not in set(related)]

    recall_audit_terms = sorted(set(related) | set(weak))

    return {
        "molecule_chembl_id": chembl_id,
        "pref_name": pref_name,
        "chembl_pref_name": (molecule.get("pref_name") or "").strip(),
        "molecule_synonyms": raw_synonyms,
        "molecule_hierarchy": hierarchy,
        "parent_molecule_chembl_id": parent_chembl_id,
        "parent_pref_name": parent_pref_name,
        "strong_match_terms": strong,
        "related_match_terms": related,
        "weak_terms": weak,
        "recall_audit_terms": recall_audit_terms,
        "exclude_names": [],
        # internal aliases consumed by alignment / client
        "_strong_match_terms": strong,
        "_related_match_terms": related,
        "_weak_terms": weak,
        "_all_terms": sorted(set(strong) | set(related) | set(weak)),
    }


class ChemblDrugNameEnricher:
    """Fetch ChEMBL molecule(s) and build drug_name_set with optional file cache."""

    def __init__(
        self,
        chembl_client: Any,
        cache_dir: Optional[Path] = None,
    ):
        self.chembl_client = chembl_client
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, chembl_id: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        safe = re.sub(r"[^\w.-]", "_", chembl_id)
        return self.cache_dir / f"{safe}.json"

    def _load_cache(self, chembl_id: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(chembl_id)
        if not path or not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("读取 ChEMBL drug_name_set 缓存失败 %s: %s", chembl_id, e)
            return None

    def _save_cache(self, chembl_id: str, payload: Dict[str, Any]) -> None:
        path = self._cache_path(chembl_id)
        if not path:
            return
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("写入 ChEMBL drug_name_set 缓存失败 %s: %s", chembl_id, e)

    def fetch_molecule(self, chembl_id: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        mol, ok = self.chembl_client.get_molecule_by_chembl_id(chembl_id)
        if not ok:
            return None, False
        return mol, True

    def enrich(
        self,
        molecule_chembl_id: str,
        *,
        pref_name_fallback: str = "",
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        对 molecule_chembl_id 拉取 ChEMBL API 并构建 drug_name_set。
        API 失败时回退到 pref_name_fallback 的最小 strong set。
        """
        chembl_id = (molecule_chembl_id or "").strip().upper()
        if use_cache:
            cached = self._load_cache(chembl_id)
            if cached:
                return cached

        if not chembl_id:
            payload = enrich_drug_name_set_from_molecule(
                {"pref_name": pref_name_fallback},
                pref_name_fallback=pref_name_fallback,
            )
            return payload

        molecule, ok = self.fetch_molecule(chembl_id)
        if not ok or not molecule:
            logger.warning(
                "ChEMBL molecule 获取失败，回退 pref_name only: %s", chembl_id
            )
            payload = enrich_drug_name_set_from_molecule(
                {"molecule_chembl_id": chembl_id, "pref_name": pref_name_fallback},
                pref_name_fallback=pref_name_fallback,
            )
            payload["enrichment_status"] = "chembl_fetch_failed"
            return payload

        hierarchy = _extract_hierarchy(molecule)
        parent_molecule: Optional[Dict[str, Any]] = None
        parent_id = hierarchy.get("parent_chembl_id", "")
        if parent_id and parent_id.upper() != chembl_id:
            parent_molecule, parent_ok = self.fetch_molecule(parent_id)
            if not parent_ok:
                parent_molecule = None

        payload = enrich_drug_name_set_from_molecule(
            molecule,
            pref_name_fallback=pref_name_fallback,
            parent_molecule=parent_molecule,
        )
        payload["enrichment_status"] = "ok"
        if use_cache:
            self._save_cache(chembl_id, payload)
        return payload


def build_drug_name_set_fallback(pref_name: str) -> Dict[str, Any]:
    """无 ChEMBL ID 时的最小 drug_name_set。"""
    return enrich_drug_name_set_from_molecule(
        {"pref_name": pref_name},
        pref_name_fallback=pref_name,
    )
