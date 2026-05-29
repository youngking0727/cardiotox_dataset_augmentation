"""ClinicalTrials.gov API客户端封装"""

import re
import time
import logging
import copy
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import requests

from utils.clinicaltrials_drug_alignment import (
    assess_drug_trial_alignment,
    assess_qt_result_attribution,
    build_drug_name_set,
    classify_evidence_tier,
    extract_trial_drug_fields,
    match_terms_in_text,
)
from utils.clinicaltrials_result_evidence import (
    classify_outcome_text,
    classify_protocol_outcomes_module,
    classify_results_outcome_measures,
    is_qt_specific_outcome_text,
    is_recall_branch_results_signal,
    is_strict_branch_protocol_signal,
    pick_protocol_qt_outcome_measure,
)

logger = logging.getLogger(__name__)


class ClinicalTrialsClient:
    """ClinicalTrials.gov API v2客户端"""

    API_BASE_URL = "https://clinicaltrials.gov/api/v2"

    # API 查询用：仅用语义较窄的 QT/hERG/复极相关词，避免「心律失常」等泛词拉爆噪声
    QT_QUERY_TERMS = [
        "QT prolongation",
        "QT interval",
        "QTc",
        "long QT",
        "torsades de pointes",
        "hERG",
        "KCNH2",
        "electrocardiogram",
        "Q-T interval",
        "Q-Tc interval",
        "Q-Tc prolongation",
        "Q-T prolongation",
        "Q-T"
    ]

    # 标题/摘要/resultsSection 判定 qt_related=True 的「强相关」短语（全部小写）。
    # 不含单独的 ecg/pr/qrs/rr；只有与 QT/QTc/TdP/hERG/IKr 直接关联的词才列入。
    QT_STRICT_MATCH_PHRASES = [
        "qt prolongation",
        "qt interval",
        "qtc",
        "qtcf",                 # Fridericia-corrected QT
        "qtcb",                 # Bazett-corrected QT
        "qtc prolongation",
        "corrected qt",
        "long qt",
        "long qt syndrome",
        "torsade",
        "torsades",
        "torsade de pointes",
        "tqt",                  # thorough QT study
        "herg",
        "kcnh2",
        "ikr",                  # cardiac rapid delayed rectifier K+ current
        "prolonged qt",
        "qt prolonged",
        "electrocardiogram qt",
        "ecg qt",
        "q-t interval",
        "q-tc interval",
        "q-tc prolongation",
        "q-t prolongation",
        "q-t",
    ]

    # 从文本中剔除 PCR 相关词，防止 qt-pcr/qpcr/rt-pcr 误匹配
    _PCR_EXCLUSION_RE = re.compile(r'\b(?:qt-pcr|qpcr|rt-pcr)\b', re.IGNORECASE)

    def __init__(self, cache_dir: Optional[Path] = None, rate_limit: float = 0.5):
        """
        初始化ClinicalTrials客户端

        Args:
            cache_dir: 缓存目录
            rate_limit: API调用间隔（秒）
        """
        self.cache_dir = cache_dir
        self.rate_limit = rate_limit
        self._last_call_time = 0
        self.session = requests.Session()

    def _rate_limit_wait(self):
        """速率限制等待"""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_call_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """发送API请求"""
        self._rate_limit_wait()
        url = f"{self.API_BASE_URL}/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"ClinicalTrials API请求失败: {url}, 错误: {e}")
            return None

    # ── credibility helpers ─────────────────────────────────────────────────

    # sponsor_class 在 _assess_institution_credibility 中已规范化为 upper+underscore，
    # 故此处使用规范化后的形式（兼容 API 返回的 "Other Gov" → "OTHER_GOV"）
    _HIGH_CRED_CLASSES = {"NIH", "FED", "U.S._FED", "OTHER_GOV"}

    _HIGH_CRED_KEYWORDS = [
        "fda", "nih", "national institutes of health", "national cancer institute",
        "national heart", "european medicines agency", "ema", "who", "cdc",
        "government", "ministry of health", "university", "hospital",
        "medical center", "academic medical center",
    ]

    _INDUSTRY_KEYWORDS = [
        " inc", " ltd", " llc", " gmbh", " s.a.", " ag ", "pharmaceutical",
        "pharma", "biotech", "therapeutics", "biosciences",
    ]

    _LARGE_PHARMA = [
        "janssen", "pfizer", "novartis", "roche", "merck", "astrazeneca",
        "sanofi", "gsk", "glaxosmithkline", "eli lilly", "bayer",
        "bristol-myers squibb", "takeda", "amgen", "abbvie",
        "boehringer", "johnson & johnson", "johnson and johnson",
    ]

    _ACADEMIC_COLLAB_KEYWORDS = [
        "university", "hospital", "medical center", "nih", "government",
    ]

    @classmethod
    def _assess_institution_credibility(
        cls, sponsor_module: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        根据 sponsorCollaboratorsModule 评估研究机构可信度。
        不修改任何 QT 判断逻辑。
        """
        empty = {
            "level": "unknown",
            "sponsor_name": "",
            "sponsor_class": "",
            "responsible_party": "",
            "collaborators": [],
            "reason_codes": ["missing_sponsor_information"],
            "summary": "Unknown credibility: sponsor information is missing.",
        }
        if not sponsor_module or not isinstance(sponsor_module, dict):
            return empty

        lead = sponsor_module.get("leadSponsor") or {}
        sponsor_name: str = (lead.get("name") or "").strip()
        # 规范化：统一大写并将空格替换为下划线（e.g. "Other Gov" → "OTHER_GOV"）
        sponsor_class: str = (lead.get("class") or "").strip().upper().replace(" ", "_")

        rp_raw = sponsor_module.get("responsibleParty") or {}
        responsible_party: str = (
            rp_raw.get("type") or rp_raw.get("investigatorFullName") or ""
        ).strip()

        collab_list = sponsor_module.get("collaborators") or []
        collaborators: List[str] = [
            (c.get("name") or "").strip()
            for c in collab_list
            if isinstance(c, dict) and (c.get("name") or "").strip()
        ]

        if not sponsor_name and not sponsor_class:
            return {**empty, "responsible_party": responsible_party,
                    "collaborators": collaborators}

        name_l = sponsor_name.lower()
        reason_codes: List[str] = []
        level = "unknown"

        # ── High: government / academic ─────────────────────────────────────
        if sponsor_class in cls._HIGH_CRED_CLASSES:
            level = "high"
            if sponsor_class == "NIH":
                reason_codes.append("nih_sponsor")
            elif sponsor_class in {"FED", "U.S._FED", "OTHER_GOV"}:
                reason_codes.append("government_sponsor")
            else:
                reason_codes.append("government_sponsor")
        elif any(kw in name_l for kw in cls._HIGH_CRED_KEYWORDS):
            level = "high"
            if "university" in name_l:
                reason_codes.append("academic_sponsor")
            elif any(w in name_l for w in ["hospital", "medical center"]):
                reason_codes.append("hospital_sponsor")
            elif any(w in name_l for w in ["nih", "national institutes"]):
                reason_codes.append("nih_sponsor")
            elif any(w in name_l for w in ["government", "ministry"]):
                reason_codes.append("government_sponsor")
            else:
                reason_codes.append("academic_sponsor")

        # ── Medium: industry ─────────────────────────────────────────────────
        elif sponsor_class == "INDUSTRY" or any(  # "INDUSTRY" 已因规范化保持大写
            kw in name_l for kw in cls._INDUSTRY_KEYWORDS
        ):
            level = "medium"
            reason_codes.append("industry_sponsor")
            if any(kw in name_l for kw in cls._LARGE_PHARMA):
                reason_codes.append("large_pharma_sponsor")
            collab_names_l = " ".join(collaborators).lower()
            if any(kw in collab_names_l for kw in cls._ACADEMIC_COLLAB_KEYWORDS):
                reason_codes.append("industry_with_academic_or_government_collaborator")

        # ── Low: individual / unrecognised ───────────────────────────────────
        else:
            level = "low"
            reason_codes.append(
                "individual_or_small_clinic_sponsor"
                if not any(kw in name_l for kw in cls._INDUSTRY_KEYWORDS + cls._HIGH_CRED_KEYWORDS)
                else "unclear_sponsor_type"
            )

        # ── summary text ─────────────────────────────────────────────────────
        summary_map = {
            "high": (
                "High credibility: the study is sponsored by a government, academic, "
                "or recognised public health institution with strong ethical oversight."
            ),
            "medium": (
                "Medium credibility: the study is sponsored by an industry organisation. "
                "It is likely conducted under clinical trial regulations, but potential "
                "commercial interest should be considered."
            ),
            "low": (
                "Low credibility: the sponsor is an individual, small clinic, or "
                "unrecognised entity. Treat evidence with caution."
            ),
        }

        return {
            "level": level,
            "sponsor_name": sponsor_name,
            "sponsor_class": sponsor_class,
            "responsible_party": responsible_party,
            "collaborators": collaborators,
            "reason_codes": reason_codes,
            "summary": summary_map.get(level, "Unknown credibility."),
        }

    # ── search methods ───────────────────────────────────────────────────────

    # search_studies 默认返回字段（含 intervention / arm）
    _DEFAULT_SEARCH_FIELDS = [
        "protocolSection.identificationModule",
        "protocolSection.statusModule",
        "protocolSection.descriptionModule",
        "protocolSection.conditionsModule",
        "protocolSection.outcomesModule",
        "protocolSection.sponsorCollaboratorsModule",
        "protocolSection.armsInterventionsModule",
    ]

    # Controlled broad recall terms. These are used only after drug-only
    # intervention recall and are never sufficient by themselves for Primary.
    # A broad candidate must still contain qt_specific results/protocol evidence
    # and pass downstream drug/result attribution.
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

    def search_studies(
        self,
        query: str,
        max_results: int = 50,
        fields: Optional[List[str]] = None,
        query_field: str = "query.term",
    ) -> List[Dict[str, Any]]:
        """
        搜索临床试验

        Args:
            query: 搜索查询
            max_results: 最大结果数
            fields: 要返回的字段列表
            query_field: API 查询字段，如 query.term / query.intr

        Returns:
            试验 protocolSection 列表
        """
        if fields is None:
            fields = self._DEFAULT_SEARCH_FIELDS

        params = {
            query_field: query,
            "pageSize": max_results,
            "fields": ",".join(fields),
        }

        data = self._make_request("studies", params)
        if not data:
            return []

        studies = data.get("studies", [])
        return [s.get("protocolSection", {}) for s in studies]

    def search_studies_by_intervention(
        self, drug_name: str, max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """Step 1：在 intervention 字段中精确查药物。"""
        return self.search_studies(
            drug_name, max_results=max_results, query_field="query.intr"
        )

    def get_study_by_nct_id(self, nct_id: str) -> Optional[Dict[str, Any]]:
        """
        通过NCT ID获取试验详情

        Args:
            nct_id: NCT ID

        Returns:
            试验详情
        """
        data = self._make_request(f"studies/{nct_id}")
        return data

    def _parse_protocol_study_basics(self, study: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """解析 protocol 基础字段，不要求 QT 信号。"""
        if not study:
            return None

        identification = study.get("identificationModule", {})
        status = study.get("statusModule", {})
        description = study.get("descriptionModule", {})

        nct_id = identification.get("nctId", "")
        if not nct_id:
            return None

        title = identification.get("briefTitle", "")
        summary = description.get("briefSummary", "") if description else ""

        sponsor_module = study.get("sponsorCollaboratorsModule") or {}
        credibility = self._assess_institution_credibility(sponsor_module)

        return {
            "nct_id": nct_id,
            "title": title,
            "status": status.get("overallStatus", "Unknown"),
            "sponsor_name": credibility.get("sponsor_name", ""),
            "sponsor_class": credibility.get("sponsor_class", ""),
            "institution_credibility_level": credibility.get("level", "unknown"),
            "summary": summary if summary else "",
            "research_institution_credibility": credibility,
        }

    def _parse_protocol_study(self, study: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """从 protocolSection 解析 trial 字段与逐条 protocol outcome 分类（strict branch）。"""
        basics = self._parse_protocol_study_basics(study)
        if not basics:
            return None

        title = basics["title"]
        summary = basics["summary"]
        outcomes = study.get("outcomesModule", {}) or {}

        protocol_outcomes = classify_protocol_outcomes_module(outcomes)
        title_cls = classify_outcome_text(f"{title} {summary}".strip())

        has_strict_signal = (
            is_strict_branch_protocol_signal(protocol_outcomes)
            or title_cls.get("evidence_type") in {"qt_specific", "ecg_conduction"}
        )
        if not has_strict_signal:
            return None

        qt_outcome_measure = ""
        for bucket in (
            "qt_specific_outcomes",
            "ecg_conduction_outcomes",
            "ecg_broad_outcomes",
            "cardiac_ae_outcomes",
        ):
            items = protocol_outcomes.get(bucket) or []
            if items:
                qt_outcome_measure = items[0].get("measure") or items[0].get("title") or ""
                break

        return {
            **basics,
            "qt_related": True,
            "qt_related_title": title_cls.get("evidence_type") == "qt_specific",
            "qt_related_outcome": protocol_outcomes.get("has_qt_specific", False),
            "qt_outcome_measure": qt_outcome_measure,
            "search_branch": "strict_protocol_qt",
            "protocol_qt_hit": True,
            "results_qt_hit": False,
            "protocol_outcomes": protocol_outcomes,
            "title_outcome_classification": title_cls,
        }

    @classmethod
    def _strong_intervention_match(
        cls, drug_name_set: Dict[str, Any], study_protocol: Dict[str, Any]
    ) -> bool:
        """目标分子（pref_name/synonym）是否出现在 intervention / arm 字段。"""
        fields = extract_trial_drug_fields({"protocolSection": study_protocol})
        strong_terms = drug_name_set.get("_strong_match_terms") or []
        texts: List[str] = []
        for iv in fields.get("interventions") or []:
            texts.append(" ".join([iv.get("name", ""), iv.get("description", "")]))
        for ag in fields.get("arm_groups") or []:
            texts.append(" ".join([ag.get("label", ""), ag.get("description", "")]))
        return any(match_terms_in_text(strong_terms, t) for t in texts if t.strip())


    @classmethod
    def _event_group_lookup(cls, adverse_events_module: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Build adverse-event groupId -> title/description map."""
        lookup: Dict[str, Dict[str, str]] = {}
        for grp in (adverse_events_module or {}).get("eventGroups") or []:
            if not isinstance(grp, dict):
                continue
            gid = str(grp.get("id") or "").strip()
            if not gid:
                continue
            lookup[gid] = {
                "id": gid,
                "title": str(grp.get("title") or grp.get("label") or "").strip(),
                "description": str(grp.get("description") or "").strip(),
            }
        return lookup

    @classmethod
    def _collect_group_ids_recursive(cls, obj: Any, out: Set[str]) -> None:
        """Collect groupId/group_id/id fields from nested AE event stats/categories."""
        if isinstance(obj, dict):
            for key in ("groupId", "group_id", "groupID"):
                val = obj.get(key)
                if val:
                    out.add(str(val).strip())
            for value in obj.values():
                cls._collect_group_ids_recursive(value, out)
        elif isinstance(obj, list):
            for item in obj:
                cls._collect_group_ids_recursive(item, out)

    @classmethod
    def _event_text(cls, event: Dict[str, Any]) -> str:
        """Flatten clinically relevant AE event fields for QT-specific classification."""
        parts: List[str] = []
        for key in (
            "term", "title", "name", "organSystem", "sourceVocabulary",
            "assessmentType", "notes", "description",
        ):
            val = event.get(key)
            if val:
                parts.append(str(val))
        for key in ("categories", "measurements", "stats"):
            val = event.get(key)
            if val:
                parts.append(str(val))
        return " ".join(parts)

    @classmethod
    def _iter_adverse_event_records(
        cls,
        adverse_events_module: Dict[str, Any],
    ) -> List[tuple[str, Dict[str, Any]]]:
        """
        Yield adverse-event records from both common ClinicalTrials.gov shapes:
        1) seriousEvents/otherEvents as a flat list of event dicts;
        2) seriousEvents/otherEvents as group/system dicts containing events[].
        """
        records: List[tuple[str, Dict[str, Any]]] = []
        for source_key in ("seriousEvents", "otherEvents", "events"):
            items = adverse_events_module.get(source_key) or []
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue

                nested_events = item.get("events")
                if isinstance(nested_events, list):
                    parent_context = {
                        "parent_term": item.get("term") or item.get("title") or item.get("name") or "",
                        "parent_organ_system": item.get("organSystem") or "",
                        "parent_source_vocabulary": item.get("sourceVocabulary") or "",
                    }
                    for ev in nested_events:
                        if isinstance(ev, dict):
                            merged = dict(ev)
                            merged.update({k: v for k, v in parent_context.items() if v})
                            records.append((source_key, merged))
                    continue

                records.append((source_key, item))
        return records

    @classmethod
    def _classify_adverse_event_results(cls, adverse_events_module: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse QT/ECG evidence from resultsSection.adverseEventsModule.

        ClinicalTrials.gov sometimes reports QTcB/QTcF thresholds as AE/event-table
        results instead of outcomeMeasuresModule. This parser is generic and applies
        to all drugs. It supports both flat and nested adverse-event structures.
        """
        out: Dict[str, Any] = {
            "qt_result_measures": [],
            "ecg_conduction_result_measures": [],
            "ecg_broad_result_measures": [],
            "cardiac_ae_result_measures": [],
            "classified_outcomes": [],
        }
        if not adverse_events_module or not isinstance(adverse_events_module, dict):
            return out

        group_lookup = cls._event_group_lookup(adverse_events_module)

        for source_key, event in cls._iter_adverse_event_records(adverse_events_module):
            text = cls._event_text(event)
            if not text.strip():
                continue
            classification = classify_outcome_text(text)
            evidence_type = classification.get("evidence_type")
            if evidence_type not in {"qt_specific", "ecg_conduction", "ecg_broad", "cardiac_ae"}:
                continue

            group_ids: Set[str] = set()
            cls._collect_group_ids_recursive(event, group_ids)
            groups: List[Dict[str, str]] = []
            for gid in sorted(g for g in group_ids if g):
                g = group_lookup.get(gid, {"id": gid, "title": "", "description": ""})
                groups.append({"id": gid, "title": g.get("title", ""), "description": g.get("description", "")})

            title = str(
                event.get("term")
                or event.get("title")
                or event.get("name")
                or event.get("parent_term")
                or text[:120]
            ).strip()
            item = {
                "measure": title,
                "title": title,
                "description": str(event.get("description") or event.get("notes") or "").strip(),
                "source": f"adverseEventsModule.{source_key}",
                "groups": groups,
                "classification": classification,
                "raw_event": event,
            }
            out["classified_outcomes"].append(item)
            if evidence_type == "qt_specific":
                out["qt_result_measures"].append(item)
            elif evidence_type == "ecg_conduction":
                out["ecg_conduction_result_measures"].append(item)
            elif evidence_type == "ecg_broad":
                out["ecg_broad_result_measures"].append(item)
            elif evidence_type == "cardiac_ae":
                out["cardiac_ae_result_measures"].append(item)
        return out

    def _build_results_recall_trial(
        self,
        basics: Dict[str, Any],
        raw_study: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        resultsSection recall branch：protocol 无 QT 信号，但 posted results 含 QT measure。
        """
        results_section = self.parse_results_section(raw_study)
        if not is_recall_branch_results_signal(results_section):
            return None

        return {
            **basics,
            "qt_related": True,
            "qt_related_title": False,
            "qt_related_outcome": False,
            "qt_outcome_measure": "",
            "search_branch": "results_recall",
            "protocol_qt_hit": False,
            "results_qt_hit": True,
            "_prefetched_raw": raw_study,
            "_clinical_results_section": results_section,
        }

    @classmethod
    def _controlled_broad_recall_terms(cls, drug_name_set: Dict[str, Any]) -> List[str]:
        """Small controlled term set for drug+QT full-text recall."""
        raw_terms = (
            list(drug_name_set.get("curated_match_terms") or [])
            + list(drug_name_set.get("strong_match_terms") or [])
            + list(drug_name_set.get("_strong_match_terms") or [])
        )
        out: List[str] = []
        seen: Set[str] = set()
        for term in raw_terms:
            t = str(term or "").strip()
            if len(t) < 3:
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
            if len(out) >= 8:
                break
        return out

    def _search_broad_drug_qt_recall(
        self,
        drug_name_set: Dict[str, Any],
        *,
        max_results: int,
        seen_nct: Set[str],
    ) -> List[Dict[str, Any]]:
        """
        Controlled full-text drug+QT recall.

        This is intentionally conservative:
        - limited to curated/strong drug aliases;
        - query combines a drug alias and a QT-specific token;
        - candidate must have qt_specific protocol/results evidence after full-study fetch;
        - candidate is labelled broad_drug_qt_recall and cannot become Primary unless
          group-level QT attribution later supports the target drug.
        """
        out: List[Dict[str, Any]] = []
        pending: Dict[str, Dict[str, Any]] = {}

        for drug_term in self._controlled_broad_recall_terms(drug_name_set):
            for qt_term in self._CONTROLLED_QT_RECALL_TERMS:
                query = f'"{drug_term}" "{qt_term}"'
                try:
                    studies = self.search_studies(query, max_results=max_results, query_field="query.term")
                except Exception:
                    continue
                for study in studies:
                    basics = self._parse_protocol_study_basics(study)
                    if not basics:
                        continue
                    nct_id = basics["nct_id"]
                    if nct_id in seen_nct or nct_id in pending:
                        continue
                    basics["_broad_query_term"] = drug_term
                    basics["_broad_qt_term"] = qt_term
                    pending[nct_id] = {"basics": basics, "protocol": study}

        for nct_id, item in pending.items():
            raw = self.get_study_by_nct_id(nct_id) or {"protocolSection": item["protocol"]}
            results_section = self.parse_results_section(raw)
            protocol = raw.get("protocolSection") or item["protocol"] or {}
            protocol_outcomes = classify_protocol_outcomes_module(protocol.get("outcomesModule") or {})
            title_cls = classify_outcome_text(
                f"{item['basics'].get('title', '')} {item['basics'].get('summary', '')}".strip()
            )
            has_qt_protocol = bool(protocol_outcomes.get("has_qt_specific")) or title_cls.get("evidence_type") == "qt_specific"
            has_qt_results = bool(results_section.get("has_qt_results"))
            if not (has_qt_protocol or has_qt_results):
                continue

            basics = item["basics"]
            trial = {
                **basics,
                "qt_related": True,
                "qt_related_title": title_cls.get("evidence_type") == "qt_specific",
                "qt_related_outcome": protocol_outcomes.get("has_qt_specific", False),
                "qt_outcome_measure": pick_protocol_qt_outcome_measure(protocol_outcomes),
                "search_branch": "broad_drug_qt_recall",
                "protocol_qt_hit": has_qt_protocol,
                "results_qt_hit": has_qt_results,
                "protocol_outcomes": protocol_outcomes,
                "title_outcome_classification": title_cls,
                "_prefetched_raw": raw,
                "_clinical_results_section": results_section,
            }
            out.append(trial)
        return out

    def search_qt_related_trials(
        self,
        drug_name: str,
        max_results: int = 30,
        drug_name_set: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        双分支搜索（strict 不覆盖 recall，同一 NCT 只保留一条）：

        strict branch  — query.intr + strong intervention/arm 匹配 + protocol QT 命中
        recall branch  — query.intr + strong intervention/arm 匹配 + 无 protocol QT，
                         按 NCT ID 拉完整 study JSON，解析
                         resultsSection.outcomeMeasuresModule.outcomeMeasures；
                         若含 QT/QTc/QTcF/QTcB/corrected QT/TQT 等 result measure 则保留，
                         标记 protocol_qt_hit=false、results_qt_hit=true、search_branch=results_recall
        """
        drug_name_set = drug_name_set or build_drug_name_set(drug_name)
        search_terms = list(drug_name_set.get("strong_match_terms") or [])
        if not search_terms:
            search_terms = list(drug_name_set.get("_strong_match_terms") or [])

        seen_nct: Set[str] = set()
        recall_pending: Set[str] = set()
        trials: List[Dict[str, Any]] = []
        recall_candidates: List[Dict[str, Any]] = []

        for term in search_terms:
            if not (term or "").strip():
                continue
            studies = self.search_studies_by_intervention(term, max_results=max_results)
            for study in studies:
                strong_match = self._strong_intervention_match(drug_name_set, study)

                # Strict protocol branch still requires strong intervention/arm alignment.
                strict = self._parse_protocol_study(study) if strong_match else None
                if strict:
                    nct_id = strict["nct_id"]
                    if nct_id not in seen_nct:
                        seen_nct.add(nct_id)
                        trials.append(strict)
                    continue

                # Results recall branch: do not drop query.intr hits before fetching
                # full results. Some QTcB/QTcF evidence is only present in
                # resultsSection.adverseEventsModule and protocol outcomes may be silent.
                basics = self._parse_protocol_study_basics(study)
                if not basics:
                    continue
                nct_id = basics["nct_id"]
                if nct_id in seen_nct or nct_id in recall_pending:
                    continue
                recall_pending.add(nct_id)
                recall_candidates.append(basics)

        for basics in recall_candidates:
            nct_id = basics["nct_id"]
            if nct_id in seen_nct:
                continue

            raw = self.get_study_by_nct_id(nct_id) or {}
            recall_trial = self._build_results_recall_trial(basics, raw)
            if not recall_trial:
                continue

            seen_nct.add(nct_id)
            trials.append(recall_trial)

        # Controlled full-text drug+QT fallback. This catches trials where the
        # drug is not indexed under intervention but QT-specific evidence is
        # present in posted results. It remains downstream-attributed and is not
        # sufficient for Primary by itself.
        broad_trials = self._search_broad_drug_qt_recall(
            drug_name_set, max_results=max_results, seen_nct=seen_nct
        )
        for trial in broad_trials:
            nct_id = trial.get("nct_id", "")
            if nct_id and nct_id not in seen_nct:
                seen_nct.add(nct_id)
                trials.append(trial)

        return trials

    def enrich_trial_evidence(
        self,
        trial: Dict[str, Any],
        raw_study: Optional[Dict[str, Any]],
        drug_name_set: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        在 raw study 基础上补充 drug_trial_alignment / qt_result_attribution / evidence_tier。
        """
        enriched = dict(trial)
        raw_study = raw_study or trial.get("_prefetched_raw") or {}
        prefetched_results = trial.get("_clinical_results_section")
        results_section = (
            prefetched_results
            if isinstance(prefetched_results, dict)
            else self.parse_results_section(raw_study)
        )

        protocol = (raw_study.get("protocolSection") or {}) if isinstance(raw_study, dict) else {}
        protocol_outcomes = trial.get("protocol_outcomes") or classify_protocol_outcomes_module(
            protocol.get("outcomesModule") or {}
        )
        title_cls = trial.get("title_outcome_classification") or classify_outcome_text(
            f"{trial.get('title', '')} {trial.get('summary', '')}".strip()
        )

        protocol_qt_hit = bool(
            trial.get("protocol_qt_hit")
            if trial.get("protocol_qt_hit") is not None
            else is_strict_branch_protocol_signal(protocol_outcomes)
            or title_cls.get("evidence_type") in {"qt_specific", "ecg_conduction"}
        )
        results_qt_hit = bool(
            trial.get("results_qt_hit")
            if trial.get("results_qt_hit") is not None
            else results_section.get("has_qt_results")
        )
        search_branch = trial.get("search_branch") or (
            "strict_protocol_qt" if protocol_qt_hit else "results_recall"
        )

        alignment = assess_drug_trial_alignment(
            drug_name_set,
            raw_study,
            has_qt_specific_protocol=protocol_outcomes.get("has_qt_specific", False),
            has_ecg_conduction_protocol=protocol_outcomes.get("has_ecg_conduction", False),
            has_ecg_broad_protocol=protocol_outcomes.get("has_ecg_broad", False),
            has_qt_specific_results=results_qt_hit,
            has_ecg_conduction_results=bool(results_section.get("has_ecg_conduction_results")),
            has_ecg_broad_results=bool(results_section.get("has_ecg_broad_results")),
        )
        qt_attr = assess_qt_result_attribution(
            drug_name_set,
            results_section.get("qt_result_measures") or [],
            raw_study,
        )
        qt_outcome_measure = trial.get("qt_outcome_measure") or pick_protocol_qt_outcome_measure(
            protocol_outcomes
        )
        tier_meta: Dict[str, str] = {}
        evidence_tier = classify_evidence_tier(
            alignment,
            qt_result_attribution=qt_attr,
            protocol_qt_hit=protocol_qt_hit,
            results_qt_hit=results_qt_hit,
            search_branch=search_branch,
            protocol_outcomes=protocol_outcomes,
            title_classification=title_cls,
            results_section=results_section,
            qt_outcome_measure=qt_outcome_measure,
            tier_reason=tier_meta,
        )

        enriched["drug_name_set"] = {
            k: drug_name_set.get(k)
            for k in (
                "molecule_chembl_id", "pref_name", "chembl_pref_name",
                "molecule_synonyms", "molecule_hierarchy",
                "parent_molecule_chembl_id", "parent_pref_name",
                "strong_match_terms", "related_match_terms", "weak_terms",
                "curated_match_terms", "curated_alias_families",
                "recall_audit_terms", "exclude_names", "enrichment_status",
            )
            if k in drug_name_set
        }
        enriched["drug_trial_alignment"] = alignment
        enriched["clinical_results_section"] = results_section
        enriched["protocol_outcomes"] = protocol_outcomes
        enriched["title_outcome_classification"] = title_cls
        enriched["qt_result_attribution"] = qt_attr
        enriched["search_branch"] = search_branch
        enriched["protocol_qt_hit"] = protocol_qt_hit
        enriched["results_qt_hit"] = results_qt_hit
        enriched["qt_related_title"] = title_cls.get("evidence_type") == "qt_specific"
        enriched["qt_related_outcome"] = protocol_outcomes.get("has_qt_specific", False)
        enriched["qt_outcome_measure"] = qt_outcome_measure
        enriched["evidence_tier"] = evidence_tier
        enriched["evidence_tier_reason"] = tier_meta.get("reason", "")
        enriched.pop("_prefetched_raw", None)
        enriched.pop("_clinical_results_section", None)
        return enriched

    @classmethod
    def _text_matches_strict_qt(cls, text_lower: str) -> bool:
        """
        文本是否含 QT/复极/hERG 直接表述。
        先剔除 PCR 相关词再匹配，防止 qt-pcr/qpcr/rt-pcr 误判。
        调用方须传入已 lower 的文本。
        """
        cleaned = cls._PCR_EXCLUSION_RE.sub(" ", text_lower)
        return any(p in cleaned for p in cls.QT_STRICT_MATCH_PHRASES)

    # ── resultsSection parser ────────────────────────────────────────────────

    #: 无结果时的默认结构，调用方可安全 .get()
    _EMPTY_RESULTS_SECTION: Dict[str, Any] = {
        "has_results_section": False,
        "has_posted_results": False,
        "has_results_outcome_measures": False,
        "has_qt_results": False,
        "qt_result_measures": [],
        "has_ecg_conduction_results": False,
        "ecg_conduction_result_measures": [],
        "has_ecg_broad_results": False,
        "ecg_broad_result_measures": [],
        "has_cardiac_ae_results": False,
        "cardiac_ae_result_measures": [],
        "classified_outcomes": [],
        "has_adverse_events": False,
        "has_more_info": False,
        "result_summary": "No posted resultsSection available from ClinicalTrials.gov.",
    }

    @classmethod
    def parse_results_section(cls, raw_study: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        从 get_study_by_nct_id 返回的原始 study dict 中解析 resultsSection。

        区别于 protocolSection.outcomesModule（计划测什么），此处解析的是
        resultsSection.outcomeMeasuresModule（实际报告了什么结果）。

        Args:
            raw_study: ClinicalTrials API /studies/{nct_id} 的原始返回，可为 None。

        Returns:
            clinical_results_section dict，字段见 _EMPTY_RESULTS_SECTION。
        """
        result: Dict[str, Any] = copy.deepcopy(cls._EMPTY_RESULTS_SECTION)

        if not raw_study or not isinstance(raw_study, dict):
            return result

        has_posted_results: bool = bool(raw_study.get("hasResults"))
        results_section: Dict = raw_study.get("resultsSection") or {}
        has_results_section: bool = bool(results_section)

        result["has_posted_results"] = has_posted_results
        result["has_results_section"] = has_results_section

        if not has_results_section:
            if has_posted_results:
                result["result_summary"] = (
                    "Study indicates posted results, but no parsed resultsSection "
                    "was available in the API response."
                )
            return result

        # ── outcome measures ─────────────────────────────────────────────────
        om_module: Dict = results_section.get("outcomeMeasuresModule") or {}
        outcome_measures: List = om_module.get("outcomeMeasures") or []
        classified_stats = classify_results_outcome_measures(outcome_measures)
        result.update(classified_stats)

        # ── adverse events & more info ────────────────────────────────────────
        adverse_events_module = results_section.get("adverseEventsModule") or {}
        result["has_adverse_events"] = bool(adverse_events_module)
        result["has_more_info"] = bool(results_section.get("moreInfoModule"))

        # Generic AE/event-table QT parser. This captures QTcB/QTcF threshold
        # results that are not represented as outcomeMeasuresModule outcomes.
        ae_stats = cls._classify_adverse_event_results(adverse_events_module)
        if ae_stats.get("qt_result_measures"):
            result.setdefault("qt_result_measures", [])
            result["qt_result_measures"].extend(ae_stats["qt_result_measures"])
            result["has_qt_results"] = True
        if ae_stats.get("ecg_conduction_result_measures"):
            result.setdefault("ecg_conduction_result_measures", [])
            result["ecg_conduction_result_measures"].extend(ae_stats["ecg_conduction_result_measures"])
            result["has_ecg_conduction_results"] = True
        if ae_stats.get("ecg_broad_result_measures"):
            result.setdefault("ecg_broad_result_measures", [])
            result["ecg_broad_result_measures"].extend(ae_stats["ecg_broad_result_measures"])
            result["has_ecg_broad_results"] = True
        if ae_stats.get("cardiac_ae_result_measures"):
            result.setdefault("cardiac_ae_result_measures", [])
            result["cardiac_ae_result_measures"].extend(ae_stats["cardiac_ae_result_measures"])
            result["has_cardiac_ae_results"] = True
        if ae_stats.get("classified_outcomes"):
            result.setdefault("classified_outcomes", [])
            result["classified_outcomes"].extend(ae_stats["classified_outcomes"])

        result["total_qt_result_measure_objects"] = len(result.get("qt_result_measures") or [])

        if result.get("has_qt_results"):
            result["result_summary"] = (
                "Posted resultsSection contains qt_specific outcome measures."
            )
        elif result.get("has_ecg_conduction_results"):
            result["result_summary"] = (
                "Posted resultsSection contains ecg_conduction outcome measures."
            )
        elif result.get("has_ecg_broad_results"):
            result["result_summary"] = (
                "Posted resultsSection contains ecg_broad outcome measures."
            )
        elif result.get("has_cardiac_ae_results"):
            result["result_summary"] = (
                "Posted resultsSection contains cardiac_ae outcome measures."
            )
        else:
            result["result_summary"] = (
                "Posted resultsSection available; no QT/ECG electrophysiology outcomes."
            )

        return result


def create_clinicaltrials_client(cache_dir: Optional[Path] = None) -> ClinicalTrialsClient:
    """创建ClinicalTrials客户端的工厂函数"""
    return ClinicalTrialsClient(cache_dir=cache_dir)