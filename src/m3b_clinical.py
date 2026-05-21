"""M3-B: 临床状态检索器模块"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional

from schemas import (
    ClinicalEvidence,
    ClinicalTrialInfo,
    ExternalDBFlags,
    WithdrawalInfo,
)
from utils.cache import get_global_cache
from utils.chembl_client import ChEMBLClient
from utils.clinicaltrials_client import ClinicalTrialsClient
from rules.cardiotox_evidence_rules import clinical_text_has_direct_qt

logger = logging.getLogger(__name__)


def _normalize_label(s: Optional[str]) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().lower().split())


def _molecule_pref_and_inchi(molecule: Dict[str, Any]) -> tuple[str, str]:
    pref = (molecule.get("pref_name") or "").strip()
    ms = molecule.get("molecule_structures")
    ikey = ""
    if isinstance(ms, dict):
        ikey = (ms.get("standard_inchi_key") or "").strip()
    return pref, ikey


def _coerce_max_phase(raw: Any) -> Optional[int]:
    """ChEMBL drug.json 常把 max_phase 打成字符串（如 \"4\"），与 int 比较会导致 approved 误判。"""
    if raw is None:
        return None
    try:
        v = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return None
    if 0 <= v <= 4:
        return v
    return None


def _approved_from_max_phase(mp: Optional[int]) -> Optional[bool]:
    if mp is None:
        return None
    return mp == 4


class ClinicalStatusRetriever:
    """临床状态检索器"""

    def __init__(
        self,
        chembl_client: Optional[ChEMBLClient] = None,
        clinicaltrials_client: Optional[ClinicalTrialsClient] = None,
    ):
        self.chembl_client = chembl_client or ChEMBLClient()
        self.clinicaltrials_client = clinicaltrials_client or ClinicalTrialsClient()
        self.cache = get_global_cache()

    def retrieve(
        self,
        chembl_id: str,
        drug_name: Optional[str] = None,
        molecule: Optional[Dict[str, Any]] = None,
        *,
        refresh_chembl_clinical: bool = False,
    ) -> ClinicalEvidence:
        """
        检索临床状态证据。每个分子独立检索，不融合其他分子信息。

        :param chembl_id: 分子 ChEMBL ID
        :param drug_name: 药品名称（用于 ClinicalTrials 检索）
        :param molecule: 若已获取分子信息（包含 pref_name），传入可避免重复查询
        :param refresh_chembl_clinical: 为 True 时删除缓存后再拉 ChEMBL drug
        """
        logger.info("检索临床状态数据: %s", chembl_id)

        # 优先使用传入的 molecule，若无则查询
        if molecule is None:
            raw_mol = self.chembl_client.get_molecule_by_chembl_id(chembl_id)
            molecule = raw_mol[0] if isinstance(raw_mol, tuple) else raw_mol
        if not isinstance(molecule, dict):
            molecule = {}

        pref_name, inchi_key = _molecule_pref_and_inchi(molecule)
        logger.info(f"M3B ChEMBL molecule | chembl_id={chembl_id} | pref_name={pref_name!r}")

        # ClinicalTrials 查询名：优先 molecule pref_name，其次 drug_name
        ct_query_name: Optional[str] = None
        user_dn = (drug_name or "").strip() or None
        pref_empty = not (pref_name or "").strip()

        if pref_name:
            ct_query_name = pref_name
        elif user_dn:
            ct_query_name = user_dn
        else:
            logger.info("M3B 跳过 ClinicalTrials：无 pref_name 且未传入 drug_name")

        if refresh_chembl_clinical:
            ck = f"chembl_clinical_{chembl_id}"
            self.cache.delete(ck)

        chembl_clinical = self._retrieve_from_chembl(chembl_id, molecule=molecule)

        clinical_trials: List[ClinicalTrialInfo] = []
        if ct_query_name:
            trials = self._retrieve_from_clinicaltrials(ct_query_name)
            clinical_trials.extend(trials)

        external_db_flags = self._get_external_db_flags(chembl_id)
        fda_warnings = self._get_fda_warnings(chembl_id)

        max_phase_i = _coerce_max_phase(chembl_clinical.get("max_phase"))
        approved = _approved_from_max_phase(max_phase_i)
        if max_phase_i == 4 and chembl_clinical.get("approved") is False:
            logger.info("M3B: 已按 max_phase==4 纠正 approved")

        withdrawn_info = chembl_clinical.get("withdrawn", WithdrawalInfo(flag=False))
        direct_qt_hit = self._compute_direct_qt_clinical_hit(
            withdrawn_info,
            clinical_trials,
            fda_warnings,
        )

        return ClinicalEvidence(
            max_phase=max_phase_i,
            approved=approved,
            withdrawn=withdrawn_info,
            black_box_warning=chembl_clinical.get("black_box_warning"),
            clinical_trials=clinical_trials,
            external_db_flags=external_db_flags,
            fda_label_warnings=fda_warnings,
            drug_info_status=chembl_clinical.get("drug_info_status"),
            direct_qt_clinical_hit=direct_qt_hit,
        )

    def _retrieve_from_chembl(
        self, chembl_id: str, molecule: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        从 ChEMBL 获取临床状态信息。

        :param chembl_id: 分子 ChEMBL ID
        :param molecule: 已获取的 molecule 数据（包含 max_phase, withdrawn_flag, black_box_warning 等）
        """
        cache_key = f"chembl_clinical_{chembl_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # 优先使用传入的 molecule
        drug_info = None
        if molecule is not None and isinstance(molecule, dict):
            drug_info = molecule
        else:
            # 兜底：若未传入则查询（保留向后兼容）
            try:
                raw = self.chembl_client.get_drug_info(chembl_id)
                drug_info = raw[0] if isinstance(raw, tuple) else raw
            except Exception as e:
                logger.warning("从ChEMBL获取drug信息失败: %s, 错误: %s", chembl_id, e)

        if not drug_info or not isinstance(drug_info, dict):
            return {"drug_info_status": "not_found"}

        # 从 molecule/drug 中提取临床相关字段
        mp = _coerce_max_phase(drug_info.get("max_phase"))
        withdrawn_flag = drug_info.get("withdrawn_flag", False)
        # drug 端点有 withdrawn_year/country/reason，molecule 端点只有 withdrawn_flag
        if "withdrawn_year" in drug_info:
            withdrawn_year = drug_info.get("withdrawn_year")
            withdrawn_country = drug_info.get("withdrawn_country")
            withdrawn_reason = drug_info.get("withdrawn_reason")
        else:
            withdrawn_year = None
            withdrawn_country = None
            withdrawn_reason = None

        result = {
            "max_phase": mp,
            "approved": _approved_from_max_phase(mp),
            "withdrawn": WithdrawalInfo(
                flag=withdrawn_flag,
                year=withdrawn_year,
                country=withdrawn_country,
                reason=withdrawn_reason,
            ),
            "black_box_warning": drug_info.get("black_box_warning", False),
            "drug_info_status": "ok",
        }

        self.cache.set(cache_key, result)
        return result

    @staticmethod
    def _dedupe_trials(trials: List[ClinicalTrialInfo]) -> List[ClinicalTrialInfo]:
        seen: Dict[str, ClinicalTrialInfo] = {}
        for t in trials:
            if t.nct_id not in seen:
                seen[t.nct_id] = t
        return list(seen.values())

    @staticmethod
    def _compute_direct_qt_clinical_hit(
        withdrawn: Any,
        trials: List[ClinicalTrialInfo],
        fda_warnings: List[str],
    ) -> bool:
        texts: List[str] = []
        if withdrawn is not None and getattr(withdrawn, "reason", None):
            texts.append(str(withdrawn.reason))
        for tr in trials:
            texts.append(tr.title)
            if tr.summary:
                texts.append(tr.summary)
        texts.extend(fda_warnings)
        if not texts:
            return False
        return clinical_text_has_direct_qt(*texts)

    def _retrieve_from_clinicaltrials(self, drug_name: str) -> List[ClinicalTrialInfo]:
        try:
            qt_trials = self.clinicaltrials_client.search_qt_related_trials(drug_name)

            return [
                ClinicalTrialInfo(
                    nct_id=trial["nct_id"],
                    title=trial["title"],
                    status=trial["status"],
                    qt_related=trial["qt_related"],
                    qt_related_title=trial.get("qt_related_title"),
                    qt_related_outcome=trial.get("qt_related_outcome"),
                    qt_outcome_measure=trial.get("qt_outcome_measure"),
                    summary=trial.get("summary"),
                )
                for trial in qt_trials
            ]

        except Exception as e:
            logger.warning("从ClinicalTrials获取数据失败: %s, 错误: %s", drug_name, e)
            return []

    def _get_external_db_flags(self, chembl_id: str) -> Dict[str, ExternalDBFlags]:
        return {
            "cardiotox": ExternalDBFlags(present=False),
            "etox": ExternalDBFlags(present=False),
            "dilirank": ExternalDBFlags(present=False),
        }

    def _get_fda_warnings(self, chembl_id: str) -> List[str]:
        return []
