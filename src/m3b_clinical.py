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

    CARDIOTOX_KEYWORDS = [
        "cardiac",
        "heart",
        "cardiotoxicity",
        "cardiovascular",
        "arrhythmia",
        "torsades",
        "qt prolongation",
        "sudden death",
    ]

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
        *,
        refresh_chembl_clinical: bool = False,
        clinicaltrials_query_name: Optional[str] = None,
    ) -> ClinicalEvidence:
        """
        检索临床状态证据。

        :param drug_name: 若未传 ``clinicaltrials_query_name``：与 ChEMBL ``pref_name`` 规范化一致才检索
            ClinicalTrials；未传入则用 ``pref_name`` 检索（若有）。
        :param clinicaltrials_query_name: 若设置（如路径 B 父药名），**直接**作为 ClinicalTrials 检索词，
            不与子分子 ``pref_name`` 做一致性校验（相似扩展化合物通常不是同一药品名）。
        :param refresh_chembl_clinical: 为 True 时删除 ``chembl_clinical_{id}`` 缓存后再拉 ChEMBL drug。
        """
        logger.info("检索临床状态数据: %s", chembl_id)

        raw_mol = self.chembl_client.get_molecule_by_chembl_id(chembl_id)
        molecule = raw_mol[0] if isinstance(raw_mol, tuple) else raw_mol
        if not isinstance(molecule, dict):
            molecule = {}

        pref_name, inchi_key = _molecule_pref_and_inchi(molecule)
        _line = (
            f"M3B ChEMBL molecule | chembl_id={chembl_id} | "
            f"pref_name={pref_name!r} | inchi_key={inchi_key!r}"
        )
        logger.info(_line)
        print(_line, file=sys.stderr, flush=True)

        clinical_trials: List[ClinicalTrialInfo] = []
        ct_query_name: Optional[str] = None
        user_dn = (drug_name or "").strip() or None
        pref_empty = not (pref_name or "").strip()

        ct_override = (clinicaltrials_query_name or "").strip()
        if ct_override:
            ct_query_name = ct_override
            _ov = (
                f"M3B ClinicalTrials：使用 clinicaltrials_query_name={ct_query_name!r} "
                f"（路径 B 以父药锚定，与子分子 pref_name 无关）"
            )
            logger.info(_ov)
            print(_ov, file=sys.stderr, flush=True)
        elif user_dn is not None:
            # 路径 B 相似分子等：ChEMBL 常无 pref_name，此时用调用方名称（多为父药 DIQTA drug_name）查试验
            if pref_empty:
                ct_query_name = user_dn
                _fb = (
                    f"M3B ClinicalTrials：子分子无 pref_name，使用传入 drug_name={user_dn!r}"
                )
                logger.info(_fb)
                print(_fb, file=sys.stderr, flush=True)
            elif _normalize_label(user_dn) != _normalize_label(pref_name):
                _skip = (
                    f"M3B 跳过 ClinicalTrials：传入 drug_name 与 ChEMBL pref_name 不一致 | "
                    f"drug_name={user_dn!r} pref_name={pref_name!r}"
                )
                logger.warning(_skip)
                print(_skip, file=sys.stderr, flush=True)
            else:
                ct_query_name = user_dn
        elif pref_name:
            ct_query_name = pref_name
        else:
            _skip = "M3B 跳过 ClinicalTrials：无 pref_name 且未传入 drug_name"
            logger.info(_skip)
            print(_skip, file=sys.stderr, flush=True)

        if refresh_chembl_clinical:
            ck = f"chembl_clinical_{chembl_id}"
            self.cache.delete(ck)
            _msg = f"M3B 已清除缓存键 {ck!r}"
            logger.info(_msg)
            print(_msg, file=sys.stderr, flush=True)

        chembl_clinical = self._retrieve_from_chembl(chembl_id)

        if ct_query_name:
            trials = self._retrieve_from_clinicaltrials(ct_query_name)
            clinical_trials.extend(trials)

        external_db_flags = self._get_external_db_flags(chembl_id)
        fda_warnings = self._get_fda_warnings(chembl_id)

        # 以规范化后的 max_phase 为准推导 approved，避免 ChEMBL 字符串 phase、旧缓存中 approved 与 phase 不一致
        max_phase_i = _coerce_max_phase(chembl_clinical.get("max_phase"))
        approved = _approved_from_max_phase(max_phase_i)
        if max_phase_i == 4 and chembl_clinical.get("approved") is False:
            logger.info(
                "M3B: 已按 max_phase==4 纠正 approved（原缓存或 drug 记录中 approved 与 phase 不一致，"
                "多因 max_phase 曾为字符串或未统一推导）"
            )

        return ClinicalEvidence(
            max_phase=max_phase_i,
            approved=approved,
            withdrawn=chembl_clinical.get("withdrawn", WithdrawalInfo(flag=False)),
            black_box_warning=chembl_clinical.get("black_box_warning"),
            clinical_trials=clinical_trials,
            external_db_flags=external_db_flags,
            fda_label_warnings=fda_warnings,
            drug_info_status=chembl_clinical.get("drug_info_status"),
        )

    def _retrieve_from_chembl(self, chembl_id: str) -> Dict[str, Any]:
        cache_key = f"chembl_clinical_{chembl_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            raw = self.chembl_client.get_drug_info(chembl_id)
            drug_info = raw[0] if isinstance(raw, tuple) else raw
            if not isinstance(drug_info, dict):
                drug_info = None

            if not drug_info:
                # 不写入缓存，避免「无 drug」被永久短路；状态由 drug_info_status 表达
                return {"drug_info_status": "not_found"}

            mp = _coerce_max_phase(drug_info.get("max_phase"))
            result = {
                "max_phase": mp,
                "approved": _approved_from_max_phase(mp),
                "withdrawn": WithdrawalInfo(
                    flag=drug_info.get("withdrawn_flag", False),
                    year=drug_info.get("withdrawn_year"),
                    country=drug_info.get("withdrawn_country"),
                    reason=drug_info.get("withdrawn_reason"),
                ),
                "black_box_warning": drug_info.get("black_box_warning", False),
                "drug_info_status": "ok",
            }

            self.cache.set(cache_key, result)
            return result

        except Exception as e:
            logger.warning("从ChEMBL获取临床状态失败: %s, 错误: %s", chembl_id, e)
            return {"drug_info_status": "request_failed"}

    def _retrieve_from_clinicaltrials(self, drug_name: str) -> List[ClinicalTrialInfo]:
        try:
            qt_trials = self.clinicaltrials_client.search_qt_related_trials(drug_name)

            return [
                ClinicalTrialInfo(
                    nct_id=trial["nct_id"],
                    title=trial["title"],
                    status=trial["status"],
                    qt_related=trial["qt_related"],
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

    def check_withdrawn_for_qt(self, chembl_id: str) -> bool:
        chembl_data = self._retrieve_from_chembl(chembl_id)
        withdrawn = chembl_data.get("withdrawn")

        if not withdrawn or not getattr(withdrawn, "flag", False):
            return False

        reason = (getattr(withdrawn, "reason", None) or "").lower()
        qt_keywords = ["qt", "torsades", "torsade", "arrhythmia", "cardiac"]

        return any(kw in reason for kw in qt_keywords)


def create_clinical_status_retriever() -> ClinicalStatusRetriever:
    return ClinicalStatusRetriever()
