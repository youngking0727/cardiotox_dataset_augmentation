"""M3-B: 临床状态检索器模块"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..schemas import (
    ClinicalEvidence, WithdrawalInfo, ClinicalTrialInfo,
    ExternalDBFlags, TargetConfig
)
from ..utils.chembl_client import ChEMBLClient
from ..utils.clinicaltrials_client import ClinicalTrialsClient
from ..utils.cache import get_global_cache

logger = logging.getLogger(__name__)


class ClinicalStatusRetriever:
    """临床状态检索器"""

    # 心脏毒性相关关键词
    CARDIOTOX_KEYWORDS = [
        "cardiac", "heart", "cardiotoxicity", "cardiovascular",
        "arrhythmia", "torsades", "qt prolongation", "sudden death"
    ]

    def __init__(self, chembl_client: Optional[ChEMBLClient] = None,
                 clinicaltrials_client: Optional[ClinicalTrialsClient] = None):
        """
        初始化临床状态检索器

        Args:
            chembl_client: ChEMBL客户端
            clinicaltrials_client: ClinicalTrials.gov客户端
        """
        self.chembl_client = chembl_client or ChEMBLClient()
        self.clinicaltrials_client = clinicaltrials_client or ClinicalTrialsClient()
        self.cache = get_global_cache()

    def retrieve(self, chembl_id: str, drug_name: Optional[str] = None) -> ClinicalEvidence:
        """
        检索临床状态证据

        Args:
            chembl_id: 分子的ChEMBL ID
            drug_name: 药物名称（用于ClinicalTrials搜索）

        Returns:
            临床证据对象
        """
        logger.info(f"检索临床状态数据: {chembl_id}")

        # 1. 从ChEMBL获取临床信息
        chembl_clinical = self._retrieve_from_chembl(chembl_id)

        # 2. 从ClinicalTrials.gov获取临床试验信息
        clinical_trials = []
        if drug_name:
            trials = self._retrieve_from_clinicaltrials(drug_name)
            clinical_trials.extend(trials)

        # 3. 外部数据库标记（暂为占位，实际需要数据库查询）
        external_db_flags = self._get_external_db_flags(chembl_id)

        # 4. FDA标签警告（暂为占位，实际需要FDA API）
        fda_warnings = self._get_fda_warnings(chembl_id)

        return ClinicalEvidence(
            max_phase=chembl_clinical.get("max_phase"),
            approved=chembl_clinical.get("approved"),
            withdrawn=chembl_clinical.get("withdrawn", WithdrawalInfo(flag=False)),
            black_box_warning=chembl_clinical.get("black_box_warning"),
            clinical_trials=clinical_trials,
            external_db_flags=external_db_flags,
            fda_label_warnings=fda_warnings
        )

    def _retrieve_from_chembl(self, chembl_id: str) -> Dict[str, Any]:
        """
        从ChEMBL获取临床状态信息

        Args:
            chembl_id: ChEMBL ID

        Returns:
            临床状态信息字典
        """
        cache_key = f"chembl_clinical_{chembl_id}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            # 获取药物信息
            drug_info = self.chembl_client.get_drug_info(chembl_id)

            if not drug_info:
                return {}

            result = {
                "max_phase": drug_info.get("max_phase"),
                "approved": drug_info.get("max_phase", 0) == 4,
                "withdrawn": WithdrawalInfo(
                    flag=drug_info.get("withdrawn_flag", False),
                    year=drug_info.get("withdrawn_year"),
                    country=drug_info.get("withdrawn_country"),
                    reason=drug_info.get("withdrawn_reason")
                ),
                "black_box_warning": drug_info.get("black_box_warning", False)
            }

            self.cache.set(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"从ChEMBL获取临床状态失败: {chembl_id}, 错误: {e}")
            return {}

    def _retrieve_from_clinicaltrials(self, drug_name: str) -> List[ClinicalTrialInfo]:
        """
        从ClinicalTrials.gov获取相关临床试验

        Args:
            drug_name: 药物名称

        Returns:
            临床试验列表
        """
        try:
            qt_trials = self.clinicaltrials_client.search_qt_related_trials(drug_name)

            return [
                ClinicalTrialInfo(
                    nct_id=trial["nct_id"],
                    title=trial["title"],
                    status=trial["status"],
                    qt_related=trial["qt_related"],
                    summary=trial.get("summary")
                )
                for trial in qt_trials
            ]

        except Exception as e:
            logger.warning(f"从ClinicalTrials获取数据失败: {drug_name}, 错误: {e}")
            return []

    def _get_external_db_flags(self, chembl_id: str) -> Dict[str, ExternalDBFlags]:
        """
        获取外部数据库标记

        Args:
            chembl_id: ChEMBL ID

        Returns:
            外部数据库标记字典
        """
        # TODO: 实际实现需要接入:
        # - CardioTox数据库
        # - eTox数据库
        # - DILIrank数据库

        # 当前返回空占位符
        return {
            "cardiotox": ExternalDBFlags(present=False),
            "etox": ExternalDBFlags(present=False),
            "dilirank": ExternalDBFlags(present=False)
        }

    def _get_fda_warnings(self, chembl_id: str) -> List[str]:
        """
        获取FDA标签警告

        Args:
            chembl_id: ChEMBL ID

        Returns:
            警告列表
        """
        # TODO: 实际实现需要接入FDA Orange Book API
        return []

    def check_withdrawn_for_qt(self, chembl_id: str) -> bool:
        """
        检查是否因QT延长被撤市

        Args:
            chembl_id: ChEMBL ID

        Returns:
            是否因QT延长被撤市
        """
        chembl_data = self._retrieve_from_chembl(chembl_id)
        withdrawn = chembl_data.get("withdrawn")

        if not withdrawn or not withdrawn.get("flag"):
            return False

        reason = withdrawn.get("reason", "").lower()
        qt_keywords = ["qt", "torsades", "torsade", "arrhythmia", "cardiac"]

        return any(kw in reason for kw in qt_keywords)


def create_clinical_status_retriever() -> ClinicalStatusRetriever:
    """创建临床状态检索器的工厂函数"""
    return ClinicalStatusRetriever()