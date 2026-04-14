"""M3-A: 靶点活性检索器模块"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..schemas import TargetBioactivity, BioactivityMeasurement, TargetConfig
from ..utils.chembl_client import ChEMBLClient
from ..utils.cache import get_global_cache

logger = logging.getLogger(__name__)


# 默认靶点配置
DEFAULT_TARGETS = [
    TargetConfig(name="hERG", chembl_id="CHEMBL240", priority=1),
    TargetConfig(name="Nav1.5", chembl_id="CHEMBL1971", priority=2),
    TargetConfig(name="Cav1.2", chembl_id="CHEMBL1940", priority=3),
]


class BioactivityRetriever:
    """靶点活性检索器"""

    # 有效测量类型
    VALID_MEASUREMENT_TYPES = ["IC50", "Ki", "Kd", "EC50", "IC90", "IC95"]

    def __init__(self, chembl_client: Optional[ChEMBLClient] = None,
                 targets: Optional[List[TargetConfig]] = None):
        """
        初始化靶点活性检索器

        Args:
            chembl_client: ChEMBL客户端
            targets: 靶点配置列表
        """
        self.chembl_client = chembl_client or ChEMBLClient()
        self.targets = targets or DEFAULT_TARGETS
        self.cache = get_global_cache()

    def retrieve(self, chembl_id: str,
                target_chembl_ids: Optional[List[str]] = None) -> Dict[str, TargetBioactivity]:
        """
        检索分子对指定靶点的活性数据

        Args:
            chembl_id: 分子的ChEMBL ID
            target_chembl_ids: 靶点ChEMBL ID列表，None表示使用默认靶点

        Returns:
            靶点活性数据字典，key为目标名称
        """
        cache_key = f"bioactivity_{chembl_id}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        logger.info(f"检索靶点活性数据: {chembl_id}")

        if target_chembl_ids is None:
            target_chembl_ids = [t.chembl_id for t in self.targets]

        # 获取活性数据
        activities = self.chembl_client.get_activities(chembl_id, target_chembl_ids)

        # 按靶点分组
        result = {}
        target_map = {t.chembl_id: t.name for t in self.targets}

        for activity in activities:
            target_id = activity.get("target_chembl_id")
            if not target_id or target_id not in target_map:
                continue

            target_name = target_map[target_id]

            # 解析测量数据
            measurement = self._parse_activity(activity)
            if measurement is None:
                continue

            if target_name not in result:
                result[target_name] = TargetBioactivity(
                    target=target_name,
                    target_chembl_id=target_id,
                    measurements=[]
                )

            result[target_name].measurements.append(measurement)

        # 对每个靶点的测量按pChEMBL值排序（活性强的在前）
        for target_name in result:
            result[target_name].measurements.sort(
                key=lambda x: x.pchembl if x.pchembl is not None else 0,
                reverse=True
            )

        self.cache.set(cache_key, result)
        return result

    def _parse_activity(self, activity: Dict[str, Any]) -> Optional[BioactivityMeasurement]:
        """
        解析活性数据

        Args:
            activity: 原始活性数据

        Returns:
            解析后的测量数据
        """
        try:
            standard_type = activity.get("standard_type")
            if not standard_type or standard_type not in self.VALID_MEASUREMENT_TYPES:
                return None

            standard_value = activity.get("standard_value")
            if standard_value is None:
                return None

            units = activity.get("standard_units")
            if not units:
                units = "nM"  # 默认单位

            # 转换单位到统一的uM
            value_in_uM = self._convert_to_uM(standard_value, units)
            pchembl = activity.get("pchembl_value")

            measurement = BioactivityMeasurement(
                type=standard_type,
                value=value_in_uM,
                units="uM",
                pchembl=pchembl,
                assay_type=activity.get("assay_type"),
                assay_description=activity.get("assay_description"),
                document_chembl_id=activity.get("document_chembl_id"),
                confidence_score=activity.get("confidence_score")
            )

            return measurement

        except Exception as e:
            logger.warning(f"解析活性数据失败: {e}")
            return None

    def _convert_to_uM(self, value: float, units: str) -> float:
        """
        将活性值转换为uM单位

        Args:
            value: 活性值
            units: 单位

        Returns:
            转换为uM后的值
        """
        units_lower = units.lower() if units else ""

        # 转换为uM
        if "nm" in units_lower or "nanomolar" in units_lower:
            return value / 1000.0  # nM -> uM
        elif "um" in units_lower or "micromolar" in units_lower:
            return value  # 已经是uM
        elif "mm" in units_lower or "millimolar" in units_lower:
            return value * 1000.0  # mM -> uM
        elif "pm" in units_lower or "picomolar" in units_lower:
            return value / 1_000_000.0  # pM -> uM
        else:
            # 未知单位，假设为nM
            return value / 1000.0

    def get_herg_only(self, chembl_id: str) -> Optional[TargetBioactivity]:
        """
        只获取hERG活性数据

        Args:
            chembl_id: 分子的ChEMBL ID

        Returns:
            hERG靶点活性数据
        """
        all_activities = self.retrieve(chembl_id)
        return all_activities.get("hERG")

    def is_herg_inhibitor(self, chembl_id: str, threshold: float = 10.0) -> bool:
        """
        判断是否为hERG抑制剂

        Args:
            chembl_id: 分子的ChEMBL ID
            threshold: 抑制阈值(uM)，默认10uM

        Returns:
            是否为hERG抑制剂
        """
        herg_data = self.get_herg_only(chembl_id)
        if not herg_data or not herg_data.measurements:
            return False

        # 取最有效的测量值
        best_measurement = herg_data.measurements[0]
        ic50 = best_measurement.value

        return ic50 < threshold


def create_bioactivity_retriever(targets: Optional[List[TargetConfig]] = None) -> BioactivityRetriever:
    """创建靶点活性检索器的工厂函数"""
    return BioactivityRetriever(targets=targets)