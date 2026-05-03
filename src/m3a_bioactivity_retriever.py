"""M3-A: 靶点活性检索器实现（可观测性日志见 retrieve）。"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional

from schemas import BioactivityMeasurement, TargetBioactivity, TargetConfig
from utils.cache import get_global_cache
from utils.chembl_client import ChEMBLClient
from rules.cardiotox_evidence_rules import (
    get_evidence_rules,
    normalize_standard_type,
    classify_activity_evidence_dict,
)

logger = logging.getLogger(__name__)


# 默认靶点配置
DEFAULT_TARGETS = [
    TargetConfig(name="hERG", chembl_id="CHEMBL240", priority=1),
    TargetConfig(name="Nav1.5", chembl_id="CHEMBL1971", priority=2),
    TargetConfig(name="Cav1.2", chembl_id="CHEMBL1940", priority=3),
]


class BioactivityRetriever:
    """靶点活性检索器"""

    def __init__(
        self,
        chembl_client: Optional[ChEMBLClient] = None,
        targets: Optional[List[TargetConfig]] = None,
    ):
        self.chembl_client = chembl_client or ChEMBLClient()
        self.targets = targets or DEFAULT_TARGETS
        self.cache = get_global_cache()
        self._raw_activities_cache: Dict[str, List[Dict[str, Any]]] = {}

    def _canonical_measurement_type(self, raw: Any) -> Optional[str]:
        r = get_evidence_rules()
        n = normalize_standard_type(raw, r)
        if not n:
            return None
        # 统一大写短名便于排序键（Percent Inhibition 等保持首字母）
        return n

    def retrieve(
        self,
        chembl_id: str,
        target_chembl_ids: Optional[List[str]] = None,
    ) -> Dict[str, TargetBioactivity]:
        cache_key = f"bioactivity_{chembl_id}"
        cached = self.cache.get(cache_key)
        if cached:
            # 缓存命中时也要填充 _raw_activities_cache（供 M3C 使用）
            if chembl_id not in self._raw_activities_cache:
                raw = self.chembl_client.get_activities(chembl_id, target_chembl_ids or [t.chembl_id for t in self.targets])
                activities = raw[0] if isinstance(raw, tuple) else raw
                if isinstance(activities, list):
                    self._raw_activities_cache[chembl_id] = list(activities)
            logger.info(
                "M3A cache hit | chembl_id=%s | keys=%s",
                chembl_id,
                list(cached.keys()),
            )
            return cached

        logger.info("检索靶点活性数据: %s", chembl_id)

        if target_chembl_ids is None:
            target_chembl_ids = [t.chembl_id for t in self.targets]

        # TODO: 这里
        raw = self.chembl_client.get_activities(chembl_id, target_chembl_ids)
        raw_is_tuple = isinstance(raw, tuple)
        activities = raw[0] if isinstance(raw, tuple) else raw
        if not isinstance(activities, list):
            activities = []
        raw_count = len(activities)
        # 保存 raw activities 供 M3C 文献服务使用（含 document_chembl_id）
        self._raw_activities_cache[chembl_id] = list(activities)

        _hdr = (
            f"M3A raw fetch | chembl_id={chembl_id} | raw_count={raw_count} | "
            f"raw_return_is_tuple={raw_is_tuple}"
        )
        logger.info(_hdr)
        print(_hdr, file=sys.stderr, flush=True)
        for i, act in enumerate(activities[:5]):
            if not isinstance(act, dict):
                _ln = f"M3A raw[#{i}] non-dict type={type(act).__name__} repr={act!r}"
                logger.info(_ln)
                print(_ln, file=sys.stderr, flush=True)
                continue
            _ln = (
                f"M3A raw[#{i}] target_chembl_id={act.get('target_chembl_id')!r} | "
                f"standard_type={act.get('standard_type')!r} | "
                f"standard_value={act.get('standard_value')!r} | "
                f"standard_units={act.get('standard_units')!r} | "
                f"assay_description={act.get('assay_description')!r}"
            )
            logger.info(_ln)
            print(_ln, file=sys.stderr, flush=True)

        result: Dict[str, TargetBioactivity] = {}
        target_map = {t.chembl_id: t.name for t in self.targets}

        skip_target_not_in_map = 0
        skip_standard_type_not_supported = 0
        skip_missing_standard_value = 0
        skip_standard_value_parse_error = 0
        supplemental_kept = 0

        for activity in activities:
            if not isinstance(activity, dict):
                continue
            target_id = activity.get("target_chembl_id")
            if not target_id or target_id not in target_map:
                skip_target_not_in_map += 1
                continue

            target_name = target_map[target_id]
            cls_info = classify_activity_evidence_dict(activity, target_name, target_id)

            measurement, skip_reason = self._parse_activity(activity, cls_info)
            if measurement is None:
                if skip_reason == "type":
                    skip_standard_type_not_supported += 1
                elif skip_reason == "missing_value":
                    skip_missing_standard_value += 1
                elif skip_reason == "parse_error":
                    skip_standard_value_parse_error += 1
                # 扩展：无数值但仍可能是 priority 证据 → supplemental
                if cls_info.get("priority_guess") or cls_info.get("direct_qt_context_hit"):
                    row = {
                        "standard_type": activity.get("standard_type"),
                        "assay_description": activity.get("assay_description"),
                        "classification": cls_info,
                    }
                    self._ensure_target(result, target_name, target_id)
                    result[target_name].supplemental_retention_rows.append(row)
                    supplemental_kept += 1
                continue

            if target_name not in result:
                result[target_name] = TargetBioactivity(
                    target=target_name,
                    target_chembl_id=target_id,
                    measurements=[],
                    supplemental_retention_rows=[],
                )

            result[target_name].measurements.append(measurement)

        for target_name in result:
            result[target_name].measurements.sort(
                key=lambda x: x.pchembl if x.pchembl is not None else 0,
                reverse=True,
            )

        if supplemental_kept:
            logger.info(
                "M3A supplemental_retention_rows 条数=%s（无标准数值但保留文本/优先级线索）",
                supplemental_kept,
            )

        if not result:
            _empty = (
                f"M3A empty result for {chembl_id} | raw_count={raw_count} | "
                f"skip_target_not_in_map={skip_target_not_in_map} | "
                f"skip_standard_type_not_supported={skip_standard_type_not_supported} | "
                f"skip_missing_standard_value={skip_missing_standard_value} | "
                f"skip_standard_value_parse_error={skip_standard_value_parse_error}"
            )
            logger.info(_empty)
            print(_empty, file=sys.stderr, flush=True)

        if result:
            self.cache.set(cache_key, result)
        return result

    def get_cached_raw_activities(self, chembl_id: str) -> List[Dict[str, Any]]:
        """获取已缓存的 raw activities（含 document_chembl_id），供 M3C 文献服务使用"""
        return self._raw_activities_cache.get(chembl_id, [])

    @staticmethod
    def _ensure_target(
        result: Dict[str, TargetBioactivity], target_name: str, target_id: str
    ) -> None:
        if target_name not in result:
            result[target_name] = TargetBioactivity(
                target=target_name,
                target_chembl_id=target_id,
                measurements=[],
                supplemental_retention_rows=[],
            )

    def _parse_activity(
        self,
        activity: Dict[str, Any],
        cls_info: Dict[str, Any],
    ) -> tuple[Optional[BioactivityMeasurement], Optional[str]]:
        """
        Returns:
            (measurement, skip_reason)
        """
        try:
            standard_type_raw = activity.get("standard_type")
            canonical = self._canonical_measurement_type(standard_type_raw)
            if not canonical:
                return None, "type"

            standard_value = activity.get("standard_value")
            if standard_value is None:
                return None, "missing_value"

            try:
                value_num = float(standard_value)
            except (TypeError, ValueError):
                return None, "parse_error"

            units = activity.get("standard_units")
            if not units:
                units = "nM"
            units_s = str(units).strip() if units is not None else "nM"

            value_in_uM = self._convert_to_uM(value_num, units_s)

            pchembl_raw = activity.get("pchembl_value")
            pchembl: Optional[float] = None
            if pchembl_raw is not None:
                try:
                    pchembl = float(pchembl_raw)
                except (TypeError, ValueError):
                    pchembl = None

            measurement = BioactivityMeasurement(
                type=canonical,
                value=value_in_uM,
                units="uM",
                pchembl=pchembl,
                assay_type=activity.get("assay_type"),
                assay_description=activity.get("assay_description"),
                document_chembl_id=activity.get("document_chembl_id"),
                confidence_score=activity.get("confidence_score"),
                normalized_type=cls_info.get("normalized_type"),
                evidence_bucket=cls_info.get("evidence_bucket"),
                mechanistic_context_hit=bool(cls_info.get("mechanistic_context_hit")),
                direct_qt_context_hit=bool(cls_info.get("direct_qt_context_hit")),
                secondary_pharmacology_context_hit=bool(
                    cls_info.get("secondary_pharmacology_context_hit")
                ),
            )

            return measurement, None

        except Exception as e:
            logger.warning("解析活性数据失败: %s", e)
            return None, "parse_error"

    def _convert_to_uM(self, value: float, units: str) -> float:
        units_lower = str(units).lower() if units else ""

        if "nm" in units_lower or "nanomolar" in units_lower:
            return value / 1000.0
        if "um" in units_lower or "micromolar" in units_lower:
            return value
        if "mm" in units_lower or "millimolar" in units_lower:
            return value * 1000.0
        if "pm" in units_lower or "picomolar" in units_lower:
            return value / 1_000_000.0
        return value / 1000.0

    def get_herg_only(self, chembl_id: str) -> Optional[TargetBioactivity]:
        all_activities = self.retrieve(chembl_id)
        return all_activities.get("hERG")

    def is_herg_inhibitor(self, chembl_id: str, threshold: float = 10.0) -> bool:
        herg_data = self.get_herg_only(chembl_id)
        if not herg_data or not herg_data.measurements:
            return False
        best_measurement = herg_data.measurements[0]
        ic50 = best_measurement.value
        return ic50 < threshold
