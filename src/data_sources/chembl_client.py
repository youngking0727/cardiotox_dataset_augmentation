"""ChEMBL API 客户端封装（生产 API：https://www.ebi.ac.uk/chembl/api/data，直连 requests）"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote, urljoin

try:
    import requests

    CHEMBL_AVAILABLE = True
except ImportError:
    CHEMBL_AVAILABLE = False
    requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data/"

# 统一：首次失败后最多重试 4 次，间隔 2s / 4s / 8s / 16s（共 5 次尝试）
_CHEMBL_MAX_RETRIES = 4
_CHEMBL_RETRY_DELAYS_SEC = (2.0, 4.0, 8.0, 16.0)
_CHEMBL_TOTAL_ATTEMPTS = 1 + _CHEMBL_MAX_RETRIES
_REQUEST_TIMEOUT_SEC = 60.0


def standard_inchi_key_from_molecule(
    molecule: Optional[Dict[str, Any]],
) -> Optional[str]:
    """
    仅在成功 GET molecule 后的 JSON 上读取 ``standard_inchi_key``。
    用途：本地缓存键、跨库对齐、去重——不作为 similarity / drug / molecule 查询的前置条件。
    """
    if not molecule or not isinstance(molecule, dict):
        return None
    v = molecule.get("standard_inchi_key")
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _format_chembl_api_error(exc: BaseException) -> str:
    s = str(exc)
    low = s.lower()
    if "<!doctype html>" in low or "<html" in low:
        return "ChEMBL API 返回 HTML 错误页（多为服务端 500），请稍后重试或检查 EBI 服务状态。"
    if len(s) > 400:
        return s[:300] + "…"
    return s


def _smiles_log_preview(smiles: str, max_len: int = 120) -> str:
    s = smiles.strip() if smiles else ""
    if len(s) <= max_len:
        return s
    return s[:max_len] + "…"


def _payload_looks_like_html(text: str) -> bool:
    if not text:
        return False
    low = text[:8000].lower()
    if "<!doctype html>" in low:
        return True
    if "<html" in low and ("<body" in low or "<head" in low):
        return True
    return False


def _is_json_content_type(content_type: str) -> bool:
    if not content_type:
        return False
    ct = content_type.lower()
    return "application/json" in ct or "application/hal+json" in ct


def _normalize_dict_like(item: Any) -> Optional[Dict[str, Any]]:
    """将单条 API 记录转为普通 dict；无法解析则返回 None。"""
    if item is None:
        return None
    if isinstance(item, dict):
        try:
            return dict(item)
        except Exception:
            return None
    keys = getattr(item, "keys", None)
    if callable(keys):
        try:
            return {str(k): item[k] for k in item.keys()}  # type: ignore[index]
        except Exception:
            return None
    return None


def _normalize_dict_list(raw: List[Any], resource: str) -> List[Dict[str, Any]]:
    """逐条安全解析，单条异常不导致整批失败。"""
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        try:
            norm = _normalize_dict_like(item)
            if norm is not None:
                out.append(norm)
        except Exception as e:
            logger.warning(f"跳过单条 {resource} 记录解析失败 (index={i}): {e}")
    return out


def _normalize_similarity_score(raw: Any) -> Optional[float]:
    """ChEMBL 可能返回 0~1 或 0~100 的相似度；统一为 0~1，无法转换则为 None。"""
    if raw is None:
        return None
    try:
        similarity = float(raw)
    except (TypeError, ValueError):
        return None
    if similarity > 1.0:
        similarity = similarity / 100.0
    return similarity


def _extract_molecules_payload(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        m = data.get("molecules") or data.get("molecule")
        if isinstance(m, list):
            return [x for x in m if isinstance(x, dict)]
        if isinstance(m, dict):
            return [m]
    return []


def _extract_named_list(data: Any, key: str) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        raw = data.get(key)
        if isinstance(raw, list):
            return [x for x in raw if isinstance(x, dict)]
    return []


class ChEMBLClient:
    """ChEMBL API 客户端封装（REST + requests）"""

    def __init__(self, cache_dir: Optional[Path] = None, rate_limit: float = 0.5):
        if not CHEMBL_AVAILABLE or requests is None:
            raise ImportError("requests 未安装，请运行: pip install requests")

        self.cache_dir = cache_dir
        self.rate_limit = rate_limit
        self._last_call_time = 0.0
        self._api_base = CHEMBL_API_BASE

    def _rate_limit_wait(self) -> None:
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_call_time = time.time()

    def _url(self, path: str) -> str:
        p = path.lstrip("/")
        return urljoin(self._api_base, p)

    def _request_json(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        resource_name: str = "",
        identifier: str = "",
        *,
        expected_404_ok: bool = False,
    ) -> Tuple[Optional[Union[Dict[str, Any], List[Any]]], bool]:
        """
        拼完整 URL，GET，统一校验与重试。

        返回 (body, ok)：
        - ok=False：传输/解析失败（用尽重试）
        - ok=True 且 expected_404_ok：404 时 body 为 None（未命中单条资源）
        - ok=True 且 200：body 为解析后的 dict 或 list
        """
        label = resource_name or path
        last_exc: Optional[BaseException] = None
        qparams = dict(params) if params else {}
        if "format" not in qparams:
            qparams["format"] = "json"

        for attempt in range(_CHEMBL_TOTAL_ATTEMPTS):
            try:
                self._rate_limit_wait()
                url = self._url(path)
                r = requests.get(
                    url,
                    params=qparams or None,
                    timeout=_REQUEST_TIMEOUT_SEC,
                )
                if r.status_code == 404 and expected_404_ok:
                    return None, True
                if r.status_code != 200:
                    raise ValueError(f"HTTP {r.status_code}")

                ct = r.headers.get("content-type", "") or ""
                text = r.text if r.text is not None else ""
                if not _is_json_content_type(ct):
                    if _payload_looks_like_html(text):
                        raise ValueError(
                            "Content-Type 非 JSON 且正文疑似 HTML 错误页"
                        )
                if _payload_looks_like_html(text):
                    raise ValueError("正文疑似 HTML 错误页")

                try:
                    data = r.json()
                except ValueError as e:
                    raise ValueError(f"JSON 解析失败: {e}") from e

                if _payload_looks_like_html(str(data)):
                    raise ValueError("解析结果疑似 HTML 错误页")

                if not isinstance(data, (dict, list)):
                    raise ValueError(f"非预期 JSON 类型: {type(data).__name__}")

                return data, True
            except Exception as e:
                last_exc = e
                if attempt < _CHEMBL_TOTAL_ATTEMPTS - 1:
                    delay = _CHEMBL_RETRY_DELAYS_SEC[attempt]
                    logger.warning(
                        f"ChEMBL {label} 请求失败 (尝试 {attempt + 1}/{_CHEMBL_TOTAL_ATTEMPTS}) "
                        f"identifier={identifier}，{delay}s 后重试 | {_format_chembl_api_error(e)}"
                    )
                    time.sleep(delay)
                else:
                    err_tail = _format_chembl_api_error(last_exc) if last_exc else ""
                    logger.warning(
                        f"ChEMBL {label} 请求最终失败 | identifier={identifier} | "
                        f"已尝试 {_CHEMBL_TOTAL_ATTEMPTS} 次 | {err_tail}"
                    )
                    return None, False
        return None, False

    def get_molecule_by_smiles(self, smiles: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        按 SMILES 查分子（官方支持 connectivity / exact 等，无需先转 InChIKey）。

        优先顺序（与 ChEMBL 文档示例一致）：
        1) ``molecule.json?molecule_structures__canonical_smiles__connectivity=...``
        2) ``molecule_structures__canonical_smiles__iexact``（精确匹配兜底）
        3) ``molecule_structures__canonical_smiles__flexmatch``（仅在前两步无命中时使用）

        传输失败返回 ({}, False)；未命中返回 (None, True)；命中返回 (dict, True)。
        """
        sm_prev = _smiles_log_preview(smiles)
        if not (smiles or "").strip():
            logger.warning(f"ChEMBL molecule.json 空 SMILES | identifier={sm_prev}")
            return None, True

        s = smiles.strip()

        def try_molecule_filter(
            layer_name: str, extra: Dict[str, Any]
        ) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
            params: Dict[str, Any] = {
                "limit": 10,
                "format": "json",
                **extra,
            }
            data, ok = self._request_json(
                "molecule.json",
                params=params,
                resource_name=f"molecule.json({layer_name})",
                identifier=sm_prev,
            )
            if not ok:
                logger.warning(
                    f"ChEMBL 按 SMILES 查询失败（{layer_name}，传输/解析）| SMILES 预览={sm_prev}"
                )
                return None, False
            rows = _extract_molecules_payload(data)
            return rows, True

        # 1) 官方文档「Connectivity search」示例
        rows, ok = try_molecule_filter(
            "connectivity",
            {"molecule_structures__canonical_smiles__connectivity": s},
        )
        if not ok:
            return {}, False
        if rows:
            return dict(rows[0]), True

        # 2) 精确匹配（仍是对 SMILES 字段的查询，非 InChIKey）
        rows2, ok2 = try_molecule_filter(
            "iexact",
            {"molecule_structures__canonical_smiles__iexact": s},
        )
        if not ok2:
            return {}, False
        if rows2:
            return dict(rows2[0]), True

        # 3) flexmatch：仅作末级兜底，不作为主查询
        rows3, ok3 = try_molecule_filter(
            "flexmatch",
            {"molecule_structures__canonical_smiles__flexmatch": smiles},
        )
        if not ok3:
            return {}, False
        if rows3:
            return dict(rows3[0]), True

        logger.warning(
            f"ChEMBL 按 SMILES 未命中（connectivity / iexact / flexmatch 均无结果）| "
            f"预览={sm_prev}"
        )
        return None, True

    def get_molecule_by_chembl_id(
        self, chembl_id: str
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """按 ChEMBL ID 查分子。失败 ({}, False)；未命中 (None, True)；命中 (dict, True)。"""
        cid = (chembl_id or "").strip()
        if not cid:
            return None, True

        data, ok = self._request_json(
            f"molecule/{cid}.json",
            resource_name="molecule.json",
            identifier=cid,
            expected_404_ok=True,
        )
        if not ok:
            return {}, False
        if data is None:
            return None, True
        if isinstance(data, dict):
            return data, True
        return {}, False

    def get_similar_molecules(
        self, smiles: str, threshold: float = 0.8, max_results: int = 100
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        相似性检索：使用官方路径式接口
        ``/similarity/{url-encoded SMILES}/{cutoff}``（文档示例为 Tanimoto 百分数阈值 40–100）。

        失败返回 ([], False)；成功返回 (list, True)，list 可为空。
        """
        if threshold <= 1.0:
            api_cutoff = int(round(threshold * 100))
        else:
            api_cutoff = int(round(threshold))
        api_cutoff = max(40, min(100, api_cutoff))

        sm_prev = _smiles_log_preview(smiles)
        if not (smiles or "").strip():
            return [], True

        s = smiles.strip()
        enc = quote(s, safe="")
        similar: List[Dict[str, Any]] = []
        page_size = 50
        offset = 0

        while len(similar) < max_results:
            # 官方文档：.../similarity/<SMILES>/<cutoff>，SMILES 需 URL 编码（如括号 → %28/%29）
            path = f"similarity/{enc}/{api_cutoff}"
            data, ok = self._request_json(
                path,
                params={
                    "format": "json",
                    "limit": page_size,
                    "offset": offset,
                },
                resource_name="similarity(path)",
                identifier=sm_prev,
            )
            if not ok:
                return [], False

            items: List[Any] = []
            if isinstance(data, dict):
                raw = (
                    data.get("molecules")
                    or data.get("molecule")
                    or data.get("similarities")
                    or []
                )
                if isinstance(raw, dict):
                    raw = raw.get("molecule") or raw.get("molecules") or []
                if isinstance(raw, list):
                    items = raw
            elif isinstance(data, list):
                items = data

            if not items:
                break

            page_added = 0
            for item in items:
                if len(similar) >= max_results:
                    break
                if not isinstance(item, dict):
                    continue
                raw_sim = item.get("similarity") or item.get("tanimoto") or item.get("score")
                similarity = _normalize_similarity_score(raw_sim)
                ms = item.get("molecule_structures")
                if not isinstance(ms, dict):
                    ms = {}
                smi = (
                    item.get("canonical_smiles")
                    or ms.get("canonical_smiles")
                    or ""
                )
                chembl_id = str(
                    item.get("molecule_chembl_id") or item.get("chembl_id") or ""
                )
                similar.append(
                    {
                        "chembl_id": chembl_id or None,
                        "smiles": smi or None,
                        "similarity": similarity,
                    }
                )
                page_added += 1

            if page_added < page_size or not items:
                break
            offset += page_size

        return similar, True

    def get_activities(
        self,
        chembl_id: str,
        target_chembl_ids: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        获取 activity 数据（附加证据）。
        传输失败时返回已拉取数据或空列表且第二项为 True，并打 warning，不把分子主流程判为取数失败。
        """

        identifier = chembl_id

        def fetch_page(
            base_params: Dict[str, Any]
        ) -> Tuple[List[Dict[str, Any]], bool]:
            acc: List[Dict[str, Any]] = []
            off = 0
            limit = 1000
            while True:
                params = {
                    **base_params,
                    "limit": limit,
                    "offset": off,
                    "format": "json",
                }
                data, ok = self._request_json(
                    "activity.json",
                    params=params,
                    resource_name="activity.json",
                    identifier=identifier,
                )
                if not ok:
                    logger.warning(
                        f"ChEMBL activity.json 不可用，使用已拉取或空活性列表（附加证据）| molecule={identifier}"
                    )
                    return acc, True
                chunk = _extract_named_list(data, "activities")
                acc.extend(_normalize_dict_list(chunk, "activity"))
                if len(chunk) < limit:
                    break
                off += limit
            return acc, True

        if target_chembl_ids:
            merged: List[Dict[str, Any]] = []
            for target_id in target_chembl_ids:
                tid = (target_id or "").strip()
                if not tid:
                    continue
                base = {
                    "molecule_chembl_id__iexact": chembl_id,
                    "target_chembl_id__iexact": tid,
                }
                part, _ok = fetch_page(base)
                merged.extend(part)
            return merged, True

        base = {"molecule_chembl_id__iexact": chembl_id}
        return fetch_page(base)

    def get_molecule_properties(self, chembl_id: str) -> Optional[Dict[str, Any]]:
        molecule, ok = self.get_molecule_by_chembl_id(chembl_id)
        if ok and molecule:
            return molecule.get("molecule_properties")
        return None

    def get_target_by_chembl_id(
        self, target_chembl_id: str
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        tid = (target_chembl_id or "").strip()
        if not tid:
            return None, True

        data, ok = self._request_json(
            f"target/{tid}.json",
            resource_name="target.json",
            identifier=tid,
            expected_404_ok=True,
        )
        if not ok:
            return {}, False
        if data is None:
            list_data, ok2 = self._request_json(
                "target.json",
                params={
                    "target_chembl_id__iexact": tid,
                    "limit": 1,
                },
                resource_name="target.json(fallback)",
                identifier=tid,
            )
            if not ok2:
                return {}, False
            rows = _extract_named_list(list_data, "targets")
            if rows:
                return dict(rows[0]), True
            return None, True
        if isinstance(data, dict):
            return data, True
        return {}, False

    def get_mechanisms(self, chembl_id: str) -> Tuple[List[Dict[str, Any]], bool]:
        """获取 mechanism 数据（附加证据）；请求失败时返回已拉取数据或空列表且第二项为 True，不抛异常。"""

        def fetch_all() -> Tuple[List[Dict[str, Any]], bool]:
            acc: List[Dict[str, Any]] = []
            offset = 0
            limit = 1000
            while True:
                params = {
                    "molecule_chembl_id__iexact": chembl_id,
                    "limit": limit,
                    "offset": offset,
                    "format": "json",
                }
                data, ok = self._request_json(
                    "mechanism.json",
                    params=params,
                    resource_name="mechanism.json",
                    identifier=chembl_id,
                )
                if not ok:
                    logger.warning(
                        f"ChEMBL mechanism.json 不可用，使用已拉取或空列表（附加证据）| molecule={chembl_id}"
                    )
                    return acc, True
                chunk = _extract_named_list(data, "mechanisms")
                acc.extend(_normalize_dict_list(chunk, "mechanism"))
                if len(chunk) < limit:
                    break
                offset += limit
            return acc, True

        return fetch_all()

    def get_drug_indications(self, chembl_id: str) -> Tuple[List[Dict[str, Any]], bool]:
        """获取 drug indication 数据（附加证据）。"""

        def fetch_all() -> Tuple[List[Dict[str, Any]], bool]:
            acc: List[Dict[str, Any]] = []
            offset = 0
            limit = 1000
            while True:
                params = {
                    "molecule_chembl_id__iexact": chembl_id,
                    "limit": limit,
                    "offset": offset,
                    "format": "json",
                }
                data, ok = self._request_json(
                    "drug_indication.json",
                    params=params,
                    resource_name="drug_indication.json",
                    identifier=chembl_id,
                )
                if not ok:
                    logger.warning(
                        f"ChEMBL drug_indication.json 不可用，使用已拉取或空列表（附加证据）| molecule={chembl_id}"
                    )
                    return acc, True
                chunk = _extract_named_list(data, "drug_indications")
                acc.extend(_normalize_dict_list(chunk, "drug_indication"))
                if len(chunk) < limit:
                    break
                offset += limit
            return acc, True

        return fetch_all()

    def get_drug_info(self, chembl_id: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        获取 drug 信息（附加证据）：按 ``molecule_chembl_id__iexact`` 过滤 drug 资源（与官方 drug 端点设计一致）。
        {"drugs": []} 或传输失败均视为无 drug 记录：(None, True)，并打 warning；有记录 (dict, True)。
        """
        params = {
            "molecule_chembl_id__iexact": chembl_id,
            "limit": 5,
            "format": "json",
        }
        data, ok = self._request_json(
            "drug.json",
            params=params,
            resource_name="drug.json",
            identifier=chembl_id,
        )
        if not ok:
            logger.warning(
                f"ChEMBL drug.json 不可用，按无 drug 记录处理（附加证据）| molecule={chembl_id}"
            )
            return None, True
        if not isinstance(data, dict):
            return None, True
        drugs = data.get("drugs")
        if drugs is None:
            return None, True
        if isinstance(drugs, list) and len(drugs) == 0:
            return None, True
        if isinstance(drugs, list) and drugs:
            first = _normalize_dict_like(drugs[0])
            if first is not None:
                return first, True
        return None, True


def create_chembl_client(cache_dir: Optional[Path] = None) -> ChEMBLClient:
    """创建 ChEMBL 客户端的工厂函数"""
    return ChEMBLClient(cache_dir=cache_dir)
