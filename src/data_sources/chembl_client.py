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


def _pref_name_log_preview(pref_name: str, max_len: int = 120) -> str:
    s = pref_name.strip() if pref_name else ""
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


def _extract_drugs_payload(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        d = data.get("drugs")
        if isinstance(d, list):
            return [x for x in d if isinstance(x, dict)]
    return []


def _normalize_drug_query(s: str) -> str:
    """去 NBSP、合并空白、小写，用于名称比对。"""
    t = (s or "").replace("\xa0", " ").replace("\u2009", " ")
    return " ".join(t.split()).strip().lower()


def _row_pref_name_matches_iexact(row: Dict[str, Any], pn: str) -> bool:
    pref = (row.get("pref_name") or "").strip()
    return _normalize_drug_query(pref) == _normalize_drug_query(pn)


def _row_pref_name_matches_icontains(row: Dict[str, Any], pn: str) -> bool:
    """查询串与 ChEMBL pref_name 互相包含（兼容盐型/制剂后缀）。"""
    npref = _normalize_drug_query(row.get("pref_name") or "")
    nq = _normalize_drug_query(pn)
    if not npref or not nq:
        return False
    return nq in npref or npref in nq


def _row_synonym_matches_query(row: Dict[str, Any], pn: str) -> bool:
    """molecule 列表项里若嵌 molecule_synonyms，则与查询比对。"""
    nq = _normalize_drug_query(pn)
    syns = row.get("molecule_synonyms")
    if not isinstance(syns, list):
        return False
    for s in syns:
        if not isinstance(s, dict):
            continue
        syn = (s.get("molecule_synonym") or s.get("synonym") or "").strip()
        if not syn:
            continue
        ns = _normalize_drug_query(syn)
        if ns == nq or nq in ns or ns in nq:
            return True
    return False


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

    def _try_molecule_filter(
        self,
        layer_name: str,
        extra: Dict[str, Any],
        identifier: str,
    ) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
        params: Dict[str, Any] = {
            "limit": 25,
            "format": "json",
            **extra,
        }
        data, ok = self._request_json(
            "molecule.json",
            params=params,
            resource_name=f"molecule.json({layer_name})",
            identifier=identifier,
        )
        if not ok:
            return None, False
        rows = _extract_molecules_payload(data)
        return rows, True

    def _get_molecule_by_smiles_connectivity(
        self, smiles: str
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        按 SMILES 查分子（connectivity → iexact → flexmatch）。
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

        rows, ok = try_molecule_filter(
            "connectivity",
            {"molecule_structures__canonical_smiles__connectivity": s},
        )
        if not ok:
            return {}, False
        if rows:
            return dict(rows[0]), True

        rows2, ok2 = try_molecule_filter(
            "iexact",
            {"molecule_structures__canonical_smiles__iexact": s},
        )
        if not ok2:
            return {}, False
        if rows2:
            return dict(rows2[0]), True

        rows3, ok3 = try_molecule_filter(
            "flexmatch",
            {"molecule_structures__canonical_smiles__flexmatch": smiles},
        )
        if not ok3:
            return {}, False
        if rows3:
            return dict(rows3[0]), True

        logger.warning(
            f"ChEMBL 按 SMILES 未命中（connectivity / iexact / flexmatch）| 预览={sm_prev}"
        )
        return None, True

    def get_drug_by_name(
        self,
        drug_name: str,
        smiles: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        按「药名」解析 ChEMBL 分子（DIQTA 通用名/商品名等），顺序：

        1) ``drug.json?pref_name__iexact`` → 取 ``molecule_chembl_id`` 再拉 ``molecule/{id}``
        2) ``molecule.json?preferred_name__iexact``（若服务端支持该 filter）
        3) ``molecule.json?pref_name__iexact``
        4) 同义词：``molecule_synonyms__molecule_synonym__iexact``、
           ``molecule_synonyms__synonyms__iexact``、``molecule_synonyms__synonym__iexact``
        5) ``pref_name__icontains``（多命中时优先全词匹配 pref_name）
        6) 若仍无结果且提供 ``smiles``：按 SMILES connectivity / iexact / flexmatch 兜底

        仅 SMILES、无药名时：直接走第 6 步。

        传输失败返回 ({}, False)；未命中返回 (None, True)；命中返回 (dict, True)。
        """
        pn = (drug_name or "").strip()
        sm = (smiles or "").strip()
        if not pn and not sm:
            logger.warning("get_drug_by_name: 空 drug_name 且无 SMILES")
            return None, True

        id_prev = _pref_name_log_preview(pn) if pn else _smiles_log_preview(sm)

        # 1) drug 端点（通用名/商品名与 drug.pref_name 对齐时）
        if pn:
            data, ok = self._request_json(
                "drug.json",
                params={
                    "pref_name__iexact": pn,
                    "limit": 10,
                    "format": "json",
                },
                resource_name="drug.json(pref_name__iexact)",
                identifier=id_prev,
            )
            if ok:
                drugs = _extract_drugs_payload(data)
            else:
                logger.warning(
                    "ChEMBL drug.json 请求失败，跳过 drug 层并继续尝试 molecule | %s",
                    id_prev,
                )
                drugs = []
            for dr in drugs:
                mid = dr.get("molecule_chembl_id")
                if not mid:
                    continue
                # drug 层必须与查询名一致（避免无效 filter 返回无关 drug）
                if not _row_pref_name_matches_icontains(
                    {"pref_name": dr.get("pref_name")}, pn
                ):
                    continue
                mol, ok_m = self.get_molecule_by_chembl_id(str(mid))
                if not ok_m:
                    return {}, False
                if mol:
                    logger.info(
                        "ChEMBL 命中: drug.json pref_name__iexact -> %s",
                        mid,
                    )
                    return mol, True

            # 仅使用 ChEMBL 文档中存在的 molecule filter；未知参数会被忽略并返回「全表前几
            # 条」→ 曾导致所有查询都命中同一条（如 CHEMBL2）。下面每条结果必须与本行 pref/同义词
            # 与查询字符串校验通过才采纳。
            syn_molecule_filters: List[str] = [
                "pref_name__iexact",
                "molecule_synonyms__molecule_synonym__iexact",
                "molecule_synonyms__synonym__iexact",
            ]
            for fk in syn_molecule_filters:
                rows, ok_f = self._try_molecule_filter(
                    fk,
                    {fk: pn},
                    id_prev,
                )
                if not ok_f:
                    logger.debug(
                        "ChEMBL molecule 层 %s 请求失败或 filter 不支持，尝试下一层",
                        fk,
                    )
                    continue
                if not rows:
                    continue
                for row in rows:
                    if fk == "pref_name__iexact":
                        if _row_pref_name_matches_iexact(row, pn):
                            logger.info(
                                "ChEMBL 命中: molecule.json pref_name__iexact（已校验）"
                            )
                            return dict(row), True
                    else:
                        if (
                            _row_synonym_matches_query(row, pn)
                            or _row_pref_name_matches_iexact(row, pn)
                            or _row_pref_name_matches_icontains(row, pn)
                        ):
                            logger.info(
                                "ChEMBL 命中: molecule.json %s（已校验）",
                                fk,
                            )
                            return dict(row), True

            rows_ic, ok_ic = self._try_molecule_filter(
                "pref_icontains",
                {"pref_name__icontains": pn},
                id_prev,
            )
            if not ok_ic:
                return {}, False
            if rows_ic:
                for row in rows_ic:
                    if _row_pref_name_matches_icontains(row, pn):
                        logger.info(
                            "ChEMBL 命中: molecule.json pref_name__icontains（已校验）"
                        )
                        return dict(row), True

            logger.warning(
                f"ChEMBL 按药名未命中 drug/molecule 各层 | drug_name={id_prev!r}"
            )

        if sm:
            mol_s, ok_s = self._get_molecule_by_smiles_connectivity(sm)
            if ok_s and mol_s:
                logger.info("ChEMBL 命中: SMILES connectivity/兜底")
            return mol_s, ok_s

        return None, True

    def get_molecule_by_pref_name(self, pref_name: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        按药名解析分子；等价于 ``get_drug_by_name(pref_name, smiles=None)``（不走 SMILES 兜底）。
        """
        return self.get_drug_by_name(pref_name, smiles=None)

    def get_molecule_by_smiles(self, smiles: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        """仅按 SMILES 查分子（connectivity / iexact / flexmatch）。"""
        return self._get_molecule_by_smiles_connectivity(smiles)

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
