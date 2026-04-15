"""
ChEMBL new_client 加载器。

官方 chembl_webresource_client.new_client 在导入时同步请求 /chembl/api/data/spore，
该端点常间歇性返回 500。此处延迟构建并带重试；也可通过 CHEMBL_SPORE_FILE 或
data/cache/chembl_data_spore.json 离线加载 SPORE 元数据。

注意：SPORE 可来自本地、生产或 wwwdev，但所有实际 API 请求（Settings 与 schema
base_url）一律强制为生产地址。
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from easydict import EasyDict

from chembl_webresource_client.query_set import Model, QuerySet
from chembl_webresource_client.settings import Settings

logger = logging.getLogger(__name__)

# 生产 API：QuerySet 与 schema 描述中的 base_url 必须始终一致且为生产
PROD_API_BASE = "https://www.ebi.ac.uk/chembl/api/data/"
PROD_SETTINGS_URL = "https://www.ebi.ac.uk/chembl/api/data"


class NewClient(object):
    pass


def _force_prod_schema_base(schema: dict) -> None:
    """无条件将 schema 中的 base_url 固定为生产 API（覆盖 wwwdev / 旧缓存等）。"""
    schema["base_url"] = PROD_API_BASE


def _force_prod_settings() -> None:
    """实际 HTTP 请求使用 chembl_webresource_client Settings.NEW_CLIENT_URL，强制为生产。"""
    Settings.Instance().NEW_CLIENT_URL = PROD_SETTINGS_URL


def _build_client(schema: dict) -> NewClient:
    _force_prod_schema_base(schema)
    if not schema["base_url"].endswith("/"):
        schema["base_url"] += "/"

    client = NewClient()
    client.description = EasyDict(schema)
    client.official = False

    keys = client.description.methods.keys()
    for method, definition in [
        (m, d)
        for (m, d) in client.description.methods.items()
        if (m.startswith("POST_") or m.startswith("GET_")) and m.endswith("_detail")
    ]:
        searchable = False
        if method.replace("dispatch_detail", "get_search") in keys:
            searchable = True
        name = definition["resource_name"]
        collection_name = definition["collection_name"]
        formats = [fmt for fmt in definition["formats"] if fmt not in ("jsonp", "html")]
        default_format = definition["default_format"].split("/")[-1]
        if not name:
            continue
        model = Model(name, collection_name, formats, searchable)
        qs = QuerySet(model=model)
        if default_format not in ("xml", "svg+xml"):
            qs.set_format(default_format)
        setattr(client, name, qs)

    return client


def client_from_url(url: str) -> NewClient:
    """从 URL 拉取 SPORE JSON；base_url 在 _build_client 中统一覆盖为生产。"""
    res = requests.get(url, timeout=120)
    if not res.ok:
        raise RuntimeError(
            f"ChEMBL SPORE schema HTTP {res.status_code} for {url!r}"
        )
    schema = res.json()
    return _build_client(schema)


def _repo_cached_spore_path() -> Path | None:
    """仓库默认缓存路径：data/cache/chembl_data_spore.json"""
    root = Path(__file__).resolve().parent.parent.parent
    p = root / "data" / "cache" / "chembl_data_spore.json"
    return p if p.is_file() else None


def _load_from_json_file(path: Path, log_prefix: str) -> NewClient:
    with open(path, encoding="utf-8") as f:
        schema = json.load(f)
    logger.info(f"{log_prefix}: {path}")
    return _build_client(schema)


def _spore_url_candidates() -> list[str]:
    """
    仅决定从何处下载 SPORE 元数据 JSON，不改变 API 基址。
    若设置 CHEMBL_SPORE_URL，则只使用该 URL。
    否则依次尝试生产 spore、wwwdev spore。
    """
    explicit = os.environ.get("CHEMBL_SPORE_URL", "").strip()
    if explicit:
        return [explicit]
    prod = "https://www.ebi.ac.uk/chembl/api/data/spore"
    dev = "https://wwwdev.ebi.ac.uk/chembl/api/data/spore"
    return [prod, dev]


def load_new_client(
    max_attempts: int = 2,
    initial_backoff: float = 1.5,
) -> NewClient:
    _force_prod_settings()

    # 1) 显式路径
    spore_file = os.environ.get("CHEMBL_SPORE_FILE", "").strip()
    if spore_file and os.path.isfile(spore_file):
        return _load_from_json_file(
            Path(spore_file), "ChEMBL SPORE 已从 CHEMBL_SPORE_FILE 加载"
        )

    # 2) 仓库 data/cache（优先于在线）
    cached = _repo_cached_spore_path()
    if cached is not None:
        return _load_from_json_file(
            cached, "ChEMBL SPORE 已从本地缓存加载（跳过在线拉取）"
        )

    # 3) 在线拉取 + 重试
    candidates = _spore_url_candidates()
    last_err: Exception | None = None
    for url in candidates:
        for attempt in range(max_attempts):
            try:
                client = client_from_url(url)
                if "wwwdev" in urlparse(url).netloc:
                    logger.info(
                        f"ChEMBL SPORE 元数据来自 wwwdev；实际 API 仍使用生产地址 "
                        f"{PROD_API_BASE.rstrip('/')}"
                    )
                return client
            except Exception as e:
                last_err = e
                if attempt < max_attempts - 1:
                    wait = initial_backoff * (2**attempt)
                    logger.warning(
                        f"ChEMBL SPORE 加载失败 {url} ({attempt + 1}/{max_attempts})，"
                        f"{wait:.1f}s 后重试: {e}"
                    )
                    time.sleep(wait)
        if len(candidates) > 1 and url == candidates[0]:
            logger.warning("生产端 SPORE 不可用，尝试备用 URL…")

    assert last_err is not None
    raise RuntimeError(
        "ChEMBL SPORE 在线不可用。可任选其一下载到 data/cache/chembl_data_spore.json 后重试：\n"
        "  curl -fsSL -o data/cache/chembl_data_spore.json "
        '"https://www.ebi.ac.uk/chembl/api/data/spore"\n'
        "  curl -fsSL -o data/cache/chembl_data_spore.json "
        '"https://wwwdev.ebi.ac.uk/chembl/api/data/spore"\n'
        "或设置 CHEMBL_SPORE_FILE 指向已下载的 JSON。"
    ) from last_err


new_client = load_new_client()
