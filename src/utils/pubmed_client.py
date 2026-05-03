"""PubMed Pipeline Service 客户端 — 调用本地 /pipeline/chembl 服务获取文献+证据段落"""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import requests

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_URL = "http://localhost:8000/pipeline/chembl"


def _env_or_arg(value: Optional[str]) -> str:
    v = (value or "").strip()
    if not v or v.startswith("${"):
        return ""
    return v


def apply_ncbi_settings_from_config(config: Dict[str, Any]) -> None:
    """保留此函数以兼容 run_pipeline.py 中的调用；本地服务不需要 NCBI 配置。"""


class PubMedClient:
    """PubMed 文献客户端 — 通过本地 /pipeline/chembl 服务获取文献与 QT/hERG 证据段落"""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        rate_limit: float = 0.5,
        ncbi_email: Optional[str] = None,
        ncbi_api_key: Optional[str] = None,
        ncbi_tool: Optional[str] = None,
        service_url: Optional[str] = None,
        timeout: int = 300,
    ):
        self.cache_dir = cache_dir
        self.rate_limit = rate_limit
        self.session = requests.Session()
        self._service_url = (service_url or "").strip() or DEFAULT_SERVICE_URL
        self._timeout = timeout

    def fetch_pipeline_articles(
        self,
        molecule_chembl_id: str,
        top_n: int = 20,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        调用 /pipeline/chembl 获取文献 + 证据段落

        Args:
            molecule_chembl_id: ChEMBL ID（必填）
            top_n: 返回文章数上限
            **kwargs: 其他参数直接传给服务（pref_name, molecule_synonyms, activities, documents 等）

        Returns:
            服务返回的完整 JSON dict，包含 articles 和 chembl_enrichment；
            服务不可达时返回 None
        """
        payload = {**kwargs, "molecule_chembl_id": molecule_chembl_id, "top_n": top_n}

        try:
            resp = self.session.post(
                self._service_url,
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles") or []
            enrichment = data.get("chembl_enrichment") or {}
            logger.info(
                "Pipeline 服务返回 | chembl_id=%s | articles=%d | name_variants=%s | known_pmids=%d",
                molecule_chembl_id,
                len(articles),
                enrichment.get("name_variants", []),
                len(enrichment.get("known_pubmed_ids") or []),
            )
            return data
        except requests.exceptions.ConnectionError:
            logger.error(
                "本地 Pipeline 服务不可达 (%s)；请确认服务已启动",
                self._service_url,
            )
            return None
        except Exception as e:
            logger.warning(f"Pipeline 服务调用失败: {e}")
            return None


def create_pubmed_client(cache_dir: Optional[Path] = None) -> PubMedClient:
    return PubMedClient(cache_dir=cache_dir)