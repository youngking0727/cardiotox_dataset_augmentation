"""PubMed API客户端封装"""

import os
import time
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from pathlib import Path
import requests

logger = logging.getLogger(__name__)

_EUTILS_EMAIL_WARNED = False
_EUTILS_MISUSE_LOGGED = False


def _env_or_arg(value: Optional[str]) -> str:
    """忽略 VS Code 未解析的 ${env:...} 占位符，避免阻断 pipeline.yaml 中的邮箱。"""
    v = (value or "").strip()
    if not v or v.startswith("${"):
        return ""
    return v


def apply_ncbi_settings_from_config(config: Dict[str, Any]) -> None:
    """将 pipeline.yaml 中的 NCBI 联系信息写入环境变量（未设置或无效占位符时）。"""
    pub = (config.get("api") or {}).get("pubmed") or {}
    email = (pub.get("ncbi_email") or "").strip()
    if email and not _env_or_arg(os.environ.get("NCBI_EMAIL")):
        os.environ["NCBI_EMAIL"] = email
    key = (pub.get("ncbi_api_key") or "").strip()
    if key and not _env_or_arg(os.environ.get("NCBI_API_KEY")):
        os.environ["NCBI_API_KEY"] = key
    tool = (pub.get("ncbi_tool") or "").strip()
    if tool and not _env_or_arg(os.environ.get("NCBI_TOOL")):
        os.environ["NCBI_TOOL"] = tool


class PubMedClient:
    """PubMed E-utilities API客户端"""

    EUTIL_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        rate_limit: float = 0.5,
        ncbi_email: Optional[str] = None,
        ncbi_api_key: Optional[str] = None,
        ncbi_tool: Optional[str] = None,
    ):
        """
        初始化PubMed客户端

        Args:
            cache_dir: 缓存目录
            rate_limit: API调用间隔（秒）
            ncbi_email: 联系邮箱（若环境变量 NCBI_EMAIL 未设置或无效则使用此值）
            ncbi_api_key: NCBI API key（可选）
            ncbi_tool: tool 名称（可选）
        """
        self.cache_dir = cache_dir
        self.rate_limit = rate_limit
        self._last_call_time = 0
        self.session = requests.Session()
        # NCBI 要求 E-utilities 请求携带 tool、email；否则易重定向至 misuse/abuse 页
        # 优先级：有效环境变量 > 构造函数参数（来自 pipeline.yaml）> EMAIL
        self._email = (
            _env_or_arg(os.environ.get("NCBI_EMAIL"))
            or _env_or_arg(ncbi_email)
            or _env_or_arg(os.environ.get("EMAIL"))
        )
        self._api_key = _env_or_arg(os.environ.get("NCBI_API_KEY")) or _env_or_arg(ncbi_api_key)
        self._tool = (
            _env_or_arg(os.environ.get("NCBI_TOOL"))
            or _env_or_arg(ncbi_tool)
            or "cardiotox_dataset_augmentation"
        )
        global _EUTILS_EMAIL_WARNED
        if not self._email and not _EUTILS_EMAIL_WARNED:
            logger.warning(
                "未设置 NCBI_EMAIL（或 EMAIL），PubMed 可能被限流或重定向到 misuse.ncbi.nlm.nih.gov；"
                "请设置环境变量 NCBI_EMAIL，可选 NCBI_API_KEY 提高限流"
            )
            _EUTILS_EMAIL_WARNED = True

    def _eutils_params(self) -> Dict[str, str]:
        p: Dict[str, str] = {"tool": self._tool}
        if self._email:
            p["email"] = self._email
        if self._api_key:
            p["api_key"] = self._api_key
        return p

    def _rate_limit_wait(self):
        """速率限制等待"""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_call_time = time.time()

    def _log_misuse_redirect(self) -> None:
        global _EUTILS_MISUSE_LOGGED
        if not _EUTILS_MISUSE_LOGGED:
            _EUTILS_MISUSE_LOGGED = True
            if self._email:
                logger.error(
                    "NCBI 将请求重定向到滥用/限流页（请求已携带 tool/email）。"
                    "若配置无误，多为本机或集群出口 IP 被限流/封禁，或共享网段滥用；"
                    "可尝试更换网络、配置 HTTPS 代理、或联系 NCBI 支持；"
                    "参见 https://www.ncbi.nlm.nih.gov/books/NBK25497/"
                )
            else:
                logger.error(
                    "NCBI 将请求重定向到滥用/限流页。请在 config/pipeline.yaml 的 api.pubmed.ncbi_email "
                    "或环境变量 NCBI_EMAIL 设置联系邮箱，可选 NCBI_API_KEY；"
                    "参见 https://www.ncbi.nlm.nih.gov/books/NBK25497/"
                )
        else:
            logger.debug("NCBI misuse 重定向（已提示过，后续请求仍失败）")

    def _make_request(self, url: str, params: Optional[Dict] = None):
        """发送 E-utilities 请求（自动合并 tool/email/api_key）。"""
        merged: Dict[str, Any] = dict(params or {})
        merged.update(self._eutils_params())
        self._rate_limit_wait()
        try:
            # 先禁止跟随重定向：若 NCBI 返回 302 到 misuse 子域，直接判定滥用/缺 email，
            # 避免跟随到 misuse 主机（部分环境对该主机路由不可达，会报 Network unreachable）。
            response = self.session.get(url, params=merged, timeout=30, allow_redirects=False)
            if response.status_code in (301, 302, 303, 307, 308):
                loc = (response.headers.get("Location") or "").lower()
                if "misuse.ncbi" in loc or "abuse.shtml" in loc:
                    self._log_misuse_redirect()
                    return None
                response = self.session.get(url, params=merged, timeout=30, allow_redirects=True)
            final_url = (response.url or "").lower()
            if "misuse.ncbi.nlm.nih.gov" in final_url or "abuse.shtml" in final_url:
                self._log_misuse_redirect()
                return None
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            err_s = str(e).lower()
            if "misuse.ncbi" in err_s or "abuse.shtml" in err_s:
                self._log_misuse_redirect()
                return None
            if "network is unreachable" in err_s or "errno 101" in err_s:
                logger.error(
                    "无法连接 NCBI（网络不可达，Errno 101）。请检查本机/集群出站、防火墙、代理或 VPN，"
                    "确认可访问 https://eutils.ncbi.nlm.nih.gov；若仅能解析 eutils 但无法访问 misuse 子域，"
                    "请勿忽略 pipeline 中的 NCBI_EMAIL。"
                )
            else:
                logger.warning(f"PubMed API请求失败: {url}, 错误: {e}")
            return None
        except Exception as e:
            logger.warning(f"PubMed API请求失败: {url}, 错误: {e}")
            return None

    def search_pubmed(self, query: str, max_results: int = 100,
                     retstart: int = 0) -> List[str]:
        """
        搜索PubMed

        Args:
            query: 搜索查询
            max_results: 最大结果数
            retstart: 起始位置

        Returns:
            PMID列表
        """
        url = f"{self.EUTIL_BASE_URL}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retstart": retstart,
            "retmode": "json",
            "sort": "relevance"
        }

        response = self._make_request(url, params)
        if not response:
            return []

        try:
            data = response.json()
            return data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            logger.warning(f"解析PubMed搜索结果失败: {e}")
            return []

    def fetch_articles(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """
        获取文章详情

        Args:
            pmids: PMID列表

        Returns:
            文章详情列表
        """
        if not pmids:
            return []

        url = f"{self.EUTIL_BASE_URL}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract"
        }

        response = self._make_request(url, params)
        if not response:
            return []

        try:
            root = ET.fromstring(response.text)
            articles = []

            for article in root.findall(".//PubmedArticle"):
                pmid = article.find(".//PMID")
                article_data = {
                    "pmid": pmid.text if pmid is not None else None,
                    "title": "",
                    "abstract": "",
                    "mesh_terms": [],
                    "is_review": False,
                    "publication_year": None
                }

                # 标题
                title = article.find(".//ArticleTitle")
                if title is not None:
                    article_data["title"] = title.text or ""

                # 摘要
                abstract = article.find(".//AbstractText")
                if abstract is not None:
                    article_data["abstract"] = abstract.text or ""

                # Mesh词
                mesh_heading_list = article.find(".//MeshHeadingList")
                if mesh_heading_list is not None:
                    for mesh in mesh_heading_list.findall(".//MeshHeading"):
                        descriptor = mesh.find(".//DescriptorName")
                        if descriptor is not None and descriptor.text:
                            article_data["mesh_terms"].append(descriptor.text)
                            # 检查是否为综述
                            if descriptor.text.lower() == "review":
                                article_data["is_review"] = True

                # 发表年份
                pub_date = article.find(".//PubDate")
                if pub_date is not None:
                    year = pub_date.find(".//Year")
                    if year is not None and year.text:
                        article_data["publication_year"] = int(year.text)

                articles.append(article_data)

            return articles
        except Exception as e:
            logger.warning(f"解析文章详情失败: {e}")
            return []

    def search_cardiotox_articles(self, drug_name: str,
                                  keywords: List[str],
                                  max_results: int = 50) -> List[Dict[str, Any]]:
        """
        搜索与心脏毒性相关的文章

        Args:
            drug_name: 药物名称
            keywords: 心脏毒性关键词列表
            max_results: 最大结果数

        Returns:
            相关文章列表
        """
        # 构建复合查询: 药物名称 AND (关键词1 OR 关键词2 OR ...)
        keyword_query = " OR ".join([f'"{kw}"' for kw in keywords])
        query = f'({drug_name}) AND ({keyword_query})'

        pmids = self.search_pubmed(query, max_results=max_results)
        if not pmids:
            return []

        articles = self.fetch_articles(pmids)

        # 过滤：标题或摘要中必须包含药物名称
        filtered_articles = []
        drug_name_lower = drug_name.lower()

        for article in articles:
            text = (article.get("title", "") + " " + article.get("abstract", "")).lower()
            article["molecule_mentioned"] = drug_name_lower in text

            # 检查关键词命中情况
            keywords_hit = []
            for kw in keywords:
                if kw.lower() in text:
                    keywords_hit.append(kw)
            article["relevance_keywords_hit"] = keywords_hit

            # 只保留标题或摘要中提及该分子的文章
            if article["molecule_mentioned"]:
                filtered_articles.append(article)

        return filtered_articles


def create_pubmed_client(cache_dir: Optional[Path] = None) -> PubMedClient:
    """创建PubMed客户端的工厂函数"""
    return PubMedClient(cache_dir=cache_dir)