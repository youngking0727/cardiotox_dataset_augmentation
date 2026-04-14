"""PubMed API客户端封装"""

import time
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from pathlib import Path
from urllib.parse import quote_plus
import requests

logger = logging.getLogger(__name__)


class PubMedClient:
    """PubMed E-utilities API客户端"""

    EUTIL_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, cache_dir: Optional[Path] = None, rate_limit: float = 0.5):
        """
        初始化PubMed客户端

        Args:
            cache_dir: 缓存目录
            rate_limit: API调用间隔（秒）
        """
        self.cache_dir = cache_dir
        self.rate_limit = rate_limit
        self._last_call_time = 0
        self.session = requests.Session()

    def _rate_limit_wait(self):
        """速率限制等待"""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_call_time = time.time()

    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """发送API请求"""
        self._rate_limit_wait()
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
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