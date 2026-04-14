"""M3-C: 文献检索器模块"""

import logging
from typing import List, Optional, Dict, Any, Set
from pathlib import Path

from ..schemas import LiteratureEvidence, PubMedArticle, PatentInfo
from ..utils.chembl_client import ChEMBLClient
from ..utils.pubmed_client import PubMedClient
from ..utils.cache import get_global_cache

logger = logging.getLogger(__name__)


# 心脏毒性相关关键词
CARDIOTOX_KEYWORDS = [
    "hERG", "HERG", "KCNH2",
    "QT prolongation", "QT interval",
    "Torsades de Pointes", "Torsade",
    "cardiotoxicity", "cardiac toxicity",
    "arrhythmia", "ventricular arrhythmia",
    "sudden cardiac death",
    "Nav1.5", "SCN5A",
    "Cav1.2", "CACNA1C",
    "L-type calcium channel"
]

# 综述Mesh词（用于过滤）
REVIEW_MESH_TERMS = ["Review", "Meta-Analysis", "Systematic Review"]


class LiteratureRetriever:
    """文献检索器"""

    def __init__(self, chembl_client: Optional[ChEMBLClient] = None,
                 pubmed_client: Optional[PubMedClient] = None):
        """
        初始化文献检索器

        Args:
            chembl_client: ChEMBL客户端
            pubmed_client: PubMed客户端
        """
        self.chembl_client = chembl_client or ChEMBLClient()
        self.pubmed_client = pubmed_client or PubMedClient()
        self.cache = get_global_cache()

    def retrieve(self, chembl_id: str, drug_name: str,
                max_pubmed: int = 30) -> LiteratureEvidence:
        """
        检索文献证据

        Args:
            chembl_id: 分子的ChEMBL ID
            drug_name: 药物名称
            max_pubmed: 最大PubMed文章数

        Returns:
            文献证据对象
        """
        logger.info(f"检索文献证据: {chembl_id}")

        # 1. 从PubMed检索心脏毒性相关文章
        pubmed_articles = self._search_pubmed(drug_name, max_pubmed)

        # 2. 从专利数据库检索（暂为占位）
        patents = self._search_patents(drug_name)

        return LiteratureEvidence(
            pubmed_articles=pubmed_articles,
            patents=patents
        )

    def _search_pubmed(self, drug_name: str, max_results: int) -> List[PubMedArticle]:
        """
        从PubMed搜索心脏毒性相关文献

        Args:
            drug_name: 药物名称
            max_results: 最大结果数

        Returns:
            PubMed文章列表
        """
        try:
            articles = self.pubmed_client.search_cardiotox_articles(
                drug_name=drug_name,
                keywords=CARDIOTOX_KEYWORDS,
                max_results=max_results
            )

            # 过滤综述文章（标记但不删除）
            result = []
            for article in articles:
                # 检查是否为综述
                is_review = article.get("is_review", False)
                if not is_review:
                    # 额外检查Mesh词
                    mesh_terms = article.get("mesh_terms", [])
                    is_review = any(term in REVIEW_MESH_TERMS for term in mesh_terms)

                result.append(PubMedArticle(
                    pmid=article.get("pmid", ""),
                    title=article.get("title", ""),
                    abstract=article.get("abstract"),
                    mesh_terms=article.get("mesh_terms", []),
                    is_review=is_review,
                    relevance_keywords_hit=article.get("relevance_keywords_hit", []),
                    molecule_mentioned=article.get("molecule_mentioned", True),
                    publication_year=article.get("publication_year")
                ))

            return result

        except Exception as e:
            logger.warning(f"PubMed搜索失败: {drug_name}, 错误: {e}")
            return []

    def _search_patents(self, drug_name: str) -> List[PatentInfo]:
        """
        搜索专利数据库

        Args:
            drug_name: 药物名称

        Returns:
            专利列表
        """
        # TODO: 实际实现需要接入:
        # - Google Patents API
        # - SureChEMBL

        # 当前返回空占位符
        return []

    def _get_related_pubmed_from_chembl(self, chembl_id: str) -> List[str]:
        """
        从ChEMBL获取直接关联的PubMed ID

        Args:
            chembl_id: ChEMBL ID

        Returns:
            PubMed ID列表
        """
        try:
            activities = self.chembl_client.get_activities(chembl_id)

            # 提取关联的document_chembl_id
            doc_ids = set()
            for activity in activities:
                doc_id = activity.get("document_chembl_id")
                if doc_id:
                    doc_ids.add(doc_id)

            # document_chembl_id需要进一步转换为PubMed ID
            # ChEMBL文档可能关联PubMed，但API不直接返回PMID
            # 这需要额外的查询

            return []

        except Exception as e:
            logger.warning(f"从ChEMBL获取关联文献失败: {chembl_id}, 错误: {e}")
            return []

    def get_cardiotox_relevant_count(self, articles: List[PubMedArticle]) -> int:
        """
        获取心脏毒性相关文章数量

        Args:
            articles: PubMed文章列表

        Returns:
            相关文章数量
        """
        count = 0
        for article in articles:
            # 跳过综述
            if article.is_review:
                continue

            # 关键词命中且分子被提及
            if article.relevance_keywords_hit and article.molecule_mentioned:
                count += 1

        return count


def create_literature_retriever() -> LiteratureRetriever:
    """创建文献检索器的工厂函数"""
    return LiteratureRetriever()