"""M3-C: 文献检索器模块"""

import logging
from typing import List, Optional, Dict, Any, Set

from schemas import LiteratureEvidence, PubMedArticle, PatentInfo
from utils.chembl_client import ChEMBLClient
from utils.pubmed_client import PubMedClient
from utils.cache import get_global_cache
from rules.cardiotox_evidence_rules import (
    get_evidence_rules,
    classify_literature_text_buckets,
    EVIDENCE_PRIORITY_1,
    EVIDENCE_PRIORITY_2,
    EVIDENCE_PRIORITY_3,
    priority_rank,
)

logger = logging.getLogger(__name__)


# 心脏毒性相关关键词（路径 A / 非 path_b 保留）
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

    def retrieve(
        self,
        chembl_id: str,
        drug_name: str,
        max_pubmed: int = 30,
        *,
        pubmed_query_name: Optional[str] = None,
        path_b: bool = False,
        child_pref_name: Optional[str] = None,
        parent_drug_name: Optional[str] = None,
    ) -> LiteratureEvidence:
        """
        检索文献证据

        Args:
            chembl_id: 分子的ChEMBL ID
            drug_name: 默认检索名（未传 pubmed_query_name 时用于 PubMed/专利）
            max_pubmed: Path A 单查询最大篇数；Path B 下为「每查询约上限」的参考，总篇数由多查询合并控制
            pubmed_query_name: 若设置（如路径 B 父药名），非 path_b 时 PubMed 用该词
            path_b: Path B 多查询合并 + 文献分层
            child_pref_name: 子分子 pref_name
            parent_drug_name: 父药/ DIQTA drug_name
        """
        logger.info(f"检索文献证据: {chembl_id} path_b={path_b}")
        if path_b:
            return self._retrieve_path_b(
                chembl_id=chembl_id,
                drug_name=drug_name,
                child_pref_name=child_pref_name or "",
                parent_drug_name=parent_drug_name or "",
                max_per_query=max(8, max_pubmed // 3),
            )

        q_pubmed = (pubmed_query_name or "").strip() or drug_name
        if pubmed_query_name and (pubmed_query_name or "").strip():
            logger.info(
                "M3C PubMed：使用 pubmed_query_name=%r（路径 B 父药锚定）",
                q_pubmed,
            )

        pubmed_articles = self._search_pubmed(q_pubmed, max_pubmed)
        patents = self._search_patents(q_pubmed)

        return LiteratureEvidence(
            pubmed_articles=pubmed_articles,
            patents=patents,
        )

    def _retrieve_path_b(
        self,
        chembl_id: str,
        drug_name: str,
        child_pref_name: str,
        parent_drug_name: str,
        max_per_query: int,
    ) -> LiteratureEvidence:
        rules = get_evidence_rules()
        direct_or = self._or_clause(rules.get("direct_qt_terms") or [])
        mech_or = self._or_clause(rules.get("mechanistic_terms") or [])
        sec_or = self._or_clause(rules.get("secondary_terms") or [])

        bases = []
        for b in (child_pref_name, parent_drug_name, drug_name):
            s = (b or "").strip()
            if s and s not in bases:
                bases.append(s)

        queries: List[str] = []
        for b in bases:
            if direct_or:
                queries.append(f'({b}) AND ({direct_or})')
            if mech_or:
                queries.append(f'({b}) AND ({mech_or})')
            if sec_or:
                queries.append(f'({b}) AND ({sec_or})')

        if not queries:
            return LiteratureEvidence(pubmed_articles=[], patents=[])

        seen_pmids: Set[str] = set()
        pmid_order: List[str] = []
        for q in queries:
            ids = self.pubmed_client.search_pubmed(q, max_results=max_per_query)
            for pmid in ids:
                if pmid not in seen_pmids:
                    seen_pmids.add(pmid)
                    pmid_order.append(pmid)

        # 总篇数上限，避免 Path B 查询组合爆炸
        cap = min(120, max(40, max_per_query * max(4, len(queries))))
        pmid_order = pmid_order[:cap]

        raw_articles = self.pubmed_client.fetch_articles(pmid_order)
        name_lower = {x.lower() for x in bases}

        classified: List[PubMedArticle] = []
        p1 = p2 = p3 = 0
        top_bucket: Optional[str] = None
        top_codes: List[str] = []
        best_rank = 0

        for art in raw_articles:
            title = art.get("title", "") or ""
            abstract = art.get("abstract", "") or ""
            mesh_terms = art.get("mesh_terms", []) or []
            text_l = (title + " " + abstract).lower()
            mol_mention = any(n in text_l for n in name_lower) if name_lower else False
            if not mol_mention and chembl_id.lower() in text_l:
                mol_mention = True

            bucket, rcodes = classify_literature_text_buckets(title, abstract, mesh_terms)
            if bucket == EVIDENCE_PRIORITY_1:
                p1 += 1
            elif bucket == EVIDENCE_PRIORITY_2:
                p2 += 1
            elif bucket == EVIDENCE_PRIORITY_3:
                p3 += 1

            rk = priority_rank(bucket if bucket != "irrelevant" else None)
            if rk > best_rank:
                best_rank = rk
                top_bucket = bucket if bucket != "irrelevant" else None
                top_codes = list(rcodes)

            is_review = art.get("is_review", False)
            if not is_review:
                is_review = any(
                    term in REVIEW_MESH_TERMS for term in mesh_terms
                )

            kw_hits: List[str] = []
            for kw in CARDIOTOX_KEYWORDS:
                if kw.lower() in text_l:
                    kw_hits.append(kw)

            classified.append(
                PubMedArticle(
                    pmid=str(art.get("pmid") or ""),
                    title=title,
                    abstract=abstract or None,
                    mesh_terms=list(mesh_terms),
                    is_review=is_review,
                    relevance_keywords_hit=kw_hits,
                    molecule_mentioned=mol_mention,
                    publication_year=art.get("publication_year"),
                    relevance_bucket=bucket,
                    relevance_reason_codes=rcodes,
                )
            )

        patents = self._search_patents(bases[0] if bases else drug_name)

        le = LiteratureEvidence(
            pubmed_articles=classified,
            patents=patents,
            priority_1_article_count=p1,
            priority_2_article_count=p2,
            priority_3_article_count=p3,
            top_relevance_bucket=top_bucket,
            top_relevance_reason_codes=top_codes,
        )
        logger.info(
            "M3C Path B 文献合并 | chembl_id=%s | n=%s | p1=%s p2=%s p3=%s top=%s",
            chembl_id,
            len(classified),
            p1,
            p2,
            p3,
            top_bucket,
        )
        return le

    @staticmethod
    def _or_clause(terms: List[str]) -> str:
        clean = [f'"{t.strip()}"' for t in terms if (t or "").strip()]
        if not clean:
            return ""
        return " OR ".join(clean[:25])

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

            result = []
            for article in articles:
                is_review = article.get("is_review", False)
                if not is_review:
                    mesh_terms = article.get("mesh_terms", [])
                    is_review = any(term in REVIEW_MESH_TERMS for term in mesh_terms)

                b, rc = classify_literature_text_buckets(
                    article.get("title", "") or "",
                    article.get("abstract") or "",
                    article.get("mesh_terms", []),
                )
                result.append(PubMedArticle(
                    pmid=article.get("pmid", ""),
                    title=article.get("title", ""),
                    abstract=article.get("abstract"),
                    mesh_terms=article.get("mesh_terms", []),
                    is_review=is_review,
                    relevance_keywords_hit=article.get("relevance_keywords_hit", []),
                    molecule_mentioned=article.get("molecule_mentioned", True),
                    publication_year=article.get("publication_year"),
                    relevance_bucket=b,
                    relevance_reason_codes=rc,
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
        return []

    def _get_related_pubmed_from_chembl(self, chembl_id: str) -> List[str]:
        try:
            raw = self.chembl_client.get_activities(chembl_id)
            activities = raw[0] if isinstance(raw, tuple) else raw
            if not isinstance(activities, list):
                activities = []

            doc_ids = set()
            for activity in activities:
                if not isinstance(activity, dict):
                    continue
                doc_id = activity.get("document_chembl_id")
                if doc_id:
                    doc_ids.add(doc_id)

            return []

        except Exception as e:
            logger.warning(f"从ChEMBL获取关联文献失败: {chembl_id}, 错误: {e}")
            return []

    @staticmethod
    def count_cardiotox_relevant(articles: List[PubMedArticle]) -> int:
        count = 0
        for article in articles:
            if article.is_review:
                continue
            if article.relevance_keywords_hit and article.molecule_mentioned:
                count += 1
        return count

    def get_cardiotox_relevant_count(self, articles: List[PubMedArticle]) -> int:
        return self.count_cardiotox_relevant(articles)


def create_literature_retriever() -> LiteratureRetriever:
    return LiteratureRetriever()
