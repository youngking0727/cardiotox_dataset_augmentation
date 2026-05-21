"""M3-C: 文献检索器模块 — 通过本地 /pipeline/chembl 服务获取文献与 QT/hERG 证据段落"""

import logging
from typing import List, Optional, Dict, Any

from schemas import LiteratureEvidence, PubMedArticle, PatentInfo, EvidenceContext
from utils.chembl_client import ChEMBLClient
from utils.pubmed_client import PubMedClient
from rules.cardiotox_evidence_rules import (
    classify_literature_text_buckets,
    EVIDENCE_PRIORITY_1,
    EVIDENCE_PRIORITY_2,
    EVIDENCE_PRIORITY_3,
    priority_rank,
)

logger = logging.getLogger(__name__)

REVIEW_MESH_TERMS = ["Review", "Meta-Analysis", "Systematic Review"]


def _map_evidence_type_to_bucket(evidence_type: str) -> str:
    """将服务返回的 evidence_type 映射为 priority bucket"""
    et = (evidence_type or "").strip().lower()
    if "direct_qt" in et or "clinical_qt" in et:
        return EVIDENCE_PRIORITY_1
    if "mechanistic" in et or "herg" in et or "ikr" in et:
        return EVIDENCE_PRIORITY_2
    if "secondary" in et or "pharmacology" in et:
        return EVIDENCE_PRIORITY_3
    return "irrelevant"


def _article_to_pubmed_article(art: Dict[str, Any]) -> PubMedArticle:
    """将服务返回的 article dict 映射为 PubMedArticle"""
    contexts_raw = art.get("contexts") or []
    contexts = [
        EvidenceContext(
            source=c.get("source", ""),
            section=c.get("section", ""),
            matched_term=c.get("matched_term", ""),
            evidence_type=c.get("evidence_type", ""),
            context=c.get("context", ""),
        )
        for c in contexts_raw
    ]

    best_bucket = "irrelevant"
    best_codes: List[str] = []
    if contexts:
        for c in contexts:
            bucket = _map_evidence_type_to_bucket(c.evidence_type)
            rk = priority_rank(bucket)
            if rk > priority_rank(best_bucket):
                best_bucket = bucket
                best_codes = [c.evidence_type]
            elif rk == priority_rank(best_bucket) and c.evidence_type not in best_codes:
                best_codes.append(c.evidence_type)

    title = art.get("title", "") or ""
    abstract = art.get("abstract", "") or ""
    if not contexts:
        mesh_terms = art.get("mesh_terms") or []
        best_bucket, best_codes = classify_literature_text_buckets(title, abstract, mesh_terms)

    is_review = False
    pub_types = art.get("publication_types") or []
    if any("Review" in pt for pt in pub_types):
        is_review = True
    mesh = art.get("mesh_terms") or []
    if not is_review and any(t in REVIEW_MESH_TERMS for t in mesh):
        is_review = True

    year = art.get("publication_year")
    if year is None:
        ji = art.get("journalInfo") or {}
        pd = ji.get("publicationDate") or {}
        y_str = pd.get("year")
        if y_str:
            try:
                year = int(y_str)
            except (ValueError, TypeError):
                pass

    kw_hits = [c.matched_term for c in contexts if c.matched_term]

    return PubMedArticle(
        pmid=str(art.get("pmid") or ""),
        title=title,
        abstract=abstract or None,
        mesh_terms=list(mesh),
        is_review=is_review,
        relevance_keywords_hit=kw_hits,
        molecule_mentioned=art.get("molecule_mentioned", True),
        publication_year=year,
        relevance_bucket=best_bucket if best_bucket != "irrelevant" else None,
        relevance_reason_codes=best_codes,
        contexts=contexts,
        source=art.get("source", ""),
        doi=art.get("doi"),
    )


class LiteratureRetriever:
    """文献检索器 — 通过本地 Pipeline 服务获取文献与 QT/hERG 证据段落"""

    def __init__(self, chembl_client: Optional[ChEMBLClient] = None,
                 pubmed_client: Optional[PubMedClient] = None):
        self.chembl_client = chembl_client or ChEMBLClient()
        self.pubmed_client = pubmed_client or PubMedClient()

    def retrieve(
        self,
        chembl_id: str,
        drug_name: str,
        max_pubmed: int = 30,
        *,
        molecule: Optional[Dict[str, Any]] = None,
        activities: Optional[List[Dict[str, Any]]] = None,
    ) -> LiteratureEvidence:
        """
        检索文献证据

        服务拿到 molecule + activities 后自行组织查询：
        - 有 pref_name/synonyms → 用名字搜 PubMed
        - 有 activities → 用 document_chembl_id 找已知论文全文
        - 两者结合 → 提取 QT/hERG 证据段落

        Args:
            chembl_id: ChEMBL ID
            drug_name: 药物名称（日志用）
            max_pubmed: 返回文章数上限
            molecule: ChEMBL molecule dict（pref_name、molecule_synonyms）
            activities: ChEMBL 活性数据（含 document_chembl_id → 论文入口）
        """
        logger.info("检索文献证据: %s", chembl_id)

        articles, enrichment = self._call_service(
            molecule_chembl_id=chembl_id,
            molecule=molecule,
            activities=activities,
            top_n=max_pubmed,
        )

        pubmed_articles: List[PubMedArticle] = []
        p1 = p2 = p3 = 0
        top_bucket: Optional[str] = None
        top_codes: List[str] = []
        best_rank = 0

        for art in articles:
            pa = _article_to_pubmed_article(art)
            pubmed_articles.append(pa)

            bucket = pa.relevance_bucket
            if bucket == EVIDENCE_PRIORITY_1:
                p1 += 1
            elif bucket == EVIDENCE_PRIORITY_2:
                p2 += 1
            elif bucket == EVIDENCE_PRIORITY_3:
                p3 += 1

            rk = priority_rank(bucket)
            if rk > best_rank:
                best_rank = rk
                top_bucket = bucket
                top_codes = list(pa.relevance_reason_codes)

        le = LiteratureEvidence(
            pubmed_articles=pubmed_articles,
            patents=self._search_patents(drug_name),
            priority_1_article_count=p1,
            priority_2_article_count=p2,
            priority_3_article_count=p3,
            top_relevance_bucket=top_bucket,
            top_relevance_reason_codes=top_codes,
            chembl_enrichment=enrichment,
        )
        logger.info(
            "M3C 文献检索完成 | chembl_id=%s | n=%s | p1=%s p2=%s p3=%s top=%s",
            chembl_id,
            len(pubmed_articles),
            p1,
            p2,
            p3,
            top_bucket,
        )
        return le

    def _call_service(
        self,
        molecule_chembl_id: str,
        molecule: Optional[Dict[str, Any]] = None,
        activities: Optional[List[Dict[str, Any]]] = None,
        top_n: int = 30,
    ) -> tuple:
        """调用本地 Pipeline 服务，返回 (articles, enrichment)"""
        kwargs: Dict[str, Any] = {}
        if molecule and isinstance(molecule, dict):
            kwargs["pref_name"] = (molecule.get("pref_name") or "").strip() or None
            synonyms = molecule.get("molecule_synonyms") or []
            if synonyms:
                kwargs["molecule_synonyms"] = [
                    {"syn_type": s.get("syn_type", ""),
                     "syn_name": s.get("synonyms") or s.get("molecule_synonym") or ""}
                    for s in synonyms
                ]
        if activities:
            kwargs["activities"] = activities

        data = self.pubmed_client.fetch_pipeline_articles(
            molecule_chembl_id=molecule_chembl_id, top_n=top_n, **kwargs,
        )
        if not data:
            return [], None
        return data.get("articles") or [], data.get("chembl_enrichment")

    def _search_patents(self, drug_name: str) -> List[PatentInfo]:
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