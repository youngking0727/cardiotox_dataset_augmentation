"""M1: 相似性检索器模块"""

import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from datetime import datetime

from schemas import SimilarityResult, SimilarMolecule
from utils.chembl_client import ChEMBLClient
from utils.cache import get_global_cache

logger = logging.getLogger(__name__)


class SimilarityRetriever:
    """相似性检索器"""

    DEFAULT_THRESHOLD = 0.8
    DEFAULT_MAX_RESULTS = 100

    def __init__(self, chembl_client: Optional[ChEMBLClient] = None,
                 threshold: float = DEFAULT_THRESHOLD,
                 max_results: int = DEFAULT_MAX_RESULTS):
        """
        初始化相似性检索器

        Args:
            chembl_client: ChEMBL客户端
            threshold: Tanimoto相似度阈值
            max_results: 最大返回结果数
        """
        self.chembl_client = chembl_client or ChEMBLClient()
        self.threshold = threshold
        self.max_results = max_results
        self.cache = get_global_cache()

    def retrieve(self, query_smiles: str, query_chembl_id: Optional[str] = None,
                diqta_smiles: Optional[Set[str]] = None) -> SimilarityResult:
        """
        检索与给定分子相似的分子

        Args:
            query_smiles: 查询分子的SMILES
            query_chembl_id: 查询分子的ChEMBL ID（可选）
            diqta_smiles: DIQTA中已有的SMILES集合，用于标记已存在的分子

        Returns:
            相似性检索结果
        """
        logger.info(f"开始相似性检索: {query_smiles[:50]}...")

        if query_chembl_id:
            molecule_info, _mol_ok = self.chembl_client.get_molecule_by_chembl_id(
                query_chembl_id
            )
            if isinstance(molecule_info, dict):
                smiles = molecule_info.get("molecule_structures", {}).get(
                    "canonical_smiles"
                )
                if smiles:
                    query_smiles = smiles

        similar_data, sim_ok = self.chembl_client.get_similar_molecules(
            smiles=query_smiles,
            threshold=self.threshold,
            max_results=self.max_results,
        )

        similar_molecules = []
        for item in similar_data:
            chembl_id = item.get("chembl_id")
            smiles = item.get("smiles")

            if not chembl_id or not smiles:
                continue

            already_in_diqta = False
            if diqta_smiles:
                already_in_diqta = smiles in diqta_smiles

            raw_sim = item.get("similarity", 0)
            try:
                tanimoto = float(raw_sim) if raw_sim is not None else 0.0
            except (TypeError, ValueError):
                tanimoto = 0.0
            if tanimoto > 1.0:
                tanimoto = tanimoto / 100.0
            tanimoto = max(0.0, min(1.0, tanimoto))

            similar_molecules.append(
                SimilarMolecule(
                    chembl_id=chembl_id,
                    smiles=smiles,
                    tanimoto=tanimoto,
                    already_in_diqta=already_in_diqta,
                )
            )

        logger.info(f"相似性检索完成，找到 {len(similar_molecules)} 个相似分子")

        return SimilarityResult(
            query_smiles=query_smiles,
            query_chembl_id=query_chembl_id,
            similar_molecules=similar_molecules,
            query_similarity_ok=sim_ok,
        )

    def batch_retrieve(self, queries: List[Dict[str, str]],
                      diqta_smiles: Optional[Set[str]] = None) -> List[SimilarityResult]:
        """
        批量检索多个分子的相似分子

        Args:
            queries: 查询列表，每项包含smiles和可选的chembl_id
            diqta_smiles: DIQTA中已有的SMILES集合

        Returns:
            相似性检索结果列表
        """
        results = []
        for query in queries:
            smiles = query.get("smiles")
            chembl_id = query.get("chembl_id")

            if not smiles:
                logger.warning(f"跳过无效查询: {query}")
                continue

            try:
                result = self.retrieve(
                    query_smiles=smiles,
                    query_chembl_id=chembl_id,
                    diqta_smiles=diqta_smiles
                )
                results.append(result)
            except Exception as e:
                logger.error(f"相似性检索失败: {smiles}, 错误: {e}")
                # 仍然添加空结果以保持索引对应
                results.append(
                    SimilarityResult(
                        query_smiles=smiles,
                        query_chembl_id=chembl_id,
                        similar_molecules=[],
                        query_similarity_ok=False,
                    )
                )

        return results


def create_similarity_retriever(threshold: float = 0.8,
                                max_results: int = 100) -> SimilarityRetriever:
    """创建相似性检索器的工厂函数"""
    return SimilarityRetriever(
        threshold=threshold,
        max_results=max_results
    )

    def batch_retrieve(self, queries: List[Dict[str, str]],
                      diqta_smiles: Optional[Set[str]] = None) -> List[SimilarityResult]:
        """
        批量检索多个分子的相似分子

        Args:
            queries: 查询列表，每项包含smiles和可选的chembl_id
            diqta_smiles: DIQTA中已有的SMILES集合

        Returns:
            相似性检索结果列表
        """
        results = []
        for query in queries:
            smiles = query.get("smiles")
            chembl_id = query.get("chembl_id")

            if not smiles:
                logger.warning(f"跳过无效查询: {query}")
                continue

            try:
                result = self.retrieve(
                    query_smiles=smiles,
                    query_chembl_id=chembl_id,
                    diqta_smiles=diqta_smiles
                )
                results.append(result)
            except Exception as e:
                logger.error(f"相似性检索失败: {smiles}, 错误: {e}")
                # 仍然添加空结果以保持索引对应
                results.append(
                    SimilarityResult(
                        query_smiles=smiles,
                        query_chembl_id=chembl_id,
                        similar_molecules=[],
                        query_similarity_ok=False,
                    )
                )

        return results


def create_similarity_retriever(threshold: float = 0.8,
                                max_results: int = 100) -> SimilarityRetriever:
    """创建相似性检索器的工厂函数"""
    return SimilarityRetriever(
        threshold=threshold,
        max_results=max_results
    )

    def batch_retrieve(self, queries: List[Dict[str, str]],
                      diqta_smiles: Optional[Set[str]] = None) -> List[SimilarityResult]:
        """
        批量检索多个分子的相似分子

        Args:
            queries: 查询列表，每项包含smiles和可选的chembl_id
            diqta_smiles: DIQTA中已有的SMILES集合

        Returns:
            相似性检索结果列表
        """
        results = []
        for query in queries:
            smiles = query.get("smiles")
            chembl_id = query.get("chembl_id")

            if not smiles:
                logger.warning(f"跳过无效查询: {query}")
                continue

            try:
                result = self.retrieve(
                    query_smiles=smiles,
                    query_chembl_id=chembl_id,
                    diqta_smiles=diqta_smiles
                )
                results.append(result)
            except Exception as e:
                logger.error(f"相似性检索失败: {smiles}, 错误: {e}")
                # 仍然添加空结果以保持索引对应
                results.append(
                    SimilarityResult(
                        query_smiles=smiles,
                        query_chembl_id=chembl_id,
                        similar_molecules=[],
                        query_similarity_ok=False,
                    )
                )

        return results


def create_similarity_retriever(threshold: float = 0.8,
                                max_results: int = 100) -> SimilarityRetriever:
    """创建相似性检索器的工厂函数"""
    return SimilarityRetriever(
        threshold=threshold,
        max_results=max_results
    )


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ap = argparse.ArgumentParser(
        description="调试 SimilarityRetriever（需 PYTHONPATH 含项目根与 src）"
    )
    ap.add_argument(
        "--smiles",
        default="CC(=O)Oc1ccccc1C(=O)O",
        help="查询 SMILES；若提供 --chembl-id 会先用 ChEMBL canonical 覆盖",
    )
    ap.add_argument("--chembl-id", default="CHEMBL25")
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--max", type=int, default=8, dest="max_results")
    ns = ap.parse_args()
    cid = (ns.chembl_id or "").strip() or None
    print(
        "正在请求 ChEMBL（ molecule + similarity ），请稍候…",
        file=sys.stderr,
        flush=True,
    )
    r = SimilarityRetriever(threshold=ns.threshold, max_results=ns.max_results)
    out = r.retrieve(ns.smiles, query_chembl_id=cid, diqta_smiles=set())
    print(
        f"query_similarity_ok={out.query_similarity_ok} count={len(out.similar_molecules)}",
        file=sys.stderr,
        flush=True,
    )
    for i, mol in enumerate(out.similar_molecules, 1):
        smi = (mol.smiles or "")[:72]
        print(f"{i}\t{mol.chembl_id}\t{mol.tanimoto:.4f}\t{smi}", flush=True)