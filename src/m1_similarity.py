"""M1: 相似性检索器模块"""

import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from datetime import datetime

from ..schemas import SimilarityResult, SimilarMolecule
from ..utils.chembl_client import ChEMBLClient
from ..utils.cache import get_global_cache

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

        # 优先使用ChEMBL ID进行检索（更准确）
        if query_chembl_id:
            molecule_info = self.chembl_client.get_molecule_by_chembl_id(query_chembl_id)
            if molecule_info:
                smiles = molecule_info.get("molecule_structures", {}).get("canonical_smiles")
                if smiles:
                    query_smiles = smiles

        # 从ChEMBL获取相似分子
        similar_data = self.chembl_client.get_similar_molecules(
            smiles=query_smiles,
            threshold=self.threshold,
            max_results=self.max_results
        )

        # 处理结果
        similar_molecules = []
        for item in similar_data:
            chembl_id = item.get("chembl_id")
            smiles = item.get("smiles")

            if not chembl_id or not smiles:
                continue

            # 检查是否已存在于DIQTA中
            already_in_diqta = False
            if diqta_smiles:
                # 简单的SMILES比较（实际可能需要标准化）
                already_in_diqta = smiles in diqta_smiles

            similar_molecules.append(SimilarMolecule(
                chembl_id=chembl_id,
                smiles=smiles,
                tanimoto=item.get("similarity", 0),
                already_in_diqta=already_in_diqta
            ))

        logger.info(f"相似性检索完成，找到 {len(similar_molecules)} 个相似分子")

        return SimilarityResult(
            query_smiles=query_smiles,
            query_chembl_id=query_chembl_id,
            similar_molecules=similar_molecules
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
                results.append(SimilarityResult(
                    query_smiles=smiles,
                    query_chembl_id=chembl_id,
                    similar_molecules=[]
                ))

        return results


def create_similarity_retriever(threshold: float = 0.8,
                                max_results: int = 100) -> SimilarityRetriever:
    """创建相似性检索器的工厂函数"""
    return SimilarityRetriever(
        threshold=threshold,
        max_results=max_results
    )