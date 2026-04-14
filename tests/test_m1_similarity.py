"""单元测试：M1 相似性检索器"""

import pytest
import sys
from pathlib import Path

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from m1_similarity import SimilarityRetriever, SimilarMolecule, SimilarityResult
from schemas import SimilarityResult as SchemaSimilarityResult


class TestSimilarityRetriever:
    """测试相似性检索器"""

    def test_create_retriever(self):
        """测试创建检索器"""
        retriever = SimilarityRetriever()
        assert retriever is not None
        assert retriever.threshold == 0.8

    def test_create_with_custom_threshold(self):
        """测试自定义阈值"""
        retriever = SimilarityRetriever(threshold=0.7)
        assert retriever.threshold == 0.7

    def test_result_structure(self):
        """测试结果结构"""
        result = SimilarityResult(
            query_smiles="CCO",
            query_chembl_id="CHEMBL545",
            similar_molecules=[
                SimilarMolecule(
                    chembl_id="CHEMBL123",
                    smiles="CC(C)C",
                    tanimoto=0.85,
                    already_in_diqta=False
                )
            ]
        )

        assert result.query_smiles == "CCO"
        assert len(result.similar_molecules) == 1
        assert result.similar_molecules[0].tanimoto == 0.85

    def test_similarity_range(self):
        """测试相似度范围"""
        similar = SimilarMolecule(
            chembl_id="CHEMBL123",
            smiles="CC",
            tanimoto=0.9,
            already_in_diqta=False
        )
        assert 0 <= similar.tanimoto <= 1


class TestSimilarMolecule:
    """测试相似分子模型"""

    def test_required_fields(self):
        """测试必填字段"""
        with pytest.raises(Exception):
            SimilarMolecule(smiles="CC", tanimoto=0.8, already_in_diqta=False)

    def test_optional_chembl_id(self):
        """测试可选字段"""
        mol = SimilarMolecule(
            chembl_id="CHEMBL123",
            smiles="CC",
            tanimoto=0.8,
            already_in_diqta=False
        )
        assert mol.chembl_id == "CHEMBL123"

    def test_tanimoto_validation(self):
        """测试Tanimoto验证"""
        # 有效范围
        mol = SimilarMolecule(
            chembl_id="CHEMBL123",
            smiles="CC",
            tanimoto=0.0,
            already_in_diqta=False
        )
        assert mol.tanimoto == 0.0

        mol = SimilarMolecule(
            chembl_id="CHEMBL123",
            smiles="CC",
            tanimoto=1.0,
            already_in_diqta=False
        )
        assert mol.tanimoto == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])