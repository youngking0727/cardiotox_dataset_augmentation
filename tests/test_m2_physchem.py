"""单元测试：M2 理化性质计算器"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from m2_physchem import PhysChemCalculator
from schemas import PhysChemProperties, SourceInfo


class TestPhysChemCalculator:
    """测试理化性质计算器"""

    def test_create_calculator(self):
        """测试创建计算器"""
        calc = PhysChemCalculator()
        assert calc is not None

    def test_source_info(self):
        """测试SourceInfo"""
        info = SourceInfo(value=450.5, source="chembl")
        assert info.value == 450.5
        assert info.source == "chembl"

    def test_physchem_properties(self):
        """测试理化性质模型"""
        props = PhysChemProperties(
            mw=SourceInfo(value=450.5, source="chembl"),
            logp=SourceInfo(value=3.2, source="chembl"),
            logd_7_4=SourceInfo(value=2.1, source="calculated"),
            tpsa=SourceInfo(value=75.0, source="chembl"),
            hbd=SourceInfo(value=2, source="chembl"),
            hba=SourceInfo(value=5, source="chembl"),
            rotatable_bonds=SourceInfo(value=6, source="chembl"),
            aromatic_rings=SourceInfo(value=2, source="rdkit"),
            heavy_atoms=SourceInfo(value=32, source="chembl"),
            basic_pka=SourceInfo(value=9.5, source="pkasolver"),
            acidic_pka=None,
            ro5_violations=SourceInfo(value=0, source="chembl"),
            qed=SourceInfo(value=0.75, source="rdkit")
        )

        assert props.mw.value == 450.5
        assert props.logp.source == "chembl"
        assert props.logd_7_4.value == 2.1
        assert props.basic_pka.value == 9.5
        assert props.acidic_pka is None


class TestLogDCalculation:
    """测试logD计算"""

    def test_calculate_logd_no_pka(self):
        """测试无pKa时的logD计算"""
        calc = PhysChemCalculator()

        # 无pKa时，logD = logP
        logd = calc.calculate_logd(logp=3.0, basic_pka=None, acidic_pka=None)
        assert logd == 3.0


class TestSMILESValidation:
    """测试SMILES验证"""

    def test_invalid_smiles(self):
        """测试无效SMILES"""
        try:
            from rdkit import Chem
            calc = PhysChemCalculator()

            # 这是一个无效的SMILES
            mol = Chem.MolFromSmiles("INVALID_SMILES")
            assert mol is None
        except ImportError:
            pytest.skip("RDKit not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])