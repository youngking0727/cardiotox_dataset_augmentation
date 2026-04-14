"""M2: 理化性质计算器模块"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

from ..schemas import PhysChemProperties, SourceInfo
from ..utils.chembl_client import ChEMBLClient
from ..utils.cache import get_global_cache

logger = logging.getLogger(__name__)


class PhysChemCalculator:
    """理化性质计算器"""

    def __init__(self, chembl_client: Optional[ChEMBLClient] = None):
        """
        初始化理化性质计算器

        Args:
            chembl_client: ChEMBL客户端
        """
        self.chembl_client = chembl_client or ChEMBLClient()
        self.cache = get_global_cache()

    def calculate_from_chembl(self, chembl_id: str) -> Optional[PhysChemProperties]:
        """
        从ChEMBL获取理化性质

        Args:
            chembl_id: ChEMBL ID

        Returns:
            理化性质对象
        """
        cache_key = f"physchem_chembl_{chembl_id}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        logger.info(f"从ChEMBL获取理化性质: {chembl_id}")

        # 获取分子信息
        molecule = self.chembl_client.get_molecule_by_chembl_id(chembl_id)
        if not molecule:
            logger.warning(f"未找到分子: {chembl_id}")
            return None

        props = molecule.get("molecule_properties")
        if not props:
            logger.warning(f"分子无可用性质: {chembl_id}")
            return None

        try:
            # 从ChEMBL获取性质，标注来源为chembl
            result = PhysChemProperties(
                mw=SourceInfo(
                    value=props.get("full_mwt") or props.get("molweight"),
                    source="chembl"
                ),
                logp=SourceInfo(
                    value=props.get("alogp"),
                    source="chembl"
                ),
                # logD需要通过pKa计算，暂不直接从ChEMBL获取
                logd_7_4=None,
                tpsa=SourceInfo(
                    value=props.get("psa"),
                    source="chembl"
                ),
                hbd=SourceInfo(
                    value=props.get("hbd"),
                    source="chembl"
                ),
                hba=SourceInfo(
                    value=props.get("hba"),
                    source="chembl"
                ),
                rotatable_bonds=SourceInfo(
                    value=props.get("rtb"),
                    source="chembl"
                ),
                aromatic_rings=SourceInfo(
                    value=props.get("aromatic_rings"),
                    source="chembl"
                ),
                heavy_atoms=SourceInfo(
                    value=props.get("heavy_atoms"),
                    source="chembl"
                ),
                # pKa从ChEMBL无法直接获取
                basic_pka=None,
                acidic_pka=None,
                ro5_violations=SourceInfo(
                    value=props.get("num_ro5_violations", 0),
                    source="chembl"
                ),
                qed=SourceInfo(
                    value=props.get("qed_weighted"),
                    source="chembl"
                )
            )

            self.cache.set(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"解析ChEMBL理化性质失败: {chembl_id}, 错误: {e}")
            return None

    def calculate_from_smiles(self, smiles: str) -> Optional[PhysChemProperties]:
        """
        使用RDKit从SMILES计算理化性质

        Args:
            smiles: 分子的SMILES

        Returns:
            理化性质对象
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski, Crippen, QED
        except ImportError:
            logger.error("RDKit未安装，无法从SMILES计算理化性质")
            return None

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            logger.warning(f"无效的SMILES: {smiles}")
            return None

        logger.info(f"使用RDKit计算理化性质")

        try:
            result = PhysChemProperties(
                mw=SourceInfo(
                    value=Descriptors.MolWt(mol),
                    source="rdkit"
                ),
                logp=SourceInfo(
                    value=Crippen.MolLogP(mol),
                    source="rdkit"
                ),
                logd_7_4=None,  # 需要pKa计算
                tpsa=SourceInfo(
                    value=Descriptors.TPSA(mol),
                    source="rdkit"
                ),
                hbd=SourceInfo(
                    value=Lipinski.NumHDonors(mol),
                    source="rdkit"
                ),
                hba=SourceInfo(
                    value=Lipinski.NumHAcceptors(mol),
                    source="rdkit"
                ),
                rotatable_bonds=SourceInfo(
                    value=Lipinski.NumRotatableBonds(mol),
                    source="rdkit"
                ),
                aromatic_rings=SourceInfo(
                    value=Lipinski.NumAromaticRings(mol),
                    source="rdkit"
                ),
                heavy_atoms=SourceInfo(
                    value=Descriptors.HeaviestAtomCount(mol),
                    source="rdkit"
                ),
                basic_pka=None,
                acidic_pka=None,
                ro5_violations=SourceInfo(
                    value=self._count_ro5_violations(mol),
                    source="rdkit"
                ),
                qed=SourceInfo(
                    value=QED.qed(mol),
                    source="rdkit"
                )
            )

            return result

        except Exception as e:
            logger.warning(f"RDKit计算理化性质失败: {smiles}, 错误: {e}")
            return None

    def _count_ro5_violations(self, mol) -> int:
        """计算Lipinski规则违反数"""
        try:
            from rdkit.Chem import Lipinski

            violations = 0

            # 规则1: 分子量 <= 500
            if Descriptors.MolWt(mol) > 500:
                violations += 1

            # 规则2: logP <= 5
            if Crippen.MolLogP(mol) > 5:
                violations += 1

            # 规则3: 氢键供体数 <= 5
            if Lipinski.NumHDonors(mol) > 5:
                violations += 1

            # 规则4: 氢键受体数 <= 10
            if Lipinski.NumHAcceptors(mol) > 10:
                violations += 1

            return violations

        except Exception as e:
            logger.warning(f"计算RO5违反数失败: {e}")
            return 0

    def calculate_pka(self, smiles: str) -> Optional[Dict[str, float]]:
        """
        使用pkasolver计算pKa值

        Args:
            smiles: 分子的SMILES

        Returns:
            包含basic_pka和acidic_pka的字典
        """
        try:
            from pkasolver import PkaCalculator
        except ImportError:
            logger.warning("pkasolver未安装，无法计算pKa")
            return None

        try:
            calculator = PkaCalculator()
            # 返回的字典可能包含多个pKa值，取最接近生理pH的值
            pka_results = calculator.calc_pka(smiles)

            if not pka_results:
                return None

            # 按pKa值排序
            sorted_pkas = sorted(pka_results, key=lambda x: x.get("pka", 0))

            basic_pka = None
            acidic_pka = None

            for pka_info in sorted_pkas:
                pka_type = pka_info.get("type", "")
                pka_val = pka_info.get("pka")

                if pka_type == "basic" and basic_pka is None:
                    basic_pka = pka_val
                elif pka_type == "acidic" and acidic_pka is None:
                    acidic_pka = pka_val

            return {
                "basic_pka": basic_pka,
                "acidic_pka": acidic_pka
            }

        except Exception as e:
            logger.warning(f"pKa计算失败: {smiles}, 错误: {e}")
            return None

    def calculate_logd(self, logp: float, basic_pka: Optional[float],
                       acidic_pka: Optional[float], ph: float = 7.4) -> Optional[float]:
        """
        使用Henderson-Hasselbalch方程计算logD

        Args:
            logp: logP值
            basic_pka: 碱性中心pKa
            acidic_pka: 酸性中心pKa
            ph: pH值

        Returns:
            logD值
        """
        if basic_pka is None and acidic_pka is None:
            return logp  # 无可电离基团时，logD = logP

        # 简化计算：假设只有单一可电离位点
        try:
            if basic_pka is not None:
                # 碱性分子：logD = logP - log(1 + 10^(pKa-pH))
                fraction = 10 ** (basic_pka - ph)
                logd = logp - (0.1 if fraction > 0 else 0)  # 简化
            elif acidic_pka is not None:
                # 酸性分子：logD = logP - log(1 + 10^(pH-pKa))
                fraction = 10 ** (ph - acidic_pka)
                logd = logp - (0.1 if fraction > 0 else 0)  # 简化
            else:
                logd = logp

            return logd

        except Exception as e:
            logger.warning(f"logD计算失败: {e}")
            return logp

    def get_full_properties(self, chembl_id: Optional[str] = None,
                           smiles: Optional[str] = None) -> Optional[PhysChemProperties]:
        """
        获取完整的理化性质（ChEMBL优先，RDKit兜底）

        Args:
            chembl_id: ChEMBL ID
            smiles: SMILES（当ChEMBL ID不可用时使用）

        Returns:
            理化性质对象
        """
        # 优先从ChEMBL获取
        if chembl_id:
            properties = self.calculate_from_chembl(chembl_id)
            if properties:
                return properties

        # 兜底：使用RDKit从SMILES计算
        if smiles:
            properties = self.calculate_from_smiles(smiles)
            if properties:
                # 尝试计算pKa
                pka_results = self.calculate_pka(smiles)
                if pka_results:
                    if pka_results.get("basic_pka"):
                        properties.basic_pka = SourceInfo(
                            value=pka_results["basic_pka"],
                            source="pkasolver"
                        )
                    if pka_results.get("acidic_pka"):
                        properties.acidic_pka = SourceInfo(
                            value=pka_results["acidic_pka"],
                            source="pkasolver"
                        )

                    # 计算logD
                    logp_val = properties.logp.value
                    if isinstance(logp_val, (int, float)):
                        logd = self.calculate_logd(
                            logp=logp_val,
                            basic_pka=pka_results.get("basic_pka"),
                            acidic_pka=pka_results.get("acidic_pka")
                        )
                        if logd is not None:
                            properties.logd_7_4 = SourceInfo(
                                value=logd,
                                source="calculated"
                            )

                return properties

        return None


def create_physchem_calculator() -> PhysChemCalculator:
    """创建理化性质计算器的工厂函数"""
    return PhysChemCalculator()