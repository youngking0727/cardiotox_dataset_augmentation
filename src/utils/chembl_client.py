"""ChEMBL API客户端封装"""

import time
import logging
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_AVAILABLE = True
except ImportError:
    CHEMBL_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChEMBLClient:
    """ChEMBL API客户端封装"""

    def __init__(self, cache_dir: Optional[Path] = None, rate_limit: float = 0.5):
        """
        初始化ChEMBL客户端

        Args:
            cache_dir: 缓存目录
            rate_limit: API调用间隔（秒）
        """
        if not CHEMBL_AVAILABLE:
            raise ImportError("chembl_webresource_client未安装，请运行: pip install chembl-webresource-client")

        self.cache_dir = cache_dir
        self.rate_limit = rate_limit
        self._last_call_time = 0

        # 初始化ChEMBL API客户端
        self.molecule = new_client.molecule
        self.activity = new_client.activity
        self.target = new_client.target
        self.mechanism = new_client.mechanism
        self.similarity = new_client.similarity
        self drug = new_client.drug
        self.indication = new_client.indication
        self.atc_classification = new_client.atc_classification

    def _rate_limit_wait(self):
        """速率限制等待"""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_call_time = time.time()

    def get_molecule_by_smiles(self, smiles: str) -> Optional[Dict[str, Any]]:
        """
        通过SMILES获取分子信息

        Args:
            smiles: 分子的SMILES表示

        Returns:
            分子信息字典
        """
        self._rate_limit_wait()
        try:
            result = self.molecule.filter(molecule_structures__canonical_smiles__flexmatch=smiles)
            result = list(result)
            return result[0] if result else None
        except Exception as e:
            logger.warning(f"通过SMILES获取分子失败: {smiles}, 错误: {e}")
            return None

    def get_molecule_by_chembl_id(self, chembl_id: str) -> Optional[Dict[str, Any]]:
        """
        通过ChEMBL ID获取分子信息

        Args:
            chembl_id: ChEMBL ID

        Returns:
            分子信息字典
        """
        self._rate_limit_wait()
        try:
            result = self.molecule.filter(molecule_chembl_id__iexact=chembl_id)
            result = list(result)
            return result[0] if result else None
        except Exception as e:
            logger.warning(f"通过ChEMBL ID获取分子失败: {chembl_id}, 错误: {e}")
            return None

    def get_similar_molecules(self, smiles: str, threshold: float = 0.8,
                              max_results: int = 100) -> List[Dict[str, Any]]:
        """
        相似性检索

        Args:
            smiles: 查询分子的SMILES
            threshold: Tanimoto相似度阈值
            max_results: 最大返回结果数

        Returns:
            相似分子列表
        """
        self._rate_limit_wait()
        try:
            # ChEMBL相似性搜索使用Morgan指纹 (radius=2, nBits=2048)
            result = self.similarity.filter(
                molecule_structures__canonical_smiles__flexmatch=smiles,
                similarity=threshold
            ).order_by('-similarity')

            similar = []
            for i, item in enumerate(result):
                if i >= max_results:
                    break
                similar.append({
                    'chembl_id': item.get('molecule_chembl_id'),
                    'smiles': item.get('molecule_structures', {}).get('canonical_smiles'),
                    'similarity': item.get('similarity') / 100.0  # 转换为0-1范围
                })
            return similar
        except Exception as e:
            logger.warning(f"相似性检索失败: {smiles}, 错误: {e}")
            return []

    def get_activities(self, chembl_id: str,
                      target_chembl_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        获取分子对特定靶点的活性数据

        Args:
            chembl_id: ChEMBL ID
            target_chembl_ids: 靶点ChEMBL ID列表，None表示所有靶点

        Returns:
            活性数据列表
        """
        self._rate_limit_wait()
        try:
            if target_chembl_ids:
                # 批量查询
                all_activities = []
                for target_id in target_chembl_ids:
                    activities = self.activity.filter(
                        molecule_chembl_id__iexact=chembl_id,
                        target_chembl_id__iexact=target_id
                    )
                    all_activities.extend(list(activities))
                return all_activities
            else:
                activities = self.activity.filter(molecule_chembl_id__iexact=chembl_id)
                return list(activities)
        except Exception as e:
            logger.warning(f"获取活性数据失败: {chembl_id}, 错误: {e}")
            return []

    def get_molecule_properties(self, chembl_id: str) -> Optional[Dict[str, Any]]:
        """
        获取分子的理化性质

        Args:
            chembl_id: ChEMBL ID

        Returns:
            分子性质字典
        """
        self._rate_limit_wait()
        try:
            molecule = self.get_molecule_by_chembl_id(chembl_id)
            if molecule:
                return molecule.get('molecule_properties')
            return None
        except Exception as e:
            logger.warning(f"获取分子性质失败: {chembl_id}, 错误: {e}")
            return None

    def get_target_by_chembl_id(self, target_chembl_id: str) -> Optional[Dict[str, Any]]:
        """
        通过ChEMBL ID获取靶点信息

        Args:
            target_chembl_id: 靶点ChEMBL ID

        Returns:
            靶点信息字典
        """
        self._rate_limit_wait()
        try:
            result = self.target.filter(target_chembl_id__iexact=target_chembl_id)
            result = list(result)
            return result[0] if result else None
        except Exception as e:
            logger.warning(f"获取靶点信息失败: {target_chembl_id}, 错误: {e}")
            return None

    def get_mechanisms(self, chembl_id: str) -> List[Dict[str, Any]]:
        """
        获取分子的作用机制

        Args:
            chembl_id: ChEMBL ID

        Returns:
            机制数据列表
        """
        self._rate_limit_wait()
        try:
            mechanisms = self.mechanism.filter(molecule_chembl_id__iexact=chembl_id)
            return list(mechanisms)
        except Exception as e:
            logger.warning(f"获取作用机制失败: {chembl_id}, 错误: {e}")
            return []

    def get_drug_info(self, chembl_id: str) -> Optional[Dict[str, Any]]:
        """
        获取药物信息（包含max_phase等临床信息）

        Args:
            chembl_id: ChEMBL ID

        Returns:
            药物信息字典
        """
        self._rate_limit_wait()
        try:
            drugs = self.drug.filter(molecule_chembl_id__iexact=chembl_id)
            drugs = list(drugs)
            return drugs[0] if drugs else None
        except Exception as e:
            logger.warning(f"获取药物信息失败: {chembl_id}, 错误: {e}")
            return None


def create_chembl_client(cache_dir: Optional[Path] = None) -> ChEMBLClient:
    """创建ChEMBL客户端的工厂函数"""
    return ChEMBLClient(cache_dir=cache_dir)