"""ClinicalTrials.gov API客户端封装"""

import time
import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests

logger = logging.getLogger(__name__)


class ClinicalTrialsClient:
    """ClinicalTrials.gov API v2客户端"""

    API_BASE_URL = "https://clinicaltrials.gov/api/v2"

    # 心脏毒性相关关键词
    QT_KEYWORDS = [
        "QT prolongation", "QT interval", "torsades", "torsade",
        "arrhythmia", "arrhythmia", "cardiac arrhythmia", "ventricular arrhythmia",
        "sudden cardiac death", "cardiac death"
    ]

    def __init__(self, cache_dir: Optional[Path] = None, rate_limit: float = 0.5):
        """
        初始化ClinicalTrials客户端

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

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """发送API请求"""
        self._rate_limit_wait()
        url = f"{self.API_BASE_URL}/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"ClinicalTrials API请求失败: {url}, 错误: {e}")
            return None

    def search_studies(self, query: str, max_results: int = 50,
                     fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        搜索临床试验

        Args:
            query: 搜索查询
            max_results: 最大结果数
            fields: 要返回的字段列表

        Returns:
            试验详情列表
        """
        if fields is None:
            fields = ["protocolSection.identificationModule",
                     "protocolSection.statusModule",
                     "protocolSection.descriptionModule",
                     "protocolSection.conditionsModule"]

        params = {
            "query.term": query,
            "pageSize": max_results,
            "fields": ",".join(fields)
        }

        data = self._make_request("studies", params)
        if not data:
            return []

        studies = data.get("studies", [])
        return [s.get("protocolSection", {}) for s in studies]

    def get_study_by_nct_id(self, nct_id: str) -> Optional[Dict[str, Any]]:
        """
        通过NCT ID获取试验详情

        Args:
            nct_id: NCT ID

        Returns:
            试验详情
        """
        data = self._make_request(f"studies/{nct_id}")
        return data

    def search_qt_related_trials(self, drug_name: str,
                                 max_results: int = 30) -> List[Dict[str, Any]]:
        """
        搜索与QT延长相关的临床试验

        Args:
            drug_name: 药物名称
            max_results: 最大结果数

        Returns:
            相关试验列表
        """
        # 构建查询: 药物名称 AND (QT延长 OR 心律失常 OR 心脏毒性)
        qt_query = " OR ".join([f'"{kw}"' for kw in self.QT_KEYWORDS])
        query = f'{drug_name} AND ({qt_query})'

        studies = self.search_studies(query, max_results=max_results)

        # 解析试验信息
        trials = []
        for study in studies:
            identification = study.get("identificationModule", {})
            status = study.get("statusModule", {})
            description = study.get("descriptionModule", {})

            nct_id = identification.get("nctId", "")
            title = identification.get("briefTitle", "")

            # 简略描述
            summary = ""
            if description and "briefSummary" in description:
                summary = description.get("briefSummary", "")

            # 试验状态
            status_str = status.get("overallStatus", "Unknown")

            # 检查是否与QT相关（通过关键词匹配）
            text = (title + " " + summary).lower()
            qt_related = any(kw.lower() in text for kw in self.QT_KEYWORDS)

            if qt_related:
                trials.append({
                    "nct_id": nct_id,
                    "title": title,
                    "status": status_str,
                    "qt_related": True,
                    "summary": summary[:500] if summary else ""  # 限制摘要长度
                })

        return trials


def create_clinicaltrials_client(cache_dir: Optional[Path] = None) -> ClinicalTrialsClient:
    """创建ClinicalTrials客户端的工厂函数"""
    return ClinicalTrialsClient(cache_dir=cache_dir)