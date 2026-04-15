"""M5: 证据充分性判定器模块

第一层（充分性）：由 EvidenceDensity.total_score 映射 sufficiency_tier，
只回答「证据是否够继续推理/入库」，与标签结论解耦。

第二层（标签）：judge_label / judge_label_detailed 给出 torsadogenic 等提示；
high 置信度需多路独立信号交叉，单条 hERG 或单条撤市叙述不会直接给 high，
避免把「强机制风险」与「最终临床致 TdP 结论」混为一谈。
"""

import logging
from typing import Optional, Dict, Any, Tuple, List

from schemas import (
    EvidenceDensity,
    ClinicalEvidence,
    LiteratureEvidence,
    TargetBioactivity,
    LabelJudgmentDetail,
    EvidenceSufficiencyTier,
)
from m3b_clinical import ClinicalStatusRetriever
from m3c_literature import LiteratureRetriever

logger = logging.getLogger(__name__)


def sufficiency_tier_from_total_score(total_score: int) -> EvidenceSufficiencyTier:
    """0–1：不足以支撑下游；2–3：可用；4+：多维度证据较强。"""
    if total_score <= 1:
        return "insufficient"
    if total_score <= 3:
        return "usable"
    return "strong"


class EvidenceSufficiencyJudge:
    """证据充分性判定器"""

    # hERG IC50 阈值（μM）：低于此视为「机制层面」提示，单独出现时置信度偏低
    HERG_IC50_THRESHOLD_TORSADOGENIC = 10.0
    HERG_IC50_THRESHOLD_STRONG = 1.0

    def __init__(self, clinical_retriever: Optional[ClinicalStatusRetriever] = None,
                 literature_retriever: Optional[LiteratureRetriever] = None):
        """
        初始化证据充分性判定器

        Args:
            clinical_retriever: 临床状态检索器
            literature_retriever: 文献检索器
        """
        # 默认不构造 ClinicalStatusRetriever / LiteratureRetriever，避免仅用于 M5 判定时
        # 意外实例化 PubMedClient 并打印 NCBI_EMAIL 警告（见 compute_evidence_density 计数逻辑）。
        self.clinical_retriever = clinical_retriever
        self.literature_retriever = literature_retriever

    def compute_evidence_density(self,
                                 bioactivity_evidence: Dict[str, TargetBioactivity],
                                 clinical_evidence: ClinicalEvidence,
                                 literature_evidence: LiteratureEvidence) -> EvidenceDensity:
        """
        计算证据密度指标

        Args:
            bioactivity_evidence: 生物活性证据
            clinical_evidence: 临床证据
            literature_evidence: 文献证据

        Returns:
            证据密度对象
        """
        # 是否有hERG测量
        has_herg_measurement = "hERG" in bioactivity_evidence and bool(
            bioactivity_evidence["hERG"].measurements
        )

        # 是否有临床阶段信息
        has_clinical_phase_info = clinical_evidence.max_phase is not None

        # 是否有撤市信息
        has_withdrawal_info = clinical_evidence.withdrawn.flag

        # 外部数据库命中数
        external_db_hit_count = sum(
            1 for db_flag in clinical_evidence.external_db_flags.values()
            if db_flag.present
        )

        # PubMed文章数
        pubmed_article_count = len(literature_evidence.pubmed_articles)

        # 心脏毒性相关PubMed文章数（无网络；优先用注入的 retriever，否则静态计数）
        if self.literature_retriever is not None:
            pubmed_cardiotox_relevant_count = self.literature_retriever.get_cardiotox_relevant_count(
                literature_evidence.pubmed_articles
            )
        else:
            pubmed_cardiotox_relevant_count = LiteratureRetriever.count_cardiotox_relevant(
                literature_evidence.pubmed_articles
            )

        # 专利数
        patent_count = len(literature_evidence.patents)

        # 总分（简单加权）
        total_score = sum([
            has_herg_measurement,
            has_clinical_phase_info,
            has_withdrawal_info,
            external_db_hit_count > 0,
            pubmed_cardiotox_relevant_count >= 3,
            patent_count > 0
        ])
        tier = sufficiency_tier_from_total_score(total_score)

        return EvidenceDensity(
            has_herg_measurement=has_herg_measurement,
            has_clinical_phase_info=has_clinical_phase_info,
            has_withdrawal_info=has_withdrawal_info,
            external_db_hit_count=external_db_hit_count,
            pubmed_article_count=pubmed_article_count,
            pubmed_cardiotox_relevant_count=pubmed_cardiotox_relevant_count,
            patent_count=patent_count,
            total_score=total_score,
            sufficiency_tier=tier,
        )

    def judge_label_detailed(
        self,
        bioactivity_evidence: Dict[str, TargetBioactivity],
        clinical_evidence: ClinicalEvidence,
    ) -> LabelJudgmentDetail:
        """
        第二层：标签判定（与充分性分层独立）。

        - **high**：至少两条独立信号同时成立（如撤市 QT 叙述 + hERG/外部库高风险等）。
        - **medium**：单条临床/监管叙述（如撤市 QT）或单条外部库高风险标注。
        - **low**：仅 hERG 机制提示（IC50 低于阈值），不等同于临床 TdP 标签。
        """
        codes: List[str] = []

        withdrawal_qt = False
        if clinical_evidence.withdrawn.flag:
            reason = (clinical_evidence.withdrawn.reason or "").lower()
            qt_keywords = ["qt", "torsades", "torsade", "arrhythmia"]
            if any(kw in reason for kw in qt_keywords):
                withdrawal_qt = True
                codes.append("withdrawal_qt_narrative")

        herg_mechanistic = False
        herg_data = bioactivity_evidence.get("hERG")
        if herg_data and herg_data.measurements:
            best_ic50 = herg_data.measurements[0].value
            if best_ic50 < self.HERG_IC50_THRESHOLD_TORSADOGENIC:
                herg_mechanistic = True
                codes.append("herg_ic50_lt_10um_mechanistic")

        cardiotox_flag = clinical_evidence.external_db_flags.get("cardiotox")
        cardiotox_high = False
        cardiotox_lowish = False
        if cardiotox_flag and cardiotox_flag.present:
            rl = (cardiotox_flag.risk_level or "").lower()
            if "high" in rl:
                cardiotox_high = True
                codes.append("external_cardiotox_high")
            elif "medium" in rl or "low" in rl:
                cardiotox_lowish = True
                codes.append("external_cardiotox_low_or_medium")

        bbw = clinical_evidence.black_box_warning is True
        if bbw:
            codes.append("black_box_warning")

        # 外部库「低风险」且无任何撤市/hERG 机制提示 → 非致 TdP 倾向（仍非金标准）
        if (
            cardiotox_lowish
            and not cardiotox_high
            and not withdrawal_qt
            and not herg_mechanistic
        ):
            return LabelJudgmentDetail(
                label="non-torsadogenic",
                confidence="medium",
                rationale_codes=codes,
                notes="外部库中低风险提示；未与机制/撤市信号冲突。非最终临床结论。",
            )

        # 独立信号计数（用于是否上调 high）
        independent_signals = sum(
            [
                withdrawal_qt,
                herg_mechanistic,
                cardiotox_high,
                bbw,
            ]
        )

        if not (withdrawal_qt or herg_mechanistic or cardiotox_high):
            return LabelJudgmentDetail(
                label=None,
                confidence=None,
                rationale_codes=codes,
                notes="未命中 torsadogenic / non-torsadogenic 规则（undetermined）。",
            )

        label = "torsadogenic"
        if independent_signals >= 2:
            conf = "high"
            notes = (
                "多路独立信号交叉（撤市叙述 / hERG 机制 / 外部库高风险 / 黑框等至少两项）。"
                "仍须与真实世界标签与监管资料核对。"
            )
        elif withdrawal_qt:
            conf = "medium"
            notes = (
                "存在撤市或 QT/心律失常相关叙述；单路信号不上调为 high。"
            )
        elif cardiotox_high:
            conf = "medium"
            notes = "外部库高风险标注；建议与临床与机制证据交叉验证。单路不上调为 high。"
        elif herg_mechanistic:
            conf = "low"
            notes = (
                "主要反映 hERG/机制风险，不等同于临床 TdP 或监管最终标签。"
            )
        else:
            conf = "medium"
            notes = "规则命中；置信度未上调。"

        return LabelJudgmentDetail(
            label=label,
            confidence=conf,
            rationale_codes=codes,
            notes=notes,
        )

    def judge_label(
        self,
        bioactivity_evidence: Dict[str, TargetBioactivity],
        clinical_evidence: ClinicalEvidence,
    ) -> Tuple[Optional[str], Optional[str]]:
        """兼容入口：仅返回 (label, confidence)。完整说明请用 judge_label_detailed。"""
        d = self.judge_label_detailed(bioactivity_evidence, clinical_evidence)
        return d.label, d.confidence


def create_evidence_sufficiency_judge() -> EvidenceSufficiencyJudge:
    """创建证据充分性判定器的工厂函数"""
    return EvidenceSufficiencyJudge()