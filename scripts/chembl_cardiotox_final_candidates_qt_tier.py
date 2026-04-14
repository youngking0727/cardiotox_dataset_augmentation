"""
对 chembl_cardiotox_screen__final_candidates.csv 做「QT / 心脏安全证据」分层，避免把整表
都当成心脏毒性（DIQT）证据。

分层（evidence_qt_tier）：
  - qt_primary                      明确 QT / proarrhythmic / hERG 安全评估语境，可优先保留
  - qt_auxiliary_mechanism          hERG/IKr/QT 语境下的 IC50、Ki、Inhibition、kon/k_off 等机制证据
  - electrophysiology_downweight    抗心律失常/复极/不应期等电生理，不宜直接当 DIQT 毒性证据
  - exclude_non_cardiac             明显非心脏（抗菌/耐药/HIV/激酶组学等），或 FC/AC50 等在无心脏语境下排除
  - review_mixed                    其余需人工判断

用法（在 cardiotoxicity_prediction 根目录）:

  python -m data_augmentation.Chembl_data.chembl_cardiotox_final_candidates_qt_tier ^
    --input data_augmentation/Chembl_data/output/full_enrichment3/chembl_cardiotox_screen__final_candidates.csv ^
    --out-dir data_augmentation/Chembl_data/output/full_enrichment3_qt_tier

默认写出：带 evidence_qt_tier / evidence_qt_reason 列的完整表，并按层拆成多个 CSV + summary.json。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

# --- standard_type 归一化 ---
def _norm_st(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip().lower()


def _blob(row: pd.Series) -> str:
    parts = [
        row.get("doc_title", ""),
        row.get("asy_description", ""),
        row.get("tgt_pref_name", ""),
        row.get("match_text_blob", ""),
    ]
    return " ".join(str(p) for p in parts if p is not None and not (isinstance(p, float) and pd.isna(p))).lower()


# 直接归为「QT/安全」优先型的 type（标题/全文未命中明显排除词时）
PRIMARY_DIRECT_TYPES: frozenset[str] = frozenset(
    _norm_st(x)
    for x in (
        "FPD measurement",
        "FPDc",
        "APDc",
    )
)

# 抗心律失常/复极语境居多，默认降权（除非整段文本已是明确 QT/致心律失常风险叙事）
ELECTROPHYS_DOWNWEIGHT_TYPES: frozenset[str] = frozenset(
    _norm_st(x)
    for x in (
        "APD",
        "APD50",
        "APD60",
        "APD90",
        "Delta APD90",
        "AERP",
        "VERP",
        "ERP",
        "ERP observations",
        "Delta ERP",
        "Delta ERPc",
        "VCT",
        "VFT",
        "HR",
        "Max delta HR",
        "Delta HR",
        "MBP",
        "Cardiac depression",
    )
)

FC_TYPES: frozenset[str] = frozenset(_norm_st(x) for x in ("FC", "Max delta FC", "Delta FC"))

# 在「QT/hERG/IKr」语境下可作为辅助机制证据的活性类型
MECHANISM_TYPES: frozenset[str] = frozenset(
    _norm_st(x)
    for x in (
        "IC50",
        "Ki",
        "Kd",
        "pIC50",
        "Inhibition",
        "Inhibition index",
        "Potency",
        "kon",
        "k_on",
        "k_off",
        "Koff",
        "koff",
        "Kif",
        "Ratio Ki",
        "Binding energy",
    )
)

NOISY_POTENCY_TYPES: frozenset[str] = frozenset(
    _norm_st(x) for x in ("AC50", "Ac50", "Activity")
)

# 明显非心脏、不宜当毒性证据的标题/文本针（子串，小写 blob）
EXCLUDE_BLOB_NEEDLES: tuple[str, ...] = (
    "hiv-1",
    " hiv ",
    "antiretroviral",
    "reverse transcriptase",
    "hepatitis c",
    "hcv ",
    "malaria",
    "plasmodium",
    "antimicrobial",
    "antibiotic",
    "antibacterial",
    "antifungal",
    "candida albicans",
    "candidiasis",
    "fungal ",
    "bacterial resistance",
    "antibiotic resistance",
    "ciprofloxacin",
    "tigecycline",
    "resistome",
    "efflux pump",
    "efflux-mediated",
    "virulence factor",
    "virulence factors",
    "susceptibility testing",
    "minimum inhibitory",
    "kinase enrichment proteomics",
    "competitive kinase enrichment",
    "abemaciclib",
    "gsk3β",
    "gsk3b",
    "wnt signaling",
    "secondary pharmacology",  # 常与 AC50 面板相关，非单通道心脏毒性叙事
)

# 明确 QT / 心脏安全 / 致心律失常风险评估语境
QT_PRIMARY_NEEDLES: tuple[str, ...] = (
    "long qt",
    "lqts",
    "qt prolongation",
    "qt interval",
    "qtc",
    "torsade",
    "torsadogenic",
    "proarrhythmic",
    "thorough qt",
    " tqt",
    "cardiomyocyte",
    "stem-cell derived cardiomyocyte",
    "stem cell derived cardiomyocyte",
    "ips-derived cardiomyocyte",
    "ion channel screen",
    "multichannel block",
    "action potential simulation",
    "clinical torsadogenic",
    "drug-induced qt",
    "diqt",
    "qt safety",
    "cardiac safety",
    "herg",
    "h erg",
    "kv11.1",
    "kcnh2",
    "ikr",
    "long-qt syndrome",
    "long qt syndrome",
)

# hERG / IKr 通道机制（标题无 QT 叙事时，IC50 等仍可作 auxiliary）
HERG_MECHANISM_NEEDLES: tuple[str, ...] = (
    "herg",
    "kv11.1",
    "kcnh2",
    "ikr",
    "long qt syndrome",
    "qt prolongation",
)

# 抗心律失常药效语境 → 与「毒性/DIQT」区分，对 APD/ERP 等降权
ANTIARRHYTHMIC_CONTEXT_NEEDLES: tuple[str, ...] = (
    "antiarrhythmic",
    "class i antiarrhythm",
    "class ii antiarrhythm",
    "class iii antiarrhythm",
    "class iv antiarrhythm",
    "novel antiarrhythmic",
    "selective class iii",
)


def _any_needle(blob: str, needles: tuple[str, ...]) -> bool:
    b = blob.lower()
    return any(n in b for n in needles)


def _exclude_blob(blob: str) -> bool:
    return _any_needle(blob, EXCLUDE_BLOB_NEEDLES)


def _qt_primary_blob(blob: str) -> bool:
    return _any_needle(blob, QT_PRIMARY_NEEDLES)


def _herg_mechanism_blob(blob: str) -> bool:
    return _any_needle(blob, HERG_MECHANISM_NEEDLES)


def _antiarrhythmic_blob(blob: str) -> bool:
    return _any_needle(blob, ANTIARRHYTHMIC_CONTEXT_NEEDLES)


def classify_evidence_qt_tier(row: pd.Series) -> tuple[str, str]:
    """
    返回 (evidence_qt_tier, evidence_qt_reason)
    """
    blob = _blob(row)
    st = _norm_st(row.get("standard_type"))

    if not st:
        return "review_mixed", "empty_standard_type"

    # 1) 明显非心脏方向
    if _exclude_blob(blob):
        return "exclude_non_cardiac", "exclude_needle_non_cardiac_topic"

    # 2) FC：歧义大；仅当有 QT 叙事或 hERG 机制文本时才保留为辅助，否则排除
    if st in FC_TYPES:
        if _qt_primary_blob(blob) or _herg_mechanism_blob(blob):
            return "qt_auxiliary_mechanism", "FC_with_qt_or_herg_text_not_primary_evidence"
        return "exclude_non_cardiac", "FC_without_qt_herg_context"

    # 3) AC50 / Activity：易混 secondary pharmacology；无 QT/hERG 机制则排除或 review
    if st in NOISY_POTENCY_TYPES:
        if "secondary pharmacology" in blob:
            return "exclude_non_cardiac", "AC50_activity_secondary_pharmacology_context"
        if _qt_primary_blob(blob):
            return "qt_primary", "potency_type_with_qt_primary_title"
        if _herg_mechanism_blob(blob):
            return "qt_auxiliary_mechanism", "potency_type_with_herg_mechanism_blob"
        return "exclude_non_cardiac", "potency_type_without_qt_herg_context"

    # 4) FPD / FPDc / APDc：优先保留（已通过排除针）
    if st in PRIMARY_DIRECT_TYPES:
        return "qt_primary", "direct_type_fpd_fpdc_apdc"

    # 5) pIC50：需 QT 叙事或 hERG 机制
    if st == "pic50":
        if _qt_primary_blob(blob):
            return "qt_primary", "pIC50_with_qt_primary_context"
        if _herg_mechanism_blob(blob):
            return "qt_auxiliary_mechanism", "pIC50_with_herg_mechanism_only"
        return "review_mixed", "pIC50_without_clear_qt_herg_context"

    # 6) APD/ERP/HR…：默认降权；若标题已是明确 proarrhythmic/torsade 叙事则抬到 primary
    if st in ELECTROPHYS_DOWNWEIGHT_TYPES:
        if _qt_primary_blob(blob):
            return "qt_primary", "ep_type_but_qt_primary_title_overrides"
        if _antiarrhythmic_blob(blob):
            return "electrophysiology_downweight", "antiarrhythmic_context_ep_type"
        return "electrophysiology_downweight", "ep_type_without_qt_toxic_frame"

    # 7) IC50 / Ki / Inhibition / kon…
    if st in MECHANISM_TYPES:
        if _qt_primary_blob(blob):
            return "qt_primary", "mechanism_type_with_qt_primary_title"
        if _herg_mechanism_blob(blob):
            return "qt_auxiliary_mechanism", "mechanism_type_with_herg_qt_mechanism_blob"
        return "review_mixed", "mechanism_type_without_qt_herg_title_or_target"

    # 8) 其余类型
    if _qt_primary_blob(blob):
        return "qt_primary", "fallback_with_qt_primary_blob"
    return "review_mixed", "unclassified_standard_type"


def main() -> None:
    p = argparse.ArgumentParser(description="final_candidates QT/DIQT 证据分层")
    p.add_argument("--input", type=Path, required=True, help="final_candidates.csv")
    p.add_argument("--out-dir", type=Path, required=True, help="输出目录")
    p.add_argument(
        "--prefix",
        type=str,
        default="chembl_cardiotox_qt_tier",
        help="输出文件前缀",
    )
    p.add_argument(
        "--no-split",
        action="store_true",
        help="只写带列的完整 CSV，不按层拆文件",
    )
    args = p.parse_args()

    inp = Path(args.input).expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"找不到输入: {inp}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, encoding="utf-8-sig", low_memory=False)
    tiers: list[str] = []
    reasons: list[str] = []
    for _, row in df.iterrows():
        t, r = classify_evidence_qt_tier(row)
        tiers.append(t)
        reasons.append(r)
    df = df.copy()
    df["evidence_qt_tier"] = tiers
    df["evidence_qt_reason"] = reasons

    stem = args.prefix
    full_path = out_dir / f"{stem}__annotated.csv"
    df.to_csv(full_path, index=False, encoding="utf-8-sig")

    counts = df["evidence_qt_tier"].value_counts().to_dict()
    summary = {
        "input": str(inp),
        "total_rows": len(df),
        "tier_counts": counts,
        "annotated_csv": str(full_path),
    }

    if not args.no_split:
        for tier in sorted(df["evidence_qt_tier"].unique()):
            sub = df[df["evidence_qt_tier"] == tier]
            safe = tier.replace(" ", "_")
            p_t = out_dir / f"{stem}__{safe}.csv"
            sub.to_csv(p_t, index=False, encoding="utf-8-sig")
            summary.setdefault("split_outputs", {})[tier] = str(p_t)

    meta_path = out_dir / f"{stem}__summary.json"
    meta_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[save] {full_path}")
    print(f"[save] {meta_path}")


if __name__ == "__main__":
    main()
