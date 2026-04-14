"""
对 chembl_cardiotox_activity_screen 生成的 CSV（推荐 __all_scored 或 __final_candidates）做三层分流：

  - core_cardiac.csv      高置信，直接用于分析/建模
  - extended_cardiac.csv  Tier2 活性类型 + 心脏离子通道白名单靶点
  - review_needed.csv     需人工复核（含 FC 歧义、Kd+怪靶点、关键词命中但靶点不像心脏等）

用法（在 cardiotoxicity_prediction 根目录）:

  python -m data_augmentation.Chembl_data.chembl_cardiotox_csv_postfilter ^
    --input data_augmentation/Chembl_data/output/full_enrichment3/chembl_cardiotox_screen__all_scored.csv ^
    --out-dir data_augmentation/Chembl_data/output/full_enrichment3_postfilter

依赖: pandas
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# --- A. 直接心脏电生理/心功能（不含 FC 系列，FC 单独规则）---
DIRECT_CORE_TYPES: frozenset[str] = frozenset(
    {
        "FPD measurement",
        "FPDc",
        "APD",
        "APDc",
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
        "HR",
        "Max delta HR",
        "Delta HR",
        "VCT",
        "VFT",
        "Cardiac depression",
        "MBP",
    }
)

FC_TYPES: frozenset[str] = frozenset({"FC", "Max delta FC", "Delta FC"})

CONDITIONAL_TYPES: frozenset[str] = frozenset(
    {
        "IC50",
        "Ki",
        "Kd",
        "pIC50",
        "AC50",
        "Ac50",
        "Inhibition",
        "Inhibition index",
        "Potency",
        "Activity",
        "Ratio Ki",
        "kon",
        "k_on",
        "k_off",
        "Koff",
        "koff",
        "Binding energy",
        "Kif",
    }
)


def _norm_st(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip().lower()


DIRECT_CORE_LOWER = frozenset(_norm_st(x) for x in DIRECT_CORE_TYPES)
FC_LOWER = frozenset(_norm_st(x) for x in FC_TYPES)
COND_LOWER = frozenset(_norm_st(x) for x in CONDITIONAL_TYPES)


def _s(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


# Tier2 靶点/assay 白名单（子串匹配，小写）
EXTENDED_TARGET_WHITELIST: tuple[str, ...] = (
    "voltage-gated inwardly rectifying potassium channel kcnh2",
    "kcnh2",
    "kv11.1",
    "herg",
    "kcnq1",
    "iks",
    "kv7.1",
    "kcne1",
    "scn5a",
    "sodium channel protein type 5",
    "nav1.5",
    "cacna1c",
    "voltage-dependent l-type calcium channel subunit alpha-1c",
    "voltage-gated l-type calcium channel",
    "voltage-dependent n-type calcium channel subunit alpha-1b",
    "l-type calcium",
    "cav1.2",
    "cav1.3",
    "ikr",
)

# 明显非心脏核心证据（子串）
TARGET_ASSAY_BLACKLIST: tuple[str, ...] = (
    "molecular identity unknown",
    "escherichia",
    "salmonella",
    "candida",
    "plasmodium",
    "trypanosoma",
    "hepatocyte",
    "hepatocellular",
    "liver microsome",
    "renal",
    "cyp3a4",
    "cyp2c9",
    "substrate",
    "permeability",
    "caco-2",
    "mdck",
    "blood-brain",
    "admet",
    "logp",
    "solubility",
)

# FC 进入 core：assay 关键词
FC_ASSAY_CORE: tuple[str, ...] = (
    "contractility",
    "contraction",
    "cardiomyocyte",
    "cardiac muscle",
    "papillary muscle",
    "langendorff",
    "left ventricular",
)

# FC：文献标题表型
FC_DOC_CORE: tuple[str, ...] = (
    "cardiac",
    "heart",
    "arrhythmia",
    "qt",
    "electrocardiogram",
    "ecg",
    "heart failure",
    "contractility",
    "cardiomyocyte",
)

# core 直接类型：靶点/assay/文献 任一层「心脏相关」支持
CARDIAC_SUPPORT_NEEDLES: tuple[str, ...] = (
    "herg",
    "kcnh2",
    "ikr",
    "kcnq1",
    "scn5a",
    "nav1.5",
    "cacna1c",
    "cav1.2",
    "cav1.3",
    "cardiac",
    "heart",
    "ventricular",
    "atrial",
    "qt",
    "arrhythmia",
    "electrophysiolog",
    "repolarization",
    "ion channel",
    "calcium channel",
    "sodium channel",
    "potassium channel",
)


def keyword_group_score(cardiotox_groups: Any) -> int:
    """A_direct:+3, B_mechanism:+2, C_phenotype:+1"""
    if pd.isna(cardiotox_groups) or not str(cardiotox_groups).strip():
        return 0
    g = str(cardiotox_groups)
    s = 0
    if "A_direct" in g:
        s += 3
    if "B_mechanism" in g:
        s += 2
    if "C_phenotype" in g:
        s += 1
    return s


def has_cardiac_layer_support(tgt: str, asy: str, doc: str) -> bool:
    blob = f"{tgt} {asy} {doc}".lower()
    return any(n in blob for n in CARDIAC_SUPPORT_NEEDLES)


def fc_qualifies_for_core(tgt: str, asy: str, doc: str) -> bool:
    t, a, d = tgt.lower(), asy.lower(), doc.lower()
    if any(n in t for n in EXTENDED_TARGET_WHITELIST):
        return True
    if any(k in a for k in FC_ASSAY_CORE):
        return True
    if any(k in d for k in FC_DOC_CORE):
        return True
    return False


def extended_whitelist_hit(tgt: str, asy: str) -> bool:
    blob = f"{tgt} {asy}".lower()
    return any(w in blob for w in EXTENDED_TARGET_WHITELIST)


def blacklist_hit(tgt: str, asy: str) -> bool:
    blob = f"{tgt} {asy}".lower()
    return any(b in blob for b in TARGET_ASSAY_BLACKLIST)


def suspicious_kd_ki(st_key: str, tgt: str, asy: str) -> bool:
    if st_key not in ("kd", "ki"):
        return False
    blob = f"{tgt} {asy}".lower()
    if not blob.strip():
        return True
    if blacklist_hit(tgt, asy):
        return True
    if not any(w in blob for w in EXTENDED_TARGET_WHITELIST) and not has_cardiac_layer_support(
        tgt, asy, ""
    ):
        return True
    return False


def classify_row(row: pd.Series) -> tuple[str, str]:
    """
    返回 (bucket, reason)，bucket in {core, extended, review, drop}
    """
    st_raw = row.get("standard_type", "")
    st_key = _norm_st(st_raw)
    if not st_key:
        return "drop", "empty_standard_type"

    tgt = _s(row.get("tgt_pref_name"))
    asy = _s(row.get("asy_description"))
    doc = _s(row.get("doc_title"))
    hit = bool(row.get("cardiotox_hit")) if pd.notna(row.get("cardiotox_hit")) else False

    # C：不在 A∪B∪FC（与 ChEMBL 大小写/空格差异兼容）
    in_direct = st_key in DIRECT_CORE_LOWER
    in_fc = st_key in FC_LOWER
    in_cond = st_key in COND_LOWER
    if not (in_direct or in_fc or in_cond):
        return "drop", "standard_type_not_in_AB_or_FC"

    # 黑名单：直接剔除到 drop（不进三层）
    if blacklist_hit(tgt, asy):
        return "drop", "blacklist_target_assay"

    # FC：不进 direct core 集合，单独
    if in_fc:
        if fc_qualifies_for_core(tgt, asy, doc):
            return "core", "FC_with_cardiac_support"
        return "review", "FC_ambiguous_no_cardiac_support"

    # 直接类型（非 FC）
    if in_direct:
        if not has_cardiac_layer_support(tgt, asy, doc):
            return "review", "direct_type_but_weak_cardiac_text"
        return "core", "direct_cardiac_standard_type_with_support"

    # 条件类型 Tier2
    if in_cond:
        if extended_whitelist_hit(tgt, asy):
            return "extended", "conditional_type_whitelist_target_or_assay"
        if hit:
            return "review", "conditional_type_keyword_hit_but_not_whitelist_target"
        if suspicious_kd_ki(st_key, tgt, asy):
            return "review", "kd_ki_non_cardiac_context"
        return "review", "conditional_type_not_whitelist"

    return "drop", "unreachable"


def main() -> None:
    p = argparse.ArgumentParser(description="cardiotox CSV 三层分流后处理")
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="输入 CSV（建议 all_scored 或 final_candidates）",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="输出目录（将写入 core/extended/review 三个 CSV 与 summary json）",
    )
    p.add_argument(
        "--recent-years",
        type=int,
        default=5,
        help="用于 meta：近 N 年（默认 5），便于与主脚本 doc_year 对照",
    )
    p.add_argument(
        "--current-year",
        type=int,
        default=None,
        help="用于标注近年列；默认当年",
    )
    args = p.parse_args()

    inp = Path(args.input).expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"找不到输入: {inp}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, encoding="utf-8-sig")
    cy = int(args.current_year or pd.Timestamp.now().year)
    cutoff = cy - int(args.recent_years)

    df = df.copy()
    if "cardiotox_groups" in df.columns:
        df["keyword_score"] = df["cardiotox_groups"].apply(keyword_group_score)
    else:
        df["keyword_score"] = 0

    def _year(r: pd.Series) -> int:
        y = r.get("doc_year")
        try:
            if y is None or (isinstance(y, float) and pd.isna(y)):
                return -1
            return int(float(y))
        except Exception:
            return -1

    df["_doc_year_i"] = df.apply(_year, axis=1)
    df["is_recent_5y"] = df["_doc_year_i"] >= cutoff

    buckets: list[str] = []
    reasons: list[str] = []
    for _, row in df.iterrows():
        b, r = classify_row(row)
        buckets.append(b)
        reasons.append(r)
    df["postfilter_bucket"] = buckets
    df["postfilter_reason"] = reasons

    core = df[df["postfilter_bucket"] == "core"].copy()
    ext = df[df["postfilter_bucket"] == "extended"].copy()
    rev = df[df["postfilter_bucket"] == "review"].copy()
    dropped = df[df["postfilter_bucket"] == "drop"].copy()

    # 收口排序：core 近年优先；extended 近年优先；review 不强制时间
    core = core.sort_values(
        by=["is_recent_5y", "keyword_score", "_doc_year_i"],
        ascending=[False, False, False],
    )
    ext = ext.sort_values(
        by=["is_recent_5y", "keyword_score", "_doc_year_i"],
        ascending=[False, False, False],
    )
    rev = rev.sort_values(by=["keyword_score", "_doc_year_i"], ascending=[False, False])

    stem = "cardiotox_postfilter"
    p_core = out_dir / f"{stem}__core_cardiac.csv"
    p_ext = out_dir / f"{stem}__extended_cardiac.csv"
    p_rev = out_dir / f"{stem}__review_needed.csv"
    p_drop = out_dir / f"{stem}__dropped_not_AB_FC.csv"

    core.drop(columns=["_doc_year_i"], errors="ignore").to_csv(
        p_core, index=False, encoding="utf-8-sig"
    )
    ext.drop(columns=["_doc_year_i"], errors="ignore").to_csv(
        p_ext, index=False, encoding="utf-8-sig"
    )
    rev.drop(columns=["_doc_year_i"], errors="ignore").to_csv(
        p_rev, index=False, encoding="utf-8-sig"
    )
    dropped.drop(columns=["_doc_year_i"], errors="ignore").to_csv(
        p_drop, index=False, encoding="utf-8-sig"
    )

    summary = {
        "input": str(inp),
        "current_year": cy,
        "recent_years_cutoff": cutoff,
        "total_rows": len(df),
        "core_cardiac_rows": len(core),
        "extended_cardiac_rows": len(ext),
        "review_needed_rows": len(rev),
        "dropped_rows": len(dropped),
        "postfilter_reason_counts": df["postfilter_reason"].value_counts().to_dict(),
        "outputs": {
            "core_cardiac": str(p_core),
            "extended_cardiac": str(p_ext),
            "review_needed": str(p_rev),
            "dropped": str(p_drop),
        },
    }
    meta_path = out_dir / f"{stem}__summary.json"
    meta_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[save] {p_core}")
    print(f"[save] {p_ext}")
    print(f"[save] {p_rev}")
    print(f"[save] {p_drop}")
    print(f"[save] {meta_path}")


if __name__ == "__main__":
    main()
