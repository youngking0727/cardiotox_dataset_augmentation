"""
【调试副本】默认只跑 10 个主分子，输出到独立目录，避免与 chembl_cardiotox_activity_screen.py 全量任务冲突。

ChEMBL 活性筛选：按「流水线」执行（先类型预筛再补全，再打分）——比单纯调并发更省请求。

流水线（与旧版「全量补全再判断」相反）：
  ① 全量拉 activity（仅 Phase1 小字段）
  ② 用 standard_type 做第一轮过滤（Tier1 / Tier2 列表）
  ③ 只对「可能相关」的 activity 所涉及的 assay / document / target 发 Phase2 请求
  ④ 合并文本后：关键词 cardiotox_hit + 类型分层 cardiotox_type_*
  ⑤ 最终候选、按分子分层 Top-N（近五年优先）

说明：关键词匹配依赖 Phase2 文本，因此仅对通过 ② 的行有意义；其它 activity 仍保留在 all_scored 中但无补全文本。

默认参数示例：phase1-concurrency=2、phase1-delay-s=0.03、concurrency=6、delay-s=0.03。
正式跑请勿使用 --max-activities-per-molecule（仅调试）。

依赖： pip install chembl-webresource-client

用法（仓库根目录 cardiotoxicity_prediction）:

  python -m data_augmentation.Chembl_data.chembl_cardiotox_activity_screen ^
    --summary-csv data_augmentation/Chembl_data/output/full_enrichment1/chembl_excel_full_enrichment_summary.csv

默认输出目录：data_augmentation/Chembl_data/output/full_enrichment3_sample10/
默认只处理前 10 个主分子；全量请用 --sample-n 0 或改用原版脚本。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    from chembl_webresource_client.new_client import new_client
except ImportError as e:
    raise SystemExit(
        "需要安装：pip install chembl-webresource-client\n" + str(e)
    ) from e

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None  # type: ignore[misc, assignment]

CHEMBL_DATA_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = CHEMBL_DATA_DIR / "output" / "full_enrichment3_sample10"
DEFAULT_SUMMARY = (
    CHEMBL_DATA_DIR
    / "output"
    / "full_enrichment1"
    / "chembl_excel_full_enrichment_summary.csv"
)

# 阶段一：仅 ID + 数值字段（不含 assay_description 等长字段，避免 API 过慢）
PHASE1_ACTIVITY_ONLY: list[str] = [
    "activity_id",
    "molecule_chembl_id",
    "assay_chembl_id",
    "document_chembl_id",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "target_chembl_id",
]

KEYWORD_GROUPS: dict[str, list[str]] = {
    "A_direct": [
        "cardiotoxic",
        "cardiotoxicity",
        "cardiac toxicity",
        "qt prolongation",
        "long qt",
        "torsade",
        "torsades",
        "arrhythmia",
        "proarrhythmic",
    ],
    "B_mechanism": [
        "herg",
        "kcnh2",
        "ikr",
        "ventricular repolarization",
    ],
    "C_phenotype": [
        "ecg",
        "electrocardiogram",
        "apd",
        "action potential duration",
        "qt",
    ],
}

# standard_type 与心脏毒性「可能相关」的两类（大小写不敏感匹配 ChEMBL 的 standard_type 文本）
# 一类：直接电生理/心功能指标
_CT_TIER1_TYPES: frozenset[str] = frozenset(
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
        "FC",
        "Max delta FC",
        "Delta FC",
        "Cardiac depression",
        "MBP",
        "VCT",
        "VFT",
    }
)
_CT_TIER1_LOWER: frozenset[str] = frozenset(x.lower() for x in _CT_TIER1_TYPES)

# 二类：通用活性/结合类型；仅当靶点/assay 文本命中心脏离子通道相关关键词时才标为可能相关
_CT_TIER2_TYPES: frozenset[str] = frozenset(
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
_CT_TIER2_LOWER: frozenset[str] = frozenset(x.lower() for x in _CT_TIER2_TYPES)


def _norm_standard_type_key(st: Any) -> str:
    if st is None or (isinstance(st, float) and pd.isna(st)):
        return ""
    return str(st).strip().lower()


def _text_suggests_cardiac_ion_target(text: str) -> bool:
    """hERG / KCNH2 / IKr / NaV1.5 / SCN5A / 钙通道等（与 ChEMBL pref_name 常见写法对齐）。"""
    if not text or not str(text).strip():
        return False
    u = str(text).lower()
    needles = (
        "herg",
        "kcnh2",
        "kv11.1",
        "ikr",
        "nav1.5",
        "scn5a",
        "kcnq1",
        "iks",
        "ikr channel",
        "l-type calcium",
        "cacna1c",
        "cacna1",
        "cav1.2",
        "cav1.3",
        "cardiac sodium",
        "cardiac calcium",
        "voltage-gated sodium channel",
        "voltage-gated calcium channel",
    )
    return any(n in u for n in needles)


def classify_cardiotox_type_relevance(
    standard_type: Any,
    tgt_pref_name: str,
    assay_description_text: str,
) -> tuple[bool, str, str]:
    """
    返回 (是否标注为可能心脏毒性相关, tier 标签, 简短说明)。
    tier: 1_direct | 2_conditional_cardiac_target | ''
    """
    key = _norm_standard_type_key(standard_type)
    if not key:
        return False, "", ""

    blob = f"{tgt_pref_name or ''} {assay_description_text or ''}"
    if key in _CT_TIER1_LOWER:
        return True, "1_direct", "standard_type=tier1_direct"

    if key in _CT_TIER2_LOWER:
        if _text_suggests_cardiac_ion_target(blob):
            return True, "2_conditional_cardiac_target", "standard_type=tier2+cardiac_ion_target"
        return False, "", "tier2_type_but_non_cardiac_ion_target"

    return False, "", ""


def _phase1_exception_is_transient(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if any(
        x in msg
        for x in (
            "500",
            "502",
            "503",
            "504",
            "429",
            "timeout",
            "timed out",
            "read timed out",
            "connecttimeout",
            "technical error",
            "httpapplicationerror",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "connection reset",
            "connection aborted",
            "temporarily unavailable",
        )
    ):
        return True
    return False


def _configure_chembl_settings(*, http_timeout_s: float, no_cache: bool) -> None:
    """chembl_webresource_client 默认 NEW_CLIENT_TIMEOUT=None，请求可能无限挂起。"""
    from chembl_webresource_client.settings import Settings

    s = Settings.Instance()
    if http_timeout_s > 0:
        s.NEW_CLIENT_TIMEOUT = float(http_timeout_s)
    if no_cache:
        s.CACHING = False


def _norm_chembl_id(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    return s if s.startswith("CHEMBL") else ""


def compile_keyword_patterns(
    groups: dict[str, list[str]],
) -> dict[str, list[re.Pattern[str]]]:
    out: dict[str, list[re.Pattern[str]]] = {}
    for gname, words in groups.items():
        pats: list[re.Pattern[str]] = []
        for w in words:
            w = w.strip()
            if not w:
                continue
            pats.append(re.compile(re.escape(w), re.IGNORECASE))
        out[gname] = pats
    return out


def match_keywords(
    text: str,
    patterns: dict[str, list[re.Pattern[str]]],
) -> tuple[bool, list[str]]:
    if not text or not str(text).strip():
        return False, []
    hit_groups: list[str] = []
    for gname, pats in patterns.items():
        for p in pats:
            if p.search(text):
                hit_groups.append(gname)
                break
    return (len(hit_groups) > 0), sorted(set(hit_groups))


def _fetch_assay_sync(aid: str, only: list[str]) -> tuple[str, dict[str, Any]]:
    assay = new_client.assay
    try:
        rows = list(assay.filter(assay_chembl_id=aid).only(only))
        return aid, (rows[0] if rows else {})
    except Exception as e:
        return aid, {"_fetch_error": str(e)}


def _fetch_document_sync(did: str, only: list[str]) -> tuple[str, dict[str, Any]]:
    doc = new_client.document
    try:
        rows = list(doc.filter(document_chembl_id=did).only(only))
        return did, (rows[0] if rows else {})
    except Exception as e:
        return did, {"_fetch_error": str(e)}


def _fetch_target_sync(tid: str, only: list[str]) -> tuple[str, dict[str, Any]]:
    tgt = new_client.target
    try:
        rows = list(tgt.filter(target_chembl_id=tid).only(only))
        return tid, (rows[0] if rows else {})
    except Exception as e:
        return tid, {"_fetch_error": str(e)}


def _phase1_sync_one_molecule(
    mid: str,
    cap: Optional[int],
    *,
    max_retries: int,
    retry_backoff_s: float,
) -> tuple[list[dict[str, Any]], set[str], set[str], set[str]]:
    activity = new_client.activity
    last_exc: Optional[BaseException] = None
    attempts = max(1, int(max_retries))
    for attempt in range(attempts):
        rows: list[dict[str, Any]] = []
        assay_ids: set[str] = set()
        doc_ids: set[str] = set()
        tgt_ids: set[str] = set()
        try:
            qs = activity.filter(molecule_chembl_id=mid).only(PHASE1_ACTIVITY_ONLY)
            n = 0
            for act in qs:
                rows.append(dict(act))
                ac = _norm_chembl_id(act.get("assay_chembl_id", ""))
                dc = _norm_chembl_id(act.get("document_chembl_id", ""))
                tc = _norm_chembl_id(act.get("target_chembl_id", ""))
                if ac:
                    assay_ids.add(ac)
                if dc:
                    doc_ids.add(dc)
                if tc:
                    tgt_ids.add(tc)
                n += 1
                if cap is not None and n >= int(cap):
                    break
            return rows, assay_ids, doc_ids, tgt_ids
        except Exception as e:
            last_exc = e
            if attempt < attempts - 1 and _phase1_exception_is_transient(e):
                time.sleep(float(retry_backoff_s) * (2**attempt))
                continue
            break

    err = last_exc if last_exc is not None else RuntimeError("phase1 unknown error")
    return (
        [
            {
                "molecule_chembl_id": mid,
                "_phase1_error": str(err),
            }
        ],
        set(),
        set(),
        set(),
    )


async def _fetch_batch_async(
    ids: set[str],
    *,
    only: list[str],
    delay_s: float,
    concurrency: int,
    sync_fn: Any,
    desc: str,
    no_progress: bool,
) -> dict[str, dict[str, Any]]:
    """用 asyncio.to_thread + Semaphore(concurrency) 并发拉取，避免阻塞事件循环。"""
    id_list = sorted(a for a in ids if a)
    if not id_list:
        return {}

    sem = asyncio.Semaphore(max(1, int(concurrency)))

    async def one(i: str) -> tuple[str, dict[str, Any]]:
        async with sem:
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            return await asyncio.to_thread(sync_fn, i, only)

    if _tqdm is not None and not no_progress:
        try:
            from tqdm.asyncio import tqdm as tqdm_async

            pairs = list(
                await tqdm_async.gather(
                    *[one(i) for i in id_list],
                    desc=desc,
                    unit="id",
                )
            )
        except Exception:
            pairs = list(await asyncio.gather(*[one(i) for i in id_list]))
            print(f"[done] {desc}: {len(pairs)} ids", flush=True)
    else:
        pairs = list(await asyncio.gather(*[one(i) for i in id_list]))
        print(f"[done] {desc}: {len(pairs)} ids", flush=True)

    return dict(pairs)


def select_time_priority(
    rows: list[dict[str, Any]],
    *,
    year_key: str,
    current_year: int,
    recent_years: int,
    top_n: int,
) -> list[dict[str, Any]]:
    """先取 document_year >= current_year - recent_years，按年降序；不足再从更早记录按年降序补满 top_n。"""
    cutoff = current_year - int(recent_years)

    def year_val(r: dict[str, Any]) -> int:
        y = r.get(year_key)
        try:
            if y is None or (isinstance(y, float) and pd.isna(y)):
                return -1
            return int(float(y))
        except Exception:
            return -1

    scored = sorted(rows, key=lambda r: -year_val(r))
    recent = [r for r in scored if year_val(r) >= cutoff]
    older = [r for r in scored if year_val(r) < cutoff]
    out = recent[:top_n]
    if len(out) < top_n:
        need = top_n - len(out)
        out.extend(older[:need])
    return out


def _standard_type_needs_phase2_fetch(act: dict[str, Any]) -> bool:
    """仅对 Tier1/Tier2 standard_type 补全 assay/document/target，降低 phase2 请求量。"""
    if "_phase1_error" in act:
        return False
    k = _norm_standard_type_key(act.get("standard_type"))
    if not k:
        return False
    return k in _CT_TIER1_LOWER or k in _CT_TIER2_LOWER


def collect_phase2_ids_prefilter(
    phase1_rows: list[dict[str, Any]],
) -> tuple[set[str], set[str], set[str]]:
    assay_ids: set[str] = set()
    doc_ids: set[str] = set()
    tgt_ids: set[str] = set()
    for act in phase1_rows:
        if not _standard_type_needs_phase2_fetch(act):
            continue
        ac = _norm_chembl_id(act.get("assay_chembl_id", ""))
        dc = _norm_chembl_id(act.get("document_chembl_id", ""))
        tc = _norm_chembl_id(act.get("target_chembl_id", ""))
        if ac:
            assay_ids.add(ac)
        if dc:
            doc_ids.add(dc)
        if tc:
            tgt_ids.add(tc)
    return assay_ids, doc_ids, tgt_ids


def select_time_priority_layered(
    rows: list[dict[str, Any]],
    *,
    year_key: str,
    current_year: int,
    recent_years: int,
    top_n: int,
) -> list[dict[str, Any]]:
    """同一分子内：优先 cardiotox_hit，再 cardiotox_type_relevant，再其余 activity；各层内按时间策略取。"""
    if top_n <= 0:
        return []
    tier_kw = [r for r in rows if r.get("cardiotox_hit")]
    tier_ty = [
        r
        for r in rows
        if r.get("cardiotox_type_relevant") and not r.get("cardiotox_hit")
    ]
    tier_rest = [
        r
        for r in rows
        if not r.get("cardiotox_hit") and not r.get("cardiotox_type_relevant")
    ]
    out: list[dict[str, Any]] = []
    out.extend(
        select_time_priority(
            tier_kw,
            year_key=year_key,
            current_year=current_year,
            recent_years=recent_years,
            top_n=top_n,
        )
    )
    if len(out) >= top_n:
        return out[:top_n]
    need = top_n - len(out)
    out.extend(
        select_time_priority(
            tier_ty,
            year_key=year_key,
            current_year=current_year,
            recent_years=recent_years,
            top_n=need,
        )
    )
    if len(out) >= top_n:
        return out[:top_n]
    need = top_n - len(out)
    out.extend(
        select_time_priority(
            tier_rest,
            year_key=year_key,
            current_year=current_year,
            recent_years=recent_years,
            top_n=need,
        )
    )
    return out[:top_n]


def enrich_activity_rows(
    phase1_rows: list[dict[str, Any]],
    assay_cache: dict[str, dict[str, Any]],
    doc_cache: dict[str, dict[str, Any]],
    tgt_cache: dict[str, dict[str, Any]],
    patterns: dict[str, list[re.Pattern[str]]],
) -> list[dict[str, Any]]:
    """
    Step ④：在 Phase2 缓存就绪后，对每条 activity 合并文本并计算
    cardiotox_hit（关键词）与 cardiotox_type_*（standard_type 分层）。
    仅 _standard_type_needs_phase2_fetch 为 True 的行带有 assay/doc/tgt 文本。
    """
    enriched: list[dict[str, Any]] = []
    for act in phase1_rows:
        if "_phase1_error" in act:
            enriched.append(
                {
                    **act,
                    "cardiotox_hit": False,
                    "cardiotox_groups": "",
                    "cardiotox_type_relevant": False,
                    "cardiotox_type_tier": "",
                    "cardiotox_type_note": "",
                }
            )
            continue

        ac = _norm_chembl_id(act.get("assay_chembl_id", ""))
        dc = _norm_chembl_id(act.get("document_chembl_id", ""))
        tc = _norm_chembl_id(act.get("target_chembl_id", ""))

        if _standard_type_needs_phase2_fetch(act):
            arow = assay_cache.get(ac, {})
            drow = doc_cache.get(dc, {})
            trow = tgt_cache.get(tc, {})
        else:
            arow, drow, trow = {}, {}, {}

        assay_txt = " ".join(
            [
                str(arow.get("description", "") or ""),
                str(arow.get("assay_type_description", "") or ""),
                str(arow.get("assay_type", "") or ""),
            ]
        )
        doc_title = str(drow.get("title", "") or "")
        doc_journal = str(drow.get("journal", "") or "")
        try:
            doc_year = drow.get("year")
            if doc_year is not None and not (isinstance(doc_year, float) and pd.isna(doc_year)):
                doc_year_i = int(float(doc_year))
            else:
                doc_year_i = None
        except Exception:
            doc_year_i = None

        tgt_name = str(trow.get("pref_name", "") or "")

        blob = " ".join([assay_txt, tgt_name, doc_title, doc_journal])
        hit, groups = match_keywords(blob, patterns)

        type_rel, type_tier, type_note = classify_cardiotox_type_relevance(
            act.get("standard_type"),
            tgt_name,
            assay_txt,
        )

        enriched.append(
            {
                **act,
                "asy_description": arow.get("description", ""),
                "asy_type": arow.get("assay_type", ""),
                "doc_title": doc_title,
                "doc_journal": doc_journal,
                "doc_year": doc_year_i,
                "tgt_pref_name": tgt_name,
                "match_text_blob": blob[:2000],
                "cardiotox_hit": hit,
                "cardiotox_groups": ",".join(groups) if groups else "",
                "cardiotox_type_relevant": type_rel,
                "cardiotox_type_tier": type_tier,
                "cardiotox_type_note": type_note,
            }
        )
    return enriched


def build_final_candidates_keyword_type_and_top(
    enriched: list[dict[str, Any]],
    *,
    current_year: int,
    recent_years: int,
    top_n: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Step ⑤：最终候选、子集列表、按分子分层时间 Top-N。"""
    final_candidates = [
        r for r in enriched if r.get("cardiotox_hit") or r.get("cardiotox_type_relevant")
    ]
    keyword_hits_only = [r for r in enriched if r.get("cardiotox_hit")]
    type_relevant_only = [r for r in enriched if r.get("cardiotox_type_relevant")]

    mol_ids_top = {
        str(r.get("molecule_chembl_id", "") or "").strip()
        for r in enriched
        if str(r.get("molecule_chembl_id", "") or "").strip()
    }
    top_rows: list[dict[str, Any]] = []
    for m in sorted(mol_ids_top):
        lst = [
            r
            for r in enriched
            if str(r.get("molecule_chembl_id", "") or "").strip() == m
        ]
        picked = select_time_priority_layered(
            lst,
            year_key="doc_year",
            current_year=current_year,
            recent_years=int(recent_years),
            top_n=int(top_n),
        )
        for r in picked:
            r2 = dict(r)
            r2["time_priority_molecule_chembl_id"] = m
            top_rows.append(r2)

    return final_candidates, keyword_hits_only, type_relevant_only, top_rows


def _short_err(msg: str, max_len: int = 400) -> str:
    s = str(msg).replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _phase2_fetch_error_pairs(cache: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for cid, row in cache.items():
        if isinstance(row, dict) and row.get("_fetch_error"):
            out.append((str(cid), _short_err(str(row["_fetch_error"]))))
    return out


def build_request_failure_report(
    phase1_rows: list[dict[str, Any]],
    assay_cache: dict[str, dict[str, Any]],
    doc_cache: dict[str, dict[str, Any]],
    tgt_cache: dict[str, dict[str, Any]],
    *,
    preview_per_section: int = 30,
) -> tuple[dict[str, Any], str]:
    """
    汇总 Phase1 分子级失败与 Phase2 单 ID 拉取失败，供控制台提示与写日志。
    返回 (写入 meta 的统计 dict, 完整日志正文)。
    """
    p1_fail: list[tuple[str, str]] = []
    seen_mol: set[str] = set()
    for act in phase1_rows:
        if "_phase1_error" not in act:
            continue
        mid = str(act.get("molecule_chembl_id", "") or "").strip()
        if not mid or mid in seen_mol:
            continue
        seen_mol.add(mid)
        p1_fail.append((mid, _short_err(str(act.get("_phase1_error", "")))))

    a_err = _phase2_fetch_error_pairs(assay_cache)
    d_err = _phase2_fetch_error_pairs(doc_cache)
    t_err = _phase2_fetch_error_pairs(tgt_cache)

    n_p1, na, nd, nt = len(p1_fail), len(a_err), len(d_err), len(t_err)
    total_events = n_p1 + na + nd + nt

    stats: dict[str, Any] = {
        "phase1_molecule_failures": n_p1,
        "phase2_assay_id_failures": na,
        "phase2_document_id_failures": nd,
        "phase2_target_id_failures": nt,
        "total_failure_events": total_events,
    }

    lines: list[str] = [
        "=== chembl_cardiotox_activity_screen 请求失败摘要 ===",
        f"Phase1 按分子拉 activity 失败: {n_p1} 个分子（每分子一条错误占位行）",
    ]
    for mid, msg in p1_fail[:preview_per_section]:
        lines.append(f"  [phase1] {mid}: {msg}")
    if n_p1 > preview_per_section:
        lines.append(f"  ... 其余 {n_p1 - preview_per_section} 个分子略（仍写入 all_scored 的 _phase1_error）")

    for label, pairs in (
        ("assay", a_err),
        ("document", d_err),
        ("target", t_err),
    ):
        lines.append(f"Phase2 {label} 单条 ID 请求失败: {len(pairs)} 个")
        for cid, msg in pairs[:preview_per_section]:
            lines.append(f"  [phase2 {label}] {cid}: {msg}")
        if len(pairs) > preview_per_section:
            lines.append(f"  ... 其余 {len(pairs) - preview_per_section} 条略")

    lines.append("")
    lines.append("说明：Phase2 失败时该行仍可能参与流程，但 assay/doc/tgt 文本为空或不全。")
    body = "\n".join(lines)
    return stats, body


def print_and_maybe_log_failures(
    *,
    out_dir: Path,
    stem: str,
    failure_stats: dict[str, Any],
    failure_log_body: str,
) -> Path:
    """控制台 [WARN]；若有失败则写 __request_failures.log。"""
    log_path = out_dir / f"{stem}__request_failures.log"
    n = int(failure_stats.get("total_failure_events") or 0)
    if n <= 0:
        print("[request failures] 无 Phase1/Phase2 请求错误。", flush=True)
        return log_path

    print(
        "[WARN] 本次存在 ChEMBL 请求失败："
        f"Phase1 分子失败={failure_stats.get('phase1_molecule_failures')}, "
        f"Phase2 assay/doc/tgt 失败数="
        f"{failure_stats.get('phase2_assay_id_failures')}/"
        f"{failure_stats.get('phase2_document_id_failures')}/"
        f"{failure_stats.get('phase2_target_id_failures')}。"
        f"详情见: {log_path}",
        flush=True,
    )
    log_path.write_text(failure_log_body, encoding="utf-8")
    print(f"[save] {log_path}", flush=True)
    return log_path


def main() -> None:
    p = argparse.ArgumentParser(description="ChEMBL 两阶段活性 + 心脏毒性关键词 + 时间优先 Top-N")
    p.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="默认 Chembl_data/output/full_enrichment3_sample10（与全量任务输出分离）",
    )
    p.add_argument(
        "--sample-n",
        type=int,
        default=10,
        help="只跑前 N 个去重主分子；默认 10。设为 0 表示不限制（与原版一致全量）",
    )
    p.add_argument("--max-molecules", type=int, default=None)
    p.add_argument(
        "--max-activities-per-molecule",
        type=int,
        default=None,
        help="仅调试：每个分子最多拉多少条 activity。正式跑请勿设，否则近五年优先等会失真",
    )
    p.add_argument(
        "--phase1-concurrency",
        type=int,
        default=4,
        help="阶段一拉 activity 并发数（默认 2；过大易触发 ChEMBL/EBI 500）",
    )
    p.add_argument(
        "--phase1-delay-s",
        type=float,
        default=0.02,
        help="阶段一每分子请求前额外 sleep（秒），减轻 API 压力",
    )
    p.add_argument(
        "--phase1-max-retries",
        type=int,
        default=5,
        help="阶段一单次拉取失败时的最大尝试次数（含首次；对 5xx/超时等可重试错误退避重试）",
    )
    p.add_argument(
        "--phase1-retry-backoff-s",
        type=float,
        default=1.25,
        help="阶段一重试退避基数（秒），实际 sleep = backoff × 2^attempt",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=6,
        help="阶段二 assay/document/target 并发数（asyncio + to_thread，默认 6）",
    )
    p.add_argument("--delay-s", type=float, default=0.03, help="阶段二每条请求前额外 sleep（秒），减轻 API 压力")
    p.add_argument("--current-year", type=int, default=None, help="默认当年")
    p.add_argument("--recent-years", type=int, default=5)
    p.add_argument("--top-n", type=int, default=10, help="每个分子输出的时间优先记录数")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--http-timeout-s",
        type=float,
        default=120.0,
        help="ChEMBL API 单次 HTTP 超时（秒）。库默认无超时，网络卡住时会一直挂起。",
    )
    p.add_argument(
        "--no-chembl-cache",
        action="store_true",
        help="关闭 ChEMBL 客户端的本地 SQLite 缓存（若怀疑缓存锁或异常可试）",
    )
    args = p.parse_args()

    _configure_chembl_settings(
        http_timeout_s=float(args.http_timeout_s),
        no_cache=bool(args.no_chembl_cache),
    )

    summary_path = Path(args.summary_csv).expanduser().resolve()
    if not summary_path.is_file():
        raise SystemExit(f"找不到: {summary_path}")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else DEFAULT_OUT_DIR
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cy = int(args.current_year or time.localtime().tm_year)

    df = pd.read_csv(summary_path, encoding="utf-8-sig")
    if "primary_chembl_id" not in df.columns or "excel_row_index" not in df.columns:
        raise SystemExit("CSV 需含 primary_chembl_id、excel_row_index")

    mol_order: list[str] = []
    seen: set[str] = set()
    for _, row in df.iterrows():
        mid = _norm_chembl_id(row.get("primary_chembl_id", ""))
        if mid and mid not in seen:
            seen.add(mid)
            mol_order.append(mid)
    mol_limit: Optional[int] = None
    if int(args.sample_n) > 0:
        mol_limit = int(args.sample_n)
    if args.max_molecules is not None:
        mm = int(args.max_molecules)
        mol_limit = mm if mol_limit is None else min(mol_limit, mm)
    if mol_limit is not None:
        mol_order = mol_order[: max(0, mol_limit)]
        print(
            f"[sample10] 仅处理前 {len(mol_order)} 个主分子（limit={mol_limit}）",
            flush=True,
        )

    patterns = compile_keyword_patterns(KEYWORD_GROUPS)

    phase1_rows: list[dict[str, Any]] = []
    assay_ids: set[str] = set()
    doc_ids: set[str] = set()
    tgt_ids: set[str] = set()

    assay_cache: dict[str, dict[str, Any]] = {}
    doc_cache: dict[str, dict[str, Any]] = {}
    tgt_cache: dict[str, dict[str, Any]] = {}

    cap = args.max_activities_per_molecule

    async def run_phase1_async() -> None:
        nonlocal phase1_rows
        p1c = max(1, int(args.phase1_concurrency))

        async def fetch_one(
            mid: str,
        ) -> tuple[list[dict[str, Any]], set[str], set[str], set[str]]:
            if args.phase1_delay_s > 0:
                await asyncio.sleep(float(args.phase1_delay_s))
            return await asyncio.to_thread(
                _phase1_sync_one_molecule,
                mid,
                cap,
                max_retries=int(args.phase1_max_retries),
                retry_backoff_s=float(args.phase1_retry_backoff_s),
            )

        results: list[tuple[list[dict[str, Any]], set[str], set[str], set[str]]] = []

        if p1c <= 1:
            # 默认串行：避免 tqdm.asyncio + gather 在 Windows 上卡住；进度条用普通 tqdm
            mol_iter: Any = mol_order
            if _tqdm is not None and not args.no_progress:
                mol_iter = _tqdm(mol_order, desc="phase1 activity", unit="mol")
            for m in mol_iter:
                results.append(await fetch_one(m))
        else:
            sem = asyncio.Semaphore(p1c)

            async def one(mid: str) -> tuple[list[dict[str, Any]], set[str], set[str], set[str]]:
                async with sem:
                    return await fetch_one(mid)

            # 普通 gather 保序；每条完成时 update 进度条（避免 tqdm.asyncio.gather 在部分环境假死）
            p1_pbar: Any = None
            if _tqdm is not None and not args.no_progress:
                p1_pbar = _tqdm(total=len(mol_order), desc="phase1 activity", unit="mol")

            async def one_tracked(mid: str) -> tuple[list[dict[str, Any]], set[str], set[str], set[str]]:
                try:
                    return await one(mid)
                finally:
                    if p1_pbar is not None:
                        p1_pbar.update(1)

            results = list(await asyncio.gather(*[one_tracked(m) for m in mol_order]))
            if p1_pbar is not None:
                p1_pbar.close()

        phase1_rows = []
        for rows, _a, _d, _t in results:
            phase1_rows.extend(rows)

    async def run_phase2() -> None:
        assay_cache.update(
            await _fetch_batch_async(
                assay_ids,
                only=[
                    "assay_chembl_id",
                    "description",
                    "assay_type",
                    "assay_type_description",
                ],
                delay_s=args.delay_s,
                concurrency=args.concurrency,
                sync_fn=_fetch_assay_sync,
                desc="phase2 assay",
                no_progress=args.no_progress,
            )
        )
        doc_cache.update(
            await _fetch_batch_async(
                doc_ids,
                only=["document_chembl_id", "title", "journal", "year"],
                delay_s=args.delay_s,
                concurrency=args.concurrency,
                sync_fn=_fetch_document_sync,
                desc="phase2 document",
                no_progress=args.no_progress,
            )
        )
        tgt_cache.update(
            await _fetch_batch_async(
                tgt_ids,
                only=["target_chembl_id", "pref_name", "target_type"],
                delay_s=args.delay_s,
                concurrency=args.concurrency,
                sync_fn=_fetch_target_sync,
                desc="phase2 target",
                no_progress=args.no_progress,
            )
        )

    async def run_pipeline() -> None:
        """① Phase1 全量 activity →② 按 standard_type 收集待补全 ID →③ Phase2 批量拉元数据。"""
        nonlocal assay_ids, doc_ids, tgt_ids
        await run_phase1_async()
        assay_ids, doc_ids, tgt_ids = collect_phase2_ids_prefilter(phase1_rows)
        if _tqdm is not None and not args.no_progress:
            print(
                f"[phase2 prefilter] unique assays={len(assay_ids)} documents={len(doc_ids)} targets={len(tgt_ids)}",
                flush=True,
            )
        await run_phase2()

    # Step ① 全量 Phase1 activity → ② standard_type 预筛 ID → ③ 仅对这些 ID 跑 Phase2
    asyncio.run(run_pipeline())

    stem = "chembl_cardiotox_screen_sample10"
    failure_stats, failure_log_body = build_request_failure_report(
        phase1_rows, assay_cache, doc_cache, tgt_cache
    )
    failure_log_path = print_and_maybe_log_failures(
        out_dir=out_dir,
        stem=stem,
        failure_stats=failure_stats,
        failure_log_body=failure_log_body,
    )

    # Step ④ 合并文本 + 关键词 + 类型分层
    enriched = enrich_activity_rows(
        phase1_rows, assay_cache, doc_cache, tgt_cache, patterns
    )

    # Step ⑤ 最终候选、导出子集、按分子分层 Top-N
    final_candidates, keyword_hits_only, type_relevant_only, top_rows = (
        build_final_candidates_keyword_type_and_top(
            enriched,
            current_year=cy,
            recent_years=int(args.recent_years),
            top_n=int(args.top_n),
        )
    )

    all_path = out_dir / f"{stem}__all_scored.csv"
    final_cand_path = out_dir / f"{stem}__final_candidates.csv"
    kw_hit_path = out_dir / f"{stem}__keyword_hits_only.csv"
    top_path = out_dir / f"{stem}__top{args.top_n}_per_molecule_time_priority.csv"
    type_rel_path = out_dir / f"{stem}__type_relevant.csv"
    meta_path = out_dir / f"{stem}__meta.json"

    pd.DataFrame(enriched).to_csv(all_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(final_candidates).to_csv(final_cand_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(keyword_hits_only).to_csv(kw_hit_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(top_rows).to_csv(top_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(type_relevant_only).to_csv(type_rel_path, index=False, encoding="utf-8-sig")

    meta = {
        "summary_csv": str(summary_path),
        "current_year": cy,
        "recent_years": args.recent_years,
        "top_n_per_molecule": args.top_n,
        "molecules": len(mol_order),
        "molecule_limit_applied": mol_limit,
        "sample_n_arg": int(args.sample_n),
        "phase1_activity_rows": len(phase1_rows),
        "keyword_groups": list(KEYWORD_GROUPS.keys()),
        "unique_assays_fetched": len(assay_cache),
        "unique_documents_fetched": len(doc_cache),
        "unique_targets_fetched": len(tgt_cache),
        "phase1_async_concurrency": args.phase1_concurrency,
        "phase1_max_retries": args.phase1_max_retries,
        "phase1_delay_s": args.phase1_delay_s,
        "http_timeout_s": args.http_timeout_s,
        "chembl_cache_disabled": args.no_chembl_cache,
        "phase2_async_concurrency": args.concurrency,
        "final_candidates_rows": len(final_candidates),
        "keyword_hits_only_rows": len(keyword_hits_only),
        "cardiotox_type_relevant_rows": len(type_relevant_only),
        "top_priority_rows": len(top_rows),
        "request_failures": failure_stats,
        "outputs": {
            "all_scored": str(all_path),
            "final_candidates": str(final_cand_path),
            "keyword_hits_only": str(kw_hit_path),
            "time_priority_top": str(top_path),
            "type_relevant": str(type_rel_path),
            "request_failures_log": (
                str(failure_log_path)
                if int(failure_stats.get("total_failure_events") or 0) > 0
                else None
            ),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[save] {all_path}")
    print(f"[save] {final_cand_path}")
    print(f"[save] {kw_hit_path}")
    print(f"[save] {top_path}")
    print(f"[save] {type_rel_path}")
    print(f"[save] {meta_path}")


if __name__ == "__main__":
    main()
