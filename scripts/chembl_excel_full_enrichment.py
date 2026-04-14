"""
从 Excel 读取每行 SMILES + 名称，在 ChEMBL 中解析主分子（InChIKey / 名称别名 / 忽略大小写），
并拉取相似度最高的 3 个分子；主分子与相似分子均保存 **完整** molecule.json 内容。

相似 Top3 每条记录带 `source_name`（与 Excel `name` 列一致），用原样本名称与主表关联，不生成额外 ID。

名称例外（在 NAME_QUERY_ALIASES 中配置，可按需扩展）：
- TEMSIROLIMUS INJECTION → 先试 TEMSIROLIMUS（去掉制剂词）
- GLYCOPYRROLATE → 试 GLYCOPYRRONIUM BROMIDE / GLYCOPYRRONIUM 等
- 多组分 SMILES（含 '.'）→ InChIKey 先试整体，再按片段（默认优先原子数最多的片段）

用法（在仓库根目录 cardiotoxicity_prediction 下）:

  python -m data_augmentation.Chembl_data.chembl_excel_full_enrichment ^
    --excel data_augmentation/input/DIQTA阴性样本为主划分.xlsx

默认输出: data_augmentation/Chembl_data/output/full_enrichment/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None  # type: ignore[misc, assignment]

ROOT = Path(__file__).resolve().parents[2]
# 本包目录（Chembl_data），默认输出见 output/full_enrichment
CHEMBL_DATA_DIR = Path(__file__).resolve().parent

try:
    from rdkit import Chem
    from rdkit.Chem import inchi
except ImportError as e:
    raise SystemExit(
        "需要安装 RDKit：pip install rdkit\n" + str(e)
    ) from e

API_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# 主分子解析三层含义说明（写入 resolution 日志 / meta）
RESOLUTION_LAYERS_EXPLANATION = """主分子解析顺序与含义（与代码一致）:
1) full（整体 InChIKey）: 用整条 canonical SMILES 生成 InChIKey，与 ChEMBL 注册结构直接比对。优先使用：结构唯一、与「一条 SMILES 代表一个分子」的假设一致。
2) fragment（片段 InChIKey）: 当 SMILES 含多组分（如盐、溶剂、水合物，用 '.' 分隔）时，整体 InChIKey 可能无法在 ChEMBL 单条记录中命中；按片段（通常重原子数优先）尝试 InChIKey。意义：把「制剂/混合物」拆成可注册的母体或离子形式。
3) name（名称）: 若结构仍无法通过 InChIKey 对齐，用 Excel 名称（及别名，如盐型、去制剂词）查 pref_name。意义：兜底，处理 SMILES 与 ChEMBL 录入不一致、或盐型/互变异构等导致 InChIKey 不一致的情况；依赖名称与真实样品一致，需人工核对。
统计中的 none 表示未解析到主分子；invalid_smiles 表示 SMILES 无法被 RDKit 解析。"""


def _json_default(o: Any) -> Any:
    """pandas/numpy 标量（如 int64）与 ChEMBL 嵌套结构可被 json 写出。"""
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if np is not None:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            f = float(o)
            return f if math.isfinite(f) else None
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def dumps_json(obj: Any, **kwargs: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_json_default, **kwargs)


def chembl_similarity_search(
    smiles: str,
    *,
    min_similarity: float,
    max_pages: int = 5,
    page_size: int = 50,
    timeout_s: float = 45.0,
) -> list[tuple[float, str, str]]:
    """
    返回 (tanimoto, molecule_chembl_id, smiles) 列表。
    与 core.plugins.chembl_similarity_search 等价，但不经 plugins 导入（避免无 RDKit 时仅因 import 失败）。
    """
    if not smiles:
        return []
    out: list[tuple[float, str, str]] = []
    base = f"{API_BASE}/similarity"
    for page in range(0, max_pages):
        params = {
            "smiles": smiles,
            "similarity": int(min_similarity * 100),
            "limit": page_size,
            "offset": page * page_size,
            "format": "json",
        }
        try:
            r = requests.get(base, params=params, timeout=timeout_s)
            if r.status_code != 200:
                break
            data = r.json()
        except Exception:
            break
        items = data.get("molecules") or data.get("molecule") or data.get("similarities") or []
        if isinstance(items, dict):
            items = items.get("molecule", []) or items.get("molecules", []) or []
        if not items:
            break
        for it in items:
            chembl_id = str(it.get("molecule_chembl_id") or it.get("chembl_id") or "")
            smi = (
                it.get("canonical_smiles")
                or (it.get("molecule_structures") or {}).get("canonical_smiles")
                or ""
            )
            sim = it.get("similarity") or it.get("tanimoto") or it.get("score")
            try:
                sim_f = float(sim) / (100.0 if float(sim) > 1.5 else 1.0)
            except Exception:
                sim_f = float(min_similarity)
            if smi and chembl_id:
                out.append((sim_f, chembl_id, smi))
        if len(items) < page_size:
            break
    return out

# Excel 名称（整行 strip 后 upper）→ 按顺序尝试的 pref_name__iexact 查询串
NAME_QUERY_ALIASES: dict[str, list[str]] = {
    "TEMSIROLIMUS INJECTION": [
        "TEMSIROLIMUS",
        "TEMSIROLIMUS INJECTION",
    ],
    "GLYCOPYRROLATE": [
        "GLYCOPYRRONIUM BROMIDE",
        "GLYCOPYRRONIUM",
        "GLYCOPYRROLATE",
    ],
}


def _sleep(delay: float) -> None:
    time.sleep(max(0.0, delay))


def _get_molecules_payload(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        m = data.get("molecules") or data.get("molecule")
        if isinstance(m, list):
            return [x for x in m if isinstance(x, dict)]
        if isinstance(m, dict):
            return [m]
    return []


def canonicalize_smiles(s: str) -> Optional[str]:
    if not s or not str(s).strip():
        return None
    s = str(s).strip()
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def inchikey_from_smiles(s: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    try:
        return inchi.MolToInchiKey(mol)
    except Exception:
        return None


def split_smiles_frags(s: str) -> list[str]:
    s = str(s).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(".") if p.strip()]
    return parts if parts else [s]


def frag_heavy_atoms(smiles: str) -> int:
    m = Chem.MolFromSmiles(smiles)
    return m.GetNumHeavyAtoms() if m else 0


def ordered_frags_for_lookup(canonical_full: str) -> list[str]:
    """多组分时：整体优先；若需按片段试 InChIKey，则按重原子数降序。"""
    frags = split_smiles_frags(canonical_full)
    if len(frags) <= 1:
        return frags
    ranked = sorted(frags, key=lambda f: -frag_heavy_atoms(f))
    # 去重保持顺序
    seen: set[str] = set()
    out: list[str] = []
    for f in ranked:
        c = canonicalize_smiles(f)
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def name_query_sequence(excel_name: str) -> list[str]:
    raw = str(excel_name).strip()
    if not raw:
        return []
    key = raw.upper()
    if key in NAME_QUERY_ALIASES:
        seq = list(NAME_QUERY_ALIASES[key])
        if raw not in seq:
            seq.append(raw)
        return seq
    return [raw]


def pick_name_hits(
    mols: list[dict[str, Any]], excel_name: str, qname: str
) -> list[dict[str, Any]]:
    """
    先取 pref_name 与 Excel 一致的；若无，再取与本次查询串 qname 一致的（别名链，如 TEMSIROLIMUS）。
    """
    ex = str(excel_name or "").strip().lower()
    qn = str(qname or "").strip().lower()
    for m in mols:
        pn = str(m.get("pref_name", "") or "").strip().lower()
        if pn and pn == ex:
            return [m]
    for m in mols:
        pn = str(m.get("pref_name", "") or "").strip().lower()
        if pn and pn == qn:
            return [m]
    return []


def fetch_molecule_by_inchikey(ik: str, *, timeout: float) -> Optional[dict[str, Any]]:
    ik = str(ik or "").strip()
    if not ik:
        return None
    try:
        r = requests.get(f"{API_BASE}/molecule/{ik}.json", timeout=timeout)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def fetch_molecule_by_id(chembl_id: str, *, timeout: float) -> Optional[dict[str, Any]]:
    cid = str(chembl_id or "").strip()
    if not cid:
        return None
    try:
        r = requests.get(f"{API_BASE}/molecule/{cid}.json", timeout=timeout)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def query_pref_name_iexact(
    name: str, *, limit: int, timeout: float
) -> tuple[list[dict[str, Any]], Optional[str]]:
    name = name.strip()
    if not name:
        return [], None
    url = f"{API_BASE}/molecule.json"
    try:
        r = requests.get(url, params={"pref_name__iexact": name, "limit": limit}, timeout=timeout)
    except requests.RequestException as e:
        return [], str(e)
    if r.status_code != 200:
        return [], f"http_{r.status_code}"
    try:
        data = r.json()
    except Exception as e:
        return [], f"json:{e}"
    return _get_molecules_payload(data), None


# --- httpx 异步 API（默认路径；与同步逻辑等价） ---


async def fetch_molecule_by_inchikey_async(
    client: Any, ik: str
) -> Optional[dict[str, Any]]:
    ik = str(ik or "").strip()
    if not ik:
        return None
    try:
        r = await client.get(f"{API_BASE}/molecule/{ik}.json")
    except Exception:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    return data if isinstance(data, dict) else None


async def fetch_molecule_by_id_async(client: Any, chembl_id: str) -> Optional[dict[str, Any]]:
    cid = str(chembl_id or "").strip()
    if not cid:
        return None
    try:
        r = await client.get(f"{API_BASE}/molecule/{cid}.json")
    except Exception:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    return data if isinstance(data, dict) else None


async def query_pref_name_iexact_async(
    client: Any, name: str, *, limit: int
) -> tuple[list[dict[str, Any]], Optional[str]]:
    name = name.strip()
    if not name:
        return [], None
    url = f"{API_BASE}/molecule.json"
    try:
        r = await client.get(url, params={"pref_name__iexact": name, "limit": limit})
    except Exception as e:
        return [], str(e)
    if r.status_code != 200:
        return [], f"http_{r.status_code}"
    try:
        data = r.json()
    except Exception as e:
        return [], f"json:{e}"
    return _get_molecules_payload(data), None


async def chembl_similarity_search_async(
    client: Any,
    smiles: str,
    *,
    min_similarity: float,
    max_pages: int = 5,
    page_size: int = 50,
) -> list[tuple[float, str, str]]:
    if not smiles:
        return []
    out: list[tuple[float, str, str]] = []
    base = f"{API_BASE}/similarity"
    for page in range(0, max_pages):
        params = {
            "smiles": smiles,
            "similarity": int(min_similarity * 100),
            "limit": page_size,
            "offset": page * page_size,
            "format": "json",
        }
        try:
            r = await client.get(base, params=params)
            if r.status_code != 200:
                break
            data = r.json()
        except Exception:
            break
        items = data.get("molecules") or data.get("molecule") or data.get("similarities") or []
        if isinstance(items, dict):
            items = items.get("molecule", []) or items.get("molecules", []) or []
        if not items:
            break
        for it in items:
            chembl_id = str(it.get("molecule_chembl_id") or it.get("chembl_id") or "")
            smi = (
                it.get("canonical_smiles")
                or (it.get("molecule_structures") or {}).get("canonical_smiles")
                or ""
            )
            sim = it.get("similarity") or it.get("tanimoto") or it.get("score")
            try:
                sim_f = float(sim) / (100.0 if float(sim) > 1.5 else 1.0)
            except Exception:
                sim_f = float(min_similarity)
            if smi and chembl_id:
                out.append((sim_f, chembl_id, smi))
        if len(items) < page_size:
            break
    return out


async def resolve_primary_molecule_async(
    client: Any,
    *,
    excel_name: str,
    canonical_smiles: str,
    delay_s: float,
    name_limit: int,
) -> tuple[Optional[dict[str, Any]], str, list[str], str]:
    notes: list[str] = []
    ikeys_tried: list[str] = []

    async def try_inchikey(ik: str, tag: str) -> Optional[dict[str, Any]]:
        if ik in ikeys_tried:
            return None
        ikeys_tried.append(ik)
        await asyncio.sleep(max(0.0, delay_s))
        mol = await fetch_molecule_by_inchikey_async(client, ik)
        if mol:
            notes.append(f"inchikey:{tag}:{ik}")
        return mol

    ik_full = inchikey_from_smiles(canonical_smiles)
    if ik_full:
        m = await try_inchikey(ik_full, "full")
        if m:
            return m, ";".join(notes), ikeys_tried, "full"

    for frag in ordered_frags_for_lookup(canonical_smiles):
        ik = inchikey_from_smiles(frag)
        if not ik:
            continue
        m = await try_inchikey(ik, "fragment")
        if m:
            return m, ";".join(notes), ikeys_tried, "fragment"

    for qname in name_query_sequence(excel_name):
        await asyncio.sleep(max(0.0, delay_s))
        mols, err = await query_pref_name_iexact_async(client, qname, limit=name_limit)
        if err:
            notes.append(f"name_err:{qname}:{err}")
            continue
        hits = pick_name_hits(mols, excel_name, qname)
        if hits:
            cid = str(hits[0].get("molecule_chembl_id", "") or "")
            notes.append(f"name_match:{qname}→{cid}")
            await asyncio.sleep(max(0.0, delay_s))
            full = await fetch_molecule_by_id_async(client, cid)
            if full:
                return full, ";".join(notes), ikeys_tried, "name"
            return hits[0], ";".join(notes), ikeys_tried, "name"

    return (
        None,
        "not_resolved:" + ";".join(notes) if notes else "not_resolved",
        ikeys_tried,
        "none",
    )


async def top_similar_full_molecules_async(
    client: Any,
    query_smiles: str,
    *,
    primary_chembl_id: str,
    top_k: int,
    min_similarity: float,
    delay_s: float,
    source_name: str = "",
) -> list[dict[str, Any]]:
    hits = await chembl_similarity_search_async(
        client,
        query_smiles,
        min_similarity=min_similarity,
        max_pages=5,
        page_size=50,
    )
    pid = str(primary_chembl_id or "").strip().upper()
    ranked: list[tuple[float, str, str]] = []
    seen: set[str] = set()
    for sim_f, cid, smi in hits:
        cid = str(cid or "").strip().upper()
        if not cid or cid in seen:
            continue
        if pid and cid == pid:
            continue
        seen.add(cid)
        ranked.append((float(sim_f), cid, smi))
    ranked.sort(key=lambda x: -x[0])

    src = str(source_name or "").strip()
    top = ranked[:top_k]
    if not top:
        return []

    async def one(rank: int, sim: float, cid: str, smi: str) -> dict[str, Any]:
        await asyncio.sleep(max(0.0, delay_s))
        full = await fetch_molecule_by_id_async(client, cid)
        return {
            "rank": rank,
            "tanimoto": sim,
            "query_smiles": query_smiles,
            "similar_smiles_from_api": smi,
            "molecule_chembl_id": cid,
            "molecule_full": full,
            "source_name": src,
        }

    coros = [
        one(rank, sim, cid, smi)
        for rank, (sim, cid, smi) in enumerate(top, start=1)
    ]
    return list(await asyncio.gather(*coros))


def resolve_primary_molecule(
    *,
    excel_name: str,
    canonical_smiles: str,
    delay_s: float,
    name_limit: int,
    timeout: float,
) -> tuple[Optional[dict[str, Any]], str, list[str], str]:
    """
    返回 (完整 molecule dict 或 None, 说明字符串, 尝试过的 inchikey 列表, 命中层级)。

    命中层级 primary_resolution_kind:
    - full: 整条 canonical SMILES 的 InChIKey 在 ChEMBL 命中
    - fragment: 多组分等情况下，按片段 InChIKey 命中（整体未命中）
    - name: InChIKey 均未命中，靠 pref_name（含别名链）命中
    - none: 未解析到主分子
    """
    notes: list[str] = []
    ikeys_tried: list[str] = []

    def try_inchikey(ik: str, tag: str) -> Optional[dict[str, Any]]:
        if ik in ikeys_tried:
            return None
        ikeys_tried.append(ik)
        _sleep(delay_s)
        mol = fetch_molecule_by_inchikey(ik, timeout=timeout)
        if mol:
            notes.append(f"inchikey:{tag}:{ik}")
        return mol

    # 1) 整体 canonical SMILES → InChIKey
    ik_full = inchikey_from_smiles(canonical_smiles)
    if ik_full:
        m = try_inchikey(ik_full, "full")
        if m:
            return m, ";".join(notes), ikeys_tried, "full"

    # 2) 多组分：按片段 InChIKey
    for frag in ordered_frags_for_lookup(canonical_smiles):
        ik = inchikey_from_smiles(frag)
        if not ik:
            continue
        m = try_inchikey(ik, "fragment")
        if m:
            return m, ";".join(notes), ikeys_tried, "fragment"

    # 3) 名称：别名链 + pref_name 与 Excel 忽略大小写一致
    for qname in name_query_sequence(excel_name):
        _sleep(delay_s)
        mols, err = query_pref_name_iexact(qname, limit=name_limit, timeout=timeout)
        if err:
            notes.append(f"name_err:{qname}:{err}")
            continue
        hits = pick_name_hits(mols, excel_name, qname)
        if hits:
            cid = str(hits[0].get("molecule_chembl_id", "") or "")
            notes.append(f"name_match:{qname}→{cid}")
            _sleep(delay_s)
            full = fetch_molecule_by_id(cid, timeout=timeout)
            if full:
                return full, ";".join(notes), ikeys_tried, "name"
            return hits[0], ";".join(notes), ikeys_tried, "name"

    return (
        None,
        "not_resolved:" + ";".join(notes) if notes else "not_resolved",
        ikeys_tried,
        "none",
    )


def smiles_for_similarity(canonical_full: str) -> str:
    """相似度查询用单组分 SMILES（多组分时取重原子最多且可解析的片段）。"""
    frags = ordered_frags_for_lookup(canonical_full)
    if not frags:
        return canonical_full
    if len(frags) == 1:
        return frags[0]
    return frags[0]


def top_similar_full_molecules(
    query_smiles: str,
    *,
    primary_chembl_id: str,
    top_k: int,
    min_similarity: float,
    delay_s: float,
    timeout: float,
    source_name: str = "",
) -> list[dict[str, Any]]:
    hits = chembl_similarity_search(
        query_smiles,
        min_similarity=min_similarity,
        max_pages=5,
        page_size=50,
        timeout_s=timeout,
    )
    # 按 Tanimoto 降序，排除主分子
    pid = str(primary_chembl_id or "").strip().upper()
    ranked: list[tuple[float, str, str]] = []
    seen: set[str] = set()
    for sim_f, cid, smi in hits:
        cid = str(cid or "").strip().upper()
        if not cid or cid in seen:
            continue
        if pid and cid == pid:
            continue
        seen.add(cid)
        ranked.append((float(sim_f), cid, smi))
    ranked.sort(key=lambda x: -x[0])

    src = str(source_name or "").strip()
    out: list[dict[str, Any]] = []
    for rank, (sim, cid, smi) in enumerate(ranked[:top_k], start=1):
        _sleep(delay_s)
        full = fetch_molecule_by_id(cid, timeout=timeout)
        out.append(
            {
                "rank": rank,
                "tanimoto": sim,
                "query_smiles": query_smiles,
                "similar_smiles_from_api": smi,
                "molecule_chembl_id": cid,
                "molecule_full": full,
                "source_name": src,
            }
        )
    return out


async def process_one_row_async(
    i: int,
    client: Any,
    df: pd.DataFrame,
    args: Any,
    sem: asyncio.Semaphore,
    sm_col: str,
    nm_col: str,
) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """处理单行；返回 (行号, jsonl 记录, summary 行)。"""
    async with sem:
        row = df.iloc[i]
        excel_name = row.get(nm_col, "")
        raw_smiles = row.get(sm_col, "")
        if pd.isna(raw_smiles):
            raw_smiles = ""
        can = canonicalize_smiles(str(raw_smiles))
        rec: dict[str, Any] = {
            "excel_row_index": int(i),
            "pubchem_id": row.get("Pubchem_ID", ""),
            "excel_name": str(excel_name) if not pd.isna(excel_name) else "",
            "label": row.get("label", ""),
            "raw_smiles": str(raw_smiles).strip(),
            "canonical_smiles": can,
            "primary_molecule_full": None,
            "primary_resolution": "",
            "primary_resolution_kind": "",
            "inchikeys_tried": [],
            "similar_top_k": [],
            "errors": [],
        }

        if not can:
            rec["errors"].append("invalid_smiles")
            rec["primary_resolution_kind"] = "invalid_smiles"
            summary = {
                "excel_row_index": i,
                "name": rec["excel_name"],
                "canonical_smiles": "",
                "primary_chembl_id": "",
                "primary_pref_name": "",
                "resolution": "",
                "primary_resolution_kind": "invalid_smiles",
                "sim1_id": "",
                "sim1_tanimoto": "",
                "sim2_id": "",
                "sim2_tanimoto": "",
                "sim3_id": "",
                "sim3_tanimoto": "",
            }
            return i, rec, summary

        pm, res_note, iks, res_kind = await resolve_primary_molecule_async(
            client,
            excel_name=rec["excel_name"],
            canonical_smiles=can,
            delay_s=args.delay_s,
            name_limit=args.name_limit,
        )
        rec["primary_resolution"] = res_note
        rec["primary_resolution_kind"] = res_kind
        rec["inchikeys_tried"] = iks

        primary_id = ""
        if pm:
            await asyncio.sleep(max(0.0, args.delay_s))
            cid = str(pm.get("molecule_chembl_id", "") or "").strip()
            primary_id = cid
            full_pm = await fetch_molecule_by_id_async(client, cid) if cid else None
            rec["primary_molecule_full"] = full_pm or pm

        qsmi = smiles_for_similarity(can)
        sims: list[dict[str, Any]] = []
        try:
            sims = await top_similar_full_molecules_async(
                client,
                qsmi,
                primary_chembl_id=primary_id,
                top_k=args.top_k,
                min_similarity=args.min_similarity,
                delay_s=args.delay_s,
                source_name=rec["excel_name"],
            )
        except Exception as e:
            rec["errors"].append(f"similarity:{e}")
        rec["similar_top_k"] = sims

        pmf = rec["primary_molecule_full"] or {}
        summary = {
            "excel_row_index": i,
            "name": rec["excel_name"],
            "canonical_smiles": can,
            "primary_chembl_id": str(pmf.get("molecule_chembl_id", "") or ""),
            "primary_pref_name": str(pmf.get("pref_name", "") or ""),
            "resolution": res_note,
            "primary_resolution_kind": res_kind,
            "sim1_id": sims[0]["molecule_chembl_id"] if len(sims) > 0 else "",
            "sim1_tanimoto": sims[0]["tanimoto"] if len(sims) > 0 else "",
            "sim2_id": sims[1]["molecule_chembl_id"] if len(sims) > 1 else "",
            "sim2_tanimoto": sims[1]["tanimoto"] if len(sims) > 1 else "",
            "sim3_id": sims[2]["molecule_chembl_id"] if len(sims) > 2 else "",
            "sim3_tanimoto": sims[2]["tanimoto"] if len(sims) > 2 else "",
        }
        return i, rec, summary


def write_resolution_log_and_meta(
    *,
    out_dir: Path,
    excel_path: Path,
    jsonl_path: Path,
    summary_path: Path,
    summary_rows: list[dict[str, Any]],
    stats: dict[str, int],
    args: Any,
    extra_meta: Optional[dict[str, Any]] = None,
) -> None:
    resolution_log_path = out_dir / "chembl_excel_full_enrichment_resolution.log"
    resolution_stats = {
        "rows_written": len(summary_rows),
        "invalid_smiles": stats["invalid_smiles"],
        "primary_by_full_inchikey": stats["primary_full"],
        "primary_by_fragment_inchikey": stats["primary_fragment"],
        "primary_by_name": stats["primary_name"],
        "primary_unresolved": stats["primary_none"],
    }
    log_body = (
        RESOLUTION_LAYERS_EXPLANATION
        + "\n\n--- 本次运行统计 ---\n"
        + f"总行数（写入）: {resolution_stats['rows_written']}\n"
        + f"无效 SMILES: {resolution_stats['invalid_smiles']}\n"
        + f"主分子-整体 InChIKey (full): {resolution_stats['primary_by_full_inchikey']}\n"
        + f"主分子-片段 InChIKey (fragment): {resolution_stats['primary_by_fragment_inchikey']}\n"
        + f"主分子-名称 (name): {resolution_stats['primary_by_name']}\n"
        + f"主分子-未解析 (none): {resolution_stats['primary_unresolved']}\n"
    )
    resolution_log_path.write_text(log_body, encoding="utf-8")

    meta: dict[str, Any] = {
        "excel": str(excel_path),
        "rows_written": len(summary_rows),
        "jsonl": str(jsonl_path),
        "summary_csv": str(summary_path),
        "resolution_log": str(resolution_log_path),
        "min_similarity": args.min_similarity,
        "top_k": args.top_k,
        "linkage": "similar_top_k[].source_name == Excel name 列，与原样本按名称对应",
        "resolution_stats": resolution_stats,
        "resolution_explanation": RESOLUTION_LAYERS_EXPLANATION.strip(),
    }
    if extra_meta:
        meta.update(extra_meta)
    (out_dir / "chembl_excel_full_enrichment_meta.json").write_text(
        dumps_json(meta, indent=2),
        encoding="utf-8",
    )
    print(dumps_json(meta, indent=2))
    print(f"[resolution log] {resolution_log_path}", flush=True)


async def run_async_enrichment(
    args: Any,
    excel_path: Path,
    out_dir: Path,
    df: pd.DataFrame,
    sm_col: str,
    nm_col: str,
    rows_range: range,
    jsonl_path: Path,
    summary_path: Path,
) -> None:
    if httpx is None:
        raise SystemExit("异步模式需要安装 httpx：pip install httpx")

    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(max_connections=max(32, args.concurrency * 4))
    sem = asyncio.Semaphore(max(1, args.concurrency))
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        coros = [
            process_one_row_async(i, client, df, args, sem, sm_col, nm_col)
            for i in rows_range
        ]
        try:
            from tqdm.asyncio import tqdm as tqdm_asyncio

            if not args.no_progress:
                results: list[Any] = await tqdm_asyncio.gather(
                    *coros,
                    desc="ChEMBL 全量(异步)",
                    unit="行",
                )
            else:
                results = await asyncio.gather(*coros)
        except Exception:
            results = await asyncio.gather(*coros)

    ordered = sorted(results, key=lambda x: x[0])
    stats: dict[str, int] = {
        "invalid_smiles": 0,
        "primary_full": 0,
        "primary_fragment": 0,
        "primary_name": 0,
        "primary_none": 0,
    }
    for _, rec, _ in ordered:
        rk = str(rec.get("primary_resolution_kind", "") or "")
        if rk == "invalid_smiles":
            stats["invalid_smiles"] += 1
        elif rk == "full":
            stats["primary_full"] += 1
        elif rk == "fragment":
            stats["primary_fragment"] += 1
        elif rk == "name":
            stats["primary_name"] += 1
        else:
            stats["primary_none"] += 1

    summary_rows = [s for _, _, s in ordered]
    with open(jsonl_path, "w", encoding="utf-8") as fj:
        for _, rec, _ in ordered:
            fj.write(dumps_json(rec) + "\n")

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    write_resolution_log_and_meta(
        out_dir=out_dir,
        excel_path=excel_path,
        jsonl_path=jsonl_path,
        summary_path=summary_path,
        summary_rows=summary_rows,
        stats=stats,
        args=args,
        extra_meta={
            "http_mode": "async",
            "async_concurrency": args.concurrency,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Excel → ChEMBL 主分子 + 相似 Top3（完整 JSON）")
    p.add_argument("--excel", type=Path, default=ROOT / "data_augmentation" / "input" / "DIQTA阴性样本为主划分.xlsx")
    p.add_argument("--sheet", default=None)
    p.add_argument("--smiles-column", default="NEW SMILES")
    p.add_argument("--name-column", default="name")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--delay-s", type=float, default=0.2)
    p.add_argument("--timeout", type=float, default=45.0)
    p.add_argument("--name-limit", type=int, default=25)
    p.add_argument("--min-similarity", type=float, default=0.45, help="ChEMBL similarity 阈值 0~1")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--max-rows", type=int, default=None, help="仅处理前 N 行（调试）")
    p.add_argument("--start-row", type=int, default=0, help="Excel 行偏移（0=首行数据）")
    p.add_argument("--no-progress", action="store_true", help="禁用 tqdm 进度条")
    p.add_argument(
        "--sync",
        action="store_true",
        help="禁用异步，改用同步 requests（更慢，不依赖 httpx）",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="异步模式下同时处理的行数上限（信号量，避免压垮 API）",
    )
    args = p.parse_args()

    excel_path = Path(args.excel).expanduser().resolve()
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else CHEMBL_DATA_DIR / "output" / "full_enrichment"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet: Any = 0 if args.sheet is None else args.sheet
    df = pd.read_excel(excel_path, sheet_name=sheet)

    sm_col = args.smiles_column
    nm_col = args.name_column
    if sm_col not in df.columns:
        raise SystemExit(f"缺少列 {sm_col!r}，当前列: {list(df.columns)}")
    if nm_col not in df.columns:
        raise SystemExit(f"缺少列 {nm_col!r}")

    n = len(df)
    end = n if args.max_rows is None else min(n, args.start_row + args.max_rows)
    rows_range = range(args.start_row, end)

    jsonl_path = out_dir / "chembl_excel_full_enrichment.jsonl"
    summary_path = out_dir / "chembl_excel_full_enrichment_summary.csv"

    if not args.sync:
        asyncio.run(
            run_async_enrichment(
                args,
                excel_path,
                out_dir,
                df,
                sm_col,
                nm_col,
                rows_range,
                jsonl_path,
                summary_path,
            )
        )
        return

    summary_rows: list[dict[str, Any]] = []

    stats: dict[str, int] = {
        "invalid_smiles": 0,
        "primary_full": 0,
        "primary_fragment": 0,
        "primary_name": 0,
        "primary_none": 0,
    }

    total_rows = end - args.start_row
    pbar: Any = None
    if _tqdm is not None and not args.no_progress:
        pbar = _tqdm(
            rows_range,
            total=total_rows,
            desc="ChEMBL 全量",
            unit="行",
            dynamic_ncols=True,
        )
    elif not args.no_progress:
        print(
            "[提示] 未安装 tqdm，请 pip install tqdm 以显示进度条；当前将静默运行。",
            flush=True,
        )

    def _iter_rows():
        if pbar is not None:
            yield from pbar
        else:
            yield from rows_range

    with open(jsonl_path, "w", encoding="utf-8") as fj:
        for i in _iter_rows():
            row = df.iloc[i]
            excel_name = row.get(nm_col, "")
            raw_smiles = row.get(sm_col, "")
            if pd.isna(raw_smiles):
                raw_smiles = ""
            can = canonicalize_smiles(str(raw_smiles))
            rec: dict[str, Any] = {
                "excel_row_index": int(i),
                "pubchem_id": row.get("Pubchem_ID", ""),
                "excel_name": str(excel_name) if not pd.isna(excel_name) else "",
                "label": row.get("label", ""),
                "raw_smiles": str(raw_smiles).strip(),
                "canonical_smiles": can,
                "primary_molecule_full": None,
                "primary_resolution": "",
                "primary_resolution_kind": "",
                "inchikeys_tried": [],
                "similar_top_k": [],
                "errors": [],
            }

            if not can:
                stats["invalid_smiles"] += 1
                rec["errors"].append("invalid_smiles")
                rec["primary_resolution_kind"] = "invalid_smiles"
                fj.write(dumps_json(rec) + "\n")
                summary_rows.append(
                    {
                        "excel_row_index": i,
                        "name": rec["excel_name"],
                        "canonical_smiles": "",
                        "primary_chembl_id": "",
                        "primary_pref_name": "",
                        "resolution": "",
                        "primary_resolution_kind": "invalid_smiles",
                        "sim1_id": "",
                        "sim1_tanimoto": "",
                        "sim2_id": "",
                        "sim2_tanimoto": "",
                        "sim3_id": "",
                        "sim3_tanimoto": "",
                    }
                )
                if pbar is not None:
                    nm = str(rec.get("excel_name", "") or "")
                    pbar.set_postfix_str(nm[:40] + ("…" if len(nm) > 40 else ""), refresh=True)
                continue

            pm, res_note, iks, res_kind = resolve_primary_molecule(
                excel_name=rec["excel_name"],
                canonical_smiles=can,
                delay_s=args.delay_s,
                name_limit=args.name_limit,
                timeout=args.timeout,
            )
            rec["primary_resolution"] = res_note
            rec["primary_resolution_kind"] = res_kind
            rec["inchikeys_tried"] = iks

            if res_kind == "full":
                stats["primary_full"] += 1
            elif res_kind == "fragment":
                stats["primary_fragment"] += 1
            elif res_kind == "name":
                stats["primary_name"] += 1
            else:
                stats["primary_none"] += 1

            primary_id = ""
            if pm:
                _sleep(args.delay_s)
                cid = str(pm.get("molecule_chembl_id", "") or "").strip()
                primary_id = cid
                full_pm = fetch_molecule_by_id(cid, timeout=args.timeout) if cid else None
                rec["primary_molecule_full"] = full_pm or pm

            qsmi = smiles_for_similarity(can)
            sims: list[dict[str, Any]] = []
            try:
                sims = top_similar_full_molecules(
                    qsmi,
                    primary_chembl_id=primary_id,
                    top_k=args.top_k,
                    min_similarity=args.min_similarity,
                    delay_s=args.delay_s,
                    timeout=args.timeout,
                    source_name=rec["excel_name"],
                )
            except Exception as e:
                rec["errors"].append(f"similarity:{e}")
            rec["similar_top_k"] = sims

            fj.write(dumps_json(rec) + "\n")

            pmf = rec["primary_molecule_full"] or {}
            summary_rows.append(
                {
                    "excel_row_index": i,
                    "name": rec["excel_name"],
                    "canonical_smiles": can,
                    "primary_chembl_id": str(pmf.get("molecule_chembl_id", "") or ""),
                    "primary_pref_name": str(pmf.get("pref_name", "") or ""),
                    "resolution": res_note,
                    "primary_resolution_kind": res_kind,
                    "sim1_id": sims[0]["molecule_chembl_id"] if len(sims) > 0 else "",
                    "sim1_tanimoto": sims[0]["tanimoto"] if len(sims) > 0 else "",
                    "sim2_id": sims[1]["molecule_chembl_id"] if len(sims) > 1 else "",
                    "sim2_tanimoto": sims[1]["tanimoto"] if len(sims) > 1 else "",
                    "sim3_id": sims[2]["molecule_chembl_id"] if len(sims) > 2 else "",
                    "sim3_tanimoto": sims[2]["tanimoto"] if len(sims) > 2 else "",
                }
            )

            if pbar is not None:
                nm = str(rec.get("excel_name", "") or "")
                pbar.set_postfix_str(nm[:40] + ("…" if len(nm) > 40 else ""), refresh=True)

            if _tqdm is None and not args.no_progress and (i - args.start_row + 1) % 25 == 0:
                print(f"[progress] {i - args.start_row + 1}/{total_rows}", flush=True)

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    write_resolution_log_and_meta(
        out_dir=out_dir,
        excel_path=excel_path,
        jsonl_path=jsonl_path,
        summary_path=summary_path,
        summary_rows=summary_rows,
        stats=stats,
        args=args,
        extra_meta={"http_mode": "sync"},
    )


if __name__ == "__main__":
    main()
