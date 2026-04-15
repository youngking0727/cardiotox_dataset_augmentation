"""
SMILES/InChIKey 直连 ChEMBL 未命中的行，用 Excel 里的「名称」再查一次。

规则：
- 仅当 ChEMBL 的 **pref_name** 与 Excel 中的名称 **字符串完全一致** 才算 hit（默认大小写也一致）。
- 只使用 `molecule?pref_name__iexact=` 拉候选，**不使用** molecule/search 模糊检索。
- 候选拉回后再本地比对 `pref_name` 与 Excel 名称。

用法（仓库根目录 cardiotoxicity_prediction）:

  python -m data_augmentation.Chembl_data.chembl_name_retry_unmatched

默认 per_row CSV 与输出目录: Chembl_data/output/registry_report/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None  # type: ignore[misc, assignment]

ROOT = Path(__file__).resolve().parents[2]
CHEMBL_DATA_DIR = Path(__file__).resolve().parent
DEFAULT_EXCEL = ROOT / "data_augmentation" / "input" / "DIQTA阴性样本为主划分.xlsx"
DEFAULT_PER_ROW = (
    CHEMBL_DATA_DIR / "output" / "registry_report" / "chembl_inchikey_per_row.csv"
)

_LOG = logging.getLogger("chembl_name_retry_unmatched")
API_BASE = "https://www.ebi.ac.uk/chembl/api/data"


def _setup_logging(path: Optional[Path], verbose: bool) -> None:
    _LOG.handlers.clear()
    _LOG.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(logging.DEBUG if verbose else logging.INFO)
        fh.setFormatter(fmt)
        _LOG.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.WARNING)
    ch.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(ch)


def _pick_name(row: pd.Series, cols: list[str]) -> tuple[str, str]:
    for c in cols:
        if c not in row.index:
            continue
        v = row[c]
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            return s, c
    return "", ""


def _get_molecules_payload(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        m = data.get("molecules") or data.get("molecule")
        if isinstance(m, list):
            return [x for x in m if isinstance(x, dict)]
        if isinstance(m, dict):
            return [m]
    return []


def query_pref_name_iexact(
    name: str, *, limit: int = 25, timeout: float = 30.0
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


def fetch_molecule_by_id(chembl_id: str, *, timeout: float = 30.0) -> Optional[dict[str, Any]]:
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


def _digest(mol: dict[str, Any]) -> dict[str, Any]:
    return {
        "molecule_chembl_id": mol.get("molecule_chembl_id", ""),
        "pref_name": mol.get("pref_name", "") or "",
        "standard_inchi_key": mol.get("standard_inchi_key", "") or "",
    }


def enrich_inchikey(d: dict[str, Any], *, delay_s: float, timeout: float) -> dict[str, Any]:
    if not str(d.get("standard_inchi_key", "")).strip() and d.get("molecule_chembl_id"):
        time.sleep(max(0.0, delay_s))
        full = fetch_molecule_by_id(str(d["molecule_chembl_id"]), timeout=timeout)
        if full:
            d = dict(d)
            d["standard_inchi_key"] = str(full.get("standard_inchi_key", "") or "")
    return d


def filter_pref_name_exact(
    mols: list[dict[str, Any]],
    excel_name: str,
    *,
    ignore_case: bool,
) -> list[dict[str, Any]]:
    ex = excel_name.strip()
    if not ex:
        return []
    out: list[dict[str, Any]] = []
    for m in mols:
        pn = str(m.get("pref_name", "") or "").strip()
        if ignore_case:
            if pn.lower() == ex.lower():
                out.append(m)
        elif pn == ex:
            out.append(m)
    return out


def _as_bool_in_chmbl(x: Any) -> bool:
    if pd.isna(x):
        return False
    return str(x).strip().lower() in ("true", "1", "yes")


def main() -> None:
    p = argparse.ArgumentParser(description="InChIKey 未命中：pref_name 与 Excel 名称须完全一致")
    p.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    p.add_argument("--sheet", type=str, default=None)
    p.add_argument("--per-row-csv", type=Path, default=DEFAULT_PER_ROW)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--name-columns",
        type=str,
        default="name,Name,名称,化合物名称,drug_name,pref_name",
    )
    p.add_argument("--delay-s", type=float, default=0.15)
    p.add_argument("--limit", type=int, default=25)
    p.add_argument(
        "--ignore-case",
        action="store_true",
        help="pref_name 与 Excel 名称忽略大小写（默认大小写也须一致）",
    )
    p.add_argument(
        "--audit",
        action="store_true",
        help="在 chembl_name_exact_all_unmatched_rows.csv 中增加 api_candidates_json 列（iexact 原始列表）",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--no-log-file", action="store_true")
    p.add_argument("--no-progress", action="store_true", help="禁用 tqdm 进度条")
    args = p.parse_args()

    excel_path = Path(args.excel).expanduser().resolve()
    per_row_path = Path(args.per_row_csv).expanduser().resolve()
    if not excel_path.is_file():
        raise SystemExit(f"Excel 不存在: {excel_path}")
    if not per_row_path.is_file():
        raise SystemExit(f"per_row CSV 不存在: {per_row_path}")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else CHEMBL_DATA_DIR / "output" / "registry_report"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = None if args.no_log_file else out_dir / "chembl_name_retry.log"
    _setup_logging(log_path, args.verbose)
    _LOG.info(f"excel={excel_path} per_row={per_row_path} ignore_case={args.ignore_case}")

    if args.sheet is None:
        sheet_param: Any = 0
    elif isinstance(args.sheet, str) and args.sheet.isdigit():
        sheet_param = int(args.sheet)
    else:
        sheet_param = args.sheet

    df_x = pd.read_excel(excel_path, sheet_name=sheet_param)
    df_p = pd.read_csv(per_row_path, encoding="utf-8-sig")
    if "excel_row_index" not in df_p.columns or "in_chembl" not in df_p.columns:
        raise SystemExit("per_row CSV 需含 excel_row_index、in_chembl")

    unmatched = df_p[~df_p["in_chembl"].map(_as_bool_in_chmbl)].copy()
    name_cols = [c.strip() for c in args.name_columns.split(",") if c.strip()]

    nu = len(unmatched)
    pbar_nm: Any = None
    row_iter: Any = unmatched.iterrows()
    if _tqdm is not None and not args.no_progress:
        row_iter = _tqdm(
            row_iter,
            total=nu,
            desc="ChEMBL 名称重试",
            unit="行",
            dynamic_ncols=True,
        )
        pbar_nm = row_iter
    elif not args.no_progress:
        print(
            "[提示] 未安装 tqdm，请 pip install tqdm 以显示进度条。",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    for _, pr in row_iter:
        try:
            ri = int(pr["excel_row_index"])
        except Exception:
            _LOG.warning(f"bad excel_row_index: {pr.get('excel_row_index')}")
            continue
        if ri < 0 or ri >= len(df_x):
            _LOG.warning(f"row out of range: {ri}")
            continue

        row_x = df_x.iloc[ri]
        excel_name, name_src = _pick_name(row_x, name_cols)
        time.sleep(max(0.0, float(args.delay_s)))

        raw_mols: list[dict[str, Any]] = []
        err: Optional[str] = None
        if excel_name:
            raw_mols, err = query_pref_name_iexact(excel_name, limit=args.limit)

        exact_mols = filter_pref_name_exact(raw_mols, excel_name, ignore_case=bool(args.ignore_case))
        digests = []
        for m in exact_mols:
            d = _digest(m)
            d = enrich_inchikey(d, delay_s=float(args.delay_s), timeout=30.0)
            digests.append(d)

        rec: dict[str, Any] = {
            "excel_row_index": ri,
            "inchikey_status": str(pr.get("status", "")),
            "excel_name": excel_name,
            "excel_name_column": name_src,
            "standard_inchi_key_from_smiles": str(pr.get("standard_inchi_key", "") or ""),
            "n_api_candidates": len(raw_mols),
            "n_name_exact_hits": len(digests),
            "name_exact_hits_json": json.dumps(digests, ensure_ascii=False),
            "api_query_error": err or "",
            "name_match_rule": "pref_name equals excel name"
            + (" (ignore case)" if args.ignore_case else " (case sensitive)"),
        }
        if args.audit:
            rec["api_candidates_json"] = json.dumps([_digest(m) for m in raw_mols], ensure_ascii=False)
        rows.append(rec)

        if pbar_nm is not None:
            en = str(excel_name or "")
            pbar_nm.set_postfix_str(en[:40] + ("…" if len(en) > 40 else ""), refresh=True)

        en_repr = repr(excel_name[:60] if excel_name else "")
        _LOG.info(
            f"row={ri} excel_name={en_repr} api_n={len(raw_mols)} exact_n={len(digests)}"
        )

    pdf = pd.DataFrame(rows)
    out_hits = out_dir / "chembl_name_exact_hits.csv"
    pdf[pdf["n_name_exact_hits"] > 0].to_csv(out_hits, index=False, encoding="utf-8-sig")

    out_all = out_dir / "chembl_name_exact_all_unmatched_rows.csv"
    pdf.to_csv(out_all, index=False, encoding="utf-8-sig")

    summary = {
        "inchikey_unmatched_rows": int(len(unmatched)),
        "rows_with_excel_name": int((pdf["excel_name"].astype(str).str.len() > 0).sum()),
        "rows_with_name_exact_hits": int((pdf["n_name_exact_hits"] > 0).sum()),
        "hits_only_csv": str(out_hits),
        "all_rows_csv": str(out_all),
        "method": "pref_name__iexact API + local strict pref_name == excel name; no molecule/search",
    }
    sum_path = out_dir / "chembl_name_retry_summary.json"
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[save] {out_hits}")
    print(f"[save] {out_all}")
    print(f"[save] {sum_path}")
    if log_path:
        print(f"[log] {log_path}")


if __name__ == "__main__":
    main()
