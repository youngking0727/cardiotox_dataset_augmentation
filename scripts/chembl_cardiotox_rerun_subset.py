"""
从完整 enrichment summary 中挑出指定 CHEMBL 分子 ID，生成小 CSV 后调用
chembl_cardiotox_activity_screen 只跑这一批。

用途：对「在 all_scored 里完全没有行」的分子单独重跑（例如某次全量跑后对比）。

说明（重要）：
  若某 CHEMBL ID 在 ChEMBL 上本来就没有任何 activity，Phase1 会得到 0 条记录，
  输出里仍不会出现该分子——这与 Phase2 assay 拉取失败无关；assay 失败只影响已有 activity 的 enrichment。

默认内置的 8 个 ID 来自 full_enrichment3 与 full_enrichment1 summary 对比时「无 activity 行」的分子。

在 cardiotoxicity_prediction 根目录执行:

  python -m data_augmentation.Chembl_data.chembl_cardiotox_rerun_subset ^
    --summary-csv data_augmentation/Chembl_data/output/full_enrichment1/chembl_excel_full_enrichment_summary.csv ^
    --out-dir data_augmentation/Chembl_data/output/full_enrichment3_rerun8

其它参数会原样传给主脚本，例如:

  python -m data_augmentation.Chembl_data.chembl_cardiotox_rerun_subset ^
    --summary-csv ... --out-dir ... -- --http-timeout-s 120 --phase1-concurrency 2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

# 与此前 full_enrichment3 统计中「summary 有、all_scored 无」的 8 个 CHEMBL ID
DEFAULT_NO_ACTIVITY_MOLECULES: tuple[str, ...] = (
    "CHEMBL1201290",
    "CHEMBL1201867",
    "CHEMBL2105745",
    "CHEMBL2364633",
    "CHEMBL3137326",
    "CHEMBL3305968",
    "CHEMBL4297516",
    "CHEMBL4597183",
)


def _norm_id(x: str) -> str:
    return str(x).strip().upper()


def _parse_ids_file(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(_norm_id(s))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="筛选 summary CSV 后仅对指定 CHEMBL 分子跑 cardiotox activity screen",
    )
    p.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
        help="完整 chembl_excel_full_enrichment_summary.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="子集输出目录（写入 subset_summary.csv 及主脚本产物）",
    )
    p.add_argument(
        "--ids",
        type=str,
        default=None,
        help="逗号分隔 CHEMBL ID；默认使用内置 8 个「无 activity」分子",
    )
    p.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="每行一个 CHEMBL ID（# 开头为注释）；若指定则与 --ids 合并",
    )
    p.add_argument(
        "--subset-name",
        type=str,
        default="chembl_cardiotox_rerun_subset_summary.csv",
        help="写入 out-dir 的子集 CSV 文件名",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只写子集 CSV 并打印将执行的命令，不调用主脚本",
    )
    args, forward = p.parse_known_args()

    summary_path = Path(args.summary_csv).expanduser().resolve()
    if not summary_path.is_file():
        raise SystemExit(f"找不到 summary: {summary_path}")

    want: set[str] = set()
    if args.ids:
        want.update(_norm_id(x) for x in args.ids.split(",") if x.strip())
    if args.ids_file:
        pth = Path(args.ids_file).expanduser().resolve()
        if not pth.is_file():
            raise SystemExit(f"找不到 --ids-file: {pth}")
        want.update(_parse_ids_file(pth))
    if not want:
        want = set(DEFAULT_NO_ACTIVITY_MOLECULES)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path, encoding="utf-8-sig")
    if "primary_chembl_id" not in df.columns or "excel_row_index" not in df.columns:
        raise SystemExit("summary CSV 需含 primary_chembl_id、excel_row_index")

    df["_pid"] = df["primary_chembl_id"].astype(str).map(_norm_id)
    sub = df[df["_pid"].isin(want)].drop(columns=["_pid"])
    missing = sorted(want - set(df["_pid"]))
    if missing:
        print(f"[warn] 下列 ID 在 summary 中不存在，已跳过: {missing}", file=sys.stderr)

    if sub.empty:
        raise SystemExit("子集为空：请检查 --ids / --ids-file 是否与 summary 中 primary_chembl_id 一致")

    subset_path = out_dir / str(args.subset_name)
    sub.to_csv(subset_path, index=False, encoding="utf-8-sig")
    print(f"[write] {subset_path} ({len(sub)} rows)")

    cmd = [
        sys.executable,
        "-m",
        "data_augmentation.Chembl_data.chembl_cardiotox_activity_screen",
        "--summary-csv",
        str(subset_path),
        "--out-dir",
        str(out_dir),
    ] + list(forward)

    print("[run]", " ".join(cmd))
    if args.dry_run:
        return

    r = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2])
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
