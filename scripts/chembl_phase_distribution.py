import time
from pathlib import Path

import requests
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"

# ChEMBL max_phase main categories
# NULL: no clinical phase annotation / preclinical or ordinary bioactive compounds
# -1: unknown phase for drug / clinical candidate
# 0: sometimes used in some datasets / versions
# 0.5: Early Phase I
# 1: Phase I
# 2: Phase II
# 3: Phase III
# 4: Approved
PHASE_ITEMS = [
    {"max_phase": None, "phase_name": "NULL / No clinical phase"},
    {"max_phase": -1, "phase_name": "Unknown"},
    {"max_phase": 0, "phase_name": "Phase 0 / Preclinical"},
    {"max_phase": 0.5, "phase_name": "Early Phase I"},
    {"max_phase": 1, "phase_name": "Phase I"},
    {"max_phase": 2, "phase_name": "Phase II"},
    {"max_phase": 3, "phase_name": "Phase III"},
    {"max_phase": 4, "phase_name": "Approved"},
]


def request_chembl(params, max_retries=5, sleep_seconds=2):
    """
    Robust ChEMBL API request with retry.
    """

    headers = {
        "User-Agent": "chembl-phase-statistics-script/1.0"
    }

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                BASE_URL,
                params=params,
                headers=headers,
                timeout=60
            )

            response.raise_for_status()
            return response.json()

        except Exception as e:
            last_error = e
            print(f"[Warning] API request failed, attempt {attempt}/{max_retries}: {e}")
            time.sleep(sleep_seconds * attempt)

    raise RuntimeError(f"ChEMBL API request failed after retries: {last_error}")


def get_total_molecule_count():
    """
    Get total number of molecule records in ChEMBL API.
    """

    params = {
        "limit": 1
    }

    data = request_chembl(params)

    if "page_meta" not in data or "total_count" not in data["page_meta"]:
        raise ValueError("Unexpected ChEMBL API response: missing page_meta.total_count")

    return data["page_meta"]["total_count"]


def get_phase_count(max_phase):
    """
    Get count for one max_phase value.

    For NULL max_phase, first try:
        max_phase__isnull=true

    If the API does not support this filter in your environment,
    the script will calculate NULL count later by:
        total - sum(non-null phase counts)
    """

    if max_phase is None:
        params = {
            "max_phase__isnull": "true",
            "limit": 1
        }
    else:
        params = {
            "max_phase": max_phase,
            "limit": 1
        }

    data = request_chembl(params)

    if "page_meta" not in data or "total_count" not in data["page_meta"]:
        raise ValueError(f"Unexpected API response for max_phase={max_phase}")

    return data["page_meta"]["total_count"]


def collect_distribution():
    """
    Collect full max_phase distribution from ChEMBL API.
    """

    print("Fetching total ChEMBL molecule count...")
    total_chembl = get_total_molecule_count()

    rows = []
    non_null_total = 0
    null_count_from_api = None

    for item in PHASE_ITEMS:
        max_phase = item["max_phase"]
        phase_name = item["phase_name"]

        print(f"Fetching count for max_phase={max_phase}, {phase_name}...")

        try:
            count = get_phase_count(max_phase)

            if max_phase is None:
                null_count_from_api = count
            else:
                non_null_total += count

        except Exception as e:
            print(f"[Warning] Failed to fetch max_phase={max_phase}: {e}")
            count = None

        rows.append({
            "max_phase": "NULL" if max_phase is None else max_phase,
            "phase_name": phase_name,
            "molecule_count": count
        })

    df = pd.DataFrame(rows)

    # Fallback: if NULL count failed, calculate it from total - non-null known phases
    if df.loc[df["max_phase"] == "NULL", "molecule_count"].isna().any():
        calculated_null = total_chembl - non_null_total
        df.loc[df["max_phase"] == "NULL", "molecule_count"] = calculated_null
        print(f"[Info] NULL count calculated as total - non-null phase counts: {calculated_null}")

    df["molecule_count"] = df["molecule_count"].astype(int)

    df["percentage_of_total_chembl"] = df["molecule_count"].apply(
        lambda x: round(x / total_chembl * 100, 4) if total_chembl > 0 else 0
    )

    clinical_df = df[df["max_phase"].isin([-1, 0, 0.5, 1, 2, 3, 4])].copy()
    clinical_total = clinical_df["molecule_count"].sum()

    df["percentage_of_non_null_phase_subset"] = df.apply(
        lambda row: round(row["molecule_count"] / clinical_total * 100, 4)
        if row["max_phase"] != "NULL" and clinical_total > 0 else None,
        axis=1
    )

    summary = {
        "total_chembl_molecule_records": total_chembl,
        "non_null_phase_subset_total": clinical_total,
        "null_or_no_clinical_phase_total": int(
            df.loc[df["max_phase"] == "NULL", "molecule_count"].iloc[0]
        )
    }

    return df, summary


def save_outputs(df, summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "chembl_all_max_phase_distribution.csv"
    xlsx_path = output_dir / "chembl_all_max_phase_distribution.xlsx"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="max_phase_distribution", index=False)

        summary_df = pd.DataFrame([summary])
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    print(f"CSV saved to: {csv_path}")
    print(f"Excel saved to: {xlsx_path}")


def plot_all_bar(df, output_dir):
    """
    Normal bar chart.
    NULL category is usually very large, so smaller clinical phases may look compressed.
    """

    output_path = Path(output_dir) / "chembl_all_max_phase_bar.png"

    plt.figure(figsize=(11, 6))

    plt.bar(df["phase_name"], df["molecule_count"])

    plt.xlabel("max_phase category")
    plt.ylabel("Number of ChEMBL molecule records")
    plt.title("ChEMBL Molecule Distribution by max_phase")

    plt.xticks(rotation=30, ha="right")

    for index, value in enumerate(df["molecule_count"]):
        plt.text(index, value, str(value), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"All-phase bar chart saved to: {output_path}")


def plot_all_bar_log(df, output_dir):
    """
    Log-scale bar chart.
    Better for viewing all phases because NULL usually dominates.
    """

    output_path = Path(output_dir) / "chembl_all_max_phase_bar_log.png"

    plt.figure(figsize=(11, 6))

    plt.bar(df["phase_name"], df["molecule_count"])
    plt.yscale("log")

    plt.xlabel("max_phase category")
    plt.ylabel("Number of ChEMBL molecule records, log scale")
    plt.title("ChEMBL Molecule Distribution by max_phase, Log Scale")

    plt.xticks(rotation=30, ha="right")

    for index, value in enumerate(df["molecule_count"]):
        plt.text(index, value, str(value), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"All-phase log bar chart saved to: {output_path}")


def plot_clinical_bar(df, output_dir):
    """
    Bar chart excluding NULL.
    This is useful for comparing Phase I, II, III and Approved.
    """

    output_path = Path(output_dir) / "chembl_clinical_phase_bar.png"

    clinical_df = df[df["max_phase"].isin([-1, 0, 0.5, 1, 2, 3, 4])].copy()

    plt.figure(figsize=(10, 6))

    plt.bar(clinical_df["phase_name"], clinical_df["molecule_count"])

    plt.xlabel("Clinical / approval phase category")
    plt.ylabel("Number of ChEMBL molecule records")
    plt.title("ChEMBL Clinical and Approved Molecules by max_phase")

    plt.xticks(rotation=30, ha="right")

    for index, value in enumerate(clinical_df["molecule_count"]):
        plt.text(index, value, str(value), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Clinical-phase bar chart saved to: {output_path}")


def plot_clinical_pie(df, output_dir):
    """
    Pie chart excluding NULL.
    If NULL is included, the pie chart is not useful because NULL dominates.
    """

    output_path = Path(output_dir) / "chembl_clinical_phase_pie.png"

    clinical_df = df[df["max_phase"].isin([-1, 0, 0.5, 1, 2, 3, 4])].copy()

    labels = [
        f"{row.phase_name}\n{row.molecule_count}"
        for row in clinical_df.itertuples()
    ]

    plt.figure(figsize=(8, 8))

    plt.pie(
        clinical_df["molecule_count"],
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("ChEMBL Clinical and Approved max_phase Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Clinical-phase pie chart saved to: {output_path}")


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df, summary = collect_distribution()

    print("\nChEMBL max_phase distribution")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)

    print("\nSummary")
    print("=" * 100)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("=" * 100)

    save_outputs(df, summary, output_dir)

    plot_all_bar(df, output_dir)
    plot_all_bar_log(df, output_dir)
    plot_clinical_bar(df, output_dir)
    plot_clinical_pie(df, output_dir)


if __name__ == "__main__":
    main()
