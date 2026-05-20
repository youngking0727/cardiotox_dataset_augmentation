import requests
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"

PHASE_MAP = {
    1: "Phase I",
    2: "Phase II",
    3: "Phase III",
    4: "Approved"
}


def get_phase_count(max_phase: int) -> int:
    """
    Call ChEMBL API and get total_count for a given max_phase.

    Example API:
    https://www.ebi.ac.uk/chembl/api/data/molecule.json?max_phase=4&limit=1
    """

    params = {
        "max_phase": max_phase,
        "limit": 1
    }

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if "page_meta" not in data or "total_count" not in data["page_meta"]:
        raise ValueError(f"Unexpected API response format for max_phase={max_phase}")

    return data["page_meta"]["total_count"]


def collect_phase_distribution() -> pd.DataFrame:
    rows = []

    for phase, phase_name in PHASE_MAP.items():
        print(f"Fetching {phase_name} count from ChEMBL API...")
        count = get_phase_count(phase)

        rows.append({
            "max_phase": phase,
            "phase_name": phase_name,
            "molecule_count": count
        })

    df = pd.DataFrame(rows)

    total = df["molecule_count"].sum()

    df["percentage"] = df["molecule_count"].apply(
        lambda x: round(x / total * 100, 2) if total > 0 else 0
    )

    return df


def save_csv(df: pd.DataFrame, output_csv: str):
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"CSV saved to: {output_csv}")


def plot_bar(df: pd.DataFrame, output_png: str):
    plt.figure(figsize=(8, 5))

    plt.bar(df["phase_name"], df["molecule_count"])

    plt.xlabel("Clinical development phase")
    plt.ylabel("Number of molecules")
    plt.title("ChEMBL Molecule Count by max_phase")

    for index, value in enumerate(df["molecule_count"]):
        plt.text(index, value, str(value), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Bar chart saved to: {output_png}")


def plot_pie(df: pd.DataFrame, output_png: str):
    plt.figure(figsize=(7, 7))

    labels = [
        f"{row.phase_name}\n{row.molecule_count} ({row.percentage}%)"
        for row in df.itertuples()
    ]

    plt.pie(
        df["molecule_count"],
        labels=labels,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("ChEMBL Phase Distribution")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    print(f"Pie chart saved to: {output_png}")


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "chembl_phase_distribution_api.csv"
    bar_path = output_dir / "chembl_phase_bar.png"
    pie_path = output_dir / "chembl_phase_pie.png"

    df = collect_phase_distribution()

    print("\nChEMBL Phase Distribution from API")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    print(f"Total Phase I + II + III + Approved molecules: {df['molecule_count'].sum()}")

    save_csv(df, csv_path)
    plot_bar(df, bar_path)
    plot_pie(df, pie_path)


if __name__ == "__main__":
    main()
