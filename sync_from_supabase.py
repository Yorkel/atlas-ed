"""
sync_from_supabase.py

Pulls data from Supabase `articles_raw` and saves as local CSVs.
Structure:
    data/training/eng_training.csv
    data/training/sco_training.csv
    data/training/irl_training.csv
    data/inference/weekly/eng_week_1.csv ... eng_week_N.csv
    data/inference/weekly/sco_week_1.csv ... sco_week_N.csv
    data/inference/weekly/irl_week_1.csv ... irl_week_N.csv

Usage:
    python sync_from_supabase.py              # sync everything
    python sync_from_supabase.py --weekly     # only sync weekly inference (faster)
    python sync_from_supabase.py --training   # only sync training data
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "data"
TRAINING_DIR = DATA_DIR / "training"
BACKFILL_DIR = DATA_DIR / "inference" / "backfill"
WEEKLY_DIR = DATA_DIR / "inference" / "weekly"

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

PAGE_SIZE = 1000

COLUMNS = [
    "id", "url", "title", "article_date", "text", "source",
    "country", "type", "institution_name", "language",
    "dataset_type", "week_number", "created_at",
]


# ── Supabase fetch ────────────────────────────────────────────────────────────
def fetch_all(client, filters: dict) -> pd.DataFrame:
    """Fetch all rows from articles_raw matching filters, paginating as needed."""
    rows = []
    page = 0
    while True:
        start = page * PAGE_SIZE
        query = client.table("articles_raw").select(",".join(COLUMNS))
        for col, val in filters.items():
            if val is None:
                query = query.is_(col, "null")
            else:
                query = query.eq(col, val)
        resp = query.range(start, start + PAGE_SIZE - 1).execute()
        rows.extend(resp.data)
        if len(resp.data) < PAGE_SIZE:
            break
        page += 1
    return pd.DataFrame(rows)


def save_csv(df: pd.DataFrame, path: Path, label: str):
    """Save DataFrame to CSV and print summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  {label}: {len(df)} rows → {path}")


# ── Sync functions ────────────────────────────────────────────────────────────
def sync_training(client):
    """Pull training data for all three countries."""
    print("\n── Training ──")
    for country in ["eng", "sco", "irl"]:
        df = fetch_all(client, {"country": country, "dataset_type": "training"})
        save_csv(df, TRAINING_DIR / f"{country}_training.csv", f"{country}/training")


def sync_weekly(client):
    """Pull weekly inference data (2026, week_number IS NOT NULL)."""
    print("\n── Weekly ──")
    for country in ["eng", "sco", "irl"]:
        # Fetch all inference rows with week numbers for this country
        df = fetch_all(client, {
            "country": country,
            "dataset_type": "inference",
        })
        # Filter to only rows with week_number set
        df_weekly = df[df["week_number"].notna()].copy()

        if df_weekly.empty:
            print(f"  {country}/weekly: no weekly data")
            continue

        # Save one file per week
        for week_num in sorted(df_weekly["week_number"].unique()):
            week_int = int(week_num)
            df_week = df_weekly[df_weekly["week_number"] == week_num]
            save_csv(
                df_week,
                WEEKLY_DIR / f"{country}_week_{week_int}.csv",
                f"{country}/week_{week_int}",
            )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sync data from Supabase to local CSVs")
    parser.add_argument("--weekly", action="store_true", help="Only sync weekly inference data")
    parser.add_argument("--training", action="store_true", help="Only sync training data")
    args = parser.parse_args()

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    if args.weekly:
        sync_weekly(client)
    elif args.training:
        sync_training(client)
    else:
        sync_training(client)
        sync_weekly(client)

    print("\nDone.")


if __name__ == "__main__":
    main()
