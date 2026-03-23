"""
sync_from_supabase.py

Pulls data from Supabase and saves as local CSVs.

Sync modes (from articles_raw):
    python sync_from_supabase.py              # sync training + weekly
    python sync_from_supabase.py --weekly     # only sync weekly inference
    python sync_from_supabase.py --training   # only sync training data

Download mode (from articles_topics — processed results):
    python sync_from_supabase.py --download   # download all processed data
"""

import argparse
import os
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "data"
TRAINING_DIR = DATA_DIR / "training"
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
def fetch_all(client, table: str, columns: list[str], filters: dict) -> pd.DataFrame:
    """Fetch all rows from a table matching filters, paginating as needed."""
    rows = []
    page = 0
    while True:
        start = page * PAGE_SIZE
        query = client.table(table).select(",".join(columns))
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
        df = fetch_all(client, "articles_raw", COLUMNS, {"country": country, "dataset_type": "training"})
        save_csv(df, TRAINING_DIR / f"{country}_training.csv", f"{country}/training")


def sync_weekly(client):
    """Pull weekly inference data (2026, week_number IS NOT NULL)."""
    print("\n── Weekly ──")
    for country in ["eng", "sco", "irl"]:
        # Fetch all inference rows with week numbers for this country
        df = fetch_all(client, "articles_raw", COLUMNS, {
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


# ── Download processed data ───────────────────────────────────────────────────

PROCESSED_COLUMNS = [
    "id", "url", "title", "article_date", "source", "country",
    "institution_name", "language", "dataset_type", "week_number",
    "model_type", "topic_num", "dominant_topic", "dominant_topic_weight",
    "topic_probabilities", "contestability_score", "election_period",
    "run_id",
]


def download_processed(client):
    """Download all processed data from articles_topics into a single CSV."""
    print("\n── Download processed data ──")
    df = fetch_all(client, "articles_topics", PROCESSED_COLUMNS, {})

    if df.empty:
        print("  No processed data found in articles_topics.")
        return

    processed_dir = DATA_DIR / "processed"
    filename = f"processed_data_{date.today().isoformat()}.csv"
    path = processed_dir / filename
    save_csv(df, path, "all processed")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sync data from Supabase to local CSVs")
    parser.add_argument("--weekly", action="store_true", help="Only sync weekly inference data")
    parser.add_argument("--training", action="store_true", help="Only sync training data")
    parser.add_argument("--download", action="store_true", help="Download all processed data from articles_topics")
    args = parser.parse_args()

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    if args.download:
        download_processed(client)
    elif args.weekly:
        sync_weekly(client)
    elif args.training:
        sync_training(client)
    else:
        sync_training(client)
        sync_weekly(client)

    print("\nDone.")


if __name__ == "__main__":
    main()
