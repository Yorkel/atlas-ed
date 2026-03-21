"""
batch_runner.py

Run inference on Scotland, Ireland, and England inference data
through the England-trained NMF model.

Three modes:
  1. Backfill Ireland (2023–2025, week=None)
  2. Backfill Scotland (2023–2025, week=None)
  3. Weekly inference (all three countries, week by week)

Usage:
  python -m model_pipeline.inference.batch_runner --mode backfill_irl
  python -m model_pipeline.inference.batch_runner --mode backfill_sco
  python -m model_pipeline.inference.batch_runner --mode weekly
  python -m model_pipeline.inference.batch_runner --mode all

Requires:
  - Saved model in experiments/outputs/runs/<run_id>/
  - Synced CSVs in data/inference/
  - .env with SUPABASE_URL and SUPABASE_SERVICE_KEY
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default to latest run
RUNS_DIR = PROJECT_ROOT / "experiments" / "outputs" / "runs"


def get_latest_run_dir() -> Path:
    """Get the most recent run directory."""
    runs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found in {RUNS_DIR}")
    return runs[-1]


def load_model(run_dir: Path):
    """Load saved NMF model and vectorizer."""
    nmf_model = joblib.load(run_dir / "nmf_model.joblib")
    vectorizer = joblib.load(run_dir / "vectorizer.joblib")
    logger.info("Loaded model from %s", run_dir)
    return nmf_model, vectorizer


def load_and_preprocess(csv_path: Path) -> pd.DataFrame:
    """Load CSV and run preprocessing (s02 cleaning + s03 spaCy)."""
    from model_pipeline.training.s02_cleaning import run_cleaning
    from model_pipeline.training.s03_spacy_processing import run_spacy_processing

    logger.info("Loading %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Raw shape: %s", df.shape)

    # Combine title + text
    df["text"] = df["title"].fillna("") + "\n\n" + df["text"].fillna("")
    df["article_date"] = pd.to_datetime(df["article_date"], errors="coerce")

    # Preprocess
    df = run_cleaning(df)
    df = run_spacy_processing(df)
    logger.info("After preprocessing: %s", df.shape)

    return df


def run_inference(df: pd.DataFrame, nmf_model, vectorizer, run_id: str, model_type: str = "nmf") -> None:
    """Run topic allocation and write results to Supabase."""
    from model_pipeline.training.s06_topic_allocation import run_topic_allocation
    from model_pipeline.training.s11_supabase_writer import write_topic_results

    df_alloc = run_topic_allocation(df, nmf_model=nmf_model, vectorizer=vectorizer)
    write_topic_results(df_alloc, run_id=run_id, model_type=model_type)


def run_backfill(country: str, nmf_model, vectorizer, run_id: str) -> None:
    """Run backfill for a single country (2023–2025 data)."""
    csv_path = PROJECT_ROOT / "data" / "inference" / "backfill" / f"{country}_backfill.csv"
    if not csv_path.exists():
        logger.error("File not found: %s. Run sync_from_supabase.py first.", csv_path)
        return

    df = load_and_preprocess(csv_path)

    if df.empty:
        logger.info("No backfill rows for %s. Skipping.", country)
        return

    logger.info("Running backfill for %s: %d articles", country, len(df))
    run_inference(df, nmf_model, vectorizer, run_id)
    logger.info("Backfill complete for %s.", country)


def run_weekly(nmf_model, vectorizer, run_id: str) -> None:
    """Run weekly inference for all three countries, week by week."""
    weekly_dir = PROJECT_ROOT / "data" / "inference" / "weekly"
    if not weekly_dir.exists():
        logger.error("Weekly directory not found: %s. Run sync_from_supabase.py first.", weekly_dir)
        return

    # Find all weekly CSVs and sort by country then week number
    csv_files = sorted(weekly_dir.glob("*_week_*.csv"))
    if not csv_files:
        logger.info("No weekly CSVs found. Nothing to do.")
        return

    for csv_path in csv_files:
        # Extract country and week from filename (e.g. eng_week_3.csv)
        parts = csv_path.stem.split("_week_")
        country = parts[0]
        week_num = int(parts[1])

        df = load_and_preprocess(csv_path)
        if df.empty:
            logger.info("Empty file: %s. Skipping.", csv_path)
            continue

        logger.info("Running %s week %d: %d articles", country, week_num, len(df))
        run_inference(df, nmf_model, vectorizer, run_id)

    logger.info("Weekly inference complete for all countries.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    logging.getLogger("gensim").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Run NMF inference on cross-jurisdiction data")
    parser.add_argument(
        "--mode",
        choices=["backfill_irl", "backfill_sco", "weekly", "all"],
        default="all",
        help="Which inference to run",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to model run directory. Defaults to latest.",
    )
    args = parser.parse_args()

    # Load model
    run_dir = Path(args.run_dir) if args.run_dir else get_latest_run_dir()
    run_id = run_dir.name
    nmf_model, vectorizer = load_model(run_dir)

    if args.mode == "backfill_irl" or args.mode == "all":
        run_backfill("irl", nmf_model, vectorizer, run_id)

    if args.mode == "backfill_sco" or args.mode == "all":
        run_backfill("sco", nmf_model, vectorizer, run_id)

    if args.mode == "weekly" or args.mode == "all":
        run_weekly(nmf_model, vectorizer, run_id)

    print("\n✅ Batch inference complete.")


if __name__ == "__main__":
    main()
