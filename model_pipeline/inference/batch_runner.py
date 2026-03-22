"""
batch_runner.py

Run inference on all three countries using their own trained NMF models.

Each country loads its own model (specified in config.yaml countries.<country>.model_run).
Weekly inference applies each country's model to that country's new articles.

Usage:
  python -m model_pipeline.inference.batch_runner --mode weekly
  python -m model_pipeline.inference.batch_runner --mode all

Requires:
  - Saved models in experiments/outputs/runs/<run_id>/
  - Synced CSVs in data/
  - .env with SUPABASE_URL and SUPABASE_SERVICE_KEY
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "experiments" / "outputs" / "runs"

with open(PROJECT_ROOT / "config.yaml") as _f:
    CONFIG = yaml.safe_load(_f)


def get_model_dir(country: str) -> Path:
    """Get the model run directory for a country from config."""
    country_cfg = CONFIG["countries"][country]
    model_run = country_cfg.get("model_run")

    if model_run:
        run_dir = RUNS_DIR / model_run
        if not run_dir.exists():
            raise FileNotFoundError(
                f"Model run '{model_run}' for {country} not found at {run_dir}. "
                "Check countries.<country>.model_run in config.yaml."
            )
        return run_dir

    # Fallback: find latest run for this country by prefix
    prefix = f"{country}_" if country != "eng" else ""
    candidates = sorted([
        d for d in RUNS_DIR.iterdir()
        if d.is_dir() and (d.name.startswith(prefix) if prefix else not any(
            d.name.startswith(p) for p in ["sco_", "irl_"]
        ))
    ])
    if not candidates:
        raise FileNotFoundError(
            f"No model runs found for {country}. Train the model first, "
            "then set countries.<country>.model_run in config.yaml."
        )
    logger.warning(
        "No model_run set for %s in config.yaml, using latest: %s",
        country, candidates[-1].name,
    )
    return candidates[-1]


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


def run_backfill(country: str) -> None:
    """Run backfill for a single country using its own model."""
    csv_path = PROJECT_ROOT / "data" / "training" / f"{country}_training.csv"
    if not csv_path.exists():
        logger.error("File not found: %s. Run sync_from_supabase.py first.", csv_path)
        return

    run_dir = get_model_dir(country)
    run_id = run_dir.name
    nmf_model, vectorizer = load_model(run_dir)

    df = load_and_preprocess(csv_path)

    if df.empty:
        logger.info("No backfill rows for %s. Skipping.", country)
        return

    logger.info("Running backfill for %s: %d articles (model: %s)", country, len(df), run_id)
    run_inference(df, nmf_model, vectorizer, run_id)
    logger.info("Backfill complete for %s.", country)


def run_weekly() -> None:
    """Run weekly inference — each country uses its own model."""
    weekly_dir = PROJECT_ROOT / "data" / "inference" / "weekly"
    if not weekly_dir.exists():
        logger.error("Weekly directory not found: %s. Run sync_from_supabase.py first.", weekly_dir)
        return

    csv_files = sorted(weekly_dir.glob("*_week_*.csv"))
    if not csv_files:
        logger.info("No weekly CSVs found. Nothing to do.")
        return

    # Load models once per country
    models = {}
    for country in CONFIG["countries"]:
        try:
            run_dir = get_model_dir(country)
            models[country] = {
                "nmf_model": joblib.load(run_dir / "nmf_model.joblib"),
                "vectorizer": joblib.load(run_dir / "vectorizer.joblib"),
                "run_id": run_dir.name,
            }
            logger.info("Loaded %s model from %s", country, run_dir.name)
        except FileNotFoundError as e:
            logger.error("Skipping %s: %s", country, e)

    for csv_path in csv_files:
        parts = csv_path.stem.split("_week_")
        country = parts[0]
        week_num = int(parts[1])

        if country not in models:
            logger.warning("No model for %s, skipping %s", country, csv_path.name)
            continue

        df = load_and_preprocess(csv_path)
        if df.empty:
            logger.info("Empty file: %s. Skipping.", csv_path)
            continue

        m = models[country]
        logger.info("Running %s week %d: %d articles (model: %s)", country, week_num, len(df), m["run_id"])
        run_inference(df, m["nmf_model"], m["vectorizer"], m["run_id"])

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
        choices=["backfill_irl", "backfill_sco", "backfill_eng", "weekly", "all"],
        default="all",
        help="Which inference to run",
    )
    args = parser.parse_args()

    if args.mode.startswith("backfill_"):
        country = args.mode.split("_")[1]
        run_backfill(country)
    elif args.mode == "weekly":
        run_weekly()
    elif args.mode == "all":
        for country in CONFIG["countries"]:
            run_backfill(country)
        run_weekly()

    print("\nBatch inference complete.")


if __name__ == "__main__":
    main()
