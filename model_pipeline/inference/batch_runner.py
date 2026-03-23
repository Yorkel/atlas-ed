"""
batch_runner.py

Run topic allocation for all three countries using their own trained NMF models.

Two pipelines, same logic (load CSV → model → allocate topics → push to Supabase):
  - training:          training data (2023-2025) through each country's model
  - inference_weekly:  new weekly articles through each country's model

Models are trained in notebooks and identified via config.yaml countries.<country>.model_run.
Data is synced from Supabase via sync_from_supabase.py before running.

Usage:
  python -m model_pipeline.inference.batch_runner --mode training_all
  python -m model_pipeline.inference.batch_runner --mode training_eng
  python -m model_pipeline.inference.batch_runner --mode inference_weekly
  python -m model_pipeline.inference.batch_runner --mode all_training_inference

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


def preprocess(csv_path: Path) -> pd.DataFrame:
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


def allocate_and_push(df: pd.DataFrame, nmf_model, vectorizer, run_id: str, model_type: str = "nmf") -> None:
    """Run topic allocation and write results to Supabase."""
    from model_pipeline.training.s06_topic_allocation import run_topic_allocation
    from model_pipeline.training.s11_supabase_writer import write_topic_results

    df_alloc = run_topic_allocation(df, nmf_model=nmf_model, vectorizer=vectorizer)
    write_topic_results(df_alloc, run_id=run_id, model_type=model_type)


def process_csvs(csv_paths: list[Path], country: str, nmf_model, vectorizer, run_id: str, label: str = "") -> None:
    """Shared pipeline: preprocess CSVs → allocate topics → push to Supabase."""
    model_type = f"nmf_{country}"
    for csv_path in csv_paths:
        df = preprocess(csv_path)
        if df.empty:
            logger.info("Empty after preprocessing: %s. Skipping.", csv_path.name)
            continue

        logger.info("Processing %s %s: %d articles (model: %s)", country, label, len(df), run_id)
        allocate_and_push(df, nmf_model, vectorizer, run_id, model_type=model_type)


def run_training(country: str) -> None:
    """Run training data through a country's model and push topics to Supabase."""
    csv_path = PROJECT_ROOT / "data" / "training" / f"{country}_training.csv"
    if not csv_path.exists():
        logger.error("File not found: %s. Run sync_from_supabase.py first.", csv_path)
        return

    run_dir = get_model_dir(country)
    nmf_model, vectorizer = load_model(run_dir)

    logger.info("── Training pipeline: %s ──", country)
    process_csvs([csv_path], country, nmf_model, vectorizer, run_dir.name, label="training")
    logger.info("Training pipeline complete for %s.", country)


def run_inference_weekly() -> None:
    """Run weekly inference — each country uses its own model."""
    weekly_dir = PROJECT_ROOT / "data" / "inference" / "weekly"
    if not weekly_dir.exists():
        logger.error("Weekly directory not found: %s. Run sync_from_supabase.py first.", weekly_dir)
        return

    csv_files = sorted(weekly_dir.glob("*_week_*.csv"))
    if not csv_files:
        logger.info("No weekly CSVs found. Nothing to do.")
        return

    # Group CSVs by country
    country_csvs: dict[str, list[Path]] = {}
    for csv_path in csv_files:
        country = csv_path.stem.split("_week_")[0]
        country_csvs.setdefault(country, []).append(csv_path)

    # Process each country's weekly files
    for country, csvs in country_csvs.items():
        try:
            run_dir = get_model_dir(country)
            nmf_model, vectorizer = load_model(run_dir)
        except FileNotFoundError as e:
            logger.error("Skipping %s: %s", country, e)
            continue

        logger.info("── Inference weekly: %s (%d files) ──", country, len(csvs))
        process_csvs(csvs, country, nmf_model, vectorizer, run_dir.name, label="weekly")

    logger.info("Weekly inference complete for all countries.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    logging.getLogger("gensim").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Run NMF topic allocation pipeline")
    parser.add_argument(
        "--mode",
        choices=[
            "training_all", "training_eng", "training_sco", "training_irl",
            "inference_weekly",
            "all_training_inference",
        ],
        default="all_training_inference",
        help="Which pipeline to run",
    )
    args = parser.parse_args()

    if args.mode == "training_all":
        for country in CONFIG["countries"]:
            run_training(country)
    elif args.mode.startswith("training_"):
        country = args.mode.split("_")[1]
        run_training(country)
    elif args.mode == "inference_weekly":
        run_inference_weekly()
    elif args.mode == "all_training_inference":
        for country in CONFIG["countries"]:
            run_training(country)
        run_inference_weekly()

    print("\nBatch pipeline complete.")


if __name__ == "__main__":
    main()
