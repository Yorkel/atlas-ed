"""
s01_data_loader.py

Data loading for AM1 NMF pipeline.
Loads articles from synced CSVs (pulled from Supabase via sync_from_supabase.py),
combines title + text into a single 'text' column, and removes PDF-sourced rows.

Data structure:
  data/training/eng_training.csv         <- England training corpus
  data/inference/eng_inference.csv       <- England weekly inference
  data/inference/sco_inference.csv       <- Scotland (backfill + weekly)
  data/inference/irl_inference.csv       <- Ireland (backfill + weekly)
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

DATA_PATHS = {
    "eng_training":  DATA_DIR / "training"  / "eng_training.csv",
    "sco_training":  DATA_DIR / "training"  / "sco_training.csv",
    "irl_training":  DATA_DIR / "training"  / "irl_training.csv",
    "eng_inference":  DATA_DIR / "inference" / "eng_inference.csv",
    "sco_inference":  DATA_DIR / "inference" / "sco_inference.csv",
    "irl_inference":  DATA_DIR / "inference" / "irl_inference.csv",
}

REQUIRED_COLS = {"title", "text", "article_date", "source", "id"}


# ── PDF detection ─────────────────────────────────────────────────────────────
def is_pdf(text: str) -> bool:
    """Detect PDF-sourced rows by checking for %PDF-1. header."""
    if not isinstance(text, str):
        return False
    return text.lstrip()[:20].startswith("%PDF-1.")


# ── Core loader ───────────────────────────────────────────────────────────────
def load_articles(dataset: str = "eng_training") -> pd.DataFrame:
    """
    Load articles from synced CSV, combine title + text, drop PDF rows.

    Args:
        dataset: One of "eng_training", "eng_inference", "sco_inference", "irl_inference"

    Returns:
        DataFrame with columns: id, url, title, text (combined), article_date,
        source, country, type, dataset_type, week_number, and others from CSV.
    """
    if dataset not in DATA_PATHS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(DATA_PATHS)}")

    path = DATA_PATHS[dataset]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset}' not found at: {path}. "
            "Run sync_from_supabase.py first."
        )

    logger.info("Loading '%s' from %s", dataset, path)
    df = pd.read_csv(path)
    logger.info("Raw shape: %s", df.shape)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Combine title + text
    df["text"] = df["title"].fillna("") + "\n\n" + df["text"].fillna("")

    # Parse date
    df["article_date"] = pd.to_datetime(df["article_date"], errors="coerce")

    # Remove PDF rows
    n_before = len(df)
    pdf_mask = df["text"].apply(is_pdf)
    df = df[~pdf_mask].reset_index(drop=True)
    n_dropped = n_before - len(df)

    logger.info("PDF rows removed: %d", n_dropped)
    logger.info("Final shape: %s", df.shape)

    return df


def load_all_inference() -> pd.DataFrame:
    """Load all inference CSVs (eng + sco + irl) into one DataFrame."""
    dfs = []
    for dataset in ["eng_inference", "sco_inference", "irl_inference"]:
        try:
            df = load_articles(dataset)
            dfs.append(df)
            logger.info("Loaded %s: %d rows", dataset, len(df))
        except FileNotFoundError:
            logger.warning("Skipping %s — file not found", dataset)
    if not dfs:
        raise FileNotFoundError("No inference CSVs found. Run sync_from_supabase.py first.")
    combined = pd.concat(dfs, ignore_index=True)
    logger.info("Total inference rows: %d", len(combined))
    return combined


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_articles("eng_training")
    print(df.shape)
    print(df.columns.tolist())
    print(df[["source", "country"]].value_counts())
    print(df.head(2))
