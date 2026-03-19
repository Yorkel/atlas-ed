"""
s01_data_loader.py

Data loading for AM1 NMF pipeline.
Loads articles from CSV, combines title + text into a single 'text' column,
and removes PDF-sourced rows.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

DATA_PATHS = {
    "full_retro": DATA_DIR / "full_retro" / "full_retro.csv",
    "long":       DATA_DIR / "long"       / "long_articles.csv",   # add when ready
    "short":      DATA_DIR / "short"      / "short_articles.csv",  # add when ready
}

REQUIRED_COLS = {"title", "text", "article_date", "source", "type"}


# ── PDF detection ─────────────────────────────────────────────────────────────
def is_pdf(text: str) -> bool:
    """Detect PDF-sourced rows by checking for %PDF-1. header."""
    if not isinstance(text, str):
        return False
    return text.lstrip()[:20].startswith("%PDF-1.")


# ── Core loader ───────────────────────────────────────────────────────────────
def load_articles(dataset: str = "full_retro") -> pd.DataFrame:
    """
    Load articles, combine title + text, drop PDF rows.
    """
    if dataset not in DATA_PATHS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(DATA_PATHS)}")

    path = DATA_PATHS[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Dataset '{dataset}' not found at: {path}")

    logger.info("Loading '%s' from %s", dataset, path)
    df = pd.read_csv(path)
    logger.info("Raw shape: %s", df.shape)

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Combine title + text (mirrors notebook exactly)
    df["text"] = df["title"].fillna("") + "\n\n" + df["text"].fillna("")

    # Parse date early so downstream can assume datetime
    df["article_date"] = pd.to_datetime(df["article_date"], errors="coerce")

    # Remove PDF rows
    n_before = len(df)
    pdf_mask = df["text"].apply(is_pdf)
    df = df[~pdf_mask].reset_index(drop=True)
    n_dropped = n_before - len(df)

    logger.info("PDF rows removed: %d", n_dropped)
    logger.info("Final shape: %s", df.shape)

    return df


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_articles("full_retro")
    print(df.shape)
    print(df.columns.tolist())
    print(df[["source", "type"]].value_counts())
    print(df.head(2))
