"""
s11_supabase_writer.py

Step 11: Write topic assignments to Supabase `articles_topics` table.

Takes df_alloc (output of s06) and writes results to `articles_topics`,
matched by article `id` from `articles_raw`.

Strategy:
  1. Build upsert payloads from df_alloc (which already has `id` from articles_raw).
  2. Upsert in chunks of CHUNK_SIZE.

Columns written per row:
  url, title, article_date, source, country, institution_name, language,
  dataset_type, week_number, model_type, topic_num, dominant_topic,
  dominant_topic_weight, topic_probabilities (JSONB), contestability_score,
  text_clean, article_text, election_period, run_id

Run standalone:
  python -m model_pipeline.training.s11_supabase_writer
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from tqdm import tqdm

logger = logging.getLogger(__name__)

load_dotenv()

ELECTION_DATE = pd.Timestamp("2024-07-04")
CHUNK_SIZE = 500


def _get_topic_cols(df: pd.DataFrame) -> list[str]:
    """Detect topic weight columns from the DataFrame.

    Topic columns are those added by s06 — numeric columns that aren't
    standard metadata or allocation columns.
    """
    skip = {
        "id", "url", "title", "text", "text_clean", "text_final",
        "article_date", "source", "country", "type", "institution_name",
        "language", "dataset_type", "week_number", "created_at",
        "topic_num", "topic_name", "dominant_topic_weight",
        "year", "month", "tokens_final", "date",
    }
    return [c for c in df.columns if c not in skip and df[c].dtype in ("float64", "float32")]


# ── Supabase client ───────────────────────────────────────────────────────────

def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in your .env file."
        )
    return create_client(url, key)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_contestability(row: pd.Series, topic_cols: list[str]) -> float:
    """
    Normalised Shannon entropy across all topic weights.
    0.0 = all weight on one topic (certain).
    1.0 = uniform distribution (maximum uncertainty).
    """
    weights = row[topic_cols].values.astype(float)
    weights = np.maximum(weights, 1e-12)
    total = weights.sum()
    probs = weights / total
    h = -np.sum(probs * np.log(probs))
    max_h = np.log(len(probs)) if len(probs) > 0 else 1.0
    return float(round(h / max_h, 6))


def _build_topic_probabilities(row: pd.Series, topic_cols: list[str]) -> dict:
    return {col: round(float(row[col]), 6) for col in topic_cols}


def _election_period(date) -> str:
    try:
        if pd.isna(date):
            return "pre_election"
    except (TypeError, ValueError):
        return "pre_election"
    return "post_election" if pd.Timestamp(date) >= ELECTION_DATE else "pre_election"


# ── Main writer ───────────────────────────────────────────────────────────────

def write_topic_results(
    df_alloc: pd.DataFrame,
    run_id: str,
    model_type: str = "nmf",
) -> None:
    """
    Upsert topic assignments into Supabase `articles_topics` table.

    Args:
        df_alloc:    Output of s06 run_topic_allocation(). Must contain id, url,
                     topic_num, topic_name, dominant_topic_weight, all 30 topic
                     weight columns, text_clean, and article_date.
        run_id:      The run directory name (e.g. "2026-03-19_160734").
        model_type:  "nmf" or "bertopic".
    """
    topic_cols = _get_topic_cols(df_alloc)
    if not topic_cols:
        raise ValueError(
            "No topic weight columns found in df_alloc. "
            "Ensure s06 has run before s11."
        )
    logger.info("Detected %d topic columns: %s...", len(topic_cols), topic_cols[:3])

    client = get_supabase_client()

    payloads: list[dict] = []
    skipped = 0

    for row in tqdm(
        df_alloc.itertuples(index=False),
        total=len(df_alloc),
        desc="Building payloads",
    ):
        row_id = getattr(row, "id", None)
        url = getattr(row, "url", None)
        if not row_id or pd.isna(row_id):
            skipped += 1
            continue

        row_series = pd.Series(row._asdict())

        payloads.append({
            "id":                     str(row_id),
            "url":                    str(url) if pd.notna(url) else None,
            "title":                  str(row.title) if pd.notna(getattr(row, "title", None)) else None,
            "article_date":           str(row.article_date) if pd.notna(getattr(row, "article_date", None)) else None,
            "source":                 str(row.source) if pd.notna(getattr(row, "source", None)) else None,
            "country":                str(row.country) if pd.notna(getattr(row, "country", None)) else None,
            "institution_name":       str(row.institution_name) if pd.notna(getattr(row, "institution_name", None)) else None,
            "language":               str(row.language) if pd.notna(getattr(row, "language", None)) else "en",
            "dataset_type":           str(row.dataset_type) if pd.notna(getattr(row, "dataset_type", None)) else "training",
            "week_number":            int(row.week_number) if pd.notna(getattr(row, "week_number", None)) else None,
            "model_type":             model_type,
            "topic_num":              int(row.topic_num),
            "dominant_topic":         str(row.topic_name),
            "dominant_topic_weight":  round(float(row.dominant_topic_weight), 6),
            "topic_probabilities":    _build_topic_probabilities(row_series, topic_cols),
            "contestability_score":   round(_compute_contestability(row_series, topic_cols), 6),
            "text_clean":             str(row.text_clean) if pd.notna(getattr(row, "text_clean", None)) else None,
            "article_text":           str(row.text) if pd.notna(getattr(row, "text", None)) else None,
            "election_period":        _election_period(getattr(row, "article_date", None)),
            "run_id":                 run_id,
        })

    logger.info("Payloads built: %d to write, %d skipped.", len(payloads), skipped)

    # Upsert in chunks
    errors: list[int] = []
    for i in tqdm(
        range(0, len(payloads), CHUNK_SIZE),
        desc="Upserting to Supabase",
    ):
        chunk = payloads[i : i + CHUNK_SIZE]
        try:
            client.table("articles_topics").upsert(chunk).execute()
        except Exception as e:
            logger.warning("Chunk %d failed: %s", i // CHUNK_SIZE, e)
            errors.append(i // CHUNK_SIZE)

    logger.info(
        "Supabase write complete. %d rows written, %d chunks failed.",
        len(payloads),
        len(errors),
    )
    if errors:
        logger.warning("Failed chunk indices: %s", errors)


# Keep old name as alias for backward compatibility
write_training_results = write_topic_results


# ── Standalone smoke test ─────────────────────────────────────────────────────

def main() -> None:
    import logging

    from model_pipeline.training.s01_data_loader import load_articles
    from model_pipeline.training.s02_cleaning import run_cleaning
    from model_pipeline.training.s03_spacy_processing import run_spacy_processing
    from model_pipeline.training.s04_vectorisation import run_vectorisation
    from model_pipeline.training.s05_nmf_training import train_nmf
    from model_pipeline.training.s06_topic_allocation import run_topic_allocation
    from model_pipeline.training.s08_save_outputs import make_run_id

    logging.basicConfig(level=logging.INFO)

    run_id = make_run_id()

    df = load_articles("full_retro")
    df = run_cleaning(df)
    df = run_spacy_processing(df)
    vec_out = run_vectorisation(df)
    nmf_out = train_nmf(vec_out.X)
    df_alloc = run_topic_allocation(
        df, nmf_model=nmf_out.nmf_model, vectorizer=vec_out.vectorizer
    )

    write_topic_results(df_alloc, run_id=run_id, model_type="nmf")
    print("\n✅ s11 complete — topic results written to Supabase (articles_topics).")


if __name__ == "__main__":
    main()
