"""
s11_supabase_writer.py

Step 11: Write training topic assignments to Supabase.

Takes df_alloc (output of s06) and updates the corresponding rows
in the Supabase `articles` table, matched by URL → UUID lookup.

Strategy:
  1. Fetch all (id, url) pairs from Supabase to build a url→id map.
  2. Build a list of upsert payloads keyed on `id`.
  3. Upsert in chunks of CHUNK_SIZE (avoids per-row round-trips).

Columns written per row:
  dataset_type, topic_num, dominant_topic, dominant_topic_weight,
  topic_probabilities (JSONB), contestability_score,
  text_clean, election_period, run_id

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

from model_pipeline.training.s06_topic_allocation import TOPIC_NAMES

logger = logging.getLogger(__name__)

load_dotenv()

ELECTION_DATE = pd.Timestamp("2024-07-04")
TOPIC_COLS = list(TOPIC_NAMES.values())
CHUNK_SIZE = 500


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

def _compute_contestability(row: pd.Series) -> float:
    """
    Normalised Shannon entropy across all topic weights.
    0.0 = all weight on one topic (certain).
    1.0 = uniform distribution (maximum uncertainty).

    Replaces the gap-based metric (1 - (top - second)) which was
    structurally broken with 30 topics — NMF spreads weight so thinly
    that every article scored 0.9+ regardless of actual certainty.
    """
    weights = row[TOPIC_COLS].values.astype(float)
    weights = np.maximum(weights, 1e-12)
    total = weights.sum()
    probs = weights / total
    h = -np.sum(probs * np.log(probs))
    max_h = np.log(len(probs)) if len(probs) > 0 else 1.0
    return float(round(h / max_h, 6))


def _build_topic_probabilities(row: pd.Series) -> dict:
    return {col: round(float(row[col]), 6) for col in TOPIC_COLS}


def _election_period(date) -> str:
    try:
        if pd.isna(date):
            return "pre_election"
    except (TypeError, ValueError):
        return "pre_election"
    return "post_election" if pd.Timestamp(date) >= ELECTION_DATE else "pre_election"


def _fetch_url_id_map(client: Client) -> dict[str, str]:
    """Fetch all (url, id) pairs from Supabase to match pipeline rows to UUIDs."""
    logger.info("Fetching url→id map from Supabase...")
    url_to_id: dict[str, str] = {}
    page = 0
    while True:
        start = page * CHUNK_SIZE
        response = (
            client.table("articles")
            .select("id, url")
            .range(start, start + CHUNK_SIZE - 1)
            .execute()
        )
        rows = response.data
        for row in rows:
            if row.get("url"):
                url_to_id[row["url"]] = row["id"]
        if len(rows) < CHUNK_SIZE:
            break
        page += 1
    logger.info("Found %d existing rows in Supabase.", len(url_to_id))
    return url_to_id


# ── Main writer ───────────────────────────────────────────────────────────────

def write_training_results(
    df_alloc: pd.DataFrame,
    run_id: str,
) -> None:
    """
    Upsert training topic assignments into Supabase in batches.

    Args:
        df_alloc:  Output of s06 run_topic_allocation(). Must contain url,
                   topic_num, topic_name, dominant_topic_weight, all 30 topic
                   weight columns, text_clean, and article_date.
        run_id:    The run directory name (e.g. "2026-02-19_223857").
    """
    missing_cols = [c for c in TOPIC_COLS if c not in df_alloc.columns]
    if missing_cols:
        raise ValueError(
            f"df_alloc is missing topic weight columns: {missing_cols[:5]}... "
            "Ensure s06 has run before s11."
        )

    client = get_supabase_client()
    url_to_id = _fetch_url_id_map(client)

    payloads: list[dict] = []
    skipped = 0

    for row in tqdm(
        df_alloc.itertuples(index=False),
        total=len(df_alloc),
        desc="Building payloads",
    ):
        url = getattr(row, "url", None)
        if not url or pd.isna(url):
            skipped += 1
            continue

        row_id = url_to_id.get(url)
        if not row_id:
            logger.debug("URL not found in Supabase, skipping: %s", url)
            skipped += 1
            continue

        row_series = pd.Series(row._asdict())

        payloads.append({
            "id":                     row_id,
            "url":                    url,
            "dataset_type":           "training",
            "topic_num":              int(row.topic_num),
            "dominant_topic":         str(row.topic_name),
            "dominant_topic_weight":  round(float(row.dominant_topic_weight), 6),
            "topic_probabilities":    _build_topic_probabilities(row_series),
            "contestability_score":   round(_compute_contestability(row_series), 6),
            "text_clean":             str(row.text_clean) if pd.notna(getattr(row, "text_clean", None)) else None,
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
            client.table("articles").upsert(chunk).execute()
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

    write_training_results(df_alloc, run_id=run_id)
    print("\n✅ s11 complete — training results written to Supabase.")


if __name__ == "__main__":
    main()
