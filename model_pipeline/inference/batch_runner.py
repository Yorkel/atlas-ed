"""
batch_runner.py

Simulate week-by-week inference: fetch unprocessed inference articles
from Supabase grouped by week, send each week to the deployed FastAPI
/predict endpoint, and write the topic assignments back.

Usage:
  python -m model_pipeline.inference.batch_runner

Requires .env with SUPABASE_URL and SUPABASE_SERVICE_KEY.
API_URL defaults to the Render deployment; override via .env if needed.
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from supabase import create_client, Client
from urllib3.util.retry import Retry

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "https://atlased-api.onrender.com")
BATCH_SIZE = 50        # articles per API request
CHUNK_SIZE = 500       # rows per Supabase upsert
FETCH_PAGE_SIZE = 500  # pagination when reading from Supabase


# ── HTTP session with retries ────────────────────────────────────────────────

def _get_http_session() -> requests.Session:
    """Session with automatic retries and backoff for transient failures."""
    retry = Retry(
        total=5,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("POST",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


SESSION = _get_http_session()


# ── Supabase client ──────────────────────────────────────────────────────────

def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in your .env file."
        )
    return create_client(url, key)


# ── Step 1: Fetch unprocessed inference articles ─────────────────────────────

def fetch_unprocessed_articles(client: Client) -> list[dict]:
    """
    Fetch all inference articles without a topic assignment.
    Paginates to avoid the Supabase 1000-row default limit.
    """
    articles: list[dict] = []
    page = 0
    while True:
        start = page * FETCH_PAGE_SIZE
        response = (
            client.table("articles")
            .select("id, article_text, article_date, week_start, week_end, week_number")
            .eq("dataset_type", "inference")
            .is_("dominant_topic", "null")
            .order("week_start")
            .range(start, start + FETCH_PAGE_SIZE - 1)
            .execute()
        )
        rows = response.data
        articles.extend(rows)
        if len(rows) < FETCH_PAGE_SIZE:
            break
        page += 1
    logger.info("Fetched %d unprocessed inference articles.", len(articles))
    return articles


def group_by_week(articles: list[dict]) -> list[tuple[str, str, int, list[dict]]]:
    """
    Group articles by (week_start, week_end, week_number).
    Returns a sorted list of (week_start, week_end, week_number, articles).
    """
    weeks: dict[tuple, list[dict]] = defaultdict(list)
    for a in articles:
        key = (a["week_start"], a["week_end"], a["week_number"])
        weeks[key].append(a)
    return [
        (ws, we, wn, arts)
        for (ws, we, wn), arts in sorted(weeks.items())
    ]


# ── Step 2: Send batch to API ────────────────────────────────────────────────

def predict_batch(articles: list[dict]) -> dict:
    """
    POST a batch of articles to /predict.
    Returns the full API response (predictions, run_id, n_articles).
    Raises ValueError if the response shape is unexpected.
    """
    payload = {
        "articles": [
            {"article_id": a["id"], "text": a["article_text"]}
            for a in articles
        ]
    }
    resp = SESSION.post(f"{API_URL}/predict", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    # Validate response shape before writing to database
    if "run_id" not in data or "predictions" not in data:
        raise ValueError(f"Malformed API response: missing required fields. Keys: {list(data.keys())}")

    return data


# ── Step 3: Build upsert payloads ────────────────────────────────────────────

def _entropy(all_weights: dict[str, float]) -> float:
    """
    Normalised Shannon entropy across all topic weights.
    0.0 = all weight on one topic (certain).
    1.0 = uniform distribution (maximum uncertainty).

    Replaces the gap-based contestability metric (1 - (top - second))
    which is structurally broken with 30 topics: NMF spreads weight so
    thinly that every article scores 0.9+ regardless of actual certainty.
    Entropy measures the shape of the entire distribution, producing a
    meaningful spread of scores.
    """
    vals = [max(float(v), 1e-12) for v in all_weights.values()]
    total = sum(vals)
    probs = [v / total for v in vals]
    h = -sum(p * math.log(p) for p in probs)
    max_h = math.log(len(probs)) if probs else 1.0
    return round(h / max_h, 6)


def build_payloads(predictions: list[dict], run_id: str) -> list[dict]:
    """Convert API predictions into Supabase upsert payloads."""
    return [
        {
            "id":                    pred["article_id"],
            "dataset_type":          "inference",
            "topic_num":             pred["topic_id"],
            "dominant_topic":        pred["topic_name"],
            "dominant_topic_weight": round(pred["confidence"], 6),
            "topic_probabilities":   pred["all_weights"],
            "contestability_score":  _entropy(pred["all_weights"]),
            "run_id":                run_id,
        }
        for pred in predictions
    ]


# ── Step 4: Write results back to Supabase ───────────────────────────────────

def upsert_results(client: Client, payloads: list[dict]) -> int:
    """Update prediction results row by row. Returns number of failed rows.

    Uses .update().eq() instead of .upsert() because the articles table has a
    NOT NULL constraint on `url`, and inference payloads don't include url.
    Upsert treats missing columns as NULL, triggering the constraint.
    """
    failed = 0
    for payload in payloads:
        row_id = payload["id"]
        update_data = {k: v for k, v in payload.items() if k != "id"}
        try:
            client.table("articles").update(update_data).eq("id", row_id).execute()
        except Exception as e:
            logger.error("Update failed for %s: %s", row_id, e)
            failed += 1
    return failed


# ── Orchestrator ─────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    client = get_supabase_client()

    # 1. Fetch all unprocessed articles
    articles = fetch_unprocessed_articles(client)
    if not articles:
        logger.info("No unprocessed inference articles. Nothing to do.")
        return

    # 2. Group by week
    weekly_groups = group_by_week(articles)
    logger.info(
        "Processing %d articles across %d weeks.\n",
        len(articles), len(weekly_groups),
    )

    total_ok = 0
    total_fail = 0

    # 3. Process each week sequentially
    for week_start, week_end, week_num, week_articles in weekly_groups:
        logger.info(
            "── Week %d (%s → %s): %d articles ──",
            week_num, week_start, week_end, len(week_articles),
        )

        week_payloads: list[dict] = []

        # Send to API in batches
        for i in range(0, len(week_articles), BATCH_SIZE):
            batch = week_articles[i : i + BATCH_SIZE]
            try:
                result = predict_batch(batch)
                payloads = build_payloads(result["predictions"], result["run_id"])
                week_payloads.extend(payloads)
            except (requests.RequestException, ValueError) as e:
                logger.error("API call failed for batch: %s", e)
                total_fail += len(batch)

        # Write this week's results
        if week_payloads:
            failed = upsert_results(client, week_payloads)
            ok = len(week_payloads) - failed
            total_ok += ok
            total_fail += failed
            logger.info(
                "  Week %d done: %d written, %d failed.\n",
                week_num, ok, failed,
            )

    logger.info(
        "Batch run complete. %d/%d articles processed successfully.",
        total_ok, len(articles),
    )
    if total_fail:
        logger.warning("%d articles failed.", total_fail)


if __name__ == "__main__":
    main()
