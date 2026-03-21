"""
drift_monitor.py

Compute Jensen-Shannon divergence between the England training baseline
and each country's topic distribution per monitoring period.

Writes results to Supabase `drift_metrics` table.

Usage:
  python -m model_pipeline.inference.drift_monitor
  python -m model_pipeline.inference.drift_monitor --backfill-only

Requires .env with SUPABASE_URL and SUPABASE_SERVICE_KEY.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from scipy.spatial.distance import jensenshannon
from supabase import create_client, Client

from model_pipeline.training.s06_topic_allocation import TOPIC_NAMES

load_dotenv()

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1000
TOPIC_COLS = list(TOPIC_NAMES.values())
N_TOPICS = len(TOPIC_COLS)


def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.")
    return create_client(url, key)


def fetch_topic_probabilities(client: Client, country: str = None, dataset_type: str = None) -> list[dict]:
    """Fetch topic_probabilities from articles_topics with optional filters."""
    articles = []
    page = 0
    while True:
        start = page * CHUNK_SIZE
        query = client.table("articles_topics").select("topic_probabilities, country, week_number, dataset_type")
        if country:
            query = query.eq("country", country)
        if dataset_type:
            query = query.eq("dataset_type", dataset_type)

        response = query.range(start, start + CHUNK_SIZE - 1).execute()
        articles.extend(response.data)
        if len(response.data) < CHUNK_SIZE:
            break
        page += 1
    return articles


def compute_topic_distribution(articles: list[dict]) -> np.ndarray:
    """Compute average topic distribution from a list of articles with topic_probabilities."""
    if not articles:
        return np.zeros(N_TOPICS)

    vectors = []
    for article in articles:
        probs = article.get("topic_probabilities", {})
        if not probs:
            continue
        vec = np.array([probs.get(col, 0.0) for col in TOPIC_COLS], dtype=float)
        vectors.append(vec)

    if not vectors:
        return np.zeros(N_TOPICS)

    avg = np.mean(vectors, axis=0)
    total = avg.sum()
    if total > 0:
        avg = avg / total
    return avg


def compute_baseline(client: Client) -> np.ndarray:
    """Compute the England training baseline distribution."""
    logger.info("Computing England training baseline...")
    articles = fetch_topic_probabilities(client, country="eng", dataset_type="training")
    baseline = compute_topic_distribution(articles)
    logger.info("Baseline computed from %d articles.", len(articles))
    return baseline


def compute_js_divergence(dist: np.ndarray, baseline: np.ndarray) -> float:
    """Compute JS divergence between two distributions. Returns 0-1."""
    eps = 1e-10
    p = dist + eps
    q = baseline + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q) ** 2)


def write_drift_metric(client: Client, row: dict) -> None:
    """Write a single drift metric row to Supabase."""
    try:
        client.table("drift_metrics").insert(row).execute()
    except Exception as e:
        logger.error("Failed to write drift metric: %s", e)


def run_weekly_drift(client: Client, baseline: np.ndarray, run_id: str) -> None:
    """Compute drift for each country for each week."""
    articles = fetch_topic_probabilities(client, dataset_type="inference")
    weekly_articles = [a for a in articles if a.get("week_number") is not None]

    # Group by country and week
    groups = defaultdict(list)
    for a in weekly_articles:
        key = (a["country"], int(a["week_number"]))
        groups[key].append(a)

    logger.info("Found %d country-week groups.", len(groups))

    for (country, week_num), group_articles in sorted(groups.items()):
        dist = compute_topic_distribution(group_articles)
        js = compute_js_divergence(dist, baseline)

        # Count unique dominant topics
        dominant_topics = set()
        for a in group_articles:
            probs = a.get("topic_probabilities", {})
            if probs:
                max_topic = max(probs, key=probs.get)
                dominant_topics.add(max_topic)

        row = {
            "week_number": week_num,
            "country": country,
            "js_divergence": round(js, 6),
            "article_count": len(group_articles),
            "unique_topics": len(dominant_topics),
            "model_type": "nmf",
            "period_type": "week",
            "run_id": run_id,
        }

        write_drift_metric(client, row)
        logger.info(
            "  %s week %d: JS=%.4f, articles=%d, unique_topics=%d",
            country, week_num, js, len(group_articles), len(dominant_topics),
        )


def run_backfill_drift(client: Client, baseline: np.ndarray, run_id: str) -> None:
    """Compute drift for backfill data (Scotland and Ireland 2023-2025 as single periods)."""
    for country in ["sco", "irl"]:
        articles = fetch_topic_probabilities(client, country=country, dataset_type="inference")
        backfill = [a for a in articles if a.get("week_number") is None]

        if not backfill:
            logger.info("No backfill data for %s.", country)
            continue

        dist = compute_topic_distribution(backfill)
        js = compute_js_divergence(dist, baseline)

        dominant_topics = set()
        for a in backfill:
            probs = a.get("topic_probabilities", {})
            if probs:
                max_topic = max(probs, key=probs.get)
                dominant_topics.add(max_topic)

        row = {
            "week_number": None,
            "country": country,
            "js_divergence": round(js, 6),
            "article_count": len(backfill),
            "unique_topics": len(dominant_topics),
            "model_type": "nmf",
            "period_type": "backfill",
            "run_id": run_id,
        }

        write_drift_metric(client, row)
        logger.info(
            "  %s backfill: JS=%.4f, articles=%d, unique_topics=%d",
            country, js, len(backfill), len(dominant_topics),
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Compute drift metrics per country per period")
    parser.add_argument("--backfill-only", action="store_true", help="Only compute backfill drift")
    args = parser.parse_args()

    client = get_supabase_client()
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    baseline = compute_baseline(client)

    logger.info("\n── Backfill drift ──")
    run_backfill_drift(client, baseline, run_id)

    if not args.backfill_only:
        logger.info("\n── Weekly drift ──")
        run_weekly_drift(client, baseline, run_id)

    print("\n✅ Drift monitoring complete.")


if __name__ == "__main__":
    main()
