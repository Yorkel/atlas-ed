"""
drift_monitor.py

Compute weekly drift metrics for NMF inference batches and write
them to the Supabase `drift_metrics` table.

Compares each inference week's topic distribution, confidence, and
contestability against the training baseline to detect model-data
mismatch, confidence degradation, and topic concentration shifts.

Run standalone:
  python -m model_pipeline.inference.drift_monitor
"""

from __future__ import annotations

import logging
import os
from collections import Counter, defaultdict

import numpy as np
from dotenv import load_dotenv
from scipy.spatial.distance import jensenshannon
from supabase import create_client, Client

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

N_TOPICS = 30
FETCH_PAGE_SIZE = 500

# Alert thresholds (informational, not hard failures)
THRESHOLD_JS_DIVERGENCE = 0.1
THRESHOLD_CONFIDENCE_DROP = 0.8       # alert if mean_conf < baseline * this
THRESHOLD_HIGH_CONTESTABILITY = 0.5   # alert if rate exceeds this
THRESHOLD_MIN_TOPICS = 15             # alert if fewer topics present


# ── Supabase client ───────────────────────────────────────────────────────────

def get_supabase_client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in your .env file."
        )
    return create_client(url, key)


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_articles(client: Client, dataset_type: str) -> list[dict]:
    """Paginated fetch of articles by dataset_type."""
    columns = "topic_num, dominant_topic_weight, contestability_score"
    if dataset_type == "inference":
        columns += ", week_number, week_start, week_end, run_id"

    articles: list[dict] = []
    page = 0
    while True:
        start = page * FETCH_PAGE_SIZE
        response = (
            client.table("articles")
            .select(columns)
            .eq("dataset_type", dataset_type)
            .not_.is_("topic_num", "null")
            .range(start, start + FETCH_PAGE_SIZE - 1)
            .execute()
        )
        rows = response.data
        articles.extend(rows)
        if len(rows) < FETCH_PAGE_SIZE:
            break
        page += 1
    return articles


# ── Metric computation ────────────────────────────────────────────────────────

def compute_topic_distribution(articles: list[dict]) -> np.ndarray:
    """Count topic_num occurrences → 30-element probability vector."""
    counts = Counter(a["topic_num"] for a in articles)
    dist = np.zeros(N_TOPICS)
    for topic_num, count in counts.items():
        if 0 <= topic_num < N_TOPICS:
            dist[topic_num] = count
    total = dist.sum()
    if total > 0:
        dist = dist / total
    return dist


def compute_baseline(training_articles: list[dict]) -> dict:
    """Compute aggregate baseline metrics from training data."""
    distribution = compute_topic_distribution(training_articles)

    weights = [a["dominant_topic_weight"] for a in training_articles]
    scores = [a["contestability_score"] for a in training_articles]

    return {
        "distribution": distribution,
        "mean_confidence": float(np.mean(weights)),
        "mean_contestability": float(np.mean(scores)),
        "high_contestability_rate": float(np.mean([s > 0.5 for s in scores])),
    }


def compute_week_metrics(week_articles: list[dict], baseline: dict) -> dict:
    """Compute all drift metrics for a single inference week."""
    week_dist = compute_topic_distribution(week_articles)

    # JS divergence (squared — jensenshannon returns the distance/sqrt)
    js_div = float(jensenshannon(baseline["distribution"], week_dist) ** 2)

    weights = [a["dominant_topic_weight"] for a in week_articles]
    scores = [a["contestability_score"] for a in week_articles]

    return {
        "js_divergence": round(js_div, 6),
        "mean_confidence": round(float(np.mean(weights)), 6),
        "mean_contestability": round(float(np.mean(scores)), 6),
        "high_contestability_rate": round(float(np.mean([s > 0.5 for s in scores])), 6),
        "topic_concentration_hhi": round(float(np.sum(week_dist ** 2)), 6),
        "n_topics_present": int(np.count_nonzero(week_dist)),
    }


def check_alerts(metrics: dict, baseline: dict, week_num: int) -> dict:
    """Apply thresholds, log warnings, add boolean alert flags."""
    alerts = {}

    # JS divergence
    alerts["alert_js_divergence"] = metrics["js_divergence"] > THRESHOLD_JS_DIVERGENCE
    if alerts["alert_js_divergence"]:
        logger.warning(
            "  Week %d: JS divergence (%.4f) exceeds threshold (%.2f)",
            week_num, metrics["js_divergence"], THRESHOLD_JS_DIVERGENCE,
        )

    # Confidence drop
    threshold_conf = baseline["mean_confidence"] * THRESHOLD_CONFIDENCE_DROP
    alerts["alert_confidence_drop"] = metrics["mean_confidence"] < threshold_conf
    if alerts["alert_confidence_drop"]:
        logger.warning(
            "  Week %d: Mean confidence (%.4f) below %.0f%% of baseline (%.4f)",
            week_num, metrics["mean_confidence"],
            THRESHOLD_CONFIDENCE_DROP * 100, baseline["mean_confidence"],
        )

    # High contestability
    alerts["alert_high_contestability"] = (
        metrics["high_contestability_rate"] > THRESHOLD_HIGH_CONTESTABILITY
    )
    if alerts["alert_high_contestability"]:
        logger.warning(
            "  Week %d: High-contestability rate (%.1f%%) exceeds threshold (%.0f%%)",
            week_num, metrics["high_contestability_rate"] * 100,
            THRESHOLD_HIGH_CONTESTABILITY * 100,
        )

    # Low topic coverage
    alerts["alert_low_topic_coverage"] = (
        metrics["n_topics_present"] < THRESHOLD_MIN_TOPICS
    )
    if alerts["alert_low_topic_coverage"]:
        logger.warning(
            "  Week %d: Only %d topics present (threshold: %d)",
            week_num, metrics["n_topics_present"], THRESHOLD_MIN_TOPICS,
        )

    return {**metrics, **alerts}


def group_by_week(articles: list[dict]) -> dict[int, list[dict]]:
    """Group inference articles by week_number."""
    weeks: dict[int, list[dict]] = defaultdict(list)
    for a in articles:
        weeks[a["week_number"]].append(a)
    return dict(weeks)


# ── Write to Supabase ─────────────────────────────────────────────────────────

def upsert_metrics(client: Client, rows: list[dict]) -> None:
    """Upsert drift metrics rows (idempotent on week_number)."""
    try:
        client.table("drift_metrics").upsert(
            rows, on_conflict="week_number"
        ).execute()
    except Exception as e:
        logger.error("Failed to upsert drift metrics: %s", e)
        raise


# ── Orchestrator ──────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    client = get_supabase_client()

    # 1. Fetch data
    logger.info("Fetching training articles...")
    training_articles = _fetch_articles(client, "training")
    logger.info("Fetched %d training articles.", len(training_articles))

    logger.info("Fetching inference articles...")
    inference_articles = _fetch_articles(client, "inference")
    logger.info("Fetched %d inference articles.", len(inference_articles))

    if not inference_articles:
        logger.info("No inference articles found. Nothing to do.")
        return

    # 2. Compute baseline from training data
    baseline = compute_baseline(training_articles)
    logger.info(
        "Training baseline: mean_confidence=%.4f, mean_contestability=%.4f, "
        "high_contestability_rate=%.1f%%",
        baseline["mean_confidence"],
        baseline["mean_contestability"],
        baseline["high_contestability_rate"] * 100,
    )

    # 3. Compute metrics per inference week
    weeks = group_by_week(inference_articles)
    rows: list[dict] = []

    for week_num in sorted(weeks):
        week_articles = weeks[week_num]
        metrics = compute_week_metrics(week_articles, baseline)
        metrics = check_alerts(metrics, baseline, week_num)

        # Add week metadata
        metrics["week_number"] = week_num
        metrics["week_start"] = week_articles[0].get("week_start")
        metrics["week_end"] = week_articles[0].get("week_end")
        metrics["n_articles"] = len(week_articles)
        metrics["run_id"] = week_articles[0].get("run_id")

        rows.append(metrics)

        logger.info(
            "  Week %d (%d articles): JS=%.4f  conf=%.4f  contest=%.4f  "
            "high_contest=%.1f%%  HHI=%.4f  topics=%d",
            week_num, len(week_articles),
            metrics["js_divergence"],
            metrics["mean_confidence"],
            metrics["mean_contestability"],
            metrics["high_contestability_rate"] * 100,
            metrics["topic_concentration_hhi"],
            metrics["n_topics_present"],
        )

    # 4. Write to Supabase
    upsert_metrics(client, rows)
    logger.info("Drift metrics written for %d weeks.", len(rows))

    # 5. Summary
    any_alerts = any(
        row.get("alert_js_divergence") or row.get("alert_confidence_drop")
        or row.get("alert_high_contestability") or row.get("alert_low_topic_coverage")
        for row in rows
    )
    if any_alerts:
        logger.warning("Alerts were triggered — review warnings above.")
    else:
        logger.info("No alerts triggered. All metrics within expected ranges.")


if __name__ == "__main__":
    main()
