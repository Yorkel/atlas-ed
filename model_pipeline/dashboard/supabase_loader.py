"""
Shared data loader for the Streamlit dashboard.

Strategy: read from a local parquet snapshot for instant loading.
The snapshot is refreshed after each weekly batch run via `refresh_snapshot()`.
Falls back to live Supabase queries if the snapshot doesn't exist.
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
from supabase import create_client, Client


_SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
_SNAPSHOT_PATH = _SNAPSHOT_DIR / "articles.parquet"
_SNAPSHOT_PROBS_PATH = _SNAPSHOT_DIR / "articles_with_probs.parquet"


# ── Supabase client ──────────────────────────────────────────────────────────

def _get_credentials() -> tuple[str, str]:
    """Resolve Supabase credentials from Streamlit secrets or env vars."""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_SERVICE_KEY"]
        return url, key
    except (KeyError, FileNotFoundError, AttributeError, TypeError):
        pass

    from dotenv import load_dotenv
    load_dotenv()

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        raise RuntimeError(
            "Supabase credentials not found. Set SUPABASE_URL and "
            "SUPABASE_SERVICE_KEY in .env or Streamlit secrets."
        )
    return url, key


@st.cache_resource
def get_client() -> Client:
    url, key = _get_credentials()
    return create_client(url, key)


# ── Column selection ─────────────────────────────────────────────────────────

_ARTICLE_COLUMNS = (
    "id, source, article_type, article_date, election_period, "
    "topic_num, dominant_topic, dominant_topic_weight, "
    "preview, text_clean, contestability_score, dataset_type"
)

_ARTICLE_COLUMNS_WITH_PROBS = (
    "id, source, article_type, article_date, election_period, "
    "topic_num, dominant_topic, dominant_topic_weight, "
    "preview, text_clean, contestability_score, dataset_type, "
    "topic_probabilities"
)


# ── Paginated fetch ──────────────────────────────────────────────────────────

def _fetch_all(client: Client, columns: str) -> list[dict]:
    """Paginate through the articles table (Supabase default limit is 1000)."""
    PAGE_SIZE = 500
    all_rows: list[dict] = []
    page = 0
    while True:
        start = page * PAGE_SIZE
        resp = (
            client.table("articles")
            .select(columns)
            .not_.is_("dominant_topic", "null")
            .range(start, start + PAGE_SIZE - 1)
            .execute()
        )
        all_rows.extend(resp.data)
        if len(resp.data) < PAGE_SIZE:
            break
        page += 1
    return all_rows


# ── Transformations ──────────────────────────────────────────────────────────

def _transform(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and apply transformations to match dashboard expectations."""
    df = df.rename(columns={
        "id": "article_id",
        "dominant_topic": "topic_name",
        "article_type": "type",
        "article_date": "date",
    })

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["country"] = "England"
    df["topic_name"] = df["topic_name"].str.replace("_", " ").str.title()
    df["source"] = df["source"].str.upper()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    return df


# ── Snapshot refresh (run after batch_runner) ────────────────────────────────

def refresh_snapshot():
    """Fetch all data from Supabase and save as parquet snapshots.

    Run this after the weekly batch to update the dashboard cache:
        python -c "from model_pipeline.dashboard.supabase_loader import refresh_snapshot; refresh_snapshot()"
    """
    _SNAPSHOT_DIR.mkdir(exist_ok=True)
    client = get_client()

    # Standard articles snapshot
    rows = _fetch_all(client, _ARTICLE_COLUMNS)
    df = pd.DataFrame(rows)
    df = _transform(df)
    df.to_parquet(_SNAPSHOT_PATH, index=False)
    print(f"Snapshot saved: {len(df)} articles → {_SNAPSHOT_PATH}")

    # Articles with probabilities snapshot
    rows_probs = _fetch_all(client, _ARTICLE_COLUMNS_WITH_PROBS)
    df_probs = pd.DataFrame(rows_probs)
    probs = df_probs["topic_probabilities"].apply(
        lambda x: x if isinstance(x, dict) else {}
    )
    probs_df = pd.json_normalize(probs)
    topic_cols = sorted(probs_df.columns.tolist())
    df_probs = pd.concat([df_probs.drop(columns=["topic_probabilities"]), probs_df], axis=1)
    df_probs = _transform(df_probs)
    df_probs.to_parquet(_SNAPSHOT_PROBS_PATH, index=False)
    print(f"Probs snapshot saved: {len(df_probs)} articles, {len(topic_cols)} topic cols → {_SNAPSHOT_PROBS_PATH}")

    return df


# ── Public loaders ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_articles() -> pd.DataFrame:
    """Load articles from parquet snapshot (fast) or Supabase fallback (slow)."""
    if _SNAPSHOT_PATH.exists():
        df = pd.read_parquet(_SNAPSHOT_PATH)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        return df

    # Fallback: live Supabase query
    client = get_client()
    rows = _fetch_all(client, _ARTICLE_COLUMNS)
    df = pd.DataFrame(rows)
    return _transform(df)


@st.cache_data(ttl=3600)
def load_articles_with_probabilities() -> tuple[pd.DataFrame, list[str]]:
    """Load articles + topic probability matrix from snapshot or Supabase."""
    if _SNAPSHOT_PROBS_PATH.exists():
        df = pd.read_parquet(_SNAPSHOT_PROBS_PATH)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        # Identify topic columns (everything not in the standard set)
        standard_cols = {
            "article_id", "source", "type", "date", "election_period",
            "topic_num", "topic_name", "dominant_topic_weight",
            "preview", "text_clean", "contestability_score", "dataset_type",
            "country", "year", "month",
        }
        topic_cols = sorted([c for c in df.columns if c not in standard_cols])
        return df, topic_cols

    # Fallback: live Supabase query
    client = get_client()
    rows = _fetch_all(client, _ARTICLE_COLUMNS_WITH_PROBS)
    df = pd.DataFrame(rows)

    probs = df["topic_probabilities"].apply(
        lambda x: x if isinstance(x, dict) else {}
    )
    probs_df = pd.json_normalize(probs)
    topic_cols = sorted(probs_df.columns.tolist())
    df = pd.concat([df.drop(columns=["topic_probabilities"]), probs_df], axis=1)

    df = _transform(df)
    return df, topic_cols
