"""
Load AtlasED dashboard data into Supabase (hybrid approach).

Uses existing articles_topics table + new atlased_* tables in the public schema.
(Supabase Python client doesn't support custom schemas via .table(),
so we use atlased_ prefix in the public schema instead.)

Prerequisites:
  1. Run the SQL migration in Supabase SQL Editor (see below)
  2. Set SUPABASE_URL and SUPABASE_KEY in .env
  3. Run: python scripts/export_for_dashboard.py (to create dashboard_export/)

Usage:
    PYTHONPATH=. python scripts/load_atlased_to_supabase.py
    PYTHONPATH=. python scripts/load_atlased_to_supabase.py --verify-only

SQL Migration (run in Supabase SQL Editor first):
-----------------------------------------------
CREATE TABLE IF NOT EXISTS atlased_models (
    model_id TEXT PRIMARY KEY,
    run_id TEXT UNIQUE NOT NULL,
    k INTEGER NOT NULL,
    n_articles INTEGER NOT NULL,
    stability DECIMAL(5,4),
    mean_weight DECIMAL(5,4),
    coherence DECIMAL(5,4),
    corpus_type TEXT NOT NULL,
    trained_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS atlased_topics (
    id SERIAL PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES atlased_models(model_id),
    topic_num INTEGER NOT NULL,
    name TEXT NOT NULL,
    top_keywords TEXT[],
    article_count INTEGER,
    pct DECIMAL(5,2),
    top_source TEXT,
    top_source_pct DECIMAL(5,4),
    single_source BOOLEAN DEFAULT FALSE,
    UNIQUE(model_id, topic_num)
);

CREATE INDEX IF NOT EXISTS idx_atlased_topics_model ON atlased_topics(model_id);

CREATE TABLE IF NOT EXISTS atlased_topic_timeseries (
    id SERIAL PRIMARY KEY,
    model_id TEXT NOT NULL,
    topic_num INTEGER NOT NULL,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    topic_name TEXT,
    article_count INTEGER NOT NULL,
    UNIQUE(model_id, topic_num, year, month)
);

CREATE INDEX IF NOT EXISTS idx_atlased_ts_lookup
    ON atlased_topic_timeseries(model_id, year, month);

CREATE TABLE IF NOT EXISTS atlased_rag_contexts (
    id SERIAL PRIMARY KEY,
    corpus_type TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources_used TEXT[],
    topics_covered TEXT[],
    n_retrieved INTEGER,
    UNIQUE(corpus_type, question)
);

-- RLS: public read access
ALTER TABLE atlased_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE atlased_topics ENABLE ROW LEVEL SECURITY;
ALTER TABLE atlased_topic_timeseries ENABLE ROW LEVEL SECURITY;
ALTER TABLE atlased_rag_contexts ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow public read" ON atlased_models FOR SELECT USING (true);
CREATE POLICY "Allow public read" ON atlased_topics FOR SELECT USING (true);
CREATE POLICY "Allow public read" ON atlased_topic_timeseries FOR SELECT USING (true);
CREATE POLICY "Allow public read" ON atlased_rag_contexts FOR SELECT USING (true);

-- Speed up dashboard queries on existing table
CREATE INDEX IF NOT EXISTS idx_articles_topics_run_source
    ON articles_topics(run_id, source);
CREATE INDEX IF NOT EXISTS idx_articles_topics_run_date
    ON articles_topics(run_id, article_date);
-----------------------------------------------
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")

EXPORT_DIR = Path(__file__).resolve().parent.parent / "dashboard_export"

# Mapping from export file key to dashboard model_id and actual run_id
MODEL_MAP = {
    "model_eng_k5.json": {
        "model_id": "eng_k5",
        "run_id": "2026-03-24_104906",
        "corpus_type": "full",
    },
    "model_eng_k15.json": {
        "model_id": "eng_k15",
        "run_id": "2026-03-24_112503",
        "corpus_type": "full",
    },
    "model_eng_k30.json": {
        "model_id": "eng_k30",
        "run_id": "eng_2026-03-24_134826",
        "corpus_type": "full",
    },
    "model_eng_k30_nm.json": {
        "model_id": "eng_k30_nm",
        "run_id": "2026-03-24_013511",
        "corpus_type": "no_media",
    },
}


def get_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SystemExit("Set SUPABASE_URL and SUPABASE_KEY in .env")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def load_models(sb: Client) -> None:
    print("Loading models + topics...")

    for filename, mapping in MODEL_MAP.items():
        filepath = EXPORT_DIR / "models" / filename
        if not filepath.exists():
            print(f"  Warning: {filename} not found, skipping")
            continue

        data = json.loads(filepath.read_text())

        # Insert model metadata
        model_record = {
            "model_id": mapping["model_id"],
            "run_id": mapping["run_id"],
            "k": data["n_topics"],
            "n_articles": data["n_articles"],
            "stability": data.get("metrics", {}).get("stability"),
            "mean_weight": data.get("metrics", {}).get("mean_dominant_weight"),
            "coherence": None,  # not in summary JSON
            "corpus_type": mapping["corpus_type"],
        }

        sb.table("atlased_models").upsert(model_record).execute()
        print(f"  {mapping['model_id']} -> run_id: {mapping['run_id']}")

        # Insert topics
        topics_batch = []
        for topic in data.get("topics", []):
            topics_batch.append({
                "model_id": mapping["model_id"],
                "topic_num": topic["topic_num"],
                "name": topic["name"],
                "top_keywords": topic.get("top_keywords", []),
                "article_count": topic["count"],
                "pct": topic["pct"],
                "top_source": topic.get("top_source"),
                "top_source_pct": topic.get("top_source_pct"),
                "single_source": topic.get("single_source", False),
            })

        if topics_batch:
            # Upsert in chunks to avoid payload limits
            for i in range(0, len(topics_batch), 50):
                sb.table("atlased_topics").upsert(topics_batch[i:i + 50]).execute()
            print(f"  Loaded {len(topics_batch)} topics")


def load_timeseries(sb: Client) -> None:
    print("\nLoading timeseries...")

    ts_file = EXPORT_DIR / "articles" / "topic_timeseries_eng.csv"
    if not ts_file.exists():
        print(f"  Warning: {ts_file} not found")
        return

    df = pd.read_csv(ts_file)

    batch = []
    for _, row in df.iterrows():
        batch.append({
            "model_id": "eng_k30",  # timeseries is for baseline
            "topic_num": int(row["topic_num"]),
            "year": int(row["year"]),
            "month": int(row["month"]),
            "topic_name": row["topic_name"],
            "article_count": int(row["article_count"]),
        })

        if len(batch) >= 500:
            sb.table("atlased_topic_timeseries").upsert(batch).execute()
            print(f"  Loaded {len(batch)} records")
            batch = []

    if batch:
        sb.table("atlased_topic_timeseries").upsert(batch).execute()
        print(f"  Loaded {len(batch)} records")

    print(f"  Total: {len(df)} timeseries rows")


def load_rag_contexts(sb: Client) -> None:
    print("\nLoading RAG contexts...")

    rag_file = EXPORT_DIR / "rag" / "rag_comparison.json"
    if not rag_file.exists():
        print(f"  Warning: {rag_file} not found")
        return

    data = json.loads(rag_file.read_text())
    questions = data.get("questions", [])

    batch = []
    for qa in questions:
        question = qa["question"]

        # Full corpus answer
        full = qa.get("full", {})
        batch.append({
            "corpus_type": "full",
            "question": question,
            "answer": full.get("answer", ""),
            "sources_used": full.get("sources_used", []),
            "topics_covered": full.get("topics_covered", []),
            "n_retrieved": full.get("n_retrieved"),
        })

        # No-media corpus answer
        nm = qa.get("nm", {})
        batch.append({
            "corpus_type": "no_media",
            "question": question,
            "answer": nm.get("answer", ""),
            "sources_used": nm.get("sources_used", []),
            "topics_covered": nm.get("topics_covered", []),
            "n_retrieved": nm.get("n_retrieved"),
        })

    if batch:
        sb.table("atlased_rag_contexts").upsert(batch).execute()
        print(f"  Loaded {len(batch)} contexts ({len(questions)} questions x 2 corpora)")


def verify(sb: Client) -> None:
    print("\n" + "=" * 50)
    print("VERIFICATION")
    print("=" * 50)

    models = sb.table("atlased_models").select("model_id, run_id, k").execute()
    print(f"\natlased_models: {len(models.data)} rows")
    for m in models.data:
        print(f"  {m['model_id']} (k={m['k']}) -> {m['run_id']}")

    topics = sb.table("atlased_topics").select("id", count="exact").execute()
    print(f"\natlased_topics: {topics.count} rows")

    ts = sb.table("atlased_topic_timeseries").select("id", count="exact").execute()
    print(f"atlased_topic_timeseries: {ts.count} rows")

    rag = sb.table("atlased_rag_contexts").select("id", count="exact").execute()
    print(f"atlased_rag_contexts: {rag.count} rows")

    # Check existing articles_topics
    try:
        articles = sb.table("articles_topics").select("url", count="exact").execute()
        print(f"\narticles_topics (existing): {articles.count} rows")
    except Exception:
        print("\narticles_topics: could not query (may need different permissions)")


def main():
    parser = argparse.ArgumentParser(description="Load AtlasED data to Supabase")
    parser.add_argument("--verify-only", action="store_true", help="Only run verification")
    args = parser.parse_args()

    sb = get_client()

    if args.verify_only:
        verify(sb)
        return

    print("Loading AtlasED dashboard data (hybrid approach)\n")
    print(f"Export dir: {EXPORT_DIR}")
    print("Strategy: existing articles_topics + new atlased_* tables\n")

    load_models(sb)
    load_timeseries(sb)
    load_rag_contexts(sb)
    verify(sb)

    print("\nDone! Dashboard can now query:")
    print("  - atlased_models / atlased_topics for model metadata")
    print("  - atlased_topic_timeseries for trend charts")
    print("  - atlased_rag_contexts for RAG dual-panel")
    print("  - articles_topics (existing) for article-level data")


if __name__ == "__main__":
    main()
