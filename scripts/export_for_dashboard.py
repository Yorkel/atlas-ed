"""
Export all data needed for the AtlasED dashboard repo.

Creates a dashboard_export/ directory with everything the dashboard needs:
- Enriched model JSONs (with top keywords)
- Analysis-ready CSVs (articles + topic assignments + probabilities)
- Temporal aggregation CSV
- RAG comparison JSON
- FAISS indexes
- Evaluation metrics (coherence, stability)
- Visualisation PNGs
- Topic names per model
- Supabase connection info template

Usage:
    python scripts/export_for_dashboard.py
    python scripts/export_for_dashboard.py --output-dir /path/to/dashboard/data
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_OUTPUTS = REPO_ROOT / "data" / "evaluation_outputs"
RUNS_DIR = REPO_ROOT / "experiments" / "outputs" / "runs"
RAG_DIR = REPO_ROOT / "data" / "rag"

# Model run mappings — latest run for each model variant
MODEL_RUNS = {
    "eng_k5": {
        "run_id": "2026-03-24_104906",
        "summary_json": "nmf_eng_5_summary.json",
        "llm_review": "llm_topic_review_k5.json",
        "has_joblib": True,
    },
    "eng_k15": {
        "run_id": "2026-03-24_112503",
        "summary_json": "nmf_eng_15_summary.json",
        "llm_review": "llm_topic_review_k15.json",
        "has_joblib": True,
    },
    "eng_k30": {
        "run_id": "eng_2026-03-24_134826",
        "summary_json": "nmf_eng_30_summary.json",
        "llm_review": "llm_topic_review_k30.json",
        "has_joblib": True,
    },
    "eng_k30_nm": {
        "run_id": "2026-03-24_013511",
        "summary_json": "nmf_eng_30_nm_summary.json",
        "llm_review": "llm_topic_review_k30_nm.json",
        "has_joblib": True,
        "topic_summary_run": "2026-03-24_174315",  # has curated names
    },
}

# Cross-jurisdiction models
CROSS_JURISDICTION_RUNS = {
    "sco_k15": {
        "run_id": "sco_2026-03-24_135938",
    },
    "irl_k15": {
        "run_id": "irl_2026-03-24_140040",
    },
}

# Analysis-ready CSVs
ANALYSIS_CSVS = [
    "topics_analysis_ready_eng.csv",
    "topics_analysis_ready_sco.csv",
    "topics_analysis_ready_irl.csv",
]

# Visualisation PNGs to include
VIZ_PNGS = [
    "4model_source_concentration.png",
    "4model_topic_distributions.png",
    "4model_weights.png",
    "trends_eng_k30.png",
    "temporal_spikes_eng_k30.png",
    "topic_growth_eng_k30.png",
    "source_topic_heatmap_eng_k30.png",
    "contestability_eng_k30.png",
    "topic_cooccurrence_eng_k30.png",
    "topic_distribution_eng_k30.png",
    "coherence_sweep_eng_k30.png",
    "framing_ai_edtech_eng_k30.png",
]

# Evaluation metric CSVs
EVAL_CSVS = [
    "coherence_sweep_eng.csv",
    "coherence_sweep_sco.csv",
    "coherence_sweep_irl.csv",
    "stability_seeds_eng.csv",
    "stability_seeds_sco.csv",
    "stability_seeds_irl.csv",
]


def extract_keywords_from_joblib(run_dir: Path, n_words: int = 10) -> dict:
    """Extract top keywords per topic from a trained NMF model."""
    import joblib

    model = joblib.load(run_dir / "nmf_model.joblib")
    vectorizer = joblib.load(run_dir / "vectorizer.joblib")
    feature_names = vectorizer.get_feature_names_out()

    keywords = {}
    for topic_idx, topic_weights in enumerate(model.components_):
        top_indices = topic_weights.argsort()[: -n_words - 1 : -1]
        keywords[topic_idx] = [feature_names[i] for i in top_indices]

    return keywords


def build_enriched_model_json(model_key: str, config: dict, out_dir: Path) -> None:
    """Build an enriched model JSON with keywords, names, and metrics."""
    run_dir = RUNS_DIR / config["run_id"]

    # Load base summary
    summary = json.loads((EVAL_OUTPUTS / config["summary_json"]).read_text())

    # Load topic names from the run (or from topic_summary_run if specified)
    names_run = config.get("topic_summary_run", config["run_id"])
    names_path = RUNS_DIR / names_run / "topic_names.json"
    topic_names = json.loads(names_path.read_text())

    # Load run metadata
    metadata = json.loads((run_dir / "run_metadata.json").read_text())

    # Extract keywords
    keywords = {}
    if config.get("has_joblib"):
        try:
            keywords = extract_keywords_from_joblib(run_dir)
        except Exception as e:
            print(f"  Warning: could not extract keywords for {model_key}: {e}")

    # Enrich each topic with keywords and ensure names match
    for topic in summary["topics"]:
        tid = topic["topic_num"]
        # Add keywords
        if tid in keywords:
            topic["top_keywords"] = keywords[tid]
        # Ensure topic name matches the curated names
        if str(tid) in topic_names:
            topic["name"] = topic_names[str(tid)]

    # Add run metadata
    summary["run_id"] = config["run_id"]
    summary["run_metadata"] = {
        "reconstruction_error": metadata.get("reconstruction_error"),
        "n_docs": metadata.get("n_docs") or metadata.get("n_articles"),
        "tfidf_max_features": metadata.get("tfidf_max_features"),
        "tfidf_ngram_range": metadata.get("tfidf_ngram_range"),
        "nmf_init": metadata.get("nmf_init") or metadata.get("init"),
        "nmf_random_state": metadata.get("nmf_random_state") or metadata.get("random_state"),
    }

    out_path = out_dir / f"model_{model_key}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"  Wrote {out_path.name}")


def generate_temporal_csv(out_dir: Path) -> None:
    """Generate monthly topic counts from analysis-ready CSV."""
    src = EVAL_OUTPUTS / "topics_analysis_ready_eng.csv"
    df = pd.read_csv(src, usecols=["article_date", "year", "month", "topic_num", "topic_name"])

    # Monthly aggregation
    monthly = (
        df.groupby(["year", "month", "topic_num", "topic_name"])
        .size()
        .reset_index(name="article_count")
    )
    monthly = monthly.sort_values(["year", "month", "topic_num"])

    out_path = out_dir / "topic_timeseries_eng.csv"
    monthly.to_csv(out_path, index=False)
    print(f"  Wrote {out_path.name} ({len(monthly)} rows)")


def copy_cross_jurisdiction(out_dir: Path) -> None:
    """Copy cross-jurisdiction model artifacts."""
    cross_dir = out_dir / "cross_jurisdiction"
    cross_dir.mkdir(exist_ok=True)

    for model_key, config in CROSS_JURISDICTION_RUNS.items():
        run_dir = RUNS_DIR / config["run_id"]
        model_dir = cross_dir / model_key
        model_dir.mkdir(exist_ok=True)

        for fname in ["topic_names.json", "run_metadata.json"]:
            src = run_dir / fname
            if src.exists():
                shutil.copy2(src, model_dir / fname)

        # Copy analysis CSV
        country = model_key.split("_")[0]
        csv_name = f"topics_analysis_ready_{country}.csv"
        csv_src = EVAL_OUTPUTS / csv_name
        if csv_src.exists():
            shutil.copy2(csv_src, cross_dir / csv_name)

    print(f"  Wrote cross-jurisdiction data for {list(CROSS_JURISDICTION_RUNS.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Export data for AtlasED dashboard")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "dashboard_export",
        help="Output directory (default: ./dashboard_export/)",
    )
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to: {out}\n")

    # --- 1. Enriched model JSONs ---
    print("1. Enriched model JSONs (with keywords, metrics)...")
    models_dir = out / "models"
    models_dir.mkdir(exist_ok=True)
    for model_key, config in MODEL_RUNS.items():
        build_enriched_model_json(model_key, config, models_dir)

    # --- 2. Analysis-ready CSVs ---
    print("\n2. Analysis-ready CSVs...")
    articles_dir = out / "articles"
    articles_dir.mkdir(exist_ok=True)
    for csv_name in ANALYSIS_CSVS:
        src = EVAL_OUTPUTS / csv_name
        if src.exists():
            shutil.copy2(src, articles_dir / csv_name)
            n_rows = sum(1 for _ in open(src)) - 1
            print(f"  Copied {csv_name} ({n_rows} rows)")
        else:
            print(f"  Warning: {csv_name} not found")

    # --- 3. Temporal aggregation ---
    print("\n3. Temporal aggregation...")
    generate_temporal_csv(articles_dir)

    # --- 4. RAG data ---
    print("\n4. RAG data...")
    rag_out = out / "rag"
    rag_out.mkdir(exist_ok=True)
    rag_src = EVAL_OUTPUTS / "rag_comparison.json"
    if rag_src.exists():
        shutil.copy2(rag_src, rag_out / "rag_comparison.json")
        print(f"  Copied rag_comparison.json")
    # FAISS indexes
    for fname in ["full_corpus.faiss", "nm_corpus.faiss"]:
        src = RAG_DIR / fname
        if src.exists():
            shutil.copy2(src, rag_out / fname)
            print(f"  Copied {fname} ({src.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  Warning: {fname} not found")

    # --- 5. LLM topic reviews ---
    print("\n5. LLM topic reviews...")
    reviews_dir = out / "llm_reviews"
    reviews_dir.mkdir(exist_ok=True)
    for config in MODEL_RUNS.values():
        fname = config["llm_review"]
        src = EVAL_OUTPUTS / fname
        if src.exists():
            shutil.copy2(src, reviews_dir / fname)
            print(f"  Copied {fname}")

    # --- 6. Evaluation metrics ---
    print("\n6. Evaluation metrics...")
    eval_dir = out / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    for csv_name in EVAL_CSVS:
        src = EVAL_OUTPUTS / csv_name
        if src.exists():
            shutil.copy2(src, eval_dir / csv_name)
            print(f"  Copied {csv_name}")

    # --- 7. Visualisations ---
    print("\n7. Visualisations...")
    viz_dir = out / "visualisations"
    viz_dir.mkdir(exist_ok=True)
    for png_name in VIZ_PNGS:
        src = EVAL_OUTPUTS / png_name
        if src.exists():
            shutil.copy2(src, viz_dir / png_name)
            print(f"  Copied {png_name}")
        else:
            print(f"  Skipped {png_name} (not found)")

    # --- 8. Framing analysis ---
    print("\n8. Framing analysis...")
    framing_src = EVAL_OUTPUTS / "framing_ai_edtech.json"
    if framing_src.exists():
        shutil.copy2(framing_src, out / "framing_ai_edtech.json")
        print(f"  Copied framing_ai_edtech.json")

    # --- 9. Cross-jurisdiction ---
    print("\n9. Cross-jurisdiction data...")
    copy_cross_jurisdiction(out)

    # --- 10. Supabase connection template ---
    print("\n10. Supabase connection template...")
    env_template = out / ".env.example"
    env_template.write_text(
        "# Supabase connection — same project as the topic modelling pipeline\n"
        "SUPABASE_URL=https://your-project.supabase.co\n"
        "SUPABASE_KEY=your-anon-key\n"
        "\n"
        "# Tables used by the dashboard:\n"
        "#   articles_topics  — article-level topic assignments (written by s11_supabase_writer.py)\n"
        "#   drift_metrics    — monthly drift monitoring (written by drift_monitor.py)\n"
        "\n"
        "# For live RAG (optional)\n"
        "ANTHROPIC_API_KEY=your-api-key\n"
    )
    print(f"  Wrote .env.example")

    # --- 11. Manifest ---
    print("\n11. Export manifest...")
    manifest = {
        "exported_at": pd.Timestamp.now().isoformat(),
        "source_repo": "AM1_topic_modelling",
        "models": {k: v["run_id"] for k, v in MODEL_RUNS.items()},
        "cross_jurisdiction": {k: v["run_id"] for k, v in CROSS_JURISDICTION_RUNS.items()},
        "contents": {
            "models/": "Enriched model JSONs with topic stats, keywords, and run metadata",
            "articles/": "Analysis-ready CSVs with article-level topic assignments and probabilities",
            "articles/topic_timeseries_eng.csv": "Monthly topic counts for trend visualisation",
            "rag/": "RAG comparison JSON and FAISS vector indexes",
            "llm_reviews/": "LLM-generated topic name reviews and suggestions",
            "evaluation/": "Coherence sweep and stability seed CSVs",
            "visualisations/": "Pre-rendered charts (PNG)",
            "cross_jurisdiction/": "Scotland and Ireland model artifacts and analysis CSVs",
            "framing_ai_edtech.json": "Framing analysis for AI/EdTech topic",
            ".env.example": "Supabase and API key template",
        },
    }
    manifest_path = out / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  Wrote MANIFEST.json")

    # --- Summary ---
    total_files = sum(1 for _ in out.rglob("*") if _.is_file())
    total_size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"\nDone! Exported {total_files} files ({total_size / 1024 / 1024:.1f} MB) to {out}/")


if __name__ == "__main__":
    main()
