"""
s10_pipeline.py

Step 10: End-to-end training pipeline runner (S01 → S09)

Run:
    python -m model_pipeline.training.s10_pipeline                  # train default country (from config)
    python -m model_pipeline.training.s10_pipeline --country sco    # train specific country
    python -m model_pipeline.training.s10_pipeline --all            # train all three countries
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml") as _f:
    CONFIG = yaml.safe_load(_f)


def train_country(country: str) -> None:
    from model_pipeline.training.s01_data_loader import load_articles
    from model_pipeline.training.s02_cleaning import run_cleaning
    from model_pipeline.training.s03_spacy_processing import run_spacy_processing
    from model_pipeline.training.s04_vectorisation import run_vectorisation, build_vectorizer
    from model_pipeline.training.s05_nmf_training import train_nmf
    from model_pipeline.training.s06_topic_allocation import run_topic_allocation, export_analysis_ready_csv, load_topic_names, make_generic_topic_names
    from model_pipeline.training.s07_evaluation import (
        evaluate_coherence_over_topic_range,
        evaluate_topic_stability,
    )
    from model_pipeline.training.s08_save_outputs import save_run_outputs, RUNS_DIR, make_run_id
    from model_pipeline.training.s09_mlflow_logging import log_run_to_mlflow
    from model_pipeline.training.s11_supabase_writer import write_topic_results

    country_cfg = CONFIG["countries"][country]
    dataset_name = country_cfg["dataset_name"]
    n_topics = country_cfg["n_topics"]
    tfidf_cfg = country_cfg["tfidf"]

    run_name = make_run_id(country)
    run_dir = RUNS_DIR / run_name
    logger.info("Pipeline run_name=%s country=%s n_topics=%d", run_name, country, n_topics)

    # S01–S05
    df = load_articles(dataset_name)
    df = run_cleaning(df)
    df = run_spacy_processing(df)
    vectorizer = build_vectorizer(
        min_df=tfidf_cfg["min_df"],
        max_df=tfidf_cfg["max_df"],
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
    )
    vec_out = run_vectorisation(df, vectorizer=vectorizer)
    nmf_out = train_nmf(
        vec_out.X,
        n_topics=n_topics,
        random_state=CONFIG["nmf"]["random_state"],
        init=CONFIG["nmf"]["init"],
        max_iter=CONFIG["nmf"]["max_iter"],
    )

    # S06: analysis-ready dataset
    topic_names = load_topic_names(country) or make_generic_topic_names(n_topics)
    df_alloc = run_topic_allocation(
        df, nmf_model=nmf_out.nmf_model, vectorizer=vec_out.vectorizer,
        topic_names=topic_names,
    )
    analysis_csv = PROJECT_ROOT / "data" / "evaluation_outputs" / f"topics_analysis_ready_{country}.csv"
    export_analysis_ready_csv(df_alloc, analysis_csv, topic_names=topic_names)

    # S07: evaluation (DataFrames)
    coh_df = evaluate_coherence_over_topic_range(
        X=vec_out.X,
        feature_names=vec_out.feature_names,
        texts_tokens=df["tokens_final"].tolist(),
        topic_range=range(*CONFIG["evaluation"]["coherence_topic_range"]),
        n_top_words=CONFIG["evaluation"]["n_top_words"],
        random_state=CONFIG["nmf"]["random_state"],
        init=CONFIG["nmf"]["init"],
        max_iter=CONFIG["nmf"]["max_iter"],
    )
    stab_df = evaluate_topic_stability(
        X=vec_out.X,
        n_topics=n_topics,
        seeds=CONFIG["evaluation"]["stability_seeds"],
    )

    # S08: persist artifacts + evaluation csvs
    save_run_outputs(
        run_dir=run_dir,
        vectorizer=vec_out.vectorizer,
        nmf_model=nmf_out.nmf_model,
        X=vec_out.X,
        dataset_name=dataset_name,
        reconstruction_error=nmf_out.reconstruction_error,
        W=nmf_out.W,
        coherence_df=coh_df,
        stability_df=stab_df,
    )

    # S09: MLflow (optional if installed)
    try:
        mlflow_run_id = log_run_to_mlflow(
            experiment_name="AM1_topic_modelling",
            run_name=run_name,
            dataset_name=dataset_name,
            X_shape=(int(vec_out.X.shape[0]), int(vec_out.X.shape[1])),
            vectorizer=vec_out.vectorizer,
            nmf_model=nmf_out.nmf_model,
            reconstruction_error=nmf_out.reconstruction_error,
            run_dir=run_dir,
            df_alloc_path=analysis_csv,
        )
    except ImportError as e:
        mlflow_run_id = None
        logger.warning("MLflow logging skipped: %s", e)

    # S11: write topic results to Supabase (articles_topics table)
    write_topic_results(df_alloc, run_id=run_name, model_type="nmf")

    # Update config.yaml with the new model_run for this country
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    config_data["countries"][country]["model_run"] = run_name
    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    logger.info("Updated config.yaml: countries.%s.model_run = %s", country, run_name)

    print(f"\nPipeline complete for {country}")
    print("Run name:", run_name)
    print("Analysis CSV:", analysis_csv.as_posix())
    print("Artifacts dir:", run_dir.as_posix())
    print("MLflow run_id:", mlflow_run_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("gensim").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Train NMF topic model")
    parser.add_argument("--country", choices=["eng", "sco", "irl"], help="Country to train")
    parser.add_argument("--all", action="store_true", help="Train all three countries")
    args = parser.parse_args()

    if args.all:
        for country in CONFIG["countries"]:
            train_country(country)
    else:
        country = args.country or CONFIG["training_country"]
        train_country(country)


if __name__ == "__main__":
    main()