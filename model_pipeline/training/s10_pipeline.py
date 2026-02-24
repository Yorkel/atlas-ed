"""
s10_pipeline.py

Step 10: End-to-end training pipeline runner (S01 → S09)

Run:
python -m model_pipeline.training.s10_pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml") as _f:
    CONFIG = yaml.safe_load(_f)


def main() -> None:
    import logging

    from model_pipeline.training.s01_data_loader import load_articles
    from model_pipeline.training.s02_cleaning import run_cleaning
    from model_pipeline.training.s03_spacy_processing import run_spacy_processing
    from model_pipeline.training.s04_vectorisation import run_vectorisation
    from model_pipeline.training.s05_nmf_training import train_nmf
    from model_pipeline.training.s06_topic_allocation import run_topic_allocation, export_analysis_ready_csv
    from model_pipeline.training.s07_evaluation import (
        evaluate_coherence_over_topic_range,
        evaluate_topic_stability,
    )
    from model_pipeline.training.s08_save_outputs import save_run_outputs, RUNS_DIR, make_run_id
    from model_pipeline.training.s09_mlflow_logging import log_run_to_mlflow

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("gensim").setLevel(logging.WARNING)

    dataset_name = CONFIG["data"]["dataset_name"]
    run_name = make_run_id()
    run_dir = RUNS_DIR / run_name
    logger.info("Pipeline run_name=%s", run_name)

    # S01–S05
    df = load_articles(dataset_name)
    df = run_cleaning(df)
    df = run_spacy_processing(df)
    vec_out = run_vectorisation(df)
    nmf_out = train_nmf(
        vec_out.X,
        n_topics=CONFIG["nmf"]["n_topics"],
        random_state=CONFIG["nmf"]["random_state"],
        init=CONFIG["nmf"]["init"],
        max_iter=CONFIG["nmf"]["max_iter"],
    )

    # S06: analysis-ready dataset
    df_alloc = run_topic_allocation(df, nmf_model=nmf_out.nmf_model, vectorizer=vec_out.vectorizer)
    analysis_csv = PROJECT_ROOT / "data" / dataset_name / "retro_topics_analysis_ready.csv"
    export_analysis_ready_csv(df_alloc, analysis_csv)

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
        n_topics=CONFIG["nmf"]["n_topics"],
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

    print("\n✅ Pipeline complete")
    print("Run name:", run_name)
    print("Analysis CSV:", analysis_csv.as_posix())
    print("Artifacts dir:", run_dir.as_posix())
    print("MLflow run_id:", mlflow_run_id)
    print("MLflow store:", (PROJECT_ROOT / "experiments" / "mlruns").as_posix())


if __name__ == "__main__":
    main()