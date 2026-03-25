"""
Retrain NMF at k=5 and k=15 to save joblib files for keyword extraction.

Reuses the SAME vectorizer and TF-IDF matrix from the production k=30 run,
so the vocabulary and document representations are identical — only k changes.

This means keywords are directly comparable across k=5, k=15, and k=30.

Usage:
    python scripts/retrain_k5_k15.py
"""

import json
import logging
from pathlib import Path

import joblib
from sklearn.decomposition import NMF

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "experiments" / "outputs" / "runs"

# The production k=30 run — we reuse its vectorizer
K30_RUN = "eng_2026-03-24_134826"

# Target run directories (already exist with topic_names.json etc)
TARGETS = {
    5: "2026-03-24_104906",
    15: "2026-03-24_112503",
}

# NMF params (same as config.yaml)
NMF_PARAMS = dict(init="nndsvd", random_state=42, max_iter=1000)


def main():
    # Load the k=30 vectorizer to get the TF-IDF matrix
    k30_dir = RUNS_DIR / K30_RUN
    logger.info("Loading vectorizer from %s ...", K30_RUN)
    vectorizer = joblib.load(k30_dir / "vectorizer.joblib")

    # We need the training texts to rebuild the TF-IDF matrix
    # Load from the analysis-ready CSV (text_clean column)
    # But the vectorizer was fit on text_final (spaCy lemmatised).
    # Since we have the FITTED vectorizer, we just need to transform the same texts.
    # The cleanest approach: load the training data and re-run spaCy,
    # OR just re-fit the vectorizer on text_clean (which gives slightly different vocab).
    #
    # Best approach: use the vectorizer to transform the training data.
    # But we need the original text_final. Let's reconstruct via the pipeline steps.

    logger.info("Loading and preprocessing training data...")
    import pandas as pd

    # Load training data
    training_csv = REPO_ROOT / "data" / "training" / "eng_training.csv"
    if not training_csv.exists():
        training_parquet = REPO_ROOT / "data" / "training" / "eng_training.parquet"
        if training_parquet.exists():
            df = pd.read_parquet(training_parquet)
        else:
            raise FileNotFoundError(f"No training data found at {training_csv} or {training_parquet}")
    else:
        df = pd.read_csv(training_csv)

    logger.info("Loaded %d articles", len(df))

    # Run cleaning + spaCy to get text_final
    from model_pipeline.training.s02_cleaning import run_cleaning
    from model_pipeline.training.s03_spacy_processing import run_spacy_processing

    logger.info("Running text cleaning...")
    df = run_cleaning(df)

    logger.info("Running spaCy processing (this takes a minute)...")
    df = run_spacy_processing(df)

    # Transform using the SAME fitted vectorizer
    logger.info("Transforming with production vectorizer...")
    X = vectorizer.transform(df["text_final"].fillna("").astype(str))
    logger.info("TF-IDF matrix shape: %s", X.shape)

    # Train and save for each k
    for k, run_id in TARGETS.items():
        run_dir = RUNS_DIR / run_id
        logger.info("\n--- Training NMF k=%d ---", k)

        nmf = NMF(n_components=k, **NMF_PARAMS)
        W = nmf.fit_transform(X)
        logger.info("Reconstruction error: %.4f", nmf.reconstruction_err_)

        # Save model and vectorizer
        joblib.dump(nmf, run_dir / "nmf_model.joblib")
        joblib.dump(vectorizer, run_dir / "vectorizer.joblib")
        logger.info("Saved nmf_model.joblib + vectorizer.joblib to %s", run_id)

        # Update run_metadata with flag
        meta_path = run_dir / "run_metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["reconstruction_error"] = float(nmf.reconstruction_err_)
        meta["joblib_added"] = True
        meta_path.write_text(json.dumps(meta, indent=2))

        # Verify keyword extraction works
        feature_names = vectorizer.get_feature_names_out()
        for t_idx in range(min(3, k)):
            top_words = [feature_names[i] for i in nmf.components_[t_idx].argsort()[:-11:-1]]
            topic_names_path = run_dir / "topic_names.json"
            tnames = json.loads(topic_names_path.read_text())
            name = tnames.get(str(t_idx), f"topic_{t_idx}")
            logger.info("  Topic %d (%s): %s", t_idx, name, ", ".join(top_words))

    logger.info("\nDone! Now re-run: python scripts/export_for_dashboard.py")


if __name__ == "__main__":
    main()
