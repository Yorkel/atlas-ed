"""
Regenerate all 4 summary JSONs from saved models + correctly-preprocessed data.

Fixes the discrepancy where notebooks generated summaries from 3,943 articles
(old parquet without MIN_CONTENT_LENGTH filter) while the pipeline uses 3,939
articles (with the filter).

For each model:
1. Load saved NMF model + vectorizer from its run directory
2. Load and preprocess articles through the pipeline's cleaning steps
3. Transform articles with the saved vectorizer/model (not fit_transform)
4. Compute topic assignments and source concentration stats
5. Write corrected summary JSON

Usage:
    python scripts/regenerate_summary_jsons.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_pipeline.training.s01_data_loader import load_articles
from model_pipeline.training.s02_cleaning import run_cleaning
from model_pipeline.training.s03_spacy_processing import run_spacy_processing
from model_pipeline.training.s08_save_outputs import generate_summary_json

logging.basicConfig(level=logging.INFO)
logging.getLogger("gensim").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "experiments" / "outputs" / "runs"
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation_outputs"

# Model configurations: run_id, output filename, topic_names source, stability value
# Stability values from existing evaluation (notebooks computed these correctly —
# they depend on the H matrix, not article count)
MODELS = {
    "nmf_eng_5": {
        "run_id": "2026-03-24_104906",
        "n_topics": 5,
        "stability": 1.0,
        "filter_source": None,  # full corpus
        "topic_names": {
            0: "pupil_welfare_and_inclusion",
            1: "academy_financial_oversight",
            2: "academy_trust_governance",
            3: "teacher_pay_and_workforce",
            4: "ofsted_inspection_reform",
        },
    },
    "nmf_eng_15": {
        "run_id": "2026-03-24_112503",
        "n_topics": 15,
        "stability": 1.0,
        "filter_source": None,
        "topic_names": {
            0: "child_welfare_and_early_years",
            1: "academy_financial_oversight",
            2: "academy_trust_governance",
            3: "teacher_recruitment_and_retention",
            4: "ofsted_inspection_reform",
            5: "gcse_attainment_and_grades",
            6: "pupil_attainment_and_disadvantage",
            7: "dfe_intervention_notices",
            8: "send_and_council_deficits",
            9: "teacher_pay_and_strikes",
            10: "skills_and_apprenticeships",
            11: "exam_regulation_and_ofqual",
            12: "raac_building_crisis",
            13: "school_absence_and_attendance",
            14: "school_funding_and_meals",
        },
    },
    "nmf_eng_30": {
        "run_id": "eng_2026-03-24_134826",
        "n_topics": 30,
        "stability": 0.9663,
        "filter_source": None,
        "topic_names": None,  # load from run dir
    },
    "nmf_eng_30_nm": {
        "run_id": "2026-03-24_013511",
        "n_topics": 30,
        "stability": 0.9609,
        "filter_source": "schoolsweek",  # remove SchoolsWeek
        "topic_names": None,  # load from topic_names_run
        "topic_names_run": "2026-03-24_174315",  # curated names are in a different run
    },
}


def load_and_preprocess(filter_source: str | None = None) -> pd.DataFrame:
    """Load articles and run pipeline preprocessing (s01 + s02 + s03)."""
    df = load_articles("eng_training")
    df = run_cleaning(df)
    df = run_spacy_processing(df)

    if filter_source:
        before = len(df)
        df = df[df["source"] != filter_source].reset_index(drop=True)
        logger.info("Filtered out '%s': %d -> %d articles", filter_source, before, len(df))

    return df


def main() -> None:
    # Preprocess once for full corpus, once for NM
    logger.info("Loading and preprocessing full corpus...")
    df_full = load_and_preprocess(filter_source=None)
    logger.info("Full corpus: %d articles", len(df_full))

    logger.info("Loading and preprocessing NM corpus (no SchoolsWeek)...")
    df_nm = load_and_preprocess(filter_source="schoolsweek")
    logger.info("NM corpus: %d articles", len(df_nm))

    # Back up existing summary JSONs
    backup_dir = EVAL_DIR / "backup_summaries"
    backup_dir.mkdir(exist_ok=True)

    for model_id, config in MODELS.items():
        logger.info("\n=== Regenerating %s ===", model_id)

        run_dir = RUNS_DIR / config["run_id"]

        # Load saved model + vectorizer
        nmf_model = joblib.load(run_dir / "nmf_model.joblib")
        vectorizer = joblib.load(run_dir / "vectorizer.joblib")

        # Load topic names (may come from a different run if curated separately)
        topic_names = config["topic_names"]
        if topic_names is None:
            names_run = config.get("topic_names_run", config["run_id"])
            names_path = RUNS_DIR / names_run / "topic_names.json"
            with open(names_path) as f:
                raw = json.load(f)
            topic_names = {int(k): v for k, v in raw.items()}
            logger.info("Loaded topic names from %s", names_path)

        # Select correct dataframe
        df = df_nm if config["filter_source"] else df_full

        # Transform (NOT fit_transform — use the saved model's learned patterns)
        texts = df["text_final"].fillna("").astype(str)
        X = vectorizer.transform(texts)
        W = nmf_model.transform(X)

        # Build df_alloc with topic assignments
        df_alloc = df.copy()
        df_alloc["topic_num"] = W.argmax(axis=1)
        df_alloc["topic_name"] = df_alloc["topic_num"].map(topic_names)
        df_alloc["dominant_topic_weight"] = W.max(axis=1)

        # Compute reconstruction error
        H = nmf_model.components_
        reconstruction_error = float(np.linalg.norm(X.toarray() - W @ H, "fro"))

        # Dominant weight stats
        dom_weights = W.max(axis=1)
        mean_dom = float(dom_weights.mean())
        max_dom = float(dom_weights.max())

        # Back up existing JSON
        existing = EVAL_DIR / f"{model_id}_summary.json"
        if existing.exists():
            backup = backup_dir / f"{model_id}_summary.json.bak"
            backup.write_bytes(existing.read_bytes())
            logger.info("Backed up: %s", backup)

        # Generate new summary JSON
        out_path = EVAL_DIR / f"{model_id}_summary.json"
        generate_summary_json(
            df_alloc=df_alloc,
            model_id=model_id,
            topic_names=topic_names,
            reconstruction_error=reconstruction_error,
            stability=config["stability"],
            mean_dominant_weight=mean_dom,
            max_dominant_weight=max_dom,
            out_path=out_path,
        )

        # Print comparison
        logger.info(
            "%s: %d articles, recon_err=%.4f, mean_wt=%.4f, max_wt=%.4f",
            model_id, len(df_alloc), reconstruction_error, mean_dom, max_dom,
        )

    # Print diff summary
    print("\n" + "=" * 60)
    print("REGENERATION COMPLETE")
    print("=" * 60)
    for model_id in MODELS:
        new = json.loads((EVAL_DIR / f"{model_id}_summary.json").read_text())
        bak_path = backup_dir / f"{model_id}_summary.json.bak"
        if bak_path.exists():
            old = json.loads(bak_path.read_text())
            print(f"\n{model_id}:")
            print(f"  n_articles: {old['n_articles']} -> {new['n_articles']}")
            print(f"  reconstruction_error: {old['metrics']['reconstruction_error']} -> {new['metrics']['reconstruction_error']}")
            print(f"  mean_dominant_weight: {old['metrics']['mean_dominant_weight']} -> {new['metrics']['mean_dominant_weight']}")
            print(f"  max_dominant_weight: {old['metrics']['max_dominant_weight']} -> {new['metrics']['max_dominant_weight']}")

            # Topic count changes
            old_counts = {t["name"]: t["count"] for t in old["topics"]}
            new_counts = {t["name"]: t["count"] for t in new["topics"]}
            big_changes = []
            for name in new_counts:
                if name in old_counts and abs(new_counts[name] - old_counts[name]) > 10:
                    big_changes.append(f"    {name}: {old_counts[name]} -> {new_counts[name]}")
            if big_changes:
                print(f"  Topic count changes (>10 diff):")
                for c in big_changes:
                    print(c)


if __name__ == "__main__":
    main()
