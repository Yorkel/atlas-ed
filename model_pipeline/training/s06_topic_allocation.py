"""
s06_topic_allocation.py

Step 06: Topic allocation + export (matches notebook "analysis-ready dataset")

Input:
- df with:
    - 'text_clean' (for export readability; from s02)
    - 'text_final' (for vectorisation; from s03)
    - metadata: url, article_date, source, type (as available)
- fitted vectorizer (from s04)
- trained NMF model (from s05)

Output:
- df with:
    - topic_num
    - topic_name
    - dominant_topic_weight
    - N continuous topic weight columns (named via topic_names)
    - year, month
- writes CSV: data/evaluation_outputs/topics_analysis_ready_<country>.csv
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_topic_names(country: str) -> Dict[int, str]:
    """Load topic names from LLM review JSON or run directory."""
    # Try LLM review file first
    llm_path = PROJECT_ROOT / "data" / "evaluation_outputs" / f"llm_{country}_topic_review.json"
    if country == "eng":
        llm_path = PROJECT_ROOT / "data" / "evaluation_outputs" / "llm_topic_review.json"

    if llm_path.exists():
        with open(llm_path) as f:
            reviews = json.load(f)
        names = {item["topic"]: item["suggested_name"] for item in reviews}
        logger.info("Loaded %d topic names from %s", len(names), llm_path.name)
        return names

    logger.warning("No topic names found for %s — using generic names", country)
    return {}


def make_generic_topic_names(n_topics: int) -> Dict[int, str]:
    """Generate generic topic_0, topic_1, ... names."""
    return {i: f"topic_{i}" for i in range(n_topics)}


# Default England names for backward compatibility
TOPIC_NAMES: Dict[int, str] = load_topic_names("eng") or {
    0: "child_and_family_support",
    1: "academy_finance_and_oversight",
    2: "mat_governance",
    3: "teacher_pay",
    4: "ofsted_inspections",
    5: "exam_results",
    6: "pupil_absence",
    7: "dfe_intervention",
    8: "local_authority_deficits",
    9: "teacher_strikes",
    10: "apprenticeships",
    11: "exam_regulation",
    12: "raac_crisis",
    13: "disadvantaged_groups",
    14: "free_school_meals",
    15: "education_politics",
    16: "education_research",
    17: "leadership_appointments",
    18: "ai_and_edtech",
    19: "mental_health",
    20: "curriculum",
    21: "safeguarding",
    22: "teacher_recruitment",
    23: "school_funding",
    24: "exclusions_suspensions",
    25: "report_cards",
    26: "send_inclusion",
    27: "primary_assessment",
    28: "school_places",
    29: "breakfast_clubs",
}


def run_topic_allocation(
    df: pd.DataFrame,
    nmf_model,
    vectorizer,
    text_final_col: str = "text_final",
    text_clean_col: str = "text_clean",
    date_col: str = "article_date",
    topic_names: Dict[int, str] = TOPIC_NAMES,
) -> pd.DataFrame:
    """
    Compute topic weights:
      X = vectorizer.transform(text_final)
      W = nmf_model.transform(X)

    Then:
      - topic_num = argmax(W)
      - dominant_topic_weight = max(W)
      - add 30 topic weight columns named via topic_names
      - add year/month from date (if date present)
    """
    if text_final_col not in df.columns:
        raise KeyError(f"Expected '{text_final_col}' not found. Available: {list(df.columns)}")

    if text_clean_col not in df.columns:
        logger.warning(
            "Column '%s' not found (export will miss readable cleaned text).",
            text_clean_col,
        )

    logger.info("Step 06 (topic allocation): starting. Input shape=%s", df.shape)

    out = df.copy()

    # Year / month (matches notebook)
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out["year"] = out[date_col].dt.year
        out["month"] = out[date_col].dt.month
    else:
        logger.warning("No '%s' column found; skipping year/month creation.", date_col)

    texts = out[text_final_col].fillna("").astype(str)
    X = vectorizer.transform(texts)     # transform (NOT fit_transform)
    W = nmf_model.transform(X)          # transform (NOT fit_transform)

    # Dominant topic assignment (matches notebook export)
    out["topic_num"] = W.argmax(axis=1)
    out["topic_name"] = out["topic_num"].map(topic_names)
    out["dominant_topic_weight"] = W.max(axis=1)

    # Continuous topic weights columns
    n_topics = W.shape[1]
    missing = set(range(n_topics)) - set(topic_names.keys())
    if missing:
        raise ValueError(f"topic_names is missing keys for topic ids: {sorted(missing)}")

    for i in range(n_topics):
        col = topic_names[i]
        out[col] = W[:, i]  # numpy 1D already; pandas handles it fine

    # Notebook safety-style warning
    if out["topic_num"].nunique() != n_topics:
        logger.warning(
            "Not all topics appear as dominant topics. unique=%d expected=%d",
            out["topic_num"].nunique(),
            n_topics,
        )

    logger.info("Step 06 (topic allocation): complete. Output shape=%s", out.shape)
    return out


def export_analysis_ready_csv(
    df: pd.DataFrame,
    out_path: Path,
    topic_names: Dict[int, str] = TOPIC_NAMES,
) -> Path:
    """
    Export in notebook-style column order.
    Only exports columns that exist in df.
    """
    topic_cols = [topic_names[i] for i in range(len(topic_names))]

    export_cols = [
        "url",
        "article_date",
        "year",
        "month",
        "source",
        "type",
        "text_clean",
        "topic_num",
        "topic_name",
        "dominant_topic_weight",
        *topic_cols,
    ]

    existing = [c for c in export_cols if c in df.columns]
    missing_cols = [c for c in export_cols if c not in df.columns]
    if missing_cols:
        logger.info("Export: skipping missing columns: %s", missing_cols)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[existing].to_csv(out_path, index=False)
    logger.info("Saved analysis-ready CSV to: %s", out_path)

    return out_path


def main() -> None:
    """
    Smoke test / end-to-end export:
    python -m model_pipeline.training.s06_topic_allocation
    """
    import logging

    from model_pipeline.training.s01_data_loader import load_articles
    from model_pipeline.training.s02_cleaning import run_cleaning
    from model_pipeline.training.s03_spacy_processing import run_spacy_processing
    from model_pipeline.training.s04_vectorisation import run_vectorisation
    from model_pipeline.training.s05_nmf_training import run_nmf_training

    logging.basicConfig(level=logging.INFO)

    # 1) Load + preprocess
    df = load_articles("full_retro")
    df = run_cleaning(df)
    df = run_spacy_processing(df)

    # 2) Vectorise (fit vectorizer on this corpus)
    vec_out = run_vectorisation(df)

    # 3) Train final NMF (final notebook settings)
    nmf_out = run_nmf_training(vec_out.X)

    # 4) Allocate topics + export
    df_alloc = run_topic_allocation(
        df,
        nmf_model=nmf_out.nmf_model,
        vectorizer=vec_out.vectorizer,
    )

    out_csv = Path("data/full_retro/retro_topics_analysis_ready.csv")
    export_analysis_ready_csv(df_alloc, out_csv)

    print("\n✅ Wrote:", out_csv)
    print("Rows:", len(df_alloc))
    print("Example topic cols:", [TOPIC_NAMES[i] for i in range(5)], "...")


if __name__ == "__main__":
    main()
