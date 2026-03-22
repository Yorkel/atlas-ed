"""
s03_spacy_processing.py

Step 03: spaCy processing + post-spaCy junk filtering (matches FINAL notebook workflow)

Input:
- df with column: 'text_clean' (from s02_cleaning)

Output:
- df with new columns:
    - tokens_after_spacy  (list[str])
    - tokens_final        (list[str])
    - text_final          (str)  joined tokens for vectorisation

NOTE (important):
- This version matches your notebook EXACTLY in stopword logic:
  it removes STOP_WORDS (spaCy default stopword set) inside spacy_clean().
- Your earlier pipeline version used CUSTOM_STOPWORDS (expanded list).
  That would shift vocabulary and topics. This version does NOT do that.
"""

import logging
from typing import Any, List, Set

import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Domain-specific stopwords (from your notebook)
# ─────────────────────────────────────────────────────────────────────────────
MEDIA: Set[str] = {"schoolsweek"}

TIME_STOPWORDS: Set[str] = {
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "today", "yesterday", "tomorrow", "daily", "weekly", "monthly", "yearly",
    "term", "terms", "spring", "summer", "autumn", "winter",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}

# ─────────────────────────────────────────────────────────────────────────────
# Post-spaCy junk terms (from your notebook)
# ─────────────────────────────────────────────────────────────────────────────
JUNK_TERMS: Set[str] = {
    # --- lemmatization & parser artefacts ---
    "cooky", "prev", "datum", "structparent", "annot",
    "font", "tabs", "rotate", "rr", "page",

    # --- statistical / reporting scaffolding ---
    "cent", "percentage", "proportion", "point", "figure",
    "survey", "rate", "number", "level", "score",

    # --- newsletter / filler language ---
    "interesting", "fact", "mover", "shaker",
    "previous", "current", "date",

    # --- admin / document language ---
    "document", "detail", "section", "annex", "appendix",
    "letter", "email", "sign", "signed", "signature",
    "submission", "item", "version", "draft",

    # --- procedural / process words ---
    "guidance", "framework", "response", "statement",
    "proposal", "approach", "review", "update",

    # --- vague policy boilerplate ---
    "summit", "voice", "stakeholder", "partnership",
    "engagement", "dialogue", "initiative",

    # --- web / gov.uk artefacts ---
    "cookie", "banner", "footer", "header",
    "subscribe", "subscription", "contact", "submit",
    "accessibility", "archive", "toggle", "skip", "select",

    # --- extra ---
    "office", "official", "issued", "signed",
    "recipient", "sender", "correspondence",
    "notification", "circulated",
    "programme", "scheme", "initiative", "pilot",
    "good", "high", "low", "new",
    "introduce", "implement", "launch", "rollout",
    "department",

    # --- source / organisation names ---
    # Removed to prevent topics clustering by publisher identity.
    # This is a specification choice: keeping them would mean some topics
    # are partly defined by who published, not what was said.
    # England sources
    "schoolsweek", "datalab", "fft", "epi", "nuffield", "fed",
    # Scotland sources
    "gtcs", "ades", "sera",
    # Ireland sources
    "esri", "erc", "rte",

    # --- country names ---
    # Appear in every document from that country, not discriminative.
    "england", "english", "scotland", "scottish", "ireland", "irish",
    "wales", "welsh", "britain", "british",
}

# ─────────────────────────────────────────────────────────────────────────────
# spaCy model loader (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────
_NLP = None

def get_nlp(model_name: str = "en_core_web_sm"):
    """Load spaCy model once per process."""
    global _NLP
    if _NLP is None:
        logger.info("Loading spaCy model: %s", model_name)
        _NLP = spacy.load(model_name)
    return _NLP


# ─────────────────────────────────────────────────────────────────────────────
# Core tokenisation/cleaning (matches notebook logic)
# ─────────────────────────────────────────────────────────────────────────────
def spacy_clean(doc: Any, nlp) -> List[str]:
    """
    Notebook-equivalent:
    - Create spaCy Doc
    - Remove PERSON entities
    - Remove MEDIA + TIME_STOPWORDS
    - Keep only POS in {NOUN, PROPN, ADJ}
    - Lemmatise
    - Remove STOP_WORDS (spaCy default stopword list)
    """
    if not isinstance(doc, str):
        return []

    parsed = nlp(doc)
    tokens: List[str] = []

    for token in parsed:
        # Remove person names
        if token.ent_type_ == "PERSON":
            continue

        # Remove domain-specific noise
        if token.lower_ in MEDIA or token.lower_ in TIME_STOPWORDS:
            continue

        # Keep only informative POS
        if token.pos_ in {"NOUN", "PROPN", "ADJ"}:
            lemma = token.lemma_.lower()
            if lemma and lemma not in STOP_WORDS:
                tokens.append(lemma)

    return tokens


def remove_junk(tokens: List[str]) -> List[str]:
    """Remove post-spaCy junk terms (list filter)."""
    return [t for t in tokens if t not in JUNK_TERMS]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline-facing function
# ─────────────────────────────────────────────────────────────────────────────
def run_spacy_processing(df: pd.DataFrame, text_col: str = "text_clean") -> pd.DataFrame:
    """
    Adds:
      - tokens_after_spacy
      - tokens_final
      - text_final
    """
    if text_col not in df.columns:
        raise KeyError(f"Expected column '{text_col}' not found. Available: {list(df.columns)}")

    logger.info("Step 03 (spacy): starting. Input shape=%s", df.shape)

    out = df.copy()
    nlp = get_nlp()

    out["tokens_after_spacy"] = out[text_col].apply(lambda x: spacy_clean(x, nlp))
    logger.info("spaCy processing complete (tokens_after_spacy created).")

    out["tokens_final"] = out["tokens_after_spacy"].apply(remove_junk)
    out["text_final"] = out["tokens_final"].apply(lambda toks: " ".join(toks))
    logger.info("Post-spaCy junk filtering complete (tokens_final + text_final created).")

    empty_final = (out["text_final"].str.len() == 0).sum()
    logger.info("Empty text_final rows: %d", empty_final)
    logger.info("Step 03 (spacy): complete. Output shape=%s", out.shape)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test (package mode ONLY)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging

    from model_pipeline.training.s01_data_loader import load_articles
    from model_pipeline.training.s02_cleaning import run_cleaning

    logging.basicConfig(level=logging.INFO)

    df = load_articles("full_retro").head(200)
    df = run_cleaning(df)
    df2 = run_spacy_processing(df)

    print("Rows in:", len(df), "Rows out:", len(df2))
    print("Columns added:",
          [c for c in ["tokens_after_spacy", "tokens_final", "text_final"] if c in df2.columns])

    i = df2.index[0]
    print("\n--- text_clean (first 300 chars) ---")
    print(df2.loc[i, "text_clean"][:300])
    print("\n--- text_final (first 300 chars) ---")
    print(df2.loc[i, "text_final"][:300])

    print("\nEmpty text_final:", (df2["text_final"].str.len() == 0).sum())
    print("Avg tokens_after_spacy:", df2["tokens_after_spacy"].apply(len).mean())
    print("Avg tokens_final:", df2["tokens_final"].apply(len).mean())

