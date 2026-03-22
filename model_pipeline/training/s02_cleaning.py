"""
s02_cleaning.py

Step 02: Cleaning (matches notebook workflow)
1) Structural + regex cleaning of scraped articles
2) Basic preprocessing chain (lowercase, URL removal, bracket removal, punctuation/numbers, whitespace)

Input:
- df with column: 'text' (from s01_data_loader)

Output:
- df with new column: 'text_clean'
"""

import logging
import re
import unicodedata
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1) Structural + regex cleaning (from your notebook)
# ─────────────────────────────────────────────────────────────────────────────
def clean_scraped_article(text: Any) -> str:
    """
    Regex-based removal of web boilerplate, PDF artefacts, nav text, etc.
    Order matters: block removal before aggressive normalization.
    """
    if not isinstance(text, str):
        return ""

    # ======== STAGE 1: STRUCTURAL FIXES (before removal) ========
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)  # zero-width chars (GTCS markdown artefacts)
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)  # fix hyphenation
    text = re.sub(r"\n+", " ", text)  # collapse newlines

    # ======== STAGE 2: REMOVE BLOCKS (largest units first) ========
    block_patterns = [
        r"accept all cookies.*?cookies policy",
        r"subscribe now.*?(?:access|continue reading)",
        r"you must be logged in.*?continue reading",
        r"sign in to continue.*?(?:subscribe|register)",
        r"start your free trial.*?(?:subscribe|access)",
        r"subscribe for full access.*?(?:here|now)",
        r"crown copyright.*?(?:all rights reserved|\d{4})",
        r"this site uses cookies.*?(?:more information|accept|privacy)",
        r"cookie preference.*?(?:settings|accept)",
        r"privacy policy.*?(?:here|terms)",
        r"accessibility statement.*?(?:here|contact)",

        # GOV.UK footer blocks — everything from these markers tends to be boilerplate
        r"updates to this page.*?(?:first published|sign up for emails)",
        r"sign up for emails or print this page.*?$",
        r"related content.*?$",
        r"explore the topic.*?$",
        r"dfe media enquiries.*?$",

        # FFT newsletter signup
        r"want to stay up-to-date with the latest research from fft education datalab\?.*?newsletter\.?",

        # Scotland gov_scot — "Media enquiries" trailing block
        r"media enquiries\s*$",
    ]
    for pat in block_patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE | re.DOTALL)

    # ======== STAGE 3: REMOVE SPECIFIC PHRASES ========
    phrase_patterns = [
        # Navigation
        r"skip to (?:main )?content",
        r"related articles?",
        r"more on this story",
        r"share this article",

        # Social media
        r"share on (?:twitter|facebook|linkedin|instagram)",
        r"follow us on",
        r"tweet this",

        # Metadata
        r"updated \d{1,2} \w+ \d{4}",
        r"published \d{1,2} \w+ \d{4}",
        r"by [A-Z][a-z]+ [A-Z][a-z]+",  # author bylines

        # Government/publication artefacts
        r"available under.*?licen[cs]e",
        r"thank you for your feedback",
        r"find out more",
        r"assistive technology",
        r"accessible format",

        # GOV.UK specific phrases
        r"applies to england",
        r"documents\s+(?:pdf|html)(?:\s*,\s*\d+\s*(?:kb|mb)\s*,\s*\d+\s*pages?)?",
        r"central newsdesk\s*-\s*for journalists\s*\d[\d\s]+",

        # SERA event registration
        r"sign up for the event here",
        r"scan the qr code",
    ]
    for pat in phrase_patterns:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)

    # ======== STAGE 4: REMOVE SINGLE WORDS ========
    noise_words = {
        # Web structure
        "advertisement", "menu", "home", "navigation", "sidebar",

        # PDF artefacts
        "pdf", "obj", "endobj", "xref", "stream", "endstream",
        "mediabox", "cropbox", "trimbox", "bleedbox", "trimbox", "cropbox",

        # Social/media
        "instagram", "flickr", "youtube", "twitter", "facebook",
        "linkedin", "newsletter", "tweet",

        # Cookie/privacy
        "cookie", "cooky", "privacy",

        # Gov.uk boilerplate
        "copyright", "licence", "license", "crown", "asset",

        # Publication metadata
        "summary", "publication", "email", "download", "document",
        "page", "section", "index", "content", "format", "type",

        # Units
        "kb", "mb", "isbn",

        # Actions
        "share", "print", "visit", "update", "view", "file",

        # Misc
        "topic", "blog", "news", "consultation", "feedback",
        "experience", "site", "thank", "ernment",  # cropped 'government'
    }
    pattern = r"\b(?:" + "|".join(noise_words) + r")\b"
    text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # ======== STAGE 5: HTML/ENTITIES ========
    text = re.sub(r"<.*?>", " ", text)       # HTML tags
    text = re.sub(r"&[a-z]+;", " ", text)    # HTML entities

    # ======== STAGE 6: CLEANUP (do this LAST) ========
    text = re.sub(r"[•●▪◆■]", " ", text)              # bullets
    text = re.sub(r"\b(\w+)\s+\1\b", r"\1", text)      # deduplicate adjacent words
    text = re.sub(r"\s+", " ", text).strip()           # final whitespace collapse

    return text


# ─────────────────────────────────────────────────────────────────────────────
# 2) Basic preprocessing chain (from notebook)
# ─────────────────────────────────────────────────────────────────────────────
def basic_preprocess_series(series: pd.Series) -> pd.Series:
    """
    Mirrors notebook:
    - lower
    - remove URLs
    - remove bracketed text
    - remove punctuation
    - remove numbers
    - collapse whitespace
    """
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"http\S+|www\.\S+", "", regex=True)
        .str.replace(r"[\(\[\{].*?[\)\]\}]", "", regex=True)
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\d+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline-facing function
# ─────────────────────────────────────────────────────────────────────────────
def run_cleaning(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Add df['text_clean'] based on df[text_col].
    Does NOT drop rows (PDF removal already happens in s01).
    """
    if text_col not in df.columns:
        raise KeyError(f"Expected column '{text_col}' not found. Available: {list(df.columns)}")

    logger.info("Step 02 (cleaning): starting. Input shape=%s", df.shape)

    out = df.copy()
    out["text_clean"] = out[text_col].apply(clean_scraped_article)
    out["text_clean"] = basic_preprocess_series(out["text_clean"])

    # Drop articles with insufficient content (e.g. ADES title-only articles)
    MIN_CONTENT_LENGTH = 200
    short_mask = out["text_clean"].str.len() < MIN_CONTENT_LENGTH
    short_count = short_mask.sum()
    if short_count > 0:
        logger.info(
            "Dropping %d articles with fewer than %d chars after cleaning",
            short_count, MIN_CONTENT_LENGTH,
        )
        out = out[~short_mask].reset_index(drop=True)

    empty_count = (out["text_clean"].str.len() == 0).sum()
    logger.info("Step 02 (cleaning): complete. Output shape=%s", out.shape)
    logger.info("Empty cleaned texts: %d", empty_count)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test (standalone run)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging

    from model_pipeline.training.s01_data_loader import load_articles

    logging.basicConfig(level=logging.INFO)

    # Small sample for speed while refactoring
    df = load_articles("full_retro").head(200)
    df2 = run_cleaning(df)

    # Basic checks
    print("Rows in:", len(df), "Rows out:", len(df2))
    print("Columns:", df2.columns.tolist())
    print("\n--- BEFORE (first 300 chars) ---")
    print(df2.loc[df2.index[0], "text"][:300])
    print("\n--- AFTER (first 300 chars) ---")
    print(df2.loc[df2.index[0], "text_clean"][:300])

    print("\nEmpty cleaned texts:", (df2["text_clean"].str.len() == 0).sum())
