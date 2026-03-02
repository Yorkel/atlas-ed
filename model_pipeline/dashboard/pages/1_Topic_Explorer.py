import sys
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(
    page_title="Topic Explorer | Education Policy Observatory",
    page_icon="🔍",
    layout="wide",
)
alt.themes.enable("dark")

st.markdown("""
<style>
[data-testid="block-container"] { padding: 1.5rem 2rem; }
[data-testid="stMetric"] {
    background-color: #2b2b2b; text-align: center;
    padding: 18px 0; border-radius: 6px;
}
[data-testid="stMetricLabel"] { justify-content: center; }
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from model_pipeline.dashboard.supabase_loader import load_articles, load_articles_with_probabilities


@st.cache_resource
def get_top_words():
    try:
        from model_pipeline.api.model_loader import get_model
        bundle = get_model()
        feat = bundle.vectorizer.get_feature_names_out()
        result = {}
        for i, vec in enumerate(bundle.nmf_model.components_):
            top_idx = vec.argsort()[-15:][::-1]
            result[i] = [{"word": feat[j], "weight": float(vec[j])} for j in top_idx]
        return result, bundle.topic_names
    except Exception:
        return None, None


df = load_articles()
top_words_by_id, topic_names_by_id = get_top_words()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Topic Explorer")
st.caption("Browse topics, read articles, and examine contestability scores.")

topics_sorted = sorted(df["topic_name"].unique())
selected_topic = st.selectbox("Select a topic", topics_sorted)
topic_df = df[df["topic_name"] == selected_topic]

m1, m2, m3 = st.columns(3)
m1.metric("Articles", len(topic_df))
m2.metric("Avg confidence", f"{topic_df['dominant_topic_weight'].mean():.3f}")
m3.metric("Sources", topic_df["source"].nunique())

st.divider()

tab_overview, tab_contest = st.tabs(["Topic Overview", "Contestability Analysis"])

# ── Tab 1: Topic Overview ─────────────────────────────────────────────────────
with tab_overview:
    col_kw, col_art = st.columns([1, 2])

    with col_kw:
        st.subheader("Top keywords")
        topic_num = int(topic_df["topic_num"].iloc[0])
        if top_words_by_id and topic_num in top_words_by_id:
            words_df = pd.DataFrame(top_words_by_id[topic_num])
            st.altair_chart(
                alt.Chart(words_df).mark_bar(color="#f28e2b").encode(
                    x=alt.X("weight:Q", title="NMF weight"),
                    y=alt.Y("word:N", sort="-x", title=""),
                    tooltip=["word", alt.Tooltip("weight:Q", format=".4f")],
                ).properties(height=420),
                use_container_width=True,
            )
        else:
            st.info("Model unavailable — run the pipeline to enable top keywords.")

    with col_art:
        st.subheader("Articles")
        f1, f2 = st.columns(2)
        src_opts = sorted(topic_df["source"].unique())
        type_opts = sorted(topic_df["type"].unique())
        sel_src  = f1.multiselect("Source", src_opts, default=src_opts, key="te_src")
        sel_type = f2.multiselect("Type",   type_opts, default=type_opts, key="te_type")

        filtered = (
            topic_df[topic_df["source"].isin(sel_src) & topic_df["type"].isin(sel_type)]
            .sort_values("date", ascending=False)
        )
        st.caption(f"Showing {min(len(filtered), 50)} of {len(filtered)} articles")

        for _, row in filtered.head(50).iterrows():
            label = (
                f"{row['source']} · {row['date'].strftime('%b %Y')} "
                f"· confidence: {row['dominant_topic_weight']:.3f}"
            )
            with st.expander(label):
                st.caption(f"Type: {row['type']}")
                text = str(row["text_clean"])
                st.write(text[:2000] + ("…" if len(text) > 2000 else ""))

# ── Tab 2: Contestability Analysis ───────────────────────────────────────────
with tab_contest:
    st.subheader("Topic Assignment Contestability")
    st.caption(
        "**Contestability score** = normalised Shannon entropy of the topic weight vector.  \n"
        "Score near **1** → weight spread evenly across topics → assignment is highly contested.  \n"
        "Score near **0** → weight concentrated on one topic → assignment is confident."
    )

    topic_contest = topic_df.dropna(subset=["contestability_score"]).copy()

    if len(topic_contest) > 0:
        # Distribution chart
        hist_df = topic_contest[["contestability_score"]].copy()
        st.altair_chart(
            alt.Chart(hist_df).mark_bar(color="#e15759").encode(
                x=alt.X("contestability_score:Q", bin=alt.Bin(maxbins=30),
                         title="Contestability score"),
                y=alt.Y("count()", title="Articles"),
            ).properties(height=220, title="Distribution of contestability scores"),
            use_container_width=True,
        )

        avg_score = topic_contest["contestability_score"].mean()
        st.metric("Average contestability score", f"{avg_score:.3f}")

        st.divider()
        threshold = st.slider(
            "Show articles with contestability score >=", 0.5, 1.0, 0.8, 0.05
        )
        high = topic_contest[topic_contest["contestability_score"] >= threshold].sort_values(
            "contestability_score", ascending=False
        )
        st.caption(f"{len(high)} articles above threshold")

        # Load topic probabilities for per-article weight breakdowns
        try:
            full_df, topic_cols = load_articles_with_probabilities()
            has_probs = len(topic_cols) > 0
        except Exception:
            has_probs = False

        for _, row in high.head(15).iterrows():
            label = (
                f"Score: {row['contestability_score']:.3f} · "
                f"{row['source']} · {row['topic_name']}"
            )
            with st.expander(label):
                if has_probs:
                    match = full_df[full_df["article_id"] == row["article_id"]]
                    if len(match) > 0:
                        st.caption(
                            "This article's assignment is ambiguous. "
                            "Top 10 topic weights:"
                        )
                        w_series = (
                            match.iloc[0][topic_cols]
                            .astype(float)
                            .sort_values(ascending=False)
                            .head(10)
                        )
                        w_df = pd.DataFrame({
                            "topic": [c.replace("_", " ").title() for c in w_series.index],
                            "weight": w_series.values,
                        })
                        st.altair_chart(
                            alt.Chart(w_df).mark_bar(color="#59a14f").encode(
                                x=alt.X("weight:Q", title="Weight"),
                                y=alt.Y("topic:N", sort="-x", title=""),
                            ).properties(height=200),
                            use_container_width=True,
                        )
                    else:
                        st.caption("Topic weight breakdown unavailable for this article.")
                else:
                    st.caption("Topic probability data unavailable.")
    else:
        st.info("No contestability scores available for this topic.")
