import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Education Policy Observatory",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.themes.enable("dark")

st.markdown("""
<style>
[data-testid="block-container"] { padding: 1.5rem 2rem; }
[data-testid="stMetric"] {
    background-color: #2b2b2b;
    text-align: center;
    padding: 18px 0;
    border-radius: 6px;
}
[data-testid="stMetricLabel"] { justify-content: center; }
h3 { margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

from model_pipeline.dashboard.supabase_loader import load_articles

df = load_articles()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Education Policy Observatory")
st.caption("Tracking who shapes the education policy conversation, and how.")

st.info(
    "**Data note:** ~69% of articles are from SchoolsWeek (education journalism). "
    "All charts offer a **normalised view** to make sources with very different "
    "volumes directly comparable.",
    icon="ℹ️",
)

st.divider()

# ── KPI cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total articles", f"{len(df):,}")
k2.metric("Topics", df["topic_name"].nunique())
k3.metric("Organisations", df["source"].nunique())
k4.metric(
    "Date range",
    f"{df['date'].dt.year.min()}–{df['date'].dt.year.max()}",
)

st.divider()

# ── Top topics + source breakdown ─────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("Top 10 topics")
    normalise = st.toggle("Normalise (% of corpus)", key="norm_topics")

    if normalise:
        counts = (
            df["topic_name"].value_counts(normalize=True)
            .mul(100).round(1).head(10).reset_index()
        )
        counts.columns = ["topic_name", "value"]
        x_title = "% of corpus"
    else:
        counts = df["topic_name"].value_counts().head(10).reset_index()
        counts.columns = ["topic_name", "value"]
        x_title = "Articles"

    st.altair_chart(
        alt.Chart(counts).mark_bar(color="#4e79a7").encode(
            x=alt.X("value:Q", title=x_title),
            y=alt.Y("topic_name:N", sort="-x", title=""),
            tooltip=["topic_name", alt.Tooltip("value:Q", title=x_title)],
        ).properties(height=320),
        use_container_width=True,
    )

with c2:
    st.subheader("Articles by organisation")
    src = df.groupby(["source", "type"]).size().reset_index(name="count")
    st.altair_chart(
        alt.Chart(src).mark_bar().encode(
            x=alt.X("count:Q", title="Articles"),
            y=alt.Y("source:N", sort="-x", title=""),
            color=alt.Color("type:N", legend=alt.Legend(title="Type")),
            tooltip=["source", "type", "count"],
        ).properties(height=320),
        use_container_width=True,
    )

st.divider()

# ── Navigation guide ──────────────────────────────────────────────────────────
st.subheader("Explore the Observatory")
n1, n2, n3, n4 = st.columns(4)
n1.info(
    "**Topic Explorer**\n\n"
    "Browse all 30 topics, view top keywords, read articles, "
    "and examine topic contestability scores."
)
n2.info(
    "**Trends Over Time**\n\n"
    "Monthly topic attention with election marker and "
    "pre/post election rank shift table."
)
n3.info(
    "**Organisation Analysis**\n\n"
    "Source × topic heatmap and organisation-type breakdowns. "
    "Filter by type and country."
)
n4.info(
    "**Framing Analysis**\n\n"
    "How different organisations frame the same topics "
    "using predefined framing keyword sets."
)
