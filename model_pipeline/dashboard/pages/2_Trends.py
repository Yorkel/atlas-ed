import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Trends | AtlasED",
    page_icon="📈",
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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from model_pipeline.dashboard.supabase_loader import load_articles

ELECTION_DATE = pd.Timestamp("2024-07-04")

df = load_articles()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    countries = st.multiselect(
        "Country", options=["England"], default=["England"], key="tr_country"
    )
    sources = st.multiselect(
        "Source", options=sorted(df["source"].unique()),
        default=sorted(df["source"].unique()), key="tr_src"
    )
    types = st.multiselect(
        "Organisation type", options=sorted(df["type"].unique()),
        default=sorted(df["type"].unique()), key="tr_type"
    )

filt = df[df["source"].isin(sources) & df["type"].isin(types)]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Trends Over Time")
st.caption("Monthly topic attention and the impact of the 2024 general election.")

st.divider()

# ── Topic trend line chart ────────────────────────────────────────────────────
st.subheader("Topic attention over time")

all_topics = sorted(filt["topic_name"].unique())
top_default = (
    filt["topic_name"].value_counts().head(6).index.tolist()
)
selected_topics = st.multiselect(
    "Topics to display", options=all_topics, default=top_default
)

normalise_trend = st.toggle("Normalise (% of monthly articles)", key="norm_trend")

if selected_topics:
    trend = filt[filt["topic_name"].isin(selected_topics)]

    if normalise_trend:
        monthly_total = filt.groupby("month").size().reset_index(name="total")
        trend_agg = (
            trend.groupby(["month", "topic_name"]).size()
            .reset_index(name="count")
            .merge(monthly_total, on="month")
        )
        trend_agg["value"] = (trend_agg["count"] / trend_agg["total"] * 100).round(2)
        y_title = "% of monthly articles"
    else:
        trend_agg = (
            trend.groupby(["month", "topic_name"]).size()
            .reset_index(name="count")
        )
        trend_agg["value"] = trend_agg["count"]
        y_title = "Articles"

    line = (
        alt.Chart(trend_agg)
        .mark_line(interpolate="monotone", point=True)
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("value:Q", title=y_title),
            color=alt.Color("topic_name:N", legend=alt.Legend(title="Topic")),
            tooltip=["month:T", "topic_name", alt.Tooltip("value:Q", title=y_title)],
        )
        .properties(height=380)
    )

    election_rule = (
        alt.Chart(pd.DataFrame({"date": [ELECTION_DATE]}))
        .mark_rule(strokeDash=[6, 3], strokeWidth=2, color="white")
        .encode(x="date:T")
    )

    election_label = (
        alt.Chart(pd.DataFrame({"date": [ELECTION_DATE], "label": ["← Election"]}))
        .mark_text(align="left", dx=6, dy=-10, color="white", fontSize=11)
        .encode(x="date:T", text="label:N")
    )

    st.altair_chart(line + election_rule + election_label, use_container_width=True)
else:
    st.info("Select at least one topic above.")

st.divider()

# ── Election shift table ──────────────────────────────────────────────────────
st.subheader("Election shift — topic rank change")
st.caption(
    "Rank of each topic by article count, pre vs. post 2024 general election. "
    "A positive rank change means the topic rose up the agenda after the election."
)

if {"pre_election", "post_election"} <= set(filt["election_period"].unique()):
    rank_df = (
        filt.groupby(["election_period", "topic_name"]).size()
        .reset_index(name="n")
    )
    pivot = rank_df.pivot(index="topic_name", columns="election_period", values="n").fillna(0)
    pivot["rank_pre"]  = pivot["pre_election"].rank(ascending=False, method="min").astype(int)
    pivot["rank_post"] = pivot["post_election"].rank(ascending=False, method="min").astype(int)
    pivot["rank_change"] = (pivot["rank_pre"] - pivot["rank_post"]).astype(int)
    pivot = pivot.reset_index().rename(columns={
        "topic_name": "Topic",
        "pre_election": "Pre-election articles",
        "post_election": "Post-election articles",
        "rank_pre": "Pre rank",
        "rank_post": "Post rank",
        "rank_change": "Rank change ↑",
    })
    pivot = pivot.sort_values("Rank change ↑", ascending=False)

    st.dataframe(
        pivot[["Topic", "Pre-election articles", "Post-election articles",
               "Pre rank", "Post rank", "Rank change ↑"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rank change ↑": st.column_config.NumberColumn(
                help="Positive = rose up the agenda after election"
            )
        },
    )
else:
    st.info("Both pre- and post-election data required for this table.")
