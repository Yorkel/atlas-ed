#######################
# Imports
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from pathlib import Path

#######################
# Page config
st.set_page_config(
    page_title="Public Conversations on Education",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

alt.themes.enable("dark")

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding: 1.5rem 2rem;
}

[data-testid="stMetric"] {
    background-color: #2b2b2b;
    text-align: center;
    padding: 18px 0;
    border-radius: 6px;
}

[data-testid="stMetricLabel"] {
    justify-content: center;
}

h3 {
    margin-bottom: 0.5rem;
}

</style>
""", unsafe_allow_html=True)

#######################
# Load data
@st.cache_data
def load_data():
    path = Path("/workspaces/AM1_topic_modelling/data/evaluation_outputs/dashboard_data.csv")
    return pd.read_csv(path, parse_dates=["date"])

df = load_data()

#######################
# Sidebar
with st.sidebar:
    st.title("📊 Education Policy Dashboard")

    selected_sources = st.multiselect(
        "Organisation",
        options=sorted(df["source"].unique()),
        default=sorted(df["source"].unique())
    )

    selected_topics = st.multiselect(
        "Topic",
        options=sorted(df["topic_name"].unique()),
        default=sorted(df["topic_name"].unique())
    )

    selected_period = st.multiselect(
        "Election period",
        options=df["election_period"].unique(),
        default=df["election_period"].unique()
    )

df_filt = df[
    df["source"].isin(selected_sources)
    & df["topic_name"].isin(selected_topics)
    & df["election_period"].isin(selected_period)
]

#######################
# Header
st.title("Mapping the Public Conversation on Education")
st.caption(
    "Tracking attention, agenda shifts, and organisational focus in education policy discourse (2023–2025)."
)

#######################
# Layout
col = st.columns((1.4, 4.6, 2), gap="large")

# =====================
# LEFT COLUMN — METRICS
# =====================
with col[0]:
    st.subheader("Agenda snapshot")

    st.metric("Articles", len(df_filt))
    st.metric("Topics", df_filt["topic_name"].nunique())
    st.metric("Organisations", df_filt["source"].nunique())

    st.divider()

    st.subheader("Election shift")

    rank = (
        df_filt
        .groupby(["election_period", "topic_name"])
        .size()
        .reset_index(name="n")
    )

    if {"pre_election", "post_election"} <= set(rank["election_period"]):
        pivot = rank.pivot(
            index="topic_name",
            columns="election_period",
            values="n"
        ).fillna(0)

        pivot["rank_pre"] = pivot["pre_election"].rank(ascending=False)
        pivot["rank_post"] = pivot["post_election"].rank(ascending=False)
        pivot["rank_change"] = pivot["rank_pre"] - pivot["rank_post"]

        top_mover = pivot.sort_values("rank_change", ascending=False).head(1)

        st.metric(
            "Biggest riser",
            top_mover.index[0].replace("_", " ").title(),
            f"+{int(top_mover['rank_change'].iloc[0])}"
        )
    else:
        st.info("Select both pre- and post-election periods.")

# =====================
# CENTRE COLUMN — TRENDS
# =====================
with col[1]:
    st.subheader("Topic attention over time")

    top_topics = (
        df_filt["topic_name"]
        .value_counts()
        .head(6)
        .index
    )

    time_df = (
        df_filt[df_filt["topic_name"].isin(top_topics)]
        .groupby(["date", "topic_name"])
        .size()
        .reset_index(name="n_articles")
    )

    line = alt.Chart(time_df).mark_line(interpolate="monotone").encode(
        x=alt.X("date:T", title="Month"),
        y=alt.Y("n_articles:Q", title="Articles"),
        color=alt.Color("topic_name:N", legend=alt.Legend(title="Topic")),
        tooltip=["topic_name", "n_articles"]
    ).properties(height=320)

    election_line = alt.Chart(
        pd.DataFrame({"date": [pd.Timestamp("2024-07-01")]})
    ).mark_rule(
        strokeDash=[4, 4],
        strokeWidth=2,
        color="white"
    ).encode(x="date:T")

    st.altair_chart(line + election_line, use_container_width=True)

    st.divider()

    st.subheader("Organisational focus (heatmap)")

    heat_df = (
        df_filt
        .groupby(["source", "topic_name"])
        .size()
        .reset_index(name="n_articles")
    )

    heatmap = alt.Chart(heat_df).mark_rect().encode(
        y=alt.Y("source:N", title="Organisation"),
        x=alt.X("topic_name:N", title="Topic"),
        color=alt.Color(
            "n_articles:Q",
            scale=alt.Scale(scheme="magma"),
            title="Articles"
        ),
        tooltip=["source", "topic_name", "n_articles"]
    ).properties(height=300)

    st.altair_chart(heatmap, use_container_width=True)

# =====================
# RIGHT COLUMN — DETAIL
# =====================
with col[2]:
    st.subheader("Top topics")

    top_topics_df = (
        df_filt["topic_name"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_topics_df.columns = ["Topic", "Articles"]

    st.dataframe(
        top_topics_df,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.subheader("Article explorer")

    selected_topic = st.selectbox(
        "Topic",
        df_filt["topic_name"].unique()
    )

    subset = df_filt[df_filt["topic_name"] == selected_topic]

    selected_id = st.selectbox(
        "Article",
        subset["article_id"].values
    )

    text = subset.loc[
        subset["article_id"] == selected_id,
        "text_clean"
    ].iloc[0]

    st.text_area(
        "Cleaned article text",
        text,
        height=350
    )
