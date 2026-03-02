"""
Framing Analysis
================
Assigns each article a framing type based on keyword matching against
predefined framing dictionaries. Edit FRAMINGS below to adjust or extend.

Framing assigned = framing with most keyword hits in text_clean.
Ties → "mixed" | Zero matches → "unclassified"
"""

import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Framing Analysis | Education Policy Observatory",
    page_icon="🔬",
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

# ── Framing definitions ───────────────────────────────────────────────────────
# Edit keyword lists here to refine framing classification.
FRAMINGS: dict[str, list[str]] = {
    "economic":   ["cost", "budget", "funding", "spend", "financial", "money",
                   "resource", "afford", "investment", "cut"],
    "rights":     ["equity", "inclusion", "access", "rights", "equality",
                   "fair", "justice", "discrimination", "disadvantaged"],
    "crisis":     ["crisis", "failing", "urgent", "emergency", "shortage",
                   "concern", "alarm", "collapse", "pressure", "strain"],
    "evidence":   ["research", "data", "evidence", "study", "report",
                   "finding", "analysis", "evaluation", "survey"],
    "political":  ["government", "labour", "minister", "policy", "reform",
                   "parliament", "legislation", "conservative", "election"],
}

from model_pipeline.dashboard.supabase_loader import load_articles


# ── Data loading + framing assignment ────────────────────────────────────────
@st.cache_data
def load_and_frame() -> pd.DataFrame:
    df = load_articles()
    df["framing"] = df["text_clean"].apply(_assign_framing)
    return df


def _assign_framing(text: object) -> str:
    if not isinstance(text, str) or not text.strip():
        return "unclassified"
    text_lower = text.lower()
    scores = {
        framing: sum(1 for kw in keywords if kw in text_lower)
        for framing, keywords in FRAMINGS.items()
    }
    max_score = max(scores.values())
    if max_score == 0:
        return "unclassified"
    top = [f for f, s in scores.items() if s == max_score]
    return "mixed" if len(top) > 1 else top[0]


df = load_and_frame()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    countries = st.multiselect(
        "Country", options=sorted(df["country"].unique()),
        default=sorted(df["country"].unique()), key="fr_country"
    )
    types = st.multiselect(
        "Organisation type", options=sorted(df["type"].unique()),
        default=sorted(df["type"].unique()), key="fr_type"
    )

filt = df[df["country"].isin(countries) & df["type"].isin(types)]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Framing Analysis")
st.caption(
    "How do different organisations frame education policy topics? "
    "Framing is assigned by keyword matching — see the `FRAMINGS` dict "
    "at the top of this file to inspect or adjust the definitions."
)

with st.expander("Framing keyword definitions"):
    for framing, keywords in FRAMINGS.items():
        st.markdown(f"**{framing.title()}**: {', '.join(keywords)}")

st.divider()

# ── Topic selector ────────────────────────────────────────────────────────────
topics_sorted = sorted(filt["topic_name"].unique())
selected_topic = st.selectbox("Select a topic", topics_sorted)
topic_df = filt[filt["topic_name"] == selected_topic]

m1, m2, m3 = st.columns(3)
m1.metric("Articles", len(topic_df))
framing_counts = topic_df["framing"].value_counts()
m2.metric("Dominant framing", framing_counts.index[0].title() if len(framing_counts) else "—")
m3.metric(
    "Unclassified",
    f"{(topic_df['framing'] == 'unclassified').sum()} "
    f"({(topic_df['framing'] == 'unclassified').mean() * 100:.0f}%)",
)

st.divider()

# ── Framing breakdown ─────────────────────────────────────────────────────────
st.subheader("Framing breakdown")
norm_frame = st.toggle("Normalise (% of topic articles)", key="norm_frame")

frame_counts = topic_df["framing"].value_counts().reset_index()
frame_counts.columns = ["framing", "count"]
if norm_frame:
    frame_counts["value"] = (frame_counts["count"] / frame_counts["count"].sum() * 100).round(1)
    y_label = "% of topic articles"
else:
    frame_counts["value"] = frame_counts["count"]
    y_label = "Articles"

frame_counts["framing_label"] = frame_counts["framing"].str.title()

st.altair_chart(
    alt.Chart(frame_counts).mark_bar().encode(
        x=alt.X("value:Q", title=y_label),
        y=alt.Y("framing_label:N", sort="-x", title="Framing"),
        color=alt.Color("framing_label:N", legend=None),
        tooltip=["framing_label", alt.Tooltip("value:Q", title=y_label)],
    ).properties(height=220),
    use_container_width=True,
)

st.divider()

# ── Framing × source heatmap ──────────────────────────────────────────────────
st.subheader("Framing by organisation")

fr_src = topic_df.groupby(["source", "framing"]).size().reset_index(name="count")
src_totals = topic_df.groupby("source").size().reset_index(name="total")
fr_src = fr_src.merge(src_totals, on="source")
fr_src["pct"] = (fr_src["count"] / fr_src["total"] * 100).round(1)
fr_src["framing_label"] = fr_src["framing"].str.title()

st.altair_chart(
    alt.Chart(fr_src).mark_rect().encode(
        x=alt.X("framing_label:N", title="Framing"),
        y=alt.Y("source:N", title="Organisation"),
        color=alt.Color("pct:Q", scale=alt.Scale(scheme="blues"), title="% of source"),
        tooltip=["source", "framing_label", "count",
                 alt.Tooltip("pct:Q", title="%")],
    ).properties(height=240),
    use_container_width=True,
)

st.divider()

# ── Framing × election period ─────────────────────────────────────────────────
st.subheader("Framing shift: pre vs. post election")

if {"pre_election", "post_election"} <= set(topic_df["election_period"].unique()):
    fr_elec = topic_df.groupby(["election_period", "framing"]).size().reset_index(name="count")
    elec_totals = topic_df.groupby("election_period").size().reset_index(name="total")
    fr_elec = fr_elec.merge(elec_totals, on="election_period")
    fr_elec["pct"] = (fr_elec["count"] / fr_elec["total"] * 100).round(1)
    fr_elec["framing_label"] = fr_elec["framing"].str.title()
    fr_elec["period_label"] = fr_elec["election_period"].str.replace("_", " ").str.title()

    st.altair_chart(
        alt.Chart(fr_elec).mark_bar().encode(
            x=alt.X("pct:Q", title="% of period articles"),
            y=alt.Y("framing_label:N", sort="-x", title="Framing"),
            color=alt.Color("period_label:N", legend=alt.Legend(title="Period")),
            tooltip=["period_label", "framing_label",
                     alt.Tooltip("pct:Q", title="%")],
        ).properties(height=240),
        use_container_width=True,
    )
else:
    st.info("Both pre- and post-election data required for this chart.")

st.divider()

# ── Framing over time ─────────────────────────────────────────────────────────
st.subheader("Framing over time")

fr_time = topic_df.groupby(["month", "framing"]).size().reset_index(name="count")
month_totals = topic_df.groupby("month").size().reset_index(name="total")
fr_time = fr_time.merge(month_totals, on="month")
fr_time["pct"] = (fr_time["count"] / fr_time["total"] * 100).round(1)
fr_time["framing_label"] = fr_time["framing"].str.title()

ELECTION_DATE = pd.Timestamp("2024-07-04")

line = (
    alt.Chart(fr_time)
    .mark_line(interpolate="monotone", point=True)
    .encode(
        x=alt.X("month:T", title="Month"),
        y=alt.Y("pct:Q", title="% of monthly topic articles"),
        color=alt.Color("framing_label:N", legend=alt.Legend(title="Framing")),
        tooltip=["month:T", "framing_label",
                 alt.Tooltip("pct:Q", title="%")],
    )
    .properties(height=320)
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
