import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

st.set_page_config(
    page_title="Organisations | Education Policy Observatory",
    page_icon="🏛️",
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
DATA_PATH = ROOT / "data" / "evaluation_outputs" / "dashboard_data.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["country"] = "England"
    df["topic_name"] = df["topic_name"].str.replace("_", " ").str.title()
    df["source"] = df["source"].str.upper()
    return df


df = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    countries = st.multiselect(
        "Country",
        options=sorted(df["country"].unique()),
        default=sorted(df["country"].unique()),
        key="org_country",
    )
    types = st.multiselect(
        "Organisation type",
        options=sorted(df["type"].unique()),
        default=sorted(df["type"].unique()),
        key="org_type",
    )

filt = df[df["country"].isin(countries) & df["type"].isin(types)]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Organisation Analysis")
st.caption("Who publishes on which topics, and how does that vary by organisation type?")

st.divider()

# ── Heatmap: source × topic ───────────────────────────────────────────────────
st.subheader("Source × topic heatmap")
normalise_heat = st.toggle("Normalise within each source (% of source output)", key="norm_heat")

heat_raw = (
    filt.groupby(["source", "topic_name"]).size()
    .reset_index(name="count")
)

if normalise_heat:
    source_totals = filt.groupby("source").size().reset_index(name="total")
    heat_raw = heat_raw.merge(source_totals, on="source")
    heat_raw["value"] = (heat_raw["count"] / heat_raw["total"] * 100).round(1)
    color_title = "% of source output"
else:
    heat_raw["value"] = heat_raw["count"]
    color_title = "Articles"

heatmap = (
    alt.Chart(heat_raw)
    .mark_rect()
    .encode(
        x=alt.X("topic_name:N", title="Topic", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("source:N", title="Organisation"),
        color=alt.Color(
            "value:Q",
            scale=alt.Scale(scheme="magma"),
            title=color_title,
        ),
        tooltip=["source", "topic_name",
                 alt.Tooltip("value:Q", title=color_title)],
    )
    .properties(height=300)
)
st.altair_chart(heatmap, use_container_width=True)

st.divider()

# ── Grouped bar: topic share by org type ─────────────────────────────────────
st.subheader("Topic focus by organisation type")
st.caption("Normalised within each organisation type to show relative emphasis.")

type_topic = (
    filt.groupby(["type", "topic_name"]).size()
    .reset_index(name="count")
)
type_totals = filt.groupby("type").size().reset_index(name="total")
type_topic = type_topic.merge(type_totals, on="type")
type_topic["pct"] = (type_topic["count"] / type_topic["total"] * 100).round(1)

top_topics_global = filt["topic_name"].value_counts().head(12).index.tolist()
type_topic_top = type_topic[type_topic["topic_name"].isin(top_topics_global)]

bar = (
    alt.Chart(type_topic_top)
    .mark_bar()
    .encode(
        x=alt.X("pct:Q", title="% of org-type output"),
        y=alt.Y("topic_name:N", sort="-x", title=""),
        color=alt.Color("type:N", legend=alt.Legend(title="Org type")),
        row=alt.Row("type:N", title=""),
        tooltip=["type", "topic_name", alt.Tooltip("pct:Q", title="%")],
    )
    .properties(height=120)
)
st.altair_chart(bar, use_container_width=True)

st.divider()

# ── Side-by-side source comparison ───────────────────────────────────────────
st.subheader("Compare two sources")

all_sources = sorted(filt["source"].unique())
if len(all_sources) >= 2:
    col_a, col_b = st.columns(2)
    src_a = col_a.selectbox("Source A", all_sources, index=0, key="cmp_a")
    src_b = col_b.selectbox("Source B", all_sources, index=min(1, len(all_sources) - 1), key="cmp_b")

    for col, src in [(col_a, src_a), (col_b, src_b)]:
        src_df = filt[filt["source"] == src]
        counts = (
            src_df["topic_name"].value_counts(normalize=True)
            .mul(100).round(1).head(10).reset_index()
        )
        counts.columns = ["topic_name", "pct"]
        chart = (
            alt.Chart(counts)
            .mark_bar(color="#4e79a7")
            .encode(
                x=alt.X("pct:Q", title="% of output"),
                y=alt.Y("topic_name:N", sort="-x", title=""),
                tooltip=["topic_name", alt.Tooltip("pct:Q", title="%")],
            )
            .properties(height=300, title=src)
        )
        col.altair_chart(chart, use_container_width=True)
else:
    st.info("Filter produces fewer than 2 sources — adjust filters to enable comparison.")
