# AtlasED Analysis Steps

After the pipeline has run (training + inference pushed to Supabase), follow these steps to validate, analyse, and evaluate the results.

## Prerequisites

Download the processed data:

```bash
python sync_from_supabase.py --download
```

This creates `data/processed_data_YYYY-MM-DD.csv` with all topic allocations from `articles_topics`.

---

## Step 1: Validation checks

Confirm the pipeline ran correctly before doing any analysis.

**Row counts:**
- Total rows per country (eng, sco, irl)
- Total rows per dataset_type (training, inference)
- Expected: eng ~3,939 training + ~231 inference, sco ~511 + ~120, irl ~1,040 + ~67

**Topic names:**
- All countries should have named topics (e.g. `child_welfare_protection`), not generic labels (`topic_0`, `topic_8`)
- England: 30 topics, Scotland: 15 topics, Ireland: 15 topics

**Data quality:**
- No missing `dominant_topic` or `topic_probabilities`
- `article_date` ranges: training should span 2023–2025, inference should span Jan 9 – Mar 20 2026
- `contestability_score` values between 0 and 1
- Each country's articles use the correct model (`run_id` matches config.yaml)

**Derived fields audit (from `articles_raw`):**

Check for missing derived variables that should be populated from existing data:
- `institution_name` — derived from URL domain
- `organisation_category` — e.g. gov, media, think tank, etc. (derived from source)
- `item_type` — article type classification

During this step, catalogue what's missing and define the derivation rules. Once rules are confirmed, build a script to:
1. Pull `articles_raw`
2. Derive missing fields from existing columns (URL, source, etc.)
3. Update `articles_raw` in Supabase
4. Re-sync and re-run the pipeline

Do this *before* moving on to analysis so downstream results are clean.

**Before re-running the pipeline**, batch up all fixes found during inspection (derived fields, data quality issues, model_type naming, etc.) and do a single clean run:
1. Fix source data in `articles_raw`
2. `TRUNCATE articles_topics; TRUNCATE drift_metrics;`
3. `python -m model_pipeline.inference.batch_runner --mode all_training_inference`
4. `python -m model_pipeline.inference.drift_monitor`

---

## Step 2: Within-country drift

Check whether the topics seen in weekly inference are stable relative to each country's own training data.

**Approach:**
- Compare each country's weekly inference topic distribution against its own training baseline
  - eng weekly vs eng training
  - sco weekly vs sco training
  - irl weekly vs irl training
- Use Jensen-Shannon divergence (already implemented in `drift_monitor.py`)
- Track drift per week to spot trends

**Questions this answers:**
- Are the topics shifting over time within each country?
- Are any weeks unusually different from the training data?
- Is the model still a good fit for incoming articles?

**Action needed:** Update `drift_monitor.py` to compute baselines per country (currently only uses England as baseline for all).

---

## Step 3: Cross-country comparison

Compare topic distributions between countries to surface structural differences in education policy discourse.

**Approach:**
- Compare training distributions: eng vs sco vs irl (overall topic landscape per country)
- Compare weekly distributions for the same time periods across countries
- Identify topics that are:
  - **Universal** — appear as dominant topics in all three countries
  - **Jurisdiction-specific** — dominant in one country but absent/rare in others
- Compare contestability score distributions across countries (are some countries' articles harder to classify?)

**Questions this answers:**
- How does education policy discourse differ across England, Scotland, and Ireland?
- Which policy themes are shared, which are unique?
- Are specification choices (what gets covered) visibly different?

---

## Step 4: Analysis of findings

Deep dive into what the topic allocations reveal about education policy.

### Country-by-country
- Dominant topics per country (ranked by article count)
- Topic trends over time (training period 2023–2025, then weekly 2026)
- High-contestability articles: what topics are they split between?
- Pre-election vs post-election topic shifts (using `election_period` field)
- Source-level analysis: do different sources cover different topics?

### Cross-country
- Topic overlap matrix: for each pair of countries, how similar are their topic distributions?
- Unique policy themes per country
- Coverage gaps: topics that exist in one country's model but have no equivalent in another

### LLM component
An LLM can add value here by:
- **Narrative summaries:** Generate plain-language descriptions of topic trends per country and across countries
- **Topic comparison:** Where topic names differ across countries but content may overlap, use an LLM to compare the top words/articles and assess semantic similarity
- **Policy interpretation:** Contextualise topic shifts in terms of known policy events (elections, legislation, reviews)
- **Cross-country synthesis:** Produce comparative summaries highlighting where countries align and diverge on education policy themes

---

## Step 5: Model evaluation

Assess whether the NMF models are performing well.

**Topic quality:**
- Coherence scores (computed during training in notebooks)
- Stability across random seeds (computed during training)
- Are topic names interpretable and distinct?

**Allocation quality:**
- Contestability score distribution: if most articles have very high contestability, topics may not be well-separated
- Dominant topic weight distribution: very low weights suggest poor fit
- Are any topics "catch-all" (assigned to a disproportionate number of articles)?

**Drift as evaluation signal:**
- Sustained high drift in weekly inference may indicate the model needs retraining
- Sudden spikes may indicate a new policy event rather than model degradation

---

## Summary of outputs

| Step | Input | Output |
|------|-------|--------|
| 1. Validation | `processed_data_*.csv` | Pass/fail checks, data quality report |
| 2. Within-country drift | `articles_topics` (Supabase) | JS divergence per country per week |
| 3. Cross-country comparison | `processed_data_*.csv` | Topic overlap matrices, unique topic lists |
| 4. Analysis | `processed_data_*.csv` + LLM | Country reports, cross-country synthesis |
| 5. Evaluation | Training notebooks + drift metrics | Model quality assessment |
