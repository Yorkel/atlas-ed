# Drift Monitor — Design & Interpretation

## What it does

The drift monitor (`model_pipeline/inference/drift_monitor.py`) compares each inference week's topic assignments against the training baseline to detect model-data mismatch. It computes six metrics per week and writes them to the `drift_metrics` table in Supabase.

Run after the batch runner completes:
```bash
python -m model_pipeline.inference.drift_monitor
```

Re-running is idempotent — it upserts on `week_number`, so results are updated rather than duplicated.

---

## Why it exists

A topic model trained on 2023–2025 articles is applied to 2026 articles. The policy landscape changes: new terminology appears, old topics become less relevant, political events shift discourse. The drift monitor answers a simple question: **is the model still a good fit for the data it's being asked to classify?**

This is not optional monitoring bolted on after the fact. It is part of the responsible AI argument that AtlasED makes: if a system classifies policy discourse, it should also monitor whether those classifications are trustworthy over time. A model that silently degrades while still producing confident-looking labels is worse than a model that flags its own uncertainty.

---

## The six metrics

### 1. JS divergence (Jensen-Shannon divergence)

**What it measures:** How different the inference week's topic distribution is from the training topic distribution.

**How it works:** Both distributions are 30-element probability vectors (one element per topic = proportion of articles assigned to that topic). JS divergence measures the distance between them on a 0–1 scale.

| Score | Meaning |
|---|---|
| 0.0 | Identical distributions — inference topics match training exactly |
| 0.0–0.05 | Very low drift — expected week-to-week variation |
| 0.05–0.15 | Low-moderate drift — may reflect genuine topic shifts or small sample noise |
| 0.15–0.30 | Moderate drift — worth investigating, but expected with small weekly batches |
| > 0.30 | High drift — the model's topic structure may not fit new data well |

**Why JS, not KL divergence:** KL divergence is undefined when any topic has zero probability (likely with 20 articles and 30 topics). JS divergence handles zeros gracefully.

**Alert threshold:** > 0.1 (triggers `alert_js_divergence`).

### 2. Mean confidence

**What it measures:** Average `dominant_topic_weight` across the week's articles. This is the weight of the strongest topic for each article.

**How to interpret:** If the model fits new data well, confidence should be similar to training. A sustained drop means new articles are further from the training vocabulary — the model is assigning topics but without strong conviction.

**Alert threshold:** Below 80% of the training baseline (triggers `alert_confidence_drop`).

### 3. Mean contestability

**What it measures:** Average normalised Shannon entropy across the week's articles.

**How to interpret:** Higher values mean topic weight is spread more uniformly — the model is less decisive. A gradual increase over time could indicate vocabulary drift (new terminology the model hasn't seen) or genuine shifts in how policy is discussed (more cross-cutting articles).

**No dedicated alert** — this is captured indirectly by the high-contestability rate.

### 4. High-contestability rate

**What it measures:** Fraction of the week's articles with `contestability_score > 0.5`.

**How to interpret:** These are articles where the model has no clear dominant topic. A rate above 50% means the majority of the week's articles have genuinely uncertain classifications. This is analytically interesting (cross-cutting policy moments) but also a signal that the model may be struggling.

**Alert threshold:** > 50% (triggers `alert_high_contestability`).

### 5. Topic concentration (HHI)

**What it measures:** Herfindahl-Hirschman Index — `sum(p_i^2)` where `p_i` is the proportion of articles on topic `i`.

| Score | Meaning |
|---|---|
| 0.033 | Perfectly uniform — all 30 topics equally represented |
| 0.05–0.10 | Mild concentration — some topics more common, typical for real data |
| 0.10–0.20 | Moderate concentration — a few topics dominate the week |
| > 0.20 | High concentration — most articles assigned to a small number of topics |

**How to interpret:** High HHI means the model is funnelling new articles into a few topics. This could be genuine (a major policy event dominates the news that week) or problematic (the model can't distinguish between topics on new vocabulary and defaults to a few catch-all topics).

**No dedicated alert threshold** — interpret alongside JS divergence and topic coverage.

### 6. Topics present

**What it measures:** Number of distinct `topic_num` values assigned in the week.

**How to interpret:** With 30 topics and only 17–28 articles per week, it's mathematically impossible to hit all 30. The training set uses all 30 because it has 3,972 articles. Low coverage is expected with small batches and is not inherently concerning — it becomes a signal only if the same topics are consistently absent.

**Alert threshold:** < 15 topics (triggers `alert_low_topic_coverage`).

---

## What to do when alerts trigger

Alerts are informational — they do not halt the pipeline or indicate a system failure. When alerts appear:

1. **Check sample size first.** With 17–28 articles per week, JS divergence, topic concentration, and topic coverage are all affected by sampling noise. A "high" JS divergence of 0.25 with 20 articles does not mean the same thing as 0.25 with 2,000 articles.

2. **Look for sustained trends.** A single week with elevated metrics is noise. Three consecutive weeks with rising JS divergence and falling confidence suggests genuine drift.

3. **Inspect the articles.** Query Supabase for the specific week's articles and check whether the topic assignments are sensible. Are high-contestability articles genuinely cross-cutting, or are they being misclassified?

4. **Consider retraining.** If drift is sustained and confidence is degrading, the model may need retraining on a corpus that includes more recent articles. This is expected — NMF topic models are static and will eventually diverge from evolving language.

---

## Interpreting the first run (6 weeks, 128 articles)

The initial backfill results show:

- **JS divergence: 0.20–0.29** — Elevated, but expected with small weekly samples. The training set has 3,972 articles distributed across 30 topics; weekly batches of 17–28 articles cannot replicate that distribution precisely. This is sampling noise, not model failure.

- **Confidence: 0.1159–0.1321 (baseline: 0.1328)** — Stable. No alert triggered. The model is similarly confident on 2026 articles as it was on 2023–2025 training data.

- **Contestability: 0.5012–0.5296 (baseline: 0.4757)** — Slightly higher. Inference articles are marginally harder to classify. This could reflect genuine topic shifts in the 2026 policy landscape or vocabulary drift.

- **High-contestability rate: 54.5–63.6% (baseline: 50.7%)** — Slightly elevated. Consistent with the contestability finding above.

- **Topics present: 11–17 per week** — Low, but a direct consequence of small sample sizes. Not concerning unless the same topics are consistently absent.

**Overall assessment:** No evidence of significant model degradation. Confidence is stable. Contestability is marginally higher — worth monitoring over time but not actionable yet.

---

## How this connects to responsible AI

The drift monitor operationalises a principle that runs through the AtlasED project: **AI systems used in policy analysis should monitor themselves, not just produce outputs.**

Most deployed classification systems have no concept of self-monitoring. They process new data with the same confidence regardless of whether the model still fits. A policymaker consuming those outputs has no way to know whether the labels were assigned with high certainty or are the model's best guess on unfamiliar text.

The drift monitor addresses this by:

1. **Quantifying model-data fit over time** — JS divergence and confidence trends make it visible whether the model is still appropriate for the data it's classifying.

2. **Flagging when human review is needed** — alerts don't automate decisions; they surface situations where a human should inspect the results before acting on them.

3. **Creating an audit trail** — the `drift_metrics` table provides a time-series record of model performance. If someone asks "was the model still reliable when it classified articles in week 4?", the answer is in the data.

4. **Making the case for retraining** — rather than retraining on an arbitrary schedule, the drift metrics provide evidence-based triggers. Retrain when the data says the model is struggling, not when a calendar says it's time.

This is the same argument the contestability score makes at the article level, extended to the system level: transparency about what the model knows and doesn't know, at every layer.

---

## Limitations

- **Small sample sizes** — 17–28 articles per week is too few for precise distributional comparisons. JS divergence and topic coverage are noisy at this scale. With 100+ articles per week in production, metrics would stabilise.

- **No OOV detection** — tracking out-of-vocabulary rate (what % of new article terms aren't in the training vocabulary) would require loading the fitted vectorizer. The drift monitor works from Supabase data only, which doesn't include per-token vocabulary information.

- **No reconstruction error** — measuring how well NMF reconstructs new article vectors would be the most direct test of model-data fit. This requires the model artefacts (vectorizer + NMF matrices). Could be added to the API as a `/diagnostics` endpoint in future.

- **Static thresholds** — the alert thresholds are heuristic, not statistically derived. With more data, adaptive thresholds (e.g., 2 standard deviations from a rolling baseline) would be more principled.
