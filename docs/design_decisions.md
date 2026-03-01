# AM1 Topic Modelling — Design Decisions, Trade-offs & Examiner Q&A

---

## 1. Project Architecture Overview

The pipeline is structured as a sequential set of numbered steps (s01–s10), each with a single responsibility:

```
s01  Data loading          → raw DataFrame
s02  Cleaning              → text_clean column
s03  spaCy processing      → text_final (lemmatised tokens)
s04  TF-IDF vectorisation  → sparse matrix X
s05  NMF training          → W (doc-topic) and H (topic-word) matrices
s06  Topic allocation      → dominant topic per document
s07  Evaluation            → coherence scores, stability scores
s08  Save outputs          → versioned run artifacts on disk
s09  MLflow logging        → experiment tracking
s10  Pipeline              → orchestrates s01–s09 end to end
```

Each step is independently runnable for debugging. Outputs flow forward as typed dataclasses, not raw variables.

---

## 2. Model Choice — Why NMF, Not LDA or BERTopic

### Decision
Non-negative Matrix Factorisation (NMF) with TF-IDF input.

### Rationale
- **NMF produces parts-based, additive representations.** Topics are combinations of words with non-negative weights, making them directly interpretable — a word either contributes to a topic or it doesn't.
- **LDA trade-off:** LDA is a generative probabilistic model that assumes a Dirichlet prior over topic distributions. In practice on short, scraped policy texts, LDA tends to produce diffuse, overlapping topics that are harder to label. NMF consistently produced cleaner, more domain-coherent topics on this corpus.
- **BERTopic trade-off:** BERTopic uses sentence-transformer embeddings (contextual, dense vectors) which are more semantically rich but computationally expensive, harder to audit (embedding space is opaque), and require GPU resources for large-scale use. NMF with TF-IDF is fully transparent — every word's contribution to every topic is directly inspectable.
- **Stakeholder interpretability:** Policy analysts need to understand and trust the topics. NMF's additive structure (top words per topic) is easier to explain than neural embedding spaces.

### Trade-off accepted
NMF cannot capture polysemy (the same word meaning different things in different contexts). BERTopic would handle this better. This is an acknowledged limitation.

---

## 3. Text Representation — Why TF-IDF, Not Embeddings

### Decision
`sklearn.TfidfVectorizer` with `min_df=3`, `max_df=0.85`, `max_features=3000`, `ngram_range=(1,2)`.

### Rationale
- TF-IDF is the standard input for NMF. It up-weights terms that are distinctive to specific documents and down-weights terms that appear everywhere — exactly the signal NMF needs to find topic clusters.
- Bigrams (`ngram_range=(1,2)`) capture domain-specific phrases like "school funding", "teacher pay", "free school meals" that unigrams would split.
- `max_features=3000` limits vocabulary to the most informative terms, reducing noise and keeping the matrix sparse and computationally tractable.
- `min_df=3` removes terms appearing in fewer than 3 documents (likely typos, scraped artefacts). `max_df=0.85` removes terms appearing in more than 85% of documents (stopword-like behaviour specific to this corpus).

### Trade-off accepted
TF-IDF ignores word order and semantics. "School funding" and "funding school" are treated identically. Contextual embeddings (BERT, sentence-transformers) would capture meaning more richly but at the cost of transparency and computational cost.

---

## 4. Number of Topics — Why 30

### Decision
`n_components=30` in the NMF model.

### Rationale
- The coherence sweep (`s07_evaluation.py`) tested NMF across `range(5, 80, 5)` topic counts using the `c_v` coherence metric (gensim). The sweep results are saved to `experiments/outputs/runs/<run_id>/evaluation/coherence_sweep.csv`.
- 30 topics was identified as the point where coherence peaks before declining — beyond ~35 topics, topics begin to fragment into near-duplicates.
- Topic stability analysis (cosine similarity of H matrices across random seeds) confirmed that 30-topic solutions are highly reproducible across different random initialisations.
- Domain validation: 30 topics covers the breadth of education policy discourse (curriculum, funding, SEND, safeguarding, teacher pay, etc.) without becoming too granular.

### Trade-off accepted
30 is a data-driven choice but still involves judgement. A different corpus sample or time window might suggest a different optimal count. This is why the sensitivity sweep exists — to show the decision is evidence-based, not arbitrary.

---

## 5. NMF Initialisation — Why `nndsvd`

### Decision
`init='nndsvd'` (Non-negative Double Singular Value Decomposition).

### Rationale
- `nndsvd` initialises the NMF factors using a deterministic SVD-based method rather than random initialisation. This produces faster convergence and more reproducible results.
- Unlike `random` init, `nndsvd` does not require a `random_state` to be deterministic — but we set one anyway for full reproducibility.
- `nndsvda` (used in stability evaluation) fills zero entries with the dataset mean, which is more appropriate for dense approximations but slightly slower for the main training run.

---

## 6. Preprocessing Pipeline — Key Decisions in s02 and s03

### s02 — Text Cleaning
- **Why structural cleaning first:** Scraped web text contains boilerplate (cookie notices, subscription prompts, navigation text) that would otherwise pollute the vocabulary. Removing these before any NLP processing prevents them from influencing topic assignments.
- **Why unicode normalisation:** Source text comes from multiple websites with inconsistent encoding. Normalising to ASCII-compatible form prevents the same word appearing as multiple vocabulary items.

### s03 — spaCy Processing
- **Why lemmatisation:** Reduces inflected forms to their base form ("schools" → "school", "funding" → "fund"), reducing vocabulary size and improving topic coherence.
- **Why POS filtering (NOUN, PROPN, ADJ only):** Verbs and adverbs carry less topical signal in policy text. Keeping only nouns, proper nouns, and adjectives improves topic purity.
- **Why remove PERSON entities:** Named individuals (politicians, academics) would create spurious topics around people rather than policy areas.
- **Why spaCy's default stopwords (not custom):** Using the default stopword list ensures the preprocessing is reproducible and auditable. A custom list risks over-fitting the preprocessing to the training corpus.

---

## 7. Pipeline Architecture — Why Numbered Steps (s01–s10)

### Decision
Each pipeline stage is a separate importable module with a `run_X()` function.

### Rationale
- **Debuggability:** Each step can be run independently (`python -m model_pipeline.training.s04_vectorisation`) to isolate failures.
- **Testability:** Each function has clearly typed inputs and outputs (dataclasses), making unit testing straightforward.
- **Separation of concerns:** Training logic (s05), evaluation logic (s07), and saving logic (s08) are independent. Changing how models are saved does not affect training.
- **Reusability:** The API (`main.py`) imports `run_cleaning` and `run_spacy_processing` directly from s02 and s03 rather than duplicating logic — ensuring training and inference use identical preprocessing.

---

## 8. Configuration — Why `config.yaml`

### Decision
All tunable parameters (n_topics, TF-IDF settings, random seeds) live in `config.yaml` at the project root.

### Rationale
- Separates hyperparameters from code — changing n_topics does not require editing source files.
- Enables sensitivity testing: running the pipeline with different configs and comparing outputs.
- Makes the experiment reproducible: the config is version-controlled alongside the code.

---

## 9. Model Serialisation — Why Joblib, Why Versioned Runs

### Decision
`joblib.dump()` for sklearn objects. Each run saved to `experiments/outputs/runs/<timestamp>/`.

### Rationale
- **Joblib over pickle:** Joblib is the recommended serialisation format for sklearn objects. It handles large numpy arrays more efficiently than standard pickle and is the sklearn-endorsed approach.
- **Versioned runs:** Timestamped run folders ensure no previous model is ever overwritten. You can compare outputs from different training runs side by side. The API auto-loads the most recent run.
- **`run_metadata.json`:** Every run saves its full configuration (n_topics, init, random_state, TF-IDF params, reconstruction error) alongside the model files. This provides a full audit trail.

---

## 10. API Design — Why FastAPI, Why Batch Endpoint

### Decision
FastAPI with a `POST /predict` endpoint accepting a list of articles.

### Rationale
- **FastAPI over Flask:** FastAPI generates automatic OpenAPI documentation (`/docs`), enforces request/response schemas via Pydantic, and is async-native. This reduces boilerplate and makes the API self-documenting.
- **Pydantic schemas:** Explicit `ArticleIn`, `PredictRequest`, `TopicResult`, `PredictResponse` models mean the API contract is clear, validated, and documented automatically.
- **Batch endpoint:** The intended use case is weekly processing of new scraped articles, not real-time single queries. A batch endpoint is more efficient — spaCy processes multiple texts in one pass.
- **Singleton model loading:** Models load once at startup (via `lifespan`) and are cached in `_bundle`. This avoids reloading 700KB+ of joblib files on every request.

### Containerisation — Why Docker (and why FastAPI + Docker together)

Docker and FastAPI are independent tools — you don't need Docker to run FastAPI (you can run it locally with `uvicorn`), and you don't need FastAPI to use Docker (you can containerise anything). Both are used here because neither alone solves the full problem.

**Why FastAPI:** `batch_runner.py` (planned — `model_pipeline/inference/`, same repo) needs to send articles to the model and receive topic assignments back without loading the model artefacts directly. FastAPI provides an HTTP interface — `POST /predict` — so the batch runner can call the model via a URL. This satisfies the apprenticeship requirement of demonstrating a deployed, callable API rather than a local model invocation.

**Why Docker:** The target deployment platforms require the app to be packaged as a container image. Docker bundles the FastAPI app, the trained model artefacts (`vectorizer.joblib`, `nmf_model.joblib`, `topic_names.json`), and all Python dependencies into a single portable image that runs identically regardless of where it is deployed.

**Together:** Docker makes the FastAPI app deployable to any cloud platform reproducibly. FastAPI makes the model callable over HTTP. Neither alone delivers a remotely-accessible, reproducible inference service.

**Deployment platform (interim):** Render (free tier, `https://atlased-api.onrender.com`). Render detects the Dockerfile in the repo, builds the image, and deploys it automatically on every push to `main`. The free tier spins down after 15 minutes of inactivity (cold-start ~30–50s on next request). This is acceptable for batch inference and development.

**Deployment platform (production):** TBC — pending UCL ISD approval via Cloud@UCL (AWS or Azure). The Docker image is platform-agnostic and can be redeployed to AWS App Runner, ECS, or Azure Container Apps with no code changes. See `docs/uclcheck.md` for the approval checklist.

### Why Two Requirements Files

- **`requirements.txt`** — full development environment. Includes everything needed to run the training pipeline, experiments, Jupyter notebooks, evaluation, and Supabase upload. This is what you install locally with `pip install -r requirements.txt`.
- **`requirements-api.txt`** — minimal API-only dependencies (9 packages). This is what the Dockerfile installs. It includes only what the FastAPI inference API needs to run: `fastapi`, `uvicorn`, `pydantic`, `pandas`, `numpy`, `scikit-learn`, `spacy`, `joblib`, `PyYAML`. No Jupyter, no gensim, no supabase, no experiment tracking.

The split exists because Docker images should be as small as possible. Installing Jupyter and gensim inside a deployed API container would add hundreds of megabytes for packages the API never uses.

---

## 11. Database — Why Supabase (PostgreSQL)

### Decision
Supabase (hosted PostgreSQL) as the central data store.

### Rationale
- **Structured data:** Articles and topic assignments have fixed, known schemas. A relational database is appropriate — document stores (MongoDB) add complexity for no benefit here.
- **SQL for dashboard queries:** Filtering articles by topic, date range, and source requires joins and aggregations that SQL handles natively and efficiently.
- **Supabase over self-hosted PostgreSQL:** Supabase provides a managed hosted instance, a built-in table viewer, and a Python client (`supabase-py`). This demonstrates production deployment rather than a local database.
- **Free tier:** Sufficient for this project's data volume (thousands of articles, not millions).

### Trade-off accepted
Supabase's free tier pauses inactive projects after 1 week. For a production system this would require a paid plan. For a university submission, this is acceptable.

---

## 12. Dashboard Design Decisions

### Structure — Why Multi-Page
A multi-page Streamlit app (using the `pages/` directory convention) gives each audience a focused surface rather than forcing all content onto one scrolling page:

| Page | Purpose |
|---|---|
| Overview | KPIs, source breakdown, navigation |
| Topic Explorer | Browse topics, top keywords, read articles, contestability |
| Trends Over Time | Monthly trends, election shift analysis |
| Organisation Analysis | Source × topic heatmap, type-stratified comparison |
| Framing Analysis | Keyword-based framing classification per topic |

### Why No Model Monitoring Page on the Dashboard
Target users (researchers, policy analysts) have no interest in reconstruction error or coherence scores. Model quality evidence is documented in `design_decisions.md` and `run_metadata.json`. Adding a monitoring page would dilute the dashboard's focus.

### Source Imbalance — Why Surface It Rather Than Hide It
SchoolsWeek accounts for ~69% of the corpus. **Decision:** Surface this explicitly on the Overview page and offer a raw/normalised toggle on every chart. Normalised = (source count / source total) × 100. This makes each source comparable regardless of volume and frames the imbalance as a finding (ed journalism dominates education policy discourse), not a flaw to hide.

### Framing Analysis — Why Keyword Matching, Not a New Model
Framing types are defined as keyword lists at the top of the page file. Each article is assigned the framing whose keywords appear most in `text_clean`. A separate classifier would require labelled training data that doesn't exist. Keyword matching is transparent, auditable, and inspectable — appropriate for a contestability analysis.

### Contestability Score — Why Entropy Replaced the Gap Metric

The original contestability metric (`1 - (top_weight - second_weight)`) measured the gap between the top two topic weights. This is structurally broken with 30 topics: NMF spreads weight so thinly that even a clearly dominant topic might score 0.12 while the runner-up scores 0.08 — producing a contestability of 0.96. Every article scores 0.9+, making the metric meaningless.

The replacement is **normalised Shannon entropy** across all 30 topic weights:
```
H = -Σ p_i * log(p_i)
contestability_score = H / log(n_topics)
```
- Score near 0 = all weight on one topic = confident, robust assignment.
- Score near 1 = weight spread uniformly = high uncertainty, genuinely contested.

Entropy measures the shape of the entire distribution, not just the top two. This produces a meaningful spread: articles dominated by one theme score low, articles spanning multiple policy framings score high. Normalising by `log(n_topics)` keeps the 0–1 scale consistent regardless of topic count.

### Country Filter — Why Placeholder Now
All current data is England. A `country = "England"` column is added at load time. When Scotland/ROI data are added later, the sidebar filter becomes functional with no code changes.

### Future Dashboard Additions (Planned)
- **Discourse concentration score**: HHI-style per-topic source concentration, showing which topics are dominated by a single organisation
- **Framing shift over time**: Whether framing diversity narrowed or broadened post-2024 election

---

## 13. Source Attribution — Considering Organisation Type Over Name

### Decision Under Consideration
Removing individual organisation names from analysis outputs and replacing them with category types only (e.g. `ed_journalism`, `think_tank`, `gov_inst`, `prof_body`, `ed_res_org`).

### Rationale

**Methodologically, this is how NMF at scale actually works.** When the model processes 3,972 documents, it extracts latent topic patterns across the entire corpus. The model does not "know" it is reading a SchoolsWeek article versus a GOV.UK press release — it sees a bag of weighted terms and assigns topic probabilities. Source metadata is added back in at the analysis stage. The unit of analysis is the document-topic relationship, not the author-text relationship.

This is methodologically distinct from a traditional literature review, where you read and cite specific texts by specific named authors. The pipeline is closer to how large-scale language modelling works: hoovering up text to learn patterns, not to attribute or reproduce individual pieces.

**This strengthens the governance argument.** A core claim of AtlasED is that AI tools encode assumptions that require governance. Surfacing organisation types rather than named organisations makes this argument more legible — it shows that the model is identifying structural patterns in discourse (how journalism, think tanks, and government bodies frame education policy differently) rather than profiling individual outlets. The governance question becomes "what assumptions are baked into the category schema?" which is a more tractable and productive question than "what does SchoolsWeek publish?"

**It also reduces copyright and attribution sensitivity.** Displaying category-level analysis rather than source-level analysis moves further from anything that could be construed as reproducing or substituting for named publications.

### Trade-off
Organisation names are retained internally (in Supabase and pipeline outputs) for data quality, deduplication, and future retraining decisions. Only the public-facing dashboard and website would show category types. Named source breakdown remains available as an internal/researcher view.

### Status
Under consideration. Not yet implemented.

---

## 14. Inference Pipeline — Batch Runner Design

### Overview

The batch inference pipeline (`model_pipeline/inference/batch_runner.py`) assigns thematic topics to new articles using the deployed NMF model while preserving robustness, reproducibility, and analytical interpretability.

The system retrieves unprocessed articles from a Supabase database, groups them by publication week, and submits them in controlled batches to the FastAPI prediction service. Predictions are written back to the database together with confidence metrics and run metadata, allowing inference to be resumed safely after interruptions without duplicating work.

### Why week-by-week processing

In production, the system receives a new batch of articles every week and classifies them against the pre-trained NMF model. Running the backfill week-by-week simulates that real-world cadence. This creates realistic weekly snapshots in the database so the dashboard can show topic trends over time exactly as it would in production, and enables evaluation — each week's predictions can be inspected for sensible topic distributions, stable confidence scores, and drift signals compared to training data.

### Uncertainty metric — normalised entropy

A key design goal is not only to classify articles but to quantify prediction uncertainty. The pipeline computes normalised Shannon entropy across all 30 topic weights (stored as `contestability_score`):

- **Near 0** = all weight concentrated on one topic — confident, robust assignment.
- **Near 1** = weight spread uniformly — high uncertainty, genuinely contested.

This replaced an earlier gap-based metric (`1 - (top - second)`) which was structurally broken with 30 topics — NMF spreads weight so thinly that every article scored 0.9+ regardless of actual certainty. See Section 12 (Dashboard Design Decisions) for the full rationale.

### Idempotent processing model

The pipeline follows an idempotent design in which database state determines execution progress. Articles lacking topic assignments (`dominant_topic IS NULL`) are automatically selected on each run. Successful writes remove those rows from future runs. This means:

- Reruns are safe — no duplicate inference
- Crashes recover automatically — only unfinished rows are reprocessed
- Partial success persists — if 4 of 6 weeks complete before a failure, only the remaining 2 weeks are processed on the next run

The database itself acts as the processing checkpoint.

### Resilience

The pipeline uses a `requests.Session` with automatic retries (5 attempts, exponential backoff) to handle transient network failures and Render cold-starts. API responses are validated before database writes. Failed upsert chunks are logged with affected article IDs for debugging without halting the overall batch.

---

## 15. Interpreting the Contestability Score & Responsible AI in Policy

### How to read the score

The contestability score is normalised Shannon entropy across all 30 topic weights. It ranges from 0 to 1:

| Score | Interpretation | What it means for the article |
|---|---|---|
| 0.0–0.3 | Low uncertainty | Strongly dominated by one topic. The assignment is robust — the article is clearly "about" one policy area. |
| 0.3–0.5 | Moderate uncertainty | One topic leads but others carry meaningful weight. The article touches multiple policy areas. Worth inspecting if used for policy analysis. |
| 0.5–0.7 | High uncertainty | No single topic dominates. The article genuinely spans several themes. Classification is contestable — an analyst might reasonably assign it differently. |
| 0.7–1.0 | Very high uncertainty | Weight is spread near-uniformly. The model has no strong opinion. These articles are often cross-cutting (e.g. a budget speech covering funding, SEND, teacher pay, and curriculum simultaneously) or outside the model's training vocabulary. |

### Why this matters for responsible AI use in policy

A central argument of AtlasED is that AI tools used in policy analysis should be transparent about what they don't know, not just what they predict. Most deployed classification systems present a single label with no indication of how confident or contestable that label is. A policymaker or journalist consuming those outputs has no way to distinguish between a clear-cut classification and a coin-flip.

The contestability score directly addresses this:

**1. Surfacing model uncertainty is a governance requirement, not a feature.**
When an NMF model assigns "School Funding" to an article with a contestability score of 0.15, that assignment can be trusted as input to policy analysis. When it assigns the same label with a score of 0.72, the assignment is a guess — the article equally belongs to several topics and the model's choice of one is arbitrary. Hiding this distinction from users is a form of epistemic dishonesty. Responsible AI requires making the distinction visible.

**2. High contestability is a finding, not a failure.**
Articles that score high on contestability are often the most analytically interesting — they sit at the intersection of multiple policy areas, which is exactly where political contestation happens. A funding announcement that also touches SEND, teacher pay, and curriculum is genuinely multi-topic. The model correctly identifies this ambiguity rather than forcing a false clarity. Surfacing these articles for human review is more valuable than hiding them behind a confident-looking label.

**3. Entropy quantifies what "contestable" actually means.**
The project's thesis is that policy classifications are inherently contestable — different actors frame the same policy differently depending on their institutional position. The contestability score operationalises this claim with a concrete metric. Rather than asserting in prose that "classifications are subjective", the system shows exactly how subjective each classification is, article by article, on a continuous scale.

**4. This connects to the broader governance argument.**
AtlasED argues that AI systems used in public policy should be auditable, transparent, and honest about their limitations. The contestability score is one concrete mechanism for this: it forces the system to say "I'm not sure about this one" rather than presenting every output with equal confidence. This is a design pattern that could (and should) be replicated in any AI system used for policy analysis, media monitoring, or public discourse classification.

---

## 16. Corpus Imbalance — Limitations and Retraining Considerations

### Date Range — Why 2023–2025

The 2023–2025 range was chosen to ensure the model captures the full spectrum of education policy discourse across distinct political periods, including pre- and post-election shifts around July 2024. This three-year window provides sufficient volume (~4,000 articles) for stable topic extraction via NMF while reflecting the longitudinal scope needed for comparative policy analysis. A narrower window would risk missing topic vocabulary that only surfaces during specific political moments, weakening the model's ability to label incoming 2026 articles against a representative baseline.

### The Problem
SchoolsWeek (~69% of corpus) means: (1) TF-IDF vocabulary reflects journalistic language; (2) NMF topic boundaries reflect how journalists categorise education policy; (3) articles from EPI/Nuffield are assigned topics learned from a vocabulary that doesn't match their register.

### What to Do Differently (for future retraining)

**Recommended: Organisation-type stratification.** Draw a balanced sample by org type (think_tank, gov_inst, ed_journalism, ed_res_org, prof_body) rather than per-source. Preserves more data than strict per-source balancing while reducing the journalism/non-journalism skew.

**More ambitious: Train separate models per type, then compare.** One NMF on think-tank articles, one on journalism. Compare topic structures — do journalists and researchers produce the same topics? This would be a genuinely original methodological contribution.

**For now:** Keep the current model and acknowledge it in the report:
> "The corpus is dominated by education journalism (~69% SchoolsWeek), meaning topic boundaries reflect journalistic framing more than institutional or governmental framing. This is both a methodological constraint and a finding: ed journalism plays a structurally dominant role in shaping the policy discourse captured in this dataset."

---

## 17. Examiner Q&A Preparation — Dashboard & Corpus

**Q: Why did you build a multi-page dashboard instead of a single page?**
A: Different users need different views. A single page forces all audiences to scroll through irrelevant content. Streamlit's `pages/` convention lets each analytical question have its own focused surface without code duplication — shared data loading is cached and reused across pages.

**Q: Why is there a raw/normalised toggle on your charts?**
A: SchoolsWeek accounts for ~69% of the corpus. Without normalisation, every topic frequency chart would just reflect SchoolsWeek's publication volume. The normalised view shows each source's topic distribution relative to its own output, making sources with very different article counts directly comparable.

**Q: How does your framing analysis work?**
A: Each article is assigned a framing type based on which predefined keyword list has the most matches in the cleaned article text. The keyword lists are defined in a plain Python dict at the top of the page — transparent, inspectable, and easy to adjust. An article with zero matches across all lists is labelled "unclassified"; ties are labelled "mixed".

**Q: Why did you use keyword matching rather than training a classifier for framing?**
A: A classifier requires labelled training data that doesn't exist for this domain. Keyword matching is transparent — the classification logic is human-readable and directly auditable. For a contestability analysis, transparency is more important than marginal accuracy gains from a black-box classifier.

**Q: What is the contestability score and what does it tell you?**
A: It is normalised Shannon entropy across all 30 topic weights. A score near 0 means all weight is concentrated on one topic — the model is confident. A score near 1 means weight is spread uniformly across topics — the assignment is genuinely uncertain. It surfaces which articles sit at topic boundaries and quantifies how much the classification should be trusted. An earlier gap-based metric (`1 - (top - second)`) was replaced because it was structurally broken with 30 topics — see Section 12 for the full rationale.

**Q: Your corpus is dominated by SchoolsWeek — how does that affect your findings?**
A: It means the NMF vocabulary and topic structure reflect journalistic framing of education policy more than institutional or governmental framing. I address this in three ways: (1) normalised chart views so SchoolsWeek volume doesn't dominate visually; (2) the framing analysis which explicitly compares how different organisation types frame the same topic; (3) acknowledging it as a finding — ed journalism's structural dominance in this corpus is itself a substantive result about agenda-setting power in education policy discourse.

**Q: What would you do differently if you retrained the model?**
A: Use organisation-type stratified sampling before training — drawing a balanced sample across think tanks, government bodies, journalism, and research organisations. This would reduce vocabulary bias toward journalistic language and produce topics that better reflect cross-institutional framing. A more ambitious extension would be training separate NMF models per organisation type and comparing the resulting topic structures.

**Q: Why is the model monitoring not on the dashboard?**
A: The dashboard's target audience — researchers and policy analysts — has no use for reconstruction error or coherence scores. Model quality is documented in `design_decisions.md` and `run_metadata.json`. Putting it on the dashboard would add noise for the intended users without adding analytical value.

---

## 18. Examiner Q&A Preparation — Model & Algorithm

### Model & Algorithm

**Q: Why did you choose NMF over LDA?**
A: NMF produces parts-based, additive topic representations that are directly interpretable — each word either contributes to a topic or it doesn't. On this corpus of policy texts, NMF consistently produced more coherent and clearly separable topics than LDA. LDA's Dirichlet prior assumptions also tend to produce more diffuse topics on shorter documents.

**Q: What is reconstruction error and what does it tell you?**
A: Reconstruction error is `||X - WH||_F` — the Frobenius norm of the difference between the original TF-IDF matrix and its NMF approximation. Lower is better. It tells you how well the low-rank factorisation captures the original data. However, lower reconstruction error doesn't always mean better topics — very high n_components would reduce reconstruction error while producing fragmented, uninterpretable topics. That's why coherence is used alongside it.

**Q: What is c_v coherence and why did you use it?**
A: C_v coherence measures how often the top words of a topic co-occur in a sliding window across the corpus. Higher coherence indicates that the top words of a topic are semantically related and co-occur frequently, which correlates with human interpretability. It is computed using gensim's CoherenceModel on the tokenised corpus.

**Q: What is topic stability and why does it matter?**
A: Stability measures how reproducible topics are across different random seeds. For each pair of NMF runs, we compute the cosine similarity between their H matrices (topic-word matrices) and take the mean of the best-matching similarities. A stability score close to 1.0 means the same topics emerge regardless of random initialisation — the solution is robust, not a local optimum artefact.

**Q: What does `nndsvd` initialisation do?**
A: It initialises the W and H factor matrices using a deterministic SVD decomposition of the input matrix rather than random values. This provides a better starting point for the NMF optimisation, leading to faster convergence and more reproducible results compared to random initialisation.

**Q: What is the W matrix and what is the H matrix?**
A: W is the document-topic matrix (shape: n_documents × n_topics). Each row is a document, each column is a topic, and the values represent how strongly that topic is expressed in that document. H is the topic-word matrix (shape: n_topics × n_vocabulary). Each row is a topic and each column is a word — the values represent each word's contribution to that topic. The top words in each H row define what the topic is "about."

**Q: How did you choose n_topics = 30?**
A: Through a coherence sweep — I trained NMF for n_topics in range(5, 80, 5) and measured c_v coherence at each point. 30 topics was where coherence peaked before declining. I also verified this with topic stability analysis — the 30-topic solution is highly stable across seeds. Finally, qualitative review of the topic word lists confirmed that 30 topics captures the breadth of education policy discourse without excessive fragmentation.

**Q: What are the limitations of NMF topic modelling?**
A: Several acknowledged limitations: (1) NMF cannot capture polysemy — a word like "funding" means the same thing regardless of context. (2) Topics are hard boundaries — each document gets one dominant topic, but real articles often span multiple topics. (3) The model is static — it was trained on a historical corpus and will not adapt to new terminology without retraining. (4) Topic quality depends heavily on preprocessing decisions — different stopword lists or POS filters would produce different topics.

---

### Preprocessing

**Q: Why did you remove PERSON entities in spaCy processing?**
A: Named individuals would create spurious topics around politicians or academics rather than policy areas. A topic dominated by a person's name would not generalise to new articles about the same policy area if the person changes.

**Q: Why did you use bigrams in TF-IDF?**
A: Domain-specific phrases like "school funding", "teacher pay", "free school meals" are meaningful as units. Splitting them into unigrams loses the semantic specificity — "free" and "school" individually are much less informative than "free school meals" as a phrase.

**Q: Why `max_df=0.85`?**
A: Terms appearing in more than 85% of documents carry little discriminative power — they behave like corpus-specific stopwords. For example, "education" appears in virtually every document in this corpus and would not help distinguish topics.

---

### Architecture & Engineering

**Q: Why did you structure the pipeline as numbered steps?**
A: Single responsibility principle — each step does one thing and can be run, tested, and debugged independently. It also means training and inference share the same preprocessing code (the API imports `run_cleaning` and `run_spacy_processing` directly) rather than duplicating logic.

**Q: How does the API ensure training and inference use the same preprocessing?**
A: The API (`main.py`) directly imports `run_cleaning` from `s02_cleaning.py` and `run_spacy_processing` from `s03_spacy_processing.py` — the exact same functions used during training. The fitted vectorizer (saved as `vectorizer.joblib`) is then called with `.transform()`, not `.fit_transform()`, ensuring the same vocabulary and IDF weights from training are applied to new documents.

**Q: Why is `.transform()` used in inference instead of `.fit_transform()`?**
A: `fit_transform()` would refit the vocabulary on the new documents, producing a completely different feature space. The NMF model was trained on a specific TF-IDF vocabulary — inference must use that exact same vocabulary and IDF weights, which is what `.transform()` does.

**Q: Why are run artifacts saved in timestamped folders?**
A: To ensure no run overwrites another. Each run folder contains the full model state (vectorizer, NMF model, topic names, metadata) needed to reproduce any prediction. This is standard MLOps practice — you should always be able to roll back to a previous model version.

**Q: What is `run_metadata.json` for?**
A: It records the full configuration and statistics of each training run — n_topics, TF-IDF parameters, random_state, reconstruction error, dominant topic weight statistics. This creates a complete audit trail so any prediction can be traced back to the exact model and parameters that produced it.

**Q: How does the API load the correct model?**
A: `model_loader.py` scans `experiments/outputs/runs/`, sorts the folders by name (which are timestamps in `YYYY-MM-DD_HHMMSS` format, so alphabetical order = chronological order), and loads the most recent one. Models are cached in a singleton after first load.

**Q: Why FastAPI over Flask?**
A: FastAPI generates automatic interactive API documentation at `/docs`, enforces request and response schemas through Pydantic models (catching malformed inputs before they reach the model), and is async-native. Flask would require additional libraries (marshmallow, flasgger) to achieve the same.

**Q: Why Supabase instead of a local database?**
A: A local database is not a deployed system — it exists only on one machine and cannot be accessed by external services (scraper, dashboard). Supabase is a hosted PostgreSQL instance that is accessible from anywhere, demonstrates genuine cloud deployment, and provides a table viewer for auditing data without writing SQL queries.

---

### How to Work Through and Understand the Code

Follow this order to learn the codebase from the ground up:

1. **Read `config.yaml`** — understand all tunable parameters before touching code.
2. **Read `s01_data_loader.py`** — understand what raw data looks like and what columns are expected.
3. **Run `s01` smoke test** (`python -m model_pipeline.training.s01_data_loader`) — confirm the data loads.
4. **Read `s02_cleaning.py`** — understand what boilerplate is removed and why. Trace `clean_scraped_article()` step by step.
5. **Read `s03_spacy_processing.py`** — understand `spacy_clean()`: which POS tags are kept, why PERSON entities are removed, what lemmatisation does.
6. **Read `s04_vectorisation.py`** — understand each TF-IDF parameter. Run the smoke test and inspect the output shape and sample features.
7. **Read `s05_nmf_training.py`** — understand W and H matrices. Look up what `reconstruction_err_` means in the sklearn NMF docs.
8. **Read `s06_topic_allocation.py`** — understand how `W.argmax(axis=1)` gives the dominant topic per document.
9. **Read `s07_evaluation.py`** — understand coherence sweep and stability analysis. Look at the CSVs in `data/evaluation_outputs/`.
10. **Read `s08_save_outputs.py`** — understand the run folder structure and what each artifact contains.
11. **Read `s10_pipeline.py`** — this is the full end-to-end view. After reading individual steps, this should read clearly.
12. **Read `api/model_loader.py` and `api/main.py`** — understand how training artifacts are loaded and served.
13. **Read `dashboard/app.py`** — understand how the data is queried and visualised.

For each step, be able to answer:
- What does this step receive as input?
- What does it produce as output?
- What would break if I removed or changed it?
