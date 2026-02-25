# Education Policy Observatory

> **Tracking who shapes the education policy conversation — and how.**

An open-source AI platform that analyses education policy discourse across England, Scotland, and Ireland. It surfaces agenda-setting patterns, institutional framing biases, and topic contestability, making the mechanisms of policy influence visible and interrogable.

---

## Why This Exists

Public education policy is shaped not just by governments, but by the organisations that frame and amplify certain issues over others. Journalists, think tanks, government bodies, and research organisations all speak about the same topics — but with different language, emphasis, and framing.

This observatory makes those differences visible. It asks:

- **Who** publishes on which topics — and who is absent?
- **How** do different organisations frame the same issue?
- **When** did the agenda shift — and did the 2024 election change what gets discussed?
- **How certain** is the model about its own classifications, and where are assignments contestable?

---

## Features

### Topic Modelling Pipeline
- Non-negative Matrix Factorisation (NMF) trained on 3,900+ education policy articles
- 30 coherence-validated topics covering curriculum, funding, SEND, teacher pay, safeguarding, and more
- Full preprocessing pipeline: structural cleaning → spaCy lemmatisation → TF-IDF vectorisation
- Versioned run artifacts with full audit trail (`run_metadata.json`)
- Coherence sweep and topic stability evaluation across random seeds

### Education Policy Observatory Dashboard
A multi-page Streamlit application with five analytical views:

| Page | What it shows |
|---|---|
| **Overview** | Corpus summary, top topics, source breakdown |
| **Topic Explorer** | Top keywords per topic, article browser, contestability scores |
| **Trends Over Time** | Monthly topic attention with election shift analysis |
| **Organisation Analysis** | Source × topic heatmap, org-type comparisons |
| **Framing Analysis** | Keyword-based framing classification by organisation, time, and country |

All charts offer a **normalised view** to account for source imbalance (ed journalism dominates volume — this is surfaced as a finding, not hidden).

### Topic Contestability Score
Each article receives a contestability score derived from the full document-topic weight matrix:

```
contestability_score = 1 - (dominant_topic_weight - second_highest_topic_weight)
```

Score near 1 → the model was uncertain between two topics → the classification is contestable.
Score near 0 → the model was confident → robust assignment.

This makes the model's own uncertainty transparent and allows users to interrogate borderline classifications.

### Framing Analysis
Articles are classified by framing type (economic, rights-based, crisis, evidence-based, political) using transparent keyword matching. Framing patterns are compared across organisations, election periods, and over time.

### REST API
FastAPI serving layer for inference on new articles:

```bash
POST /predict
# Input:  {"articles": [{"article_id": "1", "text": "..."}]}
# Output: topic_id, topic_name, confidence, all_weights (30 topics)

GET /health
# Returns: run_id, n_topics, status
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Topic modelling | scikit-learn NMF + TF-IDF |
| NLP preprocessing | spaCy (`en_core_web_sm`) |
| Sentiment analysis | Transformers (HuggingFace) *(in development)* |
| API | FastAPI + Pydantic |
| Dashboard | Streamlit + Altair |
| Database | Supabase (PostgreSQL) |
| Experiment tracking | MLflow |
| Model serialisation | joblib |

---

## Project Structure

```
AM1_topic_modelling/
├── config.yaml                          # All tunable parameters
├── data/
│   ├── full_retro/                      # Training corpus
│   └── evaluation_outputs/             # Coherence + stability CSVs, dashboard data
├── model_pipeline/
│   ├── training/
│   │   ├── s01_data_loader.py
│   │   ├── s02_cleaning.py
│   │   ├── s03_spacy_processing.py
│   │   ├── s04_vectorisation.py
│   │   ├── s05_nmf_training.py
│   │   ├── s06_topic_allocation.py
│   │   ├── s07_evaluation.py
│   │   ├── s08_save_outputs.py
│   │   ├── s09_mlflow_logging.py
│   │   └── s10_pipeline.py              # Full end-to-end run
│   ├── api/
│   │   ├── model_loader.py
│   │   └── main.py
│   └── dashboard/
│       ├── app.py                       # Overview landing page
│       └── pages/
│           ├── 1_Topic_Explorer.py
│           ├── 2_Trends.py
│           ├── 3_Organisations.py
│           └── 4_Framing_Analysis.py
├── experiments/
│   └── outputs/runs/                    # Versioned model artifacts
└── docs/
    └── design_decisions.md              # Full architecture rationale + examiner Q&A
```

---

## Getting Started

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the pipeline

```bash
python -m model_pipeline.training.s10_pipeline
```

Trained artifacts are saved to `experiments/outputs/runs/<timestamp>/`.

### 3. Launch the dashboard

```bash
streamlit run model_pipeline/dashboard/app.py
```

### 4. Start the API

```bash
uvicorn model_pipeline.api.main:app --reload
```

API docs available at `http://localhost:8000/docs`.

---

## Configuration

All tunable parameters live in [`config.yaml`](config.yaml):

```yaml
data:
  dataset_name: full_retro

nmf:
  n_topics: 30
  random_state: 42
  init: nndsvd
  max_iter: 1000

tfidf:
  min_df: 3
  max_df: 0.85
  max_features: 3000
  ngram_range: [1, 2]
```

Change `dataset_name` to point at a different corpus without touching any source code.

---

## Coverage

| Nation | Status |
|---|---|
| England | ✅ Live |
| Scotland | 🔜 In development |
| Ireland | 🔜 In development |

Sources currently included: SchoolsWeek, Education Policy Institute (EPI), FFT Education Datalab, Nuffield Foundation, UK Government, and others.

---

## Design Decisions

Full rationale for every architectural and methodological decision — including trade-offs, limitations, and answers to common questions about the modelling approach — is documented in [`docs/design_decisions.md`](docs/design_decisions.md).

Key decisions covered:
- Why NMF over LDA and BERTopic
- Why TF-IDF over embeddings
- How the number of topics (30) was chosen
- Why the source imbalance is surfaced rather than corrected
- Why framing uses keyword matching rather than a trained classifier
- What the contestability score measures and why it matters

---

## Roadmap

- [ ] Scotland and Ireland data ingestion
- [ ] Transformer-based sentiment analysis layer
- [ ] Conversational interface (chatbot) for natural language exploration
- [ ] Discourse concentration score (per-topic source dominance index)
- [ ] Framing diversity shift analysis (pre/post election)
- [ ] Balanced corpus retraining (organisation-type stratified sampling)
- [ ] Weekly automated pipeline via scheduled jobs

---

## Contributing

Contributions welcome — particularly on:
- Scotland and Ireland data sources and scraping
- Framing keyword list refinement
- Additional organisation sources
- Sentiment analysis model selection

Please open an issue before submitting a pull request.

---

## Licence

[MIT](LICENSE)

---

## Acknowledgements

Built as part of research into AI-assisted discourse analysis for democratic accountability in education policy.
