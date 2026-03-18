# AtlasED

> **Specification-aware cross-jurisdictional education policy discourse analysis.**

AtlasED is a production NLP pipeline that analyses education policy discourse across England, Scotland, and Ireland. It does two things:

1. **Surfaces how different institutional actors frame educational disadvantage** — tracking which organisations dominate the debate, how topic prominence varies across jurisdictions, and whether those patterns diverge systematically.

2. **Makes the specification choices behind the analysis visible and auditable** — surfacing how data selection, model selection, parameterisation, and preprocessing decisions shape what the analysis finds, with the goal of informing guidelines on how AI analysis should be used responsibly in public policy.

The pipeline trains on England's policy corpus as a baseline. Scottish and Irish corpora are passed through the England-trained model as out-of-distribution inference. The resulting distributional drift is not a bug — it is the finding: England's construct definitions act as the implicit norm, and the drift scores are a live, computable measure of that normative divergence.

---

## Research Questions

- How do different institutional actors across England, Scotland, and Ireland operationalise the construct of educational disadvantage — and do those operationalisations diverge systematically?
- When a model trained on England's policy corpus is applied to Scottish and Irish corpora, is the resulting distributional drift directional and persistent?
- Do specification choices — construct definitions, preprocessing decisions, model parameters — determine findings independently of the underlying policy landscape?

---

## Data

**5,885 documents** from government departments, think tanks, media outlets, professional bodies, and research organisations across three jurisdictions. Updated weekly.

| Jurisdiction | Training (2023–2025) | Inference (2023–2025) | Weekly inference (2026) |
|---|---|---|---|
| **England** | 3,943 articles | — | 219 (weeks 1–10) |
| **Scotland** | — | 511 backfill | 111 (weeks 1–10) |
| **Ireland** | — | 1,036 backfill | 65 (weeks 1–10) |

**England sources:** SchoolsWeek, UK Government, FFT Education Datalab, Education Policy Institute, Nuffield Foundation, Federation of Education.
**Scotland sources:** Scottish Government, Children in Scotland, GTCS, ADES, SERA.
**Ireland sources:** Irish Government, ESRI, Teaching Council, Education Research Centre, Education Matters, RTE.

Country is derived from the source organisation, not hardcoded.

Storage: Supabase/PostgreSQL. Raw text in `articles_raw`, topic assignments in `articles_topics`.

---

## Architecture

### Training (England baseline)
NMF and BERTopic models trained on the England corpus. The England model defines the topic space — this is a deliberate specification choice, not a default.

### Inference (cross-jurisdiction)
Scotland and Ireland corpora are passed through the England-trained model. Each document receives topic assignments relative to England's topic structure. The backfill (2023–2025) provides the cross-jurisdiction baseline; weekly runs (2026 onwards) track change over time.

### Drift detection
Jensen-Shannon divergence computed per jurisdiction per week against the England training distribution. Per-jurisdiction drift trajectories are logged as distinct time series.

### Cross-jurisdiction distributional analysis
KL divergence computed in both directions for all jurisdiction pairs. Directional asymmetry tested: does one jurisdiction systematically act as the implicit baseline? Robustness checks: balanced subsamples, parameter perturbation (k varied), model swap (NMF vs BERTopic).

### Specification scoring layer
Three computable dimensions that make specification choices visible:

- **Proxy concentration** — how far a small number of construct proxies account for observed topic variance.
- **Specification sensitivity** — how far findings shift under perturbation of modelling decisions.
- **Normative divergence** — whether one jurisdiction's corpus systematically acts as the implicit baseline.

A fourth dimension (recourse quality) was designed, tested, and scoped out of the current implementation. Inter-rater agreement was unreliable at this stage.

---

## Project Structure

```
AM1_topic_modelling/
├── config.yaml                              # All tunable parameters
├── data/
│   ├── full_retro/                          # Local CSV fallback
│   └── evaluation_outputs/                  # Coherence + stability CSVs
├── model_pipeline/
│   ├── training/                            # Model training (England)
│   │   ├── s01_data_loader.py               #   Load from Supabase or CSV
│   │   ├── s02_cleaning.py                  #   Structural text cleaning
│   │   ├── s03_spacy_processing.py          #   Lemmatisation, stopwords
│   │   ├── s04_vectorisation.py             #   TF-IDF
│   │   ├── s05_nmf_training.py              #   NMF model fitting
│   │   ├── s05b_bertopic_training.py        #   BERTopic model fitting
│   │   ├── s06_topic_allocation.py          #   Assign topics to documents
│   │   ├── s06b_topic_naming.py             #   LLM-assisted topic naming
│   │   ├── s07_evaluation.py                #   Coherence + stability
│   │   ├── s08_save_outputs.py              #   Versioned run artifacts
│   │   ├── s09_mlflow_logging.py            #   Experiment tracking
│   │   ├── s10_pipeline.py                  #   NMF orchestrator
│   │   ├── s10b_bertopic_pipeline.py        #   BERTopic orchestrator
│   │   └── s11_supabase_writer.py           #   Write results to DB
│   ├── inference/                           # Weekly + backfill inference
│   │   ├── batch_runner.py                  #   NMF inference (all countries)
│   │   ├── bertopic_batch_runner.py         #   BERTopic inference
│   │   └── drift_monitor.py                 #   Per-jurisdiction drift
│   ├── analysis/                            # Cross-jurisdiction analysis
│   │   ├── kl_divergence.py                 #   KL/JS divergence
│   │   ├── robustness.py                    #   Balanced subsamples, k perturbation
│   │   ├── specification_scores.py          #   Three computable dimensions
│   │   └── vocabulary_audit.py              #   spaCy cross-jurisdiction check
│   ├── api/                                 # FastAPI serving layer
│   │   ├── main.py
│   │   └── model_loader.py
│   └── dashboard/                           # Streamlit application
│       ├── app.py                           #   Overview + headline findings
│       ├── supabase_loader.py               #   Data loading + caching
│       ├── rag_agent.py                     #   LangGraph chatbot agent
│       └── pages/
│           ├── 1_Topic_Explorer.py           #   Browse topics by jurisdiction
│           ├── 2_Trends.py                   #   Drift + topic prominence over time
│           ├── 3_Organisations.py            #   Source × topic analysis
│           ├── 4_Framing_Analysis.py         #   Keyword framing by org/time
│           ├── 5_Build_Your_Model.py         #   Interactive specification explorer
│           └── 6_Ask_AtlasED.py              #   RAG chatbot interface
├── experiments/
│   ├── outputs/runs/                        # Versioned model artifacts
│   ├── notebooks/                           # Exploratory analysis
│   └── mlruns/                              # MLflow experiment store
└── docs/
    ├── march_working_plan.md                # Implementation plan
    ├── design_decisions.md                  # Architecture rationale
    ├── project_outline.md                   # Project specification
    └── ethics_and_data_security.md
```

Files in **bold structure** exist. Files in regular text are planned (see [implementation plan](docs/march_working_plan.md)).

---

## Tech Stack

| Component | Technology |
|---|---|
| Topic modelling | scikit-learn NMF, BERTopic |
| NLP preprocessing | spaCy (`en_core_web_sm`) |
| Vectorisation | TF-IDF (NMF), sentence-transformers (BERTopic) |
| API | FastAPI + Pydantic |
| Dashboard | Streamlit + Altair |
| Database | Supabase (PostgreSQL) |
| Experiment tracking | MLflow |
| Agent orchestration | LangGraph |
| LLM observability | Langfuse |
| Deployment | Docker, Render |

---

## Specification Choices as First-Class Outputs

Every modelling decision in this pipeline is a specification choice. The following are logged, surfaced on the dashboard, and tested under perturbation:

| Choice | Current setting | Why it matters |
|---|---|---|
| Training corpus | England only | Defines the topic space. Scotland/Ireland measured as deviation. |
| Preprocessing | spaCy `en_core_web_sm` | English-language model applied to Scottish/Irish policy text. |
| Model | NMF (baseline) + BERTopic (comparison) | Different models surface different topic structures. |
| Number of topics (k) | 30 | Varied 5–30 in robustness checks. |
| TF-IDF parameters | min_df=3, max_df=0.85, max_features=3000 | Controls what vocabulary enters the model. |
| Topic naming | LLM-generated from top keywords | Reproducible but not neutral. |
| Source selection | 6 eng, 5 sco, 6 irl organisations | Who is in the corpus determines what the model finds. |
| Date range | Jan 2023 – present | A political choice, not a neutral setting. |

---

## Getting Started

### Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Train the model

```bash
python -m model_pipeline.training.s10_pipeline
```

### Run inference

```bash
python -m model_pipeline.inference.batch_runner
```

### Launch the dashboard

```bash
streamlit run model_pipeline/dashboard/app.py
```

### Start the API

```bash
uvicorn model_pipeline.api.main:app --reload
```

---

## Configuration

All parameters in [`config.yaml`](config.yaml):

```yaml
data:
  dataset_name: full_retro
  source: supabase          # supabase or csv
  country: eng              # training country

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

---

## Theoretical Foundations

The specification scoring layer is grounded in three independent theoretical traditions:

**Proxy concentration** derives from the construct validity literature. Jacobs and Wallach (2021) argue that fairness is a property of measurement models, not algorithms — the question of what a system is actually measuring is upstream of the question of whether it measures it fairly.

**Specification sensitivity** operationalises the replication crisis argument. Botvinik-Nezer et al. (Nature, 2020) showed that seventy independent teams given identical fMRI data reached different conclusions because of unlogged pipeline choices.

**Normative divergence** derives from the situated knowledge tradition. D'Ignazio and Klein's *Data Feminism* (2020) operationalises the argument that data systems encode the perspectives of their designers. Normative divergence makes that encoding computable.

---

## Scope and Limitations

**In scope:** Text-based policy corpora. Public artefacts. Design-time specification choices.

**Out of scope:** Image-based systems. Real-time decision pipelines without document corpora. Runtime specification logging for agentic systems. Recourse quality as a computable metric (scoped out of v1).

**Known limitations:**
- England dominates the corpus by volume. This is surfaced as a finding, not corrected.
- spaCy `en_core_web_sm` is an English-language model applied to all three jurisdictions. The vocabulary divergence is audited and reported.
- Topic labels are a specification choice. They are generated by LLM from top keywords, not ground truth.
- The pipeline finds patterns in language. It cannot determine whether those patterns reflect policy reality or corpus construction. Both possibilities are surfaced.

---

## Licence

[MIT](LICENSE)

---



*UCL Institute of Education | Level 6 AI Engineering Apprenticeship | 2025–2026*
