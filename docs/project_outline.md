# Project Outline — AtlasED: Education Policy Observatory

## Overview

AtlasED is an open-source education policy intelligence tool built for the UCL Grand Challenges programme. It automatically tracks, classifies, and visualises discourse across UK education policy publications, with the goal of making policy trends accessible to researchers, educators, and policymakers.

The project is being delivered as part of a Data & AI Level 7 apprenticeship at UCL.

---

## Goals

- Track how education policy topics evolve week-by-week across major UK publications
- Surface contested and emerging themes in policy discourse
- Provide a public-facing, interactive dashboard for exploration
- Demonstrate a production-grade ML pipeline as an apprenticeship artefact

---

## Scope (This Repo — Topic Modelling)

This repository covers the topic modelling pipeline, inference API, and analytical dashboard. Sentiment analysis and the AtlasED public website are handled in separate repositories.

---

## Data

- **Source:** SchoolsWeek (UK education policy publication)
- **Corpus:** ~3,972 articles, January 2023 – December 2025 (retrospective training set)
- **Inference:** Weekly article batches, January 2019 → ongoing
- **Storage:** Supabase (PostgreSQL), `articles` table

### Key Fields in Supabase
| Field | Description |
|---|---|
| `id` | UUID primary key |
| `url` | Article URL (used for deduplication and pipeline matching) |
| `dataset_type` | `"training"` or `"inference"` |
| `topic_num` | Assigned topic index (0–29) |
| `dominant_topic` | Human-readable topic label |
| `dominant_topic_weight` | Weight of the dominant topic |
| `topic_probabilities` | JSONB — full 30-topic probability distribution |
| `contestability_score` | `1 - (top_weight - second_weight)` — how contested the topic is |
| `election_period` | `"pre_election"` / `"post_election"` relative to 4 July 2024 |
| `run_id` | Timestamp of the training run that produced this assignment |

---

## Architecture

```
SchoolsWeek articles
        │
        ▼
 Scraping pipeline          ← separate repo
        │
        ▼
   Supabase (articles table)
        │
  ┌─────┴────────────────────────────────┐
  │                                      │
  ▼                                      ▼
Training pipeline                 Inference pipeline
(s01–s11, this repo)             (batch_runner.py, this repo)
  │                                      │
  │  NMF model + vectorizer              │  Calls deployed FastAPI
  │  → Supabase (training rows)          │  → Supabase (inference rows)
  │                                      │
  └─────────────┬────────────────────────┘
                │
                ▼
        Streamlit dashboard
        (reads from Supabase)
                │
                ▼
        Public URL (Streamlit Community Cloud)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Topic model | NMF (scikit-learn) with TF-IDF vectorisation |
| NLP preprocessing | spaCy (`en_core_web_sm`) |
| Inference API | FastAPI + Pydantic, served via Uvicorn |
| Data store | Supabase (PostgreSQL) |
| Experiment tracking | MLflow (local) |
| Dashboard | Streamlit + Altair |
| Deployment (API) | Docker → Railway / Fly.io |
| Deployment (dashboard) | Streamlit Community Cloud |
| Environment | python-dotenv, venv |

---

## Model

- **Algorithm:** Non-negative Matrix Factorisation (NMF)
- **Topics:** 30
- **Vocabulary:** TF-IDF, up to 10,000 features
- **Training corpus:** 3,972 articles
- **Reconstruction error:** 53.337090 (run `2026-02-27_215011`)
- **Topic names:** Defined in `s06_topic_allocation.py` (`TOPIC_NAMES` dict)
- **Artefacts stored:** `vectorizer.joblib`, `nmf_model.joblib`, `topic_names.json` under `experiments/runs/<run_id>/`

---

## Pipeline Steps

| Step | File | Description |
|---|---|---|
| S01 | `s01_data_loader.py` | Load articles from Supabase or local CSV |
| S02 | `s02_cleaning.py` | Text cleaning (lowercase, punctuation, URLs) |
| S03 | `s03_spacy_processing.py` | Lemmatisation, POS filtering, NER via spaCy |
| S04 | `s04_vectorisation.py` | TF-IDF vectorisation |
| S05 | `s05_nmf_training.py` | Train NMF model |
| S06 | `s06_topic_allocation.py` | Assign topics, compute weights, export CSV |
| S07 | `s07_evaluation.py` | Coherence and stability evaluation |
| S08 | `s08_save_outputs.py` | Persist model artefacts and evaluation CSVs |
| S09 | `s09_mlflow_logging.py` | Log run metrics to MLflow |
| S10 | `s10_pipeline.py` | End-to-end runner (S01–S09 + S11) |
| S11 | `s11_supabase_writer.py` | Upsert training topic assignments to Supabase |

---

## Inference API

- **Endpoint:** `POST /predict` — accepts a list of articles, returns topic assignments
- **Endpoint:** `GET /health` — returns model run_id, n_topics, status
- **Model loading:** Singleton via `model_loader.py`, loaded at startup from most recent run folder
- **Deployment:** Docker container → Railway or Fly.io (pending)

---

## Dashboard (Streamlit)

5-page dashboard, currently reads from CSV. To be migrated to Supabase.

| Page | Description |
|---|---|
| Overview | Topic frequency and trend over time |
| Topic Explorer | Drill into individual topics |
| Contestability | Which topics are most contested |
| Election Analysis | Pre/post July 2024 topic shift |
| Source Comparison | Topic distribution by publication |

---

## Current Status (as of February 2026)

- [x] Training pipeline (S01–S11) complete
- [x] 3,972 training articles with topic assignments in Supabase
- [x] FastAPI inference API built and tested locally
- [x] MLflow experiment tracking working
- [ ] FastAPI deployed (blocked — Dockerfile not written yet)
- [ ] Inference batch runner written
- [ ] Inference backfill run
- [ ] Dashboard migrated to Supabase
- [ ] Dashboard deployed publicly
- [ ] ISD cloud approval obtained
- [ ] IP ownership clarified

---

## Governance & Compliance

See `docs/todo.md` for outstanding actions. Key points:

- **Scraping:** Covered by UK TDM exception (CDPA 1988, s29A) for non-commercial research
- **Data:** Non-personal — education policy text only, no PII
- **Ethics:** Application submitted
- **IP:** Needs urgent clarification — staff vs student IP rules differ at UCL
- **Cloud services:** ISD approval pending for Supabase, Railway, Streamlit Community Cloud

---

## Repo Structure

```
AM1_topic_modelling/
├── model_pipeline/
│   ├── training/          # S01–S11 training pipeline
│   ├── api/               # FastAPI inference API
│   └── inference/         # batch_runner.py (to be written)
├── dashboard/             # Streamlit dashboard
├── data/                  # Local data (gitignored)
├── experiments/
│   ├── runs/              # MLflow + model artefacts per run
│   └── supabase_scheme/   # Schema reference
├── docs/                  # Design decisions, todo, project outline
├── config.yaml            # Pipeline hyperparameters
└── requirements.txt
```
