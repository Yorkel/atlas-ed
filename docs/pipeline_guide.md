# AtlasED Topic Modelling Pipeline

End-to-end guide for training and inference across England, Scotland, and Ireland.

## Overview

The pipeline has four stages:

```
Supabase (articles_raw)
    │
    ▼
1. SYNC ─── pull data to local CSVs
    │
    ▼
2. TRAIN ── experiment in notebooks, save NMF models
    │
    ▼
3. RUN ──── push data through models, write topics back to Supabase
    │
    ▼
4. DOWNLOAD ── pull processed data for analysis
```

Each country (eng, sco, irl) has its own NMF model trained on its own corpus. The pipeline logic is identical for all three — only the data and model differ.

---

## 1. Sync data from Supabase

Pulls raw articles from the `articles_raw` table and saves them as local CSVs.

```bash
# Sync everything (training + weekly)
python sync_from_supabase.py

# Or sync selectively
python sync_from_supabase.py --training   # training data only
python sync_from_supabase.py --weekly     # weekly inference data only
```

**Output:**

| File | Source | Content |
|------|--------|---------|
| `data/training/eng_training.csv` | `dataset_type='training', country='eng'` | England 2023-2025 |
| `data/training/sco_training.csv` | `dataset_type='training', country='sco'` | Scotland 2023-2025 |
| `data/training/irl_training.csv` | `dataset_type='training', country='irl'` | Ireland 2023-2025 |
| `data/inference/weekly/eng_week_1.csv` ... | `dataset_type='inference', week_number=N` | Weekly articles per country |

---

## 2. Train models (notebooks)

Each country has a training notebook in `experiments/notebooks/`:

| Notebook | Country | Topics |
|----------|---------|--------|
| `train_nmf_england_v2_3943.ipynb` | England | 30 |
| `train_scotland_v1.ipynb` | Scotland | 15 |
| `train_ireland_v1.ipynb` | Ireland | 15 |

The notebooks:
1. Load the training CSV (`data/training/{country}_training.csv`)
2. Run preprocessing (s02 cleaning + s03 spaCy)
3. Vectorise with TF-IDF (s04)
4. Train NMF model (s05)
5. Review and name topics
6. Save model artifacts to `experiments/outputs/runs/{country}_{timestamp}/`
   - `nmf_model.joblib`
   - `vectorizer.joblib`
   - `topic_names.json`
   - `metadata.json`

After training, update `config.yaml` with the new model run ID:

```yaml
countries:
  eng:
    model_run: eng_2026-03-23_012157
  sco:
    model_run: sco_2026-03-23_013131
  irl:
    model_run: irl_2026-03-23_013238
```

---

## 3. Run data through the pipeline

The batch runner loads synced CSVs, runs them through the trained models, allocates topics, and pushes results to Supabase (`articles_topics` table).

**The same logic runs for both training and inference data:**
CSV → preprocess → model → allocate topics → push to Supabase

```bash
# Training data (all countries)
python -m model_pipeline.inference.batch_runner --mode training_all

# Training data (single country)
python -m model_pipeline.inference.batch_runner --mode training_eng
python -m model_pipeline.inference.batch_runner --mode training_sco
python -m model_pipeline.inference.batch_runner --mode training_irl

# Weekly inference (all countries)
python -m model_pipeline.inference.batch_runner --mode inference_weekly

# Everything (training + inference)
python -m model_pipeline.inference.batch_runner --mode all_training_inference
```

**What gets written to Supabase per article:**

| Column | Description |
|--------|-------------|
| `dominant_topic` | Named topic (e.g. `child_welfare_protection`) |
| `topic_num` | Topic index (0-29 for eng, 0-14 for sco/irl) |
| `dominant_topic_weight` | Strength of assignment |
| `topic_probabilities` | JSONB with all topic weights |
| `contestability_score` | 0 = certain, 1 = uncertain (normalised Shannon entropy) |
| `election_period` | `pre_election` or `post_election` (UK 2024-07-04) |
| `dataset_type` | `training` or `inference` |
| `week_number` | Week number for inference data, NULL for training |

Writes are **upserts** matched by article `id` — safe to re-run.

---

## 4. Download processed data

Pull all processed results from `articles_topics` into a single CSV for analysis:

```bash
python sync_from_supabase.py --download
```

**Output:** `data/processed_data_YYYY-MM-DD.csv`

---

## Weekly automation

The weekly pipeline (`run_weekly.py`) is designed to run as a GitHub Action:

```bash
python run_weekly.py
```

This runs:
1. `sync_from_supabase.py` — pull new weekly articles
2. `batch_runner --mode inference_weekly` — process through models and push to Supabase

---

## Drift monitoring

Monthly drift monitoring compares each country/period's topic distribution against the England training baseline using Jensen-Shannon divergence.

```bash
# Full drift (training + weekly)
python -m model_pipeline.inference.drift_monitor

# Training data drift only
python -m model_pipeline.inference.drift_monitor --training-only
```

Results are written to the `drift_metrics` table in Supabase.

---

## Pipeline modules

| Module | Step | Purpose |
|--------|------|---------|
| `s01_data_loader.py` | 01 | Load CSVs, combine title + text, drop PDFs |
| `s02_cleaning.py` | 02 | Text cleaning, drop short articles |
| `s03_spacy_processing.py` | 03 | spaCy tokenisation and lemmatisation |
| `s04_vectorisation.py` | 04 | TF-IDF vectorisation |
| `s05_nmf_training.py` | 05 | Train NMF model |
| `s06_topic_allocation.py` | 06 | Assign topics using trained model |
| `s08_save_outputs.py` | 08 | Save model artifacts |
| `s11_supabase_writer.py` | 11 | Write results to Supabase |

---

## Key files

```
sync_from_supabase.py              # Sync raw data / download processed data
config.yaml                        # Model run IDs and hyperparameters
model_pipeline/inference/
  batch_runner.py                  # Main pipeline runner (training + inference)
  drift_monitor.py                 # Jensen-Shannon drift monitoring
model_pipeline/training/
  s02-s06, s08, s11                # Pipeline steps
experiments/notebooks/
  train_nmf_england_v2_3943.ipynb  # England training notebook
  train_scotland_v1.ipynb          # Scotland training notebook
  train_ireland_v1.ipynb           # Ireland training notebook
experiments/outputs/runs/          # Saved model artifacts
run_weekly.py                      # Weekly automation script
run_monthly_drift.py               # Monthly drift automation script
```
