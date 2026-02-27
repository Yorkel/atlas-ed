# To-Do

## Technical — Critical Path

- [ ] **Deploy FastAPI** — Write `Dockerfile`, push image to Railway or Fly.io. Everything below is blocked until a live API URL exists.
- [ ] **Write `batch_runner.py`** — Calls the deployed FastAPI `/predict` endpoint for each inference article, writes results back to Supabase (`dataset_type = "inference"`).
- [ ] **Run inference backfill** — Process the synthetic weekly datasets (Jan 2019 → Feb 2020) through the batch runner.
- [ ] **Migrate Streamlit dashboard to Supabase** — Replace CSV reads with Supabase queries across all 5 pages.
- [ ] **Deploy Streamlit dashboard** — Push to Streamlit Community Cloud. Hold the public URL until ISD approval is received.

## Technical — Secondary

- [ ] **Fix URLs in scraping repo** — 2,992 training articles in Supabase have `url = NULL`, meaning they can't be matched back to topic assignments in future runs. Investigate why the scraping pipeline didn't write URLs and fix it.
- [ ] **Optional: drift monitoring** — Write `drift_monitor.py` + create `drift_metrics` table in Supabase to track topic distribution shift over inference batches (KL/JS divergence, OOV rate, confidence degradation).

## Governance — Do Now (in parallel with technical work)

- [ ] **IP ownership — PRIORITY** — Email `innovationpolicy@ucl.ac.uk` to confirm whether apprenticeship outputs are staff IP (owned by UCL) or researcher-owned. This affects the validity of the existing MIT licence. Do this this week.
- [ ] **ISD cloud approval** — Email `researchdata-support@ucl.ac.uk` asking whether Supabase, Railway, and Streamlit Community Cloud are acceptable for a non-personal research data project. Ask a direct yes/no question. Do this this week.

## Governance — Before Public Launch

- [ ] **Ethics confirmation** — Confirm ethics approval covers scraping of SchoolsWeek (commercial publication). Contact `ioe.researchethics@ucl.ac.uk` if unclear.
- [ ] **UCL branding sign-off** — Consult `brand.comms@ucl.ac.uk` before using UCL Grand Challenges branding on a public-facing site.
- [ ] **Worktribe registration** — Register the project to trigger formal departmental approvals.
- [ ] **Data Management Plan** — Complete via DMPonline: https://dmponline.dcc.ac.uk/

## Out of Scope (Separate Repos)

- Sentiment analysis pipeline
- AtlasED public website
