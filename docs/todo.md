# To-Do

## Technical — Critical Path

- [x] **Dockerise FastAPI** — `Dockerfile`, `.dockerignore`, `requirements-api.txt` created and tested locally.
- [x] **Deploy FastAPI (interim)** — Deployed to Render free tier: `https://atlased-api.onrender.com`. Health and predict endpoints verified.
- [ ] **Deploy FastAPI to Cloud@UCL (production)** — Request access to Cloud@UCL (AWS or Azure), redeploy Docker image. Blocked on UCL approval (see `docs/uclcheck.md`).
- [x] **Write `batch_runner.py`** — Calls the deployed FastAPI `/predict` endpoint for each inference article, writes results back to Supabase (`dataset_type = "inference"`).
- [x] **Run inference backfill** — 128 articles across 6 weeks (Jan 9 → Feb 20, 2026) processed successfully.
- [x] **Migrate Streamlit dashboard to Supabase** — Created shared `supabase_loader.py`, replaced CSV reads with paginated Supabase queries across all 5 pages. Contestability tab now uses real Shannon entropy scores. Tested locally: 4,100 articles (3,972 training + 128 inference).
- [ ] **Deploy Streamlit dashboard** — Push to Streamlit Community Cloud. Add `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` to Streamlit Cloud secrets. Hold the public URL until ISD approval is received.

## Technical — Secondary

- [ ] **Fix URLs in scraping repo** — 2,992 training articles in Supabase have `url = NULL`, meaning they can't be matched back to topic assignments in future runs. Investigate why the scraping pipeline didn't write URLs and fix it.
- [ ] **Remove `text_clean` from public API/dashboard** — The `text_clean` column contains full article text. Do not expose it via the public API endpoint or display it in the dashboard. For internal pipeline use only. Currently visible in Topic Explorer article expanders — must be replaced with `preview` before public launch.
- [x] **Drift monitoring** — `drift_monitor.py` written, `drift_metrics` table created in Supabase. Tracks JS divergence, confidence, contestability, topic concentration (HHI) per inference week. See `docs/drift_monitor.md`.

## SchoolsWeek & Copyright — Do Now

- [ ] **Email SchoolsWeek editor** — Brief email explaining this is UCL non-commercial research, what the project does, and asking for written permission to store and analyse their articles. They often say yes to academic use and written permission protects you. Contact via the editor form on their site.
- [ ] **Confirm ethics application covers scraping** — Check your submitted ethics application explicitly mentions automated scraping of SchoolsWeek. If not, submit an amendment to `ioe.researchethics@ucl.ac.uk`.
- [ ] **Audit dashboard and website for article text** — Confirm neither the Streamlit dashboard nor the AtlasED website displays full or partial article text. Only topic labels, trends, charts, and URLs linking back to originals are acceptable without explicit permission.

## UCL Governance — Do Now (in parallel with technical work)

- [ ] **IP ownership — PRIORITY** — Email `innovationpolicy@ucl.ac.uk` to confirm whether apprenticeship outputs are staff IP (owned by UCL) or researcher-owned. This affects the validity of the existing MIT licence. Do this this week.
- [ ] **Cloud@UCL access** — Email `isd.cloud@ucl.ac.uk` requesting access to AWS or Azure via Cloud@UCL for container deployment + database. Also ask whether Supabase is acceptable. See `docs/uclcheck.md` for full details.

## UCL Governance — Before Public Launch

- [ ] **UCL branding decision** — Decide whether to launch with or without UCL Grand Challenges branding. Launching without is simpler and avoids needing sign-off. If using UCL branding, contact `brand.comms@ucl.ac.uk` first.
- [ ] **AI use disclaimer** — Add a visible note on the site and dashboard: "Analysis produced using NMF topic modelling. This is a research tool, not an official UCL publication." UCL requires AI use to be explicitly acknowledged in outputs.
- [ ] **Privacy/cookie notice** — If the website or dashboard collects any user data (analytics, contact forms, newsletter), add a privacy notice and cookie banner. If purely read-only with no user input, this is minimal.
- [ ] **Worktribe registration** — Register the project to trigger formal departmental approvals.
- [ ] **Data Management Plan** — Complete via DMPonline: https://dmponline.dcc.ac.uk/
- [ ] **Register dataset** — Record the dataset in the UCL Research Data Repository (Figshare): https://rdr.ucl.ac.uk/

## AM1 Deliverables

- [ ] **5,000 word report** — Full report on format detailed in the apprenticeship standard. Based on a deployed ML model. Appendices not included in word count. Electronic report. 8 weeks to complete and submit after gateway.
- [ ] **20 minute presentation** — Demonstrate how you have met the assessment criteria for AM1. Cover: problem scoping, ML experimentation (NMF vs LDA vs BERTopic), deployment (Render + Supabase + Streamlit), responsible AI (contestability + drift monitoring), stakeholder engagement (dashboard + workshops).
- [ ] **Prepare for 30 min EPAO questioning** — Anticipate questions on: why NMF over deep learning, how you'd handle model degradation, what contestability means in practice, source imbalance limitations, ethical considerations of scraping, deployment trade-offs.

## Out of Scope (Separate Repos)

- Sentiment analysis pipeline
- AtlasED public website (separate repo — see `docs/website_plan.md`)
- Co-design workshops (planning stage — see website plan)
