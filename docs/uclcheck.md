# UCL Checks & Approvals

Everything you need to ask UCL before public launch, what to use in the meantime, and what happens if they say no.

---

## 1. Cloud Platform Access (Cloud@UCL)

**What you're asking:** Access to Cloud@UCL (AWS or Azure) to deploy a Docker container running your FastAPI inference API.

**Who to email:** `isd.cloud@ucl.ac.uk`

**What to say:**
> I'm a degree apprentice in the UCL Grand Challenges programme. I need to deploy a Docker container (FastAPI + Python) for my apprenticeship project — an education policy topic modelling tool. The data is non-sensitive (publicly available news articles, no personal data). Could I get access to Cloud@UCL (AWS or Azure) to use a container hosting service (e.g. AWS App Runner / ECS, or Azure Container Apps)?

**What you use in the meantime:** Run the Docker container locally (`docker run -p 8000:8000 ...`). The inference pipeline works identically against localhost.

**If they say no:** Unlikely — AWS/Azure are already UCL-approved. They may just redirect you to a specific service or team.

---

## 2. Database (Supabase vs UCL-managed PostgreSQL)

**What you're asking:** Is Supabase acceptable for non-personal research data, or must you use an UCL-managed database?

**Who to email:** Same email — `isd.cloud@ucl.ac.uk`

**What to say:**
> The project stores publicly available article metadata (titles, URLs, publication dates) and NMF topic model outputs (topic labels, weights) in Supabase, a managed PostgreSQL service. No personal data is stored. Is Supabase acceptable, or should I migrate to AWS RDS / Azure Database for PostgreSQL via Cloud@UCL?

**What you use in the meantime:** Keep using Supabase. Your data is already there and the pipeline works.

**If they say no:** Migrate to AWS RDS or Azure Database for PostgreSQL. The switch is:
1. Create a PostgreSQL instance on AWS/Azure
2. Export from Supabase: `pg_dump`
3. Import to new DB: `pg_restore`
4. Change connection string in `.env`
5. Swap `supabase-py` client calls for `psycopg2` or `asyncpg` (same SQL tables, different client library)

---

## 3. Streamlit Dashboard Hosting

**What you're asking:** Can you deploy the Streamlit dashboard on Streamlit Community Cloud, or must it be on UCL infrastructure?

**Who to email:** `isd.cloud@ucl.ac.uk` (same email, include in same message)

**What to say:**
> The project also includes a Streamlit dashboard for visualising topic trends. Can I deploy this on Streamlit Community Cloud (a free hosted service), or should it be hosted on UCL infrastructure?

**What you use in the meantime:** Run Streamlit locally (`streamlit run ...`). Works fine for development and your mentor demo.

**If they say no:** Deploy Streamlit as a second Docker container on the same AWS/Azure platform.

---

## 4. IP Ownership (Apprenticeship Work)

**What you're asking:** Who owns the intellectual property — you or UCL? This affects whether your MIT licence is valid.

**Who to email:** `innovationpolicy@ucl.ac.uk`

**What to say:**
> I'm a degree apprentice at UCL (Grand Challenges programme). My apprenticeship project produces software (a topic modelling pipeline and web dashboard). Under UCL's IP policy, is apprenticeship work treated as staff IP (owned by UCL) or student IP (owned by me)? I've currently applied an MIT open-source licence and want to confirm this is valid.

**What you use in the meantime:** Keep the MIT licence. If UCL owns the IP, they'll tell you what to change — but they rarely block open-source for non-commercial research tools.

**If they say UCL owns it:** You may need UCL's permission to open-source it, or switch to a UCL-approved licence. This doesn't affect your technical work at all — just the licence file.

---

## 5. SchoolsWeek Permission

**What you're asking:** Written permission to scrape, store, and analyse their articles for non-commercial academic research.

**Who to email:** SchoolsWeek editor via their website contact form

**What to say:**
> I'm a researcher at UCL conducting non-commercial academic research on education policy discourse in England. My project uses NMF topic modelling to analyse publicly available SchoolsWeek articles to identify policy themes and trends over time. No full article text is displayed publicly — only topic labels, statistical trends, and links back to original articles. Could I have written permission to use your articles for this research?

**What you use in the meantime:** You're already covered by the UK TDM exception (CDPA 1988, s29A) for non-commercial research. Permission is a belt-and-braces extra, not a blocker.

**If they say no:** The TDM exception still applies, but you should:
- Never display article text publicly
- Only show derived outputs (topic labels, trends, charts)
- Link back to original articles rather than quoting them

---

## 6. Ethics Application — Scraping Coverage

**What you're checking:** Does your submitted ethics application explicitly mention automated web scraping?

**Who to email:** `ioe.researchethics@ucl.ac.uk` (only if scraping isn't mentioned)

**What you use in the meantime:** Continue as normal. Ethics amendments are routine and don't block technical work.

---

## Summary: What to Send This Week

| # | Email to | Subject line | Priority |
|---|---|---|---|
| 1 | `isd.cloud@ucl.ac.uk` | Cloud@UCL access for apprenticeship project (container + database) | HIGH |
| 2 | `innovationpolicy@ucl.ac.uk` | IP ownership query — degree apprenticeship project | HIGH |
| 3 | SchoolsWeek editor (contact form) | Permission request — UCL academic research | MEDIUM |
| 4 | `ioe.researchethics@ucl.ac.uk` | Ethics amendment (only if scraping not covered) | LOW |

Emails 1 and 2 can be sent in the same sitting — 10 minutes total.

---

## What You Can Do Right Now (No Approval Needed)

Everything below runs locally and doesn't touch UCL infrastructure:

1. **Run Docker container locally** — `docker run -p 8000:8000 atlased-api`
2. **Write `batch_runner.py`** — Point it at `http://localhost:8000/predict`
3. **Run inference** — Process all articles through the local API
4. **Write results to Supabase** — Non-sensitive data, no blocker
5. **Migrate dashboard to Supabase** — Local development
6. **Demo everything to your mentor** — All runs on your machine
