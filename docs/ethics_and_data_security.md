# Ethics, Data Risk & Security

---

## 1. Ethics Approval

- Ethics application submitted to UCL
- **Action needed:** Confirm the application explicitly covers automated scraping of SchoolsWeek (a commercial publication). If not, submit an amendment.
- Contact: `ioe.researchethics@ucl.ac.uk` (IOE Research Ethics Office)
- Drop-in sessions: Tuesdays 2:30–3:30pm during term time

---

## 2. Copyright & SchoolsWeek

### Legal basis for scraping
The UK Text and Data Mining (TDM) exception (CDPA 1988, s29A) permits automated scraping for non-commercial research purposes. This exception **cannot be overridden by Terms of Service** — SchoolsWeek's ToS prohibition on scrapers does not remove this right for non-commercial academic use.

**Conditions that must hold:**
- Use is non-commercial ✓
- You have lawful access to the source material (i.e. it is publicly accessible) ✓
- You do not share full-text copies with anyone who does not have lawful access ⚠️ — see below

### What you can do
- Scrape and store article text for internal research and pipeline use
- Publish analysis outputs (topic trends, charts, topic assignments)
- Link back to original articles by URL

### What you cannot do without explicit permission
- Display full or partial article text publicly (on the dashboard or website)
- Create anything that functions as a substitute for SchoolsWeek's own service
- Use content for commercial purposes

### Action required
- [ ] Email SchoolsWeek editor requesting written permission for academic storage and analysis. Written permission removes ambiguity entirely.
- [ ] Ensure `text_clean` column in Supabase is never exposed via the public API or dashboard

---

## 3. Data Classification

| Data | Personal? | Sensitive? | Storage | Public? |
|---|---|---|---|---|
| Article text (`text_clean`) | No | No (public articles) | Supabase | No — internal only |
| Topic assignments | No | No | Supabase | Yes (via dashboard) |
| Article metadata (url, date, source) | No | No | Supabase | Yes |
| Model artefacts | No | No | Local / Railway | Via API only |
| MLflow logs | No | No | Local | No |

No personal data is collected or stored at any point in the pipeline.

---

## 4. Security

### Credentials & Secrets
- `.env` file is in `.gitignore` — never committed ✓
- `.env.example` was briefly created with real credentials — **check git history** immediately:
  ```
  git log --all --full-history -- .env.example
  ```
  If it was committed, rotate the Supabase service key at: https://supabase.com/dashboard → Project Settings → API
- `SUPABASE_SERVICE_KEY` grants full database access — treat as a secret, never expose in client-side code or public repos

### Supabase Access Control
- Service key used only in server-side pipeline code (training, batch runner)
- Dashboard and public API should use the **anon/public key** with Row Level Security (RLS) enabled, not the service key
- `text_clean` column should be excluded from any public-facing Supabase views or API responses
- Consider creating a restricted Supabase view for the dashboard that omits sensitive columns

### API Security
- FastAPI deployment should not expose raw article text via any endpoint
- `/predict` endpoint accepts article text as input but should not store or return it beyond the topic assignment
- Rate limiting should be applied before public launch to prevent abuse

### Dashboard Security
- Streamlit Community Cloud is public by default — add password protection during development/testing
- Remove or restrict access until ISD approval is received and copyright position is confirmed

---

## 5. Data Governance (UCL Policy)

UCL policy: research data should be "as open as possible, as closed as necessary."

### Storage
- Article text: closed (internal Supabase, not exposed publicly)
- Analysis outputs: open (published via dashboard)
- Model artefacts: open (MIT licence on code, artefacts shareable)

### Required registrations
- [ ] Register dataset in UCL Research Data Repository (Figshare): https://rdr.ucl.ac.uk/
- [ ] Complete Data Management Plan via DMPonline: https://dmponline.dcc.ac.uk/
- [ ] Register project via Worktribe for formal institutional approval

### Data residency
No strict UK/EU residency requirement for non-personal research data. Supabase, Railway, and Streamlit Community Cloud are likely acceptable — pending ISD confirmation.

---

## 6. IP Ownership

⚠️ **Unresolved — Priority action**

UCL's IP policy assigns ownership of software and technical works created by staff in the course of duties to UCL. As an apprentice (staff), this may mean UCL owns the codebase, not the researcher. The existing MIT licence on the repo may be invalid without UCL's agreement.

- **Action:** Email `innovationpolicy@ucl.ac.uk` this week
- **Question to ask:** Does apprenticeship project work fall under staff IP rules, or is it treated analogously to student work?
- UCL IP Policy (updated Sept 2025): https://www.ucl.ac.uk/enterprise/policies/2025/sep/intellectual-property-policy

---

## 7. AI Use Disclosure

UCL requires that use of generative AI be explicitly acknowledged in outputs. Required disclosures:

- On the AtlasED website: "This analysis was produced using NMF topic modelling trained on SchoolsWeek articles. Results represent statistical patterns in language, not editorial judgements."
- In any academic outputs or reports: acknowledge AI/ML tools used in analysis
- In apprenticeship submission: document use of AI coding tools (Claude, ChatGPT etc.) as per apprenticeship guidelines

---

## 8. Key Contacts

| Issue | Contact |
|---|---|
| Ethics | `ioe.researchethics@ucl.ac.uk` |
| IP ownership | `innovationpolicy@ucl.ac.uk` |
| ISD cloud approval | `researchdata-support@ucl.ac.uk` |
| Data protection | `data-protection@ucl.ac.uk` |
| UCL branding | `brand.comms@ucl.ac.uk` |
| SchoolsWeek permission | Via editor contact on SchoolsWeek site |
