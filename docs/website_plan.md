# AtlasED — Website Plan

> This document covers the public-facing AtlasED website. It is a separate repo from this topic modelling pipeline. The website uses outputs from this pipeline (topic assignments, trends, analysis) but does not reproduce source article text.

---

## Core Concept

**The site is an argument, not just a description.**

AtlasED is not a neutral tool with a website attached. The website itself makes an argument: that AI tools used in policy analysis encode assumptions, that those assumptions require governance, and that governance requires co-design. Every page on the site is part of that argument.

The site has a rhetorical structure as well as an informational one. A visitor who reads About, then Before you explore, then Workshops, should come away understanding not just what AtlasED does but why the governance approach is necessary.

---

## The Audience Problem the Site Solves

**Two communities who do not usually talk to each other.**

AtlasED sits at the intersection of education policy and AI governance. These are communities with different vocabularies, different concerns, and different relationships to the same data. The site serves both — not by dumbing down for one or obscuring from the other, but by providing genuinely different entry points that converge on the same destination.

The destination is the workshops: where domain expertise and technical expertise meet and shape each other. Every page points toward this. The site should feel, from either starting point, like it is building toward something.

---

## Page-by-Page Logic

### Home
The landing page for all audiences. Threefold job:
- Establish what AtlasED is
- Introduce the three-question frame (why / how / so what)
- Explain the Atlas metaphor — which carries the conceptual weight of the whole project

Routes domain and technical audiences toward their respective entry points while making clear that the workshops are where those paths converge.

**No UCL or institutional branding on this page** — this is the project speaking for itself.

### About
The political and ethical argument for the project. Not methodology — that lives in "Before you explore." About answers: why does AI analysis of education policy matter? Why are the outputs of this model contestable? Why does that require a governance response rather than a technical fix?

### Before You Explore
Methodology and transparency. How the model works, what its limitations are, what "topic" means in this context, what contestability means. Written for a non-technical policy audience but precise enough to satisfy a technical one.

This is also where the AI use disclaimer lives prominently.

### Explorer (Dashboard Embed or Link)
The Streamlit dashboard embedded or linked. Users can explore topic trends, contestability, election period analysis, source comparisons.

**Important:** The explorer only shows analysis outputs — topic labels, trends, charts, URLs linking back to original articles. No article text is displayed.

### Workshops
The destination the whole site builds toward. Information about co-design workshops bringing together education policy professionals and AI governance researchers. Sign-up or expression of interest form.

---

## What the Site Shows (and Does Not Show)

| Shown | Not Shown |
|---|---|
| Topic trend charts | Full article text |
| Contestability scores | Article excerpts or quotes |
| Topic distributions by source | Raw scraped data |
| Pre/post election comparisons | Any personal data |
| URLs linking back to SchoolsWeek originals | |
| Methodology and model limitations | |

---

## Technical Stack (Planned)

- Static site (Next.js or similar) — separate repo
- Streamlit dashboard embedded via iframe or linked
- No database — reads from published analysis outputs or directly from Supabase (read-only, anonymised views)
- Minimal or no user data collection

---

## Compliance Checklist for Launch

- [ ] No UCL branding without brand team sign-off (`brand.comms@ucl.ac.uk`)
- [ ] AI use disclaimer visible on About or Before You Explore page
- [ ] No article text displayed anywhere on site
- [ ] SchoolsWeek credited and linked as data source
- [ ] Privacy notice in place if any user data collected (analytics, contact form)
- [ ] Cookie notice if using Google Analytics or similar
- [ ] ISD cloud approval received before making URL public
- [ ] IP ownership clarified before public launch

---

## Open Questions

- Launch with or without UCL Grand Challenges branding? (Simpler without — avoids sign-off process)
- Will the site collect any user data? (Contact form? Analytics?) — determines whether a privacy policy is needed
- Will workshops be in-person, online, or hybrid?
- Who is the intended first audience at launch — researchers, policymakers, or both?
