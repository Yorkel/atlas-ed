# When the Setup Shapes the Finding: Specification Sensitivity Results from AtlasED

*AtlasED Project — University College London*

---

## The Problem

Governments increasingly use computational text analysis — topic modelling, retrieval-augmented generation, automated document synthesis — to understand policy landscapes and allocate resources. These tools process thousands of documents and return structured summaries: the main issues, the key trends, which topics are rising or falling.

But these outputs depend on choices made during setup. Which documents are included. How many topics the model looks for. Which metric determines whether the model is "good enough." These are specification choices — decisions an analyst makes before results appear. They are rarely tested, almost never disclosed, and typically invisible to the people who act on the findings.

This report presents the results of a systematic specification sensitivity analysis conducted on AtlasED, a live policy intelligence platform covering education discourse across England, Scotland, and Ireland. The analysis tested whether the platform's findings would change under alternative, equally defensible analytical setups.

They did. Substantially.

---

## What We Tested

AtlasED uses Non-negative Matrix Factorisation (NMF) topic modelling to identify policy themes across a corpus of approximately 12,000 documents from government departments, think tanks, data organisations, professional bodies, and specialist media. The England pipeline was tested across four model configurations:

- k=5 (5 topics, low resolution)
- k=15 (15 topics, medium resolution)
- k=30 (30 topics, high resolution — production baseline)
- k=30 NM (30 topics, media sources removed)

| Model | k | Corpus | Articles | Stability | Coherence (c_v) |
|-------|---|--------|----------|-----------|-----------------|
| `eng_k5` | 5 | Full (all sources) | 3,939 | 1.000 | 0.598 |
| `eng_k15` | 15 | Full (all sources) | 3,939 | 1.000 | 0.681 |
| `eng_k30` | 30 | Full (all sources) | 3,939 | 0.966 | 0.689 |
| `eng_k30_nm` | 30 | No media (Schools Week removed) | 1,198 | 0.961 | — |

All models use identical preprocessing (TF-IDF: min_df=3, max_df=0.85, max_features=3,000, ngram_range=[1,2]) and identical NMF parameters (init=nndsvd, max_iter=1,000, random_state=42).

**Source composition of the full corpus:** Schools Week 2,741 (69.6%), GOV.UK 679 (17.2%), FFT Education Datalab 202 (5.1%), Education Policy Institute 111 (2.8%), Nuffield Foundation 106 (2.7%), Federation of Education Development 100 (2.5%).

These configurations were analysed using cross-model Jaccard similarity, source concentration analysis, coherence and stability metrics, dominant weight distributions, and LLM-assisted topic labelling validation.

**The three-component lens.** Specification choices fall into three categories: which data is included (domain), how the measurement is defined (construct), and how the output is evaluated (frame). Each produces a different kind of sensitivity with different implications. The most visible sensitivity is in the data. The most dangerous is in the measurement. The most hidden is in the evaluation. The findings below are organised accordingly.

---

## Finding 1: Source Composition Determines Whether the Model Finds a Debate or a Set of Monologues

*Component: Domain — which documents are in the corpus*

Removing media sources from the corpus while holding all other settings constant changed 93% of topic names (Jaccard similarity: 0.07). This is not refinement. It is a qualitatively different picture of the policy landscape.

With media included, the model finds a connected public debate: teacher strikes, school funding crises, education politics, SEND provision. These are the topics that dominate headlines and parliamentary questions.

Without media, the model finds institutional silos: the Education and Skills Funding Agency's regulatory framework (invisible with media present), research organisations' measurement agendas, and government consultation processes. Teacher strikes, education politics, and the RAAC building safety crisis vanish entirely — they exist in the corpus only because media covered them.

One source — Schools Week — accounts for approximately 70% of the corpus. At k=30, nine of thirty topics have a single source contributing more than 90% of their content. At k=5, two of five topics exceed 90% single-source concentration (government at 99%, Schools Week at 92%), and all but one are dominated by a single publisher: the model discovers who publishes most, not what the policy landscape contains.

The severity of this concentration is partly a function of the corpus composition. At a 70/30 source split, the model is almost mechanically forced to carve source-shaped clusters. A more balanced corpus — say, six sources each contributing 10–20% — would likely show lower single-source concentration. However, the underlying sensitivity would persist: any corpus has a composition, every composition is a choice, and that choice shapes what the model finds. The skew in this corpus makes the problem acute enough to measure precisely; it does not create the problem.

An illustrative RAG comparison reinforced this finding. The same policy question asked against the full corpus and the no-media corpus produced different answers with different policy implications. The full-corpus answer was crisis-driven and politically contextualised, with seven of ten retrieved documents from Schools Week. The no-media answer was research-driven and structurally focused, with six of ten retrieved documents from the Education Policy Institute. The user had no way to know. This is a single-query illustration, not a systematic evaluation — but it demonstrates that the source composition effect extends beyond topic modelling to retrieval-based systems.

*What this means: The "education debate" is an artefact of media, not a natural property of the system. Source composition doesn't just bias the findings — it determines whether the model surfaces a shared discourse or a collection of institutional monologues. The extreme skew in this corpus amplifies the effect, but the underlying mechanism — that corpus composition determines what the model can find — applies to any system built on curated document collections.*

---

## Finding 2: The Number of Topics Changes What the Model Measures — Not Just How Well It Measures It

*Component: Construct — how the measurement is defined*

Increasing from 5 to 30 topics preserved only 6% of topic names (Jaccard similarity: 0.06, computed over exact name matches as described in the Jaccard section below). The model at k=5 and the model at k=30 are not measuring the same things at different levels of detail. They are measuring fundamentally different things.

At k=5, a single catch-all topic absorbs 49% of the corpus. This topic contains school absence, free school meals, mental health, exclusions, SEND, primary assessment, and disadvantage gaps — seven substantively distinct policy areas collapsed into one undifferentiated mass. At k=30, that single topic splits into eight distinct themes, each with its own source profile, temporal trajectory, and policy implications.

| Metric | k=5 | k=15 | k=30 |
|--------|-----|------|------|
| Largest topic share | 49.2% | 10.1% | 7.9% |
| Topic range | 4.1%–49.2% | 2.5%–10.1% | 0.8%–7.9% |
| Single-source topics (>90%) | 40% (2/5) | 33% (5/15) | 30% (9/30) |
| Multi-source topics (<70%) | 20% (1/5) | 33% (5/15) | 40% (12/30) |
| Mean dominant weight | 0.090 | 0.120 | 0.125 |

These seven policy areas are not missing from the data. They are present in the corpus at every resolution level. They are missing from the measurement — hidden by a model configuration choice that no standard diagnostic flags as problematic.

Smaller, specialist voices — the Fischer Family Trust, the Education Policy Institute, the Nuffield Foundation — never exceed 9% of any topic at k=5. At k=30, FFT becomes the plurality source for school absence (55%) and Nuffield leads research and social justice (53%) — both invisible at lower resolutions. The resolution parameter functions as a volume control for minority sources: lower resolution systematically silences them.

Two topics survived every resolution tested: academy trust governance and Ofsted inspection reform. Both are anchored to distinct institutions with dedicated vocabulary. Their robustness could reflect genuine structural features of the education system — or it could reflect that NMF mechanically rewards topics with highly distinctive word sets, regardless of their structural significance. These topics use specialised vocabulary (trust, academy, MAT, regional director; inspection, Ofsted, report card, chief inspector) that separates cleanly from the rest of the corpus at any k. Topics with less distinctive vocabulary — even if they represent equally important policy areas — are more likely to merge or shift across configurations. The robustness of these topics is real, but the explanation may be methodological rather than substantive.

*What this means: The choice of k is not a technical parameter. It is a decision about what counts as a finding. At low k, the model measures who publishes. At high k, it measures what they say about. Both pass standard evaluation. This is the most dangerous kind of specification sensitivity — not because it's the largest, but because it's invisible. The data contains the signal. The measurement definition hides it.*

---

## Finding 3: Standard Evaluation Metrics Cannot Detect Any of This

*Component: Frame — how the output is evaluated*

The most structurally concerning finding is that every standard diagnostic performed similarly across configurations that produced completely different outputs.

Coherence (c_v) jumped from 0.598 at k=5 to 0.681 at k=15, then plateaued through k=50 (0.693). The metric cannot distinguish between 15 and 50 topics. This plateau is not a failure of coherence as a measure — coherence captures local word co-occurrence quality within topics, and that quality genuinely stabilises across this range. The problem is that coherence measures something different from what analysts typically assume: it rewards well-formed word clusters, not analytically useful findings. A model with 5 topics — producing source proxies and a 49% catch-all — still returns a "usable" coherence score because those source proxies have internally coherent vocabulary.

Stability was perfect (1.0) at k=5 and k=15, dropping to 0.97 at k=30. But perfect stability at low k signals a model too simple to perturb — there is only one solution because the model lacks the complexity to have alternatives. Models that appeared most stable were those too simple to capture meaningful structure.

The two k=30 models had similar stability (0.966 vs 0.961) but divergent dominant weight profiles — the no-media model showed noticeably higher confidence (mean 0.195 vs 0.125, max 0.582 vs 0.401) because specialist sources produce sharper single-topic assignments than journalism. This difference is itself a finding: source composition changes not just what the model finds but how confidently it assigns articles to topics. A more "confident" model is not a better model — it reflects the narrower editorial scope of its sources.

| Metric | eng_k30 (full) | eng_k30_nm (no media) |
|--------|---------------|----------------------|
| Stability | 0.966 | 0.961 |
| Mean dominant weight | 0.125 | 0.195 |
| Max dominant weight | 0.401 | 0.582 |
| Topic name overlap | — | 7% Jaccard similarity |

Yet 93% of their topics differed. Similar diagnostics. Completely different picture of education policy.

LLM-generated topic labels showed high agreement across configurations: 80% at k=5 (4/5 AGREE), 93% at k=15 (14/15 AGREE), 93% at k=30 (28/30 AGREE) — but only 57% at k=30 after model retraining on a slightly different corpus. At k=15, the divergence between "reform" and "controversy" as labels for the same topic cluster represents a genuine framing difference with policy implications — not a labelling error. LLM naming also failed to detect that three topics in the no-media model were near-duplicates (the ESFA regulatory triplet consuming 11.8% of topic space) — it validates individual topics but cannot assess model structure.

Dashboard presentation compounds the problem. The dominant topic assigned to each document represents, on average, only 12.5% of that document's actual content (mean dominant weight: 0.125 at k=30). The remaining 87.5% of topical information is hidden behind a single label.

*What this means: The evaluation infrastructure that the field relies on has a blind spot. It cannot detect specification sensitivity in either the data or the measurement. Coherence cannot tell you whether your findings depend on who's in the corpus. Stability cannot tell you whether your model is robust or merely simple. Diagnostics that report similar quality across configurations with 93% different content are not diagnostic of anything that matters. This is not a gap in one metric — it is a structural limitation of evaluating models without testing specifications.*

---

## What Holds and What Changes

Across all tested configurations:

**Structurally robust findings (act with confidence):**

Academy trust governance and Ofsted inspection reform appeared in every tested specification — all resolution levels, with and without media. They are anchored to distinct institutions with dedicated vocabulary. Their survival may reflect genuine structural features of the education system, or it may reflect that NMF mechanically preserves topics with highly distinctive word sets (see Scope and Limitations). Either way, they are the most specification-resistant findings in this analysis. SEND and educational disadvantage appeared across most configurations, though they merged with other topics at low resolution. The pattern of institutional segmentation — government regulates, data organisations measure, think tanks argue, media connects — is itself a robust finding that appeared in every model comparison.

**Contingent findings (investigate further before acting):**

Teacher strikes, education politics, RAAC building safety, and school funding crises disappeared when media sources were removed. Their prominence in the full-corpus model is editorially driven, not structurally determined. This does not mean these issues are unimportant — it means their visibility in the analysis depends on whether media is included, and that dependence should be disclosed. Topic rankings — which topic is "largest" or "most important" — changed with every specification choice. No single ranking should be treated as definitive.

**Hidden findings (consider expanding the analysis):**

The ESFA regulatory framework — a triplet of topics covering funding allocation, compliance, and academy financial oversight — was invisible in the full corpus but prominent without media. Seven distinct policy areas (school absence, free school meals, mental health, exclusions, primary assessment, SEND provision, disadvantage gaps) were invisible at k=5 but distinct at k=30. These represent blind spots in any analysis conducted at low resolution or with media-dominated corpora.

---

## Cross-k Topic Progression

Topics that survive across all values of k correspond to distinct institutional structures:

| Topic | k=5 | k=15 | k=30 |
|-------|-----|------|------|
| Academy trust governance | Present | Present | Present |
| Ofsted inspection reform | Present | Present | Present + report cards split |
| Academy financial oversight | Present | Present | Present |

Topics that fragment as k increases reveal hidden policy distinctions:

| k=5 topic (49.2%) | k=15 splits | k=30 splits |
|-------------------|------------|------------|
| pupil_welfare_and_inclusion | child welfare, pupil attainment, SEND, school absence | + mental health, free school meals, exclusions, primary assessment |

| k=5 topic (20.2%) | k=15 splits | k=30 splits |
|-------------------|------------|------------|
| teacher_pay_and_workforce | recruitment/retention, pay/strikes | + strikes as own topic |

The splitting pattern follows the actual structure of education policy, not random fragmentation. But the choice of where to stop determines which distinctions are visible.

---

## Jaccard Similarity Across Model Pairs

| Comparison | Jaccard | Shared topics | What it shows |
|-----------|---------|--------------|--------------|
| k=5 vs k=15 | 0.18 | 3 / 17 | Modest k increase changes 82% of topics |
| k=5 vs k=30 | 0.06 | 2 / 33 | Almost nothing survives |
| k=15 vs k=30 | 0.15 | 6 / 39 | Six stable topics anchored to institutions |
| k=30 vs k=30 NM | 0.07 | 4 / 56 | Same k, different corpus — 93% change |

Jaccard similarity here is computed over exact topic name matches — two topics are "shared" only if they have the identical human-assigned label across configurations. This is a conservative measure: topics with overlapping content but different names (e.g., `school_absence_and_attendance` vs `pupil_absence_attendance`) are counted as different. The Jaccard values are low enough that even a more generous matching criterion (substring overlap, semantic similarity) would not change the core finding, but exact matching was chosen for reproducibility and to avoid introducing a subjective similarity threshold.

The only topics that survive every perturbation are those anchored to specific institutions (Ofsted, academy trusts, DfE). Everything else — the policy substance a policymaker would act on — is contingent on specification choices.

---

## Recommendations

**For analysts building policy AI systems:**

*On data (domain):* Disclose source composition alongside every finding. Test whether findings survive source variation before presenting them as conclusions. Flag when a single source exceeds a dominance threshold for any topic — the finding may reflect that source's editorial agenda rather than the policy landscape.

*On measurement (construct):* Report results at a minimum of two resolution levels. Disclose what is invisible at the chosen resolution — the "missing findings" list should be part of standard output. Justify model configuration choices as measurement decisions, not just technical defaults. Test whether the conclusion holds when the measurement is defined differently.

*On evaluation (frame):* Do not rely solely on coherence, stability, or other standard metrics to validate outputs. These metrics cannot detect specification sensitivity. Supplement standard evaluation with specification variation testing — hold the model constant and vary the data, hold the data constant and vary the model, and compare what changes. Report dominant weight distributions so users know how much information the single-label presentation hides.

**For policymakers using AI-generated analysis:**

Ask three questions about any AI-generated policy finding. First, what data went into the system and has the finding been tested with different data compositions? Second, how was the measurement defined and would the finding change under an alternative, equally defensible definition? Third, what evaluation standard determined that the model was "good enough" and has the finding been tested against alternative standards? Treat findings that have not been tested across all three as preliminary rather than settled.

**For regulators and standards bodies:**

Require specification sensitivity testing as part of algorithmic transparency documentation. Current standards focus on model performance metrics — accuracy, fairness, coherence — which are necessary but insufficient. They cannot detect the kinds of specification sensitivity demonstrated here, where models pass every standard check while producing fundamentally different conclusions. Develop robustness reporting standards that require disclosure across all three specification components: what holds when the data changes, what holds when the measurement changes, and what holds when the evaluation standard changes. The distinction between robust and contingent findings — not just between accurate and inaccurate ones — should be a required element of any algorithmic impact assessment.

---

## Methodology

**Model:** Non-negative Matrix Factorisation (NMF). Chosen because every specification choice is explicit and auditable — k is a parameter, vocabulary is defined by TF-IDF settings, per-article weights are inspectable, and the model is deterministic with NNDSVD initialisation. A BERTopic or LLM-based approach would bury specification choices inside embeddings, clustering algorithms, and prompts. HDBSCAN decides k for you — it cannot be perturbed the same way. Embeddings are opaque. Non-deterministic runs produce different topics. Embedding-based models may be differently sensitive to specification choices — potentially less sensitive to k (since clustering is data-driven) but more sensitive to embedding model selection, chunk size, and preprocessing decisions that are harder to enumerate. Crucially, the specification sensitivity framework used here — perturb one choice, hold the rest constant, compare outputs — is not NMF-specific. The same approach could be applied to BERTopic (perturb embedding model, clustering parameters, or corpus composition) or to RAG systems (perturb retrieval strategy, chunk size, or generation model). The specification sensitivity demonstrated here on NMF is a lower bound on what can be measured in a transparent system, not necessarily a lower bound on what exists in more complex ones.

**Corpus:** 3,939 English education policy articles (2023–2025) from six sources: Schools Week (2,741), GOV.UK (679), FFT Education Datalab (202), Education Policy Institute (111), Nuffield Foundation (106), Federation of Education Development (100). Four very short articles (<200 characters) were excluded during preprocessing.

**Evaluation:** Stability testing across 5 random seeds, coherence (c_v) sweep from k=5 to k=55, LLM naming validation (Claude), source concentration analysis, Jaccard similarity between model pairs.

**Perturbations:** Two dimensions tested — k (5, 15, 30) and corpus composition (full vs no-media). Four model variants total. Each trained with identical preprocessing and NMF parameters. The only variables are k and corpus composition.

**RAG comparison:** Two FAISS indexes built with sentence-transformer embeddings — one from the full corpus, one from the no-media corpus (1,198 articles). Same questions asked of both using Claude as the generation model.

---

## Scope and Limitations

Not all findings in this report generalise equally. It is important to distinguish between findings that reflect properties of this specific corpus and findings that reflect structural properties of the method.

**Corpus-specific findings (severity would decrease with a more balanced corpus):**

The extreme single-source concentration in Finding 1 — nine of thirty topics above 90% from one publisher — is amplified by the 70/30 source split. A corpus with fifteen sources each contributing 5–10% would almost certainly show lower concentration per topic. The specific severity of the source composition effect is partly an artefact of the corpus available for English education policy analysis. However, the underlying mechanism (that source composition determines what the model finds) would persist at any split — the effect would be harder to detect, not absent.

**Method-structural findings (would persist regardless of corpus size or composition):**

The construct sensitivity in Finding 2 — that k determines what the model can distinguish — is a property of topic modelling, not of small or skewed datasets. A corpus of 100,000 articles would still collapse seven policy areas into one topic at k=5. The coherence plateau in Finding 3, and the inability of standard metrics to detect substantive differences between configurations, are properties of those metrics. More data will not make coherence able to distinguish between configurations that produce different policy conclusions.

**Findings that could intensify with a larger corpus:**

With more sources, the space of "equally defensible" corpus definitions expands. More inclusion/exclusion combinations become available to test, and the potential for source composition effects increases rather than decreases. What looks structurally robust in a six-source corpus might turn out to be contingent on which institutions happen to be included.

**Temporal dynamics (not tested):**

The corpus covers 2023–2025 — a period that includes a UK general election, the RAAC building safety crisis, and significant shifts in education policy. Topic distributions almost certainly shifted over time, and some specification sensitivity may interact with temporal dynamics. In particular, because Schools Week accounts for 70% of the corpus, temporal shifts in one outlet's editorial priorities could appear as temporal shifts in the policy landscape — a particularly sharp instance of the domain-construct interaction that this framework is designed to surface. Temporal sensitivity was not systematically tested in this analysis and represents a limitation of the current work.

**Vocabulary distinctiveness and "robust" topics:**

The two topics that survived every specification perturbation (academy trust governance, Ofsted inspection reform) use highly distinctive vocabulary that separates cleanly from the rest of the corpus. Their robustness may reflect NMF's mechanical tendency to preserve topics with unique word sets rather than genuine structural features of the education system. Further work could test this by measuring vocabulary overlap between topics and correlating it with cross-specification survival — if only low-overlap topics survive, the robustness is methodological rather than substantive.

---

## Towards a Specification Sensitivity Standard

The following is a preliminary proposal, derived from one model type (NMF), one domain (education policy), and one corpus composition (heavily skewed towards a single media source). It is offered as a starting point for discussion, not as a validated standard. Whether these components generalise — across model architectures, corpus sizes, domain types, and deployment contexts — remains an open question that requires structured investigation.

This project proposes five components for specification transparency in AI systems:

1. **Perturbation registry.** Document at least three specification choices that could have been made differently. For each, train or run a variant.
2. **Delta reporting.** Report what changed — not just metrics, but substantive outputs (which topics appeared/disappeared, which answers changed).
3. **Diagnostic sufficiency test.** Check whether evaluation metrics detect the perturbation. If two variants score similarly but produce different outputs, diagnostics are insufficient.
4. **Confidence disclosure.** Every output shown to a user should carry a confidence measure (dominant weight, prompt sensitivity, retrieval coverage).
5. **Specification card.** Like a model card, but documenting the space of models that were not deployed and what they would have found differently. A model card says "we trained NMF at k=30 on 3,939 articles, coherence 0.689, stability 0.97." A specification card says "we also trained at k=5, k=15, and k=30 without media — here is what changed, here is what the diagnostics could not detect, and here is which findings are robust vs contingent on the choices we made."

These components emerged from a single case study. Developing them into a usable standard would require structured specification sensitivity analyses across a range of conditions — different model types (embedding-based clustering, LDA, LLM-based classification), different corpus compositions (balanced vs skewed, small vs large), different domains (health, justice, climate), and different deployment contexts (analytical tools vs user-facing chatbots vs automated decision systems). Each axis is likely to surface different specification sensitivities with different risk profiles. The three-component lens (domain, construct, frame) may prove useful across these contexts, or it may need to be extended. The current work demonstrates the principle and provides one worked example; the standard itself requires broader empirical grounding.

---

## Reproducibility

All models, evaluation outputs, and comparison data are available in this repository:

- Model summaries: `data/evaluation_outputs/nmf_eng_{5,15,30,30_nm}_summary.json`
- LLM naming reviews: `data/evaluation_outputs/llm_topic_review_k{5,15,30,30_nm}.json`
- Coherence sweeps: `data/evaluation_outputs/coherence_sweep_eng.csv`
- Stability seeds: `data/evaluation_outputs/stability_seeds_eng.csv`
- RAG comparison: `data/evaluation_outputs/rag_comparison.json`
- Analysis-ready articles: `data/evaluation_outputs/topics_analysis_ready_eng.csv`
- Trained models: `experiments/outputs/runs/` (with joblib files)

---

## About This Analysis

This specification sensitivity analysis was conducted on AtlasED, a live policy intelligence platform developed at University College London with Grand Challenges funding. AtlasED provides real-time policy landscape analysis across England, Scotland, and Ireland. The platform, source code, and supporting materials are publicly available.
