# Adaptive AI-Orchestrated Manuscript Intelligence Pipeline

> **M.Tech Project — IIT Jodhpur**  
> Developed as part of student research work in Artificial Intelligence / Machine Learning under the academic supervision of **Dr. Dinesh Mohan Joshi**.

---

## 1. Project Overview

This project explores an **Adaptive AI-Orchestrated six-stage pipeline for degraded Devanagari manuscript understanding**.

The original motivation was simple: given an old or degraded manuscript image, can an AI system restore it, identify its text structure, recognize the handwritten content, interpret the Sanskrit, and eventually translate it?

During the research, it became clear that a reliable manuscript system cannot be treated as a conventional:

```text
Image → OCR → Translation
```

pipeline.

Historical manuscripts are difficult because degradation, fading, page damage, bleed-through, script variation, segmentation errors, recognition errors, linguistic ambiguity, and incomplete external knowledge can all influence the result. A fluent-looking OCR or translation output can therefore be misleading if the underlying visual evidence is weak.

The project consequently evolved into an **evidence-preserving and uncertainty-aware architecture**:

```text
Manuscript Image
      ↓
Condition Profiling
      ↓
Stage 1 — Restoration
      ↓
Stage 2 — Damage & Uncertainty
      ↓
Stage 3 — Layout Analysis
      ↓
Stage 4 — Script-Aware Line Segmentation
      ↓
Stage 5 — Sequence-Based HTR
      ↓
Stage 6 — Semantic Interpretation & Trust
      ↓
Adaptive Retry / Scholar Review / Translation Eligibility
```

An **Adaptive AI Orchestrator** acts horizontally across the complete pipeline. It does not replace the six research stages; instead, it evaluates evidence from every stage and decides whether the system should proceed, retry, change strategy, abstain, or finally request human scholarly review.

---

## 2. Research Objective

The project aims to build a manuscript intelligence workflow that can:

1. improve readability without destroying original script evidence;
2. explicitly detect damaged or uncertain visual regions;
3. understand the structural layout of a manuscript page;
4. recover true physical text lines for downstream recognition;
5. generate and compare multiple HTR hypotheses rather than trusting one model blindly;
6. distinguish **recognition confidence/readiness** from actual transcription correctness;
7. apply Sanskrit lexical, morphological, grammatical and contextual reasoning conservatively;
8. reconstruct text only when the available evidence supports it;
9. abstain when the machine cannot justify a transcription;
10. diagnose which stage is responsible for a downstream failure;
11. retry the appropriate stage instead of restarting the complete pipeline unnecessarily;
12. permit scholar review only after available machine recovery strategies are exhausted;
13. allow translation only when the transcription is sufficiently trusted or subsequently validated.

The central design principle is:

> **A scientifically defensible unresolved result is better than a fluent but unsupported transcription.**

---

## 3. Six-Stage Research Pipeline

### Pre-processing — Manuscript Intake and Condition Analysis

Pre-processing is performed before the six formal research stages.

Its purpose is to examine visible manuscript condition, including:

- page brightness;
- contrast;
- uneven illumination;
- fading;
- background discoloration;
- general degradation.

This step provides the first indication of how aggressively the manuscript should be processed. It does not attempt text recognition.

---

### Stage 1 — Degradation-Aware Restoration & Readability Enhancement

**Purpose:** improve manuscript readability while preserving evidence required by later stages.

The research explored multiple image representations:

- original RAW image;
- tone-preserved representation;
- balanced/readability-enhanced representation;
- binarized structural representation;
- separator and special-mark preservation.

A later HTR ablation study compared:

```text
RAW
BALANCED
BINARY
```

and showed an important result:

> A visually cleaner representation is not automatically a better recognition input.

For the current manuscript, RAW/full-line evidence generally performed better than aggressively binarized input. The pipeline therefore preserves several views instead of assuming one preprocessing method is universally optimal.

**Primary orchestration evidence:** `C(p)` and `B(p)`.

- `C(p)` — enhancement/readability adequacy.
- `B(p)` — interference / bleed-through evidence.

---

### Stage 2 — Damage, Interference & Uncertainty Analysis

**Purpose:** identify which regions of the manuscript should receive less downstream trust.

Activities include:

- damage localization;
- uncertainty mapping;
- low-confidence visual-region detection;
- background/interference assessment;
- structural-risk estimation.

The stage intentionally separates:

```text
Physical damage
      from
Visual uncertainty
```

because a region may be difficult to interpret even when it is not visibly damaged.

**Primary orchestration evidence:** `G(p)` — upstream structural reliability / geometry-risk evidence.

---

### Stage 3 — Layout & Text-Region Analysis

**Purpose:** identify coarse text-bearing regions and understand page organization.

Stage 3 answers:

> Where are the meaningful text areas on this page?

It deliberately does **not** claim individual manuscript lines. That responsibility belongs to Stage 4.

Activities include:

- text-region detection;
- text coverage analysis;
- structural fragmentation analysis;
- coarse page-organization analysis.

**Primary orchestration evidence:**

- `L(p)` — layout readiness.
- `R(p)` — pre-segmentation readiness derived from upstream evidence.

`R(p)` summarizes evidence from Stages 1–3 and remains separate from Stage-4 segmentation evidence.

---

### Stage 4 — Script-Aware Physical Line Segmentation

**Purpose:** recover true manuscript text lines while preserving all meaningful foreground ink.

This stage underwent an important research correction.

The initial horizontal-projection analysis produced approximately ten strong row-profile peaks. Directly treating every peak as one line would have split the manuscript incorrectly because Devanagari physical lines can contain several strong vertical ink zones due to:

- śirorekhā / headline structure;
- character bodies;
- upper vowel signs;
- lower modifiers.

The stage therefore evolved from:

```text
Peak Detection
```

to:

```text
Peak Detection
      ↓
Valley Analysis
      ↓
Devanagari Multi-Peak Reasoning
      ↓
Physical-Line Consolidation
```

The segmentation stage evaluates:

- row-profile peaks;
- valley depth between neighboring peaks;
- physical-line grouping;
- ink preservation;
- orphan ink;
- duplicate assignment;
- line-height consistency;
- boundary uncertainty;
- suspected merged regions;
- śirorekhā evidence;
- HTR-safe crop boundaries.

For the primary experimental manuscript, ten raw peaks were consolidated into six physical text lines with strong preservation evidence.

**Primary orchestration evidence:** `S(p)` — Stage-4 segmentation readiness.

A key research conclusion emerged:

> Good segmentation and good recognition are independent properties.

A strong `S(p)` with a weak Stage-5 readiness signal should therefore be diagnosed as a recognition problem rather than repeatedly re-segmenting the page.

---

### Stage 5 — Sequence-Based Handwritten Text Recognition (HTR)

**Purpose:** generate recognition hypotheses and determine whether current HTR evidence is trustworthy enough for semantic processing.

The stage uses more than one recognition provider so that the system does not depend on a single model.

The research explored:

- Provider-A HTR;
- Provider-B HTR;
- N-best recognition hypotheses;
- transliteration-aware processing;
- selective Unicode normalization;
- decoder behavior;
- sequence quality;
- script integrity;
- suspicious repetition;
- sequence-length anomalies;
- cross-provider agreement.

Two provider outputs are compared using both strict and content-oriented similarity.

Importantly:

> Cross-provider agreement is evidence of consistency, not evidence of correctness.

Two models can agree and still be wrong.

Additional experiments included:

#### Input-view ablation

```text
RAW vs BALANCED vs BINARY
```

#### Line-width ablation

```text
Full physical line
vs
2-way chunking
vs
3-way chunking
```

These experiments indicated that neither binarization nor line chunking solved the recognition problem.

The dominant diagnosis for the current manuscript is:

> **manuscript-domain / HTR-model mismatch**

The recognition models often generate Sanskrit-like sequences influenced by learned language priors rather than reliably decoding the manuscript strokes.

**Primary orchestration evidence:** `H(p)` — HTR system readiness.

`H(p)` is intentionally **not**:

- transcription accuracy;
- CER;
- WER;
- model probability;
- scholar confidence.

It is a system-level downstream-readiness signal derived from segmentation, completion, script integrity, decoder reliability, sequence quality, and cross-provider consistency.

---

## 4. Stage 6 — Semantic Interpretation, Reconstruction & Trust

Stage 6 became a multi-step semantic recovery pipeline rather than a direct translation stage.

Its purpose is to answer:

> Can weak/noisy recognition evidence be responsibly reconstructed into manuscript text?

### Stage 6A — HTR Evidence Parser

Preserves structured evidence from Stage 5:

- exact Provider-A output;
- exact Provider-B output;
- N-best alternatives;
- Devanagari display forms;
- line ordering;
- segmentation evidence;
- `H(p)`;
- provider agreement.

No linguistic correction is performed here.

---

### Stage 6B — Position-Aware Candidate Lattice

Builds candidate relationships from observed HTR evidence using:

- transliteration-normalized comparison;
- edit-distance similarity;
- token evidence;
- position-aware clustering;
- cross-provider support;
- limited same-provider recurrence evidence.

The system is intentionally prevented from introducing unsupported Sanskrit merely because a form is linguistically plausible.

---

### Stage 6C — Morphology & Grammar Validation

Uses Sanskrit linguistic analysis to ask:

- is a candidate lexically plausible?
- can it receive a valid morphological analysis?
- is the form derivationally plausible?
- does it behave consistently with Sanskrit grammatical structure?

A crucial research distinction emerged:

```text
Linguistically valid Sanskrit
            ≠
Visually supported manuscript text
```

A word may be excellent Sanskrit but still not be what the manuscript contains.

---

### Stage 6D — Deterministic Context Retrieval

Contextual retrieval was evaluated against a Sanskrit corpus prepared from DCS material.

Retrieval evidence included:

- surface character n-gram similarity;
- lemma similarity;
- exact token overlap.

The initial corpus contained approximately 96,000 passages.

The experiment produced technically valid retrieval results but weak manuscript matches, demonstrating that context retrieval cannot compensate for severely corrupted recognition evidence.

---

### Sanskrit Semantic Embedding Experiment

A Sanskrit-capable semantic embedding model was evaluated as a possible retrieval/recovery mechanism.

The model worked correctly as an embedding system, but the experiment showed that semantic embeddings are not reliable as the first repair layer for highly corrupted HTR.

This reflects a fundamental difference between:

```text
Clean Sanskrit ↔ Clean Sanskrit semantic similarity
```

and:

```text
Severely corrupted HTR ↔ Clean Sanskrit recovery
```

Semantic embeddings remain potentially useful later for reranking cleaner candidates.

---

### Stage 6D.2 — Noisy-Surface Retrieval

A fuzzy retrieval path was introduced to tolerate:

- substitutions;
- missing characters;
- partial words;
- local HTR corruption.

Although numerical similarity improved, several retrieved passages remained contextually unrelated.

This produced another important research conclusion:

> A higher retrieval score does not automatically establish that a passage corresponds to the manuscript.

Corpus coverage and visual provenance remain essential.

---

### Stage 6E — Evidence-Constrained Reconstruction

This stage applies a strict promotion rule.

Retrieved, morphologically valid, or semantically plausible Sanskrit cannot directly become manuscript text.

A reconstructed span requires sufficient support from the existing recognition and linguistic evidence.

Otherwise the system explicitly abstains:

```text
⟦unresolved⟧
```

This prevents the semantic layer from silently converting weak HTR into fluent but fabricated Sanskrit.

---

### Stage 6F — Semantic / Transcription Trust `T(p)`

Stage 6F computes a conservative semantic-transcription trust signal.

Evidence includes:

- `H(p)`;
- cross-provider agreement;
- candidate strength;
- visual + morphological support;
- supported-span coverage;
- reconstruction completion;
- capped contextual evidence.

`T(p)` is **not**:

- a calibrated probability;
- CER;
- WER;
- scholarly correctness.

It represents whether the complete machine reconstruction is sufficiently trustworthy for the next action.

---

### Stage 6G — Adaptive Retry Controller

Stage 6G diagnoses why semantic trust remains weak.

The orchestrator examines the complete evidence vector:

```text
C  B  G  L  R  S  H  T
```

For example:

```text
S = high
H = low
T = low
```

indicates that segmentation is functioning but recognition remains weak.

The orchestrator should therefore change the recognition/recovery strategy rather than restart segmentation.

Potential retry strategies include:

- an additional materially different HTR provider;
- an alternative visual HTR route;
- expanded contextual retrieval;
- earlier-stage retry only when the corresponding evidence indicates a real upstream problem.

Scholar review is authorized only after the available machine recovery strategies have been exhausted.

---

## 5. Adaptive AI Orchestrator

The orchestrator is a horizontal control plane, not an additional research stage.

It gradually accumulates evidence:

```text
After Stages 1–3:
C B G L R

After Stage 4:
C B G L R S

After Stage 5:
C B G L R S H

After Stage 6:
C B G L R S H T
```

This allows the system to reason about the **failure domain** rather than simply declaring the complete pipeline successful or failed.

Example:

```text
Strong segmentation S
        +
Weak recognition H
        +
Weak semantic trust T
        ↓
Failure domain = recognition
```

This is the central adaptive feature of the platform.

---

## 6. Current Research Status

| Area | Status | Current Interpretation |
|---|---|---|
| Pre-processing | Implemented | Manuscript condition profiling available |
| Stage 1 | Implemented | Restoration / evidence-preserving enhancement |
| Stage 2 | Implemented | Damage and uncertainty analysis |
| Stage 3 | Implemented | Coarse layout/text-region understanding |
| Stage 4 | Implemented and validated | Strong physical line segmentation on current manuscript |
| Stage 5 | Integrated | Multiple HTR providers and system readiness `H(p)` |
| Stage 6A | Implemented | HTR evidence parser |
| Stage 6B | Implemented | Candidate lattice |
| Stage 6C | Implemented | Morphology and grammar validation |
| Stage 6D | Implemented | Deterministic contextual retrieval |
| Stage 6D.2 | Implemented | Noisy-surface retrieval |
| Stage 6E | Implemented | Evidence-constrained reconstruction / abstention |
| Stage 6F | Implemented | Semantic/transcription trust `T(p)` |
| Stage 6G | Implemented | Adaptive retry and scholar-last routing |
| Translation | Gated / future completion | Enabled only after sufficiently trusted or validated transcription |

The current experimental manuscript demonstrates a useful failure diagnosis:

```text
Stage 4 segmentation      → strong
Stage 5 recognition       → weak
Stage 6 reconstruction    → insufficient evidence
Final failure domain      → Stage-5 recognition
```

The system therefore refuses to invent a final transcription.

---

## 7. Repository Structure

```text
manuscript-demo/
│
├── backend/
│   └── FastAPI application and integrated pipeline endpoints
│
├── core/
│   └── artifact storage and shared utilities
│
├── layers/
│   ├── layer0_ingest.py
│   ├── layer1_restore.py
│   ├── layer2_damage.py
│   ├── layer3_layout.py
│   ├── layer4_segment.py
│   ├── layer5_htr.py
│   ├── htr_providers.py
│   ├── layer6_semantic.py
│   ├── layer6b_reconstruct.py
│   ├── layer6c_morphology.py
│   ├── layer6d_context_rag.py
│   ├── layer6d2_noisy_retrieval.py
│   ├── layer6e_reconstruction.py
│   └── layer6f_trust.py
│
├── orchestration/
│   ├── signals.py
│   ├── rules.py
│   ├── evaluator.py
│   ├── stage5_routing.py
│   └── stage6_routing.py
│
├── stage5_runtime/
│   └── isolated Stage-5 provider execution
│
├── stage6_runtime/
│   └── isolated Stage-6 substage execution
│
├── frontend/
│   ├── index.html
│   └── orchestration.html
│
├── test_stage5*.py
├── test_stage6*.py
├── diagnose_stage5_*.py
├── compare_stage5_providers.py
├── evaluate_stage5_readiness.py
├── prepare_stage6d_dcs.py
├── requirements.txt
└── .gitignore
```

Generated artifacts, virtual environments, models and large corpora are intentionally excluded from Git.

---

## 8. Runtime Architecture

The validated development environment uses isolated Python environments because the different AI/NLP libraries have different dependency and architecture requirements.

```text
Main Application
venv
│
├── FastAPI
├── Stages 1–4
├── Adaptive Orchestrator
├── Stage-5 Coordinator
└── Stage-6 Coordinator
        │
        ├── venv-stage5
        │      └── Stage-5 Provider A
        │
        ├── venv-provider-b
        │      └── Stage-5 Provider B
        │
        ├── venv-stage6
        │      ├── Stage 6A
        │      ├── Stage 6B
        │      ├── Stage 6C
        │      ├── Stage 6D
        │      ├── Stage 6E
        │      └── Stage 6F
        │
        └── venv-stage6-rag
               └── Stage 6D.2
```

On Apple Silicon, specialist subprocesses are executed natively as ARM64 where required.

The user normally activates **only the main application environment**. The coordinators launch the specialist environments automatically.

---

## 9. Large Assets Not Stored in Git

The repository intentionally excludes:

```text
venv/
venv-stage5/
venv-provider-b/
venv-stage6/
venv-stage6-rag/

artifacts/
models/
knowledge/
data/
```

It also excludes:

- downloaded model weights;
- PyTorch binaries;
- Vidyut data files;
- generated DCS/RAG corpora;
- embedding/vector indexes;
- manuscript runtime artifacts;
- local raw manuscript uploads;
- runtime logs.

These files can be large and should be provisioned locally or through controlled storage.

External models, libraries and corpora remain subject to their respective licenses.

---

## 10. Local Setup

### Prerequisites

The current development environment has been validated primarily on:

```text
macOS
Apple Silicon (M-series)
Python 3.12 environments
Git
```

The architecture is designed so that the same logical pipeline can later be deployed on Linux / AWS infrastructure with appropriate environment recreation.

Clone the repository:

```bash
git clone https://github.com/hashkarma/adaptive-manuscript-intelligence.git
cd adaptive-manuscript-intelligence
```

### Main application environment

Create or activate the main environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
```

Install the main project dependencies:

```bash
pip install -r requirements.txt
```

> The current research runtime also depends on specialist Stage-5 and Stage-6 environments. Provider-specific lock files should be maintained separately as the project is hardened for reproducible deployment.

---

## 11. Specialist Runtime Environments

The current integrated runtime expects these environment names at the project root:

```text
venv-stage5
venv-provider-b
venv-stage6
venv-stage6-rag
```

Conceptually:

```bash
python3.12 -m venv venv-stage5
python3.12 -m venv venv-provider-b
python3.12 -m venv venv-stage6
python3.12 -m venv venv-stage6-rag
```

Each environment should receive only the dependencies required by its research component.

The repository does **not** commit these environments.

For a fully reproducible fresh installation, environment-specific requirements/lock files should be generated and maintained for:

```text
main application
Stage-5 Provider A
Stage-5 Provider B
Stage-6 Vidyut / linguistic processing
Stage-6 RAG / fuzzy retrieval
```

---

## 12. Required External Research Assets

### Vidyut data

Stage 6B/6C expect local Vidyut linguistic data under a path such as:

```text
models/vidyut-0.4.0/
```

These files are intentionally not stored in Git.

### Sanskrit retrieval corpus

The Stage-6 contextual-retrieval experiments use a generated corpus such as:

```text
knowledge/stage6d_dcs/passages.jsonl
```

The corpus is also excluded from Git.

The repository includes a preparation utility:

```bash
python prepare_stage6d_dcs.py
```

Use the script together with the appropriate locally obtained source corpus and ensure that the source dataset's license/attribution terms are followed.

---

## 13. Running the Application

Activate only the main environment:

```bash
source venv/bin/activate
```

Start FastAPI:

```bash
uvicorn backend.main:app --reload
```

The application is normally available at:

```text
http://127.0.0.1:8000
```

The main UI provides the six-stage workflow.

The Adaptive Orchestration Dashboard is available from the UI or at:

```text
/orchestration
```

---

## 14. Normal End-to-End Workflow

A new manuscript is processed conceptually as follows:

```text
Upload
  ↓
Condition Analysis
  ↓
Stage 1 — Restoration
  ↓
Stage 2 — Damage & Uncertainty
  ↓
Stage 3 — Layout
  ↓
Stage 4 — Line Segmentation
  ↓
Stage 5 — HTR + H(p)
  ↓
Stage 6A–6F + T(p)
  ↓
Stage 6G Adaptive Routing
  ↓
Final Decision
```

The web UI exposes buttons for the stages and displays the accumulated evidence.

---

## 15. Important API Endpoints

### Upload

```text
POST /upload
```

### Condition Analysis

```text
POST /analyze/{run_id}
```

### Stage 1

```text
POST /pipeline/stage1/restore/{run_id}
```

### Stage 2

```text
POST /pipeline/stage2/damage/{run_id}
```

### Stage 3

```text
POST /pipeline/stage3/layout/{run_id}
```

### Stage 4

```text
POST /pipeline/stage4/segment/{run_id}
```

### Stage 5 — Integrated HTR Runtime

```text
POST /pipeline/stage5/htr/{run_id}
```

FastAPI keeps running in the main environment while the Stage-5 coordinator launches its HTR providers in isolated environments.

### Stage 6 — Integrated Semantic Runtime

```text
POST /pipeline/stage6/run/{run_id}
```

This launches:

```text
6A
6B
6C
6D
6D.2
6E
6F
```

in the appropriate isolated environments and then executes Stage 6G adaptive routing in the main orchestration process.

### Final Orchestration Report

```text
GET /orchestration/{run_id}
```

This exposes the accumulated pipeline state including:

```text
C B G L R S H T
```

where available.

---

## 16. Running the Integrated Stage-6 Runtime Standalone

For debugging/research validation:

```bash
source venv/bin/activate

python test_stage6_runtime.py \
  <run_id> \
  --project-root . \
  --artifacts artifacts \
  --vidyut-data models/vidyut-0.4.0 \
  --corpus knowledge/stage6d_dcs/passages.jsonl
```

A successful run executes:

```text
6A → 6B → 6C → 6D → 6D.2 → 6E → 6F
```

without manually switching virtual environments.

Stage 6G is subsequently handled by the orchestration layer/API.

---

## 17. Research and Diagnostic Utilities

The repository retains several experimental programs because they document important research decisions:

```text
compare_stage5_providers.py
evaluate_stage5_readiness.py
diagnose_stage5_input_views.py
diagnose_stage5_line_chunking.py
provider_b_model_load_test.py
provider_b_model_load_test_v2.py
test_iast_edge_normalization.py
test_iast_selective_normalization.py
test_stage5_routing.py
test_stage6d_embedding_setup.py
test_stage6g.py
```

These are intentionally retained as research evidence rather than being treated as disposable development scripts.

---

## 18. Artifact Model

Each manuscript run receives a unique `run_id`.

Generated evidence is stored under:

```text
artifacts/<run_id>/
```

Typical artifacts include:

- restored manuscript views;
- damage and uncertainty maps;
- layout masks;
- segmentation overlays and line crops;
- Provider-A/Provider-B HTR hypotheses;
- cross-provider comparison;
- HTR readiness;
- semantic candidate/morphology/context evidence;
- reconstruction evidence;
- semantic trust;
- orchestration decisions;
- runtime logs.

Artifacts are not committed to Git.

---

## 19. Interpretation of the Main Signals

| Signal | Meaning |
|---|---|
| `C(p)` | Enhancement / readability adequacy |
| `B(p)` | Interference / bleed-through evidence |
| `G(p)` | Upstream structural reliability |
| `L(p)` | Layout readiness |
| `R(p)` | Stages 1–3 pre-segmentation readiness |
| `S(p)` | Stage-4 segmentation readiness |
| `H(p)` | Stage-5 HTR system readiness |
| `T(p)` | Stage-6 semantic/transcription trust |

These signals are designed for orchestration.

In particular:

```text
H(p) ≠ recognition accuracy
T(p) ≠ probability of correctness
```

and neither should be interpreted as CER/WER unless genuine ground-truth transcription is available.

---

## 20. Key Research Findings So Far

The project has produced several important practical findings:

1. **Better-looking images do not always produce better HTR.**
2. **A Devanagari projection peak cannot automatically be treated as a physical text line.**
3. **Strong segmentation does not imply strong recognition.**
4. **Sanskrit-looking model output may still be manuscript-inaccurate.**
5. **Linguistic plausibility is different from visual provenance.**
6. **Semantic embeddings cannot reliably repair severely corrupted HTR by themselves.**
7. **Higher retrieval similarity does not prove that a retrieved passage belongs to the manuscript.**
8. **Corpus coverage limits what RAG can recover.**
9. **Explicit abstention is necessary for responsible manuscript reconstruction.**
10. **Retries should target the diagnosed failure domain.**
11. **Scholar review should occur after machine recovery, not as an early shortcut.**
12. **Translation must be gated by transcription trust.**

---

## 21. Current Limitations

The project remains an active research prototype.

Current limitations include:

- HTR models are not yet sufficiently matched to difficult historical manuscript handwriting;
- no large scholar-verified manuscript ground-truth dataset is currently integrated;
- therefore CER/WER cannot be claimed for the primary experimental manuscript;
- external Sanskrit corpus coverage remains incomplete;
- additional manuscript-specific HTR providers/fine-tuning remain future work;
- Stage-6 retrieval is conservative and intentionally cannot override weak visual evidence;
- automatic machine retry capabilities can be expanded further;
- final translation is intentionally gated and is not treated as complete while transcription trust remains insufficient;
- full deployment packaging for AWS/GPU infrastructure is future work.

---

## 22. Future Research Direction

Planned areas include:

- manuscript-domain HTR fine-tuning;
- additional materially different HTR providers;
- expanded Sanskrit / invocation / commentary corpora;
- stronger visual-linguistic candidate alignment;
- calibrated trust studies using scholar-labelled ground truth;
- controlled translation after validated transcription;
- automatic retry execution;
- broader Indian manuscript scripts and languages;
- cloud/GPU deployment;
- benchmark dataset creation;
- research publication and reproducibility studies.

The architecture is intended to remain extensible beyond Devanagari toward additional Indian manuscript traditions.

---

## 23. Academic Context

This repository represents student research and engineering work developed as part of an **M.Tech project at IIT Jodhpur** under the academic supervision of:

**Dr. Dinesh Mohan Joshi**

The work combines manuscript image processing, handwriting recognition, Sanskrit computational linguistics, retrieval, semantic reasoning, uncertainty modelling and adaptive AI orchestration.

The repository should be treated as an evolving **research prototype**, not as a production-grade manuscript transcription authority.

Any final historical or philological interpretation should be validated by appropriate manuscript scholars.

---

## 24. Acknowledgements

The project benefits from the wider open-source and research ecosystem around:

- PyTorch;
- Hugging Face Transformers;
- TrOCR-based recognition models;
- Vidyut Sanskrit linguistic tooling;
- Sanskrit textual/corpus resources used in the research experiments;
- FastAPI;
- OpenCV and scientific Python tooling.

External resources and pretrained models remain governed by their own licences and attribution requirements.

Special acknowledgement is due to **Dr. Dinesh Mohan Joshi, IIT Jodhpur**, for academic supervision and guidance of the M.Tech research direction.

---

## 25. Research Philosophy

The guiding principle of the project can be summarized as:

```text
Preserve the manuscript evidence.
Measure uncertainty.
Do not confuse plausibility with truth.
Retry intelligently.
Abstain when evidence is insufficient.
Involve scholars when machine recovery is exhausted.
Translate only what is sufficiently trusted.
```

This philosophy is what transforms the project from a conventional OCR pipeline into an **Adaptive AI-Orchestrated Manuscript Intelligence Platform**.
