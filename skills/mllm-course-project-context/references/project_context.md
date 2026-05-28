# Project Context: Multimodal Document RAG Course Project

## 1. Project Positioning

- Project type: Course project for multimodal large-model applications.
- Primary scenario: Course document QA over PDF slides, PPT screenshots, scanned pages, charts, and formulas.
- Core problem:
  - Semantic loss in pure-text pipelines.
  - Long-document retrieval and evidence grounding.
- Target outcome: A usable multimodal QA demo plus quantitative comparison against text-only RAG.

## 2. Scope from Proposal (canonical target)

### 2.1 Functional scope

- Structure-aware document parsing and semantic chunking.
- Dual-index retrieval (text + visual).
- Dual-path RAG inference with late fusion.
- Traceable answers with citation (page/figure level).

### 2.2 Evaluation scope

- Datasets: DocVQA, ChartQA, and course-owned document set.
- Metrics:
  - Retrieval: Recall@K, Precision@K
  - Generation: ANLS, Accuracy
  - Trust: citation accuracy
- Comparative groups:
  - Text-only RAG baseline
  - Multimodal RAG without semantic chunking
  - Multimodal RAG with semantic chunking

### 2.3 Expected thresholds

- End-to-end system works on multi-page course documents.
- DocVQA ANLS target >= 60%.
- Multimodal RAG improves key metrics over text-only baseline (target +10% as proposal ambition).
- Semantic chunking improves retrieval recall (target +5% as proposal ambition).

## 3. Current Implementation Status (as of 2026-05-28)

### 3.1 Completed

- Engineering skeleton established:
  - `src/serving`, `src/models`, `src/ui`, `scripts`, `configs`, `docs`.
- Backend API available:
  - `GET /health`
  - `POST /api/v1/chat`
  - `POST /api/v1/embed/text`
  - `POST /api/v1/embed/vision`
- OpenAI-compatible model clients implemented:
  - chat + embedding wrappers in `src/models/clients.py`.
- Streamlit demo available:
  - text input, optional images, optional text/file context.
- Dataset pipeline available:
  - HF download/export scripts.
  - normalization to processed JSONL.
  - health check and cleaning scripts.
- Team SOP/docs already drafted in repo (`docs/*.md`, top-level Chinese planning files).

### 3.2 Partially completed / placeholder level

- `embed/vision` endpoint exists but does not yet reflect a full visual retrieval pipeline.
- citation returned by chat is context-slot based (`ctx-i`) rather than true page/figure grounding.
- retrieval config exists (`configs/retrieval.yaml`) but no full retrieval orchestration is wired into chat flow.

### 3.3 Not completed yet (main gap)

- MinerU/PaddleOCR structure-aware parser integration.
- semantic chunk builder preserving layout semantics.
- vector-store orchestration for production dual-index retrieval (text + visual).
- dual-path retrieval + late-fusion reasoning flow.
- full benchmark runner for Recall/Precision/ANLS/citation accuracy and error taxonomy.

## 4. 3-Week Execution Blueprint (recommended)

## Week 1: Baseline Retrieval Closed Loop

- Goal: Move from API shell to runnable text-RAG baseline.
- Must deliver:
  - query -> retrieve -> generate chain in backend.
  - unified evidence schema (`source_id`, `page`, `chunk_id`, `snippet`).
  - baseline benchmark runner (small subset first).
- Exit criteria:
  - demo can answer from indexed corpus without manual context paste.

## Week 2: Multimodal Upgrade

- Goal: Introduce visual path and dual-route retrieval.
- Must deliver:
  - visual chunk extraction and indexing path.
  - text-topk + vision-topk retrieval merge.
  - late-fusion prompt/protocol for final answer synthesis.
- Exit criteria:
  - chart/image-heavy queries outperform text-only baseline on curated set.

## Week 3: Evaluation and Defense Package

- Goal: Quantitative comparison + stable demo + report assets.
- Must deliver:
  - full experiment table for all groups.
  - failure analysis by category (retrieval miss, fusion conflict, generation hallucination).
  - defense-ready artifacts (figures, pipeline diagram, representative casebook).
- Exit criteria:
  - one-click or low-friction reproducible runbook for TA/demo day.

## 5. 3-Person Team Role Split

- Member A (Data + Infra owner)
  - parser/chunk/index pipeline, dataset integrity, reproducibility scripts.
- Member B (Model + Reasoning owner)
  - retrieval strategy, multimodal prompt design, late-fusion policy.
- Member C (Evaluation + Product owner)
  - benchmark scripts, metrics dashboard, demo UX, report material.

## 6. Milestone Governance

- Daily standup: yesterday/today/blockers (15 min).
- Weekly checkpoint: mandatory runnable demo.
- Merge gate:
  - dataset validation script pass
  - core endpoint smoke test pass
  - metric report updated for experiment branches

## 7. Risk Register and Countermeasures

- Risk: parser integration complexity too high.
  - Mitigation: keep OCR baseline branch; swap parser behind common chunk schema.
- Risk: multimodal retrieval quality unstable.
  - Mitigation: preserve text-only fallback with confidence routing.
- Risk: evaluation not finished before deadline.
  - Mitigation: lock minimal benchmark subset by end of Week 1.
- Risk: member blocking due to interface drift.
  - Mitigation: freeze API/data contracts before Week 2 starts.

## 8. Reuse Rules for Future Requests

When using this context skill:

- Always first classify request into: architecture, implementation, experiment, or presentation.
- Reuse existing repo assets before suggesting new framework migration.
- Return outputs as operational plans with owner + artifact + due date.
- Keep proposal-level ambitions, but prioritize demo reliability if schedule conflict appears.
