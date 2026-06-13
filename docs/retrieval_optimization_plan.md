# Retrieval Optimization Plan

Updated: 2026-06-12

## Goal

Use the 100-sample DocVQA error breakdown to refine the retrieval optimization strategy and land code changes that are directly aligned with the main bottlenecks:

- `true_miss`
- `ranking_issue`
- `same_page_false_negative`

## Error-Driven Priorities

Current fine-grained split from `outputs/eval/docvqa_val_100_error_analysis_exp5.json`:

- `clean_hit`: 10
- `true_miss`: 43
- `ranking_issue`: 38
- `same_page_false_negative`: 6
- `generation_issue`: 3

Interpretation:

1. Retrieval remains the primary bottleneck.
2. Most remaining gain should come from better candidate selection and ranking, not answer generation.
3. `same_page_false_negative` is partly an evaluation/indexing artifact, because the same original document page is duplicated under multiple `sample_id`s.

## Landed Changes

### 1. Chunk metadata propagation

Changed:

- `scripts/07_parse_and_chunk.py`

What landed:

- benchmark sample metadata such as `ucsf_document_id`, `ucsf_document_page_no`, and `question_types` is now propagated into chunk metadata instead of being dropped at MinerU-block expansion time.

Why:

- this enables future doc-page-aware reranking, duplicate suppression, and scoped retrieval experiments.

### 2. Query-type-aware rerank

Changed:

- `src/models/retrieval.py`

What landed:

- stronger rerank now includes extra logic for:
  - layout questions
  - page-number questions
  - heading/title/logo style questions
  - short heading-like chunks
  - page-marker-like chunks

Why:

- the error table shows large `layout` and `table/list` concentrations in `true_miss` and `ranking_issue`.

### 3. Duplicate-aware diversification controls

Changed:

- `src/models/retrieval.py`
- `src/utils/settings.py`
- `configs/retrieval.yaml`
- `scripts/12_run_benchmark_eval.py`

What landed:

- added optional duplicate-aware rerank postprocessing with:
  - text fingerprint penalty
  - doc-page duplicate penalty
  - same-sample repeat penalty
- added benchmark CLI knobs for diversification ablations.

Important:

- this path is implemented, but kept `off` by default because the first ablation did not beat the current best benchmark line.

## Benchmark Ablations On The Fixed 100-Sample Scope

Reference historical best:

- `docvqa_val_100_retrieval_exp5_densefusion_w045`
- `hit@5 = 0.51`
- `precision@5 = 0.118`
- `citation_accuracy = 0.13`

New ablations after the landed code changes:

| run | retrieval profile | hit@5 | precision@5 | citation@1 |
| --- | --- | ---: | ---: | ---: |
| `docvqa_val_100_retrieval_opt_v3_nodiv` | query-aware rerank, no diversification | 0.51 | 0.118 | 0.11 |
| `docvqa_val_100_retrieval_opt_v3_exp5_nodiv` | query-aware rerank + visual dense fusion | 0.51 | 0.118 | 0.12 |
| `docvqa_val_100_retrieval_opt_v3_exp5_softdiv` | query-aware rerank + visual dense fusion + soft diversification | 0.51 | 0.116 | 0.12 |

Interpretation:

1. Query-aware rerank is stable, but by itself did not lift top-1 citation over the previous best `0.13`.
2. Duplicate-aware diversification is useful as an experimental control, but should not replace the current mainline yet.
3. The remaining gap is likely not heuristic ranking alone; it points toward a stronger scoped-retrieval mechanism for duplicated pages.

## What This Means For The Next Optimization Stage

### Keep as active mainline ideas

- query-type-aware rerank
- visual dense fusion as a strong branch
- chunk metadata propagation for doc-page-aware methods

### Do not promote to default mainline yet

- duplicate-aware diversification

Reason:

- current ablations did not exceed the existing best `citation_accuracy = 0.13`.

## Recommended Next Experiments

### Exp 6: Image-scoped retrieval for RAG path

Goal:

- specifically attack `same_page_false_negative` and duplicated-page ranking noise in the multimodal path.

Method:

- when the request includes an image/document hint, restrict or strongly bias retrieval toward chunks from the same uploaded page/document family.

Expected gain:

- mainly `citation_accuracy`
- secondarily `hit@5` on the RAG path

### Exp 7: Doc-page-aware rerank

Goal:

- use propagated `ucsf_document_id/page_no` metadata to suppress redundant chunks from the same page and re-rank by page-level evidence quality.

Method:

- score page groups first, then choose the best chunk within each page group.

Expected gain:

- reduce noisy top-1 collisions
- improve `precision@5`

### Exp 8: Split reporting into two benchmark views

Goal:

- separate "strict sample-level retrieval" from "same-original-page retrieval".

Method:

- keep the original strict metric for continuity
- add a supplemental doc-page-level metric in reports

Expected gain:

- clearer diagnosis
- avoids over-attributing benchmark artifacts to retrieval quality

## Commands

Rebuild the fixed 100-sample benchmark scope:

```powershell
.\.venv\Scripts\python.exe scripts/07_parse_and_chunk.py --source-mode benchmark --datasets docvqa --splits val --limit-per-split 100
.\.venv\Scripts\python.exe scripts/08_build_indexes.py
```

Run query-aware rerank without diversification:

```powershell
$env:RETRIEVAL_DIVERSIFY_RESULTS='false'
.\.venv\Scripts\python.exe scripts/12_run_benchmark_eval.py --suite retrieval_benchmark --datasets docvqa --splits val --limit-per-split 100 --run-name docvqa_val_100_retrieval_opt_v3_nodiv
```

Run query-aware rerank plus visual dense fusion:

```powershell
$env:RETRIEVAL_DIVERSIFY_RESULTS='false'
.\.venv\Scripts\python.exe scripts/12_run_benchmark_eval.py --suite retrieval_benchmark --datasets docvqa --splits val --limit-per-split 100 --visual-dense-fusion --visual-dense-weight 0.45 --run-name docvqa_val_100_retrieval_opt_v3_exp5_nodiv
```

Run soft diversification ablation:

```powershell
$env:RETRIEVAL_DIVERSIFY_RESULTS='true'
$env:RETRIEVAL_FINGERPRINT_DUPLICATE_PENALTY='0.04'
$env:RETRIEVAL_DOCPAGE_DUPLICATE_PENALTY='0.02'
.\.venv\Scripts\python.exe scripts/12_run_benchmark_eval.py --suite retrieval_benchmark --datasets docvqa --splits val --limit-per-split 100 --visual-dense-fusion --visual-dense-weight 0.45 --run-name docvqa_val_100_retrieval_opt_v3_exp5_softdiv
```
