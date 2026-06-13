# Unique-DocPage-100 Benchmark Protocol

## Purpose

This benchmark fixes the repeated-page bias in the original `DocVQA val first-100` slice.
The original slice contains 100 QA samples but only 29 unique `(document, page)` pairs, which makes retrieval metrics sensitive to duplicate-page collisions instead of true page retrieval quality.

## Fixed Manifest

- Manifest: `outputs/eval/docvqa_val_unique_docpage_100.manifest.jsonl`
- Summary: `outputs/eval/docvqa_val_unique_docpage_100.manifest.summary.json`
- Construction rule:
  - start from `data/processed/docvqa/val.jsonl`
  - group rows by `metadata.ucsf_document_id + metadata.ucsf_document_page_no`
  - keep one representative QA row per page
  - stratify by question type
  - sample exactly 100 unique pages

All future `unique-docpage-100` experiments should use this exact manifest without reshuffling.

## Rebuilt Data Pipeline

To avoid the criticism that the benchmark only changes the eval slice but not the retrieval corpus, this benchmark uses a dedicated rebuilt pipeline:

- documents: `data/processed/benchmarks/docvqa_unique_docpage_100/documents/documents.jsonl`
- chunks: `data/processed/benchmarks/docvqa_unique_docpage_100/chunks/chunks.jsonl`
- indexes: `data/processed/benchmarks/docvqa_unique_docpage_100/indexes/*`
- dataset config: `configs/datasets.docvqa_unique_docpage_100.yaml`

This means retrieval is evaluated against a corpus rebuilt from the same fixed manifest rather than against the old full-slice artifacts.

## Reporting Guidance

- Use the original `DocVQA val 100` as the historical baseline and failure-case benchmark.
- Use `unique-docpage-100` as the main benchmark for later retrieval optimization experiments.
- When reporting improvements, explicitly say whether the metric is:
  - sample-level on original `val-100`
  - sample-level on `unique-docpage-100`
  - doc-page-level on `unique-docpage-100`

## Reproducibility

Recommended pipeline:

```powershell
python scripts/15_build_unique_docpage_benchmark.py
python scripts/07_parse_and_chunk.py --config configs/datasets.docvqa_unique_docpage_100.yaml --datasets docvqa --splits val --sample-manifest outputs/eval/docvqa_val_unique_docpage_100.manifest.jsonl
python scripts/08_build_indexes.py --config configs/datasets.docvqa_unique_docpage_100.yaml
python scripts/13_build_visual_descriptor_index.py --config configs/datasets.docvqa_unique_docpage_100.yaml --datasets docvqa --splits val --limit-per-split 0 --overwrite
python scripts/12_run_benchmark_eval.py --datasets-config configs/datasets.docvqa_unique_docpage_100.yaml --sample-manifest outputs/eval/docvqa_val_unique_docpage_100.manifest.jsonl --mode retrieval --datasets docvqa --splits val --top-k 5 --rerank-profile stronger --visual-dense-fusion --visual-dense-weight 0.45 --run-name docvqa_val_unique_docpage_100_retrieval_exp5_rebuilt
```
