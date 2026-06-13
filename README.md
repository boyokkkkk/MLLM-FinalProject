# Multimodal Doc RAG

A minimal but extensible project skeleton for multimodal document QA.

## What is implemented now

- Repo bootstrap and data directory conventions.
- Backend API with stable contracts for multi-frontend compatibility.
- Model invocation wrappers:
  - VLM chat interface (Qwen-VL via OpenAI-compatible API)
  - Text embedding interface
  - Vision embedding interface
- Demo frontend (Streamlit) that only consumes HTTP API.

## Directory conventions

- Raw docs: `data/raw/pdf`, `data/raw/ppt`
- Parsed/chunked intermediates: `data/interim/*`
- Benchmark data: `data/processed/*`
- Runtime outputs: `outputs/*`

## Backend API contract

Base URL: `http://127.0.0.1:8000`

- `GET /health`
- `POST /api/v1/chat`
  - request:
    - `query: string`
    - `context: string[]` (optional fallback/debug context)
    - `temperature: float | null`
    - `max_tokens: int | null`
  - response:
    - `answer: string`
    - `citations: [{chunk_id, source, page, snippet, source_ref?}]`
    - `model: string`
- `POST /api/v1/embed/text`
  - request: `inputs: string[]`
  - response: `model, vectors`
- `POST /api/v1/embed/vision`
  - request: `inputs: string[]`
  - response: `model, vectors`

This contract is frontend-agnostic and can be consumed by React/Vue/Next.js/mobile clients.

## Quick start

1. Create and activate virtual env:
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

2. Install package in editable mode:
```bash
pip install -e .
```

3. Optional env vars (`.env`):
```env
OPENAI_API_KEY=your_key
VLM_BASE_URL=http://localhost:8001/v1
```

4. Run backend (recommended):
```bash
python -m src api
```

5. Run demo frontend:
```bash
python -m src ui
```

Optional short command after editable install:
```bash
mmrag api
mmrag ui
```

## Local smoke test for retrieval

Build mock retrieval chunks and vectors from local project docs:
```bash
.\.venv\Scripts\python.exe scripts/07_build_mock_retrieval_data.py
```

Run retrieval + chat smoke validation:
```bash
.\.venv\Scripts\python.exe scripts/08_smoke_test_text_retrieval.py
```

## Notes

- Current model clients assume OpenAI-compatible server endpoints.
- You can add new providers by extending `src/models/clients.py` without changing API schemas.

## Dataset preparation

1. Configure download manifest:
- `configs/dataset_downloads.yaml`

2. Download datasets by script:
```bash
python scripts/01_download_datasets.py
```

Or download directly from Hugging Face datasets:
```bash
python scripts/01_download_datasets.py --from-hf
```

Keep official HF directory layout + export project raw files:
```bash
python scripts/01_download_datasets.py --from-hf --hf-official-layout
```

3. Validate and normalize:
```bash
python scripts/01_prepare_datasets.py validate --mode eval
python scripts/01_prepare_datasets.py prepare --mode eval
```

See [docs/dataset_prep.md](docs/dataset_prep.md) for details.

## Offline data and index pipeline

Build document chunks and local indexes from the normalized benchmark records:

```bash
python scripts/07_parse_and_chunk.py --datasets docvqa,chartqa --splits val,test
python scripts/08_build_indexes.py
python scripts/09_query_indexes.py "actual value per 1000 during 1975" --top-k 3
```

Quick smoke run on one or two samples per split:

```bash
python scripts/10_run_data_index_pipeline.py --skip-download --prepare-from-hf-cache --limit-per-split 2 --run-mllm-smoke --mllm-dry-run --mllm-num-samples 2

# Optional: download MinerU weights from ModelScope before real MinerU runs.
python scripts/00_download_mineru_models.py

# Run MinerU before chunking; use --mineru-mock for schema-only smoke tests.
python scripts/10_run_data_index_pipeline.py --skip-download --prepare-from-hf-cache --limit-per-split 1 --run-mineru --mineru-mock --run-mllm-smoke --mllm-dry-run --mllm-num-samples 1
```

See [docs/data_indexing.md](docs/data_indexing.md) for schema, MinerU JSON handoff, index manifest, and query output details.

For end-to-end teammate onboarding and deployment steps, see [docs/deployment_init.md](docs/deployment_init.md).
For OpenAI-compatible LLM/VLM integration steps, see [docs/model_integration_openai_compatible.md](docs/model_integration_openai_compatible.md).

## Research-style evaluation

This project includes a reproducible evaluation protocol for retrieval, answer quality, and citation grounding so the system can be reported as a multimodal RAG architecture rather than only a chat demo.

Retrieval-only benchmark:
```bash
python scripts/12_run_benchmark_eval.py --suite retrieval_benchmark --datasets docvqa --splits val --limit-per-split 100
```

Full RAG benchmark (requires backend running):
```bash
python -m src api
python scripts/12_run_benchmark_eval.py --suite rag_benchmark --datasets docvqa --splits val --limit-per-split 100 --mode rag
```

Outputs are written to `outputs/eval/` as:
- sample-level `jsonl`
- machine-readable summary `json`
- report-ready summary `md`

See [docs/evaluation_protocol.md](docs/evaluation_protocol.md) for the full evaluation design and reporting guidance.
