# 数据与索引说明

## 目标

本阶段只要求离线脚本跑通 DocVQA / ChartQA 子集的数据处理、文档切分、索引构建和查询验证，不处理在线上传。数据链路以 `data/processed/*/*.jsonl` 为输入，输出统一文档 schema、chunk schema 和本地索引产物。

## 流程

1. 下载数据：

```bash
python scripts/01_download_datasets.py --from-hf --hf-official-layout
```

2. 标准化问答样本。若要让 DocVQA / ChartQA 图片成为本地可读文件，推荐从 `data/raw_hf` 解码导出：

```bash
python scripts/01_prepare_datasets.py prepare --mode eval --from-hf-cache --image-export-root data/images
```

只读 `data/raw/*.json` 也能生成问答 JSONL，但图片可能只是原始相对名或 `null`，不能保证后续 MLLM 能打开。

3. 文档解析与语义化切分：

```bash
python scripts/07_parse_and_chunk.py --datasets docvqa,chartqa --splits val,test
```

4. 构建索引：

```bash
python scripts/08_build_indexes.py
```

5. 查询验证：

```bash
python scripts/09_query_indexes.py "What is the actual value per 1000 during 1975?" --top-k 3
```

快速跑通一两个样本，并验证本地图片选择逻辑：

```bash
python scripts/10_run_data_index_pipeline.py --skip-download --prepare-from-hf-cache --limit-per-split 2 --run-mllm-smoke --mllm-dry-run --mllm-num-samples 2
```

真实调用阿里云百炼兼容接口前，需要在 `.env` 中配置：

```env
DASHSCOPE_API_KEY=your_key
```

然后去掉 `--mllm-dry-run`：

```bash
python scripts/10_run_data_index_pipeline.py --skip-download --prepare-from-hf-cache --limit-per-split 2 --run-mllm-smoke --mllm-num-samples 1
```

## 统一样本 Schema

标准化问答样本位于：

- `data/processed/docvqa/{val,test}.jsonl`
- `data/processed/chartqa/{val,test}.jsonl`

字段：

```json
{
  "id": "docvqa-val-0",
  "dataset": "docvqa",
  "split": "val",
  "question": "What is ...?",
  "answers": ["..."],
  "image": {"path": "optional.png", "bytes": null},
  "evidence": null,
  "metadata": {}
}
```

## Document Schema

输出文件：`data/processed/documents/documents.jsonl`

核心字段：

- `document_id`: 文档级稳定 ID
- `sample_id`: 原始标准化样本 ID
- `dataset` / `split`: 数据来源
- `source_type`: `image` 或 `benchmark_record`
- `image_path`: 页面图像路径，可为空
- `figure_id`: 图像/图表引用 ID
- `page_no`: 页码；缺少原始页码时默认 1
- `question` / `answers` / `evidence`: 保留评测样本信息
- `metadata`: 原始或派生元数据

## Chunk Schema

输出文件：`data/processed/chunks/chunks.jsonl`

核心字段：

- `chunk_id`: 块级稳定 ID
- `document_id` / `sample_id`
- `chunk_type`: `text`、`table`、`figure`、`formula`、`page_image`
- `text`: 可检索文本。当前 fallback 会使用问题和已有 evidence；接入 MinerU 后使用 MinerU block 文本
- `page_no`: 页码
- `bbox`: MinerU 坐标框；fallback 为空
- `source_ref`: 形如 `docvqa/val/docvqa-val-0#page=1#figure=fig-xxx`
- `image_path`: 关联图像路径

## MinerU 接入方式

配置项在 `configs/datasets.yaml`：

```yaml
document_chunking:
  mineru_json_root: data/interim/mineru
```

脚本会按以下路径查找 MinerU JSON：

- `data/interim/mineru/<sample_id>.json`
- `data/interim/mineru/<dataset>/<sample_id>.json`
- `data/interim/mineru/<image_stem>.json`
- `data/interim/mineru/<dataset>/<image_stem>.json`

支持的 block 字段较宽松：`type/category/block_type` 表示块类型，`text/content/html/latex/caption` 表示文本，`bbox` 和 `page_no` 会原样保留。

## 索引产物

索引目录：`data/processed/indexes/`

- `text/doc_store.json`: chunk 文档库，含词频
- `text/postings.json`: 倒排表
- `text/document_frequency.json`: 文档频率
- `vision/visual_store.json`: 图像、表格、页面图像的元数据索引
- `index_manifest.json`: 索引版本、构建时间、数据版本、输入输出路径和计数

当前文本索引是轻量本地 sparse 检索，方便无 GPU、无 embedding API 时跑通。后续可把 `visual_store.json` 替换或扩展为视觉 embedding 向量库，`chunk_id` 保持不变即可。

## RAG 中索引、Embedding、Retrieval 的关系

索引不是答案生成器，而是 retrieval 阶段用来快速找候选证据的数据结构。完整 RAG 可以理解为：

1. 文档解析：PDF / 页面图像 / 图表被切成 text/table/figure/formula/page_image chunks。
2. Embedding：把每个 chunk 的文本、表格说明或图片表示编码成向量。
3. Index：把 chunk_id、向量、文本倒排、页码、bbox、image_path、source_ref 等组织起来，方便快速搜索和回溯。
4. Retrieval：用户提问后，对问题做 embedding 或关键词检索，从 index 里取 top-k chunks。
5. Generation：把 top-k 的文本、页面图片或裁剪区域连同问题交给 MLLM，生成答案和引用。

当前 `08_build_indexes.py` 是轻量 sparse index，用来跑通 retrieval 外壳；后续做 embedding 时，不是废掉索引，而是把 `chunk_id -> vector` 加进索引，或接入 FAISS / Milvus / Chroma 等向量索引。

DocVQA 和 ChartQA 单张图确实不大，烟测阶段可以直接把图片和问题发给 MLLM，不一定非要先 embed 图片。这里建索引主要是为了把“数据样本、文档块、图片路径、页码/图号、后续 retrieval 回溯”这条链跑通。等全量评测或多页文档场景变大后，索引才会体现价值：不用把所有页面都塞给 MLLM，而是先找最可能相关的一两页或几个区域。

`index_manifest.json` 是索引清单，不参与回答本身。它记录 index 版本、构建时间、数据版本、输入文件、输出文件和数量，用来确认现在查的是哪一版数据，方便复现和排错。

任务说明里的“查询脚本”就是 `09_query_indexes.py` 这类脚本：给一个自然语言问题，从已建索引里找 top-k 候选 chunk，并输出 `source_ref/page_no/image_path/snippet`。在完整 RAG 里，它就是 retrieval 的命令行版本。

## 真实 MinerU 运行

MinerU 包装脚本：

```bash
python scripts/06_run_mineru.py --datasets docvqa --splits val --limit-per-split 1 --backend pipeline
```

完整流水线中启用 MinerU：

```bash
python scripts/10_run_data_index_pipeline.py --skip-download --prepare-from-hf-cache --limit-per-split 1 --run-mineru --run-mllm-smoke --mllm-num-samples 1
```

如果只想验证字段格式，不调用真实 MinerU：

```bash
python scripts/10_run_data_index_pipeline.py --skip-download --prepare-from-hf-cache --limit-per-split 1 --run-mineru --mineru-mock --run-mllm-smoke --mllm-dry-run --mllm-num-samples 1
```

真实 MinerU 首次运行会下载 `opendatalab/PDF-Extract-Kit-1.0` 等模型权重。若 Hugging Face 连接失败，需要提前准备模型缓存，或配置可访问的模型下载源/镜像。
