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


## DocVQA 本地数据流详解

下面用当前本地样本 `questionId=49153` 说明 DocVQA 如何一步步进入 RAG 索引。这个样本的问题是：`What is the ‘actual’ value per 1000, during the year 1975?`，标准答案是 `0.28`。

### 1. 下载后的本地结构

执行：

```bash
python scripts/01_download_datasets.py --from-hf --hf-official-layout
```

会产生两层数据：

- `data/raw_hf/docvqa/`: Hugging Face 官方 Arrow 结构，保留完整 `datasets.Image` 信息。
- `data/raw/docvqa/`: 项目交换层 JSON，便于查看和版本沟通。

DocVQA 官方层典型路径：

```text
data/raw_hf/docvqa/dataset_dict.json
data/raw_hf/docvqa/validation/data-00000-of-00008.arrow
data/raw_hf/docvqa/test/data-00000-of-00008.arrow
```

项目交换层典型路径：

```text
data/raw/docvqa/val.json
data/raw/docvqa/test.json
```

`data/raw/docvqa/val.json` 中一条样本的关键字段如下：

```json
{
  "questionId": "49153",
  "question": "What is the ‘actual’ value per 1000, during the year 1975?",
  "question_types": ["figure/diagram"],
  "image": {"bytes": null, "path": "pybv0228_81.png"},
  "docId": 14465,
  "ucsf_document_id": "pybv0228",
  "ucsf_document_page_no": "81",
  "answers": ["0.28"],
  "data_split": "val"
}
```

字段含义：

- `questionId`: 原始问题 ID，后续作为稳定 `id/sample_id` 使用。
- `question`: 问题文本。
- `answers`: 标准答案列表。
- `image`: 从 HF 导出的图片引用。普通 JSON 层通常不会保存图片 bytes。
- `docId`: DocVQA 原始文档编号。
- `ucsf_document_id`: 原文档编号。
- `ucsf_document_page_no`: 原始页码，后续写入 `page_no` 和引用。
- `question_types`: 问题类型，例如图表/布局/表单等。

注意：`data/raw/*.json` 里的 `image.path` 可能只是原始文件名，不保证本地存在图片文件。真正可靠的图片来源是 `data/raw_hf` 中的 Arrow 图像列。

### 2. 标准化与图片导出

推荐执行：

```bash
python scripts/01_prepare_datasets.py prepare --mode eval --datasets docvqa --from-hf-cache --image-export-root data/images
```

输出：

```text
data/processed/docvqa/val.jsonl
data/processed/docvqa/test.jsonl
data/images/docvqa/val/49153.png
```

脚本功能：

- 从 `data/raw_hf/docvqa` 读取 HF Arrow 数据。
- 解码 `image` 列并保存为真实本地图片。
- 将不同数据集字段统一成 `id/question/answers/image/metadata`。
- 保留 DocVQA 原始元数据，如 `questionId/docId/ucsf_document_page_no`。

重要参数：

- `prepare`: 执行标准化。
- `--mode eval`: 只处理 `val/test`，适合当前 DocVQA 无 train 的情况。
- `--datasets docvqa`: 只处理 DocVQA；也可写 `docvqa,chartqa`。
- `--from-hf-cache`: 从 `data/raw_hf` 读取并导出图片。
- `--image-export-root data/images`: 图片导出根目录。
- `--limit-per-split 2`: 每个 split 只处理 2 条，用于烟测。

标准化后一行 JSONL 示例：

```json
{
  "id": "49153",
  "dataset": "docvqa",
  "split": "val",
  "question": "What is the ‘actual’ value per 1000, during the year 1975?",
  "answers": ["0.28"],
  "image": "/home/wuchenghui/MLLM-FinalProject/data/images/docvqa/val/49153.png",
  "evidence": null,
  "metadata": {
    "questionId": "49153",
    "docId": 14465,
    "ucsf_document_id": "pybv0228",
    "ucsf_document_page_no": "81",
    "question_types": ["figure/diagram"],
    "data_split": "val",
    "raw_keys": ["..."]
  }
}
```

这里开始，`image` 已经是可被 MLLM 打开的本地图片绝对路径。

### 3. MinerU 解析

首次真实运行 MinerU 前，先下载模型：

```bash
python scripts/00_download_mineru_models.py
```

它使用 ModelScope SDK 下载官方模型：`OpenDataLab/PDF-Extract-Kit-1.0`。模型默认进入 ModelScope 缓存目录，不会提交到 Git。

对 DocVQA 样本运行 MinerU：

```bash
python scripts/06_run_mineru.py --datasets docvqa --splits val --limit-per-split 1 --backend pipeline --mineru-model-source modelscope
```

输出：

```text
data/interim/mineru_raw/docvqa/val/49153/     # MinerU 原始输出目录
data/interim/mineru/docvqa/49153.json        # 项目归一化后的 MinerU block JSON
```

脚本功能：

- 从 `data/processed/docvqa/val.jsonl` 读取样本。
- 找到样本中的本地图片路径。
- 调用 `mineru -p <image> -o <out> -b pipeline`。
- 将 MinerU 原始输出中的 JSON/Markdown 归一成统一 `blocks`。

重要参数：

- `--datasets docvqa`: 数据集名。
- `--splits val`: split 名。
- `--limit-per-split 1`: 每个 split 最多跑 1 条，避免全量解析耗时过长。
- `--backend pipeline`: 使用 MinerU pipeline 后端。
- `--mineru-model-source modelscope`: 使用 ModelScope 缓存/下载源。
- `--mock`: 不调用 MinerU，只生成兼容格式的模拟 blocks，用于快速验证字段链路。

归一化后的 MinerU JSON 结构：

```json
{
  "sample_id": "49153",
  "dataset": "docvqa",
  "split": "val",
  "source_path": "/home/.../data/images/docvqa/val/49153.png",
  "raw_output_dir": "/home/.../data/interim/mineru_raw/docvqa/val/49153",
  "raw_files": ["..."],
  "blocks": [
    {
      "type": "figure",
      "text": "figure region",
      "bbox": [118, 226, 870, 858],
      "page_no": 1,
      "metadata": {"raw_index": 0, "raw_type": "image"}
    }
  ]
}
```

`blocks[*]` 字段含义：

- `type`: 语义块类型，归一为 `text/table/figure/formula/title` 等。
- `text`: 块文本；图像块可能是 `figure region`。
- `bbox`: MinerU 给出的版面坐标，用于后续定位/裁剪。
- `page_no`: MinerU 解析页码。对于单张图片通常为 1。
- `metadata`: 原始 block 索引和类型等补充信息。

### 4. 文档记录与 chunk 切分

执行：

```bash
python scripts/07_parse_and_chunk.py --datasets docvqa --splits val --limit-per-split 1
```

输入：

```text
data/processed/docvqa/val.jsonl
data/interim/mineru/docvqa/49153.json
```

输出：

```text
data/processed/documents/documents.jsonl
data/processed/chunks/chunks.jsonl
```

`documents.jsonl` 一条记录表示一个 QA 样本对应的文档/页面：

```json
{
  "document_id": "doc-ade4608d54ffa17c",
  "sample_id": "49153",
  "dataset": "docvqa",
  "split": "val",
  "source_type": "image",
  "image_path": "/home/.../data/images/docvqa/val/49153.png",
  "figure_id": "fig-49153",
  "page_no": 81,
  "question": "What is the ‘actual’ value per 1000, during the year 1975?",
  "answers": ["0.28"],
  "evidence": "",
  "metadata": {"ucsf_document_page_no": "81", "...": "..."}
}
```

关键点：

- `page_no=81` 来自 DocVQA 原始 `ucsf_document_page_no`，用于引用。
- `figure_id=fig-49153` 来自图片文件名。
- `image_path` 指向本地真实图片。

`chunks.jsonl` 是检索单元。真实 MinerU 情况下会同时产生两类 chunk：

1. `question_context` chunk：保留问题和标准答案，保证 QA 检索链路可命中。
2. MinerU block chunk：保留 `figure/table/text` 等布局块和 bbox。

示例：

```json
{
  "chunk_id": "chk-00e797e469ea807c",
  "document_id": "doc-ade4608d54ffa17c",
  "sample_id": "49153",
  "dataset": "docvqa",
  "split": "val",
  "chunk_type": "question_context",
  "text": "Question: What is the ‘actual’ value per 1000, during the year 1975? Expected answer: 0.28",
  "page_no": 81,
  "bbox": null,
  "source_ref": "docvqa/val/49153#page=81#figure=fig-49153",
  "image_path": "/home/.../data/images/docvqa/val/49153.png",
  "metadata": {"parser": "mineru", "synthetic": true}
}
```

```json
{
  "chunk_id": "chk-a0c2da41b9a4bc60",
  "document_id": "doc-ade4608d54ffa17c",
  "sample_id": "49153",
  "dataset": "docvqa",
  "split": "val",
  "chunk_type": "figure",
  "text": "figure region",
  "page_no": 1,
  "bbox": [118, 226, 870, 858],
  "source_ref": "docvqa/val/49153#page=81#figure=fig-49153",
  "image_path": "/home/.../data/images/docvqa/val/49153.png",
  "metadata": {"parser": "mineru", "block_index": 0}
}
```

### 5. 构建索引

执行：

```bash
python scripts/08_build_indexes.py
```

输入：

```text
data/processed/documents/documents.jsonl
data/processed/chunks/chunks.jsonl
```

输出：

```text
data/processed/indexes/text/doc_store.json
data/processed/indexes/text/postings.json
data/processed/indexes/text/document_frequency.json
data/processed/indexes/vision/visual_store.json
data/processed/indexes/index_manifest.json
```

脚本功能：

- 对 `chunks.jsonl` 中每个 `text` 分词。
- 计算 chunk 内词频 `tf`。
- 建立倒排表 `postings`: token -> chunk 列表。
- 保存 chunk 全量信息到 `doc_store.json`，便于查询后回溯。
- 把 `figure/page_image/table` 或带 `image_path` 的 chunk 写入 `visual_store.json`。
- 写 `index_manifest.json` 记录索引版本、构建时间、输入输出路径和数量。

`doc_store.json` 中每个 chunk 会保留：

```json
{
  "chunk_id": "chk-...",
  "document_id": "doc-...",
  "sample_id": "49153",
  "dataset": "docvqa",
  "split": "val",
  "chunk_type": "question_context",
  "text": "Question: ... Expected answer: 0.28",
  "page_no": 81,
  "bbox": null,
  "source_ref": "docvqa/val/49153#page=81#figure=fig-49153",
  "image_path": "/home/.../49153.png",
  "tf": {"question": 1, "actual": 1, "1975": 1},
  "token_count": 13
}
```

`visual_store.json` 中每个视觉块保留：

```json
{
  "chunk_id": "chk-...",
  "document_id": "doc-...",
  "sample_id": "49153",
  "dataset": "docvqa",
  "split": "val",
  "chunk_type": "figure",
  "image_path": "/home/.../49153.png",
  "page_no": 1,
  "bbox": [118, 226, 870, 858],
  "source_ref": "docvqa/val/49153#page=81#figure=fig-49153",
  "caption": "figure region"
}
```

`index_manifest.json` 用于复现和排错，不参与模型回答。它记录：

- `index_version`: 当前索引实现版本。
- `built_at`: 构建时间。
- `dataset_version`: 基于当前 documents/chunks 生成的数据版本摘要。
- `inputs`: 使用了哪些输入文件。
- `outputs`: 生成了哪些索引文件。
- `counts`: documents/chunks/terms/visual_items 数量。

### 6. 查询和回溯

执行：

```bash
python scripts/09_query_indexes.py "actual value per 1000 during 1975" --top-k 2
```

脚本功能：

- 读取 `text/doc_store.json` 和 `document_frequency.json`。
- 对问题分词。
- 用稀疏 TF-IDF cosine 计算 query 与每个 chunk 的相似度。
- 返回 top-k chunk，并输出 `source_ref/snippet/chunk_type`。

典型输出：

```text
1. score=0.57735 docvqa/val/49153#page=81#figure=fig-49153 (question_context)
   snippet: Question: What is the ‘actual’ value per 1000, during the year 1975? Expected answer: 0.28
```

回溯链路是：

```text
query
-> 命中 chunk_id
-> chunk.document_id 找 document
-> chunk.source_ref 给出页码/图号引用
-> chunk.image_path 找本地图片
-> chunk.bbox 定位 MinerU 版面区域
-> sample_id 回到 data/processed/docvqa/val.jsonl 原始 QA 样本
```

### 7. 一键流水线

最小真实 DocVQA + MinerU + MLLM dry-run：

```bash
python scripts/10_run_data_index_pipeline.py \
  --skip-download \
  --prepare-from-hf-cache \
  --datasets docvqa \
  --splits val \
  --limit-per-split 1 \
  --run-mineru \
  --mineru-model-source modelscope \
  --run-mllm-smoke \
  --mllm-dry-run \
  --mllm-num-samples 1
```

参数含义：

- `--skip-download`: 跳过下载，复用已有 `data/raw_hf`。
- `--prepare-from-hf-cache`: 从 HF Arrow 解码图片并生成 processed JSONL。
- `--datasets docvqa`: 只跑 DocVQA。
- `--splits val`: 只跑验证集。
- `--limit-per-split 1`: 每个 split 只取 1 条。
- `--run-mineru`: 在切分前调用 MinerU。
- `--mineru-model-source modelscope`: 使用 ModelScope 模型源。
- `--run-mllm-smoke`: 抽样调用 MLLM 验证图片问答链路。
- `--mllm-dry-run`: 只选择样本，不真实调用 API。
- `--mllm-num-samples 1`: 抽 1 条样本。

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

真实 MinerU 首次运行会下载 `OpenDataLab/PDF-Extract-Kit-1.0` 等模型权重。国内环境推荐先用 ModelScope 下载：

```bash
python scripts/00_download_mineru_models.py
```

运行 MinerU 时默认会设置 `MINERU_MODEL_SOURCE=modelscope`：

```bash
python scripts/06_run_mineru.py --datasets docvqa --splits val --limit-per-split 1 --backend pipeline --mineru-model-source modelscope
```

也可以在 `.env` 或 shell 环境中设置 `MINERU_MODEL_SOURCE=modelscope`。
