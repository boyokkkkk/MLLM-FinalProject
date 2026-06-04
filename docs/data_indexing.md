# 数据与索引说明

## 设计澄清

你前面的理解是对的：真实 RAG 场景里，被 embed、被写入向量数据库的核心对象应该是 MinerU 解析出来的 block，而不是 DocVQA / ChartQA 自带的 question。

完整链路是：

```text
用户输入 PDF / 页面图片
-> MinerU 解析，得到归一化 blocks
-> 对 text/table/formula/figure 等 block 分别做 embedding
-> 向量库按 chunk_id 保存向量和回溯元数据
-> 用户提问作为 query 做 embedding
-> retrieval 命中若干 block chunks
-> 根据 chunk metadata 回到原文档、页码、bbox、图片区域
-> 把命中的文本块、图片块或裁剪区域作为上下文交给 MLLM
```

DocVQA / ChartQA 只是评测集。它们的 `question` 对应真实系统里的“用户提问”，`answers` 是评测标签，`image` 对应真实系统里的“用户上传文档或页面图像”。因此，生产 RAG 默认不把 `question` / `answers` embed 进数据库。

项目中以前出现过 `question_context` chunk，是为了早期 smoke test 在没有 MinerU block 或没有 embedding 的情况下验证查询脚本能命中样本。现在它已经改成显式调试选项：只有运行 `scripts/07_parse_and_chunk.py --include-qa-context` 才会生成，默认不会生成。

## documents 和 chunks 分别做什么

`documents.jsonl` 是文档级清单，一行表示一个输入文档、页面图像或评测样本。它不直接作为 embedding 单元，主要负责记录父级信息：文档 ID、数据来源、原始文件路径、页数、评测问题与答案等。

`chunks.jsonl` 是检索与 embedding 单元。真实 MinerU 链路里，一个 MinerU block 至少对应一个 chunk；如果文本过长，会按 `max_chars_per_chunk` 切成多个 part，但每个 part 仍保留同一个 `block_id`，可以回到原始 block。

向量数据库后续应该以 `chunk_id` 为主键，保存：

- `embedding`: 文本向量或图片向量。
- `chunk_id`: 检索命中的唯一 ID。
- `document_id`: 回到父文档。
- `block_id`: 回到 MinerU block。
- `page_no`: 回到页码。
- `bbox`: 回到页面内坐标框。
- `source_path`: 回到原始 PDF / 图片。
- `image_path`: 对图片输入或 block 图片资源的引用。
- `source_ref`: 方便日志、人读和引用展示。

当前 `08_build_indexes.py` 构建的是轻量 sparse index 和视觉元数据索引，不是真正的 embedding index；但它使用同一套 `chunk_id + metadata` 结构，后续替换成 FAISS / Milvus / Chroma / Elasticsearch dense_vector 时不需要改变上游 schema。

## DocVQA 本地数据流

DocVQA 在 Hugging Face 上的样本大致包含：

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

下载后有两层本地结构：

```text
data/raw_hf/docvqa/                 # Hugging Face Arrow 官方结构，保留 datasets.Image 信息
data/raw/docvqa/{val,test}.json     # 项目交换层 JSON，便于查看字段，但图片可能只是 path 名
```

`data/raw/docvqa/val.json` 适合检查字段，但不一定能直接打开图片。真正可靠的图片来自 `data/raw_hf` 的 Arrow 图像列。

标准化并导出图片：

```bash
python scripts/01_prepare_datasets.py prepare \
  --mode eval \
  --datasets docvqa \
  --from-hf-cache \
  --image-export-root data/images
```

输出：

```text
data/processed/docvqa/val.jsonl
data/processed/docvqa/test.jsonl
data/images/docvqa/val/49153.png
```

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
    "data_split": "val"
  }
}
```

字段含义：

- `id`: 稳定样本 ID，来自 DocVQA `questionId`。
- `question`: 评测时作为 query 使用，不默认进入数据库。
- `answers`: 标准答案，评测用，不默认进入数据库。
- `image`: 本地真实图片路径，对应 DocVQA 的文档页面。
- `metadata.ucsf_document_page_no`: 原始文档页码，可用于引用。

## MinerU 解析

首次运行前下载模型：

```bash
python scripts/00_download_mineru_models.py
```

它使用 ModelScope SDK 下载 `OpenDataLab/PDF-Extract-Kit-1.0`，模型默认在本机缓存目录，不提交 Git。

### 解析 DocVQA / ChartQA 样本

```bash
python scripts/06_run_mineru.py \
  --datasets docvqa \
  --splits val \
  --limit-per-split 1 \
  --backend pipeline \
  --mineru-model-source modelscope
```

脚本读取：

```text
data/processed/docvqa/val.jsonl
```

找到样本的 `image`，调用 MinerU，并输出：

```text
data/interim/mineru_raw/docvqa/val/49153/     # MinerU 原始输出目录
data/interim/mineru/docvqa/49153.json        # 项目归一化 MinerU JSON
```

### 解析普通用户 PDF

用户上传 PDF 的模拟输入可以直接传给脚本：

```bash
python scripts/06_run_mineru.py \
  --input-path data/raw/2511.05491v1.pdf \
  --datasets raw_pdf \
  --splits raw \
  --limit-per-split 1 \
  --backend pipeline \
  --mineru-model-source modelscope \
  --start-page 0 \
  --end-page 0
```

参数说明：

- `--input-path`: 直接处理某个 PDF / 图片文件或目录；设置后不读取 processed QA JSONL。
- `--datasets raw_pdf`: 给这批输入起一个逻辑数据源名。
- `--splits raw`: 给这批输入起一个 split 名。
- `--start-page 0 --end-page 0`: 只处理 PDF 第 1 页，适合烟测。全量处理时去掉这两个参数。
- `--backend pipeline`: MinerU pipeline 后端。
- `--mineru-model-source modelscope`: 使用 ModelScope 模型源。
- `--mock`: 不调用真实 MinerU，只生成模拟 blocks。

归一化 MinerU JSON 示例：

```json
{
  "document_id": "doc-11394522457e1392",
  "sample_id": "2511.05491v1",
  "dataset": "raw_pdf",
  "split": "raw",
  "source_path": "/home/wuchenghui/MLLM-FinalProject/data/raw/2511.05491v1.pdf",
  "source_type": "pdf",
  "raw_output_dir": "/home/.../data/interim/mineru_raw/raw_pdf/raw/2511.05491v1",
  "raw_files": ["/home/.../2511.05491v1_content_list.json"],
  "metadata": {"input_mode": "raw_document", "source_suffix": ".pdf"},
  "blocks": [
    {
      "block_id": "block-0000",
      "type": "text",
      "text": "Visual Spatial Tuning",
      "bbox": [349, 127, 647, 154],
      "page_no": 1,
      "image_path": null,
      "metadata": {"raw_index": 0, "raw_type": "text"}
    }
  ]
}
```

`blocks[*]` 字段含义：

- `block_id`: block 级稳定 ID。若 MinerU 没给 ID，脚本按顺序生成 `block-0000`。
- `type`: 归一化块类型，常见为 `text/table/figure/formula/title`。
- `text`: 块文本；图片块可能是 `figure region` 或 caption。
- `bbox`: 页面坐标框，用于定位或裁剪。
- `page_no`: 1-based 页码。
- `image_path`: block 自带图片资源路径；没有则为 `null`。
- `metadata.raw_index`: MinerU 原始 block 顺序。
- `metadata.raw_type`: MinerU 原始类型。

## 文档记录与 block 切分

### 从评测样本生成 documents/chunks

```bash
python scripts/07_parse_and_chunk.py \
  --source-mode benchmark \
  --datasets docvqa,chartqa \
  --splits val,test
```

输入：

```text
data/processed/docvqa/val.jsonl
data/processed/chartqa/val.jsonl
data/interim/mineru/docvqa/*.json
data/interim/mineru/chartqa/*.json
```

默认行为：如果某个样本有 MinerU blocks，只把 blocks 写成 chunks；不写 `question_context`。如果确实要早期调试用的 synthetic 问题 chunk，需要显式加：

```bash
python scripts/07_parse_and_chunk.py --source-mode benchmark --include-qa-context
```

### 从普通 PDF 的 MinerU JSON 生成 documents/chunks

```bash
python scripts/07_parse_and_chunk.py \
  --source-mode mineru \
  --datasets raw_pdf \
  --splits raw \
  --limit-per-split 1
```

输入：

```text
data/interim/mineru/raw_pdf/2511.05491v1.json
```

输出：

```text
data/processed/documents/documents.jsonl
data/processed/chunks/chunks.jsonl
```

`documents.jsonl` 示例：

```json
{
  "document_id": "doc-11394522457e1392",
  "sample_id": "2511.05491v1",
  "dataset": "raw_pdf",
  "split": "raw",
  "source_type": "pdf",
  "source_path": "/home/wuchenghui/MLLM-FinalProject/data/raw/2511.05491v1.pdf",
  "image_path": null,
  "page_no": 1,
  "page_count": 1,
  "question": null,
  "answers": [],
  "metadata": {
    "parser": "mineru",
    "raw_output_dir": "/home/.../data/interim/mineru_raw/raw_pdf/raw/2511.05491v1",
    "source_name": "2511.05491v1.pdf"
  }
}
```

`chunks.jsonl` 示例：

```json
{
  "chunk_id": "chk-d486207acde6455f",
  "document_id": "doc-11394522457e1392",
  "sample_id": "2511.05491v1",
  "dataset": "raw_pdf",
  "split": "raw",
  "block_id": "block-0000",
  "block_index": 0,
  "part_index": 0,
  "chunk_type": "text",
  "text": "Visual Spatial Tuning",
  "page_no": 1,
  "bbox": [349, 127, 647, 154],
  "source_ref": "raw_pdf/raw/2511.05491v1#page=1#block=block-0000",
  "source_path": "/home/wuchenghui/MLLM-FinalProject/data/raw/2511.05491v1.pdf",
  "image_path": null,
  "metadata": {"parser": "mineru", "block_index": 0, "raw_index": 0, "raw_type": "text"}
}
```

字段解释：

- `chunk_id`: 向量库主键。
- `document_id`: 父文档 ID。
- `block_id`: MinerU block ID；一个 block 被切成多个 part 时保持不变。
- `block_index`: block 在 MinerU 输出中的顺序。
- `part_index`: 长文本切分后的 part 序号。
- `chunk_type`: 检索单元类型。
- `text`: 文本 embedding 的输入；图片块可以放 caption 或占位说明。
- `page_no`: 原文档页码。
- `bbox`: 页面内定位框。
- `source_ref`: 人可读引用，包含 dataset/split/sample/page/block。
- `source_path`: 原 PDF / 原图片路径。
- `image_path`: 可直接交给 MLLM 的图片路径；PDF 文本块通常为空，页面图片或图片块可填。
- `metadata`: parser、原始类型、原始顺序等扩展信息。

## 构建索引

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

索引产物含义：

- `text/doc_store.json`: `chunk_id -> chunk 全量信息 + tf`，包括 `block_id/page_no/bbox/source_path/image_path`。
- `text/postings.json`: `token -> [{chunk_id, tf}]`，本地 sparse 检索用。
- `text/document_frequency.json`: token 出现在多少 chunk 中。
- `vision/visual_store.json`: `figure/table/formula/page_image` 等视觉块的元数据清单。
- `index_manifest.json`: 构建时间、输入输出路径、数量、版本说明。

当前 index 是“可运行的离线检索外壳”。后续做 embedding 时，可以把：

```text
chunk_id -> embedding vector
```

写入向量数据库，同时把 `doc_store.json` 里的回溯字段作为 metadata 存进去。retrieval 命中向量后，再用 `chunk_id` 找回 `source_path/page_no/bbox/text/image_path`。

## 查询与回溯逻辑

本地 sparse 查询脚本：

```bash
python scripts/09_query_indexes.py "Visual Spatial Tuning" --top-k 3
```

它做的事情：

1. 读取 `text/doc_store.json`。
2. 对 query 分词。
3. 用 TF-IDF cosine 与每个 chunk 的 `text` 做相似度。
4. 返回 top-k chunk。
5. 打印 `source_ref/chunk_type/snippet`。

真实向量检索时只是第 2-4 步换成 embedding similarity，回溯路径不变：

```text
query
-> retrieval 命中 chunk_id
-> 读取 chunk metadata
-> document_id 定位父文档
-> source_path 打开原 PDF / 图片
-> page_no 定位页
-> bbox 定位页面区域
-> text/image_path 作为 MLLM 上下文
```

## 已验证的 raw PDF 烟测

本地已用 `data/raw/2511.05491v1.pdf` 跑通第 1 页：

```bash
python scripts/06_run_mineru.py \
  --input-path data/raw/2511.05491v1.pdf \
  --datasets raw_pdf \
  --splits raw \
  --limit-per-split 1 \
  --backend pipeline \
  --mineru-model-source modelscope \
  --start-page 0 \
  --end-page 0

python scripts/07_parse_and_chunk.py --source-mode mineru --datasets raw_pdf --splits raw --limit-per-split 1
python scripts/08_build_indexes.py
```

验证结果：

```text
MinerU normalized blocks: 12
documents.jsonl: 1
chunks.jsonl: 14
index text_terms: 288
index visual_items: 0
```

`visual_items=0` 是因为该 PDF 第 1 页 MinerU 识别出的都是文本块。后续页面如果出现图片、表格或公式，`figure/table/formula/page_image` chunk 会进入 `vision/visual_store.json`。
