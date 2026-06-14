# 多模态文档语义切分与元数据构建技术报告

## 摘要

针对多模态文档问答系统中常见的语义丢失、布局结构缺失与长文档检索困难问题，本项目设计并实现了一套面向检索增强生成的文档解析、语义切分与元数据构建流程。该流程以 MinerU 解析结果中的语义块为核心索引单元，将文档页面中的文本、标题、表格、公式、图片等内容归一化为 block，再进一步转换为可检索的 chunk，并为每个 chunk 保留文档来源、页码、边界框、图像路径与原始解析信息等回溯元数据。相比直接对全文进行固定长度切分，该方法能够更好地保持语义单元完整性和版面定位能力，为后续双路 RAG、多模态检索、证据回溯与 MLLM 推理提供统一的数据基础。

## 1. 问题定义与设计目标

在文档理解任务中，传统纯文本抽取和固定窗口切分容易破坏标题、段落、表格、图像之间的语义关系。例如，一个表格标题可能被切到前一个窗口，表格主体被切到后一个窗口；页面图像或公式区域也可能因为缺少结构化元数据而无法在回答阶段回溯到原始位置。对于 DocVQA、ChartQA 或用户上传 PDF 等文档数据，若仅将问题、答案或页面文本直接写入向量库，系统虽然可以完成简单 smoke test，但并不符合真实 RAG 场景中“用户提问检索文档证据”的需求。

因此，本模块的目标不是索引评测集自带的 question/answer，而是围绕原始文档页面建立可回溯的知识单元。其核心目标包括：

1. 保持文档语义结构：以 MinerU 输出的布局块为最小语义单元，避免跨段落、跨表格、跨图文区域的硬切分。
2. 支持多模态内容：统一表达文本、标题、表格、公式、图片和整页图像等不同模态。
3. 支持证据回溯：为每个 chunk 保存 `document_id`、`block_id`、`page_no`、`bbox`、`source_path`、`image_path` 等字段。
4. 支持后续索引替换：当前实现构建轻量 sparse index 和视觉元数据索引，但 schema 可直接迁移到 FAISS、Milvus、Chroma 或 Elasticsearch dense vector 后端。

## 2. 总体流程

本模块对应项目中的数据解析与索引前处理链路，主要由以下脚本完成：

- `scripts/06_run_mineru.py`：调用 MinerU 对 PDF 或页面图片进行解析，并输出项目归一化后的 MinerU JSON。
- `scripts/07_parse_and_chunk.py`：将评测样本或 MinerU JSON 转换为 `documents.jsonl` 与 `chunks.jsonl`。
- `scripts/08_build_indexes.py`：基于 chunk 构建本地文本检索索引和视觉元数据索引。
- `scripts/10_run_data_index_pipeline.py`：串联下载、标准化、MinerU 解析、切分与索引构建步骤。

完整数据流如下：

```text
PDF / 页面图片 / 评测样本
-> MinerU 解析得到版面结构 blocks
-> 归一化为统一 block schema
-> 生成 documents.jsonl 与 chunks.jsonl
-> 以 chunk_id 为主键构建文本索引与视觉元数据索引
-> 查询命中 chunk_id
-> 根据 metadata 回溯原文档、页码、bbox、图片区域
-> 将证据文本或图像区域交给 MLLM 推理
```

该流程将“文档解析”和“检索索引”解耦：上游只需保证 chunk schema 稳定，下游无论采用 sparse retrieval 还是 dense embedding retrieval，都可以复用同一套回溯字段。

## 3. 文档解析与语义化切分方法

### 3.1 MinerU 语义块解析

本项目使用 MinerU 作为文档结构解析工具。MinerU 根据页面布局、文本区域、表格区域、公式区域、图像区域等信息，将页面解析为多个 block。项目对 MinerU 原始输出进行归一化，形成统一字段：

```json
{
  "block_id": "block-0000",
  "type": "text",
  "text": "Visual Spatial Tuning",
  "bbox": [349, 127, 647, 154],
  "page_no": 1,
  "image_path": null,
  "metadata": {
    "raw_index": 0,
    "raw_type": "text"
  }
}
```

其中，`type` 会被标准化为 `text`、`title`、`table`、`figure`、`formula` 等类型；`bbox` 保存页面坐标框；`page_no` 使用 1-based 页码；`image_path` 用于记录图像块或页面图像资源；`metadata` 保留 MinerU 原始顺序与原始类别。若 MinerU 未提供稳定 ID，系统会根据文档 ID 与 block 序号生成稳定哈希 ID，保证同一输入在重复构建时具有一致的引用路径。

### 3.2 block 到 chunk 的转换

在 `scripts/07_parse_and_chunk.py` 中，函数 `_append_block_chunks` 负责将 MinerU blocks 转换为检索 chunks。转换逻辑遵循两个原则：

1. 一个 MinerU block 至少对应一个 chunk。这样可以让检索单元天然对齐文档版面结构。
2. 当 block 文本过长时，才进行二次文本切分。切分后的多个 part 保留相同 `block_id`，并通过 `part_index` 区分。

当前配置位于 `configs/datasets.yaml`：

```yaml
document_chunking:
  document_output: data/processed/documents/documents.jsonl
  chunk_output: data/processed/chunks/chunks.jsonl
  mineru_json_root: data/interim/mineru
  default_page_no: 1
  max_chars_per_chunk: 900
  overlap_chars: 120
```

也就是说，单个 block 文本超过 900 字符时，会按 900 字符窗口、120 字符重叠进行 part 切分。该策略不同于对全文直接滑窗：它只在语义块内部处理超长文本，因此能够最大限度避免跨标题、跨段落、跨表格的语义混杂。

### 3.3 多模态块处理

对于 `figure`、`table`、`formula` 等视觉或半结构化块，即使文本为空，系统也会生成占位文本，例如 `figure region` 或 `table region`，使其能够作为独立 chunk 写入索引。同时，相关块会在 `image_path`、`bbox` 和 `chunk_type` 字段中保留视觉定位信息。后续视觉检索或 MLLM 推理可以根据这些字段恢复页面区域，或将图片块直接作为输入证据。

评测集中的 question 和 answers 默认不会进入检索库。`scripts/07_parse_and_chunk.py` 仅在显式指定 `--include-qa-context` 时才生成 `question_context` 类型的 synthetic chunk，该选项只用于早期调试，不作为生产 RAG 的默认逻辑。

## 4. 元数据结构设计

### 4.1 documents.jsonl

`documents.jsonl` 是文档级清单，一行对应一个输入文档、页面图片或评测样本。它不作为 embedding 的基本单元，而是保存父级信息和评测相关字段。核心字段包括：

| 字段 | 含义 |
| --- | --- |
| `document_id` | 文档级稳定 ID |
| `sample_id` | 评测样本或输入文件样本 ID |
| `dataset` | 数据来源，如 `docvqa`、`chartqa`、`raw_pdf` |
| `split` | 数据划分，如 `val`、`test`、`raw` |
| `source_type` | 来源类型，如 `pdf`、`image`、`benchmark_record` |
| `source_path` | 原始 PDF 或页面图片路径 |
| `image_path` | 可直接用于 MLLM 输入的页面图片路径 |
| `page_no` / `page_count` | 文档页码与页数信息 |
| `question` / `answers` | 评测字段，仅用于 evaluation，不默认索引 |
| `metadata` | 解析器、原始输出目录、数据集原始字段等扩展信息 |

该设计将“评测样本信息”和“文档证据单元”分开，避免把答案泄露进检索库，同时保留评测所需的 query 与 label。

### 4.2 chunks.jsonl

`chunks.jsonl` 是后续 embedding 与检索的核心数据表。每一行代表一个可检索证据单元，其字段如下：

| 字段 | 含义 |
| --- | --- |
| `chunk_id` | 向量库或索引库主键 |
| `document_id` | 父文档 ID |
| `sample_id` | 样本 ID |
| `dataset` / `split` | 数据来源与划分 |
| `block_id` | 对应 MinerU block ID |
| `block_index` | block 在 MinerU 输出中的顺序 |
| `part_index` | 长文本 block 的 part 序号 |
| `chunk_type` | `text`、`title`、`table`、`figure`、`formula`、`page_image` 等 |
| `text` | 文本检索和文本 embedding 输入 |
| `page_no` | 原文档页码 |
| `bbox` | 页面坐标框，用于证据定位或裁剪 |
| `source_ref` | 人可读引用，如 `raw_pdf/raw/aaav1#page=1#block=block-0000` |
| `source_path` | 原始 PDF 或图片路径 |
| `image_path` | 图片块或页面图片路径 |
| `metadata` | parser、raw_index、raw_type 等扩展字段 |

该 schema 的关键是 `chunk_id + metadata`。检索时，向量数据库只需返回 `chunk_id`，系统即可根据 `document_id`、`page_no`、`bbox` 和路径字段恢复证据上下文。

## 5. 索引构建与元数据落盘

`scripts/08_build_indexes.py` 读取 `documents.jsonl` 与 `chunks.jsonl`，构建两个索引产物。

第一类是文本索引，输出到 `data/processed/indexes/text/`：

- `doc_store.json`：以 `chunk_id` 为键，保存 chunk 全量信息、词频 `tf` 和 `token_count`。
- `postings.json`：倒排表，格式为 `token -> [{chunk_id, tf}]`。
- `document_frequency.json`：记录每个 token 出现在多少 chunk 中。

第二类是视觉元数据索引，输出到 `data/processed/indexes/vision/visual_store.json`。当 chunk 类型为 `figure`、`page_image`、`table`、`formula`，或存在 `image_path` 时，会被加入视觉索引。视觉索引不直接保存图像 embedding，而是保存可回溯字段，包括 `chunk_id`、`document_id`、`image_path`、`page_no`、`bbox`、`caption` 等。

此外，系统会生成 `data/processed/indexes/index_manifest.json`，记录索引版本、构建时间、输入输出路径、文档数量、chunk 数量、文本词项数和视觉块数量。该 manifest 可用于实验复现和索引一致性检查。

## 6. 查询与回溯机制

当前项目提供的是轻量级 sparse 检索实现，主要用于离线可运行验证。查询阶段读取 `doc_store.json`，对用户 query 分词后计算 TF-IDF cosine，相似度最高的 chunk 被返回。虽然当前实现不是 dense embedding index，但其回溯逻辑与真实向量检索一致：

```text
query
-> 检索命中 chunk_id
-> 读取 chunk metadata
-> document_id 定位父文档
-> source_path 打开原始 PDF / 页面图片
-> page_no 定位页
-> bbox 定位页面区域
-> text 或 image_path 作为 MLLM 上下文
```

因此，后续将 `08_build_indexes.py` 替换为多模态 embedding 构建时，不需要改变切分结果格式。新的向量库只需以 `chunk_id` 为主键保存文本向量、图像向量或表格向量，并保留同样的 metadata 字段。

## 7. 与双路 RAG 的衔接

该模块为双路 RAG 与晚融合提供数据基础。文本路可以使用 `chunk_type in {text,title,table,formula}` 的 `text` 字段构建文本 embedding；视觉路可以使用 `figure`、`page_image`、`table` 等 chunk 的 `image_path` 或 `bbox` 裁剪区域构建视觉 embedding。两路检索均返回 `chunk_id`，再在融合阶段根据统一 metadata 进行去重、排序和证据组织。

这种设计的优势在于，不同模态的检索结果拥有同一套引用体系。即使文本路命中的是表格 caption，视觉路命中的是表格区域图像，系统也可以通过 `document_id`、`page_no` 和 `bbox` 判断它们是否来自同一证据区域，并在 MLLM 推理时同时提供文本与图像上下文。

## 8. 质量控制与可扩展性

本模块采用以下机制提升可靠性：

1. 稳定 ID：`stable_id` 使用 SHA-1 截断哈希生成稳定的 `document_id`、`block_id` 和 `chunk_id`，减少重复构建时的 ID 漂移。
2. 解析器保真：`metadata.raw_index` 与 `metadata.raw_type` 保存 MinerU 原始顺序和类别，方便排查解析错误。
3. 语义块优先：只有在 block 内部超过长度阈值时才做二次切分，降低硬切分导致的上下文破坏。
4. 评测防泄露：question/answers 默认不写入索引，避免评测答案被直接检索。
5. 后端可替换：当前 sparse index 只承担离线验证职责，未来可平滑替换为 dense vector index。

## 9. 小结

本项目的数据切分与元数据构建模块以 MinerU 语义块为核心，将多模态文档解析结果转换为统一的 documents/chunks schema。该方法在切分阶段保留文档布局与语义单元，在元数据阶段保留页码、边界框和原始路径，在索引阶段以 `chunk_id` 为主键连接文本检索、视觉检索和证据回溯。由此，系统能够避免纯文本硬切分带来的语义丢失，并为后续多模态 RAG、晚融合和 MLLM 文档问答提供稳定、可扩展的数据底座。
