# 项目评测范式

本文定义本项目的科研化评测协议，用于支撑期末结题中的实验设计、结果汇报与架构优势说明。

## 1. 评测目标

本项目不是单纯验证“能否对话”，而是评估完整多模态 RAG 架构在以下三个层面的能力：

1. 检索能力：是否能从文档块中找回与问题对应的证据。
2. 回答能力：是否能给出与标准答案一致或近似的回答。
3. 引用能力：是否能返回正确、可追溯的 citation。

## 2. 评测对象

评测对象分为两类：

1. `retrieval` 模式
   只评估离线索引和检索模块，不调用生成模型。
2. `rag` 模式
   通过后端 `/api/v1/chat` 评估完整问答链路，包含检索、生成与 citation 返回。

## 3. 数据来源

默认使用：

1. `data/processed/docvqa/{val,test}.jsonl`
2. `data/processed/chartqa/{val,test}.jsonl`

每条样本视为一个“问题-文档页”评测单元。  
在当前协议中，若检索返回的 `source_ref/source` 前缀匹配 `dataset/split/sample_id`，则认为命中了正确样本来源。

## 4. 指标定义

### 4.1 检索指标

1. `hit_at_k`
   top-k 返回中是否至少包含一个正确来源块。
2. `recall_at_k`
   正确来源块在 top-k 中的召回率。当前基准默认每题 1 个主要相关来源，因此该值常等价于是否召回。
3. `precision_at_k`
   top-k 中相关块的占比。

### 4.2 回答指标

1. `exact_match`
   预测答案与任一标准答案归一化后完全一致。
2. `anls`
   采用 Approximate Normalized Levenshtein Similarity，适合 OCR/文档 QA 场景。

### 4.3 引用指标

1. `citation_accuracy`
   第 1 条 citation 是否指向正确样本来源。

## 5. 推荐实验设置

### 实验 A：检索能力基线

目的：
验证 block-first chunking 与本地索引是否能有效召回正确证据。

命令：

```bash
python scripts/12_run_benchmark_eval.py --suite retrieval_benchmark --datasets docvqa --splits val --limit-per-split 100
```

### 实验 B：完整 RAG 问答评测

目的：
验证系统在检索、生成、引用三个层面的整体表现。

前置：

```bash
python -m src api
```

命令：

```bash
python scripts/12_run_benchmark_eval.py --suite rag_benchmark --datasets docvqa --splits val --limit-per-split 100 --mode rag
```

### 实验 C：架构对比

建议至少比较以下版本：

1. 小样本或 fallback 版本索引
2. 真 MinerU block 版本索引
3. 不同 `top_k_text` / `score_threshold` 参数版本

结题时重点展示：

1. `hit_at_k` / `citation_accuracy` 的提升
2. `anls` / `exact_match` 的提升
3. 典型成功案例与失败案例

## 6. 输出文件

评测脚本运行后默认输出到：

```text
outputs/eval/
  <run_name>.jsonl
  <run_name>.summary.json
  <run_name>.summary.md
```

含义：

1. `jsonl`
   每条样本的详细预测与指标，适合误差分析。
2. `summary.json`
   机器可读汇总结果，适合后续画图或做表。
3. `summary.md`
   适合直接贴到实验记录或汇报材料。

## 7. 结题建议呈现方式

建议在结题报告中至少给出三类内容：

1. 总表
   展示 `hit_at_k`、`precision_at_k`、`exact_match`、`anls`、`citation_accuracy`。
2. 架构收益
   对比 mock/fallback 与真 MinerU block 版本，说明结构化解析和 block-first 索引为何有效。
3. 误差分析
   举例说明失败原因，例如：
   - 检索命中无关块
   - 多块证据融合不稳定
   - OCR 误差影响答案
   - citation 排序不够稳

## 8. 与 lm-evaluation-harness 的关系

本项目评测协议借鉴了 `lm-evaluation-harness` 的核心思想：

1. 固定评测集
2. 固定指标定义
3. 可复现命令
4. 结构化输出结果

但由于本项目是多模态文档 RAG 系统，而不是纯文本基础模型 Benchmark，因此我们采用项目定制协议，而非直接套用 MMLU/GSM8K 等标准语言模型任务。
