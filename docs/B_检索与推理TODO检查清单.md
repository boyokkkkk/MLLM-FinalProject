# B：检索与推理 TODO 检查清单

（陈宝怡）本清单用于落实 B 的开发任务，覆盖设计、实现、联调、验收四个层面。
范围以 `src/serving/`、`src/models/`、`configs/retrieval.yaml` 为主，并依赖已确认的 `chunk schema`。

## 1. 当前目标

- `/api/v1/chat` 接入 `query -> retrieve -> generate` 的 text-retrieval 最小闭环
- 返回真实 citation 基础字段：`chunk_id`、`page`、`source`
- 检索与生成关键参数配置化

## 1.1 最新开发进度（2026-06-03）

- [X]  已完成 `/api/v1/chat` 的 text-retrieval 最小闭环，基于本地 mock 索引可稳定返回 `answer + citations`
- [X]  已完成真实 citation 字段返回：`chunk_id`、`source`、`page`、`snippet`，并保留 `source_ref` 兼容前端
- [X]  已完成 retrieval / generation 配置化：`top_k_text`、`score_threshold`、`index_path`、`metadata_path`、`context_max_chars`、`default_temperature`、`default_max_tokens`
- [X]  已完成本地 mock 数据构建与 smoke test
  - [X]  mock metadata：`data/processed/retrieval/text_chunks.jsonl`
  - [X]  mock vectors：`data/processed/retrieval/text_vectors.npy`
  - [X]  smoke 脚本：`scripts/08_smoke_test_text_retrieval.py`
- [X]  当前前端已可联调测试“已进入 mock 索引的 3 个根目录文档”
- [ ]  当前仍未接通“用户上传文件 -> 自动分块 -> 自动建索引 -> 检索”的在线链路，需等待 A 同学真实 parser/index 产物

## 2. 前置确认

### 2.1 已冻结的 chunk schema

- [X]  根目录 `chunkschema规则.md` 已定义统一规则
- [ ]  A 同学确认产出字段至少包含：
  - [X]  `chunk_id`
  - [ ]  `doc_id`
  - [X]  `source`
  - [X]  `page`
  - [X]  `text`
- [ ]  A 同学确认 `page` 从 `1` 开始编号
- [ ]  A 同学确认 `chunk_id` 全局唯一且尽量稳定
- [ ]  A 同学确认 `source` 使用稳定文件名或相对路径

### 2.2 联调输入确认

- [ ]  确认 A 的 chunk 文件落盘路径
- [ ]  确认 A 的向量文件是否直接提供
- [ ]  确认 A 的最小联调样本已可用
- [ ]  确认 A 的 metadata 与 vectors 一一对应

## 3. 配置层 TODO

### 3.1 `configs/retrieval.yaml`

- [X]  增加 `enable_text_retrieval`
- [X]  增加 `top_k_text`
- [X]  增加 `score_threshold`
- [X]  增加 `index_path`
- [X]  增加 `metadata_path`
- [X]  增加 `context_max_chars`
- [X]  增加 `fallback_to_request_context`
- [X]  增加 `default_temperature`
- [X]  增加 `default_max_tokens`

### 3.2 `src/utils/settings.py`

- [X]  新增 `RetrievalConfig`
- [X]  接入 retrieval 配置读取
- [X]  提供 `settings.retrieval`
- [X]  将 chat 默认 `temperature/max_tokens` 配置化

## 4. Schema 层 TODO

### 4.1 `src/serving/schemas.py`

- [X]  调整 `ChatRequest`
  - [X]  `query` 保持必须
  - [X]  `context` 改为可选 fallback/debug 输入
  - [X]  `temperature` 允许为空并走配置默认值
  - [X]  `max_tokens` 允许为空并走配置默认值
- [X]  调整 `Citation`
  - [X]  增加 `chunk_id`
  - [X]  增加 `source`
  - [X]  增加 `page`
  - [X]  保留 `snippet`
  - [X]  兼容性需要时保留 `source_ref`
- [X]  检查 `ChatResponse` 仍兼容当前前端

## 5. 检索模块 TODO

### 5.1 新增检索实现

- [X]  新增 `src/models/retrieval.py`
- [X]  定义 `Evidence` 结构
  - [X]  `chunk_id`
  - [X]  `source`
  - [X]  `page`
  - [X]  `text`
  - [X]  `snippet`
  - [X]  `score`
- [X]  定义 `BaseTextRetriever`
- [X]  实现 `LocalTextRetriever`

### 5.2 检索能力

- [X]  实现 metadata 读取
- [X]  实现向量索引读取
- [X]  实现 query embedding
- [X]  实现相似度计算
- [X]  实现 top-k 返回
- [X]  实现 `score_threshold` 过滤
- [X]  检查 metadata / vectors 长度一致性

## 6. 依赖注入 TODO

### 6.1 `src/serving/deps.py`

- [X]  新增 `get_text_retriever()`
- [X]  用 `lru_cache` 缓存 retriever
- [X]  让 retriever 复用 `text_embedding_client`

## 7. Chat 主流程 TODO

### 7.1 `src/serving/api.py`

- [X]  修改 `/api/v1/chat` 依赖，注入 retriever
- [X]  调整主流程为：
  - [X]  读取 query
  - [X]  调用 `retriever.retrieve(query)`
  - [X]  构造 evidence context
  - [X]  使用 evidence 生成 answer
  - [X]  映射真实 citations
- [X]  删除旧的 `ctx-i` 占位 citation 逻辑
- [X]  无检索命中时支持稳定 fallback
- [X]  错误分类明确：
  - [X]  `retrieval_failed`
  - [X]  `chat_failed`

### 7.2 Prompt 组装

- [X]  system prompt 约束“优先依据检索证据回答”
- [X]  system prompt 约束“证据不足时明确说明”
- [X]  system prompt 约束“不要编造页码和来源”
- [X]  每条 evidence 使用统一格式拼接
- [X]  限制总 context 长度不超过 `context_max_chars`

## 8. 模型调用与参数 TODO

### 8.1 `src/models/clients.py`

- [X]  保持 client 仅负责调用模型接口
- [X]  不把 retrieval 逻辑混入 client 层
- [X]  检查 `temperature/max_tokens` 支持传入默认配置值

## 9. 联调 TODO

### 9.1 和 A 同学联调

- [X]  用最小样本验证 chunk schema 可读
- [X]  用最小样本验证向量与 metadata 对齐（基于本地 mock 数据）
- [X]  至少完成 1 个命中样例
- [X]  至少完成 1 个未命中样例
- [ ]  等 A 的真实 chunk/index 产物后再做一次正式联调

### 9.2 和 C 同学同步

- [ ]  同步 `citations` 字段结构变化
- [ ]  确认前端能显示 `source/page`
- [ ]  确认短期兼容旧字段是否必要

## 10. 测试与验收

### 10.1 最少测试项

- [X]  有检索命中时，`/api/v1/chat` 返回 answer + citations
- [X]  citations 至少包含 `chunk_id/page/source/snippet`
- [X]  无检索命中时接口不崩溃
- [ ]  索引缺失时错误信息清晰
- [X]  配置项如 `top_k_text` 实际生效

### 10.2 2026-06-04 前完成标准

- [X]  `/api/v1/chat` 不依赖手工传 `context` 也能回答（基于本地 mock 索引）
- [X]  text retrieval 已接入主流程
- [X]  citation 使用真实字段而不是 `ctx-i`
- [X]  检索参数和生成参数都已配置化
- [X]  最小 smoke test 通过

## 11. 风险与处理

### 11.1 主要风险

- [ ]  A 同学 schema 临时变更
- [ ]  A 的 index 路径不稳定
- [X]  chunk 过长导致 prompt 噪声过大（已通过缩小 `context_max_chars` 临时缓解）
- [ ]  A 的 metadata 与 vectors 不一致

### 11.2 对应处理

- [X]  以 `chunkschema规则.md` 为联调基准
- [X]  先固定当前 mock `metadata_path/index_path`
- [X]  限制 `top_k_text` 和 `context_max_chars`
- [ ]  在 retriever 初始化时做更早的一致性检查

## 12. 建议开发顺序

- [X]  第一步：改 `src/serving/schemas.py`
- [X]  第二步：改 `configs/retrieval.yaml`
- [X]  第三步：改 `src/utils/settings.py`
- [X]  第四步：新增 `src/models/retrieval.py`
- [X]  第五步：改 `src/serving/deps.py`
- [X]  第六步：改 `src/serving/api.py`
- [X]  第七步：补 smoke test / 联调验证

## 13. 每日检查用简表

- [X]  今天是否拿到可用 chunk 样本
- [ ]  今天是否确认 schema 无变动
- [X]  今天是否推进了一个主流程节点
- [X]  今天是否留了可验证结果
- [ ]  今天是否同步了阻塞项
