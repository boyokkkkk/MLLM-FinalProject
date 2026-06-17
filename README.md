# MLLMProject

多模态文档 RAG 项目，支持：

- 文档解析与索引构建：PDF、图片、Markdown、文本等
- 检索增强问答：文本检索、视觉辅助、工作区问答
- DocVQA 基准评测与消融实验
- Web 端演示：统一由 FastAPI 提供 API 与前端页面

## 1. 项目环境

- 操作系统：Windows 为主，脚本已按 PowerShell 使用方式组织
- Python：`>=3.10`
- 依赖安装方式：推荐使用项目内虚拟环境 `.venv`
- 外部模型服务：需准备兼容 OpenAI API 的
  - VLM 服务
  - 文本向量服务
  - 视觉向量服务

## 2. 目录结构

```text
MLLMProject/
├─ configs/                 # 数据集、评测、检索配置
├─ data/                    # 原始数据、处理中间产物、工作区数据
├─ docs/                    # 报告、说明文档、论文相关材料
├─ outputs/                 # 评测结果、表格、图表、实验输出
├─ scripts/                 # 数据准备、索引构建、评测、消融脚本
├─ src/
│  ├─ evaluation/           # 基准评测与指标计算
│  ├─ models/               # 模型客户端、检索器
│  ├─ serving/              # FastAPI 服务、工作区检索与问答逻辑
│  ├─ ui/                   # Web 前端静态资源
│  ├─ utils/                # 配置与通用工具
│  └─ cli.py                # 统一启动入口
├─ test_raw_file/           # 本地工作区演示测试文件
├─ tests/                   # 测试代码
├─ requirements.txt         # 运行依赖
├─ pyproject.toml           # 项目元数据与入口脚本
└─ README.md
```

## 3. 首次部署流程

### 3.1 创建虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2 配置模型服务

在项目根目录创建或修改 `.env`，至少配置以下变量：

```env
OPENAI_API_KEY=your_api_key

VLM_BASE_URL=http://your_vlm_endpoint/v1
TEXT_EMB_BASE_URL=http://your_text_embedding_endpoint/v1
VISION_EMB_BASE_URL=http://your_vision_embedding_endpoint/v1

VLM_MODEL=your_vlm_model
TEXT_EMB_MODEL=your_text_embedding_model
VISION_EMB_MODEL=your_vision_embedding_model
```

常用说明：

- `OPENAI_API_KEY`：统一给 OpenAI-compatible 接口使用
- `VLM_*`：回答生成模型
- `TEXT_EMB_*`：文本检索向量模型
- `VISION_EMB_*`：视觉向量模型

可先用下面命令检查模型连通性：

```powershell
.\.venv\Scripts\python.exe scripts\04_check_model_connectivity.py --project-root .
```

### 3.3 下载 MinerU 模型

如果需要解析 PDF / 图片文档，先准备 MinerU 模型：

```powershell
.\.venv\Scripts\python.exe scripts\00_download_mineru_models.py
```

## 4. 项目运行流程

### 4.1 启动后端与前端

当前前端已经直接挂载在 FastAPI 应用中，因此只需启动一个服务：

```powershell
.\.venv\Scripts\python.exe scripts\05_launch_api.py
```

或者：

```powershell
.\.venv\Scripts\python.exe -m src api
```

默认地址：

- API：`http://127.0.0.1:8000/api/v1`
- Web UI：`http://127.0.0.1:8000`

### 4.2 健康检查

浏览器或命令行访问：

```text
http://127.0.0.1:8000/health
```

### 4.3 工作区演示使用方式

1. 打开 Web UI
2. 上传 PDF、图片、Markdown 等文件到工作区
3. 等待工作区完成解析、切块和索引
4. 在 Query 页面提问

适合的工作区问题示例：

- `请只根据文件 C9 - ABE 属性基加密.md 回答：Boneh-Franklin IBE 方案的核心流程是什么？并在答案中标出引用。`
- `请只根据文件 2-ClassicCrypto.pdf 回答：page 3 在讲什么？并在答案中标出引用。`
- `请只根据文件 C5 - DigitalSignature 数字签名.md 回答：密码学 Hash 的核心安全性质有哪些？并在答案中标出引用。`

## 5. 数据集处理流程

### 5.1 下载数据集

```powershell
.\.venv\Scripts\python.exe scripts\01_download_datasets.py --project-root .
```

### 5.2 规范化数据集

```powershell
.\.venv\Scripts\python.exe scripts\01_prepare_datasets.py --project-root .
```

### 5.3 检查数据健康状态

```powershell
.\.venv\Scripts\python.exe scripts\02_check_dataset_health.py --project-root .
```

### 5.4 运行 MinerU 解析

```powershell
.\.venv\Scripts\python.exe scripts\06_run_mineru.py --project-root . --datasets docvqa --splits val
```

### 5.5 切块

```powershell
.\.venv\Scripts\python.exe scripts\07_parse_and_chunk.py --project-root . --config configs/datasets.yaml --datasets docvqa --splits val
```

### 5.6 构建索引

```powershell
.\.venv\Scripts\python.exe scripts\08_build_indexes.py --project-root . --config configs/datasets.yaml
```

### 5.7 一键跑完整索引流程

```powershell
.\.venv\Scripts\python.exe scripts\10_run_data_index_pipeline.py --project-root . --datasets docvqa --splits val
```

## 6. 评测与实验

### 6.1 构建 `unique-docpage-100` 测试集

```powershell
.\.venv\Scripts\python.exe scripts\15_build_unique_docpage_benchmark.py --project-root .
```

### 6.2 运行主评测

```powershell
.\.venv\Scripts\python.exe scripts\12_run_benchmark_eval.py ^
  --project-root . ^
  --config configs/eval.yaml ^
  --datasets-config configs/datasets.yaml ^
  --suite retrieval_benchmark ^
  --mode rag ^
  --datasets docvqa ^
  --splits val ^
  --sample-manifest outputs/eval/docvqa_val_unique_docpage_100.manifest.jsonl ^
  --top-k 5 ^
  --run-name docvqa_val_unique_docpage_100_rag_stronger_qia_base
```

### 6.3 运行消融实验

```powershell
.\.venv\Scripts\python.exe scripts\18_run_ablation_suite.py --project-root . --groups all
```

按组运行示例：

```powershell
.\.venv\Scripts\python.exe scripts\18_run_ablation_suite.py --project-root . --groups retrieval
.\.venv\Scripts\python.exe scripts\18_run_ablation_suite.py --project-root . --groups visual
.\.venv\Scripts\python.exe scripts\18_run_ablation_suite.py --project-root . --groups gating
```

### 6.4 生成实验表格与论文插图素材

```powershell
.\.venv\Scripts\python.exe scripts\16_generate_benchmark_assets.py --project-root .
.\.venv\Scripts\python.exe scripts\19_generate_ablation_tables.py --project-root .
.\.venv\Scripts\python.exe scripts\20_generate_ablation_overview.py --project-root .
```

## 7. 常用脚本说明

### 环境与连通性

- `scripts/00_prepare_env.ps1`：准备本地环境
- `scripts/00_download_mineru_models.py`：下载 MinerU 模型
- `scripts/04_check_model_connectivity.py`：检查模型服务连接

### 数据与索引

- `scripts/01_download_datasets.py`：下载数据集
- `scripts/01_prepare_datasets.py`：规范化数据集
- `scripts/06_run_mineru.py`：解析 PDF / 图片
- `scripts/07_parse_and_chunk.py`：切块
- `scripts/08_build_indexes.py`：构建检索索引
- `scripts/09_query_indexes.py`：调试索引查询
- `scripts/10_run_data_index_pipeline.py`：一键跑数据索引流程

### 评测与分析

- `scripts/11_smoke_mllm_eval.py`：评测冒烟测试
- `scripts/12_run_benchmark_eval.py`：主评测脚本
- `scripts/14_analyze_benchmark_errors.py`：错误分析
- `scripts/15_build_unique_docpage_benchmark.py`：构建评测子集
- `scripts/18_run_ablation_suite.py`：消融实验总控

### 结果整理

- `scripts/16_generate_benchmark_assets.py`：生成图表与表格素材
- `scripts/19_generate_ablation_tables.py`：生成消融表
- `scripts/20_generate_ablation_overview.py`：汇总消融结果
- `scripts/21_build_strict_visual_subset.py`：构建严格视觉子集
- `scripts/22_generate_strict_visual_tables.py`：严格视觉子集表格
- `scripts/23_generate_gate_target_table.py`：gate 目标切片表

## 8. 核心代码结构说明

### `src/serving`

- `api.py`：FastAPI 主服务，负责
  - `/api/v1/chat` 问答
  - citation 生成
  - 工作区与临时图片问答组装
  - Web 静态前端挂载
- `workspaces.py`：工作区生命周期、文件解析、工作区检索
- `deps.py`：依赖注入，加载检索器和模型客户端
- `schemas.py`：API 请求/响应结构

### `src/models`

- 模型客户端封装
- 文本/视觉检索器
- 稀疏检索与 rerank 逻辑

### `src/evaluation`

- 基准样本读取
- 指标计算
- RAG 与检索评测主流程

### `src/ui`

- `web_static/`：当前线上使用的前端页面
- `services/`：前端 API 调用逻辑

### `src/utils`

- 全局配置读取
- 路径、环境变量与 YAML 配置管理

## 9. 快速启动最短路径

如果只想快速跑通演示：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\04_check_model_connectivity.py --project-root .
.\.venv\Scripts\python.exe scripts\05_launch_api.py
```

然后打开：

```text
http://127.0.0.1:8000
```
