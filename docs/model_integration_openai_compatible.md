# OpenAI-compatible 模型接入（LLM/VLM）

本文用于把项目正式接入可用的 LLM/VLM 服务，并完成最小可用验证。

## 1. 前置条件

在项目根目录、并使用项目虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .
```

确保你已有一个 OpenAI-compatible 服务地址（例如 vLLM / Xinference / OneAPI 等），并知道模型名。

## 2. 配置 `.env`

在项目根目录创建或更新 `.env`：

```env
OPENAI_API_KEY=your_key_or_EMPTY
VLM_BASE_URL=http://127.0.0.1:8001/v1
TEXT_EMB_BASE_URL=http://127.0.0.1:8001/v1
VISION_EMB_BASE_URL=http://127.0.0.1:8001/v1

VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
TEXT_EMB_MODEL=Qwen/Qwen3-Embedding-4B
VISION_EMB_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
```

说明：
1. `*_BASE_URL` 需要是带 `/v1` 的根路径。
2. `*_MODEL` 必须与服务端实际加载模型名一致。
3. 如果服务不校验 key，可使用 `EMPTY`。

## 3. 运行模型连通性检查

只测聊天（VLM/LLM）：

```powershell
python scripts/04_check_model_connectivity.py --targets chat
```

只测文本向量：

```powershell
python scripts/04_check_model_connectivity.py --targets text_emb
```

全测：

```powershell
python scripts/04_check_model_connectivity.py --targets chat,text_emb,vision_emb
```

通过标准：输出包含 `[check] OK`。

## 4. 启动项目 API 并端到端验证

启动后端：

```powershell
python -m src api
```

新开一个终端（同样激活 `.venv`）做接口验证：

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/v1/chat" -ContentType "application/json" -Body '{"query":"测试：只回复OK","context":[],"temperature":0.0,"max_tokens":32}'
```

如果返回 `answer/model` 字段，说明项目已完成 OpenAI-compatible 接入。

## 5. 常见问题排查

1. `404`：大概率 `BASE_URL` 没带 `/v1`，或服务端不是 OpenAI-compatible 路由。
2. `401`：`OPENAI_API_KEY` 与服务端校验策略不一致。
3. `model not found`：`.env` 里的模型名与服务端加载名不一致。
4. 请求超时：增大 `VLM_TIMEOUT_S` / `TEXT_EMB_TIMEOUT_S` / `VISION_EMB_TIMEOUT_S`。
