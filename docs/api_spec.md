# API Specification (v1)

Base URL: `http://127.0.0.1:8000`
Prefix: `/api/v1`

## Design goals

- Frontend-agnostic REST API.
- Stable request/response schemas.
- Explicit versioning by URL prefix.
- Easy to consume by Web, mobile, desktop, and CLI clients.

## Endpoints

### 1) Health check

- Method: `GET`
- Path: `/health`
- Response:

```json
{
  "status": "ok",
  "version": "0.1.0"
}
```

### 2) Chat completion

- Method: `POST`
- Path: `/api/v1/chat`
- Request:

```json
{
  "query": "What is the key idea of this slide?",
  "workspace_id": "abc123workspace",
  "context": ["chunk-1 text", "chunk-2 text"],
  "image_data_urls": [],
  "temperature": 0.2,
  "max_tokens": 512,
  "history": []
}
```

- Response:

```json
{
  "answer": "...",
  "citations": [
    {
      "chunk_id": "chk-001",
      "source": "docvqa/val/10269#page=3#figure=fig-12#block=block-0002",
      "page": 3,
      "figure_id": "fig-12",
      "figure_no": "12",
      "block_id": "block-0002",
      "snippet": "...",
      "source_ref": "docvqa/val/10269#page=3#figure=fig-12#block=block-0002"
    }
  ],
  "model": "Qwen/Qwen2.5-VL-7B-Instruct"
}
```

### 3) Text embedding

- Method: `POST`
- Path: `/api/v1/embed/text`
- Request:

```json
{
  "inputs": ["text A", "text B"]
}
```

- Response:

```json
{
  "model": "Qwen/Qwen3-Embedding-4B",
  "vectors": [[0.01, 0.02], [0.03, 0.04]]
}
```

### 4) Vision embedding (placeholder interface)

- Method: `POST`
- Path: `/api/v1/embed/vision`
- Request:

```json
{
  "inputs": ["image_desc_or_serialized_feature_1"]
}
```

- Response:

```json
{
  "model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "vectors": [[0.11, 0.12]]
}
```

## Error format

HTTP status: `4xx/5xx`

```json
{
  "detail": "chat_failed: ..."
}
```

## Frontend compatibility notes

- Use `/api/v1` prefix in all clients.
- Keep client side strictly typed against these schemas.
- New fields should be additive to avoid breaking old frontends.
- Future breaking changes should use `/api/v2`.

## JS fetch example

```js
const resp = await fetch("http://127.0.0.1:8000/api/v1/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "Summarize page 3",
    context: ["..."],
    temperature: 0.2,
    max_tokens: 512
  })
});
const data = await resp.json();
```
