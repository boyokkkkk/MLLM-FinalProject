# 项目部署与初始化流程（团队版）

本文用于指导新同学从零完成项目环境初始化、数据集集成、健康检查与服务启动。

## 1. 环境初始化

在项目根目录执行（Windows PowerShell）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
```

可选：准备 `.env`

```env
OPENAI_API_KEY=your_key
VLM_BASE_URL=http://localhost:8001/v1
```

## 2. 数据集下载与集成

当前默认源：
1. DocVQA: `lmms-lab/DocVQA`（config=`DocVQA`，仅 `validation/test`）
2. ChartQA: `HuggingFaceM4/ChartQA`

建议执行（下载 + 保存官方缓存 + 导出项目 raw）：

```powershell
python scripts/01_download_datasets.py --from-hf --hf-official-layout
```

## 3. 预处理（生成项目统一 JSONL）

```powershell
python scripts/01_prepare_datasets.py validate --mode eval
python scripts/01_prepare_datasets.py prepare --mode eval --from-hf-cache --datasets docvqa,chartqa --show-progress --progress-every 500
```

说明：`--mode eval` 只处理 `val/test`，适配当前 DocVQA。

## 4. 数据集健康检查（完整性 + 可用性）

```powershell
python scripts/02_check_dataset_health.py --mode eval --datasets docvqa,chartqa --verify-image-open --sample-image-count 3
```

通过标准：
1. `[check] OK`
2. `parse_bad=0`
3. `image_exists` 与 `rows` 相等

## 5. 冗余数据清理（可选）

先预览：

```powershell
python scripts/03_clean_dataset_dirs.py --dry-run --datasets docvqa,chartqa
```

若确认删除 `data/raw_hf`：

```powershell
python scripts/03_clean_dataset_dirs.py --datasets docvqa,chartqa --remove-raw-hf
```

清理后建议再次检查：

```powershell
python scripts/02_check_dataset_health.py --mode eval --datasets docvqa,chartqa
```

## 6. 启动服务

后端：

```powershell
python -m src api
```

前端（Streamlit）：

```powershell
python -m src ui
```

## 7. 常见问题

1. 预处理看起来卡住：
   - 使用 `--show-progress` 观察进度。
   - 检查是否有残留 Python 进程占用文件。
2. DocVQA 没有 `train`：
   - 当前默认源确实仅 `validation/test`，属于预期。
3. 清理后数据异常：
   - 重新执行第 2~4 步可恢复。

