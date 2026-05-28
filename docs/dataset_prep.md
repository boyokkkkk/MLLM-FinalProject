# 数据集准备规范（DocVQA / ChartQA）

## 1. 是否需要提前下载数据集？

需要。建议由一名同学统一下载并提供下载来源、版本和日期，其他同学通过同一脚本完成校验与预处理，避免数据漂移。

原因：
1. 不同来源可能存在字段差异（如 `answer` vs `answers`）
2. 同名数据集可能有不同版本
3. 评测可复现依赖固定数据版本

## 2. 团队统一目录规范

将原始数据放在以下位置（按当前默认 Hugging Face 源）：

- `data/raw/docvqa/val.json`
- `data/raw/docvqa/test.json`
- `data/raw/chartqa/train.json`
- `data/raw/chartqa/val.json`
- `data/raw/chartqa/test.json`

脚本会统一输出：

- `data/processed/docvqa/{val,test}.jsonl`
- `data/processed/chartqa/{train,val,test}.jsonl`

## 3. 下载与预处理统一命令

### 3.1 配置下载清单

编辑：`configs/dataset_downloads.yaml`

每个文件至少填写：
1. `url`
2. `output`
3. `sha256`（强烈建议）

### 3.2 执行下载

```bash
python scripts/01_download_datasets.py
```

从 Hugging Face 直接拉取（推荐当前阶段）：

```bash
python scripts/01_download_datasets.py --from-hf
```

按“官方目录形态”保存（推荐团队长期使用）：

```bash
python scripts/01_download_datasets.py --from-hf --hf-official-layout
```

会生成：
1. 官方层：`data/raw_hf/docvqa/`、`data/raw_hf/chartqa/`（HF save_to_disk 目录）
2. 项目层：`data/raw/docvqa/*.json`、`data/raw/chartqa/*.json`（供预处理脚本消费）

若只想从已保存的官方层重新导出项目层（不重新下载）：

```bash
python scripts/01_download_datasets.py --from-hf --hf-export-from-official --force
```

对应数据集默认值：
1. DocVQA: `lmms-lab/DocVQA`（config: `DocVQA`，当前仅 `validation/test`）
2. ChartQA: `HuggingFaceM4/ChartQA`

常用参数：

```bash
python scripts/01_download_datasets.py --force
python scripts/01_download_datasets.py --skip-checksum
python scripts/01_download_datasets.py --from-hf --force
```

如果使用虚拟环境：

```bash
.\.venv\Scripts\python.exe scripts/01_download_datasets.py
```

### 3.3 校验并规范化

当前推荐（DocVQA 无 train）：

```bash
python scripts/01_prepare_datasets.py validate --mode eval
python scripts/01_prepare_datasets.py prepare --mode eval
python scripts/02_check_dataset_health.py --mode eval --datasets docvqa,chartqa --verify-image-open
```

如果使用虚拟环境：

```bash
.\.venv\Scripts\python.exe scripts/01_prepare_datasets.py validate --mode eval
.\.venv\Scripts\python.exe scripts/01_prepare_datasets.py prepare --mode eval
.\.venv\Scripts\python.exe scripts/02_check_dataset_health.py --mode eval --datasets docvqa,chartqa --verify-image-open
```

### 3.4 清理冗余目录（可选）

先预览：

```bash
python scripts/03_clean_dataset_dirs.py --dry-run --datasets docvqa,chartqa
```

删除 `data/raw_hf`：

```bash
python scripts/03_clean_dataset_dirs.py --datasets docvqa,chartqa --remove-raw-hf
```

说明：
1. `--mode train` 会要求 `train/val/test` 都存在，当前 DocVQA 默认源会校验失败。
2. 若后续切换到包含训练集的 DocVQA 源，再使用 `--mode train`。

## 4. 标准化评测输入格式（JSONL）

每一行一个样本，字段如下：

```json
{
  "id": "docvqa-val-0",
  "dataset": "docvqa",
  "split": "val",
  "question": "What is ...?",
  "answers": ["..."],
  "image": "optional/image/path.png",
  "evidence": "optional",
  "metadata": {
    "raw_keys": ["..."]
  }
}
```

## 5. 团队协作建议

1. 只提交脚本和配置，不提交大体积原始数据。
2. 每次更新数据源，更新 `docs/meeting_notes/` 记录版本、来源和日期。
3. 评测前固定使用同一批 `data/processed/*/*.jsonl`。
4. 不要手动改 `data/processed` 文件，统一由脚本重新生成。
