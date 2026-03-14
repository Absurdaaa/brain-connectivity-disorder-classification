# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 重要：每次对话开始前，请先阅读 [PLAN.md](./PLAN.md) 了解当前任务进度和计划。

## Project Overview

大创项目 (University Innovation Project) — comparing three transformer-based models for autism spectrum disorder (ASD) prediction from fMRI brain connectivity data using the ABIDE dataset.

## Conda Environments

- **ASDFormer**: `conda activate ASDFormer` (also used for Com-BrainTF)
- **BrainNetworkTransformer**: `conda activate bnt`

Both environments require Python 3.9, PyTorch 1.12.1 + CUDA 11.3, wandb, and Hydra.

## Running Models

All models are run from their respective subdirectory with `WANDB_MODE=offline`:

```bash
# ASDFormer
cd ./models/ASDFormer
export WANDB_MODE=offline
python -m source --multirun datasz=100p model=ASDFormer dataset=ABIDE repeat_time=5 preprocess=non_mixup

# Com-BrainTF
cd ./models/Com-BrainTF
export WANDB_MODE=offline
python -m source --multirun datasz=100p model=comtf dataset=ABIDE repeat_time=5 preprocess=non_mixup

# BrainNetworkTransformer (supports multiple models and datasets)
cd ./models/BrainNetworkTransformer
export WANDB_MODE=offline
python -m source --multirun datasz=100p model=bnt,fbnetgen,brainnetcnn,transformer dataset=ABIDE,ABCD repeat_time=5 preprocess=mixup
```

Or use the scripts: `bash scripts/ASDFormer.sh`, `bash scripts/Com-BrainTF.sh`, `bash scripts/bnt.sh`

## Data Processing

```bash
# Process a single fMRI subject (test)
python test.py

# Batch process Peking dataset
python process_peking_data.py
```

Both scripts use nilearn with the Schaefer 200 ROI atlas to extract time series and compute functional connectivity matrices.

## Architecture

All three models share the same project structure under `models/<ModelName>/source/`:
- `__main__.py` — entry point, runs `repeat_time` experiments with different seeds, reports mean/std of accuracy, AUC, sensitivity, specificity
- `conf/` — Hydra YAML configs (dataset, model, optimizer, training, datasz, preprocess)
- `dataset/abide.py` — loads `abide.npy` which contains keys: `timeseires` (n_subjects, n_timepoints, 200), `corr` (n_subjects, 200, 200), `label`, `site`
- `models/` — model definition
- `training/training.py` — training loop

### Model Differences

| Model | Key Architecture | Paper |
|---|---|---|
| ASDFormer | Mixture of pooling-classifier experts, hierarchical community structure | arXiv:2508.14005 |
| Com-BrainTF | Community-aware transformer, local+global transformers with learnable class tokens per community | MICCAI 2023 |
| BrainNetworkTransformer | Multi-layer transformer with DEC (Deep Embedded Clustering) pooling | NeurIPS 2022 |

### Configuration System (Hydra)

Key overridable parameters:
- `datasz`: fraction of data to use (e.g., `100p` = 100%)
- `model`: model name (e.g., `ASDFormer`, `comtf`, `bnt`, `fbnetgen`, `brainnetcnn`, `transformer`)
- `dataset`: `ABIDE` or `ABCD`
- `repeat_time`: number of repeated experiments
- `preprocess`: `mixup` or `non_mixup`

Results are saved to `result/`, experiment logs to `outputs/` or `multirun/` (all gitignored).

### Dataset Path

The ABIDE dataset (`abide.npy`) is expected at `../../../datasets/abide.npy` relative to each model's source directory, i.e., `data/abide.npy` from the repo root (gitignored due to size).
