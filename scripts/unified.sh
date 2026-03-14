#!/bin/bash
# 统一框架运行脚本
# 用法: bash scripts/unified.sh [model] [dataset] [repeat_time]
# 示例: bash scripts/unified.sh ASDFormer ABIDE 5

MODEL=${1:-ASDFormer}
DATASET=${2:-ABIDE}
REPEAT=${3:-5}

cd ./models/unified
export WANDB_MODE=offline
python -m source --multirun \
    datasz=100p \
    model=${MODEL} \
    dataset=${DATASET} \
    repeat_time=${REPEAT} \
    preprocess=non_mixup
