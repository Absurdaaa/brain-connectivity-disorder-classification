#!/bin/bash
# 批量训练脚本：在所有数据集上训练所有模型
# 用法: bash scripts/train_all.sh [repeat_time]
# 示例: bash scripts/train_all.sh 5

REPEAT=${1:-5}
MODELS=("ASDFormer" "ComBrainTF" "BrainNetworkTransformer")
DATASETS=("ABIDE" "MDD")

echo "=========================================="
echo "批量训练任务"
echo "重复次数: ${REPEAT}"
echo "模型: ${MODELS[@]}"
echo "数据集: ${DATASETS[@]}"
echo "=========================================="

cd /Users/linshangjin/Desktop/dachuang/models/unified
export WANDB_MODE=offline

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo "=========================================="
        echo "开始训练: ${MODEL} on ${DATASET}"
        echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="

        python -m source \
            datasz=100p \
            model=${MODEL} \
            dataset=${DATASET} \
            repeat_time=${REPEAT} \
            preprocess=non_mixup

        if [ $? -eq 0 ]; then
            echo "✓ ${MODEL} on ${DATASET} 训练完成"
        else
            echo "✗ ${MODEL} on ${DATASET} 训练失败"
        fi
    done
done

echo ""
echo "=========================================="
echo "所有训练任务完成"
echo "权重保存位置: models/unified/result/"
echo "=========================================="
