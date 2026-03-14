#!/bin/bash
# 训练六个模型权重用于投票推理
# 三个模型 × 两个数据集 (ABIDE / MDD) = 6 个权重文件
#
# 运行前请先激活正确的 conda 环境：
#   conda activate ASDFormer
#
# 用法：bash scripts/train_vote_models.sh
# 从项目根目录运行
# cd /Users/linshangjin/Desktop/dachuang/models/unified

set -e

UNIFIED_DIR="./models/unified"
WEIGHTS_DIR="./weights"
mkdir -p "$WEIGHTS_DIR"

export WANDB_MODE=offline
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 模型名列表（Hydra config 中的 model 参数）
MODELS=("ASDFormer" "comtf" "bnt")
DATASETS=("ABIDE" "MDD")

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo ""
        echo "========================================"
        echo " 训练: model=${MODEL}  dataset=${DATASET}"
        echo "========================================"

        # 记录训练前 result 目录中该模型的文件夹列表
        RESULT_DIR="${UNIFIED_DIR}/result/${MODEL}"
        mkdir -p "$RESULT_DIR"
        BEFORE=$(ls "$RESULT_DIR" 2>/dev/null || true)

        # 进入 unified 目录执行训练（repeat_time=1 只训练一次，获得单个权重）
        (
            cd "$UNIFIED_DIR"
            python -m source \
                datasz=100p \
                model=${MODEL} \
                dataset=${DATASET} \
                repeat_time=1 \
                preprocess=non_mixup
        )

        # 找到本次新生成的子目录（取最新的）
        AFTER=$(ls "$RESULT_DIR" 2>/dev/null || true)
        NEW_DIR=""
        for d in $AFTER; do
            if ! echo "$BEFORE" | grep -qx "$d"; then
                NEW_DIR="$d"
            fi
        done

        if [ -z "$NEW_DIR" ]; then
            # fallback：直接取最新修改的子目录
            NEW_DIR=$(ls -t "$RESULT_DIR" | head -1)
        fi

        SRC="${RESULT_DIR}/${NEW_DIR}/model.pt"
        DST="${WEIGHTS_DIR}/${MODEL}_${DATASET}.pt"

        if [ -f "$SRC" ]; then
            cp "$SRC" "$DST"
            echo "  权重已保存至: ${DST}"
        else
            echo "  警告: 未找到 ${SRC}，请手动检查 ${RESULT_DIR}/${NEW_DIR}/"
        fi
    done
done

echo ""
echo "========================================"
echo " 训练完成！权重文件列表："
ls -lh "$WEIGHTS_DIR/"
echo "========================================"
