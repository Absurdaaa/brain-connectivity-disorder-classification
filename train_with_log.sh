#!/bin/bash
# 训练脚本 - 将输出保存到日志文件

# 设置环境变量
export WANDB_MODE=offline
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 创建日志目录
LOG_DIR="logs"
mkdir -p $LOG_DIR

# 生成时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

echo "开始训练..." | tee $LOG_FILE
echo "日志文件: $LOG_FILE" | tee -a $LOG_FILE
echo "=" | tee -a $LOG_FILE

# 切换到 unified 目录
cd models/unified

# 运行训练并保存输出
conda run -n unified python -m source \
    --multirun \
    datasz=100p \
    model=ASDFormer,comtf,bnt \
    dataset=ADHD200 \
    repeat_time=1 \
    preprocess=non_mixup \
    2>&1 | tee ../../$LOG_FILE

echo "" | tee -a ../../$LOG_FILE
echo "=" | tee -a ../../$LOG_FILE
echo "训练完成！" | tee -a ../../$LOG_FILE
echo "日志已保存到: $LOG_FILE" | tee -a ../../$LOG_FILE

# 提取结果摘要
echo "" | tee -a ../../$LOG_FILE
echo "结果摘要:" | tee -a ../../$LOG_FILE
grep -E "(test acc mean|test auc mean|test sensitivity mean|test specificity mean)" ../../$LOG_FILE | tee -a ../../$LOG_FILE
