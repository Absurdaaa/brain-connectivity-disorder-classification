# 项目计划

## 任务列表

### 任务 1：分析三个模型框架异同 ✅ 已完成
**结论：** 三个模型框架高度相似（90%+ 代码重复），可以合并。详见下方分析。

### 任务 2：提供统一环境配置 ✅ 已完成
文件：`models/unified/environment.yml`
环境名：`brain-unified`，包含 Python 3.9 + PyTorch 1.12.1 + CUDA 11.3 + wandb + hydra + scipy + nilearn

### 任务 3：将 NetworkMatrix 数据转换为 abide.npy 格式 ✅ 已完成
脚本：`convert_networkmatrix.py`
输出：`data/abide_networkmatrix.npy`（767样本）、`data/mdd_networkmatrix.npy`（2428样本）

关键结论：三个模型的 forward 方法均不实际使用 time_series（接收但不参与计算），
site 也完全未使用。因此：
- timeseires：置为全 1 占位（避免 StandardScaler 除零）
- pcorr：置为全零（仅 ASDFormer 接收，但 encoder 只用 FC_matrix）
- site：置为全零

### 任务 4：合并三个模型框架 ✅ 已完成
目录：`models/unified/`
运行脚本：`bash scripts/unified.sh [model] [dataset] [repeat_time]`

统一框架设计：
- `__main__.py`：通用入口，基于 ASDFormer 版本（有随机种子、统计输出）
- `dataset/abide.py`：统一加载器，支持 abide / mdd 两个数据集名
- `dataset/dataloader.py`：始终包含 pcorr（4元组），兼容所有模型
- `training/training.py`：通过三个 flag 控制差异：
  - `use_partial`：是否传 pcorr 给模型（ASDFormer=true）
  - `use_pool_loss`：模型是否返回 (predict, loss_pool)（ComBrainTF=true）
  - `use_regularization`：是否加 CV² 正则化（ASDFormer=true）
- `models/__init__.py`：统一工厂，支持 ASDFormer / ComBrainTF / BrainNetworkTransformer / fbnetgen / brainnetcnn / transformer
- `conf/dataset/MDD.yaml`：新增 MDD 数据集配置

---

## 任务 1 分析结果

### 框架相似性

三个模型共同点：
- 都使用 Hydra 配置管理 + wandb 实验跟踪
- 都使用工厂模式：`dataset_factory`, `model_factory`, `training_factory`
- 数据加载逻辑几乎相同（都从 abide.npy 读取）
- 训练流程相同（train-val-test 循环）

关键差异：

| 差异点 | BrainNetworkTransformer | ASDFormer | Com-BrainTF |
|--------|------------------------|-----------|-------------|
| 数据输入 | 2个 (timeseries, corr) | 3个 (多了 pcorr) | 2个 |
| 模型输出 | 单输出 | 单输出+正则化 | 双输出(predict, loss_pool) |
| 最佳模型选择 | ❌ | ✅ 基于val_AUC | ✅ 基于val_AUC |
| 随机种子 | ❌ | ✅ | ✅ |
| 默认数据集 | ABCD | ABIDE | ABIDE |

### NetworkMatrix 数据格式

- ABIDE：767 个 .mat 文件，每个含 200×200 的 `NetworkMatrix`
- MDD：2428 个 .mat 文件，每个含 200×200 的 `NetworkMatrix`
- 文件名格式：`NetworkMatrix_ABIDEII-BNI_1_29006_1.mat`（ABIDE），`NetworkMatrix_S14-1-0013.mat`（MDD）

### abide.npy 目标格式

```python
{
    'timeseires': (n_subjects, 200, 100),  # 时间序列（NetworkMatrix数据无此项，需特殊处理）
    'label':      (n_subjects,),            # 标签（需从文件名或外部文件获取）
    'corr':       (n_subjects, 200, 200),  # Pearson相关矩阵 ← NetworkMatrix 直接对应此项
    'pcorr':      (n_subjects, 200, 200),  # 偏相关矩阵（可选）
    'site':       (n_subjects,),            # 站点信息（可从文件名解析）
}
```
