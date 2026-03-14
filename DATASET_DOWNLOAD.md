# 数据集下载指南

## 快速开始

### 1. 下载 ADHD-200 数据集

```bash
# 下载所有被试（约 200+ 个）
python download_datasets.py --dataset adhd

# 下载指定数量的被试（更快）
python download_datasets.py --dataset adhd --n_subjects 50

# 下载所有支持的数据集
python download_datasets.py --dataset all
```

### 2. 使用下载的数据集训练

```bash
# 在 ADHD-200 数据集上训练 ASDFormer
bash scripts/unified.sh ASDFormer ADHD200 5

# 在 ADHD-200 数据集上训练所有模型
python scripts/train_all.py --datasets ADHD200 --repeat 5
```

## 内存需求估算

### 下载过程

| 数据集 | 被试数 | 原始数据大小 | 处理后大小 | 峰值内存 |
|--------|--------|-------------|-----------|----------|
| ADHD-200 (nilearn) | 40 | ~2GB | ~15MB | ~3GB |
| ADHD-200 (完整版) | ~200 | ~10GB | ~250MB | ~8GB |
| Development | 30 | ~1.5GB | ~40MB | ~3GB |

**说明：**
- **原始数据大小**：nilearn 下载的 NIfTI 文件（会缓存到 `~/nilearn_data/`）
- **处理后大小**：生成的 `.npy` 文件大小
- **峰值内存**：处理过程中的最大内存占用

### 训练过程

| 数据集 | 被试数 | 训练内存（GPU） | 训练内存（CPU） |
|--------|--------|----------------|----------------|
| ABIDE | 1009 | ~2GB | ~4GB |
| MDD | 2428 | ~4GB | ~8GB |
| ADHD-200 | ~200 | ~1GB | ~2GB |

## 支持的数据集

### 1. ADHD-200
- **疾病**：注意力缺陷多动障碍 (ADHD)
- **被试数**：40（nilearn 精简版）或 ~200（完整版，需手动下载）
- **标签**：ADHD (1) vs Control (0)
- **来源**：[ADHD-200 Consortium](http://fcon_1000.projects.nitrc.org/indi/adhd200/)
- **注意**：`download_datasets.py` 使用 nilearn 只能下载 40 个被试。完整数据集需从官网下载。

### 2. ABIDE
- **疾病**：自闭症谱系障碍 (ASD)
- **被试数**：1009
- **标签**：ASD (1) vs Control (0)
- **来源**：已有数据

### 3. MDD
- **疾病**：重度抑郁症 (Major Depressive Disorder)
- **被试数**：2428
- **标签**：MDD (1) vs Control (0)
- **来源**：已有数据

### 4. Development（可选）
- **类型**：发育数据集
- **被试数**：~30
- **标签**：Child (1) vs Adult (0)
- **来源**：nilearn 内置

## 数据格式

所有数据集统一为以下格式（与 `abide.npy` 一致）：

```python
{
    'timeseires': (n_subjects, 200, T),  # 时间序列
    'corr':       (n_subjects, 200, 200), # Pearson 相关矩阵
    'pcorr':      (n_subjects, 200, 200), # 偏相关矩阵
    'label':      (n_subjects,),          # 标签 (0/1)
    'site':       (n_subjects,),          # 站点信息
}
```

## 常见问题

### Q: 下载速度慢怎么办？
A: nilearn 会自动缓存数据到 `~/nilearn_data/`，首次下载较慢，后续会直接使用缓存。

### Q: 内存不足怎么办？
A: 使用 `--n_subjects` 参数限制下载数量：
```bash
python download_datasets.py --dataset adhd --n_subjects 30
```

### Q: 如何添加自己的数据集？
A: 参考 `convert_networkmatrix.py`，将数据转换为相同格式，然后：
1. 保存为 `data/your_dataset.npy`
2. 创建 `models/unified/source/conf/dataset/YOUR_DATASET.yaml`
3. 更新 `dataset/__init__.py` 添加数据集名

### Q: 下载的数据保存在哪里？
A:
- 原始 NIfTI 文件：`~/nilearn_data/`
- 处理后的 .npy 文件：`data/adhd200.npy`

## Sources

- [ADHD-200 Dataset](http://fcon_1000.projects.nitrc.org/indi/adhd200)
- [ADHD-200 Preprocessed](http://preprocessed-connectomes-project.org/adhd200/)
- [Nilearn ADHD Documentation](https://nilearn.github.io/dev/modules/description/adhd.html)
- [Preprocessed Connectomes Project](https://preprocessed-connectomes-project.org/datasets.html)
