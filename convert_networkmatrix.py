"""
将 NetworkMatrix 数据集转换为与 abide.npy 相同的格式。

输入：
  data/NetworkMatrix/NetworkMatrix/ABIDE/200/*.mat  + labels.txt
  data/NetworkMatrix/NetworkMatrix/MDD/200/*.mat    + Labels.txt

输出：
  data/abide_networkmatrix.npy
  data/mdd_networkmatrix.npy

格式（与 abide.npy 一致）：
  {
    'corr':       (n, 200, 200)  - 功能连接矩阵（来自 NetworkMatrix）
    'label':      (n,)           - 0/1 标签
    'timeseires': (n, 200, 100)  - 占位数组（三个模型均不实际使用）
    'pcorr':      (n, 200, 200)  - 占位数组（仅 ASDFormer 使用，此处置零）
    'site':       (n,)           - 占位数组（所有模型均不使用）
  }
"""

import os
import numpy as np
import scipy.io as sio


def convert_dataset(mat_dir: str, label_file: str, output_path: str):
    # 读取标签
    with open(label_file, 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    labels = np.array(labels, dtype=np.float32)

    # 按文件名排序，与标签对齐
    mat_files = sorted([
        f for f in os.listdir(mat_dir) if f.endswith('.mat')
    ])

    assert len(mat_files) == len(labels), (
        f"文件数 {len(mat_files)} 与标签数 {len(labels)} 不一致！"
    )

    n = len(mat_files)
    corr = np.zeros((n, 200, 200), dtype=np.float32)

    for i, fname in enumerate(mat_files):
        mat = sio.loadmat(os.path.join(mat_dir, fname))
        matrix = mat['NetworkMatrix'].astype(np.float32)
        assert matrix.shape == (200, 200), f"{fname} 矩阵形状异常: {matrix.shape}"
        corr[i] = matrix

    # 占位数组：timeseires 用全 1（避免 StandardScaler 除零），pcorr/site 用全零
    timeseires = np.ones((n, 200, 100), dtype=np.float32)
    pcorr = np.zeros((n, 200, 200), dtype=np.float32)
    site = np.zeros(n, dtype=np.float32)

    data = {
        'corr': corr,
        'label': labels,
        'timeseires': timeseires,
        'pcorr': pcorr,
        'site': site,
    }

    np.save(output_path, data, allow_pickle=True)
    print(f"已保存 {n} 个样本 -> {output_path}")
    print(f"  标签分布: 0={int((labels==0).sum())}  1={int((labels==1).sum())}")


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    nm_base = os.path.join(base, 'data', 'NetworkMatrix', 'NetworkMatrix')

    convert_dataset(
        mat_dir=os.path.join(nm_base, 'ABIDE', '200'),
        label_file=os.path.join(nm_base, 'ABIDE', '200', 'labels.txt'),
        output_path=os.path.join(base, 'data', 'abide_networkmatrix.npy'),
    )

    convert_dataset(
        mat_dir=os.path.join(nm_base, 'MDD', '200'),
        label_file=os.path.join(nm_base, 'MDD', '200', 'Labels.txt'),
        output_path=os.path.join(base, 'data', 'mdd_networkmatrix.npy'),
    )
