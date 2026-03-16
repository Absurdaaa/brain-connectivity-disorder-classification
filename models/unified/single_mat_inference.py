#!/usr/bin/env python3
"""单.mat文件投票集成推理脚本

输入单个200×200的.mat文件，输出三个模型的投票集成分类结果。

用法:
    python single_mat_inference.py --mat_path ./your_data.mat --dataset ABIDE
    python single_mat_inference.py --mat_path ./test.mat --dataset MDD --weights_dir /path/to/weights
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from omegaconf import OmegaConf

# 将 source 目录加入 Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source'))

from models import model_factory

# 模型配置：(yaml 文件名, 权重文件前缀)
MODELS = [
    ('ASDFormer', 'ASDFormer'),
    ('comtf', 'ComBrainTF'),
    ('bnt', 'BrainNetworkTransformer'),
]


def get_device():
    """获取可用设备（CUDA > MPS > CPU）"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_cfg(dataset_name, model_yaml_name):
    """手动构建 OmegaConf 配置，不使用 Hydra"""
    conf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source', 'conf')

    # 加载各类配置文件
    dataset_cfg = OmegaConf.load(os.path.join(conf_dir, 'dataset', f'{dataset_name}.yaml'))
    model_cfg = OmegaConf.load(os.path.join(conf_dir, 'model', f'{model_yaml_name}.yaml'))
    optimizer_cfg = OmegaConf.load(os.path.join(conf_dir, 'optimizer', 'adam.yaml'))
    training_cfg = OmegaConf.load(os.path.join(conf_dir, 'training', 'basic_training.yaml'))
    datasz_cfg = OmegaConf.load(os.path.join(conf_dir, 'datasz', '100p.yaml'))
    preprocess_cfg = OmegaConf.load(os.path.join(conf_dir, 'preprocess', 'non_mixup.yaml'))

    cfg = OmegaConf.create({
        'dataset': dataset_cfg,
        'model': model_cfg,
        'optimizer': optimizer_cfg,
        'training': training_cfg,
        'datasz': datasz_cfg,
        'preprocess': preprocess_cfg,
        'log_path': 'result',
        'save_learnable_graph': False,
        'save_attn_weights': False,
    })
    return cfg


def load_mat_data(mat_path):
    """加载200×200的.mat文件，返回numpy数组
    
    Args:
        mat_path: .mat文件路径
    
    Returns:
        data: (200, 200) numpy数组
    """
    # 加载.mat文件（自动处理不同的key，优先取200×200的矩阵）
    mat_data = loadmat(mat_path)
    for key in mat_data.keys():
        # 跳过mat文件的默认key
        if key in ['__header__', '__version__', '__globals__']:
            continue
        arr = mat_data[key]
        if arr.shape == (200, 200):
            return arr.astype(np.float32)
    
    # 若未找到200×200的矩阵，抛出异常
    raise ValueError(f"未在 {mat_path} 中找到200×200的矩阵！当前mat文件中的矩阵形状：{[k+':'+str(v.shape) for k,v in mat_data.items() if not k.startswith('__')]}")


def preprocess_single_sample(mat_data, dataset_name, model_yaml_name, device):
    """预处理单样本数据，适配模型输入格式
    
    核心：补充模型所需的维度（batch维度、时间序列维度等）
    假设200×200矩阵是节点特征（node_feature），时间序列（time_series）用默认值/随机值（根据模型要求调整）
    
    Args:
        mat_data: (200, 200) numpy数组
        dataset_name: 数据集名称（ABIDE/MDD等）
        model_yaml_name: 模型配置文件名
        device: 计算设备
    
    Returns:
        time_series: 模型所需的时间序列输入 (1, T, N) 或 (1, N, T)
        node_feature: 节点特征 (1, 200, 200)
        partial_node_feature: 部分节点特征（若模型需要）
    """
    cfg = build_cfg(dataset_name, model_yaml_name)
    
    # 1. 处理node_feature（核心：200×200矩阵）
    node_feature = torch.from_numpy(mat_data).unsqueeze(0).to(device)  # (1, 200, 200)
    
    # 2. 处理time_series（根据模型配置补充，若模型不需要可设为全0）
    # 先获取模型配置中的时间序列长度/节点数
    time_series_len = cfg.dataset.get('time_series_len', 100)  # 可根据实际配置调整
    node_sz = cfg.dataset.get('node_sz', 200)
    
    # 生成适配的time_series（若有真实时间序列可替换，这里用全0/随机值占位）
    # 形状：(batch=1, time_series_len, node_sz) 或 (1, node_sz, time_series_len)，根据模型要求调整
    time_series = torch.zeros(1, time_series_len, node_sz, dtype=torch.float32).to(device)
    
    # 3. 处理partial_node_feature（若模型需要，否则设为全0）
    partial_node_feature = torch.zeros_like(node_feature).to(device)
    
    return time_series, node_feature, partial_node_feature


def run_single_inference(model_yaml_name, dataset_name, weights_path, mat_data, device):
    """加载模型并对单样本.mat数据做推理
    
    Args:
        model_yaml_name: 模型配置文件名
        dataset_name: 数据集名称
        weights_path: 模型权重路径
        mat_data: (200, 200) numpy数组
        device: 计算设备
    
    Returns:
        prob: (2,) tensor，softmax后的预测概率 [阴性概率, 阳性概率]
        pred_label: int，0=阴性，1=阳性
    """
    # 构建配置
    cfg = build_cfg(dataset_name, model_yaml_name)
    
    # 初始化模型并加载权重
    model = model_factory(cfg)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 预处理单样本数据
    time_series, node_feature, partial_node_feature = preprocess_single_sample(
        mat_data, dataset_name, model_yaml_name, device
    )
    
    # 模型推理
    with torch.no_grad():
        use_partial = cfg.model.get('use_partial', False)
        use_pool_loss = cfg.model.get('use_pool_loss', False)
        
        # 前向传播
        if use_partial:
            output = model(time_series, node_feature, partial_node_feature)
        else:
            output = model(time_series, node_feature)
        
        # 解析输出
        if use_pool_loss:
            predict, _ = output
        else:
            predict = output
        
        # 计算softmax概率
        prob = F.softmax(predict, dim=1).squeeze(0)  # (2,)
        pred_label = torch.argmax(prob).item()       # 0/1
    
    return prob, pred_label


def entropy(p):
    """计算预测熵（衡量置信度），p: (2,) tensor"""
    return -(p * (p + 1e-8).log()).sum().item()


def hard_vote_single(all_probs):
    """单样本硬投票：>=2/3模型预测阳性则判为阳性
    
    Args:
        all_probs: list of (2,) tensor，各模型预测概率
    
    Returns:
        final_label: int，0/1
        vote_score: float，投票比例（0, 1/3, 2/3, 1）
        votes: list of int，各模型的预测标签
    """
    votes = [torch.argmax(p).item() for p in all_probs]
    vote_count = sum(votes)
    final_label = 1 if vote_count >= 2 else 0
    vote_score = vote_count / len(all_probs)
    return final_label, vote_score, votes


def entropy_weighted_soft_vote_single(all_probs):
    """单样本熵加权软投票
    
    Args:
        all_probs: list of (2,) tensor，各模型预测概率
    
    Returns:
        final_label: int，0/1
        weighted_prob: float，加权后的阳性概率
        weights: list of float，各模型的权重
    """
    # 计算各模型的熵和权重
    entropies = [entropy(p) for p in all_probs]
    total_h = sum(entropies)
    weights = [(total_h - h) for h in entropies]
    total_w = sum(weights)
    
    # 计算加权阳性概率
    weighted_prob = sum(w * p[1].item() for w, p in zip(weights, all_probs)) / total_w
    final_label = 1 if weighted_prob > 0.5 else 0
    
    # 归一化权重（便于展示）
    weights = [w / total_w for w in weights]
    return final_label, weighted_prob, weights


def print_single_result(mat_path, dataset_name, all_probs, model_names):
    """打印单样本推理的详细结果"""
    print(f"\n{'='*60}")
    print(f"  单样本推理结果 - {os.path.basename(mat_path)}")
    print(f"  数据集配置: {dataset_name}")
    print(f"{'='*60}")
    
    # 打印各模型单独预测结果
    print("\n【各模型单独预测】")
    for idx, (model_name, prob) in enumerate(zip(model_names, all_probs)):
        neg_prob = prob[0].item()
        pos_prob = prob[1].item()
        pred_label = 1 if pos_prob > 0.5 else 0
        label_str = "阳性" if pred_label == 1 else "阴性"
        entropy_val = entropy(prob)
        print(f"  {model_name:20s}: {label_str} | 阴性概率: {neg_prob:.4f} | 阳性概率: {pos_prob:.4f} | 熵（置信度）: {entropy_val:.4f}")
    
    # 硬投票结果
    hard_label, hard_score, hard_votes = hard_vote_single(all_probs)
    hard_label_str = "阳性" if hard_label == 1 else "阴性"
    print(f"\n【硬投票结果】")
    print(f"  各模型投票: {[model_names[i]+'='+('阳性' if v==1 else '阴性') for i,v in enumerate(hard_votes)]}")
    print(f"  最终预测: {hard_label_str} | 投票得分: {hard_score:.2f} (≥0.67为阳性)")
    
    # 熵加权软投票结果
    soft_label, soft_prob, soft_weights = entropy_weighted_soft_vote_single(all_probs)
    soft_label_str = "阳性" if soft_label == 1 else "阴性"
    print(f"\n【熵加权软投票结果】")
    print(f"  各模型权重: {[f'{model_names[i]}={w:.4f}' for i,w in enumerate(soft_weights)]}")
    print(f"  加权阳性概率: {soft_prob:.4f} | 最终预测: {soft_label_str} (>0.5为阳性)")
    
    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='单.mat文件投票集成推理')
    parser.add_argument('--mat_path', type=str, required=True,
                        help='200×200的.mat文件路径')
    parser.add_argument('--dataset', type=str, required=True, choices=['ABIDE', 'ABIDE_NET', 'MDD'], 
                        help='数据集配置名称（用于加载对应模型配置）')
    parser.add_argument('--weights_dir', type=str, default=None,
                        help='权重文件目录（默认: ../../weights/）')
    args = parser.parse_args()

    # 检查.mat文件是否存在
    if not os.path.exists(args.mat_path):
        print(f"[ERROR] .mat文件不存在: {args.mat_path}")
        sys.exit(1)
    
    # 设置权重目录
    if args.weights_dir is None:
        args.weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', '..', 'weights')
    args.weights_dir = os.path.abspath(args.weights_dir)

    # 初始化设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 检查权重文件
    weight_paths = {}
    model_names = []
    for yaml_name, weight_prefix in MODELS:
        path = os.path.join(args.weights_dir, f'{weight_prefix}_{args.dataset}.pt')
        if not os.path.exists(path):
            print(f"[ERROR] 权重文件不存在: {path}")
            sys.exit(1)
        weight_paths[yaml_name] = path
        model_names.append(weight_prefix)
        print(f"加载权重: {weight_prefix} -> {path}")

    # 加载.mat数据
    print(f"\n加载.mat文件: {args.mat_path}")
    mat_data = load_mat_data(args.mat_path)
    print(f"数据形状: {mat_data.shape} (预期: (200, 200))")

    # 各模型单独推理
    all_probs = []
    for yaml_name, weight_prefix in MODELS:
        print(f"\n推理模型: {weight_prefix}...")
        prob, pred_label = run_single_inference(
            yaml_name, args.dataset, weight_paths[yaml_name], mat_data, device
        )
        all_probs.append(prob)
        print(f"  {weight_prefix} 预测: {'阳性' if pred_label == 1 else '阴性'} (阳性概率: {prob[1].item():.4f})")

    # 打印综合投票结果
    print_single_result(args.mat_path, args.dataset, all_probs, model_names)


if __name__ == '__main__':
    main()