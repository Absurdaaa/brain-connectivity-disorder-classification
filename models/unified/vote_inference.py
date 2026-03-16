#!/usr/bin/env python3
"""投票集成推理脚本

支持硬投票和熵加权软投票两种策略。
三个模型（ASDFormer, ComBrainTF, BrainNetworkTransformer）在同一测试集上推理，
通过投票融合得到最终预测。

用法:
    python vote_inference.py --dataset ABIDE
    python vote_inference.py --dataset MDD
    python vote_inference.py --dataset ABIDE_NET
    python vote_inference.py --dataset ABIDE --weights_dir /path/to/weights
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from sklearn.metrics import roc_auc_score, classification_report

# 将 source 目录加入 Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source'))

from models import model_factory
from dataset.abide import _load_npy_data
from dataset.dataloader import init_stratified_dataloader, init_dataloader

# 模型配置：(yaml 文件名, 权重文件前缀)
MODELS = [
    ('ASDFormer', 'ASDFormer'),
    ('comtf', 'ComBrainTF'),
    ('bnt', 'BrainNetworkTransformer'),
]


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_cfg(dataset_name, model_yaml_name):
    """手动构建 OmegaConf 配置，不使用 Hydra"""
    conf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source', 'conf')

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


def load_data_and_test_loader(dataset_name):
    """加载数据集，返回固定顺序的 test_dataloader（shuffle=False）。

    使用固定 seed 保证三个模型拿到同一批测试样本，且顺序一致。
    用 ASDFormer 的 cfg 加载数据（数据本身与模型无关）。
    """
    import torch.utils.data as utils

    cfg = build_cfg(dataset_name, 'ASDFormer')

    final_timeseires, final_pearson, final_partial, labels, site = _load_npy_data(cfg)

    # 固定种子保证数据划分一致
    np.random.seed(42)
    torch.manual_seed(42)

    if cfg.dataset.get('stratified', False):
        loaders = init_stratified_dataloader(
            cfg, final_timeseires, final_pearson, final_partial, labels, site)
    else:
        loaders = init_dataloader(
            cfg, final_timeseires, final_pearson, final_partial, labels)

    # 用 shuffle=False 重新包装测试集，保证多次遍历顺序相同
    test_dataset = loaders[2].dataset
    test_loader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False, drop_last=False)

    return test_loader


def run_inference(model_yaml_name, dataset_name, weights_path, test_loader, device):
    """加载模型并在 test_loader 上做推理，返回 softmax 概率 (N, 2) 和真实标签 (N,)"""
    cfg = build_cfg(dataset_name, model_yaml_name)

    # 调用 _load_npy_data 仅为了填充 cfg.dataset 的动态字段（node_sz 等）
    _load_npy_data(cfg)

    model = model_factory(cfg)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    use_partial = cfg.model.get('use_partial', False)
    use_pool_loss = cfg.model.get('use_pool_loss', False)

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for time_series, node_feature, partial_node_feature, label in test_loader:
            time_series = time_series.to(device)
            node_feature = node_feature.to(device)
            partial_node_feature = partial_node_feature.to(device)

            if use_partial:
                output = model(time_series, node_feature, partial_node_feature)
            else:
                output = model(time_series, node_feature)

            if use_pool_loss:
                predict, _ = output
            else:
                predict = output

            probs = F.softmax(predict, dim=1)  # (batch, 2)
            all_probs.append(probs.cpu())
            all_labels.append(label[:, 1].cpu())  # one-hot 的第 1 列即阳性标签

    all_probs = torch.cat(all_probs, dim=0)    # (N, 2)
    all_labels = torch.cat(all_labels, dim=0)   # (N,)
    return all_probs, all_labels


def entropy(p):
    """计算预测熵，p: (N, 2)"""
    return -(p * (p + 1e-8).log()).sum(dim=1)  # (N,)


def compute_metrics(binary_pred, prob_scores, labels):
    """计算 Accuracy, AUC, Sensitivity, Specificity

    Args:
        binary_pred: (N,) 二值预测 (0/1)
        prob_scores: (N,) 用于 AUC 计算的连续分数
        labels: (N,) 真实标签 (0/1)
    """
    labels_np = labels.numpy()
    pred_np = binary_pred.numpy()
    scores_np = prob_scores.numpy()

    acc = (pred_np == labels_np).mean()
    auc = roc_auc_score(labels_np, scores_np)

    report = classification_report(labels_np, pred_np, output_dict=True, zero_division=0)
    sensitivity = report.get('1.0', report.get('1', {})).get('recall', 0.0)
    specificity = report.get('0.0', report.get('0', {})).get('recall', 0.0)

    return acc, auc, sensitivity, specificity


def hard_vote(all_probs_list):
    """硬投票：>=2/3 模型预测阳性 -> 判为阳性

    Args:
        all_probs_list: list of (N, 2) tensors，每个模型的 softmax 概率

    Returns:
        binary_pred: (N,) 二值预测
        vote_scores: (N,) 投票得分（投票比例，用于 AUC）
    """
    votes = torch.stack([p[:, 1] > 0.5 for p in all_probs_list]).float()  # (3, N)
    vote_count = votes.sum(dim=0)                                          # (N,)
    binary_pred = (vote_count >= 2).long()
    vote_scores = vote_count / len(all_probs_list)  # 0, 1/3, 2/3, 1
    return binary_pred, vote_scores


# def entropy_weighted_soft_vote(all_probs_list):
#     """熵加权软投票：用预测熵的倒数作为置信度权重

#     熵越低 = 越有把握 = 权重越高

#     Args:
#         all_probs_list: list of (N, 2) tensors

#     Returns:
#         binary_pred: (N,) 二值预测
#         weighted_prob: (N,) 加权后的阳性概率（用于 AUC）
#     """
#     weights = [1.0 / (entropy(p) + 1e-6) for p in all_probs_list]  # list of (N,)
#     total_w = sum(weights)  # (N,)
#     weighted_prob = sum(w * p[:, 1] for w, p in zip(weights, all_probs_list)) / total_w  # (N,)
#     binary_pred = (weighted_prob > 0.5).long()
#     return binary_pred, weighted_prob
def entropy_weighted_soft_vote(all_probs_list):
    """熵加权软投票：用预测熵的倒数作为置信度权重

    熵越低 = 越有把握 = 权重越高

    Args:
        all_probs_list: list of (N, 2) tensors

    Returns:
        binary_pred: (N,) 二值预测
        weighted_prob: (N,) 加权后的阳性概率（用于 AUC）
    """
    entropies = [entropy(p) for p in all_probs_list]          # list of (N,)
    total_h = sum(entropies)                                   # (N,)  所有模型熵之和
    weights = [total_h - h for h in entropies]                 # w_i = Σ_{j≠i} H_j
    total_w = sum(weights)                                     # (N,)  = (n-1) * total_h
    weighted_prob = sum(w * p[:, 1] for w, p in zip(weights, all_probs_list)) / total_w  # (N,)
    binary_pred = (weighted_prob > 0.5).long()
    return binary_pred, weighted_prob


def print_metrics(strategy_name, acc, auc, sen, spec):
    print(f"\n{'='*50}")
    print(f"  {strategy_name}")
    print(f"{'='*50}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  AUC:         {auc:.4f}")
    print(f"  Sensitivity: {sen:.4f}")
    print(f"  Specificity: {spec:.4f}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='投票集成推理')
    parser.add_argument('--dataset', type=str, required=True, choices=['ABIDE', 'ABIDE_NET', 'MDD'], 
                        help='数据集名称')
    parser.add_argument('--weights_dir', type=str, default=None,
                        help='权重文件目录（默认: ../../weights/）')
    args = parser.parse_args()

    if args.weights_dir is None:
        args.weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', '..', 'weights')
    args.weights_dir = os.path.abspath(args.weights_dir)

    device = get_device()
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Weights dir: {args.weights_dir}")

    # 检查权重文件是否存在
    weight_paths = {}
    for yaml_name, weight_prefix in MODELS:
        path = os.path.join(args.weights_dir, f'{weight_prefix}_{args.dataset}.pt')
        if not os.path.exists(path):
            print(f"[ERROR] 权重文件不存在: {path}")
            sys.exit(1)
        weight_paths[yaml_name] = path
        print(f"  {weight_prefix}: {path}")

    # 加载数据集和测试集
    print("\n加载数据集...")
    test_loader = load_data_and_test_loader(args.dataset)
    print(f"测试集样本数: {len(test_loader.dataset)}")

    # 对每个模型做推理
    all_probs_list = []
    labels = None

    for yaml_name, weight_prefix in MODELS:
        print(f"\n推理模型: {weight_prefix}...")
        probs, model_labels = run_inference(
            yaml_name, args.dataset, weight_paths[yaml_name], test_loader, device)

        if labels is None:
            labels = model_labels
        else:
            assert torch.equal(labels, model_labels), "模型间测试标签不一致！"

        all_probs_list.append(probs)

        # 打印单模型指标
        single_pred = (probs[:, 1] > 0.5).long()
        acc, auc, sen, spec = compute_metrics(single_pred, probs[:, 1], labels)
        print(f"  {weight_prefix} — Acc: {acc:.4f}, AUC: {auc:.4f}, "
              f"Sen: {sen:.4f}, Spec: {spec:.4f}")

    # 硬投票
    hard_pred, hard_scores = hard_vote(all_probs_list)
    acc, auc, sen, spec = compute_metrics(hard_pred, hard_scores, labels)
    print_metrics("硬投票 (Hard Voting, >=2/3)", acc, auc, sen, spec)

    # 熵加权软投票
    ent_pred, ent_prob = entropy_weighted_soft_vote(all_probs_list)
    acc, auc, sen, spec = compute_metrics(ent_pred, ent_prob, labels)
    print_metrics("熵加权软投票 (Entropy-Weighted Soft Voting)", acc, auc, sen, spec)


if __name__ == '__main__':
    main()
