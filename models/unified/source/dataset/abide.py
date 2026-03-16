import numpy as np
import torch
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict


def load_abide_data(cfg: DictConfig):
    return _load_npy_data(cfg)


def load_mdd_data(cfg: DictConfig):
    return _load_npy_data(cfg)


def _load_npy_data(cfg: DictConfig):
    data = np.load(cfg.dataset.path, allow_pickle=True).item()
    final_timeseires = data["timeseires"]
    final_pearson = data["corr"]
    final_partial = data["pcorr"]
    labels = data["label"]
    site = data["site"]

    final_pearson = np.nan_to_num(final_pearson)
    final_partial = np.nan_to_num(final_partial)

    scaler = StandardScaler(mean=np.mean(final_timeseires), std=np.std(final_timeseires))
    final_timeseires = scaler.transform(final_timeseires)
    final_timeseires = np.nan_to_num(final_timeseires)

    final_timeseires, final_pearson, final_partial, labels = [
        torch.from_numpy(d).float()
        for d in (final_timeseires, final_pearson, final_partial, labels)
    ]

    with open_dict(cfg):
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]

    return final_timeseires, final_pearson, final_partial, labels, site
