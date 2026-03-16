from omegaconf import DictConfig, open_dict
from .abide import load_abide_data, load_mdd_data
from .dataloader import init_dataloader, init_stratified_dataloader
from typing import List
import torch.utils as utils


def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:
    assert cfg.dataset.name in ["abide", "mdd", "adhd200", "abide_net"], \
        f"Unsupported dataset: {cfg.dataset.name}. Choose from ['abide', 'mdd', 'adhd200']"

    loaders = {
        "abide": load_abide_data,
        "mdd": load_mdd_data,
        "adhd200": load_abide_data,  # 使用相同的加载器（格式一致）
        "abide_net": load_abide_data,  # 使用相同的加载器（格式一致）
    }
    datasets = loaders[cfg.dataset.name](cfg)

    dataloaders = (
        init_stratified_dataloader(cfg, *datasets)
        if cfg.dataset.stratified
        else init_dataloader(cfg, *datasets)
    )
    return dataloaders
