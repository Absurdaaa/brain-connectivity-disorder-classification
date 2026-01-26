import logging
from typing import List
from omegaconf import DictConfig
import torch

def optimizer_factory(model: torch.nn.Module, optimizer_config: DictConfig) -> torch.optim.Optimizer:
    parameters = {
        'lr': optimizer_config.lr,
        'weight_decay': optimizer_config.weight_decay
    }

    params = list(model.parameters())
    logging.info(f'Parameters [normal] length [{len(params)}]')

    parameters['params'] = params

    optimizer_type = optimizer_config.name
    if optimizer_type == 'SGD':
        parameters['momentum'] = optimizer_config.momentum
        parameters['nesterov'] = optimizer_config.nesterov
    return getattr(torch.optim, optimizer_type)(**parameters)


def optimizers_factory(model: torch.nn.Module, optimizer_configs: DictConfig) -> List[torch.optim.Optimizer]:
    if model is None:
        return None
    #return [optimizer_factory(model=model, optimizer_config=single_config) for single_config in optimizer_configs]
    return [optimizer_factory(model=model, optimizer_config=optimizer_configs)]
