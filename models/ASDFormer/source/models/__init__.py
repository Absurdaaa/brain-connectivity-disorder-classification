from omegaconf import DictConfig
from .ASDFormer import ASDFormer


def model_factory(config: DictConfig):
    return eval(config.model.name)(config).cuda()
