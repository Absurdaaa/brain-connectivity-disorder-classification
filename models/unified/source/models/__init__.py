from omegaconf import DictConfig
import torch
from .ASDFormer import ASDFormer
from .COMTF import ComBrainTF
from .BNT import BrainNetworkTransformer
from .transformer import GraphTransformer
from .brainnetcnn import BrainNetCNN
from .fbnetgen import FBNETGEN


def model_factory(config: DictConfig):
    model = eval(config.model.name)(config)
    # 自动检测设备：优先 CUDA，其次 MPS（Apple Silicon），最后 CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return model.to(device)
