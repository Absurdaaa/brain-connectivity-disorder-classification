from abc import abstractmethod
import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self,
                time_seires: torch.tensor,
                FC_matrix: torch.tensor,
                PC_matrix: torch.tensor,) -> torch.tensor:
        pass
