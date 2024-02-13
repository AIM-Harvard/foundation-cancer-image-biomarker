import torch
import torch.nn as nn
from lightly.models import SimCLR as lightly_SimCLR
from lightly.models.modules import SimCLRProjectionHead


class SimCLR(lightly_SimCLR):
    def __init__(self, backbone: nn.Module, num_ftrs: int = 32, out_dim: int = 128):
        super().__init__(backbone, num_ftrs, out_dim)
        self.projection_head = SimCLRProjectionHead(num_ftrs, num_ftrs // 2, out_dim, batch_norm=False)

    def forward(self, x, return_features=False):
        return super().forward(*x, return_features)
