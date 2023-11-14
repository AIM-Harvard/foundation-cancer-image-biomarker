from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from lightly.models.modules import NNCLRPredictionHead, NNCLRProjectionHead, NNMemoryBankModule


class NNCLR(nn.Module):
    """
    Taken largely from https://github.com/lightly-ai/lightly/blob/master/lightly/models/nnclr.py
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_ftrs: int = 4096,
        proj_hidden_dim: int = 4096,
        pred_hidden_dim: int = 4096,
        out_dim: int = 256,
        memory_bank_size: int = 4096,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.projection_head = NNCLRProjectionHead(num_ftrs, proj_hidden_dim, out_dim)
        self.prediction_head = NNCLRPredictionHead(out_dim, pred_hidden_dim, out_dim)
        self.memory_bank = NNMemoryBankModule(memory_bank_size)

    def forward(
        self,
        x: List[torch.Tensor],
        get_nearest_neighbor: bool = True,
    ):
        # forward pass of first input x0
        x0, x1 = x
        f0 = self.backbone(x0).flatten(start_dim=1)
        z0 = self.projection_head(f0)
        p0 = self.prediction_head(z0)

        if get_nearest_neighbor:
            z0 = self.memory_bank(z0, update=False)

        # forward pass of second input x1
        f1 = self.backbone(x1).flatten(start_dim=1)
        z1 = self.projection_head(f1)
        p1 = self.prediction_head(z1)

        if get_nearest_neighbor:
            z1 = self.memory_bank(z1, update=True)

        return (z0, p0), (z1, p1)
