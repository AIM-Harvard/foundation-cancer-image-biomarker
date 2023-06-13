from typing import Dict, Union

import torch
import torch.nn as nn
from lightly.models import SimCLR


class ExNegSimCLR(SimCLR):
    def __init__(self, backbone: nn.Module, num_ftrs: int = 32, out_dim: int = 128) -> None:
        print(backbone)
        super().__init__(backbone, num_ftrs, out_dim)

    def forward(self, x: Union[Dict, torch.Tensor], return_features: bool = False):
        assert isinstance(x, dict), "Input to forward must be a `dict` for ExNegSimCLR"
        out = {}
        for key, value in x.items():
            if isinstance(value, list):
                out[key] = super().forward(*value, return_features)

        return out
