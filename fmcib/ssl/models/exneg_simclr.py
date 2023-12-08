from typing import Dict, Union

import torch
import torch.nn as nn
from lightly.models import SimCLR


class ExNegSimCLR(SimCLR):
    """
    Extended Negative Sampling SimCLR model.

    Args:
        backbone (nn.Module): The backbone model.
        num_ftrs (int): Number of features in the bottleneck layer. Default is 32.
        out_dim (int): Dimension of the output feature embeddings. Default is 128.
    """

    def __init__(self, backbone: nn.Module, num_ftrs: int = 32, out_dim: int = 128) -> None:
        print(backbone)
        super().__init__(backbone, num_ftrs, out_dim)

    def forward(self, x: Union[Dict, torch.Tensor], return_features: bool = False):
        """
        Forward pass of the ExNegSimCLR model.

        Args:
            x (Union[Dict, torch.Tensor]): Input data. If a dictionary, it should contain multiple views of the same image.
            return_features (bool): Whether to return the intermediate feature embeddings. Default is False.

        Returns:
            out (Dict): Output dictionary containing the forward pass results for each input view.
        """
        assert isinstance(x, dict), "Input to forward must be a `dict` for ExNegSimCLR"
        out = {}
        for key, value in x.items():
            if isinstance(value, list):
                out[key] = super().forward(*value, return_features)

        return out
