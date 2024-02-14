from typing import Dict, Union

import torch
import torch.nn as nn
from lightly.models import SimCLR as lightly_SimCLR
from lightly.models.modules import SimCLRProjectionHead


class ExNegSimCLR(lightly_SimCLR):
    """
    Extended Negative Sampling SimCLR model.

    Args:
        backbone (nn.Module): The backbone model.
        num_ftrs (int): Number of features in the bottleneck layer. Default is 32.
        out_dim (int): Dimension of the output feature embeddings. Default is 128.
    """

    def __init__(self, backbone: nn.Module, num_ftrs: int = 32, out_dim: int = 128) -> None:
        """
        Initialize the object.

        Args:
            backbone (nn.Module): The backbone neural network.
            num_ftrs (int, optional): The number of input features for the projection head. Default is 32.
            out_dim (int, optional): The output dimension of the projection head. Default is 128.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(backbone, num_ftrs, out_dim)
        # replace the projection head with a new one
        self.projection_head = SimCLRProjectionHead(num_ftrs, num_ftrs // 2, out_dim, batch_norm=False)

    def forward(self, x: Union[Dict, torch.Tensor], return_features: bool = False):
        """
        Forward pass of the ExNegSimCLR model.

        Args:
            x (Union[Dict, torch.Tensor]): Input data. If a dictionary, it should contain multiple views of the same image.
            return_features (bool): Whether to return the intermediate feature embeddings. Default is False.

        Returns:
            Dict: Output dictionary containing the forward pass results for each input view.
        """
        assert isinstance(x, dict), "Input to forward must be a `dict` for ExNegSimCLR"
        out = {}
        for key, value in x.items():
            if isinstance(value, list):
                out[key] = super().forward(*value, return_features)

        return out
