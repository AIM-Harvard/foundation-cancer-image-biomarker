import torch
import torch.nn as nn
from lightly.models import SimCLR as lightly_SimCLR
from lightly.models.modules import SimCLRProjectionHead


class SimCLR(lightly_SimCLR):
    """
    A class representing a SimCLR model.

    Attributes:
        backbone (nn.Module): The backbone model used in the SimCLR model.
        num_ftrs (int): The number of output features from the backbone model.
        out_dim (int): The dimension of the output representations.
        projection_head (SimCLRProjectionHead): The projection head used for projection head training.
    """

    def __init__(self, backbone: nn.Module, num_ftrs: int = 32, out_dim: int = 128):
        """
        Initialize the object with a backbone network, number of features, and output dimension.

        Args:
            backbone (nn.Module): The backbone network.
            num_ftrs (int): The number of features. Default is 32.
            out_dim (int): The output dimension. Default is 128.

        Returns:
            None

        Raises:
            None
        """
        super().__init__(backbone, num_ftrs, out_dim)
        self.projection_head = SimCLRProjectionHead(num_ftrs, num_ftrs // 2, out_dim, batch_norm=False)

    def forward(self, x, return_features=False):
        """
        Perform a forward pass of the neural network.

        Args:
            x (tuple): A tuple of input data. Each element of the tuple represents a different input.
            return_features (bool, optional): Whether to return the intermediate features. Default is False.

        Returns:
            torch.Tensor or tuple: The output of the forward pass. If return_features is False, a single tensor is returned.
                If return_features is True, a tuple is returned consisting of the output tensor and the intermediate features.

        Raises:
            None.
        """
        return super().forward(*x, return_features)
