from typing import List

from lightly.loss import NTXentLoss as lightly_NTXentLoss


class NTXentLoss(lightly_NTXentLoss):
    """
    NTXentNegativeMinedLoss:
    NTXentLoss with explicitly mined negatives
    """

    def __init__(self, temperature: float = 0.1, gather_distributed: bool = False):
        """
        Initialize an instance of the class.

        Args:
            temperature (float, optional): The temperature parameter for the instance. Defaults to 0.1.
            gather_distributed (bool, optional): Whether to gather distributed data. Defaults to False.
        """
        super().__init__(temperature, gather_distributed)

    def forward(self, out: List):
        """
        Forward pass through Negative mining contrastive Cross-Entropy Loss.

        Args:
            out (List[torch.Tensor]): List of tensors

        Returns:
            float: Contrastive Cross Entropy Loss value.
        """
        return super().forward(*out)
