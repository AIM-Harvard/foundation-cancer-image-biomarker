from lightly.loss import NTXentLoss as lightly_NTXentLoss
from typing import List

class NTXentLoss(lightly_NTXentLoss):
    """
    NTXentNegativeMinedLoss:
    NTXentLoss with explicitly mined negatives
    """

    def __init__(self, temperature: float = 0.1, gather_distributed: bool = False):
        super().__init__(temperature, gather_distributed)

    def forward(self, out: List):
        """Forward pass through Negative mining contrastive Cross-Entropy Loss.

        Args:
            out: List of tensors

        Returns:
            Contrastive Cross Entropy Loss value.

        """
        return super().forward(*out)
