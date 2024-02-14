from lightly.loss import NTXentLoss


class NNCLRLoss(NTXentLoss):
    """
    A class representing the NNCLRLoss.

    This class extends the NTXentLoss class and implements a symmetric loss function for NNCLR.

    Attributes:
        temperature (float): The temperature for the loss function. Default is 0.1.
        gather_distributed (bool): A flag indicating whether the distributed gathering is used. Default is False.
    """

    def __init__(self, temperature: float = 0.1, gather_distributed: bool = False):
        """
        Initialize a new instance of the class.

        Args:
            temperature (float): The temperature to use for initialization. Default value is 0.1.
            gather_distributed (bool): Whether to use gather distributed mode. Default value is False.
        """
        super().__init__(temperature, gather_distributed)

    def forward(self, out):
        """
        Symmetric loss function for NNCLR.
        """
        (z0, p0), (z1, p1) = out
        loss0 = super().forward(z0, p0)
        loss1 = super().forward(z1, p1)
        return (loss0 + loss1) / 2
