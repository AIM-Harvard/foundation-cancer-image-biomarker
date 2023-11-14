from lightly.loss import NTXentLoss


class NNCLRLoss(NTXentLoss):
    def __init__(self, temperature: float = 0.1, gather_distributed: bool = False):
        super().__init__(temperature, gather_distributed)

    def forward(self, out):
        """ "
        Symmetric loss function for NNCLR
        """
        (z0, p0), (z1, p1) = out
        loss0 = super().forward(z0, p0)
        loss1 = super().forward(z1, p1)
        return (loss0 + loss1) / 2
