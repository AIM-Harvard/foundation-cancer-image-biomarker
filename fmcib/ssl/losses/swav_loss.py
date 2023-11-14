import lightly

class SwaVLoss(lightly.loss.swav_loss.SwaVLoss):
    def __init__(self, temperature: float = 0.1, sinkhorn_iterations: int = 3, sinkhorn_epsilon: float = 0.05, sinkhorn_gather_distributed: bool = False):
        super().__init__(temperature, sinkhorn_iterations, sinkhorn_epsilon, sinkhorn_gather_distributed)

    def forward(self, pred):
        high_resolution_outputs, low_resolution_outputs, queue_outputs = pred
        return super().forward(high_resolution_outputs, low_resolution_outputs, queue_outputs)