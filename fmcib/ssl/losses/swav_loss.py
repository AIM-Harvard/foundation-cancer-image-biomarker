import lightly


class SwaVLoss(lightly.loss.swav_loss.SwaVLoss):
    """
    A class representing a custom SwaV loss function.

    Attributes:
        temperature (float): The temperature parameter for the loss calculation. Default is 0.1.
        sinkhorn_iterations (int): The number of iterations for Sinkhorn algorithm. Default is 3.
        sinkhorn_epsilon (float): The epsilon parameter for Sinkhorn algorithm. Default is 0.05.
        sinkhorn_gather_distributed (bool): Whether to gather distributed results for Sinkhorn algorithm. Default is False.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        sinkhorn_iterations: int = 3,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_gather_distributed: bool = False,
    ):
        """
        Initialize the object with specified parameters.

        Args:
            temperature (float, optional): The temperature parameter. Default is 0.1.
            sinkhorn_iterations (int, optional): The number of Sinkhorn iterations. Default is 3.
            sinkhorn_epsilon (float, optional): The epsilon parameter for Sinkhorn algorithm. Default is 0.05.
            sinkhorn_gather_distributed (bool, optional): Whether to use distributed computation for Sinkhorn algorithm. Default is False.
        """
        super().__init__(temperature, sinkhorn_iterations, sinkhorn_epsilon, sinkhorn_gather_distributed)

    def forward(self, pred):
        """
        Perform a forward pass of the model.

        Args:
            pred (tuple): A tuple containing the predicted outputs for high resolution, low resolution, and queue.

        Returns:
            The output of the forward pass.
        """
        high_resolution_outputs, low_resolution_outputs, queue_outputs = pred
        return super().forward(high_resolution_outputs, low_resolution_outputs, queue_outputs)
