import torch
from lightly.loss.memory_bank import MemoryBankModule
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from torch import nn

torch.set_float32_matmul_precision("medium")


class SwaV(nn.Module):
    """
    Implements the SwAV (Swapping Assignments between multiple Views of the same image) model.

    Args:
        backbone (nn.Module): CNN backbone for feature extraction.
        num_ftrs (int): Number of input features for the projection head.
        out_dim (int): Output dimension for the projection head.
        n_prototypes (int): Number of prototypes to compute.
        n_queues (int): Number of memory banks (queues). Should be equal to the number of high-resolution inputs.
        queue_length (int, optional): Length of the memory bank. Defaults to 0.
        start_queue_at_epoch (int, optional): Number of the epoch at which SwaV starts using the queued features. Defaults to 0.
        n_steps_frozen_prototypes (int, optional): Number of steps during which we keep the prototypes fixed. Defaults to 0.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_ftrs: int,
        out_dim: int,
        n_prototypes: int,
        n_queues: int,
        queue_length: int = 0,
        start_queue_at_epoch: int = 0,
        n_steps_frozen_prototypes: int = 0,
    ):
        """
        Initialize a SwaV model.

        Args:
            backbone (nn.Module): The backbone model.
            num_ftrs (int): The number of input features.
            out_dim (int): The dimension of the output.
            n_prototypes (int): The number of prototypes.
            n_queues (int): The number of queues.
            queue_length (int, optional): The length of the queue. Default is 0.
            start_queue_at_epoch (int, optional): The epoch at which to start using the queue. Default is 0.
            n_steps_frozen_prototypes (int, optional): The number of steps to freeze prototypes. Default is 0.

        Returns:
            None

        Attributes:
            backbone (nn.Module): The backbone model.
            projection_head (SwaVProjectionHead): The projection head.
            prototypes (SwaVPrototypes): The prototypes.
            queues (nn.ModuleList, optional): The queues. If n_queues > 0, this will be initialized with MemoryBankModules.
            queue_length (int, optional): The length of the queue.
            num_features_queued (int): The number of features queued.
            start_queue_at_epoch (int): The epoch at which to start using the queue.
        """
        super().__init__()
        # Backbone for feature extraction
        self.backbone = backbone
        # Projection head to project features to a lower-dimensional space
        self.projection_head = SwaVProjectionHead(num_ftrs, num_ftrs // 2, out_dim)
        # SwAV Prototypes module for prototype computation
        self.prototypes = SwaVPrototypes(out_dim, n_prototypes, n_steps_frozen_prototypes)

        self.queues = None
        if n_queues > 0:
            # Initialize the memory banks (queues)
            self.queues = nn.ModuleList([MemoryBankModule(size=queue_length) for _ in range(n_queues)])
            self.queue_length = queue_length
            self.num_features_queued = 0
            self.start_queue_at_epoch = start_queue_at_epoch

    def forward(self, input, epoch=None, step=None):
        """
        Performs the forward pass for the SwAV model.

        Args:
            input (Tuple[List[Tensor], List[Tensor]]): A tuple consisting of a list of high-resolution input images
                and a list of low-resolution input images.
            epoch (int, optional): Current training epoch. Required if `start_queue_at_epoch` > 0. Defaults to None.
            step (int, optional): Current training step. Required if `n_steps_frozen_prototypes` > 0. Defaults to None.

        Returns:
            Tuple[List[Tensor], List[Tensor], List[Tensor]]: A tuple containing lists of high-resolution prototypes,
                low-resolution prototypes, and queue prototypes.
        """
        high_resolution, low_resolution = input

        # Normalize prototypes
        self.prototypes.normalize()

        # Compute high and low resolution features
        high_resolution_features = [self._subforward(x) for x in high_resolution]
        low_resolution_features = [self._subforward(x) for x in low_resolution]

        # Compute prototypes for high and low resolution features
        high_resolution_prototypes = [self.prototypes(x, epoch) for x in high_resolution_features]
        low_resolution_prototypes = [self.prototypes(x, epoch) for x in low_resolution_features]
        # Compute prototypes for queued features
        queue_prototypes = self._get_queue_prototypes(high_resolution_features, epoch)

        return high_resolution_prototypes, low_resolution_prototypes, queue_prototypes

    def _subforward(self, input):
        """
        Subforward pass to compute features for the input image.

        Args:
            input (Tensor): Input image tensor.

        Returns:
            Tensor: L2-normalized feature tensor.
        """
        # Extract features using the backbone
        features = self.backbone(input).flatten(start_dim=1)
        # Project features using the projection head
        features = self.projection_head(features)
        # L2-normalize features
        features = nn.functional.normalize(features, dim=1, p=2)
        return features

    @torch.no_grad()
    def _get_queue_prototypes(self, high_resolution_features, epoch=None):
        """
        Compute the queue prototypes for the given high-resolution features.

        Args:
            high_resolution_features (List[Tensor]): List of high-resolution feature tensors.
            epoch (int, optional): Current epoch number. Required if `start_queue_at_epoch` > 0. Defaults to None.

        Returns:
            List[Tensor] or None: List of queue prototype tensors if conditions are met, otherwise None.
        """
        if self.queues is None:
            return None

        if len(high_resolution_features) != len(self.queues):
            raise ValueError(
                f"The number of queues ({len(self.queues)}) should be equal to the number of high "
                f"resolution inputs ({len(high_resolution_features)}). Set `n_queues` accordingly."
            )

        # Get the queue features
        queue_features = []
        for i in range(len(self.queues)):
            _, features = self.queues[i](high_resolution_features[i], update=True)
            # Queue features are in (num_ftrs X queue_length) shape, while the high res
            # features are in (batch_size X num_ftrs). Swap the axes for interoperability.
            features = torch.permute(features, (1, 0))
            queue_features.append(features)

        # Do not return queue prototypes if not enough features have been queued
        self.num_features_queued += high_resolution_features[0].shape[0]
        if self.num_features_queued < self.queue_length:
            return None

        # If loss calculation with queue prototypes starts at a later epoch,
        # just queue the features and return None instead of queue prototypes.
        if self.start_queue_at_epoch > 0:
            if epoch is None:
                raise ValueError(
                    "The epoch number must be passed to the `forward()` " "method if `start_queue_at_epoch` is greater than 0."
                )
            if epoch < self.start_queue_at_epoch:
                return None

        # Assign prototypes
        queue_prototypes = [self.prototypes(x, epoch) for x in queue_features]
        # Do not return queue prototypes if not enough features have been queued
        return queue_prototypes
