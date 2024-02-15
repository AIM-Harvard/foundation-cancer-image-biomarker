""" Contrastive Loss Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

# Modified to function for explicitly selected negatives

from typing import Dict

import torch
from lightly.utils import dist
from torch import nn


class NTXentNegativeMinedLoss(torch.nn.Module):
    """
    NTXentNegativeMinedLoss:
    NTXentLoss with explicitly mined negatives

    Args:
        temperature (float): The temperature parameter for the loss calculation. Default is 0.1.
        gather_distributed (bool): Whether to gather hidden representations from other processes in a distributed setting. Default is False.

    Raises:
        ValueError: If the absolute value of temperature is less than 1e-8.
    """

    def __init__(self, temperature: float = 0.1, gather_distributed: bool = False):
        """
        Initialize the NTXentNegativeMinedLoss object.

        Args:
            temperature (float, optional): The temperature parameter for the loss function. Defaults to 0.1.
            gather_distributed (bool, optional): Whether to use distributed gathering or not. Defaults to False.

        Raises:
            ValueError: If the absolute value of the temperature is too small.

        Attributes:
            temperature (float): The temperature parameter for the loss function.
            gather_distributed (bool): Whether to use distributed gathering or not.
            cross_entropy (torch.nn.CrossEntropyLoss): The cross entropy loss function.
            eps (float): A small value to avoid division by zero.
        """
        super(NTXentNegativeMinedLoss, self).__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.eps = 1e-8

        if abs(self.temperature) < self.eps:
            raise ValueError("Illegal temperature: abs({}) < 1e-8".format(self.temperature))

    def forward(self, out: Dict):
        """
        Forward pass through Negative mining contrastive Cross-Entropy Loss.

        Args:
            out (Dict): Dictionary with `positive` and `negative` keys to represent positive selected and negative selected samples.

        Returns:
            torch.Tensor: Contrastive Cross Entropy Loss value.

        Raises:
            AssertionError: If `positive` or `negative` keys are not specified in the input dictionary.
        """

        assert "positive" in out, "`positive` key needs to be specified"
        assert "negative" in out, "`negative` key needs to be specified"

        pos0, pos1 = out["positive"]
        neg0, neg1 = out["negative"]

        device = pos0.device
        batch_size, _ = pos0.shape

        # normalize the output to length 1
        pos0 = nn.functional.normalize(pos0, dim=1)
        pos1 = nn.functional.normalize(pos1, dim=1)
        neg0 = nn.functional.normalize(neg0, dim=1)
        neg1 = nn.functional.normalize(neg1, dim=1)

        if self.gather_distributed and dist.world_size() > 1:
            # gather hidden representations from other processes
            pos0_large = torch.cat(dist.gather(pos0), 0)
            pos1_large = torch.cat(dist.gather(pos1), 0)
            neg0_large = torch.cat(dist.gather(neg0), 0)
            neg1_large = torch.cat(dist.gather(neg1), 0)
            diag_mask = dist.eye_rank(batch_size, device=pos0.device)

        else:
            # gather hidden representations from other processes
            pos0_large = pos0
            pos1_large = pos1
            neg0_large = neg0
            neg1_large = neg1
            diag_mask = torch.eye(batch_size, device=pos0.device, dtype=torch.bool)

        logits_00 = torch.einsum("nc,mc->nm", pos0, neg0_large) / self.temperature
        logits_01 = torch.einsum("nc,mc->nm", pos0, pos1_large) / self.temperature
        logits_10 = torch.einsum("nc,mc->nm", pos1, pos0_large) / self.temperature
        logits_11 = torch.einsum("nc,mc->nm", pos1, neg1_large) / self.temperature

        logits_01 = logits_01[diag_mask].view(batch_size, -1)
        logits_10 = logits_10[diag_mask].view(batch_size, -1)

        logits_0100 = torch.cat([logits_01, logits_00], dim=1)
        logits_1011 = torch.cat([logits_10, logits_11], dim=1)
        logits = torch.cat([logits_0100, logits_1011], dim=0)

        labels = torch.zeros(logits.shape[0], device=device, dtype=torch.long)
        loss = self.cross_entropy(logits, labels)

        return loss
