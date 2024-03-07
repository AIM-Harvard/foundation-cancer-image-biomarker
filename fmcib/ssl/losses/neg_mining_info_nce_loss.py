# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pprint

import numpy as np
import torch
from lightly.utils import dist
from loguru import logger
from torch import nn


class NegativeMiningInfoNCECriterion(nn.Module):
    """
    The criterion corresponding to the SimCLR loss as defined in the paper
    https://arxiv.org/abs/2002.05709.

    Args:
        temperature (float): The temperature to be applied on the logits.
        buffer_params (dict): A dictionary containing the following keys:
            - world_size (int): Total number of trainers in training.
            - embedding_dim (int): Output dimensions of the features projects.
            - effective_batch_size (int): Total batch size used (includes positives).
    """

    def __init__(
        self, embedding_dim, batch_size, world_size, gather_distributed=False, temperature: float = 0.1, balanced: bool = True
    ):
        """
        Initialize the NegativeMiningInfoNCECriterion class.

        Args:
            embedding_dim (int): The dimension of the embedding space.
            batch_size (int): The size of the input batch.
            world_size (int): The number of distributed processes.
            gather_distributed (bool): Whether to gather distributed data.
            temperature (float): The temperature used in the computation.
            balanced (bool): Whether to use balanced sampling.

        Attributes:
            embedding_dim (int): The dimension of the embedding space.
            use_gpu (bool): Whether to use GPU for computations.
            temperature (float): The temperature used in the computation.
            num_pos (int): The number of positive samples.
            num_neg (int): The number of negative samples.
            criterion (nn.CrossEntropyLoss): The loss function.
            gather_distributed (bool): Whether to gather distributed data.
            world_size (int): The number of distributed processes.
            effective_batch_size (int): The effective batch size, taking into account world size and number of positive samples.
            pos_mask (None or Tensor): Mask for positive samples.
            neg_mask (None or Tensor): Mask for negative samples.
            balanced (bool): Whether to use balanced sampling.
            setup (bool): Whether the setup has been done.
        """
        super(NegativeMiningInfoNCECriterion, self).__init__()
        self.embedding_dim = embedding_dim
        self.use_gpu = torch.cuda.is_available()
        self.temperature = temperature
        self.num_pos = 2

        # Same number of negatives as positives are loaded
        self.num_neg = self.num_pos
        self.criterion = nn.CrossEntropyLoss()
        self.gather_distributed = gather_distributed
        self.world_size = world_size
        self.effective_batch_size = batch_size * self.world_size * self.num_pos
        self.pos_mask = None
        self.neg_mask = None
        self.balanced = balanced
        self.setup = False

    def precompute_pos_neg_mask(self):
        """
        Precompute the positive and negative masks to speed up the loss calculation.
        """
        # computed once at the begining of training

        # total_images is x2 SimCLR Info-NCE loss
        # as we have negative samples for each positive sample

        total_images = self.effective_batch_size * self.num_neg
        world_size = self.world_size

        # Batch size computation is different from SimCLR paper
        batch_size = self.effective_batch_size // world_size
        orig_images = batch_size // self.num_pos
        rank = dist.rank()

        pos_mask = torch.zeros(batch_size * self.num_neg, total_images)
        neg_mask = torch.zeros(batch_size * self.num_neg, total_images)

        all_indices = np.arange(total_images)

        # Index for pairs of images (original + copy)
        pairs = orig_images * np.arange(self.num_pos)

        # Remove all indices associated with positive samples & copies (for neg_mask)
        all_pos_members = []
        for _rank in range(world_size):
            all_pos_members += list(_rank * (batch_size * 2) + np.arange(batch_size))

        all_indices_pos_removed = np.delete(all_indices, all_pos_members)

        # Index of original positive images
        orig_members = torch.arange(orig_images)

        for anchor in np.arange(self.num_pos):
            for img_idx in range(orig_images):
                # delete_inds are spaced by batch_size for each rank as
                # all_indices_pos_removed (half of the indices) is deleted first
                delete_inds = batch_size * rank + img_idx + pairs
                neg_inds = torch.tensor(np.delete(all_indices_pos_removed, delete_inds)).long()
                neg_mask[anchor * orig_images + img_idx, neg_inds] = 1

            for pos in np.delete(np.arange(self.num_pos), anchor):
                # Pos_inds are spaced by batch_size * self.num_neg for each rank
                pos_inds = (batch_size * self.num_neg) * rank + pos * orig_images + orig_members
                pos_mask[
                    torch.arange(anchor * orig_images, (anchor + 1) * orig_images).long(),
                    pos_inds.long(),
                ] = 1

        self.pos_mask = pos_mask.cuda(non_blocking=True) if self.use_gpu else pos_mask
        self.neg_mask = neg_mask.cuda(non_blocking=True) if self.use_gpu else neg_mask

    def forward(self, out: torch.Tensor):
        """
        Calculate the loss. Operates on embeddings tensor.
        """
        if not self.setup:
            logger.info(f"Running Negative Mining Info-NCE loss on Rank: {dist.rank()}")
            self.precompute_pos_neg_mask()
            self.setup = True

        pos0, pos1 = out["positive"]
        neg0, neg1 = out["negative"]
        embedding = torch.cat([pos0, pos1, neg0, neg1], dim=0)
        embedding = nn.functional.normalize(embedding, dim=1, p=2)
        assert embedding.ndim == 2
        assert embedding.shape[1] == int(self.embedding_dim)

        batch_size = embedding.shape[0]
        T = self.temperature
        num_pos = self.num_pos

        assert batch_size % num_pos == 0, "Batch size should be divisible by num_pos"
        assert batch_size == self.pos_mask.shape[0], "Batch size should be equal to pos_mask shape"

        # Step 1: gather all the embeddings. Shape example: 4096 x 128
        embeddings_buffer = self.gather_embeddings(embedding)

        # Step 2: matrix multiply: 64 x 128 with 4096 x 128 = 64 x 4096 and
        # divide by temperature.
        similarity = torch.exp(torch.mm(embedding, embeddings_buffer.t()) / T)

        pos = torch.sum(similarity * self.pos_mask, 1)
        neg = torch.sum(similarity * self.neg_mask, 1)

        # Ignore the negative samples as entries for loss calculation
        pos = pos[: (batch_size // 2)]
        neg = neg[: (batch_size // 2)]

        loss = -(torch.mean(torch.log(pos / (pos + neg))))
        return loss

    def __repr__(self):
        """
        Return a string representation of the object.

        Returns:
            str: A formatted string representation of the object.

        Examples:
            The following example shows the string representation of the object:

            {
              'name': <object_name>,
              'temperature': <temperature_value>,
              'num_negatives': <num_negatives_value>,
              'num_pos': <num_pos_value>,
              'dist_rank': <dist_rank_value>
            }

        Note:
            This function is intended to be used with the pprint module for pretty printing.
        """
        num_negatives = self.effective_batch_size - 2
        T = self.temperature
        num_pos = self.num_pos
        repr_dict = {
            "name": self._get_name(),
            "temperature": T,
            "num_negatives": num_negatives,
            "num_pos": num_pos,
            "dist_rank": dist.rank(),
        }
        return pprint.pformat(repr_dict, indent=2)

    def gather_embeddings(self, embedding: torch.Tensor):
        """
        Do a gather over all embeddings, so we can compute the loss.
        Final shape is like: (batch_size * num_gpus) x embedding_dim
        """
        if self.gather_distributed:
            embedding_gathered = torch.cat(dist.gather(embedding), 0)
        else:
            embedding_gathered = embedding
        return embedding_gathered
