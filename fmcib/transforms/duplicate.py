from typing import Any, Callable, List, Optional, Tuple

from copy import deepcopy

import torch


class Duplicate:
    """Duplicate an input and apply two different transforms. Used for SimCLR primarily."""

    def __init__(self, transforms1: Optional[Callable] = None, transforms2: Optional[Callable] = None):
        """Duplicates an input and applies the given transformations to each copy separately.

        Args:
            transforms1 (Optional[Callable], optional): _description_. Defaults to None.
            transforms2 (Optional[Callable], optional): _description_. Defaults to None.
        """
        # Wrapped into a list if it isn't one already to allow both a
        # list of transforms as well as `torchvision.transform.Compose` transforms.
        self.transforms1 = transforms1
        self.transforms2 = transforms2

    def __call__(self, input: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input (torch.Tensor or any other type supported by the given transforms): Input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: a tuple of two tensors.
        """
        out1, out2 = input, deepcopy(input)
        if self.transforms1 is not None:
            out1 = self.transforms1(out1)
        if self.transforms2 is not None:
            out2 = self.transforms2(out2)
        return (out1, out2)
