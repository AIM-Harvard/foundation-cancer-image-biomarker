from typing import List

import torch


def decollate(data: List[torch.Tensor]):
    assert isinstance(data, list), "Decollate only implemented for list of `batch` tensors"

    out = []
    for d in data:
        # Handles both cases: multiple elements and single element
        # https://pytorch.org/docs/stable/generated/torch.Tensor.tolist.html
        d = d.tolist()

        out += d
    return out
