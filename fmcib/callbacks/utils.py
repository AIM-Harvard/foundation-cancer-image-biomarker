from typing import List

import torch


def decollate(data: List[torch.Tensor]):
    """
    Decollate a list of tensors into a list of values.

    Args:
        data (list): A list of batch tensors.

    Returns:
        list: A list of values from the input tensors.

    Raises:
        AssertionError: If the input is not a list of tensors.
    """
    assert isinstance(data, list), "Decollate only implemented for list of `batch` tensors"

    out = []
    for d in data:
        # Handles both cases: multiple elements and single element
        # https://pytorch.org/docs/stable/generated/torch.Tensor.tolist.html
        d = d.tolist()

        out += d
    return out


def handle_image(image):
    """
    Handle image according to specific requirements.

    Args:
        image (tensor): An image tensor.

    Returns:
        tensor: The processed image tensor, based on the input conditions.

    Raises:
        None.
    """
    image = image.squeeze()
    if image.dim() == 3:
        return image[image.shape[0] // 2]
    else:
        return image
