from typing import Any, Callable, List, Optional, Tuple

from copy import deepcopy

import torch
from lighter.utils.misc import ensure_list


class MultiCrop:
    """
    Multi-Crop augmentation.
    """

    def __init__(self, high_resolution_transforms: List[Callable], low_resolution_transforms: Optional[List[Callable]]):
        """
        Initialize an instance of a class with transformations for high-resolution and low-resolution images.

        Args:
            high_resolution_transforms (list): A list of Callable objects representing the transformations to be applied to high-resolution images.
            low_resolution_transforms (list, optional): A list of Callable objects representing the transformations to be applied to low-resolution images. Default is None.
        """
        self.high_resolution_transforms = ensure_list(high_resolution_transforms)
        self.low_resolution_transforms = ensure_list(low_resolution_transforms)

    def __call__(self, input):
        """
        This function applies a set of transformations to an input image and returns high and low-resolution crops.

        Args:
            input (image): The input image to be transformed.

        Returns:
            tuple: A tuple containing two lists:
                - high_resolution_crops (list): A list of high-resolution cropped images.
                - low_resolution_crops (list): A list of low-resolution cropped images.
        """
        high_resolution_crops = [transform(input) for transform in self.high_resolution_transforms]
        low_resolution_crops = [transform(input) for transform in self.low_resolution_transforms]
        return high_resolution_crops, low_resolution_crops
