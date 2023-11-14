from typing import Any, Callable, List, Optional, Tuple

from copy import deepcopy

import torch
from lighter.utils.misc import ensure_list


class MultiCrop:
    """Multi-Crop augmentation."""

    def __init__(self, high_resolution_transforms: List[Callable], low_resolution_transforms: Optional[List[Callable]]):
        self.high_resolution_transforms = ensure_list(high_resolution_transforms)
        self.low_resolution_transforms = ensure_list(low_resolution_transforms)

    def __call__(self, input):
        high_resolution_crops = [transform(input) for transform in self.high_resolution_transforms]
        low_resolution_crops = [transform(input) for transform in self.low_resolution_transforms]
        return high_resolution_crops, low_resolution_crops
