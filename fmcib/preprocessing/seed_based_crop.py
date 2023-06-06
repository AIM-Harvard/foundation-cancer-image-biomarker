"""
Author: Suraj Pai
Email: bspai@bwh.harvard.edu
This script contains two classes:
1. SeedBasedPatchCropd
2. SeedBasedPatchCrop
"""

from typing import Any, Dict, Hashable, Mapping, Tuple

import numpy as np
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import MapTransform, Transform


class SeedBasedPatchCropd(MapTransform):
    def __init__(self, keys, roi_size, allow_missing_keys=False, coord_orientation="RAS", global_coordinates=True) -> None:
        """
        Initialize SeedBasedPatchCropd class.

        Args:
            keys: List of keys for images in the input data dictionary.
            roi_size: Tuple indicating the size of the region of interest (ROI).
            allow_missing_keys: If True, do not raise an error if some keys in the input data dictionary are missing.
            coord_orientation: Coordinate system (RAS or LPS) of input coordinates.
            global_coordinates: If True, coordinates are in global coordinates; otherwise, local coordinates.
        """
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.coord_orientation = coord_orientation
        self.global_coordinates = global_coordinates
        self.cropper = SeedBasedPatchCrop(roi_size=roi_size)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        """
        Apply transformation to given data.

        Args:
            data: Dictionary with image keys and required center coordinates.

        Returns:
            Dictionary with cropped patches for each key in the input data dictionary.
        """
        d = dict(data)

        assert "coordX" in d.keys(), "coordX not found in data"
        assert "coordY" in d.keys(), "coordY not found in data"
        assert "coordZ" in d.keys(), "coordZ not found in data"

        # Convert coordinates to RAS orientation to match image orientation
        if self.coord_orientation == "RAS":
            center = (d["coordX"], d["coordY"], d["coordZ"])
        elif self.coord_orientation == "LPS":
            center = (-d["coordX"], -d["coordY"], d["coordZ"])

        for key in self.key_iterator(d):
            d[key] = self.cropper(d[key], center=center, global_coordinates=self.global_coordinates)
        return d


class SeedBasedPatchCrop(Transform):
    def __init__(self, roi_size) -> None:
        """
        Initialize SeedBasedPatchCrop class.

        Args:
            roi_size: Tuple indicating the size of the region of interest (ROI).
        """
        super().__init__()
        self.roi_size = roi_size

    def __call__(self, img: NdarrayOrTensor, center: tuple, global_coordinates=False) -> NdarrayOrTensor:
        """
        Crop a patch from the input image centered around the seed coordinate.

        Args:
            img: Image to crop, with dimensions (C, H, W, D). C is the number of channels.
            center: Seed coordinate around which to crop the patch (X, Y, Z).
            global_coordinates: If True, seed coordinate is in global space; otherwise, local space.

        Returns:
            NdarrayOrTensor: Cropped patch of shape (C, Ph, Pw, Pd), where (Ph, Pw, Pd) is the patch size.
        """
        assert len(img.shape) == 4, "Input image must have dimensions: (C, H, W, D)"
        C, H, W, D = img.shape
        Ph, Pw, Pd = self.roi_size

        # If global coordinates, convert to local coordinates
        if global_coordinates:
            center = np.linalg.inv(np.array(img.affine)) @ np.array(center + (1,))
            center = [int(x) for x in center[:3]]

        # Calculate and clamp the ranges for cropping
        min_h, max_h = max(center[0] - Ph // 2, 0), min(center[0] + Ph // 2, H)
        min_w, max_w = max(center[1] - Pw // 2, 0), min(center[1] + Pw // 2, W)
        min_d, max_d = max(center[2] - Pd // 2, 0), min(center[2] + Pd // 2, D)

        # Check if coordinates are valid
        assert min_h < max_h, "Invalid coordinates: min_h >= max_h"
        assert min_w < max_w, "Invalid coordinates: min_w >= max_w"
        assert min_d < max_d, "Invalid coordinates: min_d >= max_d"

        # Crop the patch from the image
        patch = img[:, min_h:max_h, min_w:max_w, min_d:max_d]

        return patch
