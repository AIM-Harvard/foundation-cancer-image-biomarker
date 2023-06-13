from pathlib import Path

import numpy as np
import SimpleITK as sitk

# https://github.com/SimpleITK/SlicerSimpleFilters/blob/master/SimpleFilters/SimpleFilters.py
SITK_INTERPOLATOR_DICT = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "gaussian": sitk.sitkGaussian,
    "label_gaussian": sitk.sitkLabelGaussian,
    "bspline": sitk.sitkBSpline,
    "hamming_sinc": sitk.sitkHammingWindowedSinc,
    "cosine_windowed_sinc": sitk.sitkCosineWindowedSinc,
    "welch_windowed_sinc": sitk.sitkWelchWindowedSinc,
    "lanczos_windowed_sinc": sitk.sitkLanczosWindowedSinc,
}


def resample_image_to_spacing(image, new_spacing, default_value, interpolator="linear"):
    """Resample an image to a new spacing."""
    assert interpolator in SITK_INTERPOLATOR_DICT, (
        f"Interpolator '{interpolator}' not part of SimpleITK. "
        f"Please choose one of the following {list(SITK_INTERPOLATOR_DICT.keys())}."
    )

    assert image.GetDimension() == len(new_spacing), (
        f"Input is {image.GetDimension()}-dimensional while " f"the new spacing is {len(new_spacing)}-dimensional."
    )

    interpolator = SITK_INTERPOLATOR_DICT[interpolator]
    spacing = image.GetSpacing()
    size = image.GetSize()
    new_size = [int(round(siz * spac / n_spac)) for siz, spac, n_spac in zip(size, spacing, new_spacing)]
    return sitk.Resample(
        image,
        new_size,  # size
        sitk.Transform(),  # transform
        interpolator,  # interpolator
        image.GetOrigin(),  # outputOrigin
        new_spacing,  # outputSpacing
        image.GetDirection(),  # outputDirection
        default_value,  # defaultPixelValue
        image.GetPixelID(),
    )  # outputPixelType


def slice_image(image, patch_idx):
    """Slice an image."""

    start, stop = zip(*patch_idx)
    slice_filter = sitk.SliceImageFilter()
    slice_filter.SetStart(start)
    slice_filter.SetStop(stop)
    return slice_filter.Execute(image)


def is_overlapping(patch1, patch2):
    overlap_by_axis = [max(axis1[0], axis2[0]) < min(axis1[1], axis2[1]) for axis1, axis2 in zip(patch1, patch2)]

    return np.all(overlap_by_axis)
