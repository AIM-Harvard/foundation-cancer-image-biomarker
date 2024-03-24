import matplotlib.pyplot as plt
import monai.transforms as monai_transforms
import numpy as np
import torch
from monai.visualize import blend_images


def visualize_seed_point(row):
    """
    This function visualizes a seed point on an image.

    Args:
        row (pandas.Series): A row containing the information of the seed point, including the image path and the coordinates.
            The following columns are expected: "image_path", "coordX", "coordY", "coordZ".

    Returns:
        None
    """
    # Define the transformation pipeline
    is_label_provided = "label_path" in row
    keys = ["image_path", "label_path"] if is_label_provided else ["image_path"]
    all_keys = keys if is_label_provided else ["image_path", "coordX", "coordY", "coordZ"]

    T = monai_transforms.Compose(
        [
            monai_transforms.LoadImaged(keys=keys, image_only=True, reader="ITKReader"),
            monai_transforms.EnsureChannelFirstd(keys=keys),
            monai_transforms.Spacingd(keys=keys, pixdim=1, mode="bilinear", align_corners=True, diagonal=True),
            monai_transforms.ScaleIntensityRanged(keys=["image_path"], a_min=-1024, a_max=3072, b_min=0, b_max=1, clip=True),
            monai_transforms.Orientationd(keys=keys, axcodes="LPS"),
            monai_transforms.SelectItemsd(keys=all_keys),
        ]
    )

    # Apply the transformation pipeline
    out = T(row)

    # Calculate the center of the image
    image = out["image_path"]
    if not is_label_provided:
        center = (-out["coordX"], -out["coordY"], out["coordZ"])
        center = np.linalg.inv(np.array(out["image_path"].affine)) @ np.array(center + (1,))
        center = [int(x) for x in center[:3]]

        # Define the image and label
        label = torch.zeros_like(image)

        # Define the dimensions of the image and the patch
        C, H, W, D = image.shape
        Ph, Pw, Pd = 50, 50, 50

        # Calculate and clamp the ranges for cropping
        min_h, max_h = max(center[0] - Ph // 2, 0), min(center[0] + Ph // 2, H)
        min_w, max_w = max(center[1] - Pw // 2, 0), min(center[1] + Pw // 2, W)
        min_d, max_d = max(center[2] - Pd // 2, 0), min(center[2] + Pd // 2, D)

        # Check if coordinates are valid
        assert min_h < max_h, "Invalid coordinates: min_h >= max_h"
        assert min_w < max_w, "Invalid coordinates: min_w >= max_w"
        assert min_d < max_d, "Invalid coordinates: min_d >= max_d"

        # Define the label for the cropped region
        label[:, min_h:max_h, min_w:max_w, min_d:max_d] = 1
    else:
        label = out["label_path"]
        center = torch.nonzero(label).float().mean(dim=0)
        center = [int(x) for x in center][1:]

    # Blend the image and the label
    ret = blend_images(image=image, label=label, alpha=0.3, cmap="hsv", rescale_arrays=False)
    ret = ret.permute(3, 2, 1, 0)

    # Plot axial slice
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(ret[center[2], :, :])
    plt.title("Axial")
    plt.axis("off")

    # Plot sagittal slice
    plt.subplot(1, 3, 2)
    plt.imshow(np.flipud(ret[:, center[1], :]))
    plt.title("Coronal")
    plt.axis("off")

    # Plot coronal slice
    plt.subplot(1, 3, 3)
    plt.imshow(np.flipud(ret[:, :, center[0]]))
    plt.title("Sagittal")

    plt.axis("off")
    plt.show()
