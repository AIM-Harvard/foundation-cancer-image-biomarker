import numpy as np
from monai.transforms import Transform


class IntensityNormalizeOneVolume(Transform):
    """
    A class representing an intensity normalized volume.

    Attributes:
        None

    Methods:
        __call__(self, volume): Normalize the intensity of an n-dimensional volume based on the mean and standard deviation of the non-zero region.

        Args:
            volume (numpy.ndarray): The input n-dimensional volume.

        Returns:
            out (numpy.ndarray): The normalized n-dimensional volume.
    """

    def __init__(self):
        """
        Initialize the object.

        Returns:
            None
        """
        super().__init__()

    def __call__(self, volume):
        """
        Normalize the intensity of an nd volume based on the mean and std of the non-zero region.

        Args:
            volume: The input nd volume.

        Returns:
            out: The normalized nd volume.
        """
        volume = volume.astype(np.float32)
        low, high = np.percentile(volume, [0.5, 99.5])
        if high > 0:
            volume = np.clip(volume, low, high)

        pixels = volume[volume > 0]
        mean = pixels.mean()
        std = pixels.std()
        out = (volume - mean) / std
        out_random = np.random.normal(0, 1, size=volume.shape)
        out[volume == 0] = out_random[volume == 0]
        return out
