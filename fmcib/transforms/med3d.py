import numpy as np
from monai.transforms import Transform


class IntensityNormalizeOneVolume(Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
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
