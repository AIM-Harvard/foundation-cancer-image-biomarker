from copy import deepcopy
from typing import Any, Callable, Optional

import monai
from monai.transforms import MapTransform, Transform, apply_transform

class Replicate(Transform):
    """Replicate an input and apply two different transforms. Used for SimCLR primarily."""

    def __init__(
        self,
        transforms: Optional[Callable] = None,
        track_meta: bool = False,
    ):
        """Replicates an input and applies the given transformations to each copy separately.

        Args:
            transforms1 (Optional[Callable], optional): _description_. Defaults to None.
            transforms2 (Optional[Callable], optional): _description_. Defaults to None.
        """
        super().__init__()
        self.transforms = transforms
        self.track_meta = track_meta

    def __call__(self, input: Any, key: str = None):
        """
        Args:
            input (torch.Tensor or any other type supported by the given transforms): Input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: a tuple of two tensors.
        """
        transformed_outputs = []
        for transform in self.transforms:
            out = deepcopy(input)
            if transform is not None:
                out = apply_transform(transform, out)

            # Convert to tensor if necessary
            out = monai.utils.convert_to_tensor(out, track_meta=self.track_meta)
            transformed_outputs.append(out)

        if key is not None:
            transformed_outputs = [item[key] for item in transformed_outputs]
        return transformed_outputs
    

class Replicated(MapTransform):
    def __init__(
        self, keys, transforms, allow_missing_keys=False, track_meta=False
    ) -> None:
        super().__init__(keys=keys, allow_missing_keys=allow_missing_keys)
        self.Replicate = Replicate(transforms, track_meta=track_meta)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.Replicate(d, key)
        return d
