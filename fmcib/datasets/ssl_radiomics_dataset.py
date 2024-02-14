from pathlib import Path

import monai
import numpy as np
import pandas as pd
import SimpleITK as sitk
from loguru import logger
from torch.utils.data import Dataset

from .utils import resample_image_to_spacing, slice_image


class SSLRadiomicsDataset(Dataset):
    """
    Dataset class for SSL Radiomics dataset.

    Args:
        path (str): The path to the dataset.
        label (str, optional): The label column name in the dataset annotations. Default is None.
        radius (int, optional): The radius around the centroid for positive patch extraction. Default is 25.
        orient (bool, optional): Whether to orient the images to LPI orientation. Default is False.
        resample_spacing (float or tuple, optional): The desired spacing for resampling the images. Default is None.
        enable_negatives (bool, optional): Whether to include negative samples. Default is True.
        transform (callable, optional): A function/transform to apply on the images. Default is None.
    """

    def __init__(
        self,
        path,
        label=None,
        radius=25,
        orient=False,
        resample_spacing=None,
        enable_negatives=True,
        transform=None,
        orient_patch=True,
        input_is_target=False,
    ):
        """
        Creates an instance of the SSLRadiomicsDataset class with the given parameters.

        Args:
            path (str): The path to the dataset.
            label (Optional[str]): The label to use for the dataset. Defaults to None.
            radius (int): The radius parameter. Defaults to 25.
            orient (bool): True if the dataset should be oriented, False otherwise. Defaults to False.
            resample_spacing (Optional[...]): The resample spacing parameter. Defaults to None.
            enable_negatives (bool): True if negatives are enabled, False otherwise. Defaults to True.
            transform: The transformation to apply to the dataset. Defaults to None.
            orient_patch (bool): True if the patch should be oriented, False otherwise. Defaults to True.
            input_is_target (bool): True if the input is the target, False otherwise. Defaults to False.

        Raises:
            None.

        Returns:
            None.
        """
        monai.data.set_track_meta(False)
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        super(SSLRadiomicsDataset, self).__init__()
        self._path = Path(path)

        self.radius = radius
        self.orient = orient
        self.resample_spacing = resample_spacing
        self.label = label
        self.enable_negatives = enable_negatives
        self.transform = transform
        self.orient_patch = orient_patch
        self.input_is_target = input_is_target
        self.annotations = pd.read_csv(self._path)
        self._num_samples = len(self.annotations)  # set the length of the dataset

    def get_rows(self):
        """
        Get the rows of the annotations as a list of dictionaries.

        Returns:
            list of dict: The rows of the annotations as dictionaries.
        """
        return self.annotations.to_dict(orient="records")

    def get_labels(self):
        """
        Function to get labels for when they are available in the dataset.

        Args:
            None

        Returns:
            None
        """

        labels = self.annotations[self.label].values
        assert not np.any(labels == -1), "All labels must be specified"
        return labels

    def __len__(self):
        """
        Size of the dataset.
        """
        return self._num_samples

    def get_negative_sample(self, image):
        """
        Extract a negative sample from the image background with no overlap to the positive sample.

        Parameters:
            image: Image to extract sample
            positive_patch_idx: Index of the positive patch in [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        """
        positive_patch_size = [self.radius * 2] * 3
        valid_patch_size = monai.data.utils.get_valid_patch_size(image.GetSize(), positive_patch_size)

        def get_random_patch():
            """
            Get a random patch from an image.

            Returns:
                list: A list containing the start and end indices of the random patch.
            """
            random_patch_idx = [
                [x.start, x.stop] for x in monai.data.utils.get_random_patch(image.GetSize(), valid_patch_size)
            ]
            return random_patch_idx

        random_patch_idx = get_random_patch()

        # escape_count = 0
        # while is_overlapping(positive_patch_idx, random_patch_idx):
        #     if escape_count >= 3:
        #         logger.warning("Random patch has overlap with positive patch")
        #         return None

        #     random_patch_idx = get_random_patch()
        #     escape_count += 1

        random_patch = slice_image(image, random_patch_idx)
        random_patch = sitk.DICOMOrient(random_patch, "LPS") if self.orient_patch else random_patch
        negative_array = sitk.GetArrayFromImage(random_patch)

        negative_tensor = negative_array if self.transform is None else self.transform(negative_array)
        return negative_tensor

    def __getitem__(self, idx: int):
        """
        Implement how to load the data corresponding to the idx element in the dataset from your data source.
        """

        # Get a row from the CSV file
        row = self.annotations.iloc[idx]
        image_path = row["image_path"]
        image = sitk.ReadImage(str(image_path))
        image = resample_image_to_spacing(image, self.resample_spacing, -1024) if self.resample_spacing is not None else image

        centroid = (row["coordX"], row["coordY"], row["coordZ"])
        centroid = image.TransformPhysicalPointToContinuousIndex(centroid)
        centroid = [int(d) for d in centroid]

        # Orient all images to LPI orientation
        image = sitk.DICOMOrient(image, "LPI") if self.orient else image

        # Extract positive with a specified radius around centroid
        patch_idx = [(c - self.radius, c + self.radius) for c in centroid]
        patch_image = slice_image(image, patch_idx)

        patch_image = sitk.DICOMOrient(patch_image, "LPS") if self.orient_patch else patch_image

        array = sitk.GetArrayFromImage(patch_image)
        tensor = array if self.transform is None else self.transform(array)

        if self.label is not None:
            target = int(row[self.label])
        elif self.input_is_target:
            target = tensor.clone()
        else:
            target = None

        if self.enable_negatives:
            return {"positive": tensor, "negative": self.get_negative_sample(image)}, target

        return tensor, target


if __name__ == "__main__":
    from pathlib import Path

    # Test pytorch dataset
    print("Test pytorch dataset")
    dataset = SSLRadiomicsDataset(
        "/home/suraj/Repositories/cancer-imaging-ssl/src/pretraining/data_csv/deeplesion/train.csv",
        orient=True,
        resample_spacing=[1, 1, 1],
    )

    # Visualize item from dataset
    item = dataset[0]

    positive = sitk.GetImageFromArray(item[0][0])
    negative = sitk.GetImageFromArray(item[0][1])
    current_dir = Path(__file__).parent.resolve()

    sitk.WriteImage(positive, f"{str(current_dir)}/tests/positive.nrrd")
    sitk.WriteImage(negative, f"{str(current_dir)}/tests/negative.nrrd")
