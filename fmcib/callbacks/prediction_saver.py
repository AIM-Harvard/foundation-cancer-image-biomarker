from typing import Any, List

from pathlib import Path

import pandas as pd
import torchvision
from loguru import logger
from pytorch_lightning.callbacks import BasePredictionWriter

from .utils import decollate, handle_image


class SavePredictions(BasePredictionWriter):
    """
    A class that saves model predictions.

    Attributes:
        path (str): The path to save the output CSV file.
        save_preview_samples (bool): If True, save preview images.
        keys (List[str]): A list of keys.
    """

    def __init__(self, path: str, save_preview_samples: bool = False, keys: List[str] = None):
        """
        Initialize an instance of the class.

        Args:
            path (str): The path to save the output CSV file.
            save_preview_samples (bool, optional): A flag indicating whether to save preview samples. Defaults to False.
            keys (List[str], optional): A list of keys. Defaults to None.

        Raises:
            None

        Returns:
            None
        """
        super().__init__("epoch")
        self.output_csv = Path(path)
        self.keys = keys
        self.save_preview_samples = save_preview_samples
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

    def save_preview_image(self, data, tag):
        """
        Save a preview image to a specified directory.

        Args:
            self (object): The object calling the function. (self in Python)
            data (tuple): A tuple containing the image data and its corresponding tag.
            tag (str): The tag for the image.

        Returns:
            None

        Raises:
            None
        """
        self.output_dir = self.output_csv.parent / f"previews_{self.output_csv.stem}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image, _ = data
        image = handle_image(image)
        fp = self.output_dir / f"{tag}.png"
        torchvision.utils.save_image(image, fp)

    def write_on_epoch_end(
        self,
        trainer,
        pl_module: "LightningModule",
        predictions: List[Any],
        batch_indices: List[Any],
    ):
        """
        Write predictions on epoch end.

        Args:
            self: The instance of the class.
            trainer: The trainer object.
            pl_module (LightningModule): The Lightning module.
            predictions (List[Any]): A list of prediction values.
            batch_indices (List[Any]): A list of batch indices.

        Raises:
            AssertionError: If 'predict' is not present in pl_module.datasets.
            AssertionError: If 'data' is not defined in pl_module.datasets.

        Returns:
            None
        """
        rows = []
        assert "predict" in pl_module.datasets, "`data` not defined"
        dataset = pl_module.datasets["predict"]
        predictions = [pred for batch_pred in predictions for pred in batch_pred["pred"]]

        for idx, (row, pred) in enumerate(zip(dataset.get_rows(), predictions)):
            for i, v in enumerate(pred):
                row[f"pred_{i}"] = v.item()

            rows.append(row)

            # Save image previews
            if idx <= self.save_preview_samples:
                input = dataset[idx]
                self.save_preview_image(input, idx)

        df = pd.DataFrame(rows)
        df.to_csv(self.output_csv)
