from typing import Any, List

from pathlib import Path

import pandas as pd
import torchvision
from loguru import logger
from pytorch_lightning.callbacks import BasePredictionWriter

from .utils import decollate


def handle_image(image):
    image = image.squeeze()

    if image.max() > 1 or image.min() < 0:
        logger.warning(f"Normalizing image between 0 and 1, current max value is {image.max()}, min value is {image.min()}")

    # Normalize image between 0 and 1
    image = image - image.min()
    image = image / (image.max() + 1e-8)

    if image.dim() == 3:
        axial_view = image[image.shape[0] // 2].unsqueeze(0)
        sagittal_view = image[:, image.shape[1] // 2, :]. unsqueeze(0)
        coronal_view = image[:, :, image.shape[2] // 2].unsqueeze(0)
        grid_image = torchvision.utils.make_grid([axial_view, sagittal_view, coronal_view])
        return grid_image
    else:
        return image


class SavePredictions(BasePredictionWriter):
    def __init__(self, path: str, save_preview_samples: int = 0):
        super().__init__("epoch")
        self.output_csv = Path(path)
        self.save_preview_samples = save_preview_samples
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.df = pd.DataFrame()

    def save_previews(self, dataset):
        logger.info("Saving image previews")
        self.output_dir = self.output_csv.parent / "previews"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for idx, data_item in enumerate(iter(dataset)):
            if idx >= self.save_preview_samples:
                break
            image, _ = data_item
            image = handle_image(image)
            fp = self.output_dir / f"{idx}.png"
            torchvision.utils.save_image(image, fp)

    def write_on_epoch_end(self, trainer, pl_module: "LightningModule", predictions: List[Any], batch_indices: List[Any]):
        assert getattr(pl_module, "predict_dataset"), "`predict_dataset` not defined"
        dataset = pl_module.predict_dataset

        if self.save_preview_samples > 0:
            self.save_previews(dataset)

        assert getattr(dataset, "get_rows"), "The dataset must have `get_image_paths` defined for predict functionality"
        rows = dataset.get_rows()

        out = decollate(predictions)
        assert len(out) == len(rows), "Length of image_paths and predictions do not match"

        for pred, row in zip(out, rows):
            # Handle multiple output cases (features and multi-class)
            if isinstance(pred, list):
                for idx, v in enumerate(pred):
                    row[f"pred_{idx}"] = v
            # Single class case
            else:
                row["pred"] = pred

            self.df = self.df.append(row, ignore_index=True)

        self.df.to_csv(self.output_csv)