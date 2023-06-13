from typing import Any, List

from pathlib import Path

import pandas as pd
import torchvision
from loguru import logger
from pytorch_lightning.callbacks import BasePredictionWriter

from .utils import decollate


def handle_image(image):
    image = image.squeeze()
    if image.dim() == 3:
        return image[image.shape[0] // 2]
    else:
        return image


class SavePredictions(BasePredictionWriter):
    def __init__(self, path: str, save_preview: bool = False):
        super().__init__("epoch")
        self.output_csv = Path(path)
        self.save_preview = save_preview
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.df = pd.DataFrame()

    def save_previews(self, dataset):
        logger.info("Saving image previews")
        self.output_dir = self.output_csv.parent / "previews"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for idx, data_item in enumerate(iter(dataset)):
            image, _ = data_item
            image = handle_image(image)
            fp = self.output_dir / f"{idx}.png"
            torchvision.utils.save_image(image, fp)

    def write_on_epoch_end(self, trainer, pl_module: "LightningModule", predictions: List[Any], batch_indices: List[Any]):
        assert getattr(pl_module, "predict_dataset"), "`predict_dataset` not defined"
        dataset = pl_module.predict_dataset

        if self.save_preview:
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
