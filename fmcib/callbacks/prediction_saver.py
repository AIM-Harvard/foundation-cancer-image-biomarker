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
    def __init__(self, path: str, save_preview_samples: bool = False, keys: List[str] = None):
        super().__init__("epoch")
        self.output_csv = Path(path)
        self.keys = keys
        self.save_preview_samples = save_preview_samples
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

    def save_preview_image(self, data, tag):
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
