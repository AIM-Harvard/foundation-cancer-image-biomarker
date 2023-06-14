import os
from pathlib import Path

import torch
import tqdm
import wget
from loguru import logger
from monai.networks.nets import resnet50 as resnet50_monai

from fmcib.utils.download_utils import bar_progress


def resnet50(pretrained=True, device="cuda"):
    logger.info(f"Loading pretrained foundation model (Resnet50) on {device}...")

    model = resnet50_monai(pretrained=False, n_input_channels=1, widen_factor=2, conv1_t_stride=2, feed_forward=False)
    model = model.to(device)
    if pretrained:
        current_path = Path(os.getcwd())
        if not (current_path / "fmcib.torch").exists():
            wget.download("https://www.dropbox.com/s/bd7azdsvx1jhalp/fmcib.torch?dl=1", bar=bar_progress)

        model.load_state_dict(torch.load(current_path / "fmcib.torch", map_location=device)["trunk_state_dict"])

    return model
