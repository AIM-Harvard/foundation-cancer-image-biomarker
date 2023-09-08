import os
from pathlib import Path

import torch
import tqdm
import wget
from loguru import logger
from monai.networks.nets import resnet50 as resnet50_monai

from fmcib.utils.download_utils import bar_progress

def resnet50(pretrained=True, device="cuda", weights_path=None, download_url="https://www.dropbox.com/s/bd7azdsvx1jhalp/fmcib.torch?dl=1"):
    logger.info(f"Loading pretrained foundation model (Resnet50) on {device}...")

    model = resnet50_monai(pretrained=False, n_input_channels=1, widen_factor=2, conv1_t_stride=2, feed_forward=False)
    model = model.to(device)
    if pretrained:
        if weights_path is None:
            current_path = Path(os.getcwd())
            if not (current_path / "fmcib.torch").exists():
                wget.download(download_url, bar=bar_progress)
            weights_path = current_path / "fmcib.torch"

        checkpoint = torch.load(weights_path, map_location=device)

        if "trunk_state_dict" in checkpoint:
            model_state_dict = checkpoint["trunk_state_dict"]
        elif "state_dict" in checkpoint:
            model_state_dict = checkpoint["state_dict"]
            model_state_dict = {key.replace("model.backbone.", ""): value for key, value in model_state_dict.items()}
            
        model.load_state_dict(model_state_dict, strict=False)

    return model


