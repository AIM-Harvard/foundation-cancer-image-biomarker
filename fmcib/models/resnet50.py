import os
from pathlib import Path

import torch
import tqdm
import wget
from loguru import logger
from monai.networks.nets import resnet50 as resnet50_monai

from fmcib.utils.download_utils import bar_progress


def resnet50(
    pretrained=True,
    device="cuda",
    weights_path=None,
    download_url="https://www.dropbox.com/s/bd7azdsvx1jhalp/fmcib.torch?dl=1",
    n_input_channels=1,
    widen_factor=2,
    conv1_t_stride=2,
    bias_downsample=True,
    feed_forward=False,
):
    logger.info(f"Loading pretrained foundation model (Resnet50) on {device}...")

    model = resnet50_monai(
        pretrained=False,
        n_input_channels=n_input_channels,
        widen_factor=widen_factor,
        conv1_t_stride=conv1_t_stride,
        feed_forward=feed_forward,
        bias_downsample=bias_downsample,
    )
    model = model.to(device)
    if pretrained:
        if weights_path is None:
            current_path = Path(os.getcwd())
            if not (current_path / "fmcib.torch").exists():
                wget.download(download_url, bar=bar_progress)
            weights_path = current_path / "fmcib.torch"

        logger.info(f"Loading weights from {weights_path}...")
        checkpoint = torch.load(weights_path, map_location=device)

        if "trunk_state_dict" in checkpoint:
            model_state_dict = checkpoint["trunk_state_dict"]
        elif "state_dict" in checkpoint:
            model_state_dict = checkpoint["state_dict"]
            model_state_dict = {key.replace("model.backbone.", ""): value for key, value in model_state_dict.items()}
            model_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}

        msg = model.load_state_dict(model_state_dict, strict=False)
        logger.warning(f"Missing keys: {msg[0]} and unexpected keys: {msg[1]}")

    return model
