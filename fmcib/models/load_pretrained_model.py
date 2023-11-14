from collections import OrderedDict

import torch
from loguru import logger
from torch import nn


class LoadPretrainedModel(nn.Module):
    def __init__(self, trunk=None, weights_path=None, heads=[]) -> None:
        super().__init__()
        self.trunk = trunk
        head_layers = []
        for idx in range(len(heads) - 1):
            current_layers = []
            current_layers.append(nn.Linear(heads[idx], heads[idx + 1], bias=True))

            if idx != (len(heads) - 2):
                current_layers.append(nn.ReLU(inplace=True))

            head_layers.append(nn.Sequential(*current_layers))

        if len(head_layers):
            self.heads = nn.Sequential(*head_layers)
        else:
            self.heads = nn.Identity()

        if weights_path is not None:
            self.load(weights_path)

    def forward(self, x: torch.Tensor):
        out = self.trunk(x)
        out = self.heads(out)
        return out

    def load(self, weights):
        pretrained_model = torch.load(weights)

        if "trunk_state_dict" in pretrained_model:  # Loading ViSSL pretrained model
            trained_trunk = pretrained_model["trunk_state_dict"]
            msg = self.trunk.load_state_dict(trained_trunk, strict=False)
            logger.warning(f"Missing keys: {msg[0]} and unexpected keys: {msg[1]}")

        if "state_dict" in pretrained_model:  # Loading Med3D pretrained model
            trained_model = pretrained_model["state_dict"]

            # match the keys (https://github.com/Project-MONAI/MONAI/issues/6811)
            weights = {key.replace("module.", ""): value for key, value in trained_model.items()}
            weights = {key.replace("model.trunk.", ""): value for key, value in trained_model.items()}
            msg = self.trunk.load_state_dict(weights, strict=False)
            logger.warning(f"Missing keys: {msg[0]} and unexpected keys: {msg[1]}")

            weights = {key.replace("model.heads.", ""): value for key, value in trained_model.items()}
            msg = self.heads.load_state_dict(weights, strict=False)
            logger.warning(f"Missing keys: {msg[0]} and unexpected keys: {msg[1]}")

        # Load trained heads
        if "head_state_dict" in pretrained_model:
            trained_heads = pretrained_model["head_state_dict"]

            try:
                msg = self.heads.load_state_dict(trained_heads, strict=False)
            except Exception as e:
                logger.error(f"Failed to load trained heads with error {e}. This is expected if the models do not match!")
            logger.warning(f"Missing keys: {msg[0]} and unexpected keys: {msg[1]}")

        logger.info(f"Loaded pretrained model weights \n")
