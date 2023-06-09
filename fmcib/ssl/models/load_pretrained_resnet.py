import monai
from torch import nn
from monai.networks.nets.resnet import ResNetBottleneck as Bottleneck
import torch
from loguru import logger

class LoadPretrainedResnet3D(nn.Module):
    def __init__(self, pretrained=None, vissl=False, heads=[]) -> None:
        super().__init__()
        self.trunk = monai.networks.nets.resnet.ResNet(
            block=Bottleneck,
            layers=(3, 4, 6, 3),
            block_inplanes=(64, 128, 256, 512),
            spatial_dims=3,
            n_input_channels=1,
            conv1_t_stride=2,
            conv1_t_size=7,
            widen_factor=2,
            num_classes=400
        )
        self.trunk.fc = nn.Identity()

        head_layers = []
        for idx in range(len(heads) - 1):
            head_layers.append(nn.Linear(heads[idx], heads[idx+1], bias=True))

            if idx != len(heads) - 1:
                head_layers.append(nn.ReLU(inplace=True))

        if len(head_layers):
            self.heads = nn.Sequential(*head_layers)
        else:
            self.heads = nn.Identity()

        if pretrained is not None:
            self.load(pretrained)

    def forward(self, x: torch.Tensor):
        out = self.trunk(x)
        out = self.heads(out)
        return out

    def load(self, pretrained):
        pretrained_model = torch.load(pretrained)

        # Load trained trunk
        trained_trunk = pretrained_model['trunk_state_dict']
        msg = self.trunk.load_state_dict(trained_trunk, strict=False)
        logger.warning(f'Missing keys: {msg[0]} and unexpected keys: {msg[1]}')

        # Load trained heads
        if "head_state_dict" in pretrained_model:
            trained_heads = pretrained_model['head_state_dict']
            try:
                msg = self.heads.load_state_dict(trained_heads, strict=False)
            except:
                logger.error("Failed to load trained heads. This is expected if the models do not match!")
            logger.warning(f'Missing keys: {msg[0]} and unexpected keys: {msg[1]}')

        logger.info(f'Loaded pretrained model weights \n')