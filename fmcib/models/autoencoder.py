import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.nets import AutoEncoder


class CustomAE(AutoEncoder):
    """
    A custom AutoEncoder class.

    Inherits from AutoEncoder.

    Attributes:
        padding (int): The padding size for the convolutional layers.
        decoder (bool, optional): Determines if the decoder part of the network is included.
        kwargs: Additional keyword arguments passed to the parent class.

    Methods:
        _get_encode_layer(in_channels, out_channels, strides, is_last): Returns a single layer of the encoder part of the network.
        _get_decode_layer(in_channels, out_channels, strides, is_last): Returns a single layer of the decoder part of the network.
    """

    def __init__(self, padding, decoder=True, **kwargs):
        """
        Initialize the object.

        Args:
            padding (int): Padding value.
            decoder (bool, optional): If True, use a decoder. Defaults to True.
            **kwargs: Additional keyword arguments.

        Attributes:
            padding (int): Padding value.

        Raises:
            None
        """
        self.padding = padding
        super().__init__(**kwargs)
        if not decoder:
            self.decode = nn.Sequential(nn.AvgPool3d(3), nn.Flatten())

    def _get_encode_layer(self, in_channels: int, out_channels: int, strides: int, is_last: bool) -> nn.Module:
        """
        Returns a single layer of the encoder part of the network.
        """
        mod: nn.Module
        if self.num_res_units > 0:
            mod = ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                padding=self.padding,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_last,
            )
            return mod
        mod = Convolution(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            padding=self.padding,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_last,
        )
        return mod

    def _get_decode_layer(self, in_channels: int, out_channels: int, strides: int, is_last: bool) -> nn.Sequential:
        """
        Returns a single layer of the decoder part of the network.
        """
        decode = nn.Sequential()

        conv = Convolution(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            padding=self.padding,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_last and self.num_res_units == 0,
            is_transposed=True,
        )

        decode.add_module("conv", conv)

        if self.num_res_units > 0:
            ru = ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=out_channels,
                out_channels=out_channels,
                padding=self.padding,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_last,
            )

            decode.add_module("resunit", ru)

        return decode
