import torch
import torch.nn as nn
import torch.nn.functional as F


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    """
    A class representing a 3D contextual batch normalization layer.

    Attributes:
        running_mean (torch.Tensor): The running mean of the batch normalization.
        running_var (torch.Tensor): The running variance of the batch normalization.
        weight (torch.Tensor): The learnable weights of the batch normalization.
        bias (torch.Tensor): The learnable bias of the batch normalization.
        momentum (float): The momentum for updating the running statistics.
        eps (float): Small value added to the denominator for numerical stability.
    """

    def _check_input_dim(self, input):
        """
        Check if the input tensor is 5-dimensional.

        Args:
            input (torch.Tensor): Input tensor to check the dimensionality.

        Raises:
            ValueError: If the input tensor is not 5-dimensional.
        """
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))
        # super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        """
        Apply forward pass for the input through batch normalization layer.

        Args:
            input (Tensor): Input tensor to be normalized.

        Returns:
            Tensor: Normalized output tensor.

        Raises:
            ValueError: If the dimensions of the input tensor do not match the expected input dimensions.
        """
        self._check_input_dim(input)
        return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias, True, self.momentum, self.eps)


class LUConv(nn.Module):
    """
    A class representing a LUConv module.

    This module performs a convolution operation on the input data with a specified number of input channels and output channels.
    The convolution is followed by batch normalization and an activation function.

    Attributes:
        in_chan (int): The number of input channels.
        out_chan (int): The number of output channels.
        act (str): The activation function to be applied. Can be one of 'relu', 'prelu', or 'elu'.
    """

    def __init__(self, in_chan, out_chan, act):
        """
        Initialize a LUConv layer.

        Args:
            in_chan (int): Number of input channels.
            out_chan (int): Number of output channels.
            act (str): Activation function. Options: 'relu', 'prelu', 'elu'.

        Returns:
            None

        Raises:
            TypeError: If the activation function is not one of the specified options.
        """
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == "relu":
            self.activation = nn.ReLU(out_chan)
        elif act == "prelu":
            self.activation = nn.PReLU(out_chan)
        elif act == "elu":
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        """
        Apply forward pass through the neural network.

        Args:
            x (Tensor): Input tensor to the network.

        Returns:
            Tensor: Output tensor after passing through the network.
        """
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    """
    Make a two-layer convolutional neural network module.

    Args:
        in_channel (int): The number of input channels.
        depth (int): The depth of the network.
        act: Activation function to be used in the network.
        double_channel (bool, optional): If True, double the number of channels in the network. Defaults to False.

    Returns:
        nn.Sequential: A sequential module representing the two-layer convolutional network.

    Note:
        - If double_channel is True, the first layer will have 32 * 2 ** (depth + 1) channels and the second layer will have the same number of channels.
        - If double_channel is False, the first layer will have 32 * 2 ** depth channels and the second layer will have 32 * 2 ** depth * 2 channels.
    """
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth + 1)), act)
        layer2 = LUConv(32 * (2 ** (depth + 1)), 32 * (2 ** (depth + 1)), act)
    else:
        layer1 = LUConv(in_channel, 32 * (2**depth), act)
        layer2 = LUConv(32 * (2**depth), 32 * (2**depth) * 2, act)

    return nn.Sequential(layer1, layer2)


# class InputTransition(nn.Module):
#     def __init__(self, outChans, elu):
#         super(InputTransition, self).__init__()
#         self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
#         self.bn1 = ContBatchNorm3d(16)
#         self.relu1 = ELUCons(elu, 16)
#
#     def forward(self, x):
#         # do we want a PRELU here as well?
#         out = self.bn1(self.conv1(x))
#         # split input in to 16 channels
#         x16 = torch.cat((x, x, x, x, x, x, x, x,
#                          x, x, x, x, x, x, x, x), 1)
#         out = self.relu1(torch.add(out, x16))
#         return out


class DownTransition(nn.Module):
    """
    A class representing a down transition module in a neural network.

    Attributes:
        in_channel (int): The number of input channels.
        depth (int): The depth of the down transition module.
        act (nn.Module): The activation function used in the module.
    """

    def __init__(self, in_channel, depth, act):
        """
        Initialize a DownTransition object.

        Args:
            in_channel (int): The number of channels in the input.
            depth (int): The depth of the DownTransition.
            act (function): The activation function.

        Returns:
            None

        Raises:
            None
        """
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth, act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        """
        Perform a forward pass through the neural network.

        Args:
            x (Tensor): The input tensor.

        Returns:
            tuple: A tuple containing two tensors. The first tensor is the output of the forward pass. The second tensor is the output before applying the max pooling operation.

        Raises:
            None
        """
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool


class UpTransition(nn.Module):
    """
    A class representing an up transition layer in a neural network.

    Attributes:
        inChans (int): The number of input channels.
        outChans (int): The number of output channels.
        depth (int): The depth of the layer.
        act (str): The activation function to be applied.
    """

    def __init__(self, inChans, outChans, depth, act):
        """
        Initialize the UpTransition module.

        Args:
            inChans (int): The number of input channels.
            outChans (int): The number of output channels.
            depth (int): The depth of the module.
            act (nn.Module): The activation function to be used.

        Returns:
            None.

        Raises:
            None.
        """
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans + outChans // 2, depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.
            skip_x (torch.Tensor): Tensor to be concatenated with the upsampled convolution output.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
        """
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv, skip_x), 1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    """
    A class representing the output transition in a neural network.

    Attributes:
        inChans (int): The number of input channels.
        n_labels (int): The number of output labels.
    """

    def __init__(self, inChans, n_labels):
        """
        Initialize the OutputTransition class.

        Args:
            inChans (int): Number of input channels.
            n_labels (int): Number of output labels.

        Returns:
            None

        Raises:
            None
        """
        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through a neural network model.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through the model.
        """
        out = self.sigmoid(self.final_conv(x))
        return out


class UNet3D(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    """
    A class representing a 3D UNet model for segmentation.

    Attributes:
        n_class (int): The number of classes for segmentation.
        act (str): The activation function type used in the model.
        decoder (bool): Whether to include the decoder part in the model.

    Methods:
        forward(x): Forward pass of the model.
    """

    def __init__(self, n_class=1, act="relu", decoder=True):
        """
        Initialize a 3D UNet neural network model.

        Args:
            n_class (int): The number of output classes. Defaults to 1.
            act (str): The activation function to use. Defaults to 'relu'.
            decoder (bool): Whether to include the decoder layers. Defaults to True.

        Attributes:
            decoder (bool): Whether the model includes decoder layers.
            down_tr64 (DownTransition): The first down transition layer.
            down_tr128 (DownTransition): The second down transition layer.
            down_tr256 (DownTransition): The third down transition layer.
            down_tr512 (DownTransition): The fourth down transition layer.
            up_tr256 (UpTransition): The first up transition layer. (Only exists if `decoder` is True)
            up_tr128 (UpTransition): The second up transition layer. (Only exists if `decoder` is True)
            up_tr64 (UpTransition): The third up transition layer. (Only exists if `decoder` is True)
            out_tr (OutputTransition): The output transition layer. (Only exists if `decoder` is True)
            avg_pool (nn.AvgPool3d): The average pooling layer. (Only exists if `decoder` is False)
            flatten (nn.Flatten): The flattening layer. (Only exists if `decoder` is False)
        """
        super(UNet3D, self).__init__()

        self.decoder = decoder

        self.down_tr64 = DownTransition(1, 0, act)
        self.down_tr128 = DownTransition(64, 1, act)
        self.down_tr256 = DownTransition(128, 2, act)
        self.down_tr512 = DownTransition(256, 3, act)

        if self.decoder:
            self.up_tr256 = UpTransition(512, 512, 2, act)
            self.up_tr128 = UpTransition(256, 256, 1, act)
            self.up_tr64 = UpTransition(128, 128, 0, act)
            self.out_tr = OutputTransition(64, n_class)
        else:
            self.avg_pool = nn.AvgPool3d(3, stride=2)
            self.flatten = nn.Flatten()

    def forward(self, x):
        """
        Perform forward pass through the neural network.

        Args:
            x (Tensor): Input tensor to the network.

        Returns:
            Tensor: Output tensor from the network.

        Note: This function performs a series of operations to downsample the input tensor, followed by upsampling if the 'decoder' flag is set. If the 'decoder' flag is not set, the output tensor goes through average pooling and flattening.

        Raises:
            None.
        """
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128, self.skip_out128 = self.down_tr128(self.out64)
        self.out256, self.skip_out256 = self.down_tr256(self.out128)
        self.out512, self.skip_out512 = self.down_tr512(self.out256)

        if self.decoder:
            self.out_up_256 = self.up_tr256(self.out512, self.skip_out256)
            self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
            self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64)
            self.out = self.out_tr(self.out_up_64)
        else:
            self.out = self.avg_pool(self.out512)
            self.out = self.flatten(self.out)

        return self.out
