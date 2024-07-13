from typing import Union

import torch
import numpy as np
from torch import nn


class ConvBlock(nn.Module):
    """
    A Convolutional Block for a Convolutional Neural Network (CNN).

    This block consists of a 2D convolutional layer, a batch normalization layer, and a LeakyReLU activation function.

    Parameters:
    - in_channels (int): The number of input channels for the convolutional layer. Default is 64.
    - out_channels (int): The number of output channels for the convolutional layer. Default is 64.
    - kernel_size (int): The size of the convolutional kernel. Default is 3.
    - stride (int): The stride of the convolutional layer. Default is 1.
    - padding (int): The padding of the convolutional layer. Default is 1.

    Returns:
    - torch.Tensor: The output tensor after applying the convolutional, batch normalization, and activation functions.
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 64, kernel_size: int = 3, stride: int = 1,
                 padding: Union[int, str] = 1):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size,
                              stride=self.stride, padding=self.padding)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.act = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class ResidualBlock(nn.Module):
    """
    A Residual Block for a Convolutional Neural Network (CNN).

    This block consists of two convolutional layers, each followed by a batch normalization and a PReLU activation function.
    The output of the second convolutional layer is added to the input of the block, creating a skip connection.

    Parameters:
    - in_channels (int): The number of input channels for the convolutional layers. Default is 64.
    - out_channels (int): The number of output channels for the convolutional layers. Default is 64.
    - kernel_size (int): The size of the convolutional kernel. Default is 3.
    - stride (int): The stride of the convolutional layers. Default is 1.
    - padding (int): The padding of the convolutional layers. Default is 1.

    Returns:
    - torch.Tensor: The output of the Residual Block after applying the convolutional, batch normalization, and activation functions.
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 64, kernel_size: int = 3, stride: int = 1,
                 padding: Union[int, str] = 1):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv_1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(self.kernel_size, self.kernel_size),
                                stride=(self.stride, self.stride), padding=(self.padding, self.padding))
        self.conv_2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=(self.kernel_size, self.kernel_size),
                                stride=(self.stride, self.stride), padding=(self.padding, self.padding))
        self.bn_1 = nn.BatchNorm2d(self.out_channels)
        self.bn_2 = nn.BatchNorm2d(self.out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        # first
        t = self.conv_1(x)
        t = self.bn_1(t)
        t = self.act(t)

        # second
        t = self.conv_2(t)
        t = self.bn_2(t)

        return torch.add(x, t)


class UpscaleBlock(nn.Module):
    """
    A block for upscaling the input tensor using convolutional and pixel shuffle layers.

    This block consists of a convolutional layer, a PReLU activation function, and a pixel shuffle layer.
    The convolutional layer reduces the number of channels, followed by the pixel shuffle layer to increase the spatial dimensions.

    Parameters:
    - in_channels (int): The number of input channels. Default is 64.
    - out_channels (int): The number of output channels. Default is 256.
    - scale_factor (int): The factor by which to upscale the spatial dimensions. Default is 2.
    - kernel_size (int): The size of the convolutional kernel. Default is 3.
    - stride (int): The stride of the convolutional layer. Default is 1.
    - padding (Union[int, str]): The padding of the convolutional layer. Default is 1.

    Returns:
    - torch.Tensor: The output tensor after upscaling and applying the convolutional, activation, and pixel shuffle layers.
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 256, scale_factor: int = 2, kernel_size: int = 3,
                 stride: int = 1, padding: Union[int, str] = 1):
        super(UpscaleBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale_factor
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv_1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(self.kernel_size, self.kernel_size),
                                stride=(self.stride, self.stride), padding=(self.padding, self.padding))
        self.act = nn.PReLU()
        self.shuffler = nn.PixelShuffle(scale_factor)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.act(x)
        x = self.shuffler(x)

        return x


class SRResNet(nn.Module):
    """
    A PyTorch implementation of the Super Resolution Residual Network (SRResNet).

    SRResNet is a deep convolutional neural network designed for single image super-resolution.
    It consists of several convolutional layers, residual blocks, and upscaling layers to enhance the resolution of input images.

    Parameters:
    - in_channels (int): The number of input channels. Default is 3 (RGB).
    - out_channels (int): The number of output channels. Default is 3 (RGB).
    - num_of_res_blocks (int): The number of residual blocks. Default is 5.
    - scale_factor (int): The scale factor for upscaling the input image. Default is 4.
    - step_scale_factor (int): The step scale factor for upscaling in each upscale block. Default is 2.
    - stride (int): The stride for convolutional layers. Default is 1.
    - padding (Union[int, str]): The padding for convolutional layers. Default is 1.

    Returns:
    - torch.Tensor: The super-resolved output image.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_of_res_blocks: int = 5, scale_factor: int = 4,
                 step_scale_factor: int = 2,
                 stride: int = 1, padding: Union[int, str] = 1):
        super(SRResNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_of_res_blocks = num_of_res_blocks
        self.scale_factor = scale_factor
        self.step_scale_factor = step_scale_factor
        if not np.emath.logn(step_scale_factor, scale_factor).is_integer():
            raise ValueError(
                f"scale_factor {scale_factor} cannot be reached with stacked layers, each step-scale by {step_scale_factor}")

        # we need to translate the scale factor into the number of scaling layers
        self.num_of_scaling_layers = (scale_factor // step_scale_factor) - 1
        self.stride = stride
        self.padding = padding

        # low frequency information extraction
        self.in_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=stride, padding=4),
            nn.PReLU()
        )

        # high frequency information extraction
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(in_channels=64, out_channels=64, padding=padding) for _ in range(num_of_res_blocks)]
        )

        # high frequency information fusion
        self.after_res_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(64)
        )

        # zooming
        self.upscale_layers = nn.Sequential(
            UpscaleBlock(in_channels=64, out_channels=64 * step_scale_factor * step_scale_factor,
                         scale_factor=step_scale_factor, kernel_size=3, stride=stride,
                         padding=padding),
            *[UpscaleBlock(in_channels=64, out_channels=64 * step_scale_factor * step_scale_factor,
                           scale_factor=step_scale_factor, kernel_size=3, stride=stride,
                           padding=padding) for _ in range(self.num_of_scaling_layers)]
        )

        # reconstruction
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=9, stride=stride, padding=4)

    def forward(self, x):
        x = self.in_block(x)
        z = self.residual_blocks(x)
        z = self.after_res_block(z)
        z = torch.add(z, x)
        z = self.upscale_layers(z)
        return self.final_conv(z)


class Discriminator(nn.Module):
    """
    A PyTorch implementation of a Discriminator for Generative Adversarial Networks (GANs).
    The Discriminator is used to distinguish real images from generated ones.

    Parameters:
    - in_channels (int): The number of input channels. Default is 3 (RGB).
    - out_channels (int): The number of output channels. Default is 1 (scalar output).

    Returns:
    - torch.Tensor: A scalar value representing the probability of the input image being real.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # high frequency features
        self.in_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

        # conv blocks
        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.in_block(x)
        x = self.conv_blocks(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)

