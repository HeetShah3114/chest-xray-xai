#!/usr/bin/env python
"""
resnet.py

Defines the ResNet architecture with support for both standard and attention-augmented variants.
This implementation uses helper_functions.py for convolution functions (conv1x1, conv3x3) and
aa_conv2d.py for attention-augmented convolutional layers.

Usage:
    To test the implementation, run the script directly:
      $ python resnet.py
    This will instantiate a ResNet (with Bottleneck blocks) and perform a quick forward pass
    on a dummy input, printing the total number of parameters and the output shape.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from helper_functions import conv1x1, conv3x3
from aa_conv2d import AAConv2d


class BasicBlock(nn.Module):
    """
    Implements the basic residual block used in ResNet.

    Attributes:
        expansion (int): Expansion factor (1 for BasicBlock).
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 input_dims=None, attn_params=None):
        """
        Initializes a BasicBlock.

        Parameters:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int): Stride for the convolution.
            downsample (nn.Module, optional): Downsampling layer to match dimensions.
            groups (int): Number of groups for convolutions (only 1 supported).
            base_width (int): Base width for convolutions (only 64 supported).
            dilation (int): Dilation rate (dilation > 1 not supported).
            norm_layer (callable, optional): Normalization layer (default is BatchNorm2d).
            input_dims (tuple, optional): Spatial dimensions for attention adjustments.
            attn_params (dict, optional): Parameters for attention augmentation. If provided,
                                          the block uses AAConv2d instead of standard conv.
        """
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if attn_params is not None:
            nh = attn_params['nh']
            dk = max(20 * nh, int((attn_params['k'] * planes // nh) * nh))
            dv = int((attn_params['v'] * planes // nh) * nh)
            relative = attn_params['relative']
            # Adjust input dimensions to the scale of this block.
            input_dims = (int(attn_params['input_dims'][0] * 16 / planes),
                          int(attn_params['input_dims'][1] * 16 / planes))
        # Use standard conv3x3 if no attention parameters provided; otherwise, use AAConv2d.
        self.conv1 = (conv3x3(inplanes, planes, stride) if attn_params is None else
                      AAConv2d(inplanes, planes, kernel_size=3, stride=stride,
                               dk=dk, dv=dv, nh=nh, relative=relative, input_dims=input_dims))
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass for the BasicBlock.

        Parameters:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after the block.
        """
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    """
    Implements the bottleneck residual block used in deeper ResNets.

    Attributes:
        expansion (int): Expansion factor (4 for Bottleneck).
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 input_dims=None, attn_params=None):
        """
        Initializes a Bottleneck block.

        Parameters:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels before expansion.
            stride (int): Stride for the block.
            downsample (nn.Module, optional): Downsampling layer.
            groups (int): Number of groups for convolutions.
            base_width (int): Base width for the convolutions.
            dilation (int): Dilation rate.
            norm_layer (callable, optional): Normalization layer (default is BatchNorm2d).
            input_dims (tuple, optional): Spatial dimensions for attention adjustments.
            attn_params (dict, optional): Parameters for attention augmentation. If provided,
                                          the block uses AAConv2d in its second convolution.
        """
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Calculate the width for the intermediate convolution.
        width = int(planes * (base_width / 64.)) * groups
        if attn_params is not None:
            nh = attn_params['nh']
            dk = max(20 * nh, int((attn_params['k'] * width // nh) * nh))
            dv = int((attn_params['v'] * width // nh) * nh)
            relative = attn_params['relative']
            input_dims = (int(attn_params['input_dims'][0] * 16 / planes),
                          int(attn_params['input_dims'][1] * 16 / planes))
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # Use standard conv3x3 if no attention augmentation, else AAConv2d.
        self.conv2 = (conv3x3(width, width, stride, groups, dilation)
                      if attn_params is None else
                      AAConv2d(width, width, kernel_size=3, stride=stride,
                               dk=dk, dv=dv, nh=nh, relative=relative,
                               input_dims=input_dims, groups=groups, dilation=dilation))
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass for the Bottleneck block.

        Parameters:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after the block.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    Implements the ResNet architecture with support for attention augmentation.

    Attributes:
        _norm_layer (callable): Normalization layer used throughout the network.
    """
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, attn_params=None):
        """
        Initializes the ResNet model.

        Parameters:
            block (class): Block type (BasicBlock or Bottleneck).
            layers (list): List of layer counts for each of the 4 layers.
            num_classes (int): Number of output classes.
            zero_init_residual (bool): Whether to zero-initialize the last BN in each residual branch.
            groups (int): Number of groups for convolutions.
            width_per_group (int): Base width for convolutions.
            replace_stride_with_dilation (list, optional): If provided, dilates layers instead of downsampling.
            norm_layer (callable, optional): Normalization layer (default is BatchNorm2d).
            attn_params (dict, optional): Attention augmentation parameters to use in selected layers.
        """
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
        self.groups = groups
        self.base_width = width_per_group

        # Initial convolution and pooling.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build residual layers.
        self.layer1 = self._make_layer(block, 64, layers[0], attn_params=None)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       attn_params=attn_params)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       attn_params=attn_params)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       attn_params=attn_params)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight initialization.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            # Zero-initialize the last BN in each residual branch.
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
                    attn_params=None):
        """
        Creates one layer of the ResNet consisting of several residual blocks.

        Parameters:
            block (class): Residual block type (BasicBlock or Bottleneck).
            planes (int): Number of output channels for the layer.
            blocks (int): Number of blocks to stack.
            stride (int): Stride for the first block.
            dilate (bool): Whether to apply dilation.
            attn_params (dict, optional): Attention augmentation parameters.

        Returns:
            nn.Sequential: A sequential container of the stacked blocks.
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # Create a downsampling layer if dimensions do not match.
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups, self.base_width, previous_dilation,
                            norm_layer, attn_params=attn_params))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, attn_params=attn_params))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Performs the forward pass of the ResNet model.

        Parameters:
            x (Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            Tensor: Output logits of shape (B, num_classes).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # Quick test of the ResNet implementation (e.g., ResNet-50)
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=10)
    print("ResNet parameters:", sum(p.numel() for p in model.parameters()))
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
