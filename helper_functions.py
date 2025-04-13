#!/usr/bin/env python
"""
helper_functions.py

This module provides helper functions and classes for constructing convolutional layers
and DenseNet building blocks. It includes utility functions for 1x1 and 3x3 convolutions,
as well as implementations of a DenseLayer and DenseBlock commonly used in DenseNet architectures.
A sample run is provided in the __main__ block to demonstrate basic functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_planes, out_planes, stride=1):
    """
    Constructs a 1x1 convolution layer.

    Parameters:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int): Stride of the convolution.

    Returns:
        nn.Conv2d: A 1x1 convolutional layer without bias.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    Constructs a 3x3 convolution layer with padding.

    Parameters:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int): Stride of the convolution.
        groups (int): Number of blocked connections from input channels to output channels.
        dilation (int): Spacing between kernel elements.

    Returns:
        nn.Conv2d: A 3x3 convolutional layer without bias.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

# --------------------
# DenseNet Components
# --------------------
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        """
        Initializes a single DenseLayer used within a DenseBlock.

        Parameters:
            num_input_features (int): Number of input feature maps.
            growth_rate (int): Number of feature maps to add per layer.
            bn_size (int): Multiplicative factor for bottleneck layers.
            drop_rate (float): Dropout rate.
            memory_efficient (bool): If True, use a memory-efficient implementation.
        """
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        """
        Performs batch normalization, ReLU activation, and a 1x1 convolution
        on concatenated input features.

        Parameters:
            inputs (list of torch.Tensor): List of feature maps to be concatenated.

        Returns:
            torch.Tensor: Output after applying batch normalization, ReLU, and convolution.
        """
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, init_features, prev_features):
        """
        Forward pass for the DenseLayer.

        Parameters:
            init_features (torch.Tensor): The initial feature map (unused in this implementation).
            prev_features (list of torch.Tensor): List of previous feature maps.

        Returns:
            torch.Tensor: New features produced by this DenseLayer.
        """
        new_features = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        """
        Initializes a DenseBlock which consists of multiple DenseLayers.

        Parameters:
            num_layers (int): Number of DenseLayers in the block.
            num_input_features (int): Number of input feature maps.
            bn_size (int): Multiplicative factor for bottleneck layers.
            growth_rate (int): Number of feature maps to add per layer.
            drop_rate (float): Dropout rate.
            memory_efficient (bool): If True, use a memory-efficient implementation.
        """
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.layers.append(layer)

    def forward(self, init_features):
        """
        Forward pass for the DenseBlock.

        Parameters:
            init_features (torch.Tensor): Input feature map to the block.

        Returns:
            torch.Tensor: Concatenated output of all DenseLayers.
        """
        features = [init_features]
        for layer in self.layers:
            new_features = layer(init_features, features)
            features.append(new_features)
        return torch.cat(features, 1)


def main():
    """
    Demonstrates a sample run of helper functions and DenseNet components.
    """
    # Test the conv1x1 and conv3x3 helper functions.
    dummy_input = torch.randn(1, 3, 32, 32)
    conv1 = conv1x1(3, 8)
    conv3 = conv3x3(3, 8)
    print("conv1x1 output shape:", conv1(dummy_input).shape)
    print("conv3x3 output shape:", conv3(dummy_input).shape)

    # Test the DenseBlock with 2 DenseLayers.
    num_layers = 2
    num_input_features = 3
    bn_size = 2
    growth_rate = 4
    drop_rate = 0.0
    dense_block = _DenseBlock(num_layers, num_input_features, bn_size, growth_rate, drop_rate)
    dense_out = dense_block(dummy_input)
    print("DenseBlock output shape:", dense_out.shape)


if __name__ == "__main__":
    main()
