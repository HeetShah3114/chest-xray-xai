#!/usr/bin/env python
"""
densenet.py

Defines the DenseNet architecture with support for both standard and attention-augmented variants.
This implementation leverages building blocks from helper_functions.py for DenseNet components and
aa_conv2d.py for attention-augmented convolutional layers.

Classes:
    _Transition: Implements the transition layers between dense blocks with an option for attention augmentation.
    DenseNet: Implements the complete DenseNet architecture.

A sample run is provided in the main() function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from helper_functions import _DenseBlock, conv1x1
from aa_conv2d import AAConv2d

class _Transition(nn.Sequential):
    """
    Implements a transition layer between dense blocks.
    This layer performs downsampling via either a standard 1x1 conv + avg pooling or,
    if attention parameters are provided, a strided convolution with attention augmentation.
    """
    def __init__(self, num_input_features, num_output_features, attn_params=None):
        """
        Initializes the transition layer.

        Parameters:
            num_input_features (int): Number of input feature maps.
            num_output_features (int): Number of output feature maps.
            attn_params (dict, optional): Dictionary with attention augmentation parameters.
                If provided, the layer will use an attention-augmented convolution.
        """
        super(_Transition, self).__init__()
        if attn_params is not None:
            nh = attn_params['nh']
            # Compute dk based on a scaling factor (ensuring it is a multiple of nh)
            dk = max(20 * nh, int((attn_params['k'] * num_output_features // nh) * nh))
            # Compute dv similarly.
            dv = int((attn_params['v'] * num_output_features // nh) * nh)
            relative = attn_params['relative']
            # Calculate new input dimensions after downsampling by a factor of 2.
            input_dims = (attn_params['input_dims'][0] // 2, attn_params['input_dims'][1] // 2)
            self.add_module('norm', nn.InstanceNorm2d(num_input_features))
            self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('conv', AAConv2d(num_input_features, num_output_features,
                                             kernel_size=3, stride=2,
                                             dk=dk, dv=dv, nh=nh,
                                             relative=relative, input_dims=input_dims))
        else:
            # Standard DenseNet transition: 1x1 convolution followed by average pooling.
            self.add_module('norm', nn.BatchNorm2d(num_input_features))
            self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                              kernel_size=1, stride=1, bias=False))
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    """
    Implements the DenseNet architecture.

    This class builds the DenseNet model using an initial convolution block,
    a series of dense blocks interleaved with transition layers, and a final
    classifier. Attention augmentation can be enabled via the attn_params argument.
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000,
                 attn_params=None):
        """
        Initializes the DenseNet model.

        Parameters:
            growth_rate (int): Growth rate for the dense blocks.
            block_config (tuple): Tuple indicating the number of layers in each dense block.
            num_init_features (int): Number of features from the initial convolution.
            bn_size (int): Multiplicative factor for bottleneck layers.
            drop_rate (float): Dropout rate.
            num_classes (int): Number of output classes.
            attn_params (dict, optional): If provided, enables attention augmentation in transition layers.
        """
        super(DenseNet, self).__init__()

        # Initial convolution and pooling (ImageNet configuration if 4 blocks).
        if len(block_config) == 4:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7,
                                    stride=2, padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))
            if attn_params is not None:
                # Adjust input dimensions for attention modules after pooling.
                attn_params['input_dims'] = (attn_params['input_dims'][0] // 4,
                                             attn_params['input_dims'][1] // 4)
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=5,
                                    stride=1, padding=2, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
            ]))

        # Build dense blocks and transitions.
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # Create a dense block.
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            # Add a transition layer except after the last block.
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2,
                                    attn_params=attn_params)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            if attn_params is not None:
                # Downscale input_dims for subsequent layers.
                attn_params['input_dims'] = (attn_params['input_dims'][0] // 2,
                                             attn_params['input_dims'][1] // 2)

        # Final batch normalization and classifier.
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

        # Optional weight initialization.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Defines the forward pass of the DenseNet model.

        Parameters:
            x (torch.Tensor): Input tensor with shape (B, 3, H, W).

        Returns:
            torch.Tensor: Output logits with shape (B, num_classes).
        """
        features = self.features(x)
        out = F.relu(features, inplace=False)
        # Apply global average pooling and flatten.
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out

def main():
    """
    Demonstrates a sample run by instantiating a DenseNet model and performing a forward pass
    with dummy input.
    """
    # Instantiate DenseNet with example parameters.
    model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16),
                     num_init_features=64, num_classes=10)
    total_params = sum(p.numel() for p in model.parameters())
    print("DenseNet parameters:", total_params)
    # Create a dummy input tensor.
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)

if __name__ == '__main__':
    main()
