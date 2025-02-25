#!/usr/bin/env python
"""
densenet.py

Defines the DenseNet architecture with support for both standard and attention-augmented variants.
This implementation uses helper_functions.py for the DenseNet building blocks and aa_conv2d.py for
attention-augmented convolutional layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from helper_functions import _DenseBlock, conv1x1
from aa_conv2d import AAConv2d

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, attn_params=None):
        super(_Transition, self).__init__()
        if attn_params is not None:
            nh = attn_params['nh']
            dk = max(20 * nh, int((attn_params['k'] * num_output_features // nh) * nh))
            dv = int((attn_params['v'] * num_output_features // nh) * nh)
            relative = attn_params['relative']
            # Use a strided conv with attention augmentation for downsampling.
            input_dims = (attn_params['input_dims'][0] // 2, attn_params['input_dims'][1] // 2)
            self.add_module('norm', nn.InstanceNorm2d(num_input_features))
            self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('conv', AAConv2d(num_input_features, num_output_features,
                                             kernel_size=3, stride=2,
                                             dk=dk, dv=dv, nh=nh,
                                             relative=relative, input_dims=input_dims))
        else:
            # Standard DenseNet transition (1x1 conv + avg pooling)
            self.add_module('norm', nn.BatchNorm2d(num_input_features))
            self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                              kernel_size=1, stride=1, bias=False))
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000,
                 attn_params=None):
        super(DenseNet, self).__init__()

        # Initial convolution and pooling (using the ImageNet configuration if 4 blocks)
        if len(block_config) == 4:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7,
                                    stride=2, padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_features)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))
            if attn_params is not None:
                # Adjust input dimensions for the attention module after the pooling layer.
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
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2,
                                    attn_params=attn_params)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
            if attn_params is not None:
                # Downscale input_dims for subsequent layers after a transition.
                attn_params['input_dims'] = (attn_params['input_dims'][0] // 2,
                                             attn_params['input_dims'][1] // 2)

        # Final batch norm and classifier.
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
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out

if __name__ == '__main__':
    # Quick test of the DenseNet implementation.
    model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16),
                     num_init_features=64, num_classes=10)
    print("DenseNet parameters:", sum(p.numel() for p in model.parameters()))
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape)
