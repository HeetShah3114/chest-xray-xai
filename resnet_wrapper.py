#!/usr/bin/env python
"""
compare_resnet.py

Defines three variants of ResNet152 using the compare.py ResNet implementation:
    - ResNet152 base
    - ResNet152 with ImageNet pretrained weights
    - ResNet152 attention-augmented using CheXpert hyperparameters

Each model outputs a number of classes (default 5 for CheXpert) and can be imported in training scripts.
"""

import torch
from resnet import ResNet, Bottleneck, BasicBlock

def resnet152_base(num_classes=5):
    """
    Returns a ResNet152 base model (randomly initialized).
    ResNet152 uses the Bottleneck block with layer configuration [3, 8, 36, 3].
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    return model

def resnet152_imagenet(num_classes=5):
    """
    Returns a ResNet152 model with ImageNet pretrained weights.
    The pretrained weights are loaded from torchvision.
    Note: We use strict=False to adapt the final fully connected layer.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    state_dict = torch.hub.load_state_dict_from_url(
        "https://download.pytorch.org/models/resnet152-b121ed2d.pth", progress=True
    )
    model.load_state_dict(state_dict, strict=False)
    # Replace the final fully connected layer to match the desired number of output classes
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def resnet152_attention(num_classes=5):
    """
    Returns an attention-augmented ResNet152.
    The attention augmentation hyperparameters are chosen based on the CheXpert paper,
    here using an example configuration with input dimensions of 320x320.
    """
    attn_params = {
        'k': 0.2,
        'v': 0.1,
        'nh': 8,
        'relative': True,
        'input_dims': (320, 320)
    }
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, attn_params=attn_params)
    return model

if __name__ == '__main__':
    # Quick instantiation test for all ResNet152 variants
    for name, creator in zip(["base", "imagenet", "attention"],
                             [resnet152_base, resnet152_imagenet, resnet152_attention]):
        model = creator(num_classes=5)
        print(f"ResNet152 {name} model instantiated with {sum(p.numel() for p in model.parameters())} parameters.")
