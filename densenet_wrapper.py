#!/usr/bin/env python
"""
compare_densenet.py

Defines three variants of DenseNet121 using the compare.py DenseNet implementation:
    - DenseNet121 base
    - DenseNet121 with ImageNet pretrained weights
    - DenseNet121 attention-augmented using CheXpert hyperparameters

Each model outputs a number of classes (default 5 for CheXpert) and can be imported in training scripts.
"""

import torch
from densenet import DenseNet

def densenet121_base(num_classes=5):
    """
    Returns a DenseNet121 base model (randomly initialized).
    """
    model = DenseNet(32, (6, 12, 24, 16), 64, num_classes=num_classes)
    return model

def densenet121_imagenet(num_classes=5):
    """
    Returns a DenseNet121 model with ImageNet pretrained weights.
    This uses the same architecture as the base model and then loads the pretrained weights.
    Note: Loading is done with strict=False to allow adapting the final classifier.
    """
    model = DenseNet(32, (6, 12, 24, 16), 64, num_classes=num_classes)
    # Load pretrained weights from torchvision (keys may differ slightly)
    state_dict = torch.hub.load_state_dict_from_url(
        "https://download.pytorch.org/models/densenet121-a639ec97.pth", progress=True
    )
    model.load_state_dict(state_dict, strict=False)
    # Replace the classifier to match the desired number of output classes
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    return model

def densenet121_attention(num_classes=5):
    """
    Returns an attention-augmented DenseNet121.
    Hyperparameters for attention augmentation (k, v, nh, relative, input_dims) are chosen
    following the CheXpert paper (using 320x320 as the input image size).
    """
    attn_params = {
        'k': 0.2,
        'v': 0.5,
        'nh': 8,
        'relative': True,
        'input_dims': (320, 320)
    }
    model = DenseNet(32, (6, 12, 24, 16), 64, num_classes=num_classes, attn_params=attn_params)
    return model

if __name__ == '__main__':
    # Quick instantiation test for all DenseNet variants
    for name, creator in zip(["base", "imagenet", "attention"],
                             [densenet121_base, densenet121_imagenet, densenet121_attention]):
        model = creator(num_classes=5)
        print(f"DenseNet121 {name} model instantiated with {sum(p.numel() for p in model.parameters())} parameters.")
