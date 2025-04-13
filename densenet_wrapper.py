#!/usr/bin/env python
"""
densenet_wrapper.py

This module defines three variants of DenseNet121 using a custom DenseNet implementation.
The variants include:
  - DenseNet121 base: a randomly initialized DenseNet121.
  - DenseNet121 ImageNet: DenseNet121 loaded with pretrained ImageNet weights,
      then adapted to output a desired number of classes.
  - DenseNet121 attention: an attention-augmented DenseNet121 with hyperparameters
      chosen following the CheXpert paper (using 320x320 as the input image size).

A sample run is provided in the main() function to quickly instantiate and test each variant.
"""

import torch
from densenet import DenseNet

def densenet121_base(num_classes=5):
    """
    Constructs a DenseNet121 base model with random initialization.

    Parameters:
        num_classes (int): Number of output classes (default is 5).

    Returns:
        model (DenseNet): An instance of DenseNet121.
    """
    model = DenseNet(32, (6, 12, 24, 16), 64, num_classes=num_classes)
    return model

def densenet121_imagenet(num_classes=5):
    """
    Constructs a DenseNet121 model with ImageNet pretrained weights.

    The model architecture is the same as the base model. Pretrained weights are loaded
    from the official URL, and the final classifier is replaced to match the desired
    number of output classes. Loading is performed with strict=False to allow slight
    mismatches between pretrained weights and the current architecture.

    Parameters:
        num_classes (int): Number of output classes (default is 5).

    Returns:
        model (DenseNet): An instance of DenseNet121 adapted to the desired number of classes.
    """
    model = DenseNet(32, (6, 12, 24, 16), 64, num_classes=num_classes)
    state_dict = torch.hub.load_state_dict_from_url(
        "https://download.pytorch.org/models/densenet121-a639ec97.pth", progress=True
    )
    # Remove the original classifier parameters to allow replacement.
    state_dict.pop("classifier.weight", None)
    state_dict.pop("classifier.bias", None)
    model.load_state_dict(state_dict, strict=False)
    # Replace the classifier with a new one matching num_classes.
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    return model

def densenet121_attention(num_classes=5):
    """
    Constructs an attention-augmented DenseNet121.

    The model uses additional attention parameters (k, v, nh, relative, input_dims) as defined
    by the CheXpert paper. The input dimensions are assumed to be 320x320.

    Parameters:
        num_classes (int): Number of output classes (default is 5).

    Returns:
        model (DenseNet): An attention-augmented DenseNet121.
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

def main():
    """
    Demonstrates a sample run by instantiating each DenseNet121 variant and printing the
    total number of parameters.
    """
    creators = {
        "base": densenet121_base,
        "imagenet": densenet121_imagenet,
        "attention": densenet121_attention
    }
    for name, creator in creators.items():
        model = creator(num_classes=5)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"DenseNet121 {name} model instantiated with {total_params} parameters.")
        print(model)

if __name__ == '__main__':
    main()
