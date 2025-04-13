#!/usr/bin/env python
"""
resnet_wrapper.py

This module defines three variants of ResNet152 using a custom ResNet implementation.
The variants include:
  - ResNet152 base: a randomly initialized ResNet152 with the Bottleneck block.
  - ResNet152 ImageNet: ResNet152 loaded with pretrained ImageNet weights, then adapted
      by replacing the final fully connected layer.
  - ResNet152 attention: an attention-augmented ResNet152 with hyperparameters inspired by
      the CheXpert paper (using 320x320 as the input image size).

A sample run is provided in the main() function to quickly instantiate and test each variant.
"""

import torch
from resnet import ResNet, Bottleneck, BasicBlock

def resnet152_base(num_classes=5):
    """
    Constructs a base ResNet152 model with random initialization.

    Parameters:
        num_classes (int): Number of output classes (default is 5).

    Returns:
        model (ResNet): An instance of ResNet152.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    return model

def resnet152_imagenet(num_classes=5):
    """
    Constructs a ResNet152 model with ImageNet pretrained weights.

    The model is first instantiated with 1000 classes to match the pretrained weights.
    Then the final fully connected layer is replaced to output the desired number of classes.

    Parameters:
        num_classes (int): Number of output classes (default is 5).

    Returns:
        model (ResNet): A ResNet152 model adapted for the given number of classes.
    """
    # Instantiate with 1000 classes for compatibility with pretrained weights.
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=1000)
    state_dict = torch.hub.load_state_dict_from_url(
        "https://download.pytorch.org/models/resnet152-b121ed2d.pth", progress=True
    )
    model.load_state_dict(state_dict)
    # Replace the final fully connected layer.
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def resnet152_attention(num_classes=5):
    """
    Constructs an attention-augmented ResNet152.

    The attention hyperparameters are set based on guidelines from the CheXpert paper,
    here using an example configuration with input dimensions of 320x320.

    Parameters:
        num_classes (int): Number of output classes (default is 5).

    Returns:
        model (ResNet): An attention-augmented ResNet152.
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

def main():
    """
    Demonstrates a sample run by instantiating each ResNet152 variant and printing the
    total number of parameters.
    """
    creators = {
        "base": resnet152_base,
        "imagenet": resnet152_imagenet,
        "attention": resnet152_attention
    }
    for name, creator in creators.items():
        model = creator(num_classes=5)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"ResNet152 {name} model instantiated with {total_params} parameters.")

if __name__ == '__main__':
    main()
