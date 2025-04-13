#!/usr/bin/env python
"""
vit_wrapper.py

This module provides functions to load pretrained Vision Transformer (ViT) and Swin Transformer models
using the timm library. The models are adapted to output a specified number of classes.

Functions:
  - vit_pretrained: Loads a pretrained ViT model and adjusts the classification head.
  - swin_pretrained: Loads a pretrained Swin Transformer model and adjusts the classification head.

A sample run is provided in the main() function that demonstrates model instantiation,
a forward pass with dummy input, and a simple parameter freezing example for the Swin model.
"""

import torch
import torch.nn as nn
import timm

def vit_pretrained(num_classes=5, model_name='vit_base_patch16_224'):
    """
    Loads a pretrained Vision Transformer (ViT) model and adapts it for the given number of classes.

    Parameters:
        num_classes (int): Number of output classes (default is 5).
        model_name (str): The timm model name (default is 'vit_base_patch16_224').

    Returns:
        model (nn.Module): A Vision Transformer model with the classification head adjusted.
    """
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

def swin_pretrained(num_classes=5, model_name='swin_tiny_patch4_window7_224'):
    """
    Loads a pretrained Swin Transformer model and adapts it for the given number of classes.

    Parameters:
        num_classes (int): Number of output classes (default is 5).
        model_name (str): The timm model name (default is 'swin_tiny_patch4_window7_224').

    Returns:
        model (nn.Module): A Swin Transformer model with the classification head adjusted.
    """
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

def main():
    """
    Demonstrates a sample run by:
      - Instantiating the pretrained ViT model and performing a forward pass with dummy input.
      - Instantiating the pretrained Swin Transformer model.
      - Freezing the patch embedding and first two stages of the Swin model.
      - Printing out the number of parameters frozen in each Swin layer.
    """
    # Test Vision Transformer (ViT)
    model_vit = vit_pretrained(num_classes=5)
    print("ViT model parameters:", sum(p.numel() for p in model_vit.parameters()))
    dummy_input = torch.rand(2, 3, 224, 224)  # Example input
    output_vit = model_vit(dummy_input)
    print("ViT output shape:", output_vit.shape)

    # Test Swin Transformer
    model_swin = swin_pretrained(num_classes=5)
    print("Swin model parameters:", sum(p.numel() for p in model_swin.parameters()))

    # Freeze patch embedding and first two stages (layers) of Swin Transformer.
    for param in model_swin.patch_embed.parameters():
        param.requires_grad = False

    for layer in model_swin.layers[:2]:
        for param in layer.parameters():
            param.requires_grad = False

    # Print the number of frozen parameters for each layer.
    for i, layer in enumerate(model_swin.layers):
        num_frozen = sum(not p.requires_grad for p in layer.parameters())
        total = sum(1 for p in layer.parameters())
        print(f"Layer {i}: {num_frozen}/{total} parameters frozen")

    output_swin = model_swin(dummy_input)
    print("Swin output shape:", output_swin.shape)

if __name__ == "__main__":
    main()
