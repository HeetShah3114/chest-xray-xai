#!/usr/bin/env python
"""
gradcam.py

Implements GradCAM (Gradient-weighted Class Activation Mapping) for DenseNet models
trained on the CheXpert dataset. GradCAM visualizes regions of the input image that
were important for the model's prediction by highlighting them with a heatmap.

Classes:
    GradCAM: Core implementation for extracting and computing gradients.
    DenseNetGradCAM: DenseNet-specific implementation of GradCAM.
    ResNetGradCAM: ResNet-specific implementation of GradCAM.

Functions:
    visualize_cam: Creates a heatmap visualization of GradCAM output.
    overlay_heatmap: Overlays a heatmap on an image.

A sample run is provided in the main() function.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms


class GradCAM:
    """
    Base class for GradCAM implementation.

    Captures activations and gradients from a specified target layer
    and computes the weighted activation map (heatmap).
    """

    def __init__(self, model, target_layer):
        """
        Initialize GradCAM.

        Parameters:
            model (nn.Module): The model to examine.
            target_layer (nn.Module): The layer from which to extract activations.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # Register hooks for forward and backward passes
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Store activations during forward pass."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Store gradients during backward pass."""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        """
        Generate a class activation map.

        Parameters:
            input_image (Tensor): Input image tensor of shape (1, C, H, W).
            target_class (int, optional): Target class index. If None, uses the model's highest prediction.

        Returns:
            numpy.ndarray: GradCAM heatmap (normalized to 0-1).
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_image)

        if target_class is None:
            # If no target class is specified, use the class with the highest prediction
            target_class = torch.argmax(output).item()

        # Zero gradients
        self.model.zero_grad()

        # Create one-hot encoding for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1

        # Backward pass to compute gradients
        output.backward(gradient=one_hot, retain_graph=True)

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

        # Compute weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # Apply ReLU to keep only positive influences
        cam = F.relu(cam)

        # Normalize the CAM to 0-1
        cam_np = cam.squeeze().cpu().numpy()
        if np.max(cam_np) > 0:
            cam_np = cam_np / np.max(cam_np)
        return cam_np


class DenseNetGradCAM(GradCAM):
    """
    DenseNet-specific GradCAM implementation.

    Targets the final normalization layer (norm5) in DenseNet's feature extractor,
    which typically contains semantically rich features.
    """

    def __init__(self, model):
        """
        Initialize DenseNetGradCAM.

        Parameters:
            model (nn.Module): A DenseNet model (e.g., torchvision.models.densenet121).
        """
        # For DenseNet, target the final normalization layer
        target_layer = model.features.norm5
        super(DenseNetGradCAM, self).__init__(model, target_layer)

    def __call__(self, input_image, target_class=None, multilabel=True):
        """
        Generate GradCAM for DenseNet.

        Parameters:
            input_image (Tensor): Input image tensor of shape (1, C, H, W).
            target_class (int, optional): Target class index. If None, uses the model's highest prediction.
            multilabel (bool): If True, applies sigmoid to the output; otherwise softmax.

        Returns:
            tuple: (GradCAM heatmap as numpy array, class probabilities as numpy array).
        """
        # Ensure batch dimension
        if len(input_image.shape) == 3:
            input_image = input_image.unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_image)
            if multilabel:
                probs = torch.sigmoid(output)
            else:
                probs = torch.softmax(output, dim=1)

        if target_class is None:
            target_class = torch.argmax(probs).item()

        cam = self.generate_cam(input_image, target_class)
        return cam, probs.cpu().numpy()


class ResNetGradCAM(GradCAM):
    """
    ResNet-specific GradCAM implementation.

    Targets the bn3 layer of the last Bottleneck block in layer4,
    which typically contains semantically rich features.
    """

    def __init__(self, model):
        # For ResNet built with Bottleneck blocks, target bn3 of the last block in layer4.
        target_layer = model.layer4[-1].bn3
        super(ResNetGradCAM, self).__init__(model, target_layer)

    def __call__(self, input_image, target_class=None, multilabel=True):
        """
        Generate GradCAM for ResNet.

        Parameters:
            input_image (Tensor): Input image tensor of shape (1, C, H, W) or (C, H, W).
            target_class (int, optional): If None, uses the highest predicted class.
            multilabel (bool): If True, applies sigmoid; otherwise softmax.

        Returns:
            tuple: (GradCAM heatmap as numpy array, class probabilities as numpy array).
        """
        if len(input_image.shape) == 3:
            input_image = input_image.unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_image)
            if multilabel:
                probs = torch.sigmoid(output)
            else:
                probs = torch.softmax(output, dim=1)

        if target_class is None:
            target_class = torch.argmax(probs).item()

        cam = self.generate_cam(input_image, target_class)
        return cam, probs.cpu().numpy()


def visualize_cam(image, cam, alpha=0.5):
    """
    Create a visualization by overlaying a CAM heatmap on the original image.

    Parameters:
        image (numpy.ndarray): Original image in RGB format (H, W, C).
        cam (numpy.ndarray): CAM heatmap (H, W).
        alpha (float): Transparency factor for overlay.

    Returns:
        numpy.ndarray: Image with heatmap overlay.
    """
    # Resize CAM to match image dimensions
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    # Convert the CAM to a heatmap using a colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    # Convert heatmap from BGR to RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # Overlay heatmap on original image
    overlay = heatmap * alpha + image * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def overlay_heatmap(image_tensor, cam, alpha=0.5, return_pil=False):
    """
    Overlays a CAM heatmap on an image tensor.

    Parameters:
        image_tensor (Tensor): Image tensor (C, H, W).
        cam (numpy.ndarray): CAM heatmap.
        alpha (float): Transparency factor.
        return_pil (bool): If True, returns a PIL Image.

    Returns:
        numpy.ndarray or PIL.Image: Image with CAM overlay.
    """
    # Convert tensor to numpy array and transpose to (H, W, C)
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    # Scale image to 0-255 if necessary
    if image.max() <= 1.0:
        image = image * 255
    image = image.astype(np.uint8)
    # For grayscale images, convert to RGB
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    result = visualize_cam(image, cam, alpha)
    if return_pil:
        return Image.fromarray(result)
    return result


def main():
    """
    Demonstrates GradCAM visualization on a sample image using a pretrained DenseNet121 model.

    The script accepts command-line arguments for an input image and output path.
    It preprocesses the image, computes the GradCAM heatmap using DenseNetGradCAM,
    overlays the heatmap on the original image, and displays/saves the result.
    """
    parser = argparse.ArgumentParser(description="GradCAM Visualization for DenseNet Models")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Path to save output visualization")
    args = parser.parse_args()

    # Load and preprocess the image
    input_pil = Image.open(args.image).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(input_pil).unsqueeze(0)

    # Load a pretrained DenseNet121 model from torchvision
    model = models.densenet121(pretrained=True)
    model.eval()

    # Initialize DenseNetGradCAM with the model
    gradcam = DenseNetGradCAM(model)
    # Generate GradCAM heatmap and get prediction probabilities
    cam, probs = gradcam(input_tensor)
    print("Predicted probabilities:", probs)

    # Overlay the heatmap on the original (un-normalized) image for visualization.
    # Convert original image to numpy array.
    input_np = np.array(input_pil.resize((224, 224)))
    overlay = visualize_cam(input_np, cam, alpha=0.5)

    # Display the result using PIL
    result_img = Image.fromarray(overlay)
    result_img.show()

    # Save the output image if an output path is provided
    if args.output:
        result_img.save(args.output)
        print(f"Saved overlay image to {args.output}")


if __name__ == "__main__":
    main()
