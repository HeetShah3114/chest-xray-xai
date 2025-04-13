#!/usr/bin/env python
"""
visualize_gradcam.py

Visualizes GradCAM for DenseNet models trained on the CheXpert dataset.
For a given set of sample images, this script generates GradCAM visualizations
showing which regions of the chest X-rays influenced the model's predictions.

Usage:
    python visualize_gradcam.py --variant attention --checkpoint ./checkpoints/densenet_attention_best.pth
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torchvision import transforms
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader

from chexpert_small import ChexpertSmall
from densenet_wrapper import densenet121_base, densenet121_imagenet, densenet121_attention
from gradcam import DenseNetGradCAM, overlay_heatmap


def get_model_and_transform(variant, checkpoint_path, device):
    """
    Instantiates a DenseNet model and appropriate transforms.

    Parameters:
        variant (str): DenseNet variant ('base', 'imagenet', or 'attention')
        checkpoint_path (str): Path to the model checkpoint
        device (torch.device): Device to load the model on

    Returns:
        tuple: (model, transform)
    """
    # Select model variant
    if variant == "base":
        model = densenet121_base(num_classes=5)
        img_size = 224
    elif variant == "imagenet":
        model = densenet121_imagenet(num_classes=5)
        img_size = 224
    elif variant == "attention":
        model = densenet121_attention(num_classes=5)
        img_size = 320
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Define transform
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    return model, transform, img_size


def visualize_samples(model, dataset, gradcam, output_dir, device, class_names, num_samples=5):
    """
    Generates GradCAM visualizations for sample images.

    Parameters:
        model (nn.Module): The DenseNet model
        dataset (Dataset): Dataset containing samples to visualize
        gradcam (DenseNetGradCAM): GradCAM implementation for the model
        output_dir (str): Directory to save visualizations
        device (torch.device): Computation device
        class_names (list): Names of the classes/pathologies
        num_samples (int): Number of samples to visualize per class
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create visualizations for each class
    for class_idx, class_name in enumerate(class_names):
        print(f"Generating visualizations for {class_name}...")

        # Find positive samples for this class
        positive_samples = []
        for i in range(len(dataset)):
            _, label, orig_idx = dataset[i]
            if label[class_idx] == 1:
                positive_samples.append((i, orig_idx))
                if len(positive_samples) >= num_samples:
                    break

        # Generate visualizations for the positive samples
        for sample_idx, (idx, orig_idx) in enumerate(positive_samples):
            img, label, _ = dataset[idx]
            img = img.to(device)

            # Generate GradCAM for the current class
            cam, probs = gradcam(img.unsqueeze(0), target_class=class_idx, multilabel=True)

            # Get original image path for reference
            img_path = dataset.data.iloc[idx, 0]

            # Create visualization
            viz_img = overlay_heatmap(img, cam, alpha=0.6)

            # Format probabilities
            class_probs = [f"{p:.3f}" for p in probs[0]]

            # Create figure with original image, GradCAM, and overlaid image
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original image
            axes[0].imshow(img.cpu().permute(1, 2, 0).numpy())
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            # GradCAM heatmap
            heatmap = axes[1].imshow(cam, cmap='jet')
            axes[1].set_title(f"GradCAM for {class_name}")
            axes[1].axis('off')
            plt.colorbar(heatmap, ax=axes[1], fraction=0.046, pad=0.04)

            # Overlaid image
            axes[2].imshow(viz_img)
            axes[2].set_title(f"Overlay (Prob: {class_probs[class_idx]})")
            axes[2].axis('off')

            # Add information about all class probabilities
            prob_text = "\n".join([f"{name}: {prob}" for name, prob in zip(class_names, class_probs)])
            plt.figtext(0.02, 0.02, prob_text, fontsize=8)

            # Add image path for reference
            plt.figtext(0.98, 0.02, os.path.basename(img_path), fontsize=8, ha='right')

            # Save the figure
            filename = f"{class_name.lower().replace(' ', '_')}_{sample_idx + 1}.png"
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=150)
            plt.close(fig)

    print(f"Visualizations saved to {output_dir}")


def main():
    """
    Main function to parse arguments and generate GradCAM visualizations.
    """
    parser = argparse.ArgumentParser(description="Visualize GradCAM for DenseNet on CheXpert")
    parser.add_argument("--variant", type=str, default="base", choices=["base", "imagenet", "attention"],
                        help="DenseNet variant: base, imagenet, or attention")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/densenet_base_best.pth",
                        help="Path to the model checkpoint (.pth file)")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for CheXpert dataset")
    parser.add_argument("--output_dir", type=str, default=".data/CheXpert-v1.0-small/gradcam_visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to visualize per class")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get model and transform
    model, transform, img_size = get_model_and_transform(args.variant, args.checkpoint, device)

    # Create GradCAM instance
    gradcam = DenseNetGradCAM(model)

    # Load dataset
    dataset = ChexpertSmall(root=args.data_root, mode="valid", transform=transform)

    # Get class names
    class_names = ChexpertSmall.attr_names

    # Create output directory
    output_dir = os.path.join(args.output_dir, args.variant)

    # Generate visualizations
    visualize_samples(model, dataset, gradcam, output_dir, device, class_names, args.num_samples)

    print("Done!")


if __name__ == "__main__":
    main()