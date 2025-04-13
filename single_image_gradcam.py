#!/usr/bin/env python
"""
single_image_gradcam.py

Generates GradCAM visualization for a single chest X-ray image from the CheXpert dataset.
Supports both ResNet and DenseNet architectures.

Usage:
    python single_image_gradcam.py --arch resnet --variant attention --checkpoint ./checkpoints/resnet_attention_best.pth --image_path ./path/to/image.jpg
    python single_image_gradcam.py --arch densenet --variant imagenet --checkpoint ./checkpoints/densenet_imagenet_best.pth --image_path ./path/to/image.jpg
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import base64
from io import BytesIO

# Import both model wrappers and GradCAM classes
from resnet_wrapper import resnet152_base, resnet152_imagenet, resnet152_attention
from densenet_wrapper import densenet121_base, densenet121_imagenet, densenet121_attention
from gradcam import ResNetGradCAM, DenseNetGradCAM, overlay_heatmap

# CheXpert 5-class names (matching predict.py)
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']


def load_image(image_path, transform):
    """
    Loads and transforms an image.

    Parameters:
        image_path (str): Path to the image file.
        transform (callable): Transformation to apply.

    Returns:
        tuple: (transformed image tensor, original PIL image)
    """
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return transform(img), img


def load_model_for_gradcam(arch, variant, checkpoint_path, device):
    """
    Loads a model for GradCAM visualization.

    Parameters:
        arch (str): Architecture name ('resnet' or 'densenet')
        variant (str): Model variant ('base', 'imagenet', or 'attention')
        checkpoint_path (str): Path to the model checkpoint
        device (torch.device): Device to load the model on

    Returns:
        tuple: (model, gradcam_instance, input_size)
    """
    # Select model variant and set image size based on architecture and variant
    if arch == "resnet":
        if variant == "base":
            model = resnet152_base(num_classes=5)
            img_size = 224
        elif variant == "imagenet":
            model = resnet152_imagenet(num_classes=5)
            img_size = 224
        elif variant == "attention":
            model = resnet152_attention(num_classes=5)
            img_size = 320
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Create GradCAM instance for ResNet
        gradcam_inst = ResNetGradCAM(model)

    elif arch == "densenet":
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

        # Create GradCAM instance for DenseNet
        gradcam_inst = DenseNetGradCAM(model)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model, gradcam_inst, img_size


def generate_gradcam(arch, variant, checkpoint_path, image_path, target_class=None, device=None):
    """
    Generates GradCAM visualization that can be called from both CLI and Flask app.

    Parameters:
        arch (str): Architecture name ('resnet' or 'densenet')
        variant (str): Model variant ('base', 'imagenet', or 'attention')
        checkpoint_path (str): Path to the model checkpoint
        image_path (str): Path to the input image
        target_class (int, optional): Target class index (0-4)
        device (torch.device, optional): Device to run on

    Returns:
        dict: A dictionary containing:
            - target_class (str): Name of the target class
            - probabilities (dict): Class probabilities
            - cam (numpy.ndarray): GradCAM heatmap
            - overlay_img (numpy.ndarray): GradCAM overlay on original image
    """
    # Set device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model, GradCAM instance, and get image size
    model, gradcam_inst, img_size = load_model_for_gradcam(arch, variant, checkpoint_path, device)

    # Define transformation: resize image to the required dimensions and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # Load and process image
    img_tensor, original_img = load_image(image_path, transform)
    img_tensor = img_tensor.to(device)

    # Generate GradCAM heatmap; add batch dimension to image tensor
    cam, probs = gradcam_inst(img_tensor.unsqueeze(0), target_class=target_class, multilabel=True)

    # Determine target class for display
    target_idx = target_class if target_class is not None else int(probs[0].argmax())

    # Create visualization: overlay the heatmap on the original image
    viz_img = overlay_heatmap(img_tensor, cam, alpha=0.6)

    # Format results
    results = {
        "target_class": CLASS_NAMES[target_idx],
        "probabilities": {CLASS_NAMES[i]: float(probs[0][i]) for i in range(len(CLASS_NAMES))},
        "cam": cam,
        "overlay_img": viz_img
    }

    return results


def save_and_display_gradcam(results, output_path, original_img, img_size, image_path):
    """
    Saves and displays GradCAM visualization results.

    Parameters:
        results (dict): Results from generate_gradcam
        output_path (str): Path to save the output image
        original_img (PIL.Image): Original image
        img_size (int): Image size for resizing
    """
    # Extract data from results
    cam = results["cam"]
    viz_img = results["overlay_img"]
    probs = list(results["probabilities"].values())
    target_class = results["target_class"]

    # Plot original image, heatmap, and overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(np.asarray(original_img.resize((img_size, img_size))))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # GradCAM heatmap
    heatmap_im = axes[1].imshow(cam, cmap='jet')
    axes[1].set_title(f"GradCAM for {target_class}")
    axes[1].axis('off')
    plt.colorbar(heatmap_im, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlaid image
    axes[2].imshow(viz_img)
    target_idx = CLASS_NAMES.index(target_class)
    axes[2].set_title(f"Overlay (Prob: {probs[target_idx]:.3f})")
    axes[2].axis('off')

    # Display class probabilities and image filename on the figure
    prob_text = "\n".join([f"{name}: {prob:.3f}" for name, prob in zip(CLASS_NAMES, probs)])
    plt.figtext(0.02, 0.02, prob_text, fontsize=10)
    plt.figtext(0.98, 0.02, os.path.basename(image_path), horizontalalignment='right', fontsize=10)

    # Save and show the visualization
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Visualization saved to {output_path}")
    plt.show()


def convert_gradcam_to_base64(results):
    """
    Converts GradCAM overlay image to base64 string for web display.

    Parameters:
        results (dict): Results from generate_gradcam

    Returns:
        str: Base64-encoded string of the overlay image
    """
    viz_img = results["overlay_img"]

    # Convert the overlay (assumed to be a NumPy array) to a PIL image
    if viz_img.dtype != np.uint8:
        viz_img = (viz_img * 255).astype(np.uint8)
    viz_pil = Image.fromarray(viz_img)

    # Save the visualization to a bytes buffer
    buffer = BytesIO()
    viz_pil.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return img_base64


def main():
    parser = argparse.ArgumentParser(description="GradCAM visualization for a single CheXpert image")
    parser.add_argument("--arch", type=str, default="resnet", choices=["resnet", "densenet"],
                        help="Architecture: resnet or densenet")
    parser.add_argument("--variant", type=str, default="base",
                        choices=["base", "imagenet", "attention"],
                        help="Model variant: base, imagenet, or attention")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/resnet_base_best.pth",
                        help="Path to the model checkpoint (.pth file)")
    parser.add_argument("--image_path", type=str,
                        default="./data/CheXpert-v1.0-small/valid/patient64577/study1/view1_frontal.jpg",
                        help="Path to the image file")
    parser.add_argument("--output_path", type=str, default="./gradcam_result.png",
                        help="Path to save the visualization result")
    parser.add_argument("--target_class", type=int, default=None,
                        help="Target class index (0-4). If not specified, uses the class with highest prediction.")
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get GradCAM results
    results = generate_gradcam(
        args.arch,
        args.variant,
        args.checkpoint,
        args.image_path,
        args.target_class,
        device
    )

    # For CLI usage, load original image for display
    _, img_size = load_model_for_gradcam(args.arch, args.variant, args.checkpoint, device)[0::2]
    transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    _, original_img = load_image(args.image_path, transform)

    # Display and save results
    save_and_display_gradcam(results, args.output_path, original_img, img_size, args.image_path)


if __name__ == '__main__':
    main()