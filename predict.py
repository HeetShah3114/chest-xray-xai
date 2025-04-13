#!/usr/bin/env python
"""
predict.py

This script loads a trained model from a checkpoint (.pth file) and prints the five class probabilities.
The model can be a DenseNet, ResNet, or Transformer variant. It uses command-line arguments to determine
the model type, variant, and other parameters. A sample run is provided in the __main__ block.
"""

import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import timm  # for transformer models

# Import model creation functions for CNNs
from densenet_wrapper import densenet121_base, densenet121_imagenet, densenet121_attention
from resnet_wrapper import resnet152_base, resnet152_imagenet, resnet152_attention

# CheXpert 5-class names
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']


def get_model(args, device):
    """
    Instantiates and returns the model along with the expected input image size.

    Parameters:
        args (Namespace): Command-line arguments with attributes model_type, model_variant, model_name, and checkpoint.
        device (torch.device): The device on which the model will be loaded (CPU or CUDA).

    Returns:
        model (torch.nn.Module): The loaded and evaluation-ready model.
        input_size (tuple): The expected input image dimensions (width, height).
    """
    if args.model_type == "densenet":
        if args.model_variant == "base":
            model = densenet121_base(num_classes=5)
            input_size = (224, 224)
        elif args.model_variant == "imagenet":
            model = densenet121_imagenet(num_classes=5)
            input_size = (224, 224)
        elif args.model_variant == "attention":
            model = densenet121_attention(num_classes=5)
            input_size = (320, 320)
        else:
            raise ValueError("Unsupported DenseNet variant. Choose from: base, imagenet, or attention.")
    elif args.model_type == "resnet":
        if args.model_variant == "base":
            model = resnet152_base(num_classes=5)
            input_size = (224, 224)
        elif args.model_variant == "imagenet":
            model = resnet152_imagenet(num_classes=5)
            input_size = (224, 224)
        elif args.model_variant == "attention":
            model = resnet152_attention(num_classes=5)
            input_size = (320, 320)
        else:
            raise ValueError("Unsupported ResNet variant. Choose from: base, imagenet, or attention.")
    elif args.model_type == "transformer":
        if args.model_variant not in ["vit", "swin"]:
            raise ValueError("Unsupported transformer variant. Choose 'vit' or 'swin'.")
        # Set default model_name if not provided.
        if not hasattr(args, "model_name") or args.model_name is None:
            if args.model_variant == "vit":
                args.model_name = "vit_base_patch16_224"
            elif args.model_variant == "swin":
                args.model_name = "swin_tiny_patch4_window7_224"
        model = timm.create_model(args.model_name, pretrained=False, num_classes=5)
        if args.model_variant == "vit":
            in_features = model.get_classifier().in_features if hasattr(model, "get_classifier") else model.num_features
            model.head = torch.nn.Sequential(
                torch.nn.Dropout(0.3),
                torch.nn.Linear(in_features, 5)
            )
        input_size = (224, 224)
    else:
        raise ValueError("Unsupported model type. Choose from: densenet, resnet, or transformer.")

    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model, input_size


def preprocess_image(image_path, input_size, model_type, model_variant):
    """
    Loads and preprocesses an image for model evaluation.

    Parameters:
        image_path (str): Path to the image file.
        input_size (tuple): Tuple specifying the target dimensions (width, height).
        model_type (str): The type of model (e.g., 'resnet', 'transformer') to determine normalization.
        model_variant (str): Specific variant of the model, used for conditional normalization.

    Returns:
        original_image (PIL.Image.Image): The original image object.
        tensor_image (torch.Tensor): The preprocessed image tensor ready for model input.
    """
    transform_list = [
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(input_size),
        transforms.ToTensor()
    ]
    # Apply normalization for specific models/variants.
    if model_type == "resnet" and model_variant == "imagenet":
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )
    if model_type == "transformer":
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )
    transform = transforms.Compose(transform_list)
    image = Image.open(image_path)
    return image, transform(image).unsqueeze(0)


def predict_image(model_type, model_variant, model_name, checkpoint_path, image_path, device=None):
    """
    Core prediction function that can be called from both CLI and Flask app.

    Parameters:
        model_type (str): The type of model (densenet, resnet, transformer)
        model_variant (str): Specific variant of the model
        model_name (str): For transformer models, the timm model name
        checkpoint_path (str): Path to the model checkpoint
        image_path (str): Path to the input image
        device (torch.device, optional): Device to run inference on

    Returns:
        dict: Class probabilities as a dictionary {class_name: probability}
        np.ndarray: Raw probability array
    """

    # Create an args-like object for get_model()
    class Args:
        def __init__(self):
            self.model_type = model_type
            self.model_variant = model_variant
            self.model_name = model_name
            self.checkpoint = checkpoint_path
            self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = Args()

    # Load model and preprocess image
    model, input_size = get_model(args, args.device)
    _, input_tensor = preprocess_image(image_path, input_size, model_type, model_variant)
    input_tensor = input_tensor.to(args.device)

    # Get predictions
    with torch.no_grad():
        outputs = model(input_tensor)
    probabilities = torch.sigmoid(outputs).squeeze(0)
    probabilities_np = probabilities.detach().cpu().numpy()

    # Format results as dictionary
    result = {CLASS_NAMES[i]: float(probabilities_np[i]) for i in range(len(CLASS_NAMES))}

    return result, probabilities_np


def main():
    """
    Main function to parse arguments, load the model and image, and display predictions.

    This function demonstrates a sample run of the script:
      - Parses command-line arguments.
      - Loads the model using get_model.
      - Preprocesses the input image.
      - Performs a forward pass to obtain class probabilities.
      - Prints the probabilities for each class in descending order.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained model and print class probabilities.")
    parser.add_argument("--model_type", type=str, default="transformer",
                        choices=["densenet", "resnet", "transformer"],
                        help="Model type: densenet, resnet, or transformer.")
    parser.add_argument("--model_variant", type=str, default="vit",
                        help="For CNNs: variant can be base, imagenet, or attention. For transformer models: choose vit or swin.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="For transformer models, optionally specify the timm model name. Defaults are set based on model_variant if not provided.")
    parser.add_argument("--checkpoint", type=str,
                        default="./checkpoints/vit_base_patch16_224_best_auc.pth",
                        help="Path to the .pth checkpoint file.")
    parser.add_argument("--image", type=str,
                        default="./data/CheXpert-v1.0-small/valid/patient64577/study1/view1_frontal.jpg",
                        help="Path to the input image file.")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device id.")
    args = parser.parse_args()

    # Set the device based on CUDA availability.
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # Call the core prediction function
    result, probabilities_np = predict_image(
        args.model_type,
        args.model_variant,
        args.model_name,
        args.checkpoint,
        args.image,
        device
    )

    # Sort probabilities for display
    sorted_indices = np.argsort(probabilities_np)[::-1]

    print("Predicted class probabilities (from most to least likely):")
    for idx in sorted_indices:
        print(f"{CLASS_NAMES[idx]}: {probabilities_np[idx] * 100:.2f}%")


if __name__ == "__main__":
    main()