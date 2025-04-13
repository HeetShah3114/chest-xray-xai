#!/usr/bin/env python
"""
evaluate_transformers.py

Evaluates a Vision Transformer or Swin model on the CheXpert small validation dataset.
Loads the model from a checkpoint (.pth file) and computes evaluation metrics:
    - Loss
    - Accuracy
    - AUROC per class

Usage:
    Run the script with appropriate command-line arguments. For example:
      $ python evaluate_transformers.py --model_name vit_base_patch16_224 --checkpoint ./checkpoints/vit_base_patch16_224_best_auc.pth
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import timm
from chexpert_small import ChexpertSmall
from torchmetrics.classification import MultilabelAUROC


def evaluate(model, dataloader, criterion, device, num_classes):
    """
    Evaluates the model on the provided dataloader.

    Parameters:
        model (nn.Module): The transformer model to evaluate.
        dataloader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device on which to run evaluation.
        num_classes (int): Number of output classes.

    Returns:
        tuple: (epoch_loss, epoch_accuracy, epoch_auc)
            - epoch_loss (float): Average loss over the dataset.
            - epoch_accuracy (float): Overall accuracy.
            - epoch_auc (Tensor): AUROC for each class.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    auc_metric = MultilabelAUROC(num_labels=num_classes, average=None).to(device)

    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            # Convert logits to binary predictions using a threshold of 0.5.
            preds = torch.sigmoid(outputs) > 0.5
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.numel()
            auc_metric.update(torch.sigmoid(outputs), labels.int())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions / total_predictions
    epoch_auc = auc_metric.compute()
    auc_metric.reset()

    return epoch_loss, epoch_acc, epoch_auc


def main():
    """
    Main function to evaluate a Vision Transformer or Swin model on CheXpert validation data.

    Parses command-line arguments, creates the model using timm (with pretrained=False),
    loads the checkpoint, sets up the image transforms, and computes evaluation metrics:
    loss, accuracy, and AUROC per class.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate Vision Transformer or Swin model on CheXpert validation dataset"
    )
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224",
                        help="Pretrained timm model name (e.g., vit_base_patch16_224, swin_tiny_patch4_window7_224)")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/vit_base_patch16_224_best_auc.pth",
                        help="Path to the .pth checkpoint file")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for CheXpert dataset")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of dataloader workers")
    args = parser.parse_args()

    # Set device to CUDA if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the model using timm with pretrained=False; we load our checkpoint.
    model = timm.create_model(args.model_name, pretrained=False, num_classes=5)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "head.1.weight" in checkpoint and "head.weight" not in checkpoint:
        checkpoint["head.weight"] = checkpoint.pop("head.1.weight")
        checkpoint["head.bias"] = checkpoint.pop("head.1.bias")

    model.load_state_dict(checkpoint)
    model.to(device)

    # Define input image size. Default is 224 for both ViT and Swin models.
    input_size = 224
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    criterion = nn.BCEWithLogitsLoss()

    # Prepare the validation dataset and dataloader.
    valid_dataset = ChexpertSmall(root=args.data_root, mode="valid", transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    # Evaluate the model.
    loss, acc, auc = evaluate(model, valid_loader, criterion, device, num_classes=5)

    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")

    # Use disease names defined in ChexpertSmall for AUROC reporting.
    disease_names = ChexpertSmall.attr_names  # e.g., ['Atelectasis', 'Cardiomegaly', ...]
    print("Validation AUROC per disease:")
    for name, auc_score in zip(disease_names, auc):
        print(f"  {name}: {auc_score.item():.4f}")


if __name__ == "__main__":
    main()
