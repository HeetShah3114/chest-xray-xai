#!/usr/bin/env python
"""
evaluate_resnet.py

Evaluates a ResNet152 variant (base, imagenet, or attention) on the CheXpert small validation dataset.
Loads the model from a checkpoint (.pth file) and computes evaluation metrics: loss, accuracy, and AUROC per class.

Usage:
    Run the script with appropriate command-line arguments to evaluate the chosen ResNet variant.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from chexpert_small import ChexpertSmall
from resnet_wrapper import resnet152_base, resnet152_imagenet, resnet152_attention
from torchmetrics.classification import MultilabelAUROC


def evaluate(model, dataloader, criterion, device, num_classes):
    """
    Evaluates the model on the provided dataset.

    Parameters:
        model (nn.Module): The ResNet model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.
        num_classes (int): Number of output classes.

    Returns:
        tuple: (epoch_loss, epoch_accuracy, epoch_auc) where:
            epoch_loss (float): Average loss over the dataset.
            epoch_accuracy (float): Overall accuracy.
            epoch_auc (Tensor): AUROC for each class.
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
            preds = torch.sigmoid(outputs) > 0.5  # Convert logits to binary predictions
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
    Main function to evaluate a ResNet variant on the CheXpert small validation dataset.

    Parses command-line arguments, loads the corresponding model variant and checkpoint,
    prepares the validation dataset, and computes evaluation metrics (loss, accuracy, and AUROC per class).
    """
    parser = argparse.ArgumentParser(description="Evaluate ResNet variant on CheXpert validation dataset")
    parser.add_argument("--variant", type=str, default="imagenet",
                        choices=["base", "imagenet", "attention"],
                        help="ResNet variant: base, imagenet, or attention")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/resnet_imagenet-freeze_best.pth",
                        help="Path to the .pth checkpoint file")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for CheXpert dataset")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of dataloader workers")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the appropriate model variant and set up the transform.
    if args.variant in ["base", "imagenet"]:
        if args.variant == "base":
            model = resnet152_base(num_classes=5)
        else:
            model = resnet152_imagenet(num_classes=5)
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    elif args.variant == "attention":
        model = resnet152_attention(num_classes=5)
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError("Unknown variant")

    # Load the model checkpoint.
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    # Prepare the validation dataset and dataloader.
    valid_dataset = ChexpertSmall(root=args.data_root, mode="valid", transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    # Evaluate the model.
    loss, acc, auc = evaluate(model, valid_loader, criterion, device, num_classes=5)

    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")

    # Display AUROC per class using disease names from the dataset.
    disease_names = ChexpertSmall.attr_names  # e.g., ['Atelectasis', 'Cardiomegaly', ...]
    print("Validation AUROC per disease:")
    for name, auc_score in zip(disease_names, auc):
        print(f"  {name}: {auc_score.item():.4f}")


if __name__ == "__main__":
    main()
