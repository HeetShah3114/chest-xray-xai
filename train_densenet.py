#!/usr/bin/env python
"""
train_densenet.py

Trains a DenseNet121 variant (base, ImageNet-pretrained, or attention-augmented)
on the CheXpert small dataset. This script integrates wandb for logging training metrics,
tqdm for progress visualization, and torch.cuda.amp for Automatic Mixed Precision (AMP).

Functions:
    train_one_epoch: Runs one epoch of training.
    validate: Evaluates the model on the validation set.
    main: Sets up training, handles command-line arguments, and runs the training loop.

A sample run is provided in the main() function.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import wandb
from tqdm import tqdm
from torchmetrics.classification import MultilabelAUROC

from chexpert_small import ChexpertSmall
from densenet_wrapper import densenet121_base, densenet121_imagenet, densenet121_attention


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs, scaler, num_classes):
    """
    Trains the model for one epoch.

    Parameters:
        model (nn.Module): The DenseNet model.
        dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to run training on.
        epoch (int): Current epoch number.
        total_epochs (int): Total number of epochs.
        scaler (torch.cuda.amp.GradScaler): For AMP training.
        num_classes (int): Number of output classes.

    Returns:
        tuple: Epoch training loss, accuracy, and per-class AUC metrics.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    auc_metric = MultilabelAUROC(num_labels=num_classes, average=None).to(device)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
    for images, labels, _ in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        # Compute binary predictions using a 0.5 threshold.
        preds = torch.sigmoid(outputs) > 0.5
        correct_predictions += (preds == labels).sum().item()
        total_predictions += labels.numel()
        auc_metric.update(torch.sigmoid(outputs), labels.int())

        batch_acc = (preds == labels).float().mean().item()
        progress_bar.set_postfix(loss=loss.item(), acc=batch_acc)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions / total_predictions
    epoch_auc = auc_metric.compute()
    auc_metric.reset()

    return epoch_loss, epoch_acc, epoch_auc


def validate(model, dataloader, criterion, device, num_classes):
    """
    Evaluates the model on the validation set.

    Parameters:
        model (nn.Module): The DenseNet model.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.
        num_classes (int): Number of output classes.

    Returns:
        tuple: Validation loss, accuracy, and per-class AUC metrics.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    auc_metric = MultilabelAUROC(num_labels=num_classes, average=None).to(device)

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
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
    Sets up training configuration, initializes datasets and model,
    and executes the training and validation loop with wandb logging.
    """
    parser = argparse.ArgumentParser(description="Train DenseNet variants on CheXpert small with wandb tracking")
    parser.add_argument("--variant", type=str, default="attention",
                        choices=["base", "imagenet", "attention"],
                        help="DenseNet variant: base, imagenet, or attention")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for CheXpert dataset")
    parser.add_argument("--project", type=str, default="chexpert_densenet-attention",
                        help="wandb project name")
    args = parser.parse_args()

    wandb.init(project=args.project, config=vars(args))
    config = wandb.config

    device = torch.device("cuda")

    # Instantiate the selected DenseNet variant and define the appropriate transforms.
    if args.variant == "base":
        model = densenet121_base(num_classes=5)
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    elif args.variant == "imagenet":
        model = densenet121_imagenet(num_classes=5)
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    elif args.variant == "attention":
        model = densenet121_attention(num_classes=5)
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError("Unknown variant")

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    # Prepare training and validation datasets and dataloaders.
    train_dataset = ChexpertSmall(root=args.data_root, mode="train", transform=transform)
    valid_dataset = ChexpertSmall(root=args.data_root, mode="valid", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    # Training loop.
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs, scaler, num_classes=5
        )
        val_loss, val_acc, val_auc = validate(model, valid_loader, criterion, device, num_classes=5)

        elapsed = time.time() - start_time

        # Log metrics to wandb.
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time": elapsed,
            **{f"train_auc_class_{i}": v.item() for i, v in enumerate(train_auc)},
            **{f"val_auc_class_{i}": v.item() for i, v in enumerate(val_auc)}
        })

        # Save the best model based on validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, f"densenet_{args.variant}_best.pth")
            torch.save(model.state_dict(), best_path)
            wandb.save(best_path)

    # Save the final model.
    final_path = os.path.join(args.save_dir, f"densenet_{args.variant}_final.pth")
    torch.save(model.state_dict(), final_path)
    wandb.save(final_path)
    wandb.finish()


if __name__ == "__main__":
    main()
