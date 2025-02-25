#!/usr/bin/env python
"""
train_resnet.py

Trains a ResNet152 variant (base, ImageNet-pretrained, or attention-augmented)
on the CheXpert small dataset with wandb logging. Saves the best and final model checkpoints.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb

from chexpert_small import ChexpertSmall
from resnet_wrapper import resnet152_base, resnet152_imagenet, resnet152_attention

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels, _ in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def main():
    parser = argparse.ArgumentParser(description="Train ResNet variants on CheXpert small with wandb tracking")
    parser.add_argument("--variant", type=str, default="base",
                        choices=["base", "imagenet", "attention"],
                        help="ResNet variant: base, imagenet, or attention")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train (start with 3 for fine-tuning)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for CheXpert dataset")
    parser.add_argument("--project", type=str, default="chexpert_resnet",
                        help="wandb project name")
    args = parser.parse_args()

    # Initialize wandb for tracking experiments
    wandb.init(project=args.project, config={
        "variant": args.variant,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "architecture": "ResNet152",
    })
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the selected ResNet variant
    if args.variant == "base":
        model = resnet152_base(num_classes=5)
    elif args.variant == "imagenet":
        model = resnet152_imagenet(num_classes=5)
    elif args.variant == "attention":
        model = resnet152_attention(num_classes=5)
    else:
        raise ValueError("Unknown variant")

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Prepare the CheXpert training and validation datasets
    train_dataset = ChexpertSmall(root=args.data_root, mode="train", transform=None)
    valid_dataset = ChexpertSmall(root=args.data_root, mode="valid", transform=None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    best_val_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, valid_loader, criterion, device)
        elapsed = time.time() - start_time

        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Time: {elapsed:.2f}s")
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time": elapsed,
        })

        # Save the best model checkpoint based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, f"resnet_{args.variant}_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model checkpoint to {best_path}")
            wandb.save(best_path)

    # Save the final model checkpoint after training
    final_path = os.path.join(args.save_dir, f"resnet_{args.variant}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")
    wandb.save(final_path)
    wandb.finish()

if __name__ == "__main__":
    main()
