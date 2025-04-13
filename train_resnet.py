#!/usr/bin/env python
"""
train_resnet.py

Trains a ResNet152 variant (base, ImageNet-pretrained, or attention-augmented)
on the CheXpert small dataset using wandb tracking, Automatic Mixed Precision (AMP),
and progress bars. This file contains two training routines:
  1. The active routine (non-commented code) is used to train the Imagenet-freeze and attention variants;
  2. A commented-out section at the bottom shows the training routine used for the ResNet base model.

The active routine also demonstrates partial freezing of early layers,
a learning rate scheduler, and checkpointing for best and final models.
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
from resnet_wrapper import resnet152_base, resnet152_imagenet, resnet152_attention


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs, scaler, num_classes):
    """
    Trains the model for one epoch.

    Parameters:
        model (nn.Module): The ResNet model.
        dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device for training.
        epoch (int): Current epoch number.
        total_epochs (int): Total epochs.
        scaler (torch.cuda.amp.GradScaler): For AMP training.
        num_classes (int): Number of output classes.

    Returns:
        tuple: (epoch_loss, epoch_accuracy, epoch_auc)
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    auc_metric = MultilabelAUROC(num_labels=num_classes, average=None).to(device)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", leave=True)
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

        preds = torch.sigmoid(outputs) > 0.5
        correct_predictions += (preds == labels).sum().item()
        total_predictions += labels.numel()
        auc_metric.update(torch.sigmoid(outputs), labels.int())

        batch_acc = (preds == labels).float().mean().item()
        progress_bar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{batch_acc:.3f}")

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions / total_predictions
    epoch_auc = auc_metric.compute()
    auc_metric.reset()

    return epoch_loss, epoch_acc, epoch_auc


def validate(model, dataloader, criterion, device, num_classes):
    """
    Evaluates the model on the validation dataset.

    Parameters:
        model (nn.Module): The ResNet model.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device for evaluation.
        num_classes (int): Number of output classes.

    Returns:
        tuple: (validation_loss, validation_accuracy, validation_auc)
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
    Main training routine.

    Sets up argument parsing, initializes datasets, model, optimizer, scheduler,
    and runs the training and validation loop. Also implements partial freezing for
    the initial epochs (for Imagenet-freeze and attention variants).
    """
    parser = argparse.ArgumentParser(description="Train ResNet variants on CheXpert small with wandb tracking")
    parser.add_argument("--variant", type=str, default="imagenet-freeze",
                        choices=["base", "imagenet-freeze", "attention"],
                        help="ResNet variant: 'base' (for training without freezing, see commented section) or 'imagenet-freeze' / 'attention'")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--freeze_epochs", type=int, default=5, help="Initial epochs to freeze early layers")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory for CheXpert dataset")
    parser.add_argument("--project", type=str, default="chexpert_resnet-imagenet-freeze", help="wandb project name")
    args = parser.parse_args()

    wandb.init(project=args.project, config=vars(args))
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up transforms and instantiate the model based on the chosen variant.
    if args.variant == "base":
        model = resnet152_base(num_classes=5)
        train_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        val_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    elif args.variant == "imagenet-freeze":
        model = resnet152_imagenet(num_classes=5)
        train_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    elif args.variant == "attention":
        model = resnet152_attention(num_classes=5)
        train_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((360, 360)),
            transforms.RandomResizedCrop(320),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError("Unknown variant")

    model = model.to(device)

    # Partial freezing: for imagenet-freeze and attention variants, freeze early layers for initial epochs.
    if args.freeze_epochs > 0:
        if hasattr(model, "layer1"):
            for param in model.layer1.parameters():
                param.requires_grad = False
        if hasattr(model, "layer2"):
            for param in model.layer2.parameters():
                param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler()

    train_dataset = ChexpertSmall(root=args.data_root, mode="train", transform=train_transform)
    valid_dataset = ChexpertSmall(root=args.data_root, mode="valid", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # At the specified freeze_epochs, unfreeze early layers and reinitialize optimizer and scheduler.
        if epoch == args.freeze_epochs + 1:
            if hasattr(model, "layer1"):
                for param in model.layer1.parameters():
                    param.requires_grad = True
            if hasattr(model, "layer2"):
                for param in model.layer2.parameters():
                    param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3,
                                                             verbose=True)
            print(f"Unfroze early layers at epoch {epoch} and reinitialized optimizer.")

        train_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs, scaler, num_classes=5
        )
        val_loss, val_acc, val_auc = validate(
            model, valid_loader, criterion, device, num_classes=5
        )

        elapsed = time.time() - start_time
        scheduler.step(val_loss)

        summary = (
            f"Epoch {epoch}/{args.epochs}: "
            f"[Time: {elapsed:0.2f}s, train_acc: {train_acc:0.3f}, train_loss: {train_loss:0.3f}, "
            f"val_acc: {val_acc:0.3f}, val_loss: {val_loss:0.3f}]"
        )
        tqdm.write(summary)

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, f"resnet_{args.variant}_best.pth")
            torch.save(model.state_dict(), best_path)
            wandb.save(best_path)

    final_path = os.path.join(args.save_dir, f"resnet_{args.variant}_final.pth")
    torch.save(model.state_dict(), final_path)
    wandb.save(final_path)
    wandb.finish()


if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------
# The section below is commented out.
# It contains an alternative training routine that was used for training the ResNet base model.
# You can refer to this section if you wish to train the base variant without partial freezing.
# -------------------------------------------------------------------------
#
# #!/usr/bin/env python
# """
# train_resnet.py
#
# Trains a ResNet152 variant (base, ImageNet-pretrained, or attention-augmented)
# on the CheXpert small dataset with wandb tracking, Automatic Mixed Precision (AMP),
# and progress bars. This routine was used for training the ResNet base model.
# """
#
# import argparse
# import os
# import time
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms
#
# import wandb
# from tqdm import tqdm
#
# from chexpert_small import ChexpertSmall
# from resnet_wrapper import resnet152_base, resnet152_imagenet, resnet152_attention
#
# from torchmetrics.classification import MultilabelAUROC
#
#
# def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs, scaler, num_classes):
#     model.train()
#     running_loss = 0.0
#     correct_predictions = 0
#     total_predictions = 0
#     auc_metric = MultilabelAUROC(num_labels=num_classes, average=None).to(device)
#
#     progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", leave=True)
#     for images, labels, _ in progress_bar:
#         images = images.to(device)
#         labels = labels.to(device)
#
#         optimizer.zero_grad()
#         with torch.cuda.amp.autocast():
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#
#         running_loss += loss.item() * images.size(0)
#
#         preds = torch.sigmoid(outputs) > 0.5
#         correct_predictions += (preds == labels).sum().item()
#         total_predictions += labels.numel()
#         auc_metric.update(torch.sigmoid(outputs), labels.int())
#
#         batch_acc = (preds == labels).float().mean().item()
#         progress_bar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{batch_acc:.3f}")
#
#     epoch_loss = running_loss / len(dataloader.dataset)
#     epoch_acc = correct_predictions / total_predictions
#     epoch_auc = auc_metric.compute()
#     auc_metric.reset()
#
#     return epoch_loss, epoch_acc, epoch_auc
#
#
# def validate(model, dataloader, criterion, device, num_classes):
#     model.eval()
#     running_loss = 0.0
#     correct_predictions = 0
#     total_predictions = 0
#     auc_metric = MultilabelAUROC(num_labels=num_classes, average=None).to(device)
#
#     with torch.no_grad():
#         for images, labels, _ in dataloader:
#             images = images.to(device)
#             labels = labels.to(device)
#             with torch.cuda.amp.autocast():
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#
#             running_loss += loss.item() * images.size(0)
#             preds = torch.sigmoid(outputs) > 0.5
#             correct_predictions += (preds == labels).sum().item()
#             total_predictions += labels.numel()
#             auc_metric.update(torch.sigmoid(outputs), labels.int())
#
#     epoch_loss = running_loss / len(dataloader.dataset)
#     epoch_acc = correct_predictions / total_predictions
#     epoch_auc = auc_metric.compute()
#     auc_metric.reset()
#
#     return epoch_loss, epoch_acc, epoch_auc
#
#
# def main():
#     parser = argparse.ArgumentParser(description="Train ResNet variants on Chexpert small with wandb tracking")
#     parser.add_argument("--variant", type=str, default="imagenet",
#                         choices=["base", "imagenet", "attention"],
#                         help="ResNet variant: base, imagenet, or attention")
#     parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train")
#     parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
#     parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
#     parser.add_argument("--save_dir", type=str, default="checkpoints",
#                         help="Directory to save model checkpoints")
#     parser.add_argument("--data_root", type=str, default="./data",
#                         help="Root directory for Chexpert dataset")
#     parser.add_argument("--project", type=str, default="chexpert_resnet-imagenet",
#                         help="wandb project name")
#     args = parser.parse_args()
#
#     wandb.init(project=args.project, config=vars(args))
#     config = wandb.config
#
#     device = torch.device("cuda")
#
#     if args.variant == "base":
#         model = resnet152_base(num_classes=5)
#         transform = transforms.Compose([
#             transforms.Lambda(lambda img: img.convert("RGB")),
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#         ])
#     elif args.variant == "imagenet":
#         model = resnet152_imagenet(num_classes=5)
#         transform = transforms.Compose([
#             transforms.Lambda(lambda img: img.convert("RGB")),
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#         ])
#     elif args.variant == "attention":
#         model = resnet152_attention(num_classes=5)
#         transform = transforms.Compose([
#             transforms.Lambda(lambda img: img.convert("RGB")),
#             transforms.Resize((320, 320)),
#             transforms.ToTensor(),
#         ])
#     else:
#         raise ValueError("Unknown variant")
#
#     model = model.to(device)
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#     scaler = torch.cuda.amp.GradScaler()
#
#     train_dataset = ChexpertSmall(root=args.data_root, mode="train", transform=transform)
#     valid_dataset = ChexpertSmall(root=args.data_root, mode="valid", transform=transform)
#
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
#     valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
#
#     os.makedirs(args.save_dir, exist_ok=True)
#     best_val_loss = float("inf")
#
#     for epoch in range(1, args.epochs + 1):
#         start_time = time.time()
#
#         train_loss, train_acc, train_auc = train_one_epoch(
#             model, train_loader, criterion, optimizer, device, epoch, args.epochs, scaler, num_classes=5
#         )
#         val_loss, val_acc, val_auc = validate(
#             model, valid_loader, criterion, device, num_classes=5
#         )
#
#         elapsed = time.time() - start_time
#
#         summary = (
#             f"Epoch {epoch}/{args.epochs}: "
#             f"[Time: {elapsed:0.2f}s, train_acc: {train_acc:0.3f}, train_loss: {train_loss:0.3f}, "
#             f"val_acc: {val_acc:0.3f}, val_loss: {val_loss:0.3f}]"
#         )
#         tqdm.write(summary)
#
#         wandb.log({
#             "epoch": epoch,
#             "train_loss": train_loss,
#             "train_acc": train_acc,
#             "val_loss": val_loss,
#             "val_acc": val_acc,
#             "epoch_time": elapsed,
#             **{f"train_auc_class_{i}": v.item() for i, v in enumerate(train_auc)},
#             **{f"val_auc_class_{i}": v.item() for i, v in enumerate(val_auc)}
#         })
#
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_path = os.path.join(args.save_dir, f"resnet_{args.variant}_best.pth")
#             torch.save(model.state_dict(), best_path)
#             wandb.save(best_path)
#
#     final_path = os.path.join(args.save_dir, f"resnet_{args.variant}_final.pth")
#     torch.save(model.state_dict(), final_path)
#     wandb.save(final_path)
#     wandb.finish()
#
#
# if __name__ == "__main__":
#     main()
