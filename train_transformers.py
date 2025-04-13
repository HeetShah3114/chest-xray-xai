#!/usr/bin/env python
"""
train_transformers.py

Trains a pretrained transformer model on the CheXpert small dataset with wandb tracking,
Automatic Mixed Precision (AMP), and progress bars. By default, the active code trains a
Swin Transformer model using timm. The file also contains a commented-out section
that was used to train a Vision Transformer (ViT) model.

Functions:
    load_pretrained_model: Loads a pretrained model from timm and adapts the classification head.
    train_one_epoch: Runs one epoch of training.
    validate: Evaluates the model on the validation set.
    main: Sets up training configuration, data, model, and runs the training loop.

Usage:
    Run the script directly to train the ViT model:
        python train_transformer.py
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
import timm

from chexpert_small import ChexpertSmall
from torchmetrics.classification import MultilabelAUROC


def load_pretrained_model(model_name, num_classes=5):
    """
    Loads a pretrained transformer model (e.g., ViT or Swin) from timm and adapts its classification head.

    Parameters:
        model_name (str): Name of the timm model (e.g., "vit_base_patch16_224", "swin_tiny_patch4_window7_224").
        num_classes (int): Number of output classes (default is 5).

    Returns:
        model (nn.Module): The model with an updated classification head.
    """
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs, scaler, num_classes):
    """
    Runs one training epoch.

    Parameters:
        model (nn.Module): The transformer model.
        dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device for training.
        epoch (int): Current epoch number.
        total_epochs (int): Total number of epochs.
        scaler (torch.cuda.amp.GradScaler): AMP scaler for mixed precision training.
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
    Evaluates the model on the validation set.

    Parameters:
        model (nn.Module): The transformer model.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device for evaluation.
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
    Main training loop: parses arguments, sets up data, loads the model, and runs training and validation.
    """
    parser = argparse.ArgumentParser(description="Train pretrained Vision Transformer on CheXpert small with wandb tracking")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224",
                        help="Pretrained timm model name (e.g., vit_base_patch16_224, swin_tiny_patch4_window7_224)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory for CheXpert dataset")
    parser.add_argument("--project", type=str, default="chexpert_vit", help="wandb project name")
    args = parser.parse_args()

    wandb.init(project=args.project, config=vars(args))
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained model and move it to the device.
    model = load_pretrained_model(args.model_name, num_classes=5)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    # Data transformations: resize to 224x224 and normalize with ImageNet stats.
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ChexpertSmall(root=args.data_root, mode="train", transform=transform)
    valid_dataset = ChexpertSmall(root=args.data_root, mode="valid", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss, train_acc, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs, scaler, num_classes=5)
        val_loss, val_acc, val_auc = validate(model, valid_loader, criterion, device, num_classes=5)

        elapsed = time.time() - start_time

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
            best_path = os.path.join(args.save_dir, f"vit_{args.model_name}_best.pth")
            torch.save(model.state_dict(), best_path)
            wandb.save(best_path)

    final_path = os.path.join(args.save_dir, f"vit_{args.model_name}_final.pth")
    torch.save(model.state_dict(), final_path)
    wandb.save(final_path)
    wandb.finish()


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------------
# The following section is commented out. It represents the code previously used to
# train a Vision Transformer (ViT) model.
# --------------------------------------------------------------------------------
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
# import timm
# import numpy as np
#
# from chexpert_small import ChexpertSmall
# from torchmetrics.classification import MultilabelAUROC
#
#
# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#
#     def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta
#         self.path = path
#
#     def __call__(self, val_loss, model):
#         score = -val_loss
#
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             if self.verbose:
#                 print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#             self.counter = 0
#
#     def save_checkpoint(self, val_loss, model):
#         '''Saves model when validation loss decrease.'''
#         if self.verbose:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
#         torch.save(model.state_dict(), self.path)
#         self.val_loss_min = val_loss
#
#
# def load_pretrained_model(model_name, num_classes=5, gradual_unfreeze=True):
#     """
#     Loads a pretrained vision transformer with optional layer freezing strategies.
#     """
#     model = timm.create_model(model_name, pretrained=True, num_classes=0)  # Remove classifier head
#
#     # Add custom classification head with dropout for regularization
#     if 'vit' in model_name.lower():
#         # For ViT models
#         print("Training ViT")
#         in_features = model.num_features
#         model.head = nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(in_features, num_classes)
#         )
#     elif 'swin' in model_name.lower():
#         # For Swin Transformer models
#         model.reset_classifier(num_classes, global_pool='avg')
#     else:
#         # For other models
#         in_features = model.get_classifier().in_features
#         model.reset_classifier(num_classes)
#         setattr(model, 'head', nn.Sequential(
#             nn.Dropout(0.3),
#             nn.Linear(in_features, num_classes)
#         ))
#
#     # Implement gradual unfreezing for improved transfer learning
#     if gradual_unfreeze:
#         # Freeze all layers initially
#         for param in model.parameters():
#             param.requires_grad = False
#
#         # Unfreeze the head (we always train this)
#         for param in model.head.parameters():
#             param.requires_grad = True
#
#     return model
#
#
# def unfreeze_layers(model, current_epoch, total_epochs, model_name):
#     """Gradually unfreeze layers as training progresses."""
#     if 'vit' in model_name.lower():
#         # For ViT models with blocks
#         if current_epoch < total_epochs * 0.2:
#             # First 20% of training: only fine-tune the head
#             pass
#         elif current_epoch < total_epochs * 0.5:
#             # Next 30%: also fine-tune the last 2 blocks
#             for name, param in model.blocks[-2:].named_parameters():
#                 param.requires_grad = True
#         elif current_epoch < total_epochs * 0.8:
#             # Next 30%: fine-tune the last 6 blocks
#             for name, param in model.blocks[-6:].named_parameters():
#                 param.requires_grad = True
#         else:
#             # Last 20%: fine-tune all layers
#             for param in model.parameters():
#                 param.requires_grad = True
#     elif 'swin' in model_name.lower():
#         # For Swin Transformer models
#         if current_epoch < total_epochs * 0.2:
#             # First 20%: only fine-tune the head (i.e. do nothing)
#             pass
#         elif current_epoch < total_epochs * 0.5:
#             # Next 30%: also fine-tune the last stage
#             for name, param in model.layers[-1].named_parameters():
#                 param.requires_grad = True
#         elif current_epoch < total_epochs * 0.8:
#             # Next 30%: fine-tune the last two stages
#             for name, param in model.layers[-2:].named_parameters():
#                 param.requires_grad = True
#         else:
#             # Last 20%: fine-tune all layers
#             for param in model.parameters():
#                 param.requires_grad = True
#     else:
#         # For other models, implement a simpler strategy:
#         if current_epoch < total_epochs * 0.5:
#             # First half: only train head
#             pass
#         else:
#             # Second half: train all parameters
#             for param in model.parameters():
#                 param.requires_grad = True
#
#     # Count trainable parameters for logging
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return trainable_params
#
# def get_class_weights(dataset, num_classes=5):
#     """
#     Calculate class weights based on dataset distribution.
#     """
#     pos_counts = torch.zeros(num_classes)
#     neg_counts = torch.zeros(num_classes)
#
#     for _, labels, _ in tqdm(dataset, desc="Calculating class weights"):
#         for i in range(num_classes):
#             if labels[i] == 1:
#                 pos_counts[i] += 1
#             else:
#                 neg_counts[i] += 1
#
#     # Calculate positive weights
#     pos_weights = neg_counts / pos_counts
#     return pos_weights
#
#
# def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs, scaler, num_classes,
#                     model_name):
#     model.train()
#     running_loss = 0.0
#     correct_predictions = 0
#     total_predictions = 0
#     auc_metric = MultilabelAUROC(num_labels=num_classes, average=None).to(device)
#
#     # Unfreeze more layers as training progresses
#     trainable_params = unfreeze_layers(model, epoch, total_epochs, model_name)
#     wandb.log({"trainable_parameters": trainable_params, "epoch": epoch})
#
#     progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", leave=False)
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
#
#         # Add gradient clipping to prevent exploding gradients
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#
#         scaler.step(optimizer)
#         scaler.update()
#
#         running_loss += loss.item() * images.size(0)
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
#     all_outputs = []
#     all_labels = []
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
#             # Save predictions and labels for further analysis
#             all_outputs.append(outputs.cpu())
#             all_labels.append(labels.cpu())
#
#     epoch_loss = running_loss / len(dataloader.dataset)
#     epoch_acc = correct_predictions / total_predictions
#     epoch_auc = auc_metric.compute()
#     auc_metric.reset()
#
#     return epoch_loss, epoch_acc, epoch_auc, torch.cat(all_outputs), torch.cat(all_labels)
#
#
# def main():
#     parser = argparse.ArgumentParser(
#         description="Train pretrained Vision Transformer on CheXpert small with wandb tracking")
#     parser.add_argument("--model_name", type=str, default="swin_tiny_patch4_window7_224",
#                         help="Pretrained timm model name (e.g., vit_base_patch16_224, swin_tiny_patch4_window7_224)")
#     parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
#     parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
#     parser.add_argument("--lr", type=float, default=5e-5, help="Initial learning rate")
#     parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
#     parser.add_argument("--data_root", type=str, default="./data", help="Root directory for CheXpert dataset")
#     parser.add_argument("--project", type=str, default="chexpert_swin_pretrained", help="wandb project name")
#     parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for regularization")
#     parser.add_argument("--warmup_epochs", type=int, default=2, help="Number of warmup epochs")
#     parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
#     parser.add_argument("--auto_class_weights", action="store_true", help="Calculate class weights automatically")
#     args = parser.parse_args()
#
#     wandb.init(project=args.project, config=vars(args))
#     config = wandb.config
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # Data transformations with more augmentation for training
#     train_transform = transforms.Compose([
#         transforms.Lambda(lambda img: img.convert("RGB")),
#         transforms.Resize((224, 224)),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=0.1, contrast=0.1),
#         transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
#
#     # Less augmentation for validation
#     valid_transform = transforms.Compose([
#         transforms.Lambda(lambda img: img.convert("RGB")),
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
#
#     train_dataset = ChexpertSmall(root=args.data_root, mode="train", transform=train_transform)
#     valid_dataset = ChexpertSmall(root=args.data_root, mode="valid", transform=valid_transform)
#
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
#     valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
#
#     # Load model with gradual unfreezing strategy
#     model = load_pretrained_model(args.model_name, num_classes=5, gradual_unfreeze=True)
#     model = model.to(device)
#
#     # Calculate or set class weights for the loss function
#     if args.auto_class_weights:
#         print("Calculating class weights from dataset...")
#         pos_weights = get_class_weights(train_dataset)
#         print(f"Class weights: {pos_weights}")
#     else:
#         # Default weights based on CheXpert class imbalance
#         pos_weights = torch.tensor([1.5, 3.0, 2.0, 2.5, 1.8])  # Adjust based on domain knowledge
#
#     pos_weights = pos_weights.to(device)
#     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
#
#     # Optimizer with weight decay for regularization
#     optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#
#     # Learning rate scheduler with warmup
#     def warmup_cosine_schedule(epoch):
#         if epoch < args.warmup_epochs:
#             # Linear warmup
#             return float(epoch) / float(max(1, args.warmup_epochs))
#         else:
#             # Cosine annealing
#             progress = float(epoch - args.warmup_epochs) / float(args.epochs - args.warmup_epochs)
#             return 0.5 * (1. + np.cos(np.pi * progress))
#
#     scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_schedule)
#
#     scaler = torch.cuda.amp.GradScaler()
#
#     # Setup early stopping
#     os.makedirs(args.save_dir, exist_ok=True)
#     early_stopping = EarlyStopping(patience=args.patience, verbose=True,
#                                    path=os.path.join(args.save_dir, f"{args.model_name}_best.pth"))
#
#     best_val_loss = float("inf")
#     best_val_auc_avg = 0.0
#
#     for epoch in range(1, args.epochs + 1):
#         start_time = time.time()
#         current_lr = optimizer.param_groups[0]['lr']
#         wandb.log({"learning_rate": current_lr, "epoch": epoch})
#
#         train_loss, train_acc, train_auc = train_one_epoch(
#             model, train_loader, criterion, optimizer, device,
#             epoch, args.epochs, scaler, num_classes=5, model_name=args.model_name
#         )
#
#         val_loss, val_acc, val_auc, val_outputs, val_labels = validate(
#             model, valid_loader, criterion, device, num_classes=5
#         )
#
#         # Update learning rate
#         scheduler.step()
#
#         # Calculate average AUC across all classes
#         val_auc_avg = torch.mean(val_auc).item()
#
#         elapsed = time.time() - start_time
#
#         wandb.log({
#             "epoch": epoch,
#             "train_loss": train_loss,
#             "train_acc": train_acc,
#             "val_loss": val_loss,
#             "val_acc": val_acc,
#             "val_auc_avg": val_auc_avg,
#             "epoch_time": elapsed,
#             **{f"train_auc_class_{i}": v.item() for i, v in enumerate(train_auc)},
#             **{f"val_auc_class_{i}": v.item() for i, v in enumerate(val_auc)}
#         })
#
#         print(f"Epoch {epoch}/{args.epochs} - "
#               f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
#               f"Val AUC Avg: {val_auc_avg:.4f}, "
#               f"Time: {elapsed:.2f}s")
#
#         # Save model if validation AUC improves
#         if val_auc_avg > best_val_auc_avg:
#             best_val_auc_avg = val_auc_avg
#             best_auc_path = os.path.join(args.save_dir, f"{args.model_name}_best_auc.pth")
#             torch.save(model.state_dict(), best_auc_path)
#             wandb.save(best_auc_path)
#             print(f"Saved new best model with AUC: {best_val_auc_avg:.4f}")
#
#         # Early stopping check based on validation loss
#         early_stopping(val_loss, model)
#         if early_stopping.early_stop:
#             print("Early stopping triggered")
#             break
#
#     # Load the best model for final evaluation
#     model.load_state_dict(torch.load(early_stopping.path))
#     final_val_loss, final_val_acc, final_val_auc, _, _ = validate(model, valid_loader, criterion, device, num_classes=5)
#
#     print(f"Final model performance - "
#           f"Val Loss: {final_val_loss:.4f}, Val AUC Avg: {torch.mean(final_val_auc).item():.4f}")
#
#     # Save final model
#     final_path = os.path.join(args.save_dir, f"{args.model_name}_final.pth")
#     torch.save(model.state_dict(), final_path)
#     wandb.save(final_path)
#
#     # Log final metrics to wandb
#     wandb.log({
#         "final_val_loss": final_val_loss,
#         "final_val_acc": final_val_acc,
#         "final_val_auc_avg": torch.mean(final_val_auc).item(),
#         **{f"final_val_auc_class_{i}": v.item() for i, v in enumerate(final_val_auc)}
#     })
#
#     wandb.finish()
#
#
# if __name__ == "__main__":
#     main()
