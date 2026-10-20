import argparse
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score

from backend.ml.model import build_model
from backend.ml.dataset import SPECTDataset, validate_manifest
from backend.app.config import settings

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / max(len(all_targets), 1)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    return {"loss": epoch_loss, "accuracy": epoch_acc, "f1": epoch_f1}


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / max(len(all_targets), 1)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    return {"loss": epoch_loss, "accuracy": epoch_acc, "f1": epoch_f1}


def run_training(
    manifest_path: Path,
    output_checkpoint_path: Path,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    optimizer_name: str = "adamw",
    weight_decay: float = 1e-2,
    device_name: str = "cpu",
    pretrained: bool = True,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    output_checkpoint_path = Path(output_checkpoint_path)
    output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    validation_report = validate_manifest(manifest_path)
    if not validation_report["valid"]:
        details = (
            validation_report.get("error")
            or validation_report.get("unreadable_files_sample")
            or validation_report.get("missing_files_sample")
            or validation_report.get("duplicate_paths_sample")
            or f"invalid rows: {validation_report.get('invalid_row_count', 0)}"
        )
        raise ValueError(f"Manifest validation failed before training: {details}")

    # Set device
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Data augmentations for FracAtlas X-ray training
    train_transform = transforms.Compose([
        transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = SPECTDataset(manifest_path, split="train", transform=train_transform)
    val_dataset = SPECTDataset(manifest_path, split="val", transform=val_transform)

    if len(train_dataset) == 0:
        raise ValueError("Train split in manifest is empty.")
    if len(val_dataset) == 0:
        raise ValueError("Val split in manifest is empty.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=len(train_dataset) > batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(num_classes=settings.NUM_CLASSES, pretrained=pretrained)
    model.to(device)

    # Optimizer selection
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = -1.0
    history = []

    print(f"Starting training: {epochs} epochs, device={device}, batch_size={batch_size}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_split(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 4),
            "train_acc": round(train_metrics["accuracy"], 4),
            "train_f1": round(train_metrics["f1"], 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_f1": round(val_metrics["f1"], 4),
            "time_seconds": round(elapsed, 2),
        }
        history.append(record)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} "
            f"Val Loss: {val_metrics['loss']:.4f} F1: {val_metrics['f1']:.4f}"
        )

        # Save best checkpoint
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            checkpoint_data = {
                "epoch": epoch,
                "best_val_f1": best_val_f1,
                "best_val_loss": val_metrics["loss"],
                "best_val_acc": val_metrics["accuracy"],
                "state_dict": model.state_dict(),
                "architecture": "resnet18",
                "num_classes": settings.NUM_CLASSES,
                "class_names": settings.CLASS_NAMES,
                "training_timestamp": datetime.utcnow().isoformat() + "Z",
                # torch.__version__ can be a TorchVersion (a str subclass),
                # which PyTorch 2.6's weights-only loader deliberately rejects.
                "pytorch_version": str(torch.__version__),
                "train_config": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "optimizer": optimizer_name,
                    "weight_decay": weight_decay,
                },
            }
            torch.save(checkpoint_data, output_checkpoint_path)
            print(f"-> Saved new best checkpoint to {output_checkpoint_path} (F1: {best_val_f1:.4f})")

    return {
        "best_val_f1": best_val_f1,
        "checkpoint_path": str(output_checkpoint_path),
        "history": history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-18 on FracAtlas X-ray images.")
    parser.add_argument("--manifest", type=str, default=str(settings.DATASET_MANIFEST_PATH))
    parser.add_argument("--checkpoint-out", type=str, default=str(settings.CHECKPOINT_PATH))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--pretrained", action="store_true", default=True)

    args = parser.parse_args()
    run_training(
        manifest_path=Path(args.manifest),
        output_checkpoint_path=Path(args.checkpoint_out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer_name=args.optimizer,
        device_name=args.device,
        pretrained=args.pretrained,
    )
