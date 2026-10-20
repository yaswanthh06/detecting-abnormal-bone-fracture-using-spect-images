import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from backend.ml.model import load_and_validate_checkpoint
from backend.ml.dataset import SPECTDataset, validate_manifest
from backend.app.config import settings

def evaluate_test_set(
    checkpoint_path: Path,
    manifest_path: Path,
    output_summary_path: Optional[Path] = None,
    device_name: str = "cpu",
    batch_size: int = 16,
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    manifest_path = Path(manifest_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Trained checkpoint not found at: {checkpoint_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at: {manifest_path}")

    # Validate manifest integrity and patient isolation
    manifest_info = validate_manifest(manifest_path)
    if not manifest_info["valid"]:
        raise ValueError(f"Manifest failed validation: {manifest_info.get('error') or manifest_info.get('leakage_details')}")

    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")

    # Load validated model
    model, metadata, err = load_and_validate_checkpoint(checkpoint_path, num_classes=settings.NUM_CLASSES, device=str(device))
    if err or model is None:
        raise ValueError(f"Cannot evaluate: {err}")

    # Standard test transform
    test_transform = transforms.Compose([
        transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = SPECTDataset(manifest_path, split="test", transform=test_transform)
    if len(test_dataset) == 0:
        raise ValueError("Held-out test split has 0 samples.")

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs_pos: List[float] = []

    model.eval()
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            probs_pos = probs[:, 1].cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            all_probs_pos.extend(probs_pos)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs_pos)

    # Core evaluation metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))  # Sensitivity
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC-AUC calculation (requires presence of both classes)
    roc_auc = None
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_probs))
        except Exception:
            roc_auc = None

    split_counts = {
        "train_images": manifest_info["split_counts"].get("train", 0),
        "val_images": manifest_info["split_counts"].get("val", 0),
        "test_images": manifest_info["split_counts"].get("test", 0),
        "total_images": manifest_info["total_samples"],
    }

    results = {
        "has_evaluation_data": True,
        "evaluation_timestamp": datetime.utcnow().isoformat() + "Z",
        "split_counts": split_counts,
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall_sensitivity": round(rec, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "test_sample_count": len(y_true),
        },
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": settings.CLASS_NAMES,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "manifest_path": str(manifest_path),
        "message": f"Held-out image-level test set evaluated successfully ({len(y_true)} images).",
    }

    if output_summary_path:
        out_p = Path(output_summary_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved evaluation summary to {out_p}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResNet-18 on held-out test split.")
    parser.add_argument("--checkpoint", type=str, default=str(settings.CHECKPOINT_PATH))
    parser.add_argument("--manifest", type=str, default=str(settings.DATASET_MANIFEST_PATH))
    parser.add_argument("--out", type=str, default=str(settings.EVALUATION_SUMMARY_PATH))
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    summary = evaluate_test_set(
        checkpoint_path=Path(args.checkpoint),
        manifest_path=Path(args.manifest),
        output_summary_path=Path(args.out),
        device_name=args.device,
    )
    print("Evaluation Complete:")
    print(json.dumps(summary["metrics"], indent=2))
