import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

MANIFEST_FIELDS = {"image_path", "label", "split"}


def _decode_error(image_path: Path) -> Optional[str]:
    """Return a Pillow error unless the image can be verified and fully decoded."""
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            image.load()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


class FracAtlasDataset(Dataset):
    """PyTorch dataset for the FracAtlas ``image_path,label,split`` manifest."""

    def __init__(self, manifest_path: Path, split: Optional[str] = "train", transform=None, base_dir: Optional[Path] = None):
        self.manifest_path = Path(manifest_path)
        self.split = split.lower() if split else None
        self.base_dir = Path(base_dir) if base_dir else self.manifest_path.parent
        self.transform = transform
        self.samples: List[Dict[str, Any]] = []
        self._load_and_filter()

    def _load_and_filter(self) -> None:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not MANIFEST_FIELDS.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"Manifest must contain columns {sorted(MANIFEST_FIELDS)}. Found: {reader.fieldnames}")
            for row in reader:
                row_split = row["split"].strip().lower()
                if self.split and row_split != self.split:
                    continue
                label = int(row["label"])
                if label not in (0, 1):
                    raise ValueError(f"Invalid label {label}; expected 0 or 1")
                image_path = Path(row["image_path"].strip())
                if not image_path.is_absolute():
                    image_path = self.base_dir / image_path
                self.samples.append({"image_path": str(image_path), "label": label, "split": row_split})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        sample = self.samples[idx]
        try:
            with Image.open(sample["image_path"]) as image:
                image = image.convert("RGB")
                tensor = self.transform(image) if self.transform else transforms.ToTensor()(image)
        except Exception as exc:
            raise RuntimeError(f"Error loading image at {sample['image_path']}: {exc}") from exc
        return tensor, sample["label"], sample["image_path"]


# Preserve existing imports while using the FracAtlas manifest contract.
SPECTDataset = FracAtlasDataset


def validate_manifest(manifest_path: Path) -> Dict[str, Any]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return {"valid": False, "error": f"Manifest file not found: {manifest_path}"}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not MANIFEST_FIELDS.issubset(set(reader.fieldnames or [])):
            return {"valid": False, "error": f"Missing required columns {sorted(MANIFEST_FIELDS)}. Found: {reader.fieldnames}"}
        samples = list(reader)
    if not samples:
        return {"valid": False, "error": "Manifest CSV contains no data rows."}

    counts = {"train": 0, "val": 0, "test": 0, "unassigned": 0}
    class_counts: Dict[str, int] = {}
    missing_files: List[str] = []
    unreadable_files: List[str] = []
    duplicate_paths: List[str] = []
    invalid_rows = 0
    seen_paths = set()
    for row in samples:
        split, label, image_path = row["split"].strip().lower(), row["label"].strip(), row["image_path"].strip()
        if split not in ("train", "val", "test") or label not in ("0", "1"):
            invalid_rows += 1
            continue
        counts[split] += 1
        class_counts[label] = class_counts.get(label, 0) + 1
        resolved = Path(image_path) if Path(image_path).is_absolute() else manifest_path.parent / image_path
        canonical = str(resolved.resolve())
        if canonical in seen_paths and len(duplicate_paths) < 10:
            duplicate_paths.append(image_path)
        seen_paths.add(canonical)
        if not resolved.is_file() and len(missing_files) < 10:
            missing_files.append(image_path)
        elif resolved.is_file():
            decode_error = _decode_error(resolved)
            if decode_error and len(unreadable_files) < 10:
                unreadable_files.append(f"{image_path}: {decode_error}")

    valid = not missing_files and not unreadable_files and not duplicate_paths and invalid_rows == 0 and all(counts[name] > 0 for name in ("train", "val", "test"))
    return {
        "valid": valid, "total_samples": len(samples), "class_distribution": class_counts,
        "split_counts": counts, "missing_files_sample": missing_files,
        "unreadable_files_sample": unreadable_files,
        "duplicate_paths_sample": duplicate_paths, "invalid_row_count": invalid_rows,
        "split_type": "image-level",
    }
