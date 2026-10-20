"""Build a label-aware FracAtlas manifest with stratified image-level splits."""
import argparse
import csv
import random
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data" / "FracAtlas"
CSV_PATH, IMAGE_DIR = DATA_DIR / "dataset.csv", DATA_DIR / "images"
OUTPUT_PATH = BACKEND_DIR / "data" / "manifest.csv"
LABEL_FOLDERS = {0: "Non_fractured", 1: "Fractured"}
MANIFEST_FIELDS = ("image_path", "label", "split")


def _validate_image(path: Path) -> str | None:
    """Return an error reason unless Pillow can verify and decode an image."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def audit_dataset_images(dataset_csv: Path, image_dir: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Resolve and decode every CSV image without modifying source files.

    The source CSV has no patient ID; any later splits are image-level only.
    """
    df = pd.read_csv(dataset_csv)
    required = {"image_id", "fractured"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")
    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs found in dataset.csv")
    if not df["fractured"].isin([0, 1]).all():
        raise ValueError("The fractured column must contain only 0 and 1")

    by_name = defaultdict(list)
    for path in image_dir.rglob("*"):
        if path.is_file():
            by_name[path.name].append(path)

    valid_rows: List[Dict[str, object]] = []
    excluded_rows: List[Dict[str, object]] = []
    for record in df.itertuples(index=False):
        name, label = str(record.image_id), int(record.fractured)
        folder = LABEL_FOLDERS[label]
        matches = [path for path in by_name[name] if path.parent.name == folder]
        if len(matches) != 1:
            reason = f"Expected exactly one image in '{folder}'; found {len(matches)}"
            excluded_rows.append({"image_id": name, "label": label, "image_path": "", "reason": reason})
            continue
        image_path = matches[0].resolve()
        error = _validate_image(image_path)
        if error:
            excluded_rows.append({"image_id": name, "label": label, "image_path": str(image_path), "reason": error})
            continue
        valid_rows.append({"image_id": name, "image_path": str(image_path), "label": label})
    return valid_rows, excluded_rows


def assign_stratified_splits(rows: List[Dict[str, object]], seed: int) -> List[Dict[str, object]]:
    """Assign deterministic 70/15/15 splits independently within each class."""
    rng = random.Random(seed)
    for label in (0, 1):
        indices = [idx for idx, row in enumerate(rows) if row["label"] == label]
        rng.shuffle(indices)
        test_count, val_count = round(len(indices) * 0.15), round(len(indices) * 0.15)
        for position, idx in enumerate(indices):
            rows[idx]["split"] = "test" if position < test_count else "val" if position < test_count + val_count else "train"
    return rows


def build_manifest_rows(dataset_csv: Path, image_dir: Path, seed: int = 42) -> List[Dict[str, object]]:
    """Build reproducible stratified image-level splits after the full image audit."""
    audited_rows, _ = audit_dataset_images(dataset_csv, image_dir)
    rows = [{"image_path": row["image_path"], "label": row["label"]} for row in audited_rows]
    return assign_stratified_splits(rows, seed)


def write_manifest(rows: Iterable[Dict[str, object]], output_path: Path) -> None:
    """Atomically replace a manifest only after its complete temporary file is written."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.stem}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, output_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a FracAtlas image-level manifest.")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--images", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    audited_rows, excluded_rows = audit_dataset_images(args.csv, args.images)
    rows = [{"image_path": row["image_path"], "label": row["label"]} for row in audited_rows]
    rows = assign_stratified_splits(rows, args.seed)
    decode_errors = [(row["image_path"], _validate_image(Path(str(row["image_path"])))) for row in rows]
    decode_errors = [(path, error) for path, error in decode_errors if error]
    if decode_errors:
        details = "; ".join(f"{path}: {error}" for path, error in decode_errors[:10])
        raise RuntimeError(f"Manifest was not written because final decode validation failed: {details}")
    write_manifest(rows, args.output)
    print(f"Manifest created: {args.output}")
    print(f"Valid images: {len(rows)}")
    print(f"Excluded images: {len(excluded_rows)}")
    for excluded in excluded_rows:
        print(
            "EXCLUDED "
            f"image_id={excluded['image_id']} label={excluded['label']} "
            f"path={excluded['image_path'] or '<not found>'} reason={excluded['reason']}"
        )
    print(pd.DataFrame(rows).groupby(["split", "label"]).size().to_string())
    print("Note: splits are stratified image-level splits, not patient-level splits.")


if __name__ == "__main__":
    main()
