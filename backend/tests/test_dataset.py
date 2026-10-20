import csv
from pathlib import Path

from PIL import Image

from backend.ml.dataset import FracAtlasDataset, validate_manifest


def test_fracatlas_manifest_schema_and_loader(tmp_path: Path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8)).save(image_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(f"image_path,label,split\n{image_path},1,train\n", encoding="utf-8")
    dataset = FracAtlasDataset(manifest, split="train")
    tensor, label, path = dataset[0]
    assert len(dataset) == 1
    assert tensor.shape == (3, 8, 8)
    assert label == 1
    assert path == str(image_path)


def test_validate_manifest_reports_image_level_contract(tmp_path: Path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (8, 8)).save(image_path)
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "label", "split"])
        writer.writeheader()
        writer.writerows([
            {"image_path": image_path, "label": 0, "split": "train"},
            {"image_path": image_path, "label": 1, "split": "val"},
            {"image_path": image_path, "label": 1, "split": "test"},
        ])
    report = validate_manifest(manifest)
    assert report["split_type"] == "image-level"
    assert not report["valid"]
    assert report["duplicate_paths_sample"]


def test_validate_manifest_rejects_truncated_jpeg(tmp_path: Path):
    image_path = tmp_path / "truncated.jpg"
    Image.new("RGB", (16, 16), color="white").save(image_path)
    image_path.write_bytes(image_path.read_bytes()[:-20])
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        f"image_path,label,split\n{image_path},0,train\n{image_path},1,val\n{image_path},1,test\n",
        encoding="utf-8",
    )

    report = validate_manifest(manifest)

    assert not report["valid"]
    assert report["unreadable_files_sample"]
    assert "truncated" in report["unreadable_files_sample"][0].lower()
