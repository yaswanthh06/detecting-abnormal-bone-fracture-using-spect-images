from pathlib import Path

import pandas as pd
from PIL import Image

from backend.ml.prepare_fracatlas import audit_dataset_images, build_manifest_rows


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color="white").save(path)


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_preparation_accepts_valid_images_and_is_reproducible(tmp_path: Path):
    image_dir = tmp_path / "images"
    rows = []
    for label, folder in ((0, "Non_fractured"), (1, "Fractured")):
        for index in range(10):
            name = f"{label}-{index}.jpg"
            _write_image(image_dir / folder / name)
            rows.append({"image_id": name, "fractured": label})
    csv_path = _write_csv(tmp_path, rows)

    valid, excluded = audit_dataset_images(csv_path, image_dir)
    first = build_manifest_rows(csv_path, image_dir, seed=7)
    second = build_manifest_rows(csv_path, image_dir, seed=7)

    assert len(valid) == 20
    assert excluded == []
    assert first == second
    assert {row["split"] for row in first} == {"train", "val", "test"}
    assert {row["label"] for row in first} == {0, 1}


def test_audit_excludes_corrupt_image_with_decode_reason(tmp_path: Path):
    image_dir = tmp_path / "images"
    corrupt = image_dir / "Non_fractured" / "bad.jpg"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"\xff\xd8\xff\xe0truncated jpeg")
    csv_path = _write_csv(tmp_path, [{"image_id": "bad.jpg", "fractured": 0}])

    valid, excluded = audit_dataset_images(csv_path, image_dir)

    assert valid == []
    assert excluded[0]["image_id"] == "bad.jpg"
    assert excluded[0]["label"] == 0
    assert excluded[0]["image_path"] == str(corrupt.resolve())
    assert "Error" in excluded[0]["reason"]


def test_audit_excludes_truncated_jpeg_during_full_decode(tmp_path: Path):
    image_dir = tmp_path / "images"
    truncated = image_dir / "Non_fractured" / "IMG0004255.jpg"
    _write_image(truncated)
    truncated.write_bytes(truncated.read_bytes()[:-20])
    csv_path = _write_csv(tmp_path, [{"image_id": "IMG0004255.jpg", "fractured": 0}])

    valid, excluded = audit_dataset_images(csv_path, image_dir)

    assert valid == []
    assert excluded[0]["image_id"] == "IMG0004255.jpg"
    assert excluded[0]["label"] == 0
    assert "truncated" in excluded[0]["reason"].lower()


def test_audit_excludes_missing_image(tmp_path: Path):
    image_dir = tmp_path / "images"
    csv_path = _write_csv(tmp_path, [{"image_id": "missing.jpg", "fractured": 1}])

    valid, excluded = audit_dataset_images(csv_path, image_dir)

    assert valid == []
    assert excluded == [{
        "image_id": "missing.jpg", "label": 1, "image_path": "",
        "reason": "Expected exactly one image in 'Fractured'; found 0",
    }]
