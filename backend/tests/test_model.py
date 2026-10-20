import pytest
import torch
from torch.torch_version import TorchVersion
from pathlib import Path
from backend.ml.model import build_model, load_and_validate_checkpoint

def test_model_build():
    model = build_model(num_classes=2, pretrained=False)
    assert model is not None
    assert model.num_classes == 2

    # Verify forward pass shape
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 2)

def test_model_gradcam_target():
    model = build_model(num_classes=2, pretrained=False)
    target_layer = model.get_gradcam_target_layer()
    assert target_layer is not None

def test_checkpoint_nonexistent():
    fake_path = Path("/tmp/nonexistent_model_checkpoint.pt")
    model, meta, err = load_and_validate_checkpoint(fake_path, num_classes=2)
    assert model is None
    assert "not found" in err.lower()


def test_loads_legacy_checkpoint_with_torch_version_metadata(tmp_path):
    """PyTorch 2.6 must safely load locally generated legacy checkpoints."""
    checkpoint_path = tmp_path / "legacy_checkpoint.pt"
    source_model = build_model(num_classes=2, pretrained=False)
    torch.save(
        {
            "state_dict": source_model.state_dict(),
            "epoch": 5,
            # This mirrors the previous train.py behavior.
            "pytorch_version": TorchVersion(str(torch.__version__)),
        },
        checkpoint_path,
    )

    model, metadata, err = load_and_validate_checkpoint(checkpoint_path, num_classes=2)

    assert err is None
    assert model is not None
    assert metadata["epoch"] == 5
    assert metadata["pytorch_version"] == str(torch.__version__)
