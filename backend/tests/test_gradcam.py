import pytest
import torch
import numpy as np
from backend.ml.model import build_model
from backend.ml.gradcam import GradCAM

def test_gradcam_computation():
    model = build_model(num_classes=2, pretrained=False)
    grad_cam = GradCAM(model)

    dummy_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
    heatmap, pred_idx, conf = grad_cam.generate_heatmap(dummy_tensor)

    assert heatmap.shape == (224, 224)
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0
    assert pred_idx in [0, 1]
    assert 0.0 <= conf <= 1.0

    grad_cam.remove_hooks()

def test_gradcam_overlay_and_regions():
    dummy_img = np.full((224, 224, 3), 128, dtype=np.uint8)
    dummy_heatmap = np.zeros((224, 224), dtype=np.float32)
    # Simulate a hotspot in the center
    dummy_heatmap[90:130, 90:130] = 0.95

    colored, overlay = GradCAM.overlay_heatmap(dummy_img, dummy_heatmap, alpha=0.5)
    assert colored.shape == (224, 224, 3)
    assert overlay.shape == (224, 224, 3)

    regions = GradCAM.extract_focus_regions(dummy_heatmap, threshold=0.7)
    assert len(regions) >= 1
    assert regions[0].peak_intensity >= 0.9
