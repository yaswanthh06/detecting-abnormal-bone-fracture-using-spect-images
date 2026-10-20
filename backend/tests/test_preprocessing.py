import pytest
import numpy as np
from PIL import Image
from backend.app.preprocessing.quality_analyzer import SPECTQualityAnalyzer
from backend.app.preprocessing.image_processor import ImageProcessor

def test_quality_analyzer_contrast_and_sharpness():
    # Create high-contrast test image
    img = np.zeros((224, 224), dtype=np.uint8)
    img[50:150, 50:150] = 255

    metrics = SPECTQualityAnalyzer.analyze(img)
    assert metrics.resolution == [224, 224]
    assert metrics.dynamic_range_max == 255.0
    assert metrics.dynamic_range_min == 0.0
    assert metrics.contrast_ratio > 0.9

def test_quality_analyzer_blurry_image():
    # Flat image with minimal gradients
    flat = np.full((128, 128), 100, dtype=np.uint8)
    metrics = SPECTQualityAnalyzer.analyze(flat)
    assert metrics.sharpness_score == 0.0
    assert any("sharpness" in w.lower() for w in metrics.warnings)

def test_preprocess_for_inference():
    pil = Image.fromarray(np.uint8(np.random.randint(0, 255, (300, 300, 3))))
    tensor, resized_rgb = ImageProcessor.preprocess_for_inference(pil)

    assert tensor.shape == (1, 3, 224, 224)
    assert resized_rgb.shape == (224, 224, 3)

    b64 = ImageProcessor.np_to_base64_png(resized_rgb)
    assert b64.startswith("data:image/png;base64,")
