import time
import uuid
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from fastapi import HTTPException

from backend.app.config import settings
from backend.app.schemas import (
    ScreeningResponse,
    ExplainResponse,
    ModelStatusResponse,
    CheckpointMetadata,
)
from backend.app.preprocessing.image_processor import ImageProcessor
from backend.app.preprocessing.quality_analyzer import SPECTQualityAnalyzer
from backend.ml.model import SPECTResNet18Classifier, load_and_validate_checkpoint
from backend.ml.gradcam import GradCAM

class InferenceEngine:
    """
    Manages model lifecycle, state verification, PyTorch forward inference, and Grad-CAM explainability.
    Enforces that inference is strictly disabled if no trained checkpoint is loaded.
    """

    def __init__(self):
        self.device = torch.device(
            settings.DEVICE if settings.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.model: Optional[SPECTResNet18Classifier] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.checkpoint_loaded: bool = False
        self.error_message: Optional[str] = None
        self.grad_cam: Optional[GradCAM] = None

        self.reload_model()

    def reload_model(self):
        """Attempts to load and validate the checkpoint from settings.CHECKPOINT_PATH."""
        checkpoint_path = Path(settings.CHECKPOINT_PATH)
        if not checkpoint_path.exists():
            self.model = None
            self.grad_cam = None
            self.checkpoint_loaded = False
            self.error_message = (
                f"No trained checkpoint found at '{checkpoint_path}'. "
                "Inference is halted to prevent untrained or fabricated clinical predictions. "
                "Run 'python -m backend.ml.train' or mount a valid trained checkpoint."
            )
            return

        model, metadata, err = load_and_validate_checkpoint(
            checkpoint_path=checkpoint_path,
            num_classes=settings.NUM_CLASSES,
            device=str(self.device),
        )

        if err or model is None:
            self.model = None
            self.grad_cam = None
            self.checkpoint_loaded = False
            self.error_message = err
        else:
            self.model = model
            self.metadata = metadata
            self.checkpoint_loaded = True
            self.error_message = None
            self.grad_cam = GradCAM(self.model)

    def get_status(self) -> ModelStatusResponse:
        meta_obj = None
        if self.metadata:
            meta_obj = CheckpointMetadata(
                epoch=self.metadata.get("epoch"),
                best_val_f1=self.metadata.get("best_val_f1"),
                best_val_loss=self.metadata.get("best_val_loss"),
                training_timestamp=self.metadata.get("training_timestamp"),
                pytorch_version=self.metadata.get("pytorch_version"),
            )

        return ModelStatusResponse(
            model_name="FracAtlas X-ray ResNet-18 Fracture Classifier",
            architecture="ResNet-18 (Medical Fine-Tuned)",
            device=str(self.device),
            checkpoint_loaded=self.checkpoint_loaded,
            checkpoint_path=str(settings.CHECKPOINT_PATH),
            classes=settings.CLASS_NAMES,
            metadata=meta_obj,
            error_message=self.error_message,
            ready_for_inference=self.checkpoint_loaded and self.model is not None,
        )

    def run_screening(self, pil_img: Image.Image) -> ScreeningResponse:
        if not self.checkpoint_loaded or self.model is None:
            raise HTTPException(
                status_code=503,
                detail=self.error_message or "Model checkpoint is not loaded. Cannot perform inference.",
            )

        # 1. Quality Analysis
        quality_analysis = SPECTQualityAnalyzer.analyze(np.array(pil_img))

        # 2. Preprocessing
        input_tensor, _ = ImageProcessor.preprocess_for_inference(pil_img)
        input_tensor = input_tensor.to(self.device)

        # 3. Model Forward Pass
        t0 = time.time()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        elapsed_ms = (time.time() - t0) * 1000.0

        class_idx = int(np.argmax(probs))
        predicted_class = settings.CLASS_NAMES[class_idx]
        confidence = float(probs[class_idx])

        class_probs = {
            settings.CLASS_NAMES[i]: float(probs[i])
            for i in range(len(settings.CLASS_NAMES))
        }

        prediction_id = f"SCR-{uuid.uuid4().hex[:8].upper()}"

        return ScreeningResponse(
            prediction_id=prediction_id,
            predicted_class=predicted_class,
            class_index=class_idx,
            confidence=round(confidence, 4),
            class_probabilities={k: round(v, 4) for k, v in class_probs.items()},
            quality_analysis=quality_analysis,
            inference_time_ms=round(elapsed_ms, 2),
            disclaimer=settings.DISCLAIMER,
            model_version=f"ResNet-18 (epoch {self.metadata.get('epoch', 'N/A') if self.metadata else 'N/A'})",
        )

    def run_explanation(self, pil_img: Image.Image, target_class_idx: Optional[int] = None) -> ExplainResponse:
        if not self.checkpoint_loaded or self.model is None or self.grad_cam is None:
            raise HTTPException(
                status_code=503,
                detail=self.error_message or "Model checkpoint is not loaded. Cannot generate Grad-CAM explanations.",
            )

        # 1. Preprocess
        input_tensor, resized_rgb = ImageProcessor.preprocess_for_inference(pil_img)
        input_tensor = input_tensor.to(self.device)

        # 2. Grad-CAM calculation
        t0 = time.time()
        heatmap, pred_idx, confidence = self.grad_cam.generate_heatmap(
            input_tensor=input_tensor,
            target_class_idx=target_class_idx,
        )
        colored_heatmap, overlay = GradCAM.overlay_heatmap(resized_rgb, heatmap, alpha=0.45)
        focus_regions = GradCAM.extract_focus_regions(heatmap, threshold=0.60)
        elapsed_ms = (time.time() - t0) * 1000.0

        # 3. Base64 encoding for API payload
        orig_b64 = ImageProcessor.np_to_base64_png(resized_rgb)
        heat_b64 = ImageProcessor.np_to_base64_png(colored_heatmap)
        over_b64 = ImageProcessor.np_to_base64_png(overlay)

        prediction_id = f"EXP-{uuid.uuid4().hex[:8].upper()}"
        target_name = settings.CLASS_NAMES[pred_idx]

        return ExplainResponse(
            prediction_id=prediction_id,
            predicted_class=target_name,
            confidence=round(confidence, 4),
            target_layer="layer4[-1] (High-level spatial convolutions)",
            heatmap_base64=heat_b64,
            overlay_base64=over_b64,
            original_processed_base64=orig_b64,
            focus_regions=focus_regions,
            inference_time_ms=round(elapsed_ms, 2),
            disclaimer=settings.DISCLAIMER,
        )

# Global singleton engine instance
engine = InferenceEngine()
