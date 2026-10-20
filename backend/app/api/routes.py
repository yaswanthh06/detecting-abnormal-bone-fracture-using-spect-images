import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Query, HTTPException

from backend.app.config import settings
from backend.app.schemas import (
    HealthResponse,
    ModelStatusResponse,
    ScreeningResponse,
    ExplainResponse,
    EvaluationSummaryResponse,
)
from backend.app.preprocessing.image_processor import ImageProcessor
from backend.ml.inference import engine
from backend.ml.dataset import validate_manifest

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns application health and deployment readiness."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version=settings.VERSION,
        service="FracAtlas X-ray Bone Fracture Classification API",
    )

@router.get("/model/status", response_model=ModelStatusResponse, tags=["Model"])
def get_model_status():
    """Returns current model status, checkpoint presence, and hardware device."""
    return engine.get_status()

@router.post("/screen", response_model=ScreeningResponse, tags=["Inference"])
async def screen_xray_image(
    file: UploadFile = File(..., description="FracAtlas X-ray image file (PNG, JPG, TIFF)"),
):
    """
    Validates an uploaded X-ray image, performs image-quality checks, and returns
    FracAtlas fracture-classification scores. The image is processed in memory only.
    """
    image_bytes = await ImageProcessor.validate_and_read_upload(file)
    pil_image = ImageProcessor.bytes_to_pil(image_bytes)

    # Screening execution (checks checkpoint availability inside engine)
    result = engine.run_screening(pil_image)
    return result

@router.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain_xray_image(
    file: UploadFile = File(..., description="FracAtlas X-ray image file"),
    target_class_idx: Optional[int] = Query(None, description="Target class index (0=Non-fractured, 1=Fractured)"),
):
    """
    Generates Grad-CAM heatmaps and blended overlays of image regions that
    influenced the X-ray classification output; they are not diagnostic localizations.
    """
    image_bytes = await ImageProcessor.validate_and_read_upload(file)
    pil_image = ImageProcessor.bytes_to_pil(image_bytes)

    result = engine.run_explanation(pil_image, target_class_idx=target_class_idx)
    return result

@router.get("/evaluation/summary", response_model=EvaluationSummaryResponse, tags=["Evaluation"])
def get_evaluation_summary():
    """
    Returns independent test evaluation metrics and confusion matrix
    calculated on held-out patients from the manifest.
    """
    eval_path = Path(settings.EVALUATION_SUMMARY_PATH)
    manifest_path = Path(settings.DATASET_MANIFEST_PATH)

    # 1. If evaluation summary JSON file exists on disk
    if eval_path.exists():
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return EvaluationSummaryResponse(**data)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read evaluation summary file: {str(e)}",
            )

    # 2. If manifest exists but evaluate.py hasn't been run yet
    if manifest_path.exists():
        report = validate_manifest(manifest_path)
        split_counts = None
        if report.get("valid"):
            split_counts = {
                "train_images": report["split_counts"].get("train", 0),
                "val_images": report["split_counts"].get("val", 0),
                "test_images": report["split_counts"].get("test", 0),
                "total_images": report["total_samples"],
            }

        return EvaluationSummaryResponse(
            has_evaluation_data=False,
            split_counts=split_counts,
            manifest_path=str(manifest_path),
            message=(
                "Dataset manifest found, but test evaluation has not been executed. "
                "Run 'python -m backend.ml.evaluate' with a trained checkpoint to generate verifiable metrics."
            ),
        )

    # 3. Neither evaluation file nor manifest is mounted
    return EvaluationSummaryResponse(
        has_evaluation_data=False,
        manifest_path=str(manifest_path),
        message=(
            "No evaluation data or dataset manifest found. "
            "Mount a valid CSV manifest at data/manifest.csv and run evaluation to populate metrics."
        ),
    )
