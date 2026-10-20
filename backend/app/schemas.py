from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    timestamp: str
    version: str
    service: str

class CheckpointMetadata(BaseModel):
    epoch: Optional[int] = None
    best_val_f1: Optional[float] = None
    best_val_loss: Optional[float] = None
    training_timestamp: Optional[str] = None
    pytorch_version: Optional[str] = None
    weights_sha256: Optional[str] = None

class ModelStatusResponse(BaseModel):
    model_name: str
    architecture: str
    device: str
    checkpoint_loaded: bool
    checkpoint_path: str
    classes: List[str]
    metadata: Optional[CheckpointMetadata] = None
    error_message: Optional[str] = None
    ready_for_inference: bool

class QualityMetrics(BaseModel):
    resolution: List[int] = Field(..., description="[height, width]")
    channels: int
    dynamic_range_min: float
    dynamic_range_max: float
    mean_intensity: float
    std_intensity: float
    contrast_ratio: float
    sharpness_score: float
    hotspot_ratio: float = Field(..., description="Ratio of high-intensity image pixels (legacy field name retained)")
    quality_flag: str = Field(..., description="'optimal', 'acceptable', or 'suboptimal'")
    warnings: List[str] = Field(default_factory=list)

class ScreeningResponse(BaseModel):
    prediction_id: str
    predicted_class: str
    class_index: int
    confidence: float
    class_probabilities: Dict[str, float]
    quality_analysis: QualityMetrics
    inference_time_ms: float
    disclaimer: str
    model_version: str

class FocusRegion(BaseModel):
    region_id: int
    center_normalized: List[float] = Field(..., description="[y_norm, x_norm]")
    peak_intensity: float
    relative_area_pct: float
    intensity_category: str = Field(..., description="Grad-CAM activation category")

class ExplainResponse(BaseModel):
    prediction_id: str
    predicted_class: str
    confidence: float
    target_layer: str
    heatmap_base64: str
    overlay_base64: str
    original_processed_base64: str
    focus_regions: List[FocusRegion]
    inference_time_ms: float
    disclaimer: str

class ConfusionMatrixData(BaseModel):
    matrix: List[List[int]]
    labels: List[str]
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

class EvaluationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall_sensitivity: float
    specificity: float
    f1_score: float
    roc_auc: Optional[float] = None
    test_sample_count: int

class SplitCounts(BaseModel):
    train_images: int
    val_images: int
    test_images: int
    total_images: int

class EvaluationSummaryResponse(BaseModel):
    has_evaluation_data: bool
    evaluation_timestamp: Optional[str] = None
    split_counts: Optional[SplitCounts] = None
    metrics: Optional[EvaluationMetrics] = None
    confusion_matrix: Optional[ConfusionMatrixData] = None
    manifest_path: str
    message: str
