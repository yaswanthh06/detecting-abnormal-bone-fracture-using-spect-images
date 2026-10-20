import os
from pathlib import Path
from typing import List

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"

class Settings:
    # Application Info
    PROJECT_NAME: str = "FracAtlas X-ray Bone Fracture Classification"
    VERSION: str = "1.0.0"
    API_PREFIX: str = ""

    # Server Configuration
    HOST: str = os.getenv("BACKEND_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("BACKEND_PORT", "8000"))
    CORS_ORIGINS: List[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173",
        ).split(",")
        if origin.strip()
    ]

    # File Upload Limits & Formats
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "25"))
    MAX_UPLOAD_SIZE_BYTES: int = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    ALLOWED_EXTENSIONS: set = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}

    # ML Pipeline & Model Settings
    CHECKPOINT_PATH: Path = Path(os.getenv("CHECKPOINT_PATH", str(BACKEND_DIR / "checkpoints" / "best_resnet18_fracatlas.pt")))
    DATASET_MANIFEST_PATH: Path = Path(os.getenv("DATASET_MANIFEST_PATH", str(BACKEND_DIR / "data" / "manifest.csv")))
    EVALUATION_SUMMARY_PATH: Path = Path(os.getenv("EVALUATION_SUMMARY_PATH", str(BACKEND_DIR / "data" / "evaluation_summary.json")))

    MODEL_ARCH: str = "resnet18"
    NUM_CLASSES: int = 2
    CLASS_NAMES: List[str] = ["Non-fractured", "Fractured"]
    IMG_SIZE: int = 224

    # Device selection (CPU by default unless CUDA is explicitly configured & available)
    DEVICE: str = os.getenv("ML_DEVICE", "cpu")

    # Research Disclaimer
    DISCLAIMER: str = (
        "RESEARCH USE ONLY: This system is an investigational deep learning tool for FracAtlas X-ray image research. "
        "Outputs and Grad-CAM activations do not constitute medical advice, diagnostic conclusions, or treatment plans. "
        "All clinical decisions must be rendered by a licensed radiologist or other qualified clinician."
    )

settings = Settings()
