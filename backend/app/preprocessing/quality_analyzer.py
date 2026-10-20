import numpy as np
import cv2
from typing import List, Tuple
from backend.app.schemas import QualityMetrics

class SPECTQualityAnalyzer:
    """
    Analyzes basic X-ray image quality metrics: resolution, dynamic range, contrast,
    sharpness, and high-intensity pixel distribution.

    The legacy class name is retained for compatibility with the existing inference API.
    """

    @staticmethod
    def analyze(image_np: np.ndarray) -> QualityMetrics:
        """
        Expects image_np as 2D (H, W) or 3D (H, W, C) numpy array.
        Values in range [0, 255] or [0, 1].
        """
        warnings: List[str] = []

        # Convert to single-channel 8-bit grayscale for metric calculations
        if image_np.ndim == 3:
            h, w, c = image_np.shape
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if c == 3 else image_np[:, :, 0]
        else:
            h, w = image_np.shape
            c = 1
            gray = image_np.copy()

        # Normalize to uint8 [0, 255] if float
        if gray.dtype != np.uint8:
            min_v = float(np.min(gray))
            max_v = float(np.max(gray))
            if max_v > min_v:
                gray_u8 = np.clip((gray - min_v) / (max_v - min_v) * 255.0, 0, 255).astype(np.uint8)
            else:
                gray_u8 = np.zeros_like(gray, dtype=np.uint8)
        else:
            gray_u8 = gray

        # 1. Resolution
        if h < 128 or w < 128:
            warnings.append(f"Low image resolution ({w}x{h}). A minimum of 128x128 pixels is recommended.")
        elif h < 224 or w < 224:
            warnings.append(f"Image resolution ({w}x{h}) will be upsampled to 224x224 for ResNet-18.")

        # 2. Dynamic range & Intensity stats
        dyn_min = float(np.min(gray_u8))
        dyn_max = float(np.max(gray_u8))
        dynamic_range = dyn_max - dyn_min
        mean_intensity = float(np.mean(gray_u8))
        std_intensity = float(np.std(gray_u8))

        if dynamic_range < 50:
            warnings.append("Severely restricted dynamic range. X-ray detail may be poorly represented.")

        # 3. Contrast Ratio (Michelson Contrast)
        if (dyn_max + dyn_min) > 0:
            contrast_ratio = float((dyn_max - dyn_min) / (dyn_max + dyn_min))
        else:
            contrast_ratio = 0.0

        if contrast_ratio < 0.35:
            warnings.append("Low contrast ratio detected. Difficulty distinguishing skeletal structures from soft tissue.")

        # 4. Sharpness (Laplacian Variance)
        laplacian = cv2.Laplacian(gray_u8, cv2.CV_64F)
        sharpness_score = float(laplacian.var())

        if sharpness_score < 40.0:
            warnings.append("Low gradient sharpness. Possible motion blur or acquisition blur.")

        # 5. High-intensity area ratio (legacy response-field name retained for API compatibility)
        threshold_hotspot = np.percentile(gray_u8, 85)
        hotspot_pixels = np.sum(gray_u8 >= max(threshold_hotspot, 180))
        total_pixels = float(h * w)
        hotspot_ratio = float(hotspot_pixels / total_pixels) if total_pixels > 0 else 0.0

        # Overall quality flag assignment
        if len(warnings) == 0 and contrast_ratio >= 0.6 and sharpness_score >= 80.0:
            quality_flag = "optimal"
        elif len(warnings) <= 2 and contrast_ratio >= 0.3:
            quality_flag = "acceptable"
        else:
            quality_flag = "suboptimal"

        return QualityMetrics(
            resolution=[h, w],
            channels=c,
            dynamic_range_min=dyn_min,
            dynamic_range_max=dyn_max,
            mean_intensity=round(mean_intensity, 2),
            std_intensity=round(std_intensity, 2),
            contrast_ratio=round(contrast_ratio, 3),
            sharpness_score=round(sharpness_score, 2),
            hotspot_ratio=round(hotspot_ratio, 4),
            quality_flag=quality_flag,
            warnings=warnings,
        )
