import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, List, Optional
from backend.app.schemas import FocusRegion

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for ResNet-18 bone fracture localization.
    Visualizes image regions that contribute to a selected fracture-classification score.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        if target_layer is None:
            # Default to last convolutional block in layer4 of ResNet-18
            if hasattr(model, "get_gradcam_target_layer"):
                self.target_layer = model.get_gradcam_target_layer()
            elif hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
                self.target_layer = model.backbone.layer4[-1]
            else:
                raise ValueError("Target layer could not be determined automatically.")
        else:
            self.target_layer = target_layer

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float]:
        """
        Runs forward + backward pass and computes the 2D Grad-CAM heatmap normalized to [0, 1].
        input_tensor: (1, 3, H, W)
        Returns (heatmap: (H, W) float32 in [0, 1], predicted_class: int, confidence: float)
        """
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class_idx]
        confidence = float(probs[0, target_class_idx].item())

        # Backward pass for targeted class score
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture feature maps or gradients for Grad-CAM.")

        # Global average pooling of gradients: weights alpha_k = (1/Z) * sum(grad_k)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, h, w)

        # Apply ReLU to retain features that positively correlate with the target class
        cam = F.relu(cam)

        # Upsample heatmap to match input tensor dimensions
        _, _, h, w = input_tensor.shape
        cam_upsampled = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        cam_np = cam_upsampled.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        max_val = np.max(cam_np)
        min_val = np.min(cam_np)
        if max_val > min_val:
            heatmap = (cam_np - min_val) / (max_val - min_val)
        else:
            heatmap = np.zeros_like(cam_np)

        return heatmap, target_class_idx, confidence

    @staticmethod
    def overlay_heatmap(
        original_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a colorized heatmap and blended overlay with the original X-ray image.
        original_rgb: (H, W, 3) uint8
        heatmap: (H, W) float32 [0, 1]
        Returns (colored_heatmap_rgb: (H, W, 3), overlay_rgb: (H, W, 3))
        """
        h, w = original_rgb.shape[:2]

        # Resize heatmap if not matching original resolution
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        heatmap_u8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(heatmap_u8, colormap)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        # Alpha blend overlay
        overlay = cv2.addWeighted(original_rgb, 1.0 - alpha, colored_rgb, alpha, 0)
        return colored_rgb, overlay

    @staticmethod
    def extract_focus_regions(heatmap: np.ndarray, threshold: float = 0.65) -> List[FocusRegion]:
        """
        Identifies connected components / hotspots of high Grad-CAM activation.
        """
        h, w = heatmap.shape[:2]
        binary = (heatmap >= threshold).astype(np.uint8) * 255

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        total_pixels = float(h * w)

        regions: List[FocusRegion] = []
        for i in range(1, num_labels):  # Skip background component 0
            area = stats[i, cv2.CC_STAT_AREA]
            area_pct = (area / total_pixels) * 100.0

            cx, cy = centroids[i]
            # Region mask
            mask = (labels == i)
            peak_val = float(np.max(heatmap[mask]))

            if peak_val >= 0.85:
                cat = "High activation"
            elif peak_val >= 0.70:
                cat = "Moderate activation"
            else:
                cat = "Low activation"

            regions.append(FocusRegion(
                region_id=i,
                center_normalized=[round(float(cy / h), 3), round(float(cx / w), 3)],
                peak_intensity=round(peak_val, 3),
                relative_area_pct=round(area_pct, 2),
                intensity_category=cat,
            ))

        # Sort descending by peak intensity
        regions.sort(key=lambda r: r.peak_intensity, reverse=True)
        return regions[:5]

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
