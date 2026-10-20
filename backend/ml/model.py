import torch
import torch.nn as nn
from torch.torch_version import TorchVersion
from torchvision.models import resnet18, ResNet18_Weights
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

class SPECTResNet18Classifier(nn.Module):
    """
    Modular PyTorch ResNet-18 classifier for FracAtlas X-ray fracture classification.
    The legacy class name is retained to preserve checkpoint-loading compatibility.
    Includes dropout regularization and designated hook targets for Grad-CAM.
    """

    def __init__(self, num_classes: int = 2, pretrained_backbone: bool = False, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # Load backbone
        if pretrained_backbone:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.backbone = resnet18(weights=weights)

        # Replace classification head with medical dropout + linear projection
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: shape (B, 3, 224, 224)
        Returns logits: shape (B, num_classes)
        """
        return self.backbone(x)

    def get_gradcam_target_layer(self) -> nn.Module:
        """
        Returns the last convolutional layer in the ResNet-18 architecture (layer4[1].conv2),
        which captures high-level spatial features used by the X-ray classifier.
        """
        return self.backbone.layer4[-1]


def build_model(num_classes: int = 2, pretrained: bool = False) -> SPECTResNet18Classifier:
    return SPECTResNet18Classifier(num_classes=num_classes, pretrained_backbone=pretrained)


def load_and_validate_checkpoint(
    checkpoint_path: Path,
    num_classes: int = 2,
    device: str = "cpu"
) -> Tuple[Optional[SPECTResNet18Classifier], Optional[Dict[str, Any]], Optional[str]]:
    """
    Loads and validates weights from a checkpoint file.
    Detects missing files, corrupt files, or incompatible tensor dimensions.
    Returns (model, metadata_dict, error_string).
    """
    if not checkpoint_path.exists():
        return None, None, f"Checkpoint not found at '{checkpoint_path}'."

    try:
        # Checkpoints contain tensors plus simple metadata.  Older local
        # checkpoints saved ``torch.__version__`` as a TorchVersion instance;
        # permit only that known metadata type while retaining PyTorch's safe
        # weights-only unpickler.  Do not change this to weights_only=False for
        # checkpoints from outside this project, since pickle can execute code.
        with torch.serialization.safe_globals([TorchVersion]):
            loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = build_model(num_classes=num_classes, pretrained=False)

        metadata: Dict[str, Any] = {}
        if isinstance(loaded, dict) and "state_dict" in loaded:
            state_dict = loaded["state_dict"]
            metadata = {
                "epoch": loaded.get("epoch"),
                "best_val_f1": loaded.get("best_val_f1"),
                "best_val_loss": loaded.get("best_val_loss"),
                "training_timestamp": loaded.get("training_timestamp"),
                "pytorch_version": str(loaded.get("pytorch_version", torch.__version__)),
            }
        elif isinstance(loaded, dict):
            state_dict = loaded
        else:
            return None, None, "Invalid checkpoint format: expected state dict or mapping."

        # Filter and validate keys
        model_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())

        # Support state dicts saved with 'module.' prefix (from DataParallel)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            clean_k = k.replace("module.", "")
            cleaned_state_dict[clean_k] = v

        # Check dimension compatibility of the final classification layer
        fc_weight_key = "backbone.fc.1.weight"
        if fc_weight_key in cleaned_state_dict:
            weight_shape = cleaned_state_dict[fc_weight_key].shape
            if weight_shape[0] != num_classes:
                return (
                    None,
                    None,
                    f"Incompatible checkpoint: classifier head has {weight_shape[0]} classes, expected {num_classes}.",
                )

        model.load_state_dict(cleaned_state_dict, strict=True)
        model.to(device)
        model.eval()
        return model, metadata, None

    except Exception as e:
        return None, None, f"Failed to load checkpoint: {str(e)}"
