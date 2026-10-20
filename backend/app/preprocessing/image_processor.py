import io
import base64
from typing import Tuple
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms
from fastapi import HTTPException, UploadFile
from backend.app.config import settings

# Standard PyTorch normalization for ResNet-18
IMAGE_TRANSFORMS = transforms.Compose([
    transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImageProcessor:
    @staticmethod
    async def validate_and_read_upload(file: UploadFile) -> bytes:
        """
        Validates uploaded file size and extension without persisting to disk.
        Returns the raw image bytes in memory.
        """
        # Validate filename and extension
        filename = file.filename or ""
        ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format '{ext}'. Allowed formats: {', '.join(settings.ALLOWED_EXTENSIONS)}",
            )

        # Read into memory safely
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        if len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB.",
            )

        return contents

    @staticmethod
    def bytes_to_pil(image_bytes: bytes) -> Image.Image:
        """
        Safely opens image bytes into a PIL Image and converts to RGB mode.
        """
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.verify()
            # Re-open because verify() closes the file pointer
            pil_img = Image.open(io.BytesIO(image_bytes))
            # Convert to RGB to ensure 3-channel input for ResNet
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            return pil_img
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Corrupt or invalid image file: {str(e)}",
            )

    @staticmethod
    def preprocess_for_inference(pil_img: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocesses a PIL Image identically to the training pipeline:
        - Resizes to (224, 224)
        - Normalizes using standard ImageNet mean and std
        Returns (batch_tensor: (1, 3, 224, 224), resized_np_rgb: (224, 224, 3))
        """
        # Resized numpy image for Grad-CAM overlay and visualization
        resized_pil = pil_img.resize((settings.IMG_SIZE, settings.IMG_SIZE), Image.Resampling.BILINEAR)
        resized_np_rgb = np.array(resized_pil, dtype=np.uint8)

        # Transform to normalized tensor
        tensor = IMAGE_TRANSFORMS(resized_pil)
        batch_tensor = tensor.unsqueeze(0)  # Shape (1, 3, 224, 224)
        return batch_tensor, resized_np_rgb

    @staticmethod
    def np_to_base64_png(np_img: np.ndarray) -> str:
        """
        Converts uint8 numpy array (H, W, 3) RGB or BGR to base64-encoded PNG.
        """
        if np_img.dtype != np.uint8:
            np_img = np.clip(np_img * 255.0, 0, 255).astype(np.uint8)

        # Convert RGB to BGR for cv2 imencode
        if np_img.ndim == 3 and np_img.shape[2] == 3:
            bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        else:
            bgr = np_img

        success, buffer = cv2.imencode(".png", bgr)
        if not success:
            raise RuntimeError("Failed to encode processed image to PNG.")
        b64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"
