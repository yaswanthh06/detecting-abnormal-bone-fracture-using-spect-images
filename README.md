# FracAtlas X-ray Bone Fracture Classification

Research-use-only web application for binary fracture classification of FracAtlas X-ray images. The frontend uses React, TypeScript, and Vite; the FastAPI backend serves a trained PyTorch ResNet-18 checkpoint with Grad-CAM visual explanations.

## Run locally

1. Install frontend dependencies: `npm install`
2. Configure the optional frontend API URL in `.env.local` (see `.env.example`).
3. Start the backend from the project root: `python -m uvicorn backend.app.main:app --reload`
4. Start the frontend: `npm run dev`

The default backend checkpoint is `backend/checkpoints/best_resnet18_fracatlas.pt`; do not replace it unless intentionally deploying a compatible trained checkpoint.

## Limitations and intended use

This is an investigational research tool, not a medical device. Predictions and Grad-CAM activations do not provide medical advice, diagnostic conclusions, fracture localization, or treatment recommendations. Performance is limited to the model, FracAtlas data distribution, image quality, and binary class mapping used during training; it may not generalize to other imaging protocols, populations, acquisition devices, or clinical settings. A licensed clinician must make all clinical decisions.
