"""
=============================================================================
FastAPI Inference Server — Judo Throws Classification
=============================================================================

USAGE:
    .\.venv\Scripts\python.exe -m uvicorn predict_api:app --reload --port 8000

ENDPOINTS:
    POST /predict   — Upload a video file, get back the predicted throw
    GET  /health    — Check server is running
    GET  /classes   — List the 4 judo throw classes
=============================================================================
"""

import os
import sys
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import av
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

# Make sure project modules are importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from configs.config import (
    CLASS_NAMES, NUM_FRAMES, CROP_SIZE, RESIZE_SHORT_SIDE, MEAN, STD, DEVICE
)
from src.model import build_model

# =============================================================================
# App setup
# =============================================================================
app = FastAPI(
    title="Judo Throws Classifier",
    description="Upload a judo video and get the predicted throw technique.",
    version="1.0.0",
)

# Allow requests from the Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Model loading (done once at startup)
# =============================================================================
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "outputs", "best_model.pth")

model = None  # loaded in startup event


@app.on_event("startup")
def load_model():
    global model
    print(f"Loading model from {CHECKPOINT_PATH} ...")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model ready on {DEVICE}.")


# =============================================================================
# Video preprocessing (mirrors src/dataset.py)
# =============================================================================
def decode_video(video_path: str, num_frames: int = NUM_FRAMES) -> np.ndarray:
    """Decode a video and uniformly sample `num_frames` frames."""
    container = av.open(video_path)
    all_frames = []
    for frame in container.decode(video=0):
        all_frames.append(frame.to_ndarray(format="rgb24"))
    container.close()

    total = len(all_frames)
    if total == 0:
        raise ValueError("Video has no frames.")

    if total >= num_frames:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    else:
        indices = list(range(total)) + [total - 1] * (num_frames - total)
        indices = np.array(indices)

    return np.array([all_frames[i] for i in indices])  # (T, H, W, C)


# Same normalisation as training (centre-crop, no flip for inference)
_inference_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(RESIZE_SHORT_SIDE),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def preprocess(video_path: str) -> torch.Tensor:
    """Return a (1, 3, T, H, W) tensor ready for the model."""
    frames = decode_video(video_path)           # (T, H, W, C) uint8
    tensors = [_inference_transform(f) for f in frames]  # list of (3, H, W)
    clip = torch.stack(tensors, dim=1)          # (3, T, H, W)
    return clip.unsqueeze(0).to(DEVICE)         # (1, 3, T, H, W)


# =============================================================================
# Routes
# =============================================================================
@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.get("/classes")
def classes():
    return {"classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a video file (.mp4, .avi, .mov) and receive a prediction.

    Returns:
        {
            "predicted_class": "osoto_gari",
            "confidence": 0.97,
            "scores": {
                "ippon_seoi_nage": 0.01,
                "o_goshi": 0.01,
                "osoto_gari": 0.97,
                "uchi_mata": 0.01
            }
        }
    """
    # Accept common video formats
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}"
        )

    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        tensor = preprocess(tmp_path)
    except Exception as e:
        os.unlink(tmp_path)
        raise HTTPException(status_code=422, detail=f"Video processing error: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    with torch.no_grad():
        logits = model(tensor)                          # (1, 4)
        probs = F.softmax(logits, dim=1)[0]             # (4,)

    predicted_idx = probs.argmax().item()
    scores = {cls: round(probs[i].item(), 4) for i, cls in enumerate(CLASS_NAMES)}

    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": round(probs[predicted_idx].item(), 4),
        "scores": scores,
    }
