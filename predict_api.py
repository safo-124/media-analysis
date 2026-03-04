"""
=============================================================================
FastAPI Inference Server — Judo Throws Classification  (dual-model)
=============================================================================

USAGE:
    .venv\\Scripts\\python.exe -m uvicorn predict_api:app --reload --port 8000

ENDPOINTS:
    POST /predict          — Predict with X3D-S (default) or specify model
    POST /compare          — Run BOTH models on the same video and compare
    GET  /health           — Check server is running
    GET  /classes          — List the 4 judo throw classes
    GET  /models           — List loaded models and their info
=============================================================================
"""

import os
import sys
import time
import shutil
import tempfile
import urllib.parse
import numpy as np
import torch
import torch.nn.functional as F
import av
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torchvision import transforms

# Make sure project modules are importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import configs.config as cfg
from configs.config import (
    CLASS_NAMES, NUM_FRAMES, CROP_SIZE, RESIZE_SHORT_SIDE, MEAN, STD, DEVICE
)
from src.model import build_model

# =============================================================================
# App setup
# =============================================================================
app = FastAPI(
    title="Judo Throws Classifier",
    description="Upload a judo video and compare X3D-S vs X3D-M predictions.",
    version="2.0.0",
)

# Allow requests from the Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Model loading — both X3D-S and X3D-M (done once at startup)
# =============================================================================
MODELS: dict = {}  # {"x3d_s": model, "x3d_m": model}

MODEL_META = {
    "x3d_s": {
        "name": "X3D-S",
        "checkpoint": os.path.join(PROJECT_ROOT, "outputs", "x3d_s", "best_model.pth"),
        "test_accuracy": 0.8812,
        "params": "2.98M",
    },
    "x3d_m": {
        "name": "X3D-M",
        "checkpoint": os.path.join(PROJECT_ROOT, "outputs", "x3d_m", "best_model.pth"),
        "test_accuracy": 0.7525,
        "params": "2.98M",
    },
}


def _load_one_model(model_key: str) -> torch.nn.Module:
    """Build model architecture and load checkpoint weights."""
    meta = MODEL_META[model_key]
    path = meta["checkpoint"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Temporarily override config so build_model() picks the right arch
    old_name = cfg.MODEL_NAME
    cfg.MODEL_NAME = model_key
    try:
        m = build_model()
    finally:
        cfg.MODEL_NAME = old_name

    ckpt = torch.load(path, map_location=DEVICE)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m


@app.on_event("startup")
def load_models():
    for key in MODEL_META:
        path = MODEL_META[key]["checkpoint"]
        if os.path.exists(path):
            print(f"Loading {MODEL_META[key]['name']} from {path} ...")
            MODELS[key] = _load_one_model(key)
            print(f"  {MODEL_META[key]['name']} ready on {DEVICE}.")
        else:
            print(f"  [SKIP] {MODEL_META[key]['name']} checkpoint not found.")
    if not MODELS:
        raise RuntimeError("No model checkpoints found — cannot start server.")


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
def _run_inference(model_obj: torch.nn.Module, tensor: torch.Tensor) -> dict:
    """Run a single model on the preprocessed tensor and return result dict."""
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model_obj(tensor)
        probs = F.softmax(logits, dim=1)[0]
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    predicted_idx = probs.argmax().item()
    scores = {cls: round(probs[i].item(), 4) for i, cls in enumerate(CLASS_NAMES)}
    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": round(probs[predicted_idx].item(), 4),
        "scores": scores,
        "inference_ms": elapsed_ms,
    }


def _save_upload(file_bytes: bytes, ext: str) -> str:
    """Write uploaded bytes to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        return tmp.name


def _validate_ext(filename: str) -> str:
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(filename)[-1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}",
        )
    return ext


# =============================================================================
# Routes
# =============================================================================
@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "models_loaded": list(MODELS.keys())}


@app.get("/classes")
def classes():
    return {"classes": CLASS_NAMES}


@app.get("/models")
def models_info():
    return {
        key: {
            "name": MODEL_META[key]["name"],
            "loaded": key in MODELS,
            "test_accuracy": MODEL_META[key]["test_accuracy"],
            "params": MODEL_META[key]["params"],
        }
        for key in MODEL_META
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Query("x3d_s", description="x3d_s or x3d_m"),
):
    """Predict using a single model (default: X3D-S)."""
    if model_name not in MODELS:
        raise HTTPException(404, f"Model '{model_name}' is not loaded.")

    ext = _validate_ext(file.filename)
    raw = await file.read()
    tmp_path = _save_upload(raw, ext)

    try:
        tensor = preprocess(tmp_path)
    except Exception as e:
        raise HTTPException(422, f"Video processing error: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    result = _run_inference(MODELS[model_name], tensor)
    result["model"] = model_name
    return result


@app.post("/compare")
async def compare(file: UploadFile = File(...)):
    """
    Run BOTH models on the same video and return side-by-side results.

    Returns:
        {
            "x3d_s": { predicted_class, confidence, scores, inference_ms },
            "x3d_m": { predicted_class, confidence, scores, inference_ms },
            "agree": true/false
        }
    """
    ext = _validate_ext(file.filename)
    raw = await file.read()
    tmp_path = _save_upload(raw, ext)

    try:
        tensor = preprocess(tmp_path)
    except Exception as e:
        raise HTTPException(422, f"Video processing error: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    results = {}
    for key, m in MODELS.items():
        results[key] = _run_inference(m, tensor)

    agree = len(set(r["predicted_class"] for r in results.values())) == 1

    return {**results, "agree": agree}


# =============================================================================
# URL-based endpoints  (supports YouTube, TikTok, direct links, etc.)
# =============================================================================
import re as _re
import subprocess as _sp

MAX_DOWNLOAD_MB = 100
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Patterns that need yt-dlp (YouTube, TikTok, Shorts, etc.)
_YTDLP_PATTERNS = [
    r"(youtube\.com|youtu\.be)",
    r"tiktok\.com",
    r"instagram\.com\/reel",
    r"facebook\.com\/.*\/videos",
    r"twitter\.com|x\.com",
]


def _needs_ytdlp(url: str) -> bool:
    """Return True if the URL is from a platform that requires yt-dlp."""
    return any(_re.search(p, url, _re.IGNORECASE) for p in _YTDLP_PATTERNS)


class URLRequest(BaseModel):
    url: str
    model_name: str = "x3d_s"


class CompareURLRequest(BaseModel):
    url: str


def _download_with_ytdlp(url: str) -> str:
    """Use yt-dlp to download the best mp4 ≤720p to a temp file.

    Returns the path to the downloaded .mp4 file.
    The file lives inside a temp directory — callers should use
    _cleanup_path() to remove it.
    """
    tmp_dir = tempfile.mkdtemp(prefix="judo_")
    out_template = os.path.join(tmp_dir, "video.%(ext)s")

    # Use a format string that prefers a single pre-merged mp4 stream
    # so yt-dlp doesn't leave separate .m4a / .webm fragments behind.
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--format", "best[height<=720][ext=mp4]/best[ext=mp4]/best[height<=720]/best",
        "--output", out_template,
        "--no-warnings",
        "--quiet",
        url,
    ]

    try:
        result = _sp.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            err_msg = result.stderr.strip() or "Unknown yt-dlp error"
            raise HTTPException(400, f"yt-dlp failed: {err_msg}")
    except _sp.TimeoutExpired:
        raise HTTPException(400, "Video download timed out (120 s limit).")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to run yt-dlp: {e}")

    # Find the downloaded VIDEO file (skip audio-only .m4a, .aac, etc.)
    VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
    all_files = os.listdir(tmp_dir)
    video_files = [
        os.path.join(tmp_dir, f)
        for f in all_files
        if os.path.splitext(f)[-1].lower() in VIDEO_EXTS
    ]

    if not video_files:
        # Fallback: just take any file yt-dlp produced
        video_files = [os.path.join(tmp_dir, f) for f in all_files]

    if not video_files:
        raise HTTPException(400, "yt-dlp produced no output file.")

    path = video_files[0]
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > MAX_DOWNLOAD_MB:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, f"Downloaded video is {size_mb:.0f} MB — exceeds {MAX_DOWNLOAD_MB} MB limit.")

    return path


def _download_direct(url: str) -> str:
    """Direct HTTP download for plain video file URLs."""
    parsed = urllib.parse.urlparse(url)
    path_part = parsed.path.split("?")[0]
    ext = os.path.splitext(path_part)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".mp4"

    try:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(400, f"Failed to download video: HTTP {e.response.status_code}")
    except Exception as e:
        raise HTTPException(400, f"Failed to download video: {e}")

    if len(resp.content) > MAX_DOWNLOAD_MB * 1024 * 1024:
        raise HTTPException(400, f"Video exceeds {MAX_DOWNLOAD_MB} MB limit.")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(resp.content)
        return tmp.name


def _download_video(url: str) -> str:
    """Download a video — auto-detects YouTube/TikTok vs direct link."""
    if _needs_ytdlp(url):
        return _download_with_ytdlp(url)
    return _download_direct(url)


def _cleanup_path(path: str) -> None:
    """Remove a temp file or its parent temp directory (safe on Windows)."""
    try:
        parent = os.path.dirname(path)
        # If parent is a judo_* temp dir, remove the whole thing
        if os.path.basename(parent).startswith("judo_"):
            shutil.rmtree(parent, ignore_errors=True)
        elif os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass  # best-effort cleanup


@app.post("/predict-url")
async def predict_url(body: URLRequest):
    """Predict using a single model from a video URL."""
    if body.model_name not in MODELS:
        raise HTTPException(404, f"Model '{body.model_name}' is not loaded.")

    tmp_path = _download_video(body.url)
    try:
        tensor = preprocess(tmp_path)
    except Exception as e:
        raise HTTPException(422, f"Video processing error: {e}")
    finally:
        _cleanup_path(tmp_path)

    result = _run_inference(MODELS[body.model_name], tensor)
    result["model"] = body.model_name
    return result


@app.post("/compare-url")
async def compare_url(body: CompareURLRequest):
    """Run BOTH models on a video URL and return side-by-side results."""
    tmp_path = _download_video(body.url)
    try:
        tensor = preprocess(tmp_path)
    except Exception as e:
        raise HTTPException(422, f"Video processing error: {e}")
    finally:
        _cleanup_path(tmp_path)

    results = {}
    for key, m in MODELS.items():
        results[key] = _run_inference(m, tensor)

    agree = len(set(r["predicted_class"] for r in results.values())) == 1
    return {**results, "agree": agree}
