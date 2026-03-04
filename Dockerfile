# ------------------------------------------------------------------
# Dockerfile for Hugging Face Spaces  (FastAPI + PyTorch CPU backend)
# ------------------------------------------------------------------
FROM python:3.11-slim

# System deps (ffmpeg for yt-dlp merging, libgl for opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Install Python dependencies (CPU-only PyTorch to keep image small)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy project files
COPY configs/ configs/
COPY src/ src/
COPY outputs/ outputs/
COPY predict_api.py .

# HF Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

USER user

CMD ["uvicorn", "predict_api:app", "--host", "0.0.0.0", "--port", "7860"]
