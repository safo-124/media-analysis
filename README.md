---
title: Judo Throw Classifier API
emoji: 🥋
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
---

# Judo Throw Classifier — FastAPI Backend

Dual X3D model API for classifying judo throws from video.

**Models:** X3D-S (88.1% acc) and X3D-M (75.2% acc)  
**Classes:** Ippon Seoi Nage, O Goshi, Osoto Gari, Uchi Mata  

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/compare` | Upload video, compare both models |
| POST | `/compare-url` | URL/YouTube/TikTok, compare both |
| POST | `/predict` | Single-model prediction |
| POST | `/predict-url` | Single-model from URL |
| GET | `/health` | Server status |
| GET | `/classes` | List throw classes |
| GET | `/models` | Model info |
