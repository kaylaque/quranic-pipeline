# ──────────────────────────────────────────────
# Quranic Pipeline — CPU image (faster-whisper int8)
# For CUDA server: change FROM to nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
#   and add: RUN apt-get install -y python3.9 python3-pip && ln -s python3.9 /usr/bin/python
# ──────────────────────────────────────────────
FROM python:3.9-slim

# libsndfile1 — required by soundfile (audio loading, no ffmpeg)
# git — needed by some HuggingFace hub downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies before copying code (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source packages
COPY src/   src/
COPY scripts/ scripts/
COPY eval/  eval/

# Pre-converted CTranslate2 model (74 MB — baked in, zero-download startup)
# The larger wasimlhr model is NOT included; mount via volume if needed
COPY models/whisper-quran-ct2/ models/whisper-quran-ct2/

# data/, results/ and optional large HF models are mounted at runtime (see docker-compose.yml)

ENV PYTHONPATH=/app
# MPS fallback env var — harmless on Linux, required on macOS containers
ENV PYTORCH_ENABLE_MPS_FALLBACK=1
# Point HuggingFace cache to a path we can mount as a named volume
ENV HF_HOME=/app/.cache/huggingface
# Suppress HF tokenizer parallelism warnings in subprocess pipelines
ENV TOKENIZERS_PARALLELISM=false

CMD ["python", "scripts/run_pipeline.py", "--help"]
