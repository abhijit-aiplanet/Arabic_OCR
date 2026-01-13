# RunPod Dockerfile for AIN Vision Language Model OCR
# Using RunPod's Python base with CUDA support
# Build trigger: 2025-01-13-v2 (Pre-baked model for faster cold starts)
FROM runpod/base:0.4.0-cuda11.8.0

# Set working directory
WORKDIR /app

# =============================================================================
# ENVIRONMENT VARIABLES FOR HUGGINGFACE CACHE
# =============================================================================
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/app/hf_cache

# Update pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Uninstall any pre-existing conflicting packages from base image
RUN pip3 uninstall -y torch torchvision transformers || true

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with --force-reinstall to ensure clean versions
RUN pip3 install --no-cache-dir --force-reinstall -r requirements.txt

# Try to install Flash Attention 2 (optional - will use SDPA fallback if fails)
RUN pip3 install --no-cache-dir ninja packaging || true
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation || echo "Flash Attention 2 not installed - using SDPA fallback"

# =============================================================================
# PRE-DOWNLOAD MODEL DURING BUILD (CRITICAL FOR FAST COLD STARTS)
# =============================================================================
# Create download script for cleaner execution
RUN echo '#!/usr/bin/env python3\n\
import os\n\
os.environ["HF_HOME"] = "/app/hf_cache"\n\
os.environ["TRANSFORMERS_CACHE"] = "/app/hf_cache"\n\
\n\
from huggingface_hub import snapshot_download\n\
\n\
model_id = "MBZUAI/AIN"\n\
cache_dir = "/app/hf_cache"\n\
\n\
print("📥 Downloading AIN model files...")\n\
snapshot_download(model_id, cache_dir=cache_dir)\n\
print("✅ Model download complete!")\n\
' > /app/download_model.py

# Run the download script
RUN python3 /app/download_model.py

# Verify model files exist
RUN echo "📊 Verifying model cache:" && \
    ls -la /app/hf_cache/ && \
    du -sh /app/hf_cache/ && \
    echo "✅ Model files verified!"

# Copy handler code
COPY handler.py .

# Cleanup download script
RUN rm -f /app/download_model.py

# =============================================================================
# RUNTIME
# =============================================================================
CMD ["python3", "-u", "handler.py"]
