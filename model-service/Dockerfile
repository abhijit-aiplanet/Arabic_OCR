# RunPod Dockerfile for AIN Vision Language Model OCR
# Using RunPod's Python base with CUDA support
# Build trigger: 2026-01-08-v3 (Superior anti-hallucination: aggressive params)
#
# SETUP: Attach a Network Volume to store the model cache.
# The model (~30GB) will download on first run and be cached.
# Subsequent runs use the cached model (fast cold starts).
FROM runpod/base:0.4.0-cuda11.8.0

# Set working directory
WORKDIR /app

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================
# Default to network volume path - handler.py will detect and use it
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface
ENV PYTHONUNBUFFERED=1

# Update pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Uninstall any pre-existing conflicting packages from base image
RUN pip3 uninstall -y torch torchvision transformers || true

# Copy requirements first (for Docker cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --force-reinstall -r requirements.txt

# Try to install Flash Attention 2 (optional - will use SDPA fallback if fails)
RUN pip3 install --no-cache-dir ninja packaging || true
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation || echo "Flash Attention 2 not installed - using SDPA fallback"

# =============================================================================
# NOTE: Model is NOT pre-downloaded during build anymore.
# It will be downloaded on first run to the Network Volume and cached there.
# This makes Docker builds fast (~5 min) and cold starts fast after first run.
# =============================================================================

# =============================================================================
# COPY HANDLER
# =============================================================================
COPY handler.py .

# Verify syntax
RUN python3 -m py_compile handler.py && echo "Handler syntax OK"

# =============================================================================
# RUNTIME
# =============================================================================
CMD ["python3", "-u", "handler.py"]
