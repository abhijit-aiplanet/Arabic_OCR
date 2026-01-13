# RunPod Dockerfile for AIN Vision Language Model OCR
# Using RunPod's Python base with CUDA support
# Build trigger: 2025-01-13-v3 (Pre-baked model for faster cold starts)
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

# Copy requirements first (for Docker cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --force-reinstall -r requirements.txt

# Try to install Flash Attention 2 (optional - will use SDPA fallback if fails)
RUN pip3 install --no-cache-dir ninja packaging || true
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation || echo "Flash Attention 2 not installed - using SDPA fallback"

# =============================================================================
# PRE-DOWNLOAD MODEL DURING BUILD (CRITICAL FOR FAST COLD STARTS)
# =============================================================================
# Copy and run the download script
COPY download_model.py .
RUN python3 download_model.py

# =============================================================================
# COPY HANDLER AND CLEANUP
# =============================================================================
COPY handler.py .

# Remove download script (not needed at runtime)
RUN rm -f download_model.py

# =============================================================================
# RUNTIME
# =============================================================================
CMD ["python3", "-u", "handler.py"]
