# RunPod Dockerfile for AIN Vision Language Model OCR
# Using RunPod's Python base with CUDA support
# Build trigger: 2025-01-13-v1 (Pre-baked model for faster cold starts)
FROM runpod/base:0.4.0-cuda11.8.0

# Set working directory
WORKDIR /app

# =============================================================================
# ENVIRONMENT VARIABLES FOR HUGGINGFACE CACHE
# =============================================================================
# Set HuggingFace cache directory inside the container
# This ensures the pre-downloaded model is used at runtime
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
# This requires ninja for faster compilation
RUN pip3 install --no-cache-dir ninja packaging || true
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation || echo "Flash Attention 2 not installed - using SDPA fallback"

# =============================================================================
# PRE-DOWNLOAD MODEL DURING BUILD (CRITICAL FOR FAST COLD STARTS)
# =============================================================================
# This downloads the ~7GB AIN model during Docker build, NOT at runtime
# Cold start goes from ~180s (download+load) to ~30-60s (just load)

RUN echo "📥 Pre-downloading AIN model for faster cold starts..." && \
    python3 -c "\
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer; \
from huggingface_hub import snapshot_download; \
import os; \
\
model_id = 'MBZUAI/AIN'; \
cache_dir = '/app/hf_cache'; \
\
print('Downloading model files...'); \
snapshot_download(model_id, cache_dir=cache_dir); \
\
print('Verifying model can be loaded...'); \
model = Qwen2VLForConditionalGeneration.from_pretrained( \
    model_id, \
    cache_dir=cache_dir, \
    trust_remote_code=True, \
    torch_dtype='auto', \
    device_map='cpu' \
); \
del model; \
\
print('Verifying processor can be loaded...'); \
try: \
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True); \
except: \
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True); \
\
print('✅ Model pre-download complete!'); \
" && echo "✅ Model baked into image successfully!"

# Copy handler code
COPY handler.py .

# =============================================================================
# RUNTIME
# =============================================================================
# Set the handler - model loading will now be fast since files are local
CMD ["python3", "-u", "handler.py"]
