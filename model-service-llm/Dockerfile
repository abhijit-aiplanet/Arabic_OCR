# RunPod Dockerfile for Qwen 2.5 Reasoning LLM
# Used for agentic OCR orchestration - analysis, region estimation, merging
# Build trigger: 2025-01-21-v3-network-volume
#
# SETUP: This container expects a Network Volume mounted at /runpod-volume
# with the model pre-downloaded. See README.md for setup instructions.

FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================
# Model will be loaded from Network Volume at /runpod-volume/huggingface
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface

# Update pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# =============================================================================
# INSTALL DEPENDENCIES
# =============================================================================

# Copy requirements first (for Docker cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# =============================================================================
# COPY HANDLER AND DOWNLOAD SCRIPT
# =============================================================================

COPY llm_handler.py .
COPY download_model.py .

# NOTE: Model is NOT downloaded during build.
# It will be downloaded to Network Volume on first run, or you can
# pre-download it using a GPU Pod. See README.md for instructions.

# =============================================================================
# RUNTIME
# =============================================================================

EXPOSE 8000

CMD ["python3", "-u", "llm_handler.py"]
