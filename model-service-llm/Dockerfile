# RunPod Dockerfile for Qwen 2.5 7B Reasoning LLM
# Used for agentic OCR orchestration - analysis, region estimation, merging
# Build trigger: 2026-01-21-v4-7b-model
#
# SETUP: This container expects a Network Volume mounted at /runpod-volume
# Model will be downloaded on first run (~14GB for 7B model)

FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================
# Model will be loaded from Network Volume at /runpod-volume/huggingface
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface
ENV PYTHONUNBUFFERED=1

# =============================================================================
# INSTALL DEPENDENCIES
# =============================================================================

# Update pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Copy and install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# =============================================================================
# COPY HANDLER
# =============================================================================

COPY llm_handler.py .

# Verify Python syntax before container starts
RUN python3 -m py_compile llm_handler.py && echo "Handler syntax OK"

# =============================================================================
# RUNTIME
# =============================================================================

EXPOSE 8000

CMD ["python3", "-u", "llm_handler.py"]
