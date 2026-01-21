# RunPod Dockerfile for Qwen 2.5 72B Reasoning LLM
# Used for agentic OCR orchestration - analysis, region estimation, merging
# Build trigger: 2025-01-08-v1

FROM runpod/pytorch:2.1.1-py3.10-cuda11.8.0

WORKDIR /app

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV HUGGINGFACE_HUB_CACHE=/app/hf_cache
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

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
# PRE-DOWNLOAD MODEL DURING BUILD (CRITICAL FOR FAST COLD STARTS)
# =============================================================================

# Copy and run the download script
COPY download_model.py .
RUN python3 download_model.py

# =============================================================================
# COPY HANDLER
# =============================================================================

COPY llm_handler.py .

# Remove download script (not needed at runtime)
RUN rm -f download_model.py

# =============================================================================
# RUNTIME
# =============================================================================

# Expose port for health checks (optional)
EXPOSE 8000

CMD ["python3", "-u", "llm_handler.py"]
