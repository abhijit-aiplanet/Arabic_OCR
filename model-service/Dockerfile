# RunPod Dockerfile for AIN Vision Language Model OCR
# Using RunPod's Python base with CUDA support (much faster!)
FROM runpod/base:0.4.0-cuda11.8.0

# Set working directory
WORKDIR /app

# Update pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Uninstall any pre-existing conflicting packages from base image
RUN pip3 uninstall -y torch torchvision transformers || true

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with --force-reinstall to ensure clean versions
RUN pip3 install --no-cache-dir --force-reinstall -r requirements.txt

# Copy handler code
COPY handler.py .

# Set the handler
CMD ["python3", "-u", "handler.py"]

