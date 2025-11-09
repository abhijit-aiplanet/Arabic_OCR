# RunPod Dockerfile for AIN Vision Language Model OCR
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler code
COPY handler.py .

# Set the handler
CMD ["python", "-u", "handler.py"]

