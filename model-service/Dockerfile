# RunPod Dockerfile for AIN Vision Language Model OCR
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies and runpod
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install runpod SDK first
RUN pip install --no-cache-dir runpod

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler code
COPY handler.py .

# Set the handler
CMD ["python", "-u", "handler.py"]

