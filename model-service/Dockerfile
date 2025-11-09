# RunPod Dockerfile for AIN Vision Language Model OCR
# Using RunPod's Python base with CUDA support (much faster!)
FROM runpod/base:0.4.0-cuda11.8.0

# Set working directory
WORKDIR /app

# Update pip
RUN python -m pip install --no-cache-dir --upgrade pip

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler code
COPY handler.py .

# Set the handler
CMD ["python", "-u", "handler.py"]

