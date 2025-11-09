# AIN OCR Model Service for RunPod

This directory contains the model service that runs on RunPod GPU instances.

## Features

- Runs MBZUAI/AIN vision language model for OCR
- Optimized for GPU inference
- Serverless deployment on RunPod
- Handles base64-encoded images
- Configurable inference parameters

## GPU Requirements

### Recommended Configuration:
- **GPU**: NVIDIA A40 (48GB) or RTX A6000 (48GB)
- **Alternatives**: 
  - RTX 4090 (24GB) - Budget option
  - A100 (40GB/80GB) - High performance
  - RTX 3090 (24GB) - Budget option

The model requires approximately 20-25GB of VRAM, so a GPU with at least 24GB is recommended.

## Setup on RunPod

### Option 1: Using RunPod Template (Recommended)

1. **Log in to RunPod**: https://www.runpod.io/

2. **Create a Template**:
   - Go to "Templates" in the sidebar
   - Click "New Template"
   - Configure:
     - **Container Image**: `runpod/pytorch:2.1.1-py3.10-cuda11.8.0-devel-ubuntu22.04`
     - **Docker Command**: Leave empty (we'll use the handler)
     - **Container Disk**: 50GB minimum
     - **Volume Disk**: Optional (for caching)

3. **Create a Serverless Endpoint**:
   - Go to "Serverless" → "Endpoints"
   - Click "New Endpoint"
   - Select your template
   - Choose GPU type (A40/RTX A6000 recommended)
   - Set:
     - **Idle Timeout**: 30 seconds
     - **Max Workers**: Based on your needs (1-5)
     - **GPU IDs**: Select your preferred GPU
   
4. **Upload Handler**:
   ```bash
   # Install RunPod CLI
   pip install runpod
   
   # Deploy
   runpod project create
   runpod project deploy
   ```

### Option 2: Using Docker Build

1. **Build the Docker image**:
   ```bash
   docker build -t ain-ocr-model .
   ```

2. **Push to a container registry** (Docker Hub, GitHub Container Registry, etc.):
   ```bash
   docker tag ain-ocr-model your-username/ain-ocr-model:latest
   docker push your-username/ain-ocr-model:latest
   ```

3. **Create endpoint in RunPod** using your custom image.

### Option 3: Using RunPod Web Terminal (Quick Start)

1. **Create a GPU Pod**:
   - Go to "Pods" → "GPU Instances"
   - Click "Deploy"
   - Select a GPU (A40 or RTX A6000)
   - Choose PyTorch template

2. **SSH into the pod** and run:
   ```bash
   cd /workspace
   git clone <your-repo-url>
   cd model-service
   pip install -r requirements.txt
   python handler.py
   ```

## Testing the Handler Locally

You can test the handler locally before deploying:

```python
import base64
from PIL import Image
import io

# Load your test image
with open("test_image.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Create test job
test_job = {
    "input": {
        "image": image_b64,
        "prompt": "Extract all text from this image.",
        "max_new_tokens": 2048,
        "min_pixels": 200704,
        "max_pixels": 1003520
    }
}

# Test handler
from handler import handler
result = handler(test_job)
print(result)
```

## API Usage

Once deployed, you'll get an endpoint URL like:
```
https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync
```

### Request Format:

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "base64_encoded_image_here",
      "prompt": "Extract all text from this image.",
      "max_new_tokens": 2048,
      "min_pixels": 200704,
      "max_pixels": 1003520
    }
  }'
```

### Response Format:

```json
{
  "output": {
    "text": "Extracted text content here",
    "status": "success"
  },
  "status": "COMPLETED"
}
```

## Environment Variables

No environment variables required for the model service itself. All configuration is passed via the API request.

## Cost Estimation

For RunPod serverless:
- **A40 (48GB)**: ~$0.0004/sec = ~$1.44/hour when active
- **RTX A6000 (48GB)**: ~$0.00068/sec = ~$2.45/hour when active
- **RTX 4090 (24GB)**: ~$0.00034/sec = ~$1.22/hour when active

Typical inference time: 2-5 seconds per image

You only pay for actual compute time (not idle time), making serverless very cost-effective for variable workloads.

## Troubleshooting

### Out of Memory (OOM) Errors
- Use a GPU with more VRAM (A40 or A6000)
- Reduce `max_pixels` in the request
- Reduce `max_new_tokens`

### Model Loading Timeout
- Increase container disk size
- Use volume mounting for model caching
- Pre-download model weights into the container

### Slow Cold Starts
- Increase min workers to keep instances warm
- Use volume mounting to cache model weights
- Consider using dedicated pods for consistent traffic

## Model Details

- **Model**: MBZUAI/AIN
- **Type**: Vision Language Model (VLM)
- **Architecture**: Qwen2-VL based
- **Size**: ~20GB
- **Primary Use**: OCR and text extraction from images
- **Languages**: Optimized for Arabic, supports multiple languages

