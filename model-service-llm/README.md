# Qwen 2.5 LLM Service for Agentic OCR

This service provides reasoning capabilities for the agentic OCR system on RunPod.

## Purpose

The LLM service handles:
- **Analysis**: Analyzing OCR output for issues and uncertainties
- **Region Estimation**: Estimating bounding box coordinates for uncertain fields
- **Merging**: Intelligently merging multiple OCR passes

## Network Volume Setup (Required)

Since the Qwen 2.5 32B model is ~60GB, we use a **Network Volume** to store it persistently.

### Step 1: Create a Network Volume

1. Go to [RunPod Storage](https://www.runpod.io/console/user/storage)
2. Click **"+ New Network Volume"**
3. Configure:
   - **Name**: `qwen-model-cache`
   - **Region**: Same region as your serverless endpoint
   - **Size**: **100 GB** (model is ~60GB + overhead)
4. Click **Create**

### Step 2: Pre-download the Model (One-time setup)

You need to download the model to the Network Volume once. The easiest way:

#### Option A: Let it download on first request
1. Deploy the endpoint with the Network Volume attached
2. Send a test request - it will download the model (takes ~10-15 min)
3. Subsequent requests will be fast (model loads from cache)

#### Option B: Use a GPU Pod to pre-download (faster)
1. Go to [RunPod Pods](https://www.runpod.io/console/pods)
2. Create a new GPU Pod with your Network Volume attached
3. SSH into the pod and run:

```bash
# Install dependencies
pip install transformers huggingface_hub

# Download model to network volume
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen2.5-32B-Instruct',
    cache_dir='/runpod-volume/huggingface',
    ignore_patterns=['*.md', '*.txt', 'LICENSE*']
)
print('Download complete!')
"
```

4. Terminate the pod (the model stays on the Network Volume)

### Step 3: Attach Network Volume to Serverless Endpoint

1. Go to your **Arabic_OCR_LLM** endpoint settings
2. Click **"Edit Endpoint"**
3. Under **"Network Volume"**, select `qwen-model-cache`
4. Save and redeploy

## Model Options

| Model | VRAM Required | Download Size | Quality |
|-------|--------------|---------------|---------|
| Qwen2.5-7B-Instruct | ~8GB | ~14GB | Good |
| Qwen2.5-32B-Instruct | ~20GB (4-bit) | ~60GB | Excellent |
| Qwen2.5-72B-Instruct-AWQ | ~40GB | ~40GB | Best |

**Default**: Qwen2.5-32B-Instruct (best balance of quality and speed)

To change the model, edit `MODEL_ID` in `llm_handler.py`.

## Endpoint Configuration

| Setting | Recommended Value |
|---------|-------------------|
| GPU | RTX A6000 (48GB) or RTX 4090 (24GB) |
| Container Disk | 20 GB (code only, model on volume) |
| Network Volume | 100 GB |
| Idle Timeout | 60-120 seconds |
| Max Workers | 1 |

## Testing

Test the endpoint:

```bash
curl -X POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "What is 2+2? Answer briefly.",
      "max_tokens": 50,
      "temperature": 0.1
    }
  }'
```

Expected response:
```json
{
  "status": "COMPLETED",
  "output": {
    "text": "2+2 equals 4.",
    "tokens_generated": 8,
    "status": "success"
  }
}
```

## Cost Estimate

| Component | Cost |
|-----------|------|
| Network Volume (100GB) | ~$10/month |
| GPU (RTX A6000) | ~$0.45/hour (only when processing) |
| Per Request | ~$0.002-0.005 |

## Troubleshooting

### "Model not found" error
- Ensure Network Volume is attached to the endpoint
- Check that the model was downloaded to `/runpod-volume/huggingface`

### Out of Memory
- Use 4-bit quantization (enabled by default)
- Use smaller model (7B instead of 32B)
- Ensure GPU has 24GB+ VRAM

### Slow first request
- First request downloads/loads the model (~2-5 min if cached, ~15 min if downloading)
- Subsequent requests are fast (~2-5 seconds)
- Increase idle timeout to keep model in memory

### Build timeout
- This is expected! The Dockerfile no longer downloads the model during build
- Model is downloaded at runtime to the Network Volume
