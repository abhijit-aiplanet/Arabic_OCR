# Qwen 2.5 LLM Service for Agentic OCR

This service provides reasoning capabilities for the agentic OCR system on RunPod.

## Purpose

The LLM service handles:
- **Analysis**: Analyzing OCR output for issues and uncertainties
- **Region Estimation**: Estimating bounding box coordinates for uncertain fields
- **Merging**: Intelligently merging multiple OCR passes

## Model Options

| Model | VRAM Required | Quality | Speed |
|-------|--------------|---------|-------|
| Qwen2.5-7B-Instruct | ~8GB | Good | Fast |
| Qwen2.5-32B-Instruct | ~20GB (4-bit) | Excellent | Medium |
| Qwen2.5-72B-Instruct | ~40GB (4-bit) | Best | Slower |

**Default**: Qwen2.5-32B-Instruct (best balance)

## Deployment to RunPod

### Step 1: Build Docker Image

```bash
cd model-service-llm
docker build -t your-dockerhub-username/qwen-llm-service:latest .
docker push your-dockerhub-username/qwen-llm-service:latest
```

### Step 2: Create RunPod Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **"+ New Endpoint"**
3. Configure:
   - **Name**: `qwen-llm-agentic`
   - **GPU**: RTX A6000 (48GB) recommended, or RTX 4090 (24GB) with 4-bit quantization
   - **Container Image**: `your-dockerhub-username/qwen-llm-service:latest`
   - **Container Disk**: 100GB (model is ~60GB)
   - **Idle Timeout**: 60 seconds
   - **Max Workers**: 1

4. Click **Deploy**

### Step 3: Get Endpoint URL

After deployment:
1. Copy the **Endpoint ID**
2. Your endpoint URL is: `https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync`

### Step 4: Configure Backend

Add to your backend `.env`:

```env
RUNPOD_LLM_ENDPOINT_URL=https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync
RUNPOD_LLM_API_KEY=your_runpod_api_key
```

## Testing

Test the endpoint:

```bash
curl -X POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "What is 2+2?",
      "max_tokens": 100,
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

| GPU | Cost/Hour | Typical Usage |
|-----|-----------|---------------|
| RTX A6000 | ~$0.45 | ~$0.002/request |
| RTX 4090 | ~$0.35 | ~$0.0015/request |

Agentic OCR typically makes 3-5 LLM calls per image.

## Troubleshooting

### Out of Memory
- Use 4-bit quantization (enabled by default)
- Use smaller model (7B instead of 32B)
- Ensure GPU has 24GB+ VRAM

### Slow First Request
- First request loads model (~30-60s)
- Subsequent requests are fast (~2-5s)
- Increase idle timeout to keep model warm

### Model Download Fails
- Check HuggingFace is accessible
- Increase container disk size
- Check build logs for errors
