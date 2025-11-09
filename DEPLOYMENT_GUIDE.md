# 🚀 Deployment Guide - AIN OCR Application

Complete guide to deploy the AIN OCR application with separated frontend, backend, and model service.

## Architecture Overview

```
┌─────────────────┐
│   Frontend      │  ← Deployed on Vercel
│   (Next.js)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Backend       │  ← Deployed on Vercel/Render
│   (FastAPI)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Service  │  ← Deployed on RunPod
│  (AIN VLM)      │
└─────────────────┘
```

## Prerequisites

Before you begin, make sure you have:

- [ ] GitHub account
- [ ] Vercel account (free tier is fine)
- [ ] RunPod account with credits
- [ ] Git installed locally
- [ ] Node.js 18+ installed (for frontend)
- [ ] Python 3.10+ installed (for backend)

---

## Step 1: Deploy Model Service on RunPod 🎯

### 1.1 Create RunPod Account

1. Visit [runpod.io](https://www.runpod.io/)
2. Sign up for an account
3. Add credits (minimum $10 recommended)

### 1.2 Choose GPU

**Recommended GPUs (in order of preference):**

| GPU | VRAM | Price/hr | Notes |
|-----|------|----------|-------|
| RTX A6000 | 48GB | ~$0.45 | Best balance |
| A40 | 48GB | ~$0.40 | Great option |
| RTX 4090 | 24GB | ~$0.35 | Budget option |
| A100 (40GB) | 40GB | ~$1.00 | High performance |

**Why 24GB+ VRAM?** The AIN model requires approximately 20-25GB of VRAM.

### 1.3 Deploy Using Serverless

**Option A: Web UI Deployment (Easiest)**

1. Go to **"Serverless"** → **"Endpoints"**
2. Click **"New Endpoint"**
3. Configure:
   - **Name**: `ain-ocr-model`
   - **GPU Type**: Select RTX A6000 or A40
   - **Container Image**: `runpod/pytorch:2.1.1-py3.10-cuda11.8.0-devel-ubuntu22.04`
   - **Container Disk**: 50 GB
   - **Idle Timeout**: 30 seconds
   - **Max Workers**: 1 (increase if needed)

4. Click **"Advanced"** and add startup command:
```bash
cd /workspace && git clone YOUR_REPO_URL && cd Dots-OCR/model-service && pip install -r requirements.txt && python handler.py
```

5. Click **"Deploy"**

**Option B: Docker Deployment (Recommended for Production)**

1. Build and push Docker image:

```bash
cd model-service

# Build
docker build -t YOUR_DOCKERHUB_USERNAME/ain-ocr-model:latest .

# Push
docker push YOUR_DOCKERHUB_USERNAME/ain-ocr-model:latest
```

2. In RunPod:
   - Go to **"Serverless"** → **"Endpoints"**
   - Click **"New Endpoint"**
   - **Container Image**: `YOUR_DOCKERHUB_USERNAME/ain-ocr-model:latest`
   - Select GPU and deploy

### 1.4 Get Endpoint Details

After deployment, note down:
- **Endpoint ID**: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
- **Endpoint URL**: `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync`

Go to **"Settings"** → **"API Keys"** and create a new API key:
- **API Key**: `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

**Save these for the backend configuration!**

### 1.5 Test the Endpoint

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "base64_encoded_image_string",
      "prompt": "Extract all text from this image.",
      "max_new_tokens": 2048,
      "min_pixels": 200704,
      "max_pixels": 1003520
    }
  }'
```

Expected response:
```json
{
  "output": {
    "text": "Extracted text...",
    "status": "success"
  },
  "status": "COMPLETED"
}
```

---

## Step 2: Deploy Backend on Vercel 🔧

### 2.1 Prepare Backend

1. Make sure your code is pushed to GitHub
2. Backend is in the `backend/` directory

### 2.2 Deploy to Vercel

1. Visit [vercel.com](https://vercel.com)
2. Click **"New Project"**
3. **Import** your GitHub repository
4. Configure:
   - **Framework Preset**: Other
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Output Directory**: (leave empty)

5. **Environment Variables** - Add these:

```
RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync
RUNPOD_API_KEY=your_runpod_api_key_here
FRONTEND_URL=https://your-frontend-url.vercel.app
```

6. Click **"Deploy"**

### 2.3 Verify Backend

Once deployed, test:

```bash
curl https://your-backend.vercel.app/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_service": "configured"
}
```

### 2.4 Alternative: Deploy on Render

If you prefer Render over Vercel for the backend:

1. Visit [render.com](https://render.com)
2. Click **"New"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `ain-ocr-backend`
   - **Root Directory**: `backend`
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free (or paid for better performance)

5. Add Environment Variables (same as Vercel)
6. Click **"Create Web Service"**

---

## Step 3: Deploy Frontend on Vercel 🎨

### 3.1 Prepare Frontend

1. Make sure your code is pushed to GitHub
2. Frontend is in the `frontend/` directory

### 3.2 Deploy to Vercel

1. Visit [vercel.com](https://vercel.com)
2. Click **"New Project"**
3. **Import** your GitHub repository (or create a new project if same repo)
4. Configure:
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`
   - **Install Command**: `npm install`

5. **Environment Variables** - Add:

```
NEXT_PUBLIC_API_URL=https://your-backend.vercel.app
```

6. Click **"Deploy"**

### 3.3 Update Backend CORS

After frontend is deployed, update backend's `FRONTEND_URL` environment variable:

1. Go to your backend project on Vercel
2. **Settings** → **Environment Variables**
3. Update `FRONTEND_URL` to your frontend URL: `https://your-frontend.vercel.app`
4. **Redeploy** the backend

---

## Step 4: Final Configuration & Testing ✅

### 4.1 Update Environment Variables

**Backend** (on Vercel/Render):
```env
RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync
RUNPOD_API_KEY=your_runpod_api_key_here
FRONTEND_URL=https://your-frontend.vercel.app
```

**Frontend** (on Vercel):
```env
NEXT_PUBLIC_API_URL=https://your-backend.vercel.app
```

### 4.2 Full End-to-End Test

1. Visit your frontend: `https://your-frontend.vercel.app`
2. Upload an image
3. Click "Process Image"
4. Verify text extraction works

### 4.3 Monitor & Debug

**Check Logs:**
- **Frontend**: Vercel Dashboard → Your Project → Logs
- **Backend**: Vercel Dashboard → Your Project → Logs
- **Model**: RunPod → Serverless → Your Endpoint → Logs

**Common Issues:**

1. **CORS Error**: Update backend `FRONTEND_URL` and redeploy
2. **Timeout Error**: Increase RunPod max workers or use dedicated pod
3. **Out of Memory**: Use GPU with more VRAM (A40/A6000)

---

## Cost Estimation 💰

### RunPod (Model Service)
- **Serverless**: Only pay for compute time
  - RTX A6000: ~$0.45/hr when active (~$0.00012/sec)
  - Typical inference: 3-5 seconds per image
  - Cost per image: ~$0.0006-$0.001
- **Dedicated Pod** (if high traffic):
  - RTX A6000: ~$0.45-$0.65/hr
  - Good for 24/7 availability

### Vercel (Frontend + Backend)
- **Free Tier**: 
  - 100 GB-hours serverless function execution
  - 100 GB bandwidth
  - Usually sufficient for personal/small projects
- **Pro Plan**: $20/month for higher limits

### Render (Backend Alternative)
- **Free Tier**: Limited resources, spins down after inactivity
- **Starter**: $7/month for always-on service

**Estimated Monthly Cost for Low-Medium Traffic:**
- RunPod: $5-$20 (serverless)
- Vercel: $0-$20
- **Total**: ~$5-$40/month

---

## Scaling & Optimization 📈

### For High Traffic:

1. **Increase RunPod Workers**:
   - Go to your endpoint settings
   - Increase "Max Workers" to 5-10
   - Enable auto-scaling

2. **Use Dedicated Pods** (instead of serverless):
   - Faster cold starts
   - More consistent performance
   - Better for high-frequency requests

3. **Add Caching**:
   - Cache frequently processed images
   - Use Redis for response caching

4. **CDN Configuration**:
   - Vercel automatically uses CDN
   - Optimize images before upload

### For Production:

- [ ] Add rate limiting to backend
- [ ] Implement API authentication
- [ ] Add monitoring (Sentry, LogRocket)
- [ ] Set up custom domain
- [ ] Add health check alerts
- [ ] Implement queue system for batch processing

---

## Troubleshooting 🔧

### Model Service Issues

**Problem**: Out of Memory (OOM)
- **Solution**: Use GPU with more VRAM (A40 or A6000)

**Problem**: Slow cold starts
- **Solution**: Increase min workers or use dedicated pod

**Problem**: Model not loading
- **Solution**: Check container disk size (needs 50GB+)

### Backend Issues

**Problem**: Timeout errors
- **Solution**: Increase timeout in backend code or use async processing

**Problem**: CORS errors
- **Solution**: Verify `FRONTEND_URL` matches actual frontend URL

### Frontend Issues

**Problem**: API URL not working
- **Solution**: Ensure `NEXT_PUBLIC_API_URL` is set and starts with `https://`

**Problem**: Images not uploading
- **Solution**: Check file size limits (default 5MB)

---

## Security Checklist 🔒

- [ ] RunPod API key is kept secret
- [ ] Backend validates file types and sizes
- [ ] CORS is properly configured
- [ ] Rate limiting is enabled
- [ ] No sensitive data in client-side code
- [ ] HTTPS is used for all services
- [ ] Environment variables are not committed to git

---

## Support & Resources 📚

- **RunPod Docs**: https://docs.runpod.io/
- **Vercel Docs**: https://vercel.com/docs
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Next.js Docs**: https://nextjs.org/docs

---

## Next Steps 🎯

After successful deployment:

1. ✅ Set up custom domain (optional)
2. ✅ Add analytics (Google Analytics, Vercel Analytics)
3. ✅ Implement authentication (if needed)
4. ✅ Add more features (batch processing, PDF support)
5. ✅ Set up monitoring and alerts

---

**Congratulations! Your application is now fully deployed!** 🎉

