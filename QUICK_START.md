# ⚡ Quick Start Guide

Get your AIN OCR application running in under 30 minutes!

## 🎯 What You'll Deploy

- ✅ **Frontend** (Next.js) on Vercel - ~5 minutes
- ✅ **Backend** (FastAPI) on Vercel - ~10 minutes  
- ✅ **Model Service** (AIN VLM) on RunPod - ~15 minutes

**Total Time**: ~30 minutes

**Total Cost**: ~$5-20/month (low traffic)

---

## 📋 Prerequisites Checklist

Before starting, make sure you have:

- [ ] GitHub account (to store code)
- [ ] Vercel account (free tier) - [Sign up](https://vercel.com)
- [ ] RunPod account with $10+ credits - [Sign up](https://runpod.io)
- [ ] Your code pushed to GitHub

---

## 🚀 Step-by-Step Deployment

### Step 1: Deploy Model on RunPod (15 min) 🎯

1. **Log in to RunPod**: https://runpod.io

2. **Go to Serverless**:
   - Click **"Serverless"** in sidebar
   - Click **"+ New Endpoint"**

3. **Configure Endpoint**:
   ```
   Name: ain-ocr-model
   GPU Type: RTX A6000 (48GB) or A40 (48GB)
   Container Image: runpod/pytorch:2.1.1-py3.10-cuda11.8.0-devel-ubuntu22.04
   Container Disk: 50 GB
   Idle Timeout: 30 seconds
   Max Workers: 1
   ```

4. **Add Startup Command** (Advanced section):
   ```bash
   cd /workspace && \
   git clone YOUR_GITHUB_REPO_URL && \
   cd Dots-OCR/model-service && \
   pip install -r requirements.txt && \
   python handler.py
   ```
   
   Replace `YOUR_GITHUB_REPO_URL` with your actual GitHub repo URL.

5. **Click "Deploy"**

6. **Save These Values**:
   - Endpoint ID: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
   - Full URL: `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync`
   
7. **Get API Key**:
   - Go to **Settings** → **API Keys**
   - Click **"+ API Key"**
   - Copy and save the key

✅ **Model Service Deployed!**

---

### Step 2: Deploy Backend on Vercel (10 min) 🔧

1. **Log in to Vercel**: https://vercel.com

2. **Create New Project**:
   - Click **"Add New..."** → **"Project"**
   - Click **"Import"** next to your GitHub repo

3. **Configure Project**:
   ```
   Framework Preset: Other
   Root Directory: backend
   Build Command: (leave empty)
   Output Directory: (leave empty)
   Install Command: (leave empty)
   ```

4. **Add Environment Variables**:
   
   Click **"Environment Variables"** and add these three:
   
   | Name | Value |
   |------|-------|
   | `RUNPOD_ENDPOINT_URL` | `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync` |
   | `RUNPOD_API_KEY` | Your RunPod API key from Step 1 |
   | `FRONTEND_URL` | `http://localhost:3000` (we'll update this) |

5. **Click "Deploy"**

6. **Wait for deployment** (~2 minutes)

7. **Save Your Backend URL**:
   - After deployment, copy the URL: `https://your-backend-xyz.vercel.app`

8. **Test Backend**:
   ```bash
   curl https://your-backend-xyz.vercel.app/health
   ```
   
   Should return:
   ```json
   {"status": "healthy", "model_service": "configured"}
   ```

✅ **Backend Deployed!**

---

### Step 3: Deploy Frontend on Vercel (5 min) 🎨

1. **Still in Vercel**, click **"Add New..."** → **"Project"** again

2. **Import Same Repo** (or create new project)

3. **Configure Project**:
   ```
   Framework Preset: Next.js
   Root Directory: frontend
   Build Command: npm run build
   Output Directory: (leave default)
   Install Command: npm install
   ```

4. **Add Environment Variable**:
   
   | Name | Value |
   |------|-------|
   | `NEXT_PUBLIC_API_URL` | `https://your-backend-xyz.vercel.app` |
   
   Use the backend URL from Step 2.

5. **Click "Deploy"**

6. **Wait for deployment** (~2 minutes)

7. **Save Your Frontend URL**:
   - Copy the URL: `https://your-frontend-abc.vercel.app`

✅ **Frontend Deployed!**

---

### Step 4: Update Backend CORS (2 min) 🔄

Now update the backend to allow requests from your frontend:

1. **Go to Vercel Dashboard** → Your Backend Project

2. **Settings** → **Environment Variables**

3. **Edit `FRONTEND_URL`**:
   - Change from `http://localhost:3000`
   - To: `https://your-frontend-abc.vercel.app`

4. **Save** and **Redeploy**:
   - Go to **Deployments** tab
   - Click **"⋯"** on latest deployment
   - Click **"Redeploy"**

✅ **Backend Updated!**

---

## 🎉 You're Done! Test Your Application

1. **Visit your frontend**: `https://your-frontend-abc.vercel.app`

2. **Upload a test image** (drag & drop or click to browse)

3. **Click "Process Image"**

4. **Wait 3-5 seconds** for processing

5. **See extracted text!**

---

## 📊 Quick Reference

### Your Deployed URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | `https://your-frontend-abc.vercel.app` | User interface |
| Backend | `https://your-backend-xyz.vercel.app` | API service |
| Model | `https://api.runpod.ai/v2/YOUR_ID/runsync` | GPU inference |

### Your Environment Variables

**Backend** (Vercel):
```env
RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync
RUNPOD_API_KEY=sk_xxxxxxxxxxxxxxxxxx
FRONTEND_URL=https://your-frontend-abc.vercel.app
```

**Frontend** (Vercel):
```env
NEXT_PUBLIC_API_URL=https://your-backend-xyz.vercel.app
```

---

## 🐛 Common Issues & Solutions

### Issue: "CORS Error" in frontend

**Solution**: 
1. Make sure `FRONTEND_URL` in backend matches your actual frontend URL exactly
2. Redeploy backend after changing environment variables
3. Hard refresh browser (Ctrl+F5)

### Issue: "Timeout Error" when processing

**Solution**:
1. Check RunPod endpoint is active (RunPod Dashboard → Serverless → Your Endpoint)
2. Increase max workers in RunPod (Settings → Max Workers: 2-3)
3. Check RunPod logs for errors

### Issue: "Model not configured" error

**Solution**:
1. Verify `RUNPOD_ENDPOINT_URL` is correct in backend
2. Verify `RUNPOD_API_KEY` is correct in backend
3. Check RunPod endpoint is running
4. Test RunPod endpoint directly (see Step 1.8)

### Issue: Frontend shows "Failed to fetch"

**Solution**:
1. Check backend URL is correct in frontend environment variables
2. Test backend health endpoint: `curl https://your-backend.vercel.app/health`
3. Check backend logs in Vercel dashboard

---

## 💰 Cost Monitoring

### How to Check Costs

**RunPod**:
- Dashboard → Billing → Usage
- Only charges for active compute time
- ~$0.0006-$0.001 per image

**Vercel**:
- Dashboard → Account → Usage
- Free tier: 100 GB-hours
- Usually free for low-medium traffic

### Expected Costs

| Usage | RunPod | Vercel | Total |
|-------|--------|--------|-------|
| 100 images/month | $0.06-$0.10 | $0 | ~$0.10 |
| 1000 images/month | $0.60-$1.00 | $0 | ~$1.00 |
| 10,000 images/month | $6-$10 | $0-$20 | ~$10-$30 |

---

## 📈 Next Steps

Now that your application is deployed:

1. ✅ **Test thoroughly** with different images
2. ✅ **Share the URL** with users
3. ✅ **Monitor usage** in dashboards
4. ✅ **Set up custom domain** (optional)
5. ✅ **Add analytics** (Vercel Analytics)

### Want to Customize?

- **Change UI colors**: Edit `frontend/tailwind.config.js`
- **Modify API behavior**: Edit `backend/main.py`
- **Adjust model parameters**: Change default values in `backend/main.py`

### Want to Scale?

- **More traffic**: Increase RunPod max workers (5-10)
- **Faster processing**: Use A100 GPU on RunPod
- **Better reliability**: Use dedicated RunPod pods instead of serverless

---

## 📚 More Resources

- **Full Deployment Guide**: See `DEPLOYMENT_GUIDE.md` for detailed info
- **Architecture Details**: See `PROJECT_STRUCTURE.md`
- **Backend API Docs**: Visit `https://your-backend.vercel.app/docs`
- **Component Docs**: See README files in each directory

---

## 🆘 Need Help?

If you're stuck:

1. **Check logs**:
   - Frontend: Vercel Dashboard → Frontend Project → Logs
   - Backend: Vercel Dashboard → Backend Project → Logs
   - Model: RunPod Dashboard → Endpoint → Logs

2. **Test each service independently**:
   - Model: Use test_handler.py locally
   - Backend: Test /health endpoint
   - Frontend: Check browser console (F12)

3. **Review documentation**:
   - `DEPLOYMENT_GUIDE.md` - Comprehensive guide
   - `SETUP_INSTRUCTIONS.txt` - Quick checklist
   - Component READMEs - Specific guides

---

## ✅ Deployment Checklist

Use this to track your progress:

- [ ] RunPod account created with credits
- [ ] Vercel account created
- [ ] Code pushed to GitHub
- [ ] Model service deployed on RunPod
- [ ] RunPod endpoint ID and API key saved
- [ ] Backend deployed on Vercel
- [ ] Backend environment variables set
- [ ] Backend health check passes
- [ ] Frontend deployed on Vercel
- [ ] Frontend environment variables set
- [ ] Backend CORS updated with frontend URL
- [ ] End-to-end test successful
- [ ] Costs monitored

---

**🎉 Congratulations! Your AIN OCR application is live!**

Visit your application: `https://your-frontend-abc.vercel.app`

