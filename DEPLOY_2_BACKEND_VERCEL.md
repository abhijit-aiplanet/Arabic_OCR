# 🔧 Step 2: Deploy Backend on Vercel

**Estimated Time**: 10 minutes
**Cost**: Free (Free tier is usually sufficient)

**Prerequisites**: You must have completed RunPod deployment and have your Endpoint URL and API Key ready.

---

## ✅ What You Need Before Starting

- [ ] Vercel account (create at https://vercel.com)
- [ ] Your GitHub repository: `https://github.com/abhijit-aiplanet/Arabic_OCR`
- [ ] RunPod Endpoint URL from Step 1
- [ ] RunPod API Key from Step 1

---

## 📝 Step-by-Step Instructions

### **Step 1: Create Vercel Account**

1. Go to **https://vercel.com**
2. Click **"Sign Up"** (top right)
3. Choose **"Continue with GitHub"** (recommended)

   - This connects your GitHub account automatically
   - Click "Authorize Vercel"
4. Complete the setup

---

### **Step 2: Import Your Repository**

1. After logging in, you'll see the Vercel dashboard
2. Click **"Add New..."** button (top right)
3. Select **"Project"**
4. You'll see "Import Git Repository" page
5. Find your repository:

   - If you don't see it, click **"Adjust GitHub App Permissions"**
   - Select your GitHub account
   - Give Vercel access to `Arabic_OCR` repository
   - Click "Install"
6. Now you should see `abhijit-aiplanet/Arabic_OCR`
7. Click **"Import"** next to it

---

### **Step 3: Configure Backend Project**

You'll see the "Configure Project" page. Set these values:

#### **Project Settings:**

| Field                      | Value                                                 |
| -------------------------- | ----------------------------------------------------- |
| **Project Name**     | `arabic-ocr-backend` (or any name you like)         |
| **Framework Preset** | Select**"Other"** from dropdown                 |
| **Root Directory**   | Click**"Edit"** → Type `backend` → Click ✓ |
| **Build Command**    | Leave empty (Vercel will auto-detect)                 |
| **Output Directory** | Leave empty                                           |
| **Install Command**  | Leave empty                                           |

💡 **Important**: Make sure "Root Directory" is set to `backend` - this tells Vercel to deploy only the backend folder.

---

### **Step 4: Add Environment Variables**

This is the most important step! Scroll down to **"Environment Variables"** section.

Add these **THREE** environment variables:

#### **Variable 1: RUNPOD_ENDPOINT_URL**

1. Click **"Add"** under Environment Variables
2. **Key**:

   ```
   RUNPOD_ENDPOINT_URL
   ```
3. **Value**: Your full RunPod endpoint URL from Step 1

   ```
   https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync
   ```

   ⚠️ Replace `YOUR_ENDPOINT_ID` with your actual endpoint ID!
4. **Environment**: Select all three (Production, Preview, Development)

---

#### **Variable 2: RUNPOD_API_KEY**

1. Click **"Add"** again
2. **Key**:

   ```
   RUNPOD_API_KEY
   ```
3. **Value**: Your RunPod API Key from Step 1

   ```
   your_actual_api_key_here
   ```

   ⚠️ Paste your actual API key (starts with letters/numbers)
4. **Environment**: Select all three (Production, Preview, Development)

---

#### **Variable 3: FRONTEND_URL**

1. Click **"Add"** one more time
2. **Key**:

   ```
   FRONTEND_URL
   ```
3. **Value**: For now, use localhost

   ```
   http://localhost:3000
   ```

   💡 We'll update this after deploying the frontend
4. **Environment**: Select all three

---

### **Step 5: Deploy**

1. Review your settings:

   - ✅ Root Directory: `backend`
   - ✅ Framework: Other
   - ✅ 3 Environment Variables added
2. Click **"Deploy"** button (bottom)
3. Wait 2-3 minutes while Vercel:

   - Builds your backend
   - Deploys it globally
   - Assigns a URL
4. You'll see a success screen with confetti! 🎉

---

### **Step 6: Get Your Backend URL**

1. After deployment, you'll see your project dashboard
2. At the top, you'll see **"Domains"** section
3. Your backend URL will look like:

   ```
   https://arabic-ocr-backend-xxx.vercel.app
   ```
4. **SAVE THIS URL** - you'll need it for frontend deployment!
5. Click on the URL to verify it's working

---

### **Step 7: Test Your Backend**

Test if the backend is working:

#### **Test 1: Health Check**

Open your browser or use curl:

```bash
curl https://your-backend-url.vercel.app/health
```

**Expected Response**:

```json
{
  "status": "healthy",
  "model_service": "configured"
}
```

#### **Test 2: API Documentation**

Visit in your browser:

```
https://your-backend-url.vercel.app/docs
```

You should see the FastAPI Swagger documentation page.

---

## 📋 Summary - Save This Value!

Before moving to the next step, save:

✅ **Backend URL**: `https://________________________.vercel.app`

**Write this down!** You'll need it for frontend deployment.

---

## 🔧 Troubleshooting

### Issue: Build Failed

**Solution**:

1. Go to your project dashboard
2. Click "Deployments" tab
3. Click on the failed deployment
4. Check the build logs
5. Common fix: Verify Root Directory is set to `backend`

### Issue: Health check returns error

**Solution**:

1. Go to Vercel dashboard → Your project
2. Click "Settings" → "Environment Variables"
3. Verify all three variables are set correctly
4. Redeploy: Go to "Deployments" → "⋯" → "Redeploy"

### Issue: "CORS Error" or "model_service": "not_configured"

**Solution**:

1. Check `RUNPOD_ENDPOINT_URL` is correct (should end with `/runsync`)
2. Check `RUNPOD_API_KEY` is correct (no extra spaces)
3. Redeploy after fixing

### Issue: Can't find my repository

**Solution**:

1. Go to Vercel dashboard
2. Click "Add New..." → "Project"
3. Click "Adjust GitHub App Permissions"
4. Give Vercel access to your repository
5. Try importing again

---

## 💰 Cost Information

**Vercel Free Tier Includes**:

- ✅ 100 GB-hours serverless function execution
- ✅ 100 GB bandwidth
- ✅ Unlimited projects
- ✅ Automatic HTTPS
- ✅ Global CDN

**For this backend**: Free tier is usually sufficient for low-medium traffic.

**If you exceed free tier**: $20/month Pro plan with higher limits.

---

## 🔄 Updating Environment Variables Later

If you need to update the `FRONTEND_URL` after deploying frontend:

1. Go to Vercel dashboard
2. Select your backend project
3. Click **"Settings"** tab
4. Click **"Environment Variables"**
5. Find `FRONTEND_URL`
6. Click **"Edit"** (pencil icon)
7. Update the value
8. Click **"Save"**
9. **Important**: Go to "Deployments" tab → "⋯" → "Redeploy" to apply changes

---

## ✅ Checklist

Before proceeding to frontend deployment:

- [ ] Vercel account created
- [ ] Backend project deployed
- [ ] Deployment successful (green checkmark)
- [ ] Backend URL saved
- [ ] Health check returns {"status": "healthy"}
- [ ] API docs accessible at `/docs`

---

## 🚀 Next Step

Backend deployed successfully! 🎉

**Next**: Deploy the frontend on Vercel

👉 Open file: `DEPLOY_3_FRONTEND_VERCEL.md`

---

## 📞 Need Help?

- **Vercel Documentation**: https://vercel.com/docs
- **Check Logs**: Vercel Dashboard → Your Project → View Function Logs
- **Support**: Vercel has excellent documentation and community

---

## 📝 Quick Reference

**Your Backend Endpoints**:

- Health: `https://your-backend.vercel.app/health`
- API Docs: `https://your-backend.vercel.app/docs`
- OCR: `https://your-backend.vercel.app/api/ocr`

**Environment Variables**:

```
RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/YOUR_ID/runsync
RUNPOD_API_KEY=your_api_key_here
FRONTEND_URL=http://localhost:3000 (update after frontend deployment)
```

---

**Last Updated**: November 2025
**Repository**: https://github.com/abhijit-aiplanet/Arabic_OCR
