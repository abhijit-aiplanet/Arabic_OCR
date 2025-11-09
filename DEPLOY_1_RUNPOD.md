# 🎯 Step 1: Deploy Model Service on RunPod

**Estimated Time**: 15-20 minutes  
**Cost**: ~$0.45/hour when active (serverless = pay only when processing images)

---

## ✅ What You Need Before Starting

- [ ] RunPod account created at https://runpod.io
- [ ] $10+ credits added to your RunPod account
- [ ] Your GitHub repo URL: `https://github.com/abhijit-aiplanet/Arabic_OCR`
- [ ] A notepad to save important values

---

## 📝 Step-by-Step Instructions

### **Step 1: Create RunPod Account**

1. Go to **https://runpod.io**
2. Click **"Sign Up"** (top right)
3. Create account with email/password or Google
4. Verify your email

### **Step 2: Add Credits**

1. After logging in, click your profile icon (top right)
2. Click **"Billing"**
3. Click **"Add Credits"**
4. Add **$10 minimum** (recommended: $20-50 for testing)
5. Complete payment

💡 **Note**: You only pay for actual compute time. $10 can process hundreds of images.

---

### **Step 3: Create Serverless Endpoint**

1. In RunPod dashboard, click **"Serverless"** in left sidebar

2. Click **"+ New Endpoint"** (green button)

3. You'll see a configuration form. Fill it out:

---

#### **Basic Settings:**

| Field | Value |
|-------|-------|
| **Endpoint Name** | `ain-ocr-model` |
| **Select GPU** | RTX A6000 (48GB) ⭐ Recommended<br>_or_ A40 (48GB)<br>_or_ RTX 4090 (24GB) - budget option |

💡 **GPU Selection Tips**:
- **RTX A6000**: Best balance of price and performance
- **A40**: Slightly cheaper, same performance
- **RTX 4090**: Budget option, but only 24GB VRAM (model needs ~20-25GB)
- **Don't use**: GPUs with less than 24GB VRAM (will crash)

---

#### **Container Configuration:**

| Field | Value |
|-------|-------|
| **Container Image** | `runpod/pytorch:2.1.1-py3.10-cuda11.8.0-devel-ubuntu22.04` |
| **Container Disk** | `50 GB` (minimum) |
| **Volume Disk** | Leave empty (optional) |

---

#### **Scaling Configuration:**

| Field | Value |
|-------|-------|
| **Idle Timeout** | `30` seconds |
| **Max Workers** | `1` (start with 1, increase later if needed) |
| **GPU IDs** | Leave as selected |

---

#### **Advanced Settings** (Click "Advanced"):

**Start Command** field - Copy and paste this EXACTLY:

```bash
cd /workspace && git clone https://github.com/abhijit-aiplanet/Arabic_OCR.git && cd Arabic_OCR/model-service && pip install -r requirements.txt && python handler.py
```

💡 **What this does**:
1. Clones your GitHub repository
2. Goes into the model-service directory
3. Installs all required packages
4. Starts the model service

---

### **Step 4: Deploy**

1. Review all settings
2. Click **"Deploy"** button (bottom right)
3. Wait 2-5 minutes for deployment

You'll see a status indicator. Wait until it shows **"Ready"** or **"Idle"**.

---

### **Step 5: Get Your Endpoint Details**

After deployment completes:

1. Click on your endpoint name (`ain-ocr-model`)

2. You'll see the endpoint details page

3. **SAVE THESE VALUES** (you'll need them for backend deployment):

   **Endpoint ID**: 
   ```
   xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
   ```
   
   **Full Endpoint URL**:
   ```
   https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync
   ```

4. Copy the **Endpoint ID** - you'll need it!

---

### **Step 6: Get Your API Key**

1. Click on your profile icon (top right)

2. Click **"Settings"**

3. Click **"API Keys"** tab

4. Click **"+ Create API Key"**

5. Give it a name: `OCR Backend`

6. Click **"Create"**

7. **COPY THE API KEY IMMEDIATELY** (it won't be shown again):
   ```
   xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

8. Save this key securely!

---

### **Step 7: Test Your Endpoint** (Optional but Recommended)

Test if your endpoint is working:

1. In your endpoint details page, look for **"Test"** or **"Run"** button

2. Or use this curl command (replace YOUR_ENDPOINT_ID and YOUR_API_KEY):

```bash
curl -X POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
      "prompt": "Test prompt",
      "max_new_tokens": 100,
      "min_pixels": 200704,
      "max_pixels": 1003520
    }
  }'
```

**Expected Response** (means it's working):
```json
{
  "status": "COMPLETED",
  "output": {
    "text": "...",
    "status": "success"
  }
}
```

---

## 📋 Summary - Save These Values!

Before moving to the next step, make sure you have:

✅ **Endpoint ID**: `________________________`

✅ **Full Endpoint URL**: `https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync`

✅ **API Key**: `________________________`

**Write these down!** You'll need them for the backend deployment.

---

## 💰 Cost Breakdown

| Usage | Cost per Image | Monthly (1000 images) |
|-------|---------------|----------------------|
| RTX A6000 | $0.0006-0.001 | $0.60-$1.00 |
| Processing Time | 3-5 seconds/image | Only pay when active |

**Serverless Advantage**: Pod spins down when idle = **$0 cost when not processing**

---

## 🔧 Troubleshooting

### Issue: "Out of Memory" Error
**Solution**: 
- Use A6000 or A40 (48GB VRAM)
- Don't use GPUs with less than 24GB

### Issue: "Container failed to start"
**Solution**:
- Check container disk is 50GB minimum
- Verify the start command was pasted correctly
- Check logs in RunPod dashboard

### Issue: Model takes too long to load
**Solution**:
- First load takes 5-10 minutes (downloading model)
- Subsequent requests are fast (3-5 seconds)
- Use volume mounting to cache model (advanced)

### Issue: Endpoint shows "Error" status
**Solution**:
1. Click on the endpoint
2. Click **"Logs"** tab
3. Read the error messages
4. Common fix: Increase container disk size

---

## ✅ Checklist

Before proceeding to backend deployment:

- [ ] RunPod account created and funded
- [ ] GPU endpoint deployed successfully
- [ ] Endpoint status shows "Ready" or "Idle"
- [ ] Endpoint ID saved
- [ ] API Key saved
- [ ] Test successful (optional)

---

## 🚀 Next Step

You're done with RunPod! 🎉

**Next**: Deploy the backend on Vercel

👉 Open file: `DEPLOY_2_BACKEND_VERCEL.md`

---

## 📞 Need Help?

- **RunPod Documentation**: https://docs.runpod.io/
- **Check Logs**: RunPod Dashboard → Your Endpoint → Logs tab
- **Community**: RunPod Discord (link in their dashboard)

---

**Last Updated**: November 2025  
**Repository**: https://github.com/abhijit-aiplanet/Arabic_OCR

