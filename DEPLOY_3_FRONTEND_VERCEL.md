# 🎨 Step 3: Deploy Frontend on Vercel

**Estimated Time**: 5-10 minutes  
**Cost**: Free (Free tier is sufficient)

**Prerequisites**: You must have completed backend deployment and have your Backend URL ready.

---

## ✅ What You Need Before Starting

- [ ] Vercel account (already created in Step 2)
- [ ] Your GitHub repository: `https://github.com/abhijit-aiplanet/Arabic_OCR`
- [ ] Backend URL from Step 2 (e.g., `https://your-backend.vercel.app`)

---

## 📝 Step-by-Step Instructions

### **Step 1: Import Repository Again**

1. Go to **Vercel Dashboard**: https://vercel.com/dashboard

2. Click **"Add New..."** button (top right)

3. Select **"Project"**

4. Find your repository: `abhijit-aiplanet/Arabic_OCR`

5. Click **"Import"** next to it

💡 **Note**: Yes, you're importing the same repository again, but this time for the frontend (different root directory).

---

### **Step 2: Configure Frontend Project**

You'll see the "Configure Project" page. Set these values:

#### **Project Settings:**

| Field | Value |
|-------|-------|
| **Project Name** | `arabic-ocr-frontend` (or any name you like) |
| **Framework Preset** | Select **"Next.js"** from dropdown |
| **Root Directory** | Click **"Edit"** → Type `frontend` → Click ✓ |
| **Build Command** | Leave as default: `npm run build` |
| **Output Directory** | Leave as default: `.next` |
| **Install Command** | Leave as default: `npm install` |

💡 **Important**: Make sure "Root Directory" is set to `frontend` - this deploys only the frontend folder.

---

### **Step 3: Add Environment Variable**

Scroll down to **"Environment Variables"** section.

Add **ONE** environment variable:

#### **Variable: NEXT_PUBLIC_API_URL**

1. Click **"Add"** under Environment Variables

2. **Key**: 
   ```
   NEXT_PUBLIC_API_URL
   ```
   
   ⚠️ **Important**: The name must be EXACTLY this, including `NEXT_PUBLIC_` prefix!

3. **Value**: Your backend URL from Step 2
   ```
   https://your-backend-url.vercel.app
   ```
   
   ⚠️ **Important**: 
   - Use your ACTUAL backend URL
   - Do NOT add a trailing slash (/)
   - Should start with `https://`

4. **Environment**: Select all three (Production, Preview, Development)

---

### **Step 4: Deploy**

1. Review your settings:
   - ✅ Root Directory: `frontend`
   - ✅ Framework: Next.js
   - ✅ 1 Environment Variable added (`NEXT_PUBLIC_API_URL`)

2. Click **"Deploy"** button (bottom)

3. Wait 2-3 minutes while Vercel:
   - Installs dependencies (npm packages)
   - Builds your Next.js frontend
   - Deploys it globally
   - Assigns a URL

4. You'll see a success screen with confetti! 🎉

---

### **Step 5: Get Your Frontend URL**

1. After deployment, you'll see your project dashboard

2. At the top, you'll see **"Domains"** section

3. Your frontend URL will look like:
   ```
   https://arabic-ocr-frontend-xxx.vercel.app
   ```

4. **SAVE THIS URL** - you'll need it to update the backend!

5. Click on the URL to open your application

---

### **Step 6: Update Backend CORS** ⚠️ IMPORTANT!

Now you need to update the backend to allow requests from your frontend:

1. Go back to **Vercel Dashboard**

2. Select your **BACKEND** project (not frontend)

3. Click **"Settings"** tab

4. Click **"Environment Variables"**

5. Find the variable named `FRONTEND_URL`

6. Click the **"⋯"** (three dots) on the right

7. Click **"Edit"**

8. **Update the value** to your frontend URL:
   ```
   https://your-frontend-url.vercel.app
   ```
   
   ⚠️ Replace with your ACTUAL frontend URL from Step 5!

9. Click **"Save"**

---

### **Step 7: Redeploy Backend** ⚠️ IMPORTANT!

The environment variable change requires a redeploy:

1. Still in your **backend project** on Vercel

2. Click **"Deployments"** tab (top)

3. Find the latest deployment (top of the list)

4. Click the **"⋯"** (three dots) on the right

5. Click **"Redeploy"**

6. Click **"Redeploy"** again to confirm

7. Wait 1-2 minutes for redeployment

---

### **Step 8: Test Your Application!** 🎉

1. Visit your frontend URL:
   ```
   https://your-frontend-url.vercel.app
   ```

2. You should see the AIN OCR application with a beautiful UI

3. **Test the full flow**:
   - Drag and drop an image (or click to browse)
   - Click "Process Image"
   - Wait 3-5 seconds
   - See the extracted text appear!

---

## 📋 Summary - Your Deployed Application!

Congratulations! Save these URLs:

✅ **Frontend URL**: `https://________________________.vercel.app`

✅ **Backend URL**: `https://________________________.vercel.app`

✅ **RunPod Endpoint**: `https://api.runpod.ai/v2/________/runsync`

---

## 🎉 Success! Application is Live!

Your application is now fully deployed and accessible worldwide! 🌍

**Share your application**: Send the frontend URL to anyone!

---

## 🔧 Troubleshooting

### Issue: Page loads but "Process Image" doesn't work
**Solution**:
1. Open browser console (Press F12)
2. Look for errors (likely CORS error)
3. Verify backend `FRONTEND_URL` is set correctly
4. Verify you redeployed backend after updating variable
5. Hard refresh browser (Ctrl + F5)

### Issue: "Failed to fetch" error
**Solution**:
1. Check `NEXT_PUBLIC_API_URL` is correct
2. Test backend health: `https://your-backend.vercel.app/health`
3. Verify backend is responding
4. Redeploy frontend if you changed the variable

### Issue: CORS error in browser console
**Solution**:
```
Access to fetch at 'https://backend...' from origin 'https://frontend...' 
has been blocked by CORS policy
```
This means:
1. Backend `FRONTEND_URL` doesn't match actual frontend URL
2. Go to backend project → Settings → Environment Variables
3. Update `FRONTEND_URL` to exact frontend URL (no trailing slash)
4. Redeploy backend
5. Hard refresh frontend (Ctrl + F5)

### Issue: Build failed
**Solution**:
1. Go to Deployments tab
2. Click on failed deployment
3. Check build logs
4. Common issues:
   - Root directory not set to `frontend`
   - Missing `NEXT_PUBLIC_API_URL` variable
5. Fix and redeploy

### Issue: Application loads but looks broken
**Solution**:
1. Hard refresh browser (Ctrl + F5)
2. Clear browser cache
3. Check browser console for errors
4. Verify all assets loaded (Network tab in F12)

---

## 💰 Cost Information

**Vercel Free Tier for Frontend**:
- ✅ Unlimited websites
- ✅ 100 GB bandwidth
- ✅ Global CDN
- ✅ Automatic HTTPS
- ✅ Custom domains

**Total Monthly Cost** (all services):
- Frontend: $0 (free tier)
- Backend: $0 (free tier)
- RunPod: $5-20 (pay per use)
- **Total**: ~$5-20/month for low-medium traffic

---

## 🎨 Customization Tips

### Change UI Colors

Edit `frontend/tailwind.config.js` in your repository:

```javascript
colors: {
  primary: {
    500: '#0ea5e9',  // Change this!
    600: '#0284c7',  // And this!
  }
}
```

Then commit and push - Vercel will auto-deploy!

### Change Title

Edit `frontend/src/app/layout.tsx`:

```typescript
export const metadata: Metadata = {
  title: 'Your Custom Title',
  description: 'Your description',
}
```

### Add Custom Domain

1. Buy a domain (Namecheap, GoDaddy, etc.)
2. In Vercel frontend project → Settings → Domains
3. Add your domain
4. Follow DNS instructions
5. Wait for SSL certificate (automatic)

---

## 🚀 You're Done! 🎉

### What You've Accomplished:

✅ Deployed GPU model service on RunPod  
✅ Deployed FastAPI backend on Vercel  
✅ Deployed Next.js frontend on Vercel  
✅ Connected all three services  
✅ Application is live and accessible worldwide!

---

## 📊 Your Architecture

```
User's Browser
     ↓
Frontend (Vercel Global CDN)
https://your-frontend.vercel.app
     ↓
Backend (Vercel Serverless)
https://your-backend.vercel.app
     ↓
Model Service (RunPod GPU)
https://api.runpod.ai/v2/YOUR_ID/runsync
```

---

## 🎯 Next Steps

### Immediate:
- [ ] Test with various images
- [ ] Share URL with others
- [ ] Monitor usage in dashboards

### Optional:
- [ ] Set up custom domain
- [ ] Add Google Analytics
- [ ] Monitor costs (Vercel + RunPod dashboards)
- [ ] Customize UI colors/branding

### Advanced:
- [ ] Add user authentication
- [ ] Add processing history
- [ ] Add batch processing
- [ ] Scale RunPod workers for high traffic

---

## 📱 Mobile Access

Your application works great on mobile too!

- Visit the frontend URL on your phone
- Add to home screen for app-like experience
- Fully responsive design

---

## 📈 Monitoring

### Check Logs:

**Frontend**:
- Vercel Dashboard → Frontend Project → Logs

**Backend**:
- Vercel Dashboard → Backend Project → View Function Logs

**Model**:
- RunPod Dashboard → Your Endpoint → Logs

### Monitor Costs:

**Vercel**:
- Dashboard → Your Account → Usage

**RunPod**:
- Dashboard → Billing → Usage

---

## ✅ Final Checklist

- [ ] Frontend deployed and accessible
- [ ] Backend deployed and responding
- [ ] Model service running on RunPod
- [ ] Can upload and process images
- [ ] Text extraction works
- [ ] No CORS errors
- [ ] All URLs saved
- [ ] Application shared with others

---

## 🆘 Still Having Issues?

### Check All Environment Variables:

**Backend** should have:
```
RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/YOUR_ID/runsync
RUNPOD_API_KEY=your_key_here
FRONTEND_URL=https://your-frontend.vercel.app
```

**Frontend** should have:
```
NEXT_PUBLIC_API_URL=https://your-backend.vercel.app
```

### Test Each Service:

1. **Model**: Check RunPod dashboard - should be "Ready" or "Idle"
2. **Backend**: Visit `https://your-backend.vercel.app/health`
3. **Frontend**: Visit your frontend URL - page should load

### Common Fix:

If nothing works:
1. Redeploy backend (Deployments → Redeploy)
2. Redeploy frontend (Deployments → Redeploy)
3. Wait 2 minutes
4. Hard refresh browser (Ctrl + F5)
5. Try again

---

## 📞 Support Resources

- **Vercel Docs**: https://vercel.com/docs
- **Next.js Docs**: https://nextjs.org/docs
- **RunPod Docs**: https://docs.runpod.io/
- **Your Repository**: https://github.com/abhijit-aiplanet/Arabic_OCR

---

## 🎊 Congratulations!

You've successfully deployed a production-ready, globally-distributed OCR application!

**Your live application**: `https://your-frontend-url.vercel.app`

**Share it with the world!** 🌍

---

**Last Updated**: November 2025  
**Repository**: https://github.com/abhijit-aiplanet/Arabic_OCR  
**Deployment**: Complete ✅

