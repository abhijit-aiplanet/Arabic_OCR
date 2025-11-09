# 🔍 AIN OCR - Vision Language Model for Text Extraction

<div align="center">

![AIN OCR](https://img.shields.io/badge/AIN-OCR-blue?style=for-the-badge)
![Next.js](https://img.shields.io/badge/Next.js-14-black?style=for-the-badge&logo=next.js)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch)

Advanced OCR application using the MBZUAI/AIN Vision Language Model for accurate text extraction from images.

[Features](#features) • [Architecture](#architecture) • [Quick Start](#quick-start) • [Deployment](#deployment) • [Documentation](#documentation)

</div>

---

## 📖 Overview

AIN OCR is a production-ready OCR application that leverages the power of Vision Language Models (VLM) for superior text extraction. Unlike traditional OCR systems, it understands context and can handle handwritten text, complex layouts, and multiple languages with high accuracy.

### Why AIN OCR?

- **🎯 High Accuracy**: Uses MBZUAI/AIN model, specialized for understanding text in context
- **🌍 Multi-language**: Optimized for Arabic, supports many other languages
- **🚀 Production Ready**: Separated frontend/backend/model architecture for scalability
- **💰 Cost Effective**: Serverless deployment with pay-per-use model
- **🎨 Modern UI**: Beautiful, responsive interface built with Next.js and Tailwind CSS
- **📱 Mobile Friendly**: Works seamlessly on all device sizes

## ✨ Features

### Core Features
- ✅ Advanced text extraction using Vision Language Models
- ✅ Support for handwritten and printed text
- ✅ Multi-language support (Arabic, English, and more)
- ✅ Maintains original text structure and formatting
- ✅ Configurable inference parameters
- ✅ Custom prompt support for specific extraction needs

### User Interface
- ✅ Drag & drop image upload
- ✅ Real-time processing status
- ✅ One-click copy to clipboard
- ✅ Character and word count
- ✅ Advanced settings panel
- ✅ Toast notifications
- ✅ Responsive design

### Technical Features
- ✅ RESTful API architecture
- ✅ Async request handling
- ✅ GPU-accelerated inference
- ✅ Auto-scaling capabilities
- ✅ Error handling and recovery
- ✅ Health check endpoints
- ✅ CORS support

## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend      │  ← Next.js 14 on Vercel
│   (TypeScript)  │     • Modern UI with Tailwind CSS
│   Port: 3000    │     • Drag & drop upload
└────────┬────────┘     • Real-time notifications
         │
         │ HTTPS
         ▼
┌─────────────────┐
│   Backend       │  ← FastAPI on Vercel/Render
│   (Python)      │     • RESTful API
│   Port: 8000    │     • Image validation
└────────┬────────┘     • Request orchestration
         │
         │ HTTPS
         ▼
┌─────────────────┐
│  Model Service  │  ← MBZUAI/AIN on RunPod
│  (PyTorch)      │     • GPU inference (RTX A6000)
│  GPU Required   │     • Serverless scaling
└─────────────────┘     • 3-5s per image
```

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Frontend** | Next.js 14, TypeScript, Tailwind CSS | Modern web interface |
| **Backend** | FastAPI, Python, httpx | API orchestration |
| **Model Service** | PyTorch, Transformers, RunPod | GPU inference |
| **Deployment** | Vercel, RunPod | Cloud hosting |

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ (for frontend)
- Python 3.10+ (for backend)
- RunPod account with GPU credits
- Vercel account (free tier works)

### 1. Clone Repository

```bash
git clone https://github.com/your-username/Dots-OCR.git
cd Dots-OCR
```

### 2. Local Development

#### Backend

```bash
cd backend
pip install -r requirements.txt

# Create .env file
cp env.example .env
# Edit .env with your RunPod credentials

# Run server
python main.py
```

Backend runs on `http://localhost:8000`

#### Frontend

```bash
cd frontend
npm install

# Create .env.local file
cp env.local.example .env.local
# Edit .env.local with backend URL

# Run dev server
npm run dev
```

Frontend runs on `http://localhost:3000`

#### Model Service (Optional - for local testing)

```bash
cd model-service
pip install -r requirements.txt

# Test handler
python test_handler.py
```

### 3. Deploy to Production

Follow the comprehensive deployment guide:

```bash
# Read the deployment guide
cat DEPLOYMENT_GUIDE.md

# Quick checklist
cat SETUP_INSTRUCTIONS.txt
```

**Deployment Order:**
1. 🎯 Deploy Model Service on RunPod (15-20 min)
2. 🔧 Deploy Backend on Vercel (5-10 min)
3. 🎨 Deploy Frontend on Vercel (5 min)
4. ✅ Configure environment variables and test

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Complete deployment instructions |
| [SETUP_INSTRUCTIONS.txt](SETUP_INSTRUCTIONS.txt) | Quick setup checklist |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Project organization details |
| [backend/README.md](backend/README.md) | Backend API documentation |
| [frontend/README.md](frontend/README.md) | Frontend development guide |
| [model-service/README.md](model-service/README.md) | Model service setup |

## 🔧 Configuration

### Backend Environment Variables

```env
RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync
RUNPOD_API_KEY=your_runpod_api_key_here
FRONTEND_URL=https://your-frontend.vercel.app
PORT=8000
```

### Frontend Environment Variables

```env
NEXT_PUBLIC_API_URL=https://your-backend.vercel.app
```

## 💰 Cost Estimation

### Low-Medium Traffic (~1000 images/month)

- **RunPod (Serverless)**: $5-$20/month
  - RTX A6000: ~$0.45/hour when active
  - ~$0.0006-$0.001 per image
  - Only pay for actual compute time
  
- **Vercel (Frontend + Backend)**: $0-$20/month
  - Free tier: 100 GB-hours serverless execution
  - Usually sufficient for small-medium projects
  
- **Total**: ~$5-$40/month

### High Traffic

- Increase RunPod max workers (5-10)
- Vercel Pro plan: $20/month
- Consider dedicated GPU pods for consistency
- Estimated: $50-$200/month

## 📊 Performance

- **Frontend Load Time**: < 1 second (CDN cached)
- **API Response Time**: 3-5 seconds (inference time)
- **Model Inference**: 3-5 seconds per image
- **Throughput**: 10-20 images/minute (serverless)
- **Scalability**: Auto-scales with demand

## 🖼️ Supported Image Formats

- PNG
- JPEG/JPG
- GIF
- WebP
- BMP

**Recommended**: PNG or JPEG, RGB mode, 300+ DPI for best results

## 🌐 API Endpoints

### POST `/api/ocr`
Process an image and extract text.

**Request:**
```bash
curl -X POST https://your-backend.vercel.app/api/ocr \
  -F "file=@image.png" \
  -F "max_new_tokens=2048" \
  -F "min_pixels=200704" \
  -F "max_pixels=1003520"
```

**Response:**
```json
{
  "extracted_text": "Text content from image...",
  "status": "success",
  "error": null
}
```

### GET `/health`
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "model_service": "configured"
}
```

### GET `/api/prompt`
Get the default OCR prompt.

## 🔐 Security

- ✅ HTTPS everywhere
- ✅ Environment variables for secrets
- ✅ CORS configured
- ✅ File type validation
- ✅ Size limits on uploads
- ✅ API key authentication
- ✅ No sensitive data in client code

## 🧪 Testing

### Backend Tests
```bash
cd backend
# Start server
python main.py

# Test health endpoint
curl http://localhost:8000/health
```

### Frontend Tests
```bash
cd frontend
npm run dev
# Open http://localhost:3000 and test UI
```

### Model Service Tests
```bash
cd model-service
python test_handler.py
```

## 🐛 Troubleshooting

### Common Issues

**CORS Errors**
- Ensure `FRONTEND_URL` in backend matches actual frontend URL exactly
- Redeploy backend after changing environment variables

**Timeout Errors**
- Increase RunPod max workers
- Check model service logs in RunPod dashboard
- Verify GPU has sufficient VRAM (24GB+)

**Out of Memory**
- Use GPU with more VRAM (A40 or A6000)
- Reduce `max_pixels` parameter
- Reduce `max_new_tokens` parameter

**Model Not Loading**
- Check container disk size (needs 50GB+)
- Verify RunPod endpoint is active
- Check RunPod logs for errors

## 📈 Scaling Guide

### For Increased Traffic

1. **Increase RunPod Workers**: 1 → 5-10 workers
2. **Upgrade Vercel**: Free → Pro plan ($20/month)
3. **Add Caching**: Implement Redis for response caching
4. **Use Dedicated Pods**: Switch from serverless to dedicated for consistency
5. **Load Balancing**: Multiple RunPod endpoints with round-robin

### Monitoring

- Use Vercel Analytics (built-in)
- Monitor RunPod usage in dashboard
- Set up Sentry for error tracking
- Add UptimeRobot for uptime monitoring

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MBZUAI** for the AIN Vision Language Model
- **Qwen2-VL** for the base architecture
- **RunPod** for GPU infrastructure
- **Vercel** for serverless deployment platform

## 📞 Support

- 📧 Email: your-email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/Dots-OCR/issues)
- 📖 Docs: See [Documentation](#documentation) section

## 🗺️ Roadmap

- [ ] Batch processing support
- [ ] PDF and multi-page document support
- [ ] User authentication and accounts
- [ ] Processing history
- [ ] Multiple export formats (JSON, CSV, PDF)
- [ ] Webhook notifications
- [ ] Mobile app (React Native)
- [ ] Custom model fine-tuning interface

---

<div align="center">

**Made with ❤️ using Vision Language Models**

[Report Bug](https://github.com/your-username/Dots-OCR/issues) • [Request Feature](https://github.com/your-username/Dots-OCR/issues)

</div>

