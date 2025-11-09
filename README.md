# 🔍 AIN OCR - Arabic Text Extraction

Advanced OCR application using the MBZUAI/AIN Vision Language Model for accurate text extraction from images.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🚀 Quick Deploy (30 minutes)

Deploy your OCR application in 3 simple steps:

1. **Model Service** → RunPod (15 min)
2. **Backend API** → Vercel (10 min)
3. **Frontend UI** → Vercel (5 min)

**👉 Start Here: [DEPLOY_START_HERE.md](DEPLOY_START_HERE.md)**

---

## ✨ Features

- 🎯 **High Accuracy** - Vision Language Model for context-aware text extraction
- 🌍 **Multi-language** - Optimized for Arabic, supports many languages
- 🎨 **Modern UI** - Beautiful, responsive Next.js interface
- 📱 **Mobile Ready** - Works seamlessly on all devices
- 💰 **Cost Effective** - Serverless deployment, pay only for usage (~$5-20/month)
- 🚀 **Production Ready** - Auto-scaling microservices architecture

---

## 📁 Project Structure

```
Arabic_OCR/
├── backend/              # FastAPI backend service
├── frontend/             # Next.js web application
├── model-service/        # RunPod GPU service
└── DEPLOY_*.md          # Step-by-step deployment guides
```

---

## 🎯 Deployment Guides

| Guide | Service | Platform | Time |
|-------|---------|----------|------|
| [DEPLOY_1_RUNPOD.md](DEPLOY_1_RUNPOD.md) | Model | RunPod | 15 min |
| [DEPLOY_2_BACKEND_VERCEL.md](DEPLOY_2_BACKEND_VERCEL.md) | Backend | Vercel | 10 min |
| [DEPLOY_3_FRONTEND_VERCEL.md](DEPLOY_3_FRONTEND_VERCEL.md) | Frontend | Vercel | 5 min |

---

## 💰 Cost Estimate

- **RunPod GPU**: $5-20/month (pay per use)
- **Vercel**: Free tier (usually sufficient)
- **Total**: ~$5-20/month for low-medium traffic

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Next.js 14, TypeScript, Tailwind CSS |
| **Backend** | FastAPI, Python |
| **Model** | MBZUAI/AIN (Qwen2-VL based) |
| **Deployment** | Vercel + RunPod |

---

## 📖 Quick Start

### Prerequisites

- RunPod account with credits
- Vercel account (free)
- GitHub account
- 30-40 minutes

### Deploy Now

```bash
# 1. Clone this repository (if needed)
git clone https://github.com/abhijit-aiplanet/Arabic_OCR.git
cd Arabic_OCR

# 2. Follow the deployment guides in order
# Start with: DEPLOY_START_HERE.md
```

---

## 🎨 Example Images

Test images are available in the `image/app/` directory.

---

## 📞 Support

- **Documentation**: See `DEPLOY_*.md` files
- **Issues**: [GitHub Issues](https://github.com/abhijit-aiplanet/Arabic_OCR/issues)

---

## 📄 License

MIT License - feel free to use for personal or commercial projects.

---

## 🙏 Acknowledgments

- **MBZUAI** for the AIN Vision Language Model
- **Qwen2-VL** for the base architecture
- **RunPod** for GPU infrastructure
- **Vercel** for serverless deployment

---

**Ready to deploy?** 👉 [Start with DEPLOY_START_HERE.md](DEPLOY_START_HERE.md)
