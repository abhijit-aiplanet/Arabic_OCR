# 📁 Project Structure

Complete overview of the AIN OCR project structure after refactoring.

## Directory Layout

```
Dots-OCR/
│
├── 📁 backend/                      # FastAPI Backend Service
│   ├── main.py                      # Main FastAPI application
│   ├── requirements.txt             # Python dependencies
│   ├── vercel.json                  # Vercel deployment configuration
│   ├── env.example                  # Example environment variables
│   └── README.md                    # Backend documentation
│
├── 📁 frontend/                     # Next.js Frontend Application
│   ├── 📁 src/
│   │   ├── 📁 app/                 # Next.js App Router
│   │   │   ├── layout.tsx          # Root layout component
│   │   │   └── page.tsx            # Home page
│   │   │
│   │   ├── 📁 components/          # React Components
│   │   │   ├── ImageUploader.tsx   # Image upload component
│   │   │   ├── ExtractedText.tsx   # Text display component
│   │   │   └── AdvancedSettings.tsx # Settings panel
│   │   │
│   │   ├── 📁 lib/                 # Utilities & API
│   │   │   └── api.ts              # API client functions
│   │   │
│   │   └── 📁 styles/              # Global Styles
│   │       └── globals.css         # Global CSS with Tailwind
│   │
│   ├── 📁 public/                  # Static Assets
│   ├── package.json                # Node.js dependencies
│   ├── tsconfig.json               # TypeScript configuration
│   ├── next.config.js              # Next.js configuration
│   ├── tailwind.config.js          # Tailwind CSS configuration
│   ├── postcss.config.js           # PostCSS configuration
│   ├── env.local.example           # Example environment variables
│   ├── .gitignore                  # Git ignore rules
│   └── README.md                   # Frontend documentation
│
├── 📁 model-service/               # RunPod Model Service
│   ├── handler.py                  # RunPod serverless handler
│   ├── requirements.txt            # Python dependencies
│   ├── Dockerfile                  # Docker configuration
│   ├── test_handler.py             # Local testing script
│   └── README.md                   # Model service documentation
│
├── 📁 image/                       # Example Images (from original)
│   └── 📁 app/
│       ├── 1762329983969.png
│       ├── 1762330009302.png
│       └── 1762330020168.png
│
├── 📄 DEPLOYMENT_GUIDE.md          # Complete deployment guide
├── 📄 SETUP_INSTRUCTIONS.txt       # Quick setup checklist
├── 📄 PROJECT_STRUCTURE.md         # This file
├── 📄 README.md                    # Main project README
│
└── 📁 (Original files - can be archived)
    ├── ain_app.py                  # Original Gradio app
    ├── app.py                      # Original app
    ├── deepseek_app.py            # Alternative implementation
    ├── arabic_corrector.py        # Arabic correction module
    └── requirements.txt            # Original requirements
```

## Components Description

### Backend (FastAPI)

**Purpose**: RESTful API service that acts as middleware between frontend and model service.

**Key Files**:
- `main.py`: Core API with endpoints for OCR processing, health checks, and configuration
- `requirements.txt`: FastAPI, httpx, Pillow, and other dependencies
- `vercel.json`: Configuration for Vercel serverless deployment
- `env.example`: Template for environment variables

**Key Features**:
- CORS middleware for frontend communication
- Image validation and preprocessing
- Async communication with RunPod model service
- Error handling and timeout management

**API Endpoints**:
- `POST /api/ocr`: Process image and extract text
- `GET /api/prompt`: Get default OCR prompt
- `GET /health`: Health check endpoint
- `GET /`: API information

### Frontend (Next.js + TypeScript)

**Purpose**: Modern, responsive web interface for image upload and text extraction.

**Tech Stack**:
- Next.js 14 (App Router)
- TypeScript
- Tailwind CSS
- React Dropzone
- Axios
- React Hot Toast

**Key Components**:

1. **ImageUploader.tsx**
   - Drag & drop image upload
   - Image preview
   - File type validation

2. **ExtractedText.tsx**
   - Display extracted text
   - Copy to clipboard functionality
   - Character and word count

3. **AdvancedSettings.tsx**
   - Collapsible settings panel
   - Custom prompt input
   - Resolution and token configuration

4. **api.ts**
   - API client functions
   - HTTP request handling
   - Error management

**Key Features**:
- Responsive design (mobile, tablet, desktop)
- Real-time processing status
- Toast notifications
- Arabic text support (RTL)
- Modern gradient UI

### Model Service (RunPod)

**Purpose**: GPU-powered inference service running the MBZUAI/AIN model.

**Key Files**:
- `handler.py`: RunPod serverless handler implementing the model inference
- `Dockerfile`: Container configuration for deployment
- `test_handler.py`: Local testing without deploying
- `requirements.txt`: Model dependencies (transformers, torch, etc.)

**Key Features**:
- Automatic model loading with error recovery
- Base64 image processing
- GPU optimization
- Configurable inference parameters
- Error handling and logging

**Model Details**:
- Model: MBZUAI/AIN (Vision Language Model)
- Size: ~20GB
- VRAM Required: 20-25GB
- Inference Time: 3-5 seconds per image

## Data Flow

```
1. User uploads image in Frontend
   ↓
2. Frontend sends to Backend API (multipart/form-data)
   ↓
3. Backend validates and converts image to base64
   ↓
4. Backend sends request to RunPod Model Service
   ↓
5. Model Service processes image with AIN VLM
   ↓
6. Model Service returns extracted text
   ↓
7. Backend forwards response to Frontend
   ↓
8. Frontend displays extracted text to user
```

## Deployment Architecture

```
┌─────────────────────────────────────────┐
│         User's Browser                   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Frontend (Vercel Edge Network)        │
│   • Next.js Static + Server Components  │
│   • Global CDN Distribution              │
│   • Auto HTTPS                           │
└──────────────┬──────────────────────────┘
               │ HTTPS API Call
               ▼
┌─────────────────────────────────────────┐
│   Backend (Vercel Serverless)           │
│   • FastAPI on AWS Lambda               │
│   • Auto-scaling                         │
│   • 10s request timeout                  │
└──────────────┬──────────────────────────┘
               │ HTTPS API Call
               ▼
┌─────────────────────────────────────────┐
│   Model Service (RunPod GPU)            │
│   • RTX A6000 / A40 GPU                 │
│   • Serverless or Dedicated Pod         │
│   • Auto-scaling workers                 │
│   • Container-based deployment          │
└─────────────────────────────────────────┘
```

## Environment Variables

### Backend
```env
RUNPOD_ENDPOINT_URL=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync
RUNPOD_API_KEY=your_runpod_api_key
FRONTEND_URL=https://your-frontend.vercel.app
PORT=8000
```

### Frontend
```env
NEXT_PUBLIC_API_URL=https://your-backend.vercel.app
```

### Model Service
No environment variables needed - fully configured via API calls.

## Technology Stack

### Backend
- **Framework**: FastAPI (Python)
- **HTTP Client**: httpx (async)
- **Image Processing**: Pillow
- **Server**: Uvicorn
- **Deployment**: Vercel Serverless Functions

### Frontend
- **Framework**: Next.js 14
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **HTTP Client**: Axios
- **File Upload**: React Dropzone
- **Notifications**: React Hot Toast
- **Icons**: Lucide React
- **Deployment**: Vercel Edge Network

### Model Service
- **Framework**: RunPod Serverless
- **Model**: Qwen2VL (MBZUAI/AIN)
- **ML Libraries**: Transformers, PyTorch
- **Image Processing**: Pillow
- **Container**: Docker
- **Deployment**: RunPod GPU Pods

## Security Features

- ✅ CORS configured for specific origins
- ✅ Environment variables for sensitive data
- ✅ File type validation
- ✅ Size limits on uploads
- ✅ HTTPS everywhere
- ✅ No sensitive data in client code
- ✅ API key authentication for model service

## Performance Optimizations

### Frontend
- Next.js SSR and SSG
- Code splitting
- Image optimization
- CDN delivery
- Lazy loading components

### Backend
- Async request handling
- Connection pooling
- Timeout management
- Error recovery

### Model Service
- GPU acceleration
- Model weight caching
- Efficient tokenization
- Batch processing capability

## Scalability

### Current Capacity
- Frontend: Unlimited (CDN)
- Backend: Auto-scales (Vercel)
- Model: 1-5 workers (configurable)

### For High Traffic
1. Increase RunPod max workers (10-20)
2. Use dedicated GPU pods
3. Add request queue (Redis)
4. Implement caching layer
5. Enable auto-scaling on all services

## Monitoring & Logging

### Available Logs
- **Frontend**: Vercel Dashboard → Logs
- **Backend**: Vercel Dashboard → Logs
- **Model Service**: RunPod Dashboard → Endpoint Logs

### Recommended Monitoring
- Vercel Analytics (built-in)
- RunPod usage dashboard
- Custom error tracking (Sentry)
- Uptime monitoring (UptimeRobot)

## Future Enhancements

Potential improvements:
- [ ] Batch processing support
- [ ] PDF and multi-page document support
- [ ] User authentication and accounts
- [ ] Processing history and saved results
- [ ] Multiple language support in UI
- [ ] Export formats (JSON, CSV, PDF)
- [ ] API rate limiting
- [ ] Webhook notifications
- [ ] Custom model fine-tuning
- [ ] Mobile app (React Native)

## Migration from Original

### Changes from Original Application
- ✅ Separated Gradio UI → Modern Next.js frontend
- ✅ Monolithic app → Microservices architecture
- ✅ Local deployment → Cloud deployment
- ✅ Single service → Three independent services
- ✅ Direct model loading → API-based model service

### Preserved Features
- ✅ AIN VLM model for OCR
- ✅ Custom prompt support
- ✅ Advanced settings
- ✅ Arabic text support
- ✅ Example images

### New Features
- ✅ Modern responsive UI
- ✅ Drag & drop upload
- ✅ Real-time notifications
- ✅ Better error handling
- ✅ Scalable architecture
- ✅ Production-ready deployment

## Getting Started

1. **Read Documentation**
   - `SETUP_INSTRUCTIONS.txt` - Quick start
   - `DEPLOYMENT_GUIDE.md` - Detailed deployment steps
   - Component READMEs - Specific guides

2. **Deploy Services** (in order)
   - Model Service (RunPod)
   - Backend (Vercel/Render)
   - Frontend (Vercel)

3. **Configure & Test**
   - Set environment variables
   - Test each service independently
   - Test full integration

4. **Monitor & Maintain**
   - Check logs regularly
   - Monitor costs
   - Update dependencies

---

For detailed deployment instructions, see `DEPLOYMENT_GUIDE.md`

For quick setup, see `SETUP_INSTRUCTIONS.txt`

