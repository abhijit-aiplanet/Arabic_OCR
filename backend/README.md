# AIN OCR Backend API

FastAPI backend service for the AIN OCR application.

## Features

- RESTful API for OCR processing
- Integration with RunPod model service
- CORS support for frontend integration
- Error handling and validation
- Health check endpoints

## Setup

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file from `.env.example`:
```bash
cp .env.example .env
```

3. Configure environment variables in `.env`:
   - `RUNPOD_ENDPOINT_URL`: Your RunPod endpoint URL
   - `RUNPOD_API_KEY`: Your RunPod API key
   - `FRONTEND_URL`: Your frontend URL (for CORS)

4. Run the development server:
```bash
python main.py
# or
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Deployment

### Vercel Deployment

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Login to Vercel:
```bash
vercel login
```

3. Deploy:
```bash
vercel
```

4. Set environment variables in Vercel dashboard:
   - `RUNPOD_ENDPOINT_URL`
   - `RUNPOD_API_KEY`
   - `FRONTEND_URL`

### Render Deployment

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables in Render dashboard

## API Endpoints

### POST /api/ocr
Process an image and extract text.

**Request:**
- `file`: Image file (multipart/form-data)
- `custom_prompt`: Optional custom prompt (string)
- `max_new_tokens`: Maximum tokens to generate (int, default: 2048)
- `min_pixels`: Minimum image resolution (int, default: 200704)
- `max_pixels`: Maximum image resolution (int, default: 1003520)

**Response:**
```json
{
  "extracted_text": "Extracted text from image",
  "status": "success",
  "error": null
}
```

### GET /api/prompt
Get the default OCR prompt.

### GET /health
Health check endpoint.

### GET /
Root endpoint with API information.

