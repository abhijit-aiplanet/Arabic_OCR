"""
FastAPI Backend for AIN OCR Application
Handles API requests and communicates with the model service on RunPod
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import httpx
import os
from PIL import Image
import io
import base64
from dotenv import load_dotenv
import traceback

load_dotenv()

app = FastAPI(
    title="AIN OCR API",
    description="Backend API for Arabic OCR using AIN Vision Language Model",
    version="1.0.0"
)

# CORS middleware - Configure this properly for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        os.getenv("FRONTEND_URL", "*"),  # Production frontend URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RunPod Configuration
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT_URL")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
TIMEOUT_SECONDS = 120

# Default OCR Prompt
DEFAULT_OCR_PROMPT = """Extract all text from this image exactly as it appears. 

Requirements:
1. Extract ONLY the text content - do not describe, analyze, or interpret the image
2. Maintain the original text structure, layout, and formatting
3. Preserve line breaks, paragraphs, and spacing as they appear
4. Do not translate the text - keep it in its original language
5. Do not add any explanations, descriptions, or additional commentary
6. If there are tables, maintain their structure
7. If there are headers, titles, or sections, preserve their hierarchy

Output only the extracted text, nothing else."""


class OCRRequest(BaseModel):
    """Request model for OCR processing"""
    custom_prompt: Optional[str] = None
    max_new_tokens: int = 2048
    min_pixels: Optional[int] = 200704  # 256 * 28 * 28
    max_pixels: Optional[int] = 1003520  # 1280 * 28 * 28


class OCRResponse(BaseModel):
    """Response model for OCR processing"""
    extracted_text: str
    status: str
    error: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AIN OCR API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_service": "configured" if RUNPOD_ENDPOINT else "not_configured"
    }


@app.post("/api/ocr", response_model=OCRResponse)
async def process_ocr(
    file: UploadFile = File(...),
    custom_prompt: Optional[str] = Form(None),
    max_new_tokens: int = Form(2048),
    min_pixels: Optional[int] = Form(200704),
    max_pixels: Optional[int] = Form(1003520),
):
    """
    Process an image and extract text using OCR.
    
    Args:
        file: Image file to process
        custom_prompt: Optional custom prompt (uses default if not provided)
        max_new_tokens: Maximum tokens to generate
        min_pixels: Minimum image resolution
        max_pixels: Maximum image resolution
        
    Returns:
        OCRResponse with extracted text and status
    """
    try:
        # Validate RunPod configuration
        if not RUNPOD_ENDPOINT:
            raise HTTPException(
                status_code=500,
                detail="Model service endpoint not configured. Please set RUNPOD_ENDPOINT_URL environment variable."
            )
        
        # Read and validate image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Prepare request payload for RunPod
        prompt = custom_prompt if custom_prompt and custom_prompt.strip() else DEFAULT_OCR_PROMPT
        
        payload = {
            "input": {
                "image": img_base64,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels
            }
        }
        
        # Call RunPod endpoint
        headers = {
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(
                RUNPOD_ENDPOINT,
                json=payload,
                headers=headers
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Model service error: {response.text}"
                )
            
            result = response.json()
            
            # Extract text from response
            # RunPod response format: {"output": {"text": "extracted text"}, "status": "COMPLETED"}
            if result.get("status") == "COMPLETED":
                extracted_text = result.get("output", {}).get("text", "")
                if not extracted_text:
                    extracted_text = "No text extracted from image"
                
                return OCRResponse(
                    extracted_text=extracted_text,
                    status="success"
                )
            else:
                return OCRResponse(
                    extracted_text="",
                    status="error",
                    error=result.get("error", "Unknown error from model service")
                )
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing OCR request: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/api/prompt")
async def get_default_prompt():
    """Get the default OCR prompt"""
    return {
        "default_prompt": DEFAULT_OCR_PROMPT
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )

