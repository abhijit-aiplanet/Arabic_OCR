"""
FastAPI Backend for AIN OCR Application
Handles API requests and communicates with the model service on RunPod
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import httpx
import os
from PIL import Image
import io
import base64
from dotenv import load_dotenv
import traceback
import asyncio
import fitz  # PyMuPDF
import json
from datetime import datetime
import uuid

load_dotenv()

# Vercel Blob Storage Configuration
BLOB_READ_WRITE_TOKEN = os.getenv("BLOB_READ_WRITE_TOKEN")
BLOB_API_URL = "https://blob.vercel-storage.com"


async def store_file_to_blob(file_data: bytes, filename: str, content_type: str = "application/octet-stream") -> dict:
    """
    Store a file in Vercel Blob Storage.
    
    Args:
        file_data: The file content as bytes
        filename: Original filename
        content_type: MIME type of the file
        
    Returns:
        dict with 'url' and 'pathname' if successful, or 'error' if failed
    """
    if not BLOB_READ_WRITE_TOKEN:
        print("⚠️ BLOB_READ_WRITE_TOKEN not configured, skipping file storage")
        return {"error": "Blob storage not configured", "stored": False}
    
    try:
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # Extract extension from original filename
        ext = filename.split('.')[-1] if '.' in filename else 'bin'
        stored_filename = f"uploads/{timestamp}_{unique_id}.{ext}"
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.put(
                f"{BLOB_API_URL}/{stored_filename}",
                content=file_data,
                headers={
                    "Authorization": f"Bearer {BLOB_READ_WRITE_TOKEN}",
                    "x-content-type": content_type,
                    "access": "public"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ File stored successfully: {result.get('url', stored_filename)}")
                return {
                    "stored": True,
                    "url": result.get("url"),
                    "pathname": result.get("pathname", stored_filename),
                    "original_filename": filename
                }
            else:
                print(f"❌ Failed to store file: {response.status_code} - {response.text}")
                return {"error": f"Storage failed: {response.status_code}", "stored": False}
                
    except Exception as e:
        print(f"❌ Error storing file to blob: {str(e)}")
        return {"error": str(e), "stored": False}

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
        "https://arabic-ocr-frontend-beryl.vercel.app",  # Production frontend
        os.getenv("FRONTEND_URL", "*"),  # Additional frontend URL from env
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RunPod Configuration
RUNPOD_ENDPOINT = os.getenv("RUNPOD_ENDPOINT_URL")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
TIMEOUT_SECONDS = 600  # 10 minutes - enough for cold starts and large documents

# Default OCR Prompt
DEFAULT_OCR_PROMPT = """Extract all Arabic text from this image with maximum accuracy.

CRITICAL: This image contains ARABIC text that may be:
- Handwritten Arabic text
- Typed/printed Arabic text
- Small Arabic annotations or notes
- Arabic text in forms, tables, or documents

Your task: Find and extract EVERY piece of Arabic text visible in the image, no matter how small.

Requirements:
1. Extract ALL Arabic text - handwritten, typed, printed, or annotations
2. Accuracy is CRITICAL - extract Arabic text exactly as it appears
3. Do NOT miss any Arabic text, even small notes or annotations
4. Maintain the original text structure, layout, and formatting
5. Preserve line breaks, paragraphs, and spacing as they appear
6. Do NOT translate - keep all text in Arabic
7. Do NOT add descriptions, interpretations, or commentary
8. If there are tables or forms, maintain their structure
9. If there are headers, titles, or sections, preserve their hierarchy
10. Focus on ACCURACY over speed

Output only the extracted Arabic text, nothing else."""


class OCRRequest(BaseModel):
    """Request model for OCR processing"""
    custom_prompt: Optional[str] = None
    max_new_tokens: int = 4096  # Balanced for speed and capacity
    min_pixels: Optional[int] = 200704  # 256 * 28 * 28
    max_pixels: Optional[int] = 1003520  # 1280 * 28 * 28 - Reduced for faster processing


class OCRResponse(BaseModel):
    """Response model for OCR processing"""
    extracted_text: str
    status: str
    error: Optional[str] = None


class PDFPageResult(BaseModel):
    """Result for a single PDF page"""
    page_number: int
    extracted_text: str
    status: str
    error: Optional[str] = None
    page_image: Optional[str] = None  # Base64 encoded image


class PDFOCRResponse(BaseModel):
    """Response model for PDF OCR processing"""
    total_pages: int
    results: List[PDFPageResult]
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
        
        # Store file to Vercel Blob (async, non-blocking)
        storage_result = await store_file_to_blob(
            contents,
            file.filename or "image.png",
            file.content_type or "image/png"
        )
        if storage_result.get("stored"):
            print(f"📁 Image stored: {storage_result.get('url')}")
        
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
            
            # Debug: Print the actual response
            print(f"🔍 RunPod response: {result}")
            print(f"🔍 Response status: {result.get('status')}")
            print(f"🔍 Response output: {result.get('output')}")
            
            # Handle IN_QUEUE status - poll for results
            if result.get("status") == "IN_QUEUE":
                job_id = result.get("id")
                print(f"⏳ Job in queue, polling for results. Job ID: {job_id}")
                
                # Poll the status endpoint
                status_url = RUNPOD_ENDPOINT.replace("/runsync", f"/status/{job_id}")
                print(f"🔄 Polling status URL: {status_url}")
                
                max_polls = 180  # Poll for up to 6 minutes (180 × 2 = 360 seconds)
                poll_interval = 2  # Poll every 2 seconds
                
                for i in range(max_polls):
                    await asyncio.sleep(poll_interval)
                    
                    status_response = await client.get(
                        status_url,
                        headers=headers
                    )
                    
                    if status_response.status_code == 200:
                        result = status_response.json()
                        print(f"🔄 Poll {i+1}: Status = {result.get('status')}")
                        
                        if result.get("status") == "COMPLETED":
                            print(f"✅ Job completed after {i+1} polls")
                            break
                        elif result.get("status") in ["FAILED", "CANCELLED"]:
                            print(f"❌ Job failed or cancelled: {result.get('status')}")
                            return OCRResponse(
                                extracted_text="",
                                status="error",
                                error=f"Job {result.get('status')}: {result.get('error', 'Unknown error')}"
                            )
                else:
                    # Timeout after max_polls
                    print(f"⏰ Polling timed out after {max_polls * poll_interval} seconds")
                    return OCRResponse(
                        extracted_text="",
                        status="error",
                        error="Request timed out while processing"
                    )
            
            # Extract text from response
            # RunPod response format: {"output": {"text": "extracted text"}, "status": "COMPLETED"}
            if result.get("status") == "COMPLETED":
                extracted_text = result.get("output", {}).get("text", "")
                print(f"✅ Extracted text from output: {extracted_text[:100]}...")
                
                if not extracted_text:
                    extracted_text = "No text extracted from image"
                
                return OCRResponse(
                    extracted_text=extracted_text,
                    status="success"
                )
            else:
                print(f"❌ Status not COMPLETED: {result.get('status')}")
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


@app.post("/api/ocr-pdf")
async def process_pdf_ocr(
    file: UploadFile = File(...),
    custom_prompt: Optional[str] = Form(None),
    max_new_tokens: int = Form(2048),
    min_pixels: Optional[int] = Form(200704),
    max_pixels: Optional[int] = Form(1003520),
):
    """
    Process a PDF and extract text from each page using OCR.
    Returns results page by page as they complete.
    
    Args:
        file: PDF file to process
        custom_prompt: Optional custom prompt (uses default if not provided)
        max_new_tokens: Maximum tokens to generate
        min_pixels: Minimum image resolution
        max_pixels: Maximum image resolution
        
    Returns:
        StreamingResponse with PDFPageResult for each page as JSON lines
    """
    try:
        # Validate RunPod configuration
        if not RUNPOD_ENDPOINT:
            raise HTTPException(
                status_code=500,
                detail="Model service endpoint not configured. Please set RUNPOD_ENDPOINT_URL environment variable."
            )
        
        # Validate PDF file
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="File must be a PDF"
            )
        
        # Read PDF file
        pdf_contents = await file.read()
        
        # Store PDF to Vercel Blob
        storage_result = await store_file_to_blob(
            pdf_contents,
            file.filename or "document.pdf",
            "application/pdf"
        )
        if storage_result.get("stored"):
            print(f"📁 PDF stored: {storage_result.get('url')}")
        
        # Generator function to process pages and yield results
        async def process_pages():
            try:
                # Open PDF with PyMuPDF
                pdf_document = fitz.open(stream=pdf_contents, filetype="pdf")
                total_pages = len(pdf_document)
                
                # Send total pages first
                yield json.dumps({
                    "type": "metadata",
                    "total_pages": total_pages
                }) + "\n"
                
                # Process each page
                for page_num in range(total_pages):
                    try:
                        # Get page
                        page = pdf_document[page_num]
                        
                        # Convert page to image (300 DPI for good quality)
                        mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
                        pix = page.get_pixmap(matrix=mat)
                        
                        # Convert to PIL Image
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        
                        # Convert image to base64 for display
                        buffered_display = io.BytesIO()
                        img.save(buffered_display, format="PNG")
                        page_image_base64 = base64.b64encode(buffered_display.getvalue()).decode('utf-8')
                        
                        # Convert image to base64 for OCR
                        buffered_ocr = io.BytesIO()
                        img.save(buffered_ocr, format="PNG")
                        img_base64 = base64.b64encode(buffered_ocr.getvalue()).decode('utf-8')
                        
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
                                # Yield error for this page
                                yield json.dumps({
                                    "type": "page_result",
                                    "page_number": page_num + 1,
                                    "status": "error",
                                    "error": f"Model service error: {response.text}",
                                    "page_image": page_image_base64
                                }) + "\n"
                                continue
                            
                            result = response.json()
                            
                            # Handle IN_QUEUE status - poll for results
                            if result.get("status") == "IN_QUEUE":
                                job_id = result.get("id")
                                status_url = RUNPOD_ENDPOINT.replace("/runsync", f"/status/{job_id}")
                                
                                max_polls = 180
                                poll_interval = 2
                                
                                for i in range(max_polls):
                                    await asyncio.sleep(poll_interval)
                                    
                                    status_response = await client.get(
                                        status_url,
                                        headers=headers
                                    )
                                    
                                    if status_response.status_code == 200:
                                        result = status_response.json()
                                        
                                        if result.get("status") == "COMPLETED":
                                            break
                                        elif result.get("status") in ["FAILED", "CANCELLED"]:
                                            yield json.dumps({
                                                "type": "page_result",
                                                "page_number": page_num + 1,
                                                "status": "error",
                                                "error": f"Job {result.get('status')}: {result.get('error', 'Unknown error')}",
                                                "page_image": page_image_base64
                                            }) + "\n"
                                            break
                                else:
                                    # Timeout
                                    yield json.dumps({
                                        "type": "page_result",
                                        "page_number": page_num + 1,
                                        "status": "error",
                                        "error": "Request timed out while processing",
                                        "page_image": page_image_base64
                                    }) + "\n"
                                    continue
                            
                            # Extract text from response
                            if result.get("status") == "COMPLETED":
                                extracted_text = result.get("output", {}).get("text", "")
                                
                                if not extracted_text:
                                    extracted_text = "No text extracted from this page"
                                
                                # Yield success result
                                yield json.dumps({
                                    "type": "page_result",
                                    "page_number": page_num + 1,
                                    "status": "success",
                                    "extracted_text": extracted_text,
                                    "page_image": page_image_base64
                                }) + "\n"
                            else:
                                yield json.dumps({
                                    "type": "page_result",
                                    "page_number": page_num + 1,
                                    "status": "error",
                                    "error": result.get("error", "Unknown error from model service"),
                                    "page_image": page_image_base64
                                }) + "\n"
                    
                    except Exception as e:
                        # Error processing this specific page
                        yield json.dumps({
                            "type": "page_result",
                            "page_number": page_num + 1,
                            "status": "error",
                            "error": f"Error processing page: {str(e)}",
                            "page_image": ""
                        }) + "\n"
                
                # Close PDF document
                pdf_document.close()
                
                # Send completion message
                yield json.dumps({
                    "type": "complete",
                    "status": "success"
                }) + "\n"
                
            except Exception as e:
                yield json.dumps({
                    "type": "error",
                    "error": f"Error processing PDF: {str(e)}"
                }) + "\n"
        
        return StreamingResponse(
            process_pages(),
            media_type="application/x-ndjson"
        )
                
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing PDF OCR request: {str(e)}")
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

