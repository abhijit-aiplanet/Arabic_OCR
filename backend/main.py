"""
FastAPI Backend for AIN OCR Application
Handles API requests and communicates with the model service on RunPod
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header, Depends
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
import jwt
from jwt import PyJWKClient
from supabase import create_client, Client
import time

load_dotenv()

# Vercel Blob Storage Configuration
BLOB_READ_WRITE_TOKEN = os.getenv("BLOB_READ_WRITE_TOKEN")
BLOB_API_URL = "https://blob.vercel-storage.com"

# Clerk Configuration
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
CLERK_JWKS_URL = "https://rational-coral-39.clerk.accounts.dev/.well-known/jwks.json"

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Use service key for backend
supabase: Optional[Client] = None

if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("✅ Supabase client initialized")
    except Exception as e:
        print(f"⚠️ Failed to initialize Supabase: {str(e)}")
else:
    print("⚠️ Supabase not configured, history will not be saved")


async def verify_clerk_token(authorization: Optional[str] = Header(None)) -> dict:
    """
    Verify Clerk JWT token from Authorization header.
    
    Args:
        authorization: Authorization header with Bearer token
        
    Returns:
        dict: Decoded token payload with user information
        
    Raises:
        HTTPException: If token is invalid or missing
    """
    if not CLERK_SECRET_KEY:
        print("⚠️ CLERK_SECRET_KEY not configured, skipping authentication")
        return {"user_id": "anonymous", "auth_disabled": True}
    
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization header. Please sign in."
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format. Expected 'Bearer <token>'"
        )
    
    token = authorization.replace("Bearer ", "")
    
    try:
        # Use PyJWKClient to fetch public keys from Clerk's JWKS endpoint
        jwks_client = PyJWKClient(CLERK_JWKS_URL)
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        
        # Decode and verify the token
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False}  # Clerk doesn't always include aud
        )
        
        print(f"✅ Authenticated user: {payload.get('sub', 'unknown')}")
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired. Please sign in again."
        )
    except jwt.InvalidTokenError as e:
        print(f"❌ Invalid token: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token. Please sign in again."
        )
    except Exception as e:
        print(f"❌ Authentication error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Authentication failed. Please sign in again."
        )


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


async def save_ocr_history(
    user_id: str,
    file_name: str,
    file_type: str,
    file_size: int,
    total_pages: int,
    extracted_text: str,
    status: str,
    error_message: Optional[str],
    processing_time: float,
    settings: dict,
    blob_url: Optional[str]
) -> Optional[dict]:
    """
    Save OCR history to Supabase.
    
    Args:
        user_id: User ID from Clerk
        file_name: Original filename
        file_type: 'image' or 'pdf'
        file_size: File size in bytes
        total_pages: Number of pages (1 for images)
        extracted_text: Extracted OCR text
        status: 'success' or 'error'
        error_message: Error message if status is 'error'
        processing_time: Time taken to process in seconds
        settings: OCR settings used
        blob_url: URL of file in Vercel Blob
        
    Returns:
        dict: Saved record or None if failed
    """
    if not supabase:
        print("⚠️ Supabase not configured, skipping history save")
        return None
    
    try:
        data = {
            "user_id": user_id,
            "file_name": file_name,
            "file_type": file_type,
            "file_size": file_size,
            "total_pages": total_pages,
            "extracted_text": extracted_text,
            "status": status,
            "error_message": error_message,
            "processing_time": processing_time,
            "settings": settings,
            "blob_url": blob_url
        }
        
        result = supabase.table("ocr_history").insert(data).execute()
        print(f"✅ OCR history saved for user {user_id}")
        return result.data[0] if result.data else None
        
    except Exception as e:
        print(f"❌ Error saving OCR history: {str(e)}")
        traceback.print_exc()
        return None

app = FastAPI(
    title="AIN OCR API",
    description="Backend API for Arabic OCR using AIN Vision Language Model",
    version="1.0.0"
)

# CORS middleware - Configure this properly for production
# Important: do NOT include "*" while allow_credentials=True (browsers will block / omit headers).
frontend_url_env = (os.getenv("FRONTEND_URL") or "").strip()
allowed_origins = [
    "http://localhost:3000",  # Local development
    "https://arabic-ocr-frontend-beryl.vercel.app",  # Production frontend
]
if frontend_url_env and frontend_url_env != "*":
    allowed_origins.append(frontend_url_env)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://arabic-ocr-frontend-.*\.vercel\.app",
    allow_credentials=False,  # We use Bearer tokens, not cross-site cookies
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
    confidence: Optional[Dict[str, Any]] = None


class PDFPageResult(BaseModel):
    """Result for a single PDF page"""
    page_number: int
    extracted_text: str
    status: str
    error: Optional[str] = None
    page_image: Optional[str] = None  # Base64 encoded image
    confidence: Optional[Dict[str, Any]] = None


class PDFOCRResponse(BaseModel):
    """Response model for PDF OCR processing"""
    total_pages: int
    results: List[PDFPageResult]
    status: str
    error: Optional[str] = None


class OCRHistoryItem(BaseModel):
    """Model for OCR history item"""
    id: str
    user_id: str
    file_name: str
    file_type: str
    file_size: int
    total_pages: int
    extracted_text: str
    edited_text: Optional[str]
    edited_at: Optional[str]
    status: str
    error_message: Optional[str]
    processing_time: float
    settings: dict
    blob_url: Optional[str]
    created_at: str


class OCRHistoryResponse(BaseModel):
    """Response model for OCR history"""
    history: List[OCRHistoryItem]
    total_count: int


class UpdateHistoryRequest(BaseModel):
def analyze_image_quality(image: Image.Image) -> Dict[str, Any]:
    """
    Lightweight image quality analysis using only PIL (no numpy).
    Returns per-factor scores in [0,1] and a pre_ocr_confidence.
    """
    try:
        from PIL import ImageStat, ImageFilter
        
        w, h = image.size
        pixels = w * h
        
        # Convert to grayscale for analysis
        gray = image.convert("L")
        
        # Get basic statistics using PIL's ImageStat
        stats = ImageStat.Stat(gray)
        brightness_avg = stats.mean[0]  # Average brightness (0-255)
        contrast_std = stats.stddev[0]  # Standard deviation (contrast measure)
        
        # 1. Sharpness: use variance of edges (Laplacian filter)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_stats = ImageStat.Stat(edges)
        sharpness_var = edge_stats.stddev[0] ** 2  # Variance approximation
        sharpness = float(min(sharpness_var / 500.0, 1.0))  # Normalize
        
        # 2. Contrast: use stddev from stats
        contrast = float(min(contrast_std / 80.0, 1.0))
        
        # 3. Brightness: prefer 100-180 range
        if brightness_avg < 100:
            brightness = float(max(brightness_avg / 100.0, 0.0))
        elif brightness_avg > 180:
            brightness = float(max(1.0 - ((brightness_avg - 180.0) / 75.0), 0.0))
        else:
            brightness = 1.0
        brightness = float(min(max(brightness, 0.0), 1.0))
        
        # 4. Resolution adequacy
        if pixels < 500_000:
            resolution = pixels / 500_000.0
        elif pixels < 1_000_000:
            resolution = 0.8 + (pixels - 500_000) / 500_000.0 * 0.2
        else:
            resolution = 1.0
        resolution = float(min(max(resolution, 0.0), 1.0))
        
        # 5. Noise estimate: compare original to slightly blurred
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=1))
        # Approximate noise by RMS difference
        original_stats = ImageStat.Stat(gray)
        blur_stats = ImageStat.Stat(blurred)
        noise_level = abs(original_stats.rms[0] - blur_stats.rms[0])
        noise = float(1.0 - min(noise_level / 20.0, 1.0))
        
        factors = {
            "sharpness": sharpness,
            "contrast": contrast,
            "brightness": brightness,
            "resolution": resolution,
            "noise": noise,
            "raw": {
                "sharpness_var": sharpness_var,
                "contrast_std": contrast_std,
                "brightness_avg": brightness_avg,
                "pixels": pixels,
                "noise_level": noise_level,
            },
        }
        
        pre = (
            sharpness * 0.25
            + contrast * 0.25
            + brightness * 0.20
            + resolution * 0.15
            + noise * 0.15
        )
        pre = float(min(max(pre, 0.0), 1.0))
        
        warnings: List[str] = []
        if sharpness < 0.4:
            warnings.append("Low sharpness detected (blurry image)")
        if contrast < 0.4:
            warnings.append("Low contrast detected (faded text/background)")
        if brightness < 0.5:
            warnings.append("Suboptimal brightness detected (too dark/too bright)")
        if resolution < 0.6:
            warnings.append("Low resolution detected (small text may be missed)")
        if noise < 0.5:
            warnings.append("High noise detected (scan artifacts or compression)")
        
        recommendation = (
            "excellent" if pre >= 0.85 else
            "good" if pre >= 0.7 else
            "fair" if pre >= 0.5 else
            "poor"
        )
        
        return {
            "pre_ocr_confidence": pre,
            "quality_factors": {k: v for k, v in factors.items() if k != "raw"},
            "raw_metrics": factors["raw"],
            "recommendation": recommendation,
            "warnings": warnings,
        }
    except Exception as e:
        print(f"⚠️ Image quality analysis failed: {str(e)}")
        traceback.print_exc()
        # Return safe fallback
        return {
            "pre_ocr_confidence": 0.7,  # Neutral fallback
            "quality_factors": {},
            "raw_metrics": {},
            "recommendation": "unknown",
            "warnings": ["Image quality analysis unavailable"],
        }


def analyze_text_quality(text: str) -> Dict[str, Any]:
    """Heuristic text quality analysis for Arabic OCR output."""
    t = (text or "").strip()
    if not t:
        return {
            "text_quality_confidence": 0.0,
            "quality_factors": {},
            "warnings": ["Empty text"],
            "validation_passed": False,
        }

    length = len(t)
    # 1) length score
    if length < 10:
        length_score = 0.2
    elif length < 50:
        length_score = 0.2 + (length - 10) / 40.0 * 0.5
    elif length <= 5000:
        length_score = 1.0
    else:
        length_score = 0.9

    # 2) arabic ratio
    arabic_chars = sum(1 for c in t if "\u0600" <= c <= "\u06FF")
    arabic_ratio = arabic_chars / max(length, 1)
    if arabic_ratio > 0.9:
        arabic_score = 1.0
    elif arabic_ratio > 0.7:
        arabic_score = 0.8
    elif arabic_ratio > 0.5:
        arabic_score = 0.6
    else:
        arabic_score = 0.3

    # 3) special chars ratio
    special_chars = sum(1 for c in t if (not c.isalnum()) and (not c.isspace()))
    special_ratio = special_chars / max(length, 1)
    if special_ratio <= 0.15:
        special_score = 1.0
    elif special_ratio <= 0.30:
        special_score = 0.7
    elif special_ratio <= 0.50:
        special_score = 0.4
    else:
        special_score = 0.1

    # 4) repetition/uniqueness
    words = [w for w in t.split() if w.strip()]
    unique_ratio = (len(set(words)) / len(words)) if words else 0.0
    if unique_ratio > 0.7:
        unique_score = 1.0
    elif unique_ratio > 0.5:
        unique_score = 0.8
    elif unique_ratio > 0.3:
        unique_score = 0.5
    else:
        unique_score = 0.2

    # 5) structure
    lines = [ln for ln in t.split("\n") if ln.strip()]
    if len(lines) >= 3:
        structure_score = 1.0
    elif len(lines) == 2:
        structure_score = 0.8
    elif len(lines) == 1:
        structure_score = 0.6
    else:
        structure_score = 0.4

    # 6) simple Arabic pattern presence (very lightweight)
    common_tokens = ["ال", "في", "من", "على", "إلى", "و", "هذا", "هذه"]
    common_hits = sum(1 for tok in common_tokens if tok in t)
    linguistic_score = 1.0 if common_hits >= 3 else 0.7 if common_hits == 2 else 0.4 if common_hits == 1 else 0.2

    score = (
        float(length_score) * 0.15
        + float(arabic_score) * 0.30
        + float(special_score) * 0.15
        + float(unique_score) * 0.15
        + float(structure_score) * 0.10
        + float(linguistic_score) * 0.15
    )
    score = float(min(max(score, 0.0), 1.0))

    warnings: List[str] = []
    if length < 10:
        warnings.append("Very short text (possible OCR failure)")
    if arabic_ratio < 0.5:
        warnings.append("Low Arabic character ratio (possible noise or wrong extraction)")
    if special_ratio > 0.3:
        warnings.append("High special-character density (possible OCR artifacts)")
    if unique_ratio < 0.3 and len(words) >= 10:
        warnings.append("High repetition detected (possible generation loop)")

    return {
        "text_quality_confidence": score,
        "quality_factors": {
            "length_score": float(length_score),
            "arabic_ratio": float(arabic_ratio),
            "special_chars": float(special_score),
            "uniqueness": float(unique_ratio),
            "structure": float(structure_score),
            "linguistic_patterns": float(linguistic_score),
        },
        "warnings": warnings,
        "validation_passed": score >= 0.5,
    }


def calculate_final_confidence(
    image_quality: Dict[str, Any],
    token_confidence: Optional[Dict[str, Any]],
    text_quality: Dict[str, Any],
) -> Dict[str, Any]:
    image_conf = image_quality.get("pre_ocr_confidence")
    token_conf = (token_confidence or {}).get("overall_token_confidence")
    text_conf = text_quality.get("text_quality_confidence")

    # Weighted combination (fallback if token_conf missing)
    if token_conf is None:
        # Reweight image/text to sum=1.0
        overall = (float(image_conf or 0.0) * 0.4) + (float(text_conf or 0.0) * 0.6)
        sources = {"image_quality": image_conf, "token_logits": None, "text_quality": text_conf}
    else:
        overall = (float(image_conf or 0.0) * 0.20) + (float(token_conf) * 0.50) + (float(text_conf or 0.0) * 0.30)
        sources = {"image_quality": image_conf, "token_logits": token_conf, "text_quality": text_conf}

    overall = float(min(max(overall, 0.0), 1.0))
    level = "high" if overall >= 0.9 else "medium" if overall >= 0.75 else "low_medium" if overall >= 0.6 else "low"

    # Pass through word confidences if present
    per_word = (token_confidence or {}).get("word_confidences") or []
    # Basic per-line (optional later): keep empty for now
    per_line = (token_confidence or {}).get("line_confidences") or []

    recommendations: List[str] = []
    if level in ["high"]:
        recommendations.append("Overall quality looks excellent")
    elif level in ["medium"]:
        recommendations.append("Good quality; minor review recommended")
    else:
        recommendations.append("Low confidence; please review carefully")

    warnings = []
    warnings.extend(image_quality.get("warnings") or [])
    warnings.extend(text_quality.get("warnings") or [])

    return {
        "overall_confidence": overall,
        "confidence_level": level,
        "confidence_sources": sources,
        "image_quality": image_quality,
        "text_quality": text_quality,
        "per_word": per_word,
        "per_line": per_line,
        "warnings": warnings,
        "recommendations": recommendations,
    }

    """Request model for updating OCR history"""
    edited_text: str


# ----------------------------
# Templates (Form/Doc presets)
# ----------------------------
class OCRTemplate(BaseModel):
    """Model for an OCR template (prompt preset + optional structure metadata)."""
    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    content_type: str  # e.g. form, document, receipt, table, id_card, certificate, handwritten, mixed
    language: str = "ar"  # ar | en | mixed
    custom_prompt: Optional[str] = None
    sections: Dict[str, Any] = {}  # stored as JSONB in DB (MVP: optional)
    tables: Optional[Dict[str, Any]] = None  # stored as JSONB in DB (MVP: optional)
    keywords: Optional[List[str]] = None
    is_public: bool = False
    usage_count: int = 0
    example_image_url: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CreateTemplateRequest(BaseModel):
    """Request model for creating an OCR template."""
    name: str
    description: Optional[str] = None
    content_type: str
    language: str = "ar"
    custom_prompt: Optional[str] = None
    sections: Optional[Dict[str, Any]] = None
    tables: Optional[Dict[str, Any]] = None
    keywords: Optional[List[str]] = None
    is_public: bool = False
    example_image_url: Optional[str] = None


class UpdateTemplateRequest(BaseModel):
    """Request model for updating an OCR template."""
    name: Optional[str] = None
    description: Optional[str] = None
    content_type: Optional[str] = None
    language: Optional[str] = None
    custom_prompt: Optional[str] = None
    sections: Optional[Dict[str, Any]] = None
    tables: Optional[Dict[str, Any]] = None
    keywords: Optional[List[str]] = None
    is_public: Optional[bool] = None
    example_image_url: Optional[str] = None


@app.get("/api/templates", response_model=List[OCRTemplate])
async def list_templates(user: dict = Depends(verify_clerk_token)):
    """List templates owned by the authenticated user."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Templates service not available")

    user_id = user.get("sub", "anonymous")
    try:
        response = supabase.table("ocr_templates")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("updated_at", desc=True)\
            .execute()
        return [OCRTemplate(**t) for t in (response.data or [])]
    except Exception as e:
        print(f"❌ Error listing templates: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")


@app.get("/api/templates/public", response_model=List[OCRTemplate])
async def list_public_templates():
    """List public templates (no auth)."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Templates service not available")

    try:
        response = supabase.table("ocr_templates")\
            .select("*")\
            .eq("is_public", True)\
            .order("usage_count", desc=True)\
            .limit(100)\
            .execute()
        return [OCRTemplate(**t) for t in (response.data or [])]
    except Exception as e:
        print(f"❌ Error listing public templates: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to list public templates: {str(e)}")


@app.post("/api/templates", response_model=OCRTemplate)
async def create_template(payload: CreateTemplateRequest, user: dict = Depends(verify_clerk_token)):
    """Create a new OCR template for the authenticated user."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Templates service not available")

    user_id = user.get("sub", "anonymous")
    try:
        data = {
            "user_id": user_id,
            "name": payload.name,
            "description": payload.description,
            "content_type": payload.content_type,
            "language": payload.language,
            "custom_prompt": payload.custom_prompt,
            "sections": payload.sections or {},
            "tables": payload.tables,
            "keywords": payload.keywords,
            "is_public": payload.is_public,
            "example_image_url": payload.example_image_url,
            "usage_count": 0,
        }
        result = supabase.table("ocr_templates").insert(data).execute()
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create template")
        return OCRTemplate(**result.data[0])
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error creating template: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create template: {str(e)}")


@app.patch("/api/templates/{template_id}", response_model=OCRTemplate)
async def update_template(template_id: str, payload: UpdateTemplateRequest, user: dict = Depends(verify_clerk_token)):
    """Update an existing template (must be owned by the authenticated user)."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Templates service not available")

    user_id = user.get("sub", "anonymous")
    try:
        # Ensure ownership
        existing = supabase.table("ocr_templates").select("*").eq("id", template_id).execute()
        if not existing.data:
            raise HTTPException(status_code=404, detail="Template not found")
        if existing.data[0].get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Not allowed to update this template")

        update_data: Dict[str, Any] = {}
        for k, v in payload.model_dump(exclude_unset=True).items():
            update_data[k] = v

        if not update_data:
            return OCRTemplate(**existing.data[0])

        update_data["updated_at"] = datetime.now().isoformat()

        updated = supabase.table("ocr_templates")\
            .update(update_data)\
            .eq("id", template_id)\
            .execute()
        if not updated.data:
            raise HTTPException(status_code=500, detail="Failed to update template")
        return OCRTemplate(**updated.data[0])
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating template: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to update template: {str(e)}")


@app.delete("/api/templates/{template_id}")
async def delete_template(template_id: str, user: dict = Depends(verify_clerk_token)):
    """Delete a template (must be owned by the authenticated user)."""
    if not supabase:
        raise HTTPException(status_code=503, detail="Templates service not available")

    user_id = user.get("sub", "anonymous")
    try:
        existing = supabase.table("ocr_templates").select("*").eq("id", template_id).execute()
        if not existing.data:
            raise HTTPException(status_code=404, detail="Template not found")
        if existing.data[0].get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Not allowed to delete this template")

        supabase.table("ocr_templates").delete().eq("id", template_id).execute()
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error deleting template: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to delete template: {str(e)}")


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


@app.get("/api/history", response_model=OCRHistoryResponse)
async def get_ocr_history(
    user: dict = Depends(verify_clerk_token),
    limit: int = 50,
    offset: int = 0
):
    """
    Get OCR history for the authenticated user.
    
    Args:
        user: Authenticated user from Clerk
        limit: Maximum number of records to return (default 50)
        offset: Offset for pagination (default 0)
        
    Returns:
        OCRHistoryResponse with user's OCR history
    """
    if not supabase:
        raise HTTPException(
            status_code=503,
            detail="History service not available"
        )
    
    try:
        user_id = user.get("sub", "anonymous")
        
        # Query history with pagination, ordered by created_at descending
        response = supabase.table("ocr_history")\
            .select("*", count="exact")\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .range(offset, offset + limit - 1)\
            .execute()
        
        total_count = response.count if hasattr(response, 'count') else len(response.data)
        
        history_items = []
        for item in response.data:
            history_items.append(OCRHistoryItem(
                id=item["id"],
                user_id=item["user_id"],
                file_name=item["file_name"],
                file_type=item["file_type"],
                file_size=item["file_size"],
                total_pages=item["total_pages"],
                extracted_text=item["extracted_text"],
                edited_text=item.get("edited_text"),
                edited_at=item.get("edited_at"),
                status=item["status"],
                error_message=item.get("error_message"),
                processing_time=item["processing_time"],
                settings=item["settings"],
                blob_url=item.get("blob_url"),
                created_at=item["created_at"]
            ))
        
        return OCRHistoryResponse(
            history=history_items,
            total_count=total_count
        )
        
    except Exception as e:
        print(f"❌ Error fetching OCR history: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch history: {str(e)}"
        )


@app.patch("/api/history/{history_id}", response_model=OCRHistoryItem)
async def update_ocr_history(
    history_id: str,
    update_data: UpdateHistoryRequest,
    user: dict = Depends(verify_clerk_token)
):
    """
    Update OCR history item (edit the extracted text).
    
    Args:
        history_id: ID of the history item to update
        update_data: New edited text
        user: Authenticated user from Clerk
        
    Returns:
        Updated OCRHistoryItem
    """
    if not supabase:
        raise HTTPException(
            status_code=503,
            detail="History service not available"
        )
    
    try:
        user_id = user.get("sub", "anonymous")
        
        # First, verify the item belongs to the user
        check_response = supabase.table("ocr_history")\
            .select("user_id")\
            .eq("id", history_id)\
            .single()\
            .execute()
        
        if not check_response.data:
            raise HTTPException(
                status_code=404,
                detail="History item not found"
            )
        
        if check_response.data["user_id"] != user_id:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to edit this item"
            )
        
        # Update the item
        update_response = supabase.table("ocr_history")\
            .update({
                "edited_text": update_data.edited_text,
                "edited_at": datetime.now().isoformat()
            })\
            .eq("id", history_id)\
            .execute()
        
        if not update_response.data:
            raise HTTPException(
                status_code=500,
                detail="Failed to update history item"
            )
        
        item = update_response.data[0]
        
        return OCRHistoryItem(
            id=item["id"],
            user_id=item["user_id"],
            file_name=item["file_name"],
            file_type=item["file_type"],
            file_size=item["file_size"],
            total_pages=item["total_pages"],
            extracted_text=item["extracted_text"],
            edited_text=item.get("edited_text"),
            edited_at=item.get("edited_at"),
            status=item["status"],
            error_message=item.get("error_message"),
            processing_time=item["processing_time"],
            settings=item["settings"],
            blob_url=item.get("blob_url"),
            created_at=item["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating OCR history: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update history: {str(e)}"
        )


@app.post("/api/ocr", response_model=OCRResponse)
async def process_ocr(
    file: UploadFile = File(...),
    custom_prompt: Optional[str] = Form(None),
    max_new_tokens: int = Form(2048),
    min_pixels: Optional[int] = Form(200704),
    max_pixels: Optional[int] = Form(1003520),
    user: dict = Depends(verify_clerk_token),
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
    start_time = time.time()
    storage_url = None
    
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

        image_quality = analyze_image_quality(image)
        
        # Store file to Vercel Blob (async, non-blocking)
        storage_result = await store_file_to_blob(
            contents,
            file.filename or "image.png",
            file.content_type or "image/png"
        )
        if storage_result.get("stored"):
            storage_url = storage_result.get('url')
            print(f"📁 Image stored: {storage_url}")
        
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
                output = result.get("output", {}) or {}
                extracted_text = output.get("text", "")
                token_confidence = output.get("token_confidence")
                print(f"✅ Extracted text from output: {extracted_text[:100]}...")
                
                if not extracted_text:
                    extracted_text = "No text extracted from image"

                text_quality = analyze_text_quality(extracted_text)
                confidence = calculate_final_confidence(image_quality, token_confidence, text_quality)
                
                # Save to history
                processing_time = time.time() - start_time
                await save_ocr_history(
                    user_id=user.get("sub", "anonymous"),
                    file_name=file.filename or "image.png",
                    file_type="image",
                    file_size=len(contents),
                    total_pages=1,
                    extracted_text=extracted_text,
                    status="success",
                    error_message=None,
                    processing_time=processing_time,
                    settings={
                        "max_new_tokens": max_new_tokens,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                        "custom_prompt": custom_prompt if custom_prompt else None,
                        "confidence": confidence
                    },
                    blob_url=storage_url
                )
                
                return OCRResponse(
                    extracted_text=extracted_text,
                    status="success",
                    confidence=confidence
                )
            else:
                error_msg = result.get("error", "Unknown error from model service")
                print(f"❌ Status not COMPLETED: {result.get('status')}")
                
                # Save error to history
                processing_time = time.time() - start_time
                await save_ocr_history(
                    user_id=user.get("sub", "anonymous"),
                    file_name=file.filename or "image.png",
                    file_type="image",
                    file_size=len(contents),
                    total_pages=1,
                    extracted_text="",
                    status="error",
                    error_message=error_msg,
                    processing_time=processing_time,
                    settings={
                        "max_new_tokens": max_new_tokens,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                        "custom_prompt": custom_prompt if custom_prompt else None
                    },
                    blob_url=storage_url
                )
                
                return OCRResponse(
                    extracted_text="",
                    status="error",
                    error=error_msg,
                    confidence=None
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
    user: dict = Depends(verify_clerk_token),
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

                        # Pre-OCR image quality (per page)
                        page_image_quality = analyze_image_quality(img)
                        
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
                                output = result.get("output", {}) or {}
                                extracted_text = output.get("text", "")
                                token_confidence = output.get("token_confidence")
                                
                                if not extracted_text:
                                    extracted_text = "No text extracted from this page"

                                text_quality = analyze_text_quality(extracted_text)
                                confidence = calculate_final_confidence(page_image_quality, token_confidence, text_quality)
                                
                                # Yield success result
                                yield json.dumps({
                                    "type": "page_result",
                                    "page_number": page_num + 1,
                                    "status": "success",
                                    "extracted_text": extracted_text,
                                    "page_image": page_image_base64,
                                    "confidence": confidence
                                }) + "\n"
                            else:
                                yield json.dumps({
                                    "type": "page_result",
                                    "page_number": page_num + 1,
                                    "status": "error",
                                    "error": result.get("error", "Unknown error from model service"),
                                    "page_image": page_image_base64,
                                    "confidence": None
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

