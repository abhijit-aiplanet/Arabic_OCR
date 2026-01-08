"""
FastAPI Backend for AIN OCR Application
Handles API requests and communicates with the model service on RunPod

ARCHITECTURE NOTES:
- Uses connection pooling for RunPod requests
- Implements retry logic with exponential backoff for cold starts
- Supports 15+ page PDFs with robust error handling
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
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
import random

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Vercel Blob Storage Configuration
BLOB_READ_WRITE_TOKEN = os.getenv("BLOB_READ_WRITE_TOKEN")
BLOB_API_URL = "https://blob.vercel-storage.com"

# Clerk Configuration
CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
CLERK_JWKS_URL = "https://rational-coral-39.clerk.accounts.dev/.well-known/jwks.json"

# Supabase Configuration
_supabase_url_raw = os.getenv("SUPABASE_URL", "")
SUPABASE_URL = _supabase_url_raw.strip() if _supabase_url_raw else None
if SUPABASE_URL and not SUPABASE_URL.startswith(("http://", "https://")):
    print(f"SUPABASE_URL missing protocol, adding https://")
    SUPABASE_URL = f"https://{SUPABASE_URL}"

SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Optional[Client] = None

if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print(f"Supabase client initialized: {SUPABASE_URL[:40]}...")
    except Exception as e:
        print(f"Failed to initialize Supabase: {str(e)}")
else:
    print("Supabase not configured, history will not be saved")

# RunPod Configuration
_runpod_endpoint_raw = os.getenv("RUNPOD_ENDPOINT_URL", "")
RUNPOD_ENDPOINT = _runpod_endpoint_raw.strip() if _runpod_endpoint_raw else None
if RUNPOD_ENDPOINT and not RUNPOD_ENDPOINT.startswith(("http://", "https://")):
    print(f"RUNPOD_ENDPOINT_URL missing protocol, adding https://")
    RUNPOD_ENDPOINT = f"https://{RUNPOD_ENDPOINT}"
print(f"RUNPOD_ENDPOINT configured: {RUNPOD_ENDPOINT[:50] if RUNPOD_ENDPOINT else 'NOT SET'}...")

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

# =============================================================================
# RUNPOD CLIENT CONFIGURATION (CRITICAL FOR COLD START HANDLING)
# =============================================================================

# Timeouts (in seconds)
RUNPOD_CONNECT_TIMEOUT = 30          # Time to establish connection
RUNPOD_READ_TIMEOUT = 600            # Time to wait for response (10 min for cold starts)
RUNPOD_INITIAL_REQUEST_TIMEOUT = 60  # Initial POST request timeout

# Retry Configuration
MAX_RETRIES = 3                      # Number of retries for failed requests
RETRY_BASE_DELAY = 2                 # Base delay for exponential backoff (seconds)
RETRY_MAX_DELAY = 30                 # Maximum delay between retries (seconds)

# Polling Configuration (for IN_QUEUE jobs)
POLL_INTERVAL_INITIAL = 3            # Initial poll interval (seconds)
POLL_INTERVAL_MAX = 10               # Max poll interval (seconds)
POLL_TIMEOUT_COLD_START = 480        # Total polling timeout for cold starts (8 minutes)
POLL_TIMEOUT_WARM = 300              # Total polling timeout for warm workers (5 minutes)

# Cold Start Detection
COLD_START_INDICATORS = ["IN_QUEUE", "IN_PROGRESS"]
COLD_START_WAIT_THRESHOLD = 30       # If queued for 30+ seconds, likely cold start

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

# Structured Extraction Prompt (for forms - returns JSON)
STRUCTURED_EXTRACTION_PROMPT = """Extract all fields from this Arabic form as structured JSON.

You are analyzing a form image. Extract ALL visible fields, labels, and their values.

OUTPUT FORMAT (strict JSON only, no markdown):
{
  "form_title": "title of the form if visible, or null",
  "sections": [
    {
      "name": "section name if visible, or null for ungrouped fields",
      "fields": [
        {
          "label": "the field label exactly as shown",
          "value": "the handwritten or typed value",
          "type": "text"
        }
      ]
    }
  ],
  "tables": [
    {
      "headers": ["column1", "column2"],
      "rows": [["value1", "value2"]]
    }
  ],
  "checkboxes": [
    {"label": "checkbox label", "checked": true}
  ]
}

EXTRACTION RULES:
1. Extract EVERY label and its corresponding value (handwritten or typed)
2. Preserve Arabic text exactly - do NOT translate
3. For empty fields, use value = ""
4. For checkboxes: checked = true or false
5. For tables: extract headers and all rows
6. Group related fields into sections when there are clear visual separators
7. If no clear sections, put all fields in one section with name = null
8. Output ONLY valid JSON - no markdown code blocks, no explanation

IMPORTANT: The output must be parseable JSON. Start with { and end with }"""


# =============================================================================
# RUNPOD CLIENT WITH RETRY LOGIC
# =============================================================================

class RunPodClient:
    """
    Robust RunPod client with:
    - Connection pooling
    - Retry logic with exponential backoff
    - Cold start handling
    - Proper timeout management
    """
    
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._last_successful_call: float = 0
        self._is_likely_cold: bool = True
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create the shared httpx client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=RUNPOD_CONNECT_TIMEOUT,
                    read=RUNPOD_READ_TIMEOUT,
                    write=30.0,
                    pool=30.0
                ),
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                    keepalive_expiry=60.0
                ),
                http2=True  # Enable HTTP/2 for better performance
            )
        return self._client
    
    async def close(self):
        """Close the client connection."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
        jitter = random.uniform(0, delay * 0.3)  # Add 0-30% jitter
        return delay + jitter
    
    def _is_cold_start_likely(self) -> bool:
        """Detect if cold start is likely based on recent activity."""
        if self._last_successful_call == 0:
            return True
        time_since_last = time.time() - self._last_successful_call
        return time_since_last > 300  # Cold if no activity for 5 minutes
    
    async def call_runpod(
        self,
        payload: dict,
        page_info: str = ""
    ) -> Tuple[dict, bool]:
        """
        Call RunPod endpoint with robust retry logic and cold start handling.
        
        Args:
            payload: Request payload for RunPod
            page_info: Optional string for logging (e.g., "Page 3/15")
            
        Returns:
            Tuple of (result_dict, is_success)
        """
        if not RUNPOD_ENDPOINT:
            return {"error": "RunPod endpoint not configured"}, False
        
        client = await self.get_client()
        headers = {
            "Authorization": f"Bearer {RUNPOD_API_KEY}",
            "Content-Type": "application/json"
        }
        
        is_cold = self._is_cold_start_likely()
        poll_timeout = POLL_TIMEOUT_COLD_START if is_cold else POLL_TIMEOUT_WARM
        
        prefix = f"[{page_info}] " if page_info else ""
        print(f"{prefix}Calling RunPod (cold_start_likely={is_cold}, poll_timeout={poll_timeout}s)")
        
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                # Initial request with shorter timeout
                try:
                    response = await asyncio.wait_for(
                        client.post(RUNPOD_ENDPOINT, json=payload, headers=headers),
                        timeout=RUNPOD_INITIAL_REQUEST_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    print(f"{prefix}Initial request timeout (attempt {attempt + 1}/{MAX_RETRIES})")
                    if attempt < MAX_RETRIES - 1:
                        delay = self._calculate_backoff(attempt)
                        print(f"{prefix}Retrying in {delay:.1f}s...")
                        await asyncio.sleep(delay)
                        continue
                    return {"error": "Request timeout - RunPod may be experiencing cold start"}, False
                
                if response.status_code != 200:
                    error_text = response.text[:500]
                    print(f"{prefix}RunPod error {response.status_code}: {error_text}")
                    
                    # Retry on 5xx errors (server issues)
                    if response.status_code >= 500 and attempt < MAX_RETRIES - 1:
                        delay = self._calculate_backoff(attempt)
                        print(f"{prefix}Server error, retrying in {delay:.1f}s...")
                        await asyncio.sleep(delay)
                        continue
                    
                    return {"error": f"RunPod error: {response.status_code}"}, False
                
                result = response.json()
                status = result.get("status")
                
                # Handle immediate completion
                if status == "COMPLETED":
                    self._last_successful_call = time.time()
                    self._is_likely_cold = False
                    print(f"{prefix}Completed immediately")
                    return result, True
                
                # Handle queued/in-progress jobs (cold start scenario)
                if status in COLD_START_INDICATORS:
                    job_id = result.get("id")
                    if not job_id:
                        return {"error": "No job ID returned"}, False
                    
                    print(f"{prefix}Job queued (ID: {job_id}), polling for completion...")
                    
                    # Poll for completion with adaptive intervals
                    poll_result = await self._poll_for_completion(
                        client, job_id, headers, poll_timeout, prefix
                    )
                    
                    if poll_result.get("status") == "COMPLETED":
                        self._last_successful_call = time.time()
                        self._is_likely_cold = False
                        return poll_result, True
                    
                    return poll_result, False
                
                # Handle failed status
                if status in ["FAILED", "CANCELLED"]:
                    error_msg = result.get("error", "Job failed")
                    print(f"{prefix}Job {status}: {error_msg}")
                    return {"error": error_msg, "status": status}, False
                
                # Unknown status
                print(f"{prefix}Unknown status: {status}")
                return result, status == "COMPLETED"
                
            except httpx.ConnectError as e:
                last_error = f"Connection error: {str(e)}"
                print(f"{prefix}{last_error} (attempt {attempt + 1}/{MAX_RETRIES})")
            except httpx.ReadTimeout as e:
                last_error = f"Read timeout: {str(e)}"
                print(f"{prefix}{last_error} (attempt {attempt + 1}/{MAX_RETRIES})")
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                print(f"{prefix}{last_error} (attempt {attempt + 1}/{MAX_RETRIES})")
                traceback.print_exc()
            
            # Exponential backoff before retry
            if attempt < MAX_RETRIES - 1:
                delay = self._calculate_backoff(attempt)
                print(f"{prefix}Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
        
        return {"error": last_error or "Max retries exceeded"}, False
    
    async def _poll_for_completion(
        self,
        client: httpx.AsyncClient,
        job_id: str,
        headers: dict,
        timeout: float,
        prefix: str
    ) -> dict:
        """
        Poll RunPod status endpoint for job completion with adaptive intervals.
        """
        status_url = RUNPOD_ENDPOINT.replace("/runsync", f"/status/{job_id}")
        start_time = time.time()
        poll_count = 0
        poll_interval = POLL_INTERVAL_INITIAL
        
        while time.time() - start_time < timeout:
            poll_count += 1
            elapsed = time.time() - start_time
            
            try:
                status_response = await client.get(status_url, headers=headers)
                
                if status_response.status_code == 200:
                    result = status_response.json()
                    status = result.get("status")
                    
                    # Log progress every 5 polls
                    if poll_count % 5 == 0 or status == "COMPLETED":
                        print(f"{prefix}Poll #{poll_count} ({elapsed:.0f}s): {status}")
                    
                    if status == "COMPLETED":
                        print(f"{prefix}Job completed after {elapsed:.1f}s ({poll_count} polls)")
                        return result
                    
                    if status in ["FAILED", "CANCELLED"]:
                        print(f"{prefix}Job {status} after {elapsed:.1f}s")
                        return result
                    
                    # Adaptive polling: increase interval over time
                    if elapsed > 60 and poll_interval < POLL_INTERVAL_MAX:
                        poll_interval = min(poll_interval + 1, POLL_INTERVAL_MAX)
                        
            except Exception as e:
                print(f"{prefix}Poll error: {str(e)}")
            
            await asyncio.sleep(poll_interval)
        
        print(f"{prefix}Polling timed out after {timeout}s ({poll_count} polls)")
        return {"error": "Processing timed out - please try again", "status": "TIMEOUT"}


# Global RunPod client instance
runpod_client = RunPodClient()


# =============================================================================
# AUTHENTICATION
# =============================================================================

async def verify_clerk_token(authorization: Optional[str] = Header(None)) -> dict:
    """Verify Clerk JWT token from Authorization header."""
    if not CLERK_SECRET_KEY:
        print("CLERK_SECRET_KEY not configured, skipping authentication")
        return {"user_id": "anonymous", "auth_disabled": True}
    
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header. Please sign in.")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format. Expected 'Bearer <token>'")
    
    token = authorization.replace("Bearer ", "")
    
    try:
        jwks_client = PyJWKClient(CLERK_JWKS_URL)
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False}
        )
        
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired. Please sign in again.")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail="Invalid authentication token. Please sign in again.")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication failed. Please sign in again.")


# =============================================================================
# STORAGE HELPERS
# =============================================================================

async def store_file_to_blob(file_data: bytes, filename: str, content_type: str = "application/octet-stream") -> dict:
    """Store a file in Vercel Blob Storage."""
    if not BLOB_READ_WRITE_TOKEN:
        return {"error": "Blob storage not configured", "stored": False}
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
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
                return {
                    "stored": True,
                    "url": result.get("url"),
                    "pathname": result.get("pathname", stored_filename),
                    "original_filename": filename
                }
            else:
                return {"error": f"Storage failed: {response.status_code}", "stored": False}
                
    except Exception as e:
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
    """Save OCR history to Supabase."""
    if not supabase:
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
        return result.data[0] if result.data else None
        
    except Exception as e:
        print(f"Error saving OCR history: {str(e)}")
        return None


# =============================================================================
# QUALITY ANALYSIS
# =============================================================================

def analyze_image_quality(image: Image.Image) -> Dict[str, Any]:
    """Analyze image quality for confidence scoring."""
    w, h = image.size
    pixels = w * h
    
    if pixels < 500_000:
        resolution = pixels / 500_000.0
    elif pixels < 1_000_000:
        resolution = 0.8 + (pixels - 500_000) / 500_000.0 * 0.2
    else:
        resolution = 1.0
    
    return {
        "pre_ocr_confidence": 0.75,
        "quality_factors": {"resolution": float(resolution)},
        "raw_metrics": {"pixels": pixels},
        "recommendation": "good",
        "warnings": [],
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
    
    # Length score
    if length < 10:
        length_score = 0.2
    elif length < 50:
        length_score = 0.2 + (length - 10) / 40.0 * 0.5
    elif length <= 5000:
        length_score = 1.0
    else:
        length_score = 0.9

    # Arabic ratio
    arabic_chars = sum(1 for c in t if "\u0600" <= c <= "\u06FF")
    arabic_ratio = arabic_chars / max(length, 1)
    arabic_score = 1.0 if arabic_ratio > 0.9 else 0.8 if arabic_ratio > 0.7 else 0.6 if arabic_ratio > 0.5 else 0.3

    # Special chars
    special_chars = sum(1 for c in t if (not c.isalnum()) and (not c.isspace()))
    special_ratio = special_chars / max(length, 1)
    special_score = 1.0 if special_ratio <= 0.15 else 0.7 if special_ratio <= 0.30 else 0.4 if special_ratio <= 0.50 else 0.1

    # Uniqueness
    words = [w for w in t.split() if w.strip()]
    unique_ratio = (len(set(words)) / len(words)) if words else 0.0
    unique_score = 1.0 if unique_ratio > 0.7 else 0.8 if unique_ratio > 0.5 else 0.5 if unique_ratio > 0.3 else 0.2

    # Structure
    lines = [ln for ln in t.split("\n") if ln.strip()]
    structure_score = 1.0 if len(lines) >= 3 else 0.8 if len(lines) == 2 else 0.6 if len(lines) == 1 else 0.4

    # Linguistic patterns
    common_tokens = ["ال", "في", "من", "على", "إلى", "و", "هذا", "هذه"]
    common_hits = sum(1 for tok in common_tokens if tok in t)
    linguistic_score = 1.0 if common_hits >= 3 else 0.7 if common_hits == 2 else 0.4 if common_hits == 1 else 0.2

    score = (
        float(length_score) * 0.15 +
        float(arabic_score) * 0.30 +
        float(special_score) * 0.15 +
        float(unique_score) * 0.15 +
        float(structure_score) * 0.10 +
        float(linguistic_score) * 0.15
    )
    score = float(min(max(score, 0.0), 1.0))

    warnings: List[str] = []
    if length < 10:
        warnings.append("Very short text")
    if arabic_ratio < 0.5:
        warnings.append("Low Arabic character ratio")
    if special_ratio > 0.3:
        warnings.append("High special-character density")
    if unique_ratio < 0.3 and len(words) >= 10:
        warnings.append("High repetition detected")

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
    """Calculate final confidence score from all sources."""
    image_conf = image_quality.get("pre_ocr_confidence")
    token_conf = (token_confidence or {}).get("overall_token_confidence")
    text_conf = text_quality.get("text_quality_confidence")

    if token_conf is None:
        overall = (float(image_conf or 0.0) * 0.4) + (float(text_conf or 0.0) * 0.6)
        sources = {"image_quality": image_conf, "token_logits": None, "text_quality": text_conf}
    else:
        overall = (float(image_conf or 0.0) * 0.20) + (float(token_conf) * 0.50) + (float(text_conf or 0.0) * 0.30)
        sources = {"image_quality": image_conf, "token_logits": token_conf, "text_quality": text_conf}

    overall = float(min(max(overall, 0.0), 1.0))
    level = "high" if overall >= 0.9 else "medium" if overall >= 0.75 else "low_medium" if overall >= 0.6 else "low"

    per_word = (token_confidence or {}).get("word_confidences") or []
    per_line = (token_confidence or {}).get("line_confidences") or []

    recommendations = []
    if level == "high":
        recommendations.append("Overall quality looks excellent")
    elif level == "medium":
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


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class OCRRequest(BaseModel):
    custom_prompt: Optional[str] = None
    max_new_tokens: int = 4096
    min_pixels: Optional[int] = 200704
    max_pixels: Optional[int] = 1003520


class OCRResponse(BaseModel):
    extracted_text: str
    status: str
    error: Optional[str] = None
    confidence: Optional[Dict[str, Any]] = None


class PDFPageResult(BaseModel):
    page_number: int
    extracted_text: str
    status: str
    error: Optional[str] = None
    page_image: Optional[str] = None
    confidence: Optional[Dict[str, Any]] = None


class OCRHistoryItem(BaseModel):
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
    history: List[OCRHistoryItem]
    total_count: int


class UpdateHistoryRequest(BaseModel):
    edited_text: str


class OCRTemplate(BaseModel):
    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    content_type: str
    language: str = "ar"
    custom_prompt: Optional[str] = None
    sections: Dict[str, Any] = {}
    tables: Optional[Dict[str, Any]] = None
    keywords: Optional[List[str]] = None
    is_public: bool = False
    usage_count: int = 0
    example_image_url: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CreateTemplateRequest(BaseModel):
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


# Structured Extraction Models
class ExtractedField(BaseModel):
    label: str
    value: str
    type: str = "text"  # text, date, number, checkbox


class ExtractedSection(BaseModel):
    name: Optional[str] = None
    fields: List[ExtractedField] = []


class ExtractedTable(BaseModel):
    headers: List[str] = []
    rows: List[List[str]] = []


class ExtractedCheckbox(BaseModel):
    label: str
    checked: bool = False


class StructuredExtractionData(BaseModel):
    form_title: Optional[str] = None
    sections: List[ExtractedSection] = []
    tables: List[ExtractedTable] = []
    checkboxes: List[ExtractedCheckbox] = []


class StructuredOCRResponse(BaseModel):
    raw_text: str
    structured_data: Optional[StructuredExtractionData] = None
    status: str
    error: Optional[str] = None
    confidence: Optional[Dict[str, Any]] = None
    parsing_successful: bool = False


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Arabic OCR API",
    description="Backend API for Arabic OCR using AIN Vision Language Model",
    version="2.0.0"
)

# CORS middleware
frontend_url_env = (os.getenv("FRONTEND_URL") or "").strip()
allowed_origins = [
    "http://localhost:3000",
    "https://arabic-ocr-frontend-beryl.vercel.app",
]
if frontend_url_env and frontend_url_env != "*":
    allowed_origins.append(frontend_url_env)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://arabic-ocr-frontend-.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# LIFECYCLE EVENTS
# =============================================================================

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await runpod_client.close()


# =============================================================================
# HEALTH & INFO ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "message": "Arabic OCR API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_service": "configured" if RUNPOD_ENDPOINT else "not_configured",
        "features": {
            "retry_enabled": True,
            "max_retries": MAX_RETRIES,
            "cold_start_timeout": POLL_TIMEOUT_COLD_START
        }
    }


@app.get("/api/prompt")
async def get_default_prompt():
    return {"default_prompt": DEFAULT_OCR_PROMPT}


# =============================================================================
# HISTORY ENDPOINTS
# =============================================================================

@app.get("/api/history", response_model=OCRHistoryResponse)
async def get_ocr_history(
    user: dict = Depends(verify_clerk_token),
    limit: int = 50,
    offset: int = 0
):
    if not supabase:
        raise HTTPException(status_code=503, detail="History service not available")
    
    try:
        user_id = user.get("sub", "anonymous")
        
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
        
        return OCRHistoryResponse(history=history_items, total_count=total_count)
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


@app.patch("/api/history/{history_id}", response_model=OCRHistoryItem)
async def update_ocr_history(
    history_id: str,
    update_data: UpdateHistoryRequest,
    user: dict = Depends(verify_clerk_token)
):
    if not supabase:
        raise HTTPException(status_code=503, detail="History service not available")
    
    try:
        user_id = user.get("sub", "anonymous")
        
        check_response = supabase.table("ocr_history")\
            .select("user_id")\
            .eq("id", history_id)\
            .single()\
            .execute()
        
        if not check_response.data:
            raise HTTPException(status_code=404, detail="History item not found")
        
        if check_response.data["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Permission denied")
        
        update_response = supabase.table("ocr_history")\
            .update({
                "edited_text": update_data.edited_text,
                "edited_at": datetime.now().isoformat()
            })\
            .eq("id", history_id)\
            .execute()
        
        if not update_response.data:
            raise HTTPException(status_code=500, detail="Failed to update")
        
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to update: {str(e)}")


# =============================================================================
# TEMPLATE ENDPOINTS
# =============================================================================

@app.get("/api/templates", response_model=List[OCRTemplate])
async def list_templates(user: dict = Depends(verify_clerk_token)):
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
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")


@app.get("/api/templates/public", response_model=List[OCRTemplate])
async def list_public_templates():
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
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")


@app.post("/api/templates", response_model=OCRTemplate)
async def create_template(payload: CreateTemplateRequest, user: dict = Depends(verify_clerk_token)):
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
        raise HTTPException(status_code=500, detail=f"Failed to create template: {str(e)}")


@app.patch("/api/templates/{template_id}", response_model=OCRTemplate)
async def update_template(template_id: str, payload: UpdateTemplateRequest, user: dict = Depends(verify_clerk_token)):
    if not supabase:
        raise HTTPException(status_code=503, detail="Templates service not available")

    user_id = user.get("sub", "anonymous")
    try:
        existing = supabase.table("ocr_templates").select("*").eq("id", template_id).execute()
        if not existing.data:
            raise HTTPException(status_code=404, detail="Template not found")
        if existing.data[0].get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Permission denied")

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
            raise HTTPException(status_code=500, detail="Failed to update")
        return OCRTemplate(**updated.data[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update: {str(e)}")


@app.delete("/api/templates/{template_id}")
async def delete_template(template_id: str, user: dict = Depends(verify_clerk_token)):
    if not supabase:
        raise HTTPException(status_code=503, detail="Templates service not available")

    user_id = user.get("sub", "anonymous")
    try:
        existing = supabase.table("ocr_templates").select("*").eq("id", template_id).execute()
        if not existing.data:
            raise HTTPException(status_code=404, detail="Template not found")
        if existing.data[0].get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Permission denied")

        supabase.table("ocr_templates").delete().eq("id", template_id).execute()
        return {"status": "success"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")


# =============================================================================
# OCR ENDPOINTS (CORE FUNCTIONALITY)
# =============================================================================

@app.post("/api/ocr", response_model=OCRResponse)
async def process_ocr(
    file: UploadFile = File(...),
    custom_prompt: Optional[str] = Form(None),
    max_new_tokens: int = Form(4096),
    min_pixels: Optional[int] = Form(200704),
    max_pixels: Optional[int] = Form(1003520),
    user: dict = Depends(verify_clerk_token),
):
    """
    Process an image and extract text using OCR.
    Includes robust cold start handling and retry logic.
    """
    start_time = time.time()
    storage_url = None
    
    try:
        if not RUNPOD_ENDPOINT:
            raise HTTPException(status_code=500, detail="Model service not configured")
        
        # Read and validate image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        image_quality = analyze_image_quality(image)
        
        # Store file (non-blocking)
        storage_result = await store_file_to_blob(
            contents,
            file.filename or "image.png",
            file.content_type or "image/png"
        )
        if storage_result.get("stored"):
            storage_url = storage_result.get('url')
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Prepare payload
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
        
        # Call RunPod with retry logic
        result, success = await runpod_client.call_runpod(payload)
        
        processing_time = time.time() - start_time
        
        if success and result.get("status") == "COMPLETED":
            output = result.get("output", {}) or {}
            extracted_text = output.get("text", "")
            token_confidence = output.get("token_confidence")
            
            if not extracted_text:
                extracted_text = "No text extracted from image"

            text_quality = analyze_text_quality(extracted_text)
            confidence = calculate_final_confidence(image_quality, token_confidence, text_quality)
            
            # Save to history
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
                },
                blob_url=storage_url
            )
            
            return OCRResponse(
                extracted_text=extracted_text,
                status="success",
                confidence=confidence
            )
        else:
            error_msg = result.get("error", "Processing failed")
            
            # Save error to history
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/ocr-pdf")
async def process_pdf_ocr(
    file: UploadFile = File(...),
    custom_prompt: Optional[str] = Form(None),
    max_new_tokens: int = Form(4096),
    min_pixels: Optional[int] = Form(200704),
    max_pixels: Optional[int] = Form(1003520),
    user: dict = Depends(verify_clerk_token),
):
    """
    Process a PDF and extract text from each page.
    Optimized for 15+ page PDFs with robust error handling.
    """
    start_time = time.time()
    
    try:
        if not RUNPOD_ENDPOINT:
            raise HTTPException(status_code=500, detail="Model service not configured")
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        pdf_contents = await file.read()
        pdf_file_size = len(pdf_contents)
        pdf_filename = file.filename or "document.pdf"
        user_id = user.get("sub", "anonymous")
        
        # Store PDF
        storage_result = await store_file_to_blob(pdf_contents, pdf_filename, "application/pdf")
        storage_url = storage_result.get('url') if storage_result.get("stored") else None
        
        async def process_pages():
            all_extracted_texts = []
            successful_pages = 0
            error_pages = 0
            total_pages = 0
            
            try:
                pdf_document = fitz.open(stream=pdf_contents, filetype="pdf")
                total_pages = len(pdf_document)
                
                print(f"Processing PDF: {pdf_filename} ({total_pages} pages)")
                
                # Send metadata
                yield json.dumps({
                    "type": "metadata",
                    "total_pages": total_pages
                }) + "\n"
                
                prompt = custom_prompt if custom_prompt and custom_prompt.strip() else DEFAULT_OCR_PROMPT
                
                # Process each page
                for page_num in range(total_pages):
                    page_info = f"Page {page_num + 1}/{total_pages}"
                    
                    try:
                        page = pdf_document[page_num]
                        
                        # Convert page to image (300 DPI)
                        mat = fitz.Matrix(300/72, 300/72)
                        pix = page.get_pixmap(matrix=mat)
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        
                        # Create base64 for display
                        buffered_display = io.BytesIO()
                        img.save(buffered_display, format="PNG", optimize=True)
                        page_image_base64 = base64.b64encode(buffered_display.getvalue()).decode('utf-8')
                        
                        # Create base64 for OCR
                        buffered_ocr = io.BytesIO()
                        img.save(buffered_ocr, format="PNG")
                        img_base64 = base64.b64encode(buffered_ocr.getvalue()).decode('utf-8')

                        page_image_quality = analyze_image_quality(img)
                        
                        payload = {
                            "input": {
                                "image": img_base64,
                                "prompt": prompt,
                                "max_new_tokens": max_new_tokens,
                                "min_pixels": min_pixels,
                                "max_pixels": max_pixels
                            }
                        }
                        
                        # Call RunPod with retry logic (using shared client)
                        result, success = await runpod_client.call_runpod(payload, page_info)
                        
                        if success and result.get("status") == "COMPLETED":
                            page_output = result.get("output", {}) or {}
                            extracted_text = page_output.get("text", "")
                            token_confidence = page_output.get("token_confidence")
                            
                            if not extracted_text:
                                extracted_text = "No text extracted from this page"

                            text_quality = analyze_text_quality(extracted_text)
                            confidence = calculate_final_confidence(page_image_quality, token_confidence, text_quality)
                            
                            successful_pages += 1
                            all_extracted_texts.append(f"--- Page {page_num + 1} ---\n{extracted_text}")
                            
                            yield json.dumps({
                                "type": "page_result",
                                "page_number": page_num + 1,
                                "status": "success",
                                "extracted_text": extracted_text,
                                "page_image": page_image_base64,
                                "confidence": confidence
                            }) + "\n"
                        else:
                            error_msg = result.get("error", "Processing failed")
                            error_pages += 1
                            
                            yield json.dumps({
                                "type": "page_result",
                                "page_number": page_num + 1,
                                "status": "error",
                                "error": error_msg,
                                "page_image": page_image_base64,
                                "confidence": None
                            }) + "\n"
                    
                    except Exception as e:
                        error_pages += 1
                        print(f"Error processing {page_info}: {str(e)}")
                        
                        yield json.dumps({
                            "type": "page_result",
                            "page_number": page_num + 1,
                            "status": "error",
                            "error": f"Error: {str(e)}",
                            "page_image": ""
                        }) + "\n"
                
                pdf_document.close()
                
                # Save to history
                processing_time = time.time() - start_time
                combined_text = "\n\n".join(all_extracted_texts) if all_extracted_texts else ""
                history_status = "success" if successful_pages > 0 else "error"
                error_message = None if successful_pages > 0 else f"Failed to process all {total_pages} pages"
                
                await save_ocr_history(
                    user_id=user_id,
                    file_name=pdf_filename,
                    file_type="pdf",
                    file_size=pdf_file_size,
                    total_pages=total_pages,
                    extracted_text=combined_text,
                    status=history_status,
                    error_message=error_message,
                    processing_time=processing_time,
                    settings={
                        "max_new_tokens": max_new_tokens,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                        "custom_prompt": custom_prompt if custom_prompt else None,
                        "successful_pages": successful_pages,
                        "error_pages": error_pages
                    },
                    blob_url=storage_url
                )
                
                print(f"PDF complete: {successful_pages}/{total_pages} pages, {processing_time:.1f}s")
                
                yield json.dumps({
                    "type": "complete",
                    "status": "success",
                    "successful_pages": successful_pages,
                    "error_pages": error_pages
                }) + "\n"
                
            except Exception as e:
                processing_time = time.time() - start_time
                print(f"PDF processing error: {str(e)}")
                
                await save_ocr_history(
                    user_id=user_id,
                    file_name=pdf_filename,
                    file_type="pdf",
                    file_size=pdf_file_size,
                    total_pages=total_pages,
                    extracted_text="",
                    status="error",
                    error_message=str(e),
                    processing_time=processing_time,
                    settings={
                        "max_new_tokens": max_new_tokens,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    },
                    blob_url=storage_url
                )
                
                yield json.dumps({
                    "type": "error",
                    "error": f"PDF processing error: {str(e)}"
                }) + "\n"
        
        return StreamingResponse(
            process_pages(),
            media_type="application/x-ndjson"
        )
                
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# =============================================================================
# STRUCTURED EXTRACTION ENDPOINT
# =============================================================================

def parse_structured_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from VLM output, handling common issues like markdown code blocks.
    """
    if not text or not text.strip():
        return None
    
    # Remove markdown code blocks if present
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    # Try to find JSON object boundaries
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    
    if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
        return None
    
    json_str = cleaned[start_idx:end_idx + 1]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        # Try to fix common issues
        try:
            # Replace single quotes with double quotes
            fixed = json_str.replace("'", '"')
            return json.loads(fixed)
        except:
            pass
        return None


def build_template_enhanced_prompt(template: Optional[Dict[str, Any]]) -> str:
    """
    Build a structured extraction prompt, optionally enhanced with template field schema.
    """
    base_prompt = STRUCTURED_EXTRACTION_PROMPT
    
    if not template:
        return base_prompt
    
    # Get field schema from template sections
    field_schema = template.get("sections", {})
    if not field_schema:
        return base_prompt
    
    # Build expected fields list
    expected_fields = []
    sections_info = field_schema.get("sections", [])
    
    for section in sections_info:
        section_name = section.get("name", "")
        fields = section.get("fields", [])
        for field in fields:
            label = field.get("label", "")
            field_type = field.get("type", "text")
            if label:
                expected_fields.append(f"- {label} ({field_type})")
    
    if not expected_fields:
        return base_prompt
    
    # Enhance prompt with expected fields
    enhanced_prompt = f"""{base_prompt}

EXPECTED FIELDS (from template):
{chr(10).join(expected_fields)}

Make sure to extract values for ALL of these expected fields. If a field is not found in the image, set its value to ""."""
    
    return enhanced_prompt


@app.post("/api/ocr-structured", response_model=StructuredOCRResponse)
async def process_structured_ocr(
    file: UploadFile = File(...),
    template_id: Optional[str] = Form(None),
    max_new_tokens: int = Form(4096),
    min_pixels: Optional[int] = Form(200704),
    max_pixels: Optional[int] = Form(1003520),
    user: dict = Depends(verify_clerk_token),
):
    """
    Process an image and extract structured data (key-value pairs) from forms.
    Optionally uses a template to improve extraction accuracy.
    
    Returns both raw text and structured JSON data.
    """
    start_time = time.time()
    storage_url = None
    
    try:
        if not RUNPOD_ENDPOINT:
            raise HTTPException(status_code=500, detail="Model service not configured")
        
        # Read and validate image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        image_quality = analyze_image_quality(image)
        
        # Store file
        storage_result = await store_file_to_blob(
            contents,
            file.filename or "form.png",
            file.content_type or "image/png"
        )
        if storage_result.get("stored"):
            storage_url = storage_result.get('url')
        
        # Fetch template if provided
        template_data = None
        if template_id and supabase:
            try:
                template_response = supabase.table("ocr_templates")\
                    .select("*")\
                    .eq("id", template_id)\
                    .single()\
                    .execute()
                if template_response.data:
                    template_data = template_response.data
                    # Increment usage count
                    supabase.table("ocr_templates")\
                        .update({"usage_count": (template_data.get("usage_count", 0) or 0) + 1})\
                        .eq("id", template_id)\
                        .execute()
            except Exception as e:
                print(f"Template fetch error: {e}")
        
        # Build prompt (enhanced with template if available)
        prompt = build_template_enhanced_prompt(template_data)
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        payload = {
            "input": {
                "image": img_base64,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels
            }
        }
        
        # Call RunPod
        result, success = await runpod_client.call_runpod(payload, "structured")
        
        processing_time = time.time() - start_time
        
        if success and result.get("status") == "COMPLETED":
            output = result.get("output", {}) or {}
            raw_text = output.get("text", "")
            token_confidence = output.get("token_confidence")
            
            if not raw_text:
                raw_text = ""
            
            # Parse structured data from response
            structured_data = None
            parsing_successful = False
            
            parsed_json = parse_structured_json(raw_text)
            if parsed_json:
                try:
                    # Convert to our model structure
                    sections = []
                    for section in parsed_json.get("sections", []):
                        fields = []
                        for field in section.get("fields", []):
                            fields.append(ExtractedField(
                                label=field.get("label", ""),
                                value=field.get("value", ""),
                                type=field.get("type", "text")
                            ))
                        sections.append(ExtractedSection(
                            name=section.get("name"),
                            fields=fields
                        ))
                    
                    tables = []
                    for table in parsed_json.get("tables", []):
                        tables.append(ExtractedTable(
                            headers=table.get("headers", []),
                            rows=table.get("rows", [])
                        ))
                    
                    checkboxes = []
                    for cb in parsed_json.get("checkboxes", []):
                        checkboxes.append(ExtractedCheckbox(
                            label=cb.get("label", ""),
                            checked=cb.get("checked", False)
                        ))
                    
                    structured_data = StructuredExtractionData(
                        form_title=parsed_json.get("form_title"),
                        sections=sections,
                        tables=tables,
                        checkboxes=checkboxes
                    )
                    parsing_successful = True
                except Exception as e:
                    print(f"Structured data parsing error: {e}")
            
            text_quality = analyze_text_quality(raw_text)
            confidence = calculate_final_confidence(image_quality, token_confidence, text_quality)
            
            # Save to history
            await save_ocr_history(
                user_id=user.get("sub", "anonymous"),
                file_name=file.filename or "form.png",
                file_type="structured",
                file_size=len(contents),
                total_pages=1,
                extracted_text=raw_text,
                status="success",
                error_message=None,
                processing_time=processing_time,
                settings={
                    "max_new_tokens": max_new_tokens,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "template_id": template_id,
                    "structured_extraction": True,
                    "parsing_successful": parsing_successful
                },
                blob_url=storage_url
            )
            
            return StructuredOCRResponse(
                raw_text=raw_text,
                structured_data=structured_data,
                status="success",
                confidence=confidence,
                parsing_successful=parsing_successful
            )
        else:
            error_msg = result.get("error", "Processing failed")
            
            await save_ocr_history(
                user_id=user.get("sub", "anonymous"),
                file_name=file.filename or "form.png",
                file_type="structured",
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
                    "template_id": template_id,
                    "structured_extraction": True
                },
                blob_url=storage_url
            )
            
            return StructuredOCRResponse(
                raw_text="",
                structured_data=None,
                status="error",
                error=error_msg,
                confidence=None,
                parsing_successful=False
            )
                
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
