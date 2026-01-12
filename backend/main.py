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
import re
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
# RUNPOD CLIENT CONFIGURATION (OPTIMIZED FOR HIGH TRAFFIC & LONG WAITS)
# =============================================================================

# Timeouts (in seconds) - SIGNIFICANTLY INCREASED for heavy traffic scenarios
RUNPOD_CONNECT_TIMEOUT = 60          # Time to establish connection
RUNPOD_READ_TIMEOUT = 1800           # Time to wait for response (30 minutes max)
RUNPOD_INITIAL_REQUEST_TIMEOUT = 120 # Initial POST request timeout

# Retry Configuration
MAX_RETRIES = 5                      # More retries for reliability
RETRY_BASE_DELAY = 3                 # Base delay for exponential backoff (seconds)
RETRY_MAX_DELAY = 60                 # Maximum delay between retries (seconds)

# Polling Configuration (for IN_QUEUE jobs) - Extended for heavy queue
POLL_INTERVAL_INITIAL = 2            # Start polling quickly
POLL_INTERVAL_MAX = 15               # Max poll interval (seconds)
POLL_TIMEOUT_COLD_START = 1200       # Total polling timeout for cold starts (20 minutes)
POLL_TIMEOUT_WARM = 900              # Total polling timeout for warm workers (15 minutes)
POLL_TIMEOUT_HEAVY_LOAD = 1800       # Total polling timeout under heavy load (30 minutes)

# Cold Start Detection
COLD_START_INDICATORS = ["IN_QUEUE", "IN_PROGRESS"]
COLD_START_WAIT_THRESHOLD = 30       # If queued for 30+ seconds, likely cold start

# =============================================================================
# PROCESSING TIME TRACKING (for ETA calculations)
# =============================================================================

# Rolling average of processing times (in seconds)
processing_times_history: list = []
MAX_HISTORY_SIZE = 100  # Keep last 100 processing times

# Average times by operation type (defaults, updated dynamically)
avg_processing_times = {
    "image": 15.0,      # Average time to process single image
    "pdf_page": 20.0,   # Average time to process one PDF page
    "structured": 25.0, # Average time for structured extraction
}

def record_processing_time(operation_type: str, duration: float):
    """Record a processing time for ETA calculations."""
    global processing_times_history, avg_processing_times
    
    processing_times_history.append({
        "type": operation_type,
        "duration": duration,
        "timestamp": time.time()
    })
    
    # Keep only recent history
    if len(processing_times_history) > MAX_HISTORY_SIZE:
        processing_times_history = processing_times_history[-MAX_HISTORY_SIZE:]
    
    # Update rolling average for this operation type
    type_times = [p["duration"] for p in processing_times_history if p["type"] == operation_type]
    if type_times:
        avg_processing_times[operation_type] = sum(type_times) / len(type_times)

def get_avg_processing_time(operation_type: str) -> float:
    """Get average processing time for an operation type."""
    return avg_processing_times.get(operation_type, 20.0)

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

# =============================================================================
# STRUCTURED EXTRACTION PROMPT - OPTIMIZED FOR ARABIC HANDWRITTEN FORMS
# =============================================================================
#
# KEY INSIGHT: Arabic forms have TWO visual layers:
#   1. PRINTED TEMPLATE: Labels + dotted lines (........) as placeholders
#   2. HANDWRITTEN VALUES: Written ON TOP of or NEAR the dotted areas
#
# The VLM must READ THE HANDWRITTEN TEXT, not just see dots!
# =============================================================================

STRUCTURED_EXTRACTION_PROMPT = """اقرأ هذا النموذج العربي واستخرج جميع الحقول والقيم المكتوبة بخط اليد.

Read this Arabic form and extract ALL field names and their HANDWRITTEN values.

THIS IS A FILLED FORM:
- PRINTED TEXT: Field labels like "اسم المالك:", "رقم الهوية:", "تاريخ الميلاد:"
- HANDWRITTEN TEXT: Values written by someone filling the form
- DOTTED LINES: These are just placeholder backgrounds, IGNORE them

YOUR TASK: Find each field label and read the handwritten value next to it.

EXAMPLES OF WHAT TO LOOK FOR:

اسم المالك: [handwritten name like محمد أحمد العلي]
→ Output: اسم المالك: محمد أحمد العلي

رقم الهوية: [handwritten number like ١٠٥٤٣٢١٩٨٧]
→ Output: رقم الهوية: ١٠٥٤٣٢١٩٨٧

تاريخ الميلاد: [handwritten date like ١٤٠٥/٣/١٥]
→ Output: تاريخ الميلاد: ١٤٠٥/٣/١٥

المدينة: [handwritten city like جدة]
→ Output: المدينة: جدة

ONLY USE "-" IF THE FIELD IS COMPLETELY EMPTY (no handwriting at all).

OUTPUT FORMAT:
field_name: value
field_name: value
...

For section headers in boxes:
[قسم] section_name

Now read and extract ALL fields with their handwritten values:"""


# Alternative detailed prompt with more context
DETAILED_EXTRACTION_PROMPT = """أنت قارئ نماذج عربية محترف. اقرأ كل ما هو مكتوب بخط اليد في هذا النموذج.

You are a professional Arabic form reader. Read ALL handwritten content in this form.

UNDERSTANDING THE FORM:
1. This is a PRINTED form template with HANDWRITTEN fill-in values
2. Each field has a LABEL (printed) followed by a VALUE (handwritten)
3. The dots (......) and lines (____) are just placeholder backgrounds - ignore them
4. Focus on reading the HANDWRITTEN TEXT that was written to fill the form

STEP BY STEP:
1. Scan the form for field labels (printed text ending with : or followed by dots)
2. For each label, look at the space next to/below it
3. Read any handwritten text in that space
4. Write: label: handwritten_value

COMMON FIELDS TO FIND:
- Names (اسم، الاسم الكامل)
- ID numbers (رقم الهوية، رقم البطاقة)  
- Dates (تاريخ الميلاد، تاريخ الإصدار)
- Places (المدينة، العنوان)
- Numbers (رقم الهاتف، رقم اللوحة)

OUTPUT: One field per line in format "label: value"
ONLY use "-" for truly empty fields with NO handwriting.

Extract all fields now:"""


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
        
        # Use the longest timeout to ensure we never give up prematurely
        # User experience is better with a long wait than a timeout error
        poll_timeout = POLL_TIMEOUT_HEAVY_LOAD  # Always use max timeout (30 min)
        
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
        return {"error": "Processing took too long due to high server load. Please try again in a few minutes.", "status": "TIMEOUT"}


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


# =============================================================================
# QUEUE STATUS & ETA ENDPOINT
# =============================================================================

class QueueStatusResponse(BaseModel):
    queue_length: int
    workers_running: int
    workers_total: int
    estimated_wait_seconds: int
    estimated_wait_display: str
    avg_processing_time: float
    status: str  # "low_load", "moderate_load", "high_load", "very_high_load"
    message: str


@app.get("/api/queue-status", response_model=QueueStatusResponse)
async def get_queue_status(operation_type: str = "image"):
    """
    Get current RunPod queue status and estimated wait time.
    
    Args:
        operation_type: "image", "pdf_page", or "structured"
    
    Returns:
        Queue status with estimated wait time
    """
    try:
        if not RUNPOD_ENDPOINT or not RUNPOD_API_KEY:
            return QueueStatusResponse(
                queue_length=0,
                workers_running=0,
                workers_total=3,
                estimated_wait_seconds=15,
                estimated_wait_display="~15 seconds",
                avg_processing_time=get_avg_processing_time(operation_type),
                status="unknown",
                message="Queue status unavailable"
            )
        
        # Query RunPod for queue status
        queue_info = await get_runpod_queue_status()
        
        queue_length = queue_info.get("queue_length", 0)
        workers_running = queue_info.get("workers_running", 0)
        workers_total = queue_info.get("workers_total", 3)
        
        # Get average processing time for this operation type
        avg_time = get_avg_processing_time(operation_type)
        
        # Calculate estimated wait time
        # If queue is empty and workers available, minimal wait
        if queue_length == 0 and workers_running < workers_total:
            estimated_wait = int(avg_time)
            status = "low_load"
            message = "Servers ready - processing will start immediately"
        elif queue_length == 0:
            # All workers busy but no queue
            estimated_wait = int(avg_time * 1.5)
            status = "low_load"
            message = "Servers busy - short wait expected"
        elif queue_length <= 2:
            # Small queue
            estimated_wait = int((queue_length + 1) * avg_time)
            status = "moderate_load"
            message = f"{queue_length} request(s) ahead of you"
        elif queue_length <= 5:
            # Moderate queue
            estimated_wait = int((queue_length + 1) * avg_time * 1.2)  # Add buffer
            status = "high_load"
            message = f"{queue_length} requests in queue - please be patient"
        else:
            # Heavy load
            estimated_wait = int((queue_length + 1) * avg_time * 1.5)  # Add larger buffer
            status = "very_high_load"
            message = f"High traffic ({queue_length} in queue) - longer wait expected"
        
        # Format display string
        if estimated_wait < 60:
            display = f"~{estimated_wait} seconds"
        elif estimated_wait < 120:
            display = "~1-2 minutes"
        elif estimated_wait < 300:
            minutes = estimated_wait // 60
            display = f"~{minutes}-{minutes+1} minutes"
        elif estimated_wait < 600:
            display = "~5-10 minutes"
        elif estimated_wait < 1200:
            display = "~10-20 minutes"
        else:
            display = "~20+ minutes"
        
        return QueueStatusResponse(
            queue_length=queue_length,
            workers_running=workers_running,
            workers_total=workers_total,
            estimated_wait_seconds=estimated_wait,
            estimated_wait_display=display,
            avg_processing_time=avg_time,
            status=status,
            message=message
        )
        
    except Exception as e:
        print(f"Queue status error: {e}")
        # Return safe defaults
        return QueueStatusResponse(
            queue_length=0,
            workers_running=0,
            workers_total=3,
            estimated_wait_seconds=30,
            estimated_wait_display="~30 seconds",
            avg_processing_time=get_avg_processing_time(operation_type),
            status="unknown",
            message="Processing will begin shortly"
        )


async def get_runpod_queue_status() -> dict:
    """Query RunPod API for current queue status."""
    try:
        if not RUNPOD_ENDPOINT or not RUNPOD_API_KEY:
            return {"queue_length": 0, "workers_running": 0, "workers_total": 3}
        
        # Extract endpoint ID from URL
        # URL format: https://api.runpod.ai/v2/{endpoint_id}/...
        import re
        match = re.search(r'/v2/([a-zA-Z0-9]+)/', RUNPOD_ENDPOINT)
        if not match:
            return {"queue_length": 0, "workers_running": 0, "workers_total": 3}
        
        endpoint_id = match.group(1)
        
        # Query RunPod health endpoint for queue info
        health_url = f"https://api.runpod.ai/v2/{endpoint_id}/health"
        
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                health_url,
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # RunPod health response includes workers and jobs info
                workers = data.get("workers", {})
                jobs = data.get("jobs", {})
                
                return {
                    "queue_length": jobs.get("inQueue", 0) + jobs.get("inProgress", 0),
                    "workers_running": workers.get("running", 0),
                    "workers_total": workers.get("ready", 0) + workers.get("running", 0),
                    "workers_idle": workers.get("idle", 0),
                    "jobs_completed": jobs.get("completed", 0),
                    "jobs_failed": jobs.get("failed", 0)
                }
            else:
                print(f"RunPod health check returned {response.status_code}")
                return {"queue_length": 0, "workers_running": 0, "workers_total": 3}
                
    except Exception as e:
        print(f"Error getting RunPod queue status: {e}")
        return {"queue_length": 0, "workers_running": 0, "workers_total": 3}


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
        
        # Record processing time for ETA calculations
        record_processing_time("image", processing_time)

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


@app.post("/api/ocr-pdf-structured")
async def process_pdf_structured_ocr(
    file: UploadFile = File(...),
    template_id: Optional[str] = Form(None),
    max_new_tokens: int = Form(4096),
    min_pixels: Optional[int] = Form(200704),
    max_pixels: Optional[int] = Form(1003520),
    user: dict = Depends(verify_clerk_token),
):
    """
    Process a PDF with structured extraction (key-value pairs) for each page.
    Streams results page by page with structured data.
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
            except Exception as e:
                print(f"Template fetch error: {e}")
        
        # Build prompt
        prompt = build_template_enhanced_prompt(template_data)
        
        async def process_pages_structured():
            all_structured_data = []
            successful_pages = 0
            error_pages = 0
            total_pages = 0
            
            try:
                pdf_document = fitz.open(stream=pdf_contents, filetype="pdf")
                total_pages = len(pdf_document)
                
                print(f"Processing PDF (structured): {pdf_filename} ({total_pages} pages)")
                
                # Send metadata
                yield json.dumps({
                    "type": "metadata",
                    "total_pages": total_pages
                }) + "\n"
                
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
                        
                        # Call RunPod
                        result, success = await runpod_client.call_runpod(payload, f"{page_info} structured")
                        
                        if success and result.get("status") == "COMPLETED":
                            page_output = result.get("output", {}) or {}
                            raw_text = page_output.get("text", "")
                            token_confidence = page_output.get("token_confidence")
                            
                            # Parse structured data
                            parsed = parse_structured_response(raw_text)
                            
                            # Convert to response format
                            sections = []
                            for section in parsed.get("sections", []):
                                fields = []
                                for field in section.get("fields", []):
                                    fields.append({
                                        "label": field.get("label", ""),
                                        "value": field.get("value", ""),
                                        "type": field.get("type", "text")
                                    })
                                if fields:
                                    sections.append({
                                        "name": section.get("name"),
                                        "fields": fields
                                    })
                            
                            page_structured_data = {
                                "form_title": parsed.get("form_title"),
                                "sections": sections,
                                "tables": parsed.get("tables", []),
                                "checkboxes": parsed.get("checkboxes", [])
                            }
                            
                            total_fields = sum(len(s["fields"]) for s in sections)
                            parsing_successful = total_fields > 0
                            
                            successful_pages += 1
                            all_structured_data.append({
                                "page": page_num + 1,
                                "data": page_structured_data
                            })
                            
                            text_quality = analyze_text_quality(raw_text)
                            confidence = calculate_final_confidence(page_image_quality, token_confidence, text_quality)
                            
                            yield json.dumps({
                                "type": "page_result",
                                "page_number": page_num + 1,
                                "status": "success",
                                "raw_text": raw_text,
                                "structured_data": page_structured_data,
                                "parsing_successful": parsing_successful,
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
                                "page_image": page_image_base64
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
                
                # Combine all structured data into text for history
                combined_text = json.dumps({
                    "pages": all_structured_data,
                    "total_pages": total_pages,
                    "successful_pages": successful_pages
                }, ensure_ascii=False, indent=2)
                
                history_status = "success" if successful_pages > 0 else "error"
                error_message = None if successful_pages > 0 else f"Failed to process all {total_pages} pages"
                
                await save_ocr_history(
                    user_id=user_id,
                    file_name=pdf_filename,
                    file_type="pdf-structured",
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
                        "template_id": template_id,
                        "structured_extraction": True,
                        "successful_pages": successful_pages,
                        "error_pages": error_pages
                    },
                    blob_url=storage_url
                )
                
                print(f"PDF structured complete: {successful_pages}/{total_pages} pages, {processing_time:.1f}s")
                
                yield json.dumps({
                    "type": "complete",
                    "status": "success",
                    "successful_pages": successful_pages,
                    "error_pages": error_pages
                }) + "\n"
                
            except Exception as e:
                processing_time = time.time() - start_time
                print(f"PDF structured error: {str(e)}")
                
                await save_ocr_history(
                    user_id=user_id,
                    file_name=pdf_filename,
                    file_type="pdf-structured",
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
                        "template_id": template_id,
                        "structured_extraction": True
                    },
                    blob_url=storage_url
                )
                
                yield json.dumps({
                    "type": "error",
                    "error": f"PDF processing error: {str(e)}"
                }) + "\n"
        
        return StreamingResponse(
            process_pages_structured(),
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

def normalize_empty_value(value: str) -> str:
    """
    Normalize empty field values to empty string.
    
    IMPORTANT: Be CONSERVATIVE - only mark as empty if we're CERTAIN there's no value.
    Arabic forms often have dots/lines as placeholders, but the actual handwritten
    values might be mixed with these characters.
    """
    if not value:
        return ""

    value = value.strip()
    
    # If empty after strip
    if not value:
        return ""

    # EXACT empty indicators (be strict - must match exactly)
    exact_empty = [
        '-', '—', '−',  # Single dashes only
        '[فارغ]', '[empty]', 'فارغ', 'empty', 'n/a', 'N/A', 'none', 'None',
        '/ /', '/ / /', 
    ]
    
    if value in exact_empty:
        return ""
    
    # Check if value is PURELY dots/dashes/underscores with NO letters/digits
    # This catches "............" but NOT "محمد............" or "١٢٣...."
    if re.match(r'^[\.\-_\s/]+$', value):
        return ""
    
    # Check if it's just an empty Islamic date placeholder "/ / ١٤هـ" with NO actual date
    # But NOT if there are actual numbers like "١٤٤٥/١/١هـ"
    if re.match(r'^[\s/]*١٤هـ[\s/]*$', value):
        return ""
    
    # If value contains ANY Arabic letters or actual digits, it's NOT empty
    # This is the key check - real values have letters or numbers
    has_arabic_letters = bool(re.search(r'[\u0600-\u06FF]', value.replace('١٤هـ', '')))  # Exclude date marker
    has_arabic_digits = bool(re.search(r'[٠-٩]', value))
    has_western_chars = bool(re.search(r'[a-zA-Z0-9]', value))
    
    if has_arabic_letters or has_arabic_digits or has_western_chars:
        # It has actual content - clean it up by removing trailing dots/lines
        # but keep the actual text
        cleaned = re.sub(r'[\.\-_]+$', '', value).strip()  # Remove trailing dots
        cleaned = re.sub(r'^[\.\-_]+', '', cleaned).strip()  # Remove leading dots
        return cleaned if cleaned else value
    
    # Default: keep the value (be conservative)
    return value


def split_field_value(text: str) -> tuple:
    """Split a text segment into label and value, handling Arabic forms."""
    if ':' not in text:
        return ("", "")
    
    # Find the most appropriate colon
    colon_idx = -1
    for idx, char in enumerate(text):
        if char == ':':
            before = text[:idx].strip()
            after = text[idx+1:].strip() if idx + 1 < len(text) else ""
            
            # Skip time patterns like 12:30
            if before and after and before[-1].isdigit() and after and after[0].isdigit():
                if len(after) > 2 and '/' in after:
                    colon_idx = idx
                    break
                continue
            
            colon_idx = idx
            break
    
    if colon_idx <= 0:
        return ("", "")
    
    label = text[:colon_idx].strip()
    value = text[colon_idx + 1:].strip()
    
    return (label, value)


def parse_structured_response(text: str) -> Dict[str, Any]:
    """
    Parse the VLM's structured extraction response.
    Handles multiple formats with robust parsing:
    1. Natural "label: value" format (primary)
    2. [FIELD]/[VALUE] format (including 3-line values)
    3. FIELD:/VALUE: format (legacy)
    4. Bullet/numbered lists
    5. JSON format (fallback)

    Returns a structured dict with sections, tables, checkboxes.
    """
    if not text or not text.strip():
        return {"sections": [], "tables": [], "checkboxes": [], "form_title": None}

    text = text.strip()

    # Filter out common garbage patterns
    garbage_patterns = [
        "here's the extracted", "here is the extracted", "i'll extract",
        "let me extract", "the form contains", "based on the image",
        "from the form", "extracted data:", "form data:", "your task",
        "begin extraction", "extract everything", "output format",
        "instructions:", "important:", "rules:"
    ]

    # Remove garbage intro lines but keep content
    lines = text.split('\n')
    filtered_lines = []
    started = False

    for line in lines:
        line_lower = line.lower().strip()
        
        if not started:
            is_garbage = any(pat in line_lower for pat in garbage_patterns)
            # Start when we see actual content
            has_arabic = re.search(r'[\u0600-\u06FF]', line)
            has_colon = ':' in line
            has_field_marker = '[field]' in line_lower or line_lower.startswith('field:')
            has_bullet = line.strip().startswith(('•', '-', '*', '١', '٢', '٣')) or re.match(r'^\d+[\.\)]', line.strip())
            
            if has_field_marker or (has_arabic and has_colon) or has_bullet:
                started = True
                filtered_lines.append(line)
            elif not is_garbage and line.strip() and len(line.strip()) > 2:
                if has_colon or has_bullet:
                    started = True
                    filtered_lines.append(line)
        else:
            filtered_lines.append(line)

    text = '\n'.join(filtered_lines)
    
    # Try JSON parsing first
    if text.strip().startswith("{"):
        try:
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx > start_idx:
                parsed = json.loads(text[start_idx:end_idx + 1])
                if isinstance(parsed.get("sections"), list):
                    return parsed
        except:
            pass

    fields = []
    current_section = None
    checkboxes = []
    form_title = None

    lines = text.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Pattern 0: Arabic section headers [قسم] or "بيانات ..."
        section_match = re.match(r'^\[قسم\]\s*(.+)$', line)
        if section_match:
            current_section = section_match.group(1).strip()
            i += 1
            continue
        
        # Detect section headers like "بيانات عامة" or "بيانات المركبة"
        if re.match(r'^بيانات\s+[\u0600-\u06FF]+', line) and ':' not in line:
            current_section = line.strip()
            i += 1
            continue
        
        # Detect checklist/requirements sections: "طلبات الإصدار الجديد :" or "طلبات التجديد :"
        checklist_match = re.match(r'^(طلبات|متطلبات|شروط)\s+[\u0600-\u06FF\s]+\s*:?\s*$', line)
        if checklist_match:
            current_section = line.replace(':', '').strip()
            i += 1
            continue
        
        # Detect [قائمة] list marker
        list_marker_match = re.match(r'^\[قائمة\]\s*(.+)$', line)
        if list_marker_match:
            current_section = list_marker_match.group(1).strip()
            i += 1
            continue
        
        # Pattern 1: [FIELD] with multi-line value support
        if line.upper().startswith('[FIELD]'):
            field_label = line[7:].strip()
            value = ""
            
            # Check if [VALUE] is inline (e.g., "[FIELD] label [VALUE] value")
            if '[VALUE]' in field_label.upper():
                parts = re.split(r'\[VALUE\]', field_label, flags=re.IGNORECASE)
                field_label = parts[0].strip()
                value = parts[1].strip() if len(parts) > 1 else ""
            else:
                # Look for [VALUE] on next lines (up to 3 lines ahead)
                j = i + 1
                while j < len(lines) and j <= i + 3:
                    next_line = lines[j].strip()
                    
                    if next_line.upper().startswith('[VALUE]'):
                        value_content = next_line[7:].strip()
                        
                        # If [VALUE] line has content, use it
                        if value_content and value_content != '-':
                            value = value_content
                            i = j
                            break
                        else:
                            # CRITICAL FIX: Value might be on the NEXT line after [VALUE]
                            if j + 1 < len(lines):
                                potential = lines[j + 1].strip()
                                # Check it's not another marker
                                if (potential and 
                                    not potential.upper().startswith('[FIELD]') and
                                    not potential.upper().startswith('[VALUE]') and
                                    not potential.upper().startswith('FIELD:') and
                                    not potential.upper().startswith('[SECTION]')):
                                    value = potential
                                    i = j + 1
                                    break
                            i = j
                            break
                    elif next_line.upper().startswith('[FIELD]'):
                        break  # Next field found
                    j += 1

            if field_label and len(field_label) > 1:
                if not any(pat in field_label.lower() for pat in garbage_patterns):
                    fields.append({
                        "label": field_label,
                        "value": value if value != '-' else "",
                        "type": infer_field_type(field_label, value),
                        "section": current_section
                    })
            i += 1
            continue

        # Pattern 2: Section headers
        if line.upper().startswith('SECTION:') or line.upper().startswith('[SECTION]'):
            marker_len = 9 if '[SECTION]' in line.upper() else 8
            current_section = line[marker_len:].strip()
            i += 1
            continue

        # Pattern 3: Bullet points and numbered lists (Western + Arabic numerals)
        bullet_match = re.match(r'^[•\-\*]\s*(.+)$', line)
        # Match both Western (1, 2, 3) and Arabic (١، ٢، ٣) numerals with various separators
        number_match = re.match(r'^[\d١٢٣٤٥٦٧٨٩٠]+[\.\)\-–]\s*(.+)$', line)
        
        if bullet_match or number_match:
            content = (bullet_match or number_match).group(1).strip()
            # Remove trailing period/dot
            content = re.sub(r'\s*\.\s*$', '', content)
            
            if ':' in content:
                # This is a field:value pair within a list
                label, value = split_field_value(content)
                if label and len(label) > 1 and len(label) < 80:
                    fields.append({
                        "label": label,
                        "value": normalize_empty_value(value),
                        "type": infer_field_type(label, value),
                        "section": current_section
                    })
            else:
                # This is a standalone list item (checklist/requirements)
                if content and len(content) > 2 and len(content) < 200:
                    fields.append({
                        "label": content,
                        "value": "",  # No value for checklist items
                        "type": "list_item",
                        "section": current_section
                    })
            i += 1
            continue

        # Pattern 4: Legacy FIELD:/VALUE: format
        if line.upper().startswith('FIELD:'):
            field_label = line[6:].strip()
            value = ""
            
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.upper().startswith('VALUE:'):
                    value = next_line[6:].strip()
                    # Check next line for value if empty
                    if not value and i + 2 < len(lines):
                        potential = lines[i + 2].strip()
                        if not potential.upper().startswith('FIELD:'):
                            value = potential
                            i += 1
                    i += 1
            
            if field_label and len(field_label) > 1:
                if value.upper() in ['CHECKED', 'UNCHECKED', '☑', '☐', 'نعم', 'لا']:
                    checkboxes.append({
                        "label": field_label,
                        "checked": value.upper() in ['CHECKED', '☑', 'نعم']
                    })
                else:
                    fields.append({
                        "label": field_label,
                        "value": value if value not in ['-', 'EMPTY'] else "",
                        "type": infer_field_type(field_label, value),
                        "section": current_section
                    })
            i += 1
            continue
        
        # Pattern 5: Natural "label: value" format - HANDLES MULTIPLE FIELDS PER LINE
        if ':' in line and not line.startswith('http') and not line.startswith('['):
            # Split by common table separators (multiple spaces, tabs, pipes)
            # This handles forms where multiple fields appear on the same row
            segments = re.split(r'\s{3,}|\t|\|', line)
            
            for segment in segments:
                segment = segment.strip()
                if not segment or ':' not in segment:
                    continue
                
                label, value = split_field_value(segment)
                
                # Clean label (remove bullets, numbers)
                label = re.sub(r'^[•\-\*\d\.\)]+\s*', '', label).strip()
                
                # Validate
                if (label and 
                    len(label) > 1 and 
                    len(label) < 80 and 
                    not label.isdigit() and
                    not any(pat in label.lower() for pat in garbage_patterns)):
                    
                    # Normalize empty values
                    normalized_value = normalize_empty_value(value)
                    
                    # Check for checkbox values
                    if value in ['☑', '☐', '✓', '✗'] or value.lower() in ['نعم', 'لا', 'yes', 'no']:
                        checkboxes.append({
                            "label": label,
                            "checked": value in ['☑', '✓'] or value.lower() in ['نعم', 'yes']
                        })
                    else:
                        fields.append({
                            "label": label,
                            "value": normalized_value,
                            "type": infer_field_type(label, value),
                            "section": current_section
                        })
        
        i += 1
    
    # Group fields by section
    sections_dict: Dict[str, List] = {}
    for field in fields:
        section_name = field.pop("section", None) or "General"
        if section_name not in sections_dict:
            sections_dict[section_name] = []
        sections_dict[section_name].append(field)
    
    sections = [
        {"name": name if name != "General" else None, "fields": section_fields}
        for name, section_fields in sections_dict.items()
    ]
    
    return {
        "form_title": form_title,
        "sections": sections,
        "tables": [],
        "checkboxes": checkboxes
    }


def infer_field_type(label: str, value: str) -> str:
    """Infer field type from label and value."""
    label_lower = label.lower()
    
    # Date patterns
    date_keywords = ['date', 'تاريخ', 'يوم', 'شهر', 'سنة', 'الميلاد', 'التسجيل', 'الإصدار', 'الانتهاء']
    if any(kw in label_lower or kw in label for kw in date_keywords):
        return 'date'
    
    # Number patterns  
    number_keywords = ['number', 'رقم', 'هاتف', 'جوال', 'هوية', 'جواز', 'عدد', 'كمية', 'سعر', 'مبلغ']
    if any(kw in label_lower or kw in label for kw in number_keywords):
        return 'number'
    
    # Check if value looks like a number
    if value and re.match(r'^[\d\s\-\+\.٠-٩]+$', value.replace(' ', '')):
        return 'number'
    
    # Check if value looks like a date
    if value and re.match(r'^\d{1,4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,4}$', value.strip()):
        return 'date'
    
    return 'text'


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
        
        if section_name:
            expected_fields.append(f"\nSECTION: {section_name}")
        
        for field in fields:
            label = field.get("label", "")
            if label:
                expected_fields.append(f"FIELD: {label}")
    
    if not expected_fields:
        return base_prompt
    
    # Enhance prompt with expected fields
    enhanced_prompt = f"""{base_prompt}

IMPORTANT: This form should contain these specific fields. Find and extract them:
{chr(10).join(expected_fields)}

Extract these fields AND any other fields you see in the form."""
    
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
        
        # Record processing time for ETA calculations
        record_processing_time("structured", processing_time)

        if success and result.get("status") == "COMPLETED":
            output = result.get("output", {}) or {}
            raw_text = output.get("text", "")
            token_confidence = output.get("token_confidence")

            if not raw_text:
                raw_text = ""

            # Parse structured data from response using robust parser
            structured_data = None
            parsing_successful = False

            try:
                parsed = parse_structured_response(raw_text)
                
                # Check if we got any meaningful data
                has_data = (
                    len(parsed.get("sections", [])) > 0 or
                    len(parsed.get("tables", [])) > 0 or
                    len(parsed.get("checkboxes", [])) > 0
                )
                
                # Also check if sections have actual fields
                total_fields = sum(
                    len(s.get("fields", [])) 
                    for s in parsed.get("sections", [])
                )
                
                if has_data or total_fields > 0:
                    # Convert to our model structure
                    sections = []
                    for section in parsed.get("sections", []):
                        fields = []
                        for field in section.get("fields", []):
                            fields.append(ExtractedField(
                                label=field.get("label", ""),
                                value=field.get("value", ""),
                                type=field.get("type", "text")
                            ))
                        if fields:  # Only add sections with fields
                            sections.append(ExtractedSection(
                                name=section.get("name"),
                                fields=fields
                            ))

                    tables = []
                    for table in parsed.get("tables", []):
                        tables.append(ExtractedTable(
                            headers=table.get("headers", []),
                            rows=table.get("rows", [])
                        ))

                    checkboxes = []
                    for cb in parsed.get("checkboxes", []):
                        checkboxes.append(ExtractedCheckbox(
                            label=cb.get("label", ""),
                            checked=cb.get("checked", False)
                        ))

                    structured_data = StructuredExtractionData(
                        form_title=parsed.get("form_title"),
                        sections=sections,
                        tables=tables,
                        checkboxes=checkboxes
                    )
                    parsing_successful = total_fields > 0
                    
                    print(f"Structured extraction: {total_fields} fields, {len(checkboxes)} checkboxes")
                else:
                    print(f"No structured data extracted from response")
                    
            except Exception as e:
                print(f"Structured data parsing error: {e}")
                traceback.print_exc()
            
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
