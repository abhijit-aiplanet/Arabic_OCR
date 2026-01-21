"""
API Clients for Agentic OCR System

Provides clients for communicating with:
- VLM (AIN model) on RunPod for vision/OCR tasks
- LLM (Qwen 2.5) on RunPod for reasoning tasks
"""

import httpx
import asyncio
import base64
import io
import json
import time
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image

from .prompts import (
    INITIAL_OCR_PROMPT,
    ANALYSIS_PROMPT,
    REGION_ESTIMATION_PROMPT,
    FIELD_REEXTRACT_PROMPT,
    MERGE_PROMPT,
    ANALYSIS_SYSTEM_PROMPT,
    REGION_SYSTEM_PROMPT,
    MERGE_SYSTEM_PROMPT,
    get_content_hint,
)
from .models import AnalysisResult, RegionEstimate, MergeResult


# =============================================================================
# BASE CLIENT
# =============================================================================

class BaseRunPodClient:
    """Base client for RunPod API calls."""
    
    def __init__(
        self,
        endpoint_url: str,
        api_key: str,
        timeout: float = 300.0,  # 5 minutes default
        max_retries: int = 3
    ):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=10)
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def _call_runpod(self, payload: dict) -> Tuple[dict, bool]:
        """
        Call RunPod endpoint with retry logic.
        
        Returns:
            Tuple of (result_dict, success_bool)
        """
        client = await self.get_client()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    self.endpoint_url,
                    json=payload,
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Check for RunPod status
                    status = result.get("status", "").upper()
                    
                    if status == "COMPLETED":
                        return result.get("output", {}), True
                    elif status == "IN_QUEUE" or status == "IN_PROGRESS":
                        # Poll for completion
                        job_id = result.get("id")
                        if job_id:
                            return await self._poll_job(job_id)
                    elif status == "FAILED":
                        return {"error": result.get("error", "Job failed")}, False
                    else:
                        # Direct response (for sync endpoints)
                        return result.get("output", result), True
                
                elif response.status_code == 429:
                    # Rate limited, wait and retry
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    
            except httpx.TimeoutException:
                last_error = "Request timed out"
            except Exception as e:
                last_error = str(e)
            
            # Wait before retry
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        
        return {"error": last_error or "Unknown error"}, False
    
    async def _poll_job(self, job_id: str, max_wait: float = 300.0) -> Tuple[dict, bool]:
        """Poll for job completion."""
        client = await self.get_client()
        status_url = self.endpoint_url.replace("/runsync", f"/status/{job_id}")
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        start_time = time.time()
        poll_interval = 1.0
        
        while time.time() - start_time < max_wait:
            try:
                response = await client.get(status_url, headers=headers)
                
                if response.status_code == 200:
                    result = response.json()
                    status = result.get("status", "").upper()
                    
                    if status == "COMPLETED":
                        return result.get("output", {}), True
                    elif status == "FAILED":
                        return {"error": result.get("error", "Job failed")}, False
                    elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                        await asyncio.sleep(poll_interval)
                        poll_interval = min(poll_interval * 1.2, 5.0)
                        continue
                
            except Exception as e:
                print(f"Poll error: {e}")
            
            await asyncio.sleep(poll_interval)
        
        return {"error": "Job timed out"}, False


# =============================================================================
# VLM CLIENT (AIN Model for Vision/OCR)
# =============================================================================

class VLMClient(BaseRunPodClient):
    """
    Client for AIN VLM model on RunPod.
    
    Used for:
    - Initial full-page OCR extraction
    - Focused field re-extraction from cropped regions
    """
    
    def __init__(
        self,
        endpoint_url: str,
        api_key: str,
        default_max_tokens: int = 8192,
        default_min_pixels: int = 200704,
        default_max_pixels: int = 1317120,  # Optimized for handwriting
        **kwargs
    ):
        super().__init__(endpoint_url, api_key, **kwargs)
        self.default_max_tokens = default_max_tokens
        self.default_min_pixels = default_min_pixels
        self.default_max_pixels = default_max_pixels
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        # Use PNG for lossless quality
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    async def extract_initial(
        self,
        image: Image.Image,
        custom_prompt: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Perform initial full-page OCR extraction.
        
        Args:
            image: PIL Image of the form
            custom_prompt: Optional custom prompt (uses INITIAL_OCR_PROMPT by default)
            
        Returns:
            Tuple of (extracted_text, success_bool)
        """
        prompt = custom_prompt or INITIAL_OCR_PROMPT
        
        payload = {
            "input": {
                "image": self._image_to_base64(image),
                "prompt": prompt,
                "max_new_tokens": self.default_max_tokens,
                "min_pixels": self.default_min_pixels,
                "max_pixels": self.default_max_pixels,
            }
        }
        
        result, success = await self._call_runpod(payload)
        
        if success:
            return result.get("text", ""), True
        else:
            return result.get("error", "VLM extraction failed"), False
    
    async def extract_field(
        self,
        cropped_image: Image.Image,
        field_name: str,
        previous_value: str = "",
        content_hint: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Extract a specific field from a cropped region.
        
        Args:
            cropped_image: Cropped PIL Image of the field region
            field_name: Name of the field being extracted
            previous_value: Previous extraction value for context
            content_hint: Hint about expected content
            
        Returns:
            Tuple of (extracted_value, success_bool)
        """
        hint = content_hint or get_content_hint(field_name)
        
        prompt = FIELD_REEXTRACT_PROMPT.format(
            field_name=field_name,
            content_hint=hint,
            previous_value=previous_value or "[not previously extracted]"
        )
        
        # Use higher resolution for cropped regions (they're smaller)
        payload = {
            "input": {
                "image": self._image_to_base64(cropped_image),
                "prompt": prompt,
                "max_new_tokens": 512,  # Field values are short
                "min_pixels": self.default_min_pixels,
                "max_pixels": self.default_max_pixels,
            }
        }
        
        result, success = await self._call_runpod(payload)
        
        if success:
            text = result.get("text", "").strip()
            # Clean up common artifacts
            text = text.replace("Your transcription:", "").strip()
            return text, True
        else:
            return result.get("error", "Field extraction failed"), False


# =============================================================================
# LLM CLIENT (Qwen 2.5 for Reasoning)
# =============================================================================

class LLMClient(BaseRunPodClient):
    """
    Client for Qwen 2.5 LLM on RunPod.
    
    Used for:
    - Analyzing OCR output for issues
    - Estimating field region coordinates
    - Merging and reconciling multiple OCR passes
    """
    
    def __init__(
        self,
        endpoint_url: str,
        api_key: str,
        default_max_tokens: int = 4096,
        default_temperature: float = 0.1,
        **kwargs
    ):
        super().__init__(endpoint_url, api_key, **kwargs)
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
    
    async def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        parse_json: bool = True,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Tuple[Any, bool]:
        """
        Call LLM with prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            parse_json: Whether to parse response as JSON
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Tuple of (result, success_bool)
        """
        payload = {
            "input": {
                "prompt": prompt,
                "max_tokens": max_tokens or self.default_max_tokens,
                "temperature": temperature if temperature is not None else self.default_temperature,
                "system_prompt": system_prompt,
                "parse_json": parse_json,
            }
        }
        
        result, success = await self._call_runpod(payload)
        
        if success:
            if parse_json and "json" in result:
                return result["json"], True
            return result.get("text", ""), True
        else:
            return result.get("error", "LLM call failed"), False
    
    async def analyze(self, initial_extraction: str) -> Tuple[Optional[AnalysisResult], bool]:
        """
        Analyze OCR extraction for issues.
        
        Args:
            initial_extraction: Raw OCR output text
            
        Returns:
            Tuple of (AnalysisResult or None, success_bool)
        """
        prompt = ANALYSIS_PROMPT.format(initial_extraction=initial_extraction)
        
        result, success = await self._call_llm(
            prompt=prompt,
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
            parse_json=True
        )
        
        if success and isinstance(result, dict):
            try:
                return AnalysisResult.from_dict(result), True
            except Exception as e:
                print(f"Failed to parse analysis result: {e}")
                return None, False
        
        return None, False
    
    async def estimate_regions(
        self,
        fields_to_reexamine: List[Dict[str, Any]],
        image_width: int,
        image_height: int
    ) -> Tuple[List[RegionEstimate], bool]:
        """
        Estimate bounding box coordinates for fields.
        
        Args:
            fields_to_reexamine: List of field dicts from analysis
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Tuple of (list of RegionEstimate, success_bool)
        """
        prompt = REGION_ESTIMATION_PROMPT.format(
            fields_to_reexamine=json.dumps(fields_to_reexamine, ensure_ascii=False, indent=2),
            width=image_width,
            height=image_height
        )
        
        result, success = await self._call_llm(
            prompt=prompt,
            system_prompt=REGION_SYSTEM_PROMPT,
            parse_json=True
        )
        
        if success and isinstance(result, dict):
            try:
                regions = []
                for r in result.get("regions", []):
                    regions.append(RegionEstimate.from_dict(r))
                return regions, True
            except Exception as e:
                print(f"Failed to parse region estimates: {e}")
                return [], False
        
        return [], False
    
    async def merge(
        self,
        original_extraction: str,
        refined_values: Dict[str, str]
    ) -> Tuple[Optional[MergeResult], bool]:
        """
        Merge original and refined OCR results.
        
        Args:
            original_extraction: Original full-page OCR text
            refined_values: Dict of field_name -> refined_value from cropped re-OCR
            
        Returns:
            Tuple of (MergeResult or None, success_bool)
        """
        prompt = MERGE_PROMPT.format(
            original_extraction=original_extraction,
            refined_values=json.dumps(refined_values, ensure_ascii=False, indent=2)
        )
        
        result, success = await self._call_llm(
            prompt=prompt,
            system_prompt=MERGE_SYSTEM_PROMPT,
            parse_json=True
        )
        
        if success and isinstance(result, dict):
            try:
                return MergeResult.from_dict(result), True
            except Exception as e:
                print(f"Failed to parse merge result: {e}")
                return None, False
        
        return None, False


# =============================================================================
# CLIENT FACTORY
# =============================================================================

def create_clients(
    vlm_endpoint: str,
    vlm_api_key: str,
    llm_endpoint: str,
    llm_api_key: str,
    **kwargs
) -> Tuple[VLMClient, LLMClient]:
    """
    Create VLM and LLM clients.
    
    Args:
        vlm_endpoint: RunPod endpoint URL for VLM
        vlm_api_key: API key for VLM endpoint
        llm_endpoint: RunPod endpoint URL for LLM
        llm_api_key: API key for LLM endpoint
        
    Returns:
        Tuple of (VLMClient, LLMClient)
    """
    vlm_client = VLMClient(vlm_endpoint, vlm_api_key, **kwargs)
    llm_client = LLMClient(llm_endpoint, llm_api_key, **kwargs)
    return vlm_client, llm_client
