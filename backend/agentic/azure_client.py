"""
Azure OpenAI Vision Client for Surgical OCR

Wraps the Azure OpenAI GPT-4o-mini vision API for:
- Section-by-section OCR extraction
- Field-level zoom-in refinement
- Self-critique analysis
- Multi-pass verification

Key features:
- Async support for parallel processing
- Automatic retry with backoff
- Response parsing and validation
"""

import os
import asyncio
import json
import re
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

# Azure OpenAI SDK
try:
    from openai import AzureOpenAI, AsyncAzureOpenAI
except ImportError:
    raise ImportError(
        "openai package not installed. Install with: pip install openai>=1.0.0"
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default Azure OpenAI settings from environment
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "o4-mini")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExtractionResult:
    """Result from an OCR extraction call."""
    text: str
    success: bool
    error: Optional[str] = None
    tokens_used: int = 0
    raw_response: Optional[Dict] = None


@dataclass
class FieldExtraction:
    """Extracted value for a single field."""
    field_name: str
    value: str
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    is_empty: bool = False
    is_unclear: bool = False
    notes: Optional[str] = None


@dataclass
class CritiqueResult:
    """Result from self-critique analysis."""
    has_issues: bool
    issues: List[Dict[str, Any]]
    fields_to_recheck: List[str]
    overall_confidence: str
    raw_response: Optional[str] = None


# =============================================================================
# AZURE VISION OCR CLIENT
# =============================================================================

class AzureVisionOCR:
    """
    Azure OpenAI GPT-4o-mini Vision client for surgical OCR.
    
    Provides methods for:
    - Full document extraction
    - Section-specific extraction
    - Field-level zoom extraction
    - Self-critique analysis
    - Multi-pass verification
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the Azure Vision OCR client.
        
        Args:
            api_key: Azure OpenAI API key (or use env var)
            endpoint: Azure OpenAI endpoint URL (or use env var)
            deployment: Deployment name (or use env var)
            api_version: API version (or use env var)
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries (exponential backoff)
        """
        self.api_key = api_key or AZURE_OPENAI_API_KEY
        self.endpoint = endpoint or AZURE_OPENAI_ENDPOINT
        self.deployment = deployment or AZURE_DEPLOYMENT
        self.api_version = api_version or AZURE_API_VERSION
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not self.api_key:
            raise ValueError("Azure OpenAI API key not provided")
        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint not provided")
        
        # Initialize sync client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
        )
        
        # Initialize async client
        self.async_client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
        )
    
    async def extract(
        self,
        image_base64: str,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> ExtractionResult:
        """
        Extract text from an image using GPT-4o vision.
        
        Args:
            image_base64: Base64 encoded image
            prompt: Extraction prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            ExtractionResult with extracted text and metadata
        """
        for attempt in range(self.max_retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_base64}",
                                        "detail": "high",
                                    }
                                }
                            ]
                        }
                    ],
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                )
                
                text = response.choices[0].message.content or ""
                tokens = response.usage.total_tokens if response.usage else 0
                
                return ExtractionResult(
                    text=text,
                    success=True,
                    tokens_used=tokens,
                    raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
                )
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    print(f"[Azure] Retry {attempt + 1}/{self.max_retries} after {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    return ExtractionResult(
                        text="",
                        success=False,
                        error=str(e),
                    )
        
        return ExtractionResult(text="", success=False, error="Max retries exceeded")
    
    async def extract_section(
        self,
        image_base64: str,
        section_prompt: str,
        learning_context: str = "",
    ) -> ExtractionResult:
        """
        Extract fields from a specific document section.
        
        Args:
            image_base64: Base64 encoded section image
            section_prompt: Section-specific extraction prompt
            learning_context: Few-shot examples from past corrections
            
        Returns:
            ExtractionResult with extracted fields
        """
        # Combine prompt with learning context
        full_prompt = section_prompt
        if learning_context:
            full_prompt = f"{section_prompt}\n\n## Learning from past corrections:\n{learning_context}"
        
        return await self.extract(
            image_base64=image_base64,
            prompt=full_prompt,
            max_tokens=2048,
            temperature=0.1,
        )
    
    async def extract_field(
        self,
        image_base64: str,
        field_name: str,
        content_hint: str = "",
    ) -> FieldExtraction:
        """
        Extract a single field from a zoomed-in image.
        
        Uses a very simple prompt to minimize hallucination.
        
        Args:
            image_base64: Base64 encoded field image (zoomed)
            field_name: Name of the field being extracted
            content_hint: Hint about expected content format
            
        Returns:
            FieldExtraction with value and confidence
        """
        prompt = f"""هذه صورة مقصوصة لحقل "{field_name}".

اقرأ القيمة المكتوبة بخط اليد فقط.

قواعد:
- إذا فارغ (نقاط/خطوط فقط): اكتب "---"
- إذا غير واضح: اكتب "؟ [أفضل تخمين]"
- خلاف ذلك: اكتب القيمة فقط

{f'التنسيق المتوقع: {content_hint}' if content_hint else ''}

القيمة:"""

        result = await self.extract(
            image_base64=image_base64,
            prompt=prompt,
            max_tokens=256,
            temperature=0.05,
        )
        
        if not result.success:
            return FieldExtraction(
                field_name=field_name,
                value="",
                confidence="LOW",
                notes=f"Extraction failed: {result.error}",
            )
        
        value = result.text.strip()
        
        # Parse confidence from markers
        is_empty = value == "---" or "[فارغ]" in value
        is_unclear = value.startswith("؟") or "[غير واضح" in value
        
        if is_empty:
            confidence = "HIGH"
        elif is_unclear:
            confidence = "LOW"
            # Extract the guess from "؟ [guess]"
            if value.startswith("؟"):
                value = value[1:].strip().strip("[]")
        else:
            confidence = "HIGH"
        
        return FieldExtraction(
            field_name=field_name,
            value=value,
            confidence=confidence,
            is_empty=is_empty,
            is_unclear=is_unclear,
        )
    
    async def self_critique(
        self,
        extraction: Dict[str, str],
    ) -> CritiqueResult:
        """
        Have the model critique its own extraction for errors.
        
        Checks for:
        - Duplicate values across fields
        - Format mismatches
        - Suspicious patterns
        
        Args:
            extraction: Dictionary of field_name -> value
            
        Returns:
            CritiqueResult with issues and fields to recheck
        """
        extraction_text = "\n".join(
            f"{name}: {value}" 
            for name, value in extraction.items()
        )
        
        prompt = f"""Review this OCR extraction for errors:

## EXTRACTED DATA:
{extraction_text}

## CHECK FOR:
1. Same value appearing in multiple unrelated fields (hallucination)
2. Values that don't match field format:
   - رقم الهوية should be 10 digits starting with 1 or 2
   - رقم الجوال should be 10 digits starting with 05
   - Dates should be day/month/year format
3. Arabic name fields containing only numbers
4. Suspiciously complete forms (real forms have empty fields)

## OUTPUT (JSON only):
{{
  "has_issues": true/false,
  "issues": [
    {{"field": "field_name", "issue": "description", "severity": "high/medium/low"}}
  ],
  "fields_to_recheck": ["field1", "field2"],
  "overall_confidence": "high/medium/low"
}}"""

        result = await self.extract(
            image_base64="",  # No image for critique
            prompt=prompt,
            max_tokens=1024,
            temperature=0.1,
        )
        
        # Since we can't do text-only with vision API, use a workaround
        # Call the text completion endpoint instead
        try:
            response = await self.async_client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an OCR quality analyst. Output valid JSON only."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_completion_tokens=1024,
                temperature=0.1,
            )
            
            text = response.choices[0].message.content or "{}"
            parsed = self._parse_json(text)
            
            return CritiqueResult(
                has_issues=parsed.get("has_issues", False),
                issues=parsed.get("issues", []),
                fields_to_recheck=parsed.get("fields_to_recheck", []),
                overall_confidence=parsed.get("overall_confidence", "medium"),
                raw_response=text,
            )
            
        except Exception as e:
            print(f"[Azure] Critique failed: {e}")
            return CritiqueResult(
                has_issues=False,
                issues=[],
                fields_to_recheck=[],
                overall_confidence="low",
            )
    
    async def analyze_document_structure(
        self,
        image_base64: str,
    ) -> Dict[str, Any]:
        """
        Analyze document structure before detailed extraction.
        
        Returns overview of:
        - Document type
        - Visible sections
        - Fields with content vs empty
        - Overall handwriting quality
        """
        prompt = """Analyze this Arabic government form image:

1. What type of document is this?
2. List all visible sections (header, body sections, footer)
3. For each section, describe what you see:
   - Field labels visible
   - Fields with handwritten content
   - Fields that appear empty
4. Rate overall handwriting clarity: clear / moderate / poor
5. Any notable features or issues?

Output as JSON:
{
  "document_type": "string",
  "sections": [
    {
      "name": "string",
      "fields_with_content": ["field1", "field2"],
      "fields_empty": ["field3", "field4"]
    }
  ],
  "handwriting_quality": "clear/moderate/poor",
  "notes": "string"
}"""

        result = await self.extract(
            image_base64=image_base64,
            prompt=prompt,
            max_tokens=2048,
            temperature=0.1,
        )
        
        if not result.success:
            return {"error": result.error}
        
        return self._parse_json(result.text)
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from model response, handling common issues."""
        if not text:
            return {}
        
        # Remove markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end]
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end]
        
        text = text.strip()
        
        # Find JSON object
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            text = text[first_brace:last_brace + 1]
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
    
    async def close(self):
        """Close the client connections."""
        # AsyncAzureOpenAI doesn't require explicit closing
        pass


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_azure_client(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    deployment: Optional[str] = None,
) -> AzureVisionOCR:
    """
    Create an Azure Vision OCR client with configuration from environment.
    
    Args:
        api_key: Override API key
        endpoint: Override endpoint
        deployment: Override deployment name
        
    Returns:
        Configured AzureVisionOCR instance
    """
    return AzureVisionOCR(
        api_key=api_key,
        endpoint=endpoint,
        deployment=deployment,
    )


async def test_connection() -> bool:
    """Test Azure OpenAI connection."""
    try:
        client = create_azure_client()
        # Simple test without image
        response = await client.async_client.chat.completions.create(
            model=client.deployment,
            messages=[{"role": "user", "content": "Say 'connected'"}],
            max_completion_tokens=10,
        )
        return bool(response.choices)
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False
