"""
Production-Grade OCR Pipeline for Arabic Documents

Multi-stage pipeline architecture:
1. Image quality assessment
2. Layout analysis (detect document structure)
3. Field detection and region extraction
4. Per-region OCR with specialized prompts
5. Validation and confidence scoring
6. Routing to auto-accept or human review

Key Principles:
- NEVER hallucinate - only transcribe what's visually present
- Express uncertainty explicitly
- Route low-confidence extractions to human review
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal, Tuple
from enum import Enum
import re
from PIL import Image
import io
import base64


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ConfidenceLevel(Enum):
    HIGH = "high"           # > 0.85 - Auto-accept
    MEDIUM = "medium"       # 0.60-0.85 - Flag for optional review
    LOW = "low"             # 0.40-0.60 - Require human verification
    UNREADABLE = "unreadable"  # < 0.40 - Mark as unreadable
    EMPTY = "empty"         # Field is blank


class ValidationStatus(Enum):
    VALID = "valid"         # Matches expected pattern
    INVALID = "invalid"     # Doesn't match (possible hallucination)
    UNCHECKED = "unchecked" # No validator for this field type
    SUSPICIOUS = "suspicious"  # Matches hallucination patterns


@dataclass
class FieldResult:
    """Result for a single extracted field."""
    field_name: str
    raw_value: str
    cleaned_value: str
    confidence_score: float
    confidence_level: ConfidenceLevel
    validation_status: ValidationStatus
    validation_message: Optional[str] = None
    needs_review: bool = False
    is_empty: bool = False
    is_unclear: bool = False
    unclear_guess: Optional[str] = None
    region_bounds: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h


@dataclass
class QualityAssessment:
    """Image quality assessment result."""
    overall_score: float  # 0.0 - 1.0
    is_acceptable: bool
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Specific metrics
    brightness_score: float = 0.0
    contrast_score: float = 0.0
    sharpness_score: float = 0.0
    skew_angle: float = 0.0


@dataclass
class PipelineResult:
    """Complete result from the OCR pipeline."""
    status: Literal["success", "partial", "failed", "poor_quality"]
    fields: List[FieldResult] = field(default_factory=list)
    raw_text: str = ""
    
    # Quality info
    quality_assessment: Optional[QualityAssessment] = None
    
    # Review routing
    needs_review: bool = False
    review_reason: Optional[str] = None
    fields_needing_review: List[str] = field(default_factory=list)
    
    # Metadata
    processing_time_ms: float = 0.0
    confidence_summary: Dict[str, int] = field(default_factory=dict)
    
    # Warnings
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# CONFIDENCE THRESHOLDS
# =============================================================================

CONFIDENCE_THRESHOLDS = {
    "auto_accept": 0.85,     # High confidence - accept automatically
    "flag_review": 0.60,     # Medium - flag for optional review  
    "require_review": 0.40,  # Low - require human verification
    # Below 0.40 - mark as unreadable
}


# =============================================================================
# MARKER PATTERNS (for parsing model output)
# =============================================================================

EMPTY_MARKERS = ["[فارغ]", "[EMPTY]", "[empty]", "فارغ"]
UNCLEAR_PATTERN = r"\[غير واضح:\s*(.+?)\]|\[UNCLEAR:\s*(.+?)\]"
UNREADABLE_MARKERS = ["[غير مقروء]", "[UNREADABLE]", "[unreadable]"]
SIGNATURE_MARKERS = ["[توقيع]", "[SIGNATURE]", "[signature]"]
STAMP_MARKERS = ["[ختم]", "[STAMP]", "[stamp]"]


# =============================================================================
# OCR PIPELINE CLASS
# =============================================================================

class OCRPipeline:
    """
    Multi-stage OCR pipeline for production-grade Arabic document processing.
    
    Usage:
        pipeline = OCRPipeline(runpod_client)
        result = await pipeline.process(image_bytes, options)
    """
    
    def __init__(self, runpod_client=None, validators=None):
        """
        Initialize the OCR pipeline.
        
        Args:
            runpod_client: Client for calling the RunPod model service
            validators: Optional validators instance for field validation
        """
        self.runpod_client = runpod_client
        self.validators = validators
    
    async def process(
        self,
        image: Image.Image,
        prompt: str,
        options: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Process an image through the full OCR pipeline.
        
        Args:
            image: PIL Image to process
            prompt: OCR prompt to use
            options: Optional processing options
            
        Returns:
            PipelineResult with extracted fields and confidence scores
        """
        import time
        start_time = time.time()
        
        options = options or {}
        warnings = []
        
        # Stage 1: Quality Assessment
        quality = self.assess_quality(image)
        if not quality.is_acceptable:
            return PipelineResult(
                status="poor_quality",
                quality_assessment=quality,
                needs_review=True,
                review_reason="Image quality too low for reliable OCR",
                warnings=quality.issues
            )
        
        if quality.issues:
            warnings.extend(quality.issues)
        
        # Stage 2-4: OCR Processing (currently single-shot, can be extended to region-based)
        try:
            raw_text = await self._run_ocr(image, prompt, options)
        except Exception as e:
            return PipelineResult(
                status="failed",
                quality_assessment=quality,
                warnings=[f"OCR processing failed: {str(e)}"]
            )
        
        # Stage 5: Parse and Validate Results
        fields = self.parse_and_validate(raw_text)
        
        # Stage 6: Determine Review Routing
        fields_needing_review = []
        for f in fields:
            if f.needs_review:
                fields_needing_review.append(f.field_name)
        
        needs_review = len(fields_needing_review) > 0
        review_reason = None
        if needs_review:
            review_reason = f"{len(fields_needing_review)} field(s) need human review"
        
        # Calculate confidence summary
        confidence_summary = {
            "high": sum(1 for f in fields if f.confidence_level == ConfidenceLevel.HIGH),
            "medium": sum(1 for f in fields if f.confidence_level == ConfidenceLevel.MEDIUM),
            "low": sum(1 for f in fields if f.confidence_level == ConfidenceLevel.LOW),
            "empty": sum(1 for f in fields if f.confidence_level == ConfidenceLevel.EMPTY),
            "unreadable": sum(1 for f in fields if f.confidence_level == ConfidenceLevel.UNREADABLE),
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        return PipelineResult(
            status="success" if not needs_review else "partial",
            fields=fields,
            raw_text=raw_text,
            quality_assessment=quality,
            needs_review=needs_review,
            review_reason=review_reason,
            fields_needing_review=fields_needing_review,
            processing_time_ms=processing_time,
            confidence_summary=confidence_summary,
            warnings=warnings
        )
    
    def assess_quality(self, image: Image.Image) -> QualityAssessment:
        """
        Assess image quality for OCR processing.
        
        Returns quality metrics and recommendations.
        """
        import numpy as np
        
        issues = []
        recommendations = []
        
        # Convert to numpy for analysis
        img_array = np.array(image.convert('L'))  # Grayscale
        
        # Brightness check
        brightness = img_array.mean() / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Best around 0.5
        
        if brightness < 0.2:
            issues.append("Image is too dark")
            recommendations.append("Increase brightness or use better lighting")
        elif brightness > 0.85:
            issues.append("Image is too bright/washed out")
            recommendations.append("Reduce exposure or brightness")
        
        # Contrast check
        contrast = img_array.std() / 128.0  # Normalize to 0-1 range
        contrast_score = min(contrast, 1.0)
        
        if contrast < 0.15:
            issues.append("Low contrast - text may be hard to read")
            recommendations.append("Improve lighting or scan quality")
        
        # Sharpness check (using Laplacian variance)
        try:
            import cv2
            laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
            sharpness = laplacian.var() / 1000.0  # Normalize
            sharpness_score = min(sharpness, 1.0)
            
            if sharpness < 0.05:
                issues.append("Image appears blurry")
                recommendations.append("Use a higher resolution or steadier camera")
        except ImportError:
            sharpness_score = 0.5  # Default if cv2 not available
        
        # Overall score
        overall_score = (brightness_score * 0.3 + contrast_score * 0.4 + sharpness_score * 0.3)
        is_acceptable = overall_score >= 0.3 and len(issues) < 3
        
        return QualityAssessment(
            overall_score=overall_score,
            is_acceptable=is_acceptable,
            issues=issues,
            recommendations=recommendations,
            brightness_score=brightness_score,
            contrast_score=contrast_score,
            sharpness_score=sharpness_score,
            skew_angle=0.0  # TODO: Add skew detection
        )
    
    async def _run_ocr(
        self,
        image: Image.Image,
        prompt: str,
        options: Dict[str, Any]
    ) -> str:
        """
        Run OCR on the image using the model service.
        
        This is a placeholder - actual implementation depends on RunPod client.
        """
        if self.runpod_client is None:
            raise ValueError("RunPod client not configured")
        
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Build payload
        payload = {
            "input": {
                "image": img_base64,
                "prompt": prompt,
                "max_new_tokens": options.get("max_new_tokens", 8192),
                "min_pixels": options.get("min_pixels", 200704),
                "max_pixels": options.get("max_pixels", 1317120),  # Higher for handwriting
            }
        }
        
        # Call RunPod
        result, success = await self.runpod_client.call_runpod(payload)
        
        if not success:
            raise Exception(f"RunPod call failed: {result.get('error', 'Unknown error')}")
        
        if result.get("status") != "COMPLETED":
            raise Exception(f"RunPod job not completed: {result.get('status')}")
        
        output = result.get("output", {})
        return output.get("text", "")
    
    def parse_and_validate(self, raw_text: str) -> List[FieldResult]:
        """
        Parse the raw OCR output and validate each field.
        
        Handles special markers:
        - [فارغ] / [EMPTY] - Empty field
        - [غير واضح: X] / [UNCLEAR: X] - Unclear, with best guess
        - [غير مقروء] / [UNREADABLE] - Completely unreadable
        """
        fields = []
        
        if not raw_text:
            return fields
        
        lines = raw_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip section headers
            if line.startswith('[قسم]') or line.startswith('[SECTION]'):
                continue
            
            # Parse field:value format
            if ':' in line:
                parts = line.split(':', 1)
                field_name = parts[0].strip()
                raw_value = parts[1].strip() if len(parts) > 1 else ""
            else:
                # Not a field:value line, skip
                continue
            
            # Analyze the value
            field_result = self._analyze_field_value(field_name, raw_value)
            fields.append(field_result)
        
        return fields
    
    def _analyze_field_value(self, field_name: str, raw_value: str) -> FieldResult:
        """
        Analyze a field value and determine confidence, validation status, etc.
        """
        # Check for empty markers
        is_empty = any(marker in raw_value for marker in EMPTY_MARKERS)
        if is_empty:
            return FieldResult(
                field_name=field_name,
                raw_value=raw_value,
                cleaned_value="",
                confidence_score=1.0,  # High confidence that it's empty
                confidence_level=ConfidenceLevel.EMPTY,
                validation_status=ValidationStatus.VALID,
                is_empty=True,
                needs_review=False
            )
        
        # Check for unclear markers
        unclear_match = re.search(UNCLEAR_PATTERN, raw_value)
        if unclear_match:
            guess = unclear_match.group(1) or unclear_match.group(2) or ""
            return FieldResult(
                field_name=field_name,
                raw_value=raw_value,
                cleaned_value=guess,
                confidence_score=0.5,  # Medium confidence
                confidence_level=ConfidenceLevel.MEDIUM,
                validation_status=ValidationStatus.UNCHECKED,
                is_unclear=True,
                unclear_guess=guess,
                needs_review=True  # Unclear fields should be reviewed
            )
        
        # Check for unreadable markers
        is_unreadable = any(marker in raw_value for marker in UNREADABLE_MARKERS)
        if is_unreadable:
            return FieldResult(
                field_name=field_name,
                raw_value=raw_value,
                cleaned_value="",
                confidence_score=0.0,
                confidence_level=ConfidenceLevel.UNREADABLE,
                validation_status=ValidationStatus.UNCHECKED,
                needs_review=True  # Unreadable fields need review
            )
        
        # Check for signature/stamp markers
        is_signature = any(marker in raw_value for marker in SIGNATURE_MARKERS)
        is_stamp = any(marker in raw_value for marker in STAMP_MARKERS)
        if is_signature or is_stamp:
            return FieldResult(
                field_name=field_name,
                raw_value=raw_value,
                cleaned_value=raw_value,
                confidence_score=0.9,
                confidence_level=ConfidenceLevel.HIGH,
                validation_status=ValidationStatus.VALID,
                needs_review=False
            )
        
        # Regular value - validate and score
        cleaned_value = raw_value
        validation_status = ValidationStatus.UNCHECKED
        validation_message = None
        confidence_score = 0.75  # Default medium-high confidence
        
        # Run validators if available
        if self.validators:
            validation_result = self.validators.validate_field(field_name, cleaned_value)
            validation_status = validation_result.status
            validation_message = validation_result.message
            
            # Adjust confidence based on validation
            if validation_status == ValidationStatus.VALID:
                confidence_score = min(confidence_score + 0.15, 1.0)
            elif validation_status == ValidationStatus.INVALID:
                confidence_score = max(confidence_score - 0.25, 0.3)
            elif validation_status == ValidationStatus.SUSPICIOUS:
                confidence_score = 0.3  # Likely hallucination
        
        # Determine confidence level
        if confidence_score >= CONFIDENCE_THRESHOLDS["auto_accept"]:
            confidence_level = ConfidenceLevel.HIGH
            needs_review = False
        elif confidence_score >= CONFIDENCE_THRESHOLDS["flag_review"]:
            confidence_level = ConfidenceLevel.MEDIUM
            needs_review = False  # Optional review
        elif confidence_score >= CONFIDENCE_THRESHOLDS["require_review"]:
            confidence_level = ConfidenceLevel.LOW
            needs_review = True
        else:
            confidence_level = ConfidenceLevel.UNREADABLE
            needs_review = True
        
        return FieldResult(
            field_name=field_name,
            raw_value=raw_value,
            cleaned_value=cleaned_value,
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            validation_status=validation_status,
            validation_message=validation_message,
            needs_review=needs_review
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_pipeline_result_dict(result: PipelineResult) -> Dict[str, Any]:
    """Convert PipelineResult to a JSON-serializable dictionary."""
    return {
        "status": result.status,
        "fields": [
            {
                "field_name": f.field_name,
                "value": f.cleaned_value,
                "raw_value": f.raw_value,
                "confidence": f.confidence_score,
                "confidence_level": f.confidence_level.value,
                "validation_status": f.validation_status.value,
                "validation_message": f.validation_message,
                "needs_review": f.needs_review,
                "is_empty": f.is_empty,
                "is_unclear": f.is_unclear,
            }
            for f in result.fields
        ],
        "raw_text": result.raw_text,
        "quality": {
            "score": result.quality_assessment.overall_score if result.quality_assessment else None,
            "issues": result.quality_assessment.issues if result.quality_assessment else [],
        } if result.quality_assessment else None,
        "needs_review": result.needs_review,
        "review_reason": result.review_reason,
        "fields_needing_review": result.fields_needing_review,
        "confidence_summary": result.confidence_summary,
        "warnings": result.warnings,
        "processing_time_ms": result.processing_time_ms,
    }
