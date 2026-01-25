"""
Surgical Agentic OCR Controller

Complete OCR pipeline with:
1. Image preprocessing
2. Document structure analysis
3. Section-by-section extraction
4. Iterative zoom-in refinement for unclear fields
5. Format validation
6. Self-critique and merge
7. Learning integration

Inspired by how humans read forms - surgical precision approach.
"""

import asyncio
import time
from typing import Optional, Dict, List, Any, Tuple
from PIL import Image
from dataclasses import dataclass, field

from .image_processor import ImageProcessor, Section, SectionType
from .azure_client import AzureVisionOCR, FieldExtraction
from .format_validator import FormatValidator, DocumentValidation
from .learning import LearningModule, LearningContext
from .prompts import (
    get_section_prompt,
    get_field_prompt,
    get_critique_prompt,
    DOCUMENT_ANALYSIS_PROMPT,
)
from .models import (
    AgenticResult,
    FieldResult,
    QualityReport,
    ValidationIssue,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SectionResult:
    """Result from processing a single section."""
    section_type: str
    fields: Dict[str, str]
    confidence: Dict[str, str]  # field -> confidence level
    unclear_fields: List[str]
    processing_time: float


@dataclass
class ProcessingTrace:
    """Trace of processing steps for debugging."""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    
    def add(self, step: str, data: Dict[str, Any] = None):
        self.steps.append({
            "step": step,
            "timestamp": time.time(),
            "data": data or {},
        })


# =============================================================================
# QUALITY THRESHOLDS
# =============================================================================

MIN_CONFIDENCE_FOR_ZOOM = "MEDIUM"  # Zoom in on MEDIUM and LOW confidence
MAX_ZOOM_ATTEMPTS = 3  # Maximum zoom refinement attempts per field
MIN_QUALITY_SCORE = 40  # Minimum score to accept result
HALLUCINATION_THRESHOLD = 20  # Below this, reject as hallucination


# =============================================================================
# SURGICAL OCR CONTROLLER
# =============================================================================

class AgenticOCRController:
    """
    Surgical precision OCR controller.
    
    Orchestrates the complete pipeline:
    1. Preprocess image (grayscale, contrast enhancement)
    2. Analyze document structure
    3. Detect and process each section
    4. Zoom-in refinement for unclear fields
    5. Format validation
    6. Self-critique
    7. Final merge with confidence scoring
    
    Uses Azure OpenAI GPT-4o-mini for vision tasks.
    Integrates learning from user corrections.
    """
    
    def __init__(
        self,
        azure_client: Optional[AzureVisionOCR] = None,
        learning_module: Optional[LearningModule] = None,
        enable_zoom_refinement: bool = True,
        enable_self_critique: bool = True,
        strict_validation: bool = True,
    ):
        """
        Initialize the surgical OCR controller.
        
        Args:
            azure_client: Azure OpenAI vision client (created if not provided)
            learning_module: Learning module for corrections (created if not provided)
            enable_zoom_refinement: Whether to zoom in on unclear fields
            enable_self_critique: Whether to run self-critique pass
            strict_validation: Use strict format validation
        """
        self.azure = azure_client or AzureVisionOCR()
        self.learning = learning_module or LearningModule()
        self.processor = ImageProcessor()
        self.validator = FormatValidator()
        
        self.enable_zoom_refinement = enable_zoom_refinement
        self.enable_self_critique = enable_self_critique
        self.strict_validation = strict_validation
    
    async def process(
        self,
        image: Image.Image,
        user_id: Optional[str] = None,
    ) -> AgenticResult:
        """
        Process an image through the full surgical OCR pipeline.
        
        Args:
            image: PIL Image of the document
            user_id: Optional user ID for learning context
            
        Returns:
            AgenticResult with all extracted fields, quality scores, and metadata
        """
        start_time = time.time()
        trace = ProcessingTrace()
        
        print(f"[Surgical OCR] Starting pipeline...")
        trace.add("start", {"image_size": image.size})
        
        # =================================================================
        # STAGE 1: Image Preprocessing
        # =================================================================
        trace.add("preprocessing")
        print(f"[Surgical OCR] Stage 1: Preprocessing image...")
        
        enhanced = self.processor.preprocess(image, contrast=2.0)
        
        trace.add("preprocessing_complete", {
            "original_size": image.size,
            "enhanced": True,
        })
        
        # =================================================================
        # STAGE 2: Get Learning Context
        # =================================================================
        trace.add("learning_context")
        print(f"[Surgical OCR] Stage 2: Loading learning context...")
        
        learning_context = await self.learning.get_context()
        learning_prompt = learning_context.to_prompt() if learning_context else ""
        
        trace.add("learning_context_complete", {
            "has_examples": bool(learning_prompt),
        })
        
        # =================================================================
        # STAGE 3: Section Detection
        # =================================================================
        trace.add("section_detection")
        print(f"[Surgical OCR] Stage 3: Detecting sections...")
        
        sections = self.processor.detect_sections(enhanced)
        
        trace.add("section_detection_complete", {
            "sections_found": len(sections),
        })
        print(f"[Surgical OCR] Found {len(sections)} sections")
        
        # =================================================================
        # STAGE 4: Section-by-Section Extraction
        # =================================================================
        trace.add("section_extraction")
        print(f"[Surgical OCR] Stage 4: Extracting sections...")
        
        all_fields: Dict[str, str] = {}
        all_confidence: Dict[str, str] = {}
        unclear_fields: List[str] = []
        
        for section in sections:
            print(f"[Surgical OCR]   Processing: {section.name}")
            
            # Crop and upscale section
            section_image = self.processor.crop_section(
                enhanced, section, upscale=2
            )
            
            # Get section-specific prompt with learning context
            section_prompt = get_section_prompt(
                section.section_type.value,
                learning_context=learning_prompt,
            )
            
            # Extract using Azure Vision
            image_b64 = self.processor.image_to_base64(section_image)
            result = await self.azure.extract_section(
                image_b64,
                section_prompt,
            )
            
            if result.success:
                # Parse extraction results
                fields, confidence = self._parse_extraction(result.text)
                
                all_fields.update(fields)
                all_confidence.update(confidence)
                
                # Track unclear fields for zoom refinement
                for field_name, conf in confidence.items():
                    if conf in ["LOW", "MEDIUM"]:
                        unclear_fields.append(field_name)
                
                print(f"[Surgical OCR]   Extracted {len(fields)} fields")
            else:
                print(f"[Surgical OCR]   Extraction failed: {result.error}")
        
        trace.add("section_extraction_complete", {
            "total_fields": len(all_fields),
            "unclear_fields": len(unclear_fields),
        })
        
        # =================================================================
        # STAGE 5: Zoom-In Refinement for Unclear Fields
        # =================================================================
        if self.enable_zoom_refinement and unclear_fields:
            trace.add("zoom_refinement")
            print(f"[Surgical OCR] Stage 5: Refining {len(unclear_fields)} unclear fields...")
            
            refined_count = 0
            for field_name in unclear_fields[:10]:  # Limit to top 10
                print(f"[Surgical OCR]   Zooming: {field_name}")
                
                # Get zoom levels for field
                zoom_levels = self.processor.iterative_zoom(enhanced, field_name)
                
                if not zoom_levels:
                    continue
                
                # Try each zoom level until confident
                best_result: Optional[FieldExtraction] = None
                
                for zoomed_image, scale in zoom_levels:
                    # Check if region is blank
                    if self.processor.is_blank_region(zoomed_image):
                        best_result = FieldExtraction(
                            field_name=field_name,
                            value="---",
                            confidence="HIGH",
                            is_empty=True,
                        )
                        break
                    
                    # Extract at this zoom level
                    image_b64 = self.processor.image_to_base64(zoomed_image)
                    result = await self.azure.extract_field(
                        image_b64,
                        field_name,
                    )
                    
                    if result.confidence == "HIGH":
                        best_result = result
                        break
                    elif not best_result or result.confidence == "MEDIUM":
                        best_result = result
                
                # Update with refined result
                if best_result:
                    if best_result.value and best_result.value != "---":
                        all_fields[field_name] = best_result.value
                    all_confidence[field_name] = best_result.confidence
                    refined_count += 1
            
            trace.add("zoom_refinement_complete", {
                "refined_count": refined_count,
            })
            print(f"[Surgical OCR] Refined {refined_count} fields")
        
        # =================================================================
        # STAGE 6: Format Validation
        # =================================================================
        trace.add("validation")
        print(f"[Surgical OCR] Stage 6: Validating formats...")
        
        validation = self.validator.validate_document(all_fields)
        
        # Update confidence based on validation
        for field_name, result in validation.field_results.items():
            if not result.is_valid:
                all_confidence[field_name] = "LOW"
        
        trace.add("validation_complete", {
            "score": validation.overall_score,
            "issues": len(validation.critical_issues),
        })
        print(f"[Surgical OCR] Validation score: {validation.overall_score}/100")
        
        # =================================================================
        # STAGE 7: Self-Critique
        # =================================================================
        if self.enable_self_critique:
            trace.add("self_critique")
            print(f"[Surgical OCR] Stage 7: Self-critique...")
            
            critique = await self.azure.self_critique(all_fields)
            
            if critique.has_issues:
                print(f"[Surgical OCR] Found {len(critique.issues)} issues")
                
                # Mark problematic fields for review
                for issue in critique.issues:
                    field_name = issue.get("field", "")
                    if field_name in all_confidence:
                        all_confidence[field_name] = "LOW"
                
                # Try to re-extract flagged fields
                for field_name in critique.fields_to_recheck[:5]:
                    if field_name in all_fields:
                        zoom_levels = self.processor.iterative_zoom(enhanced, field_name)
                        if zoom_levels:
                            zoomed_image, _ = zoom_levels[-1]  # Use max zoom
                            image_b64 = self.processor.image_to_base64(zoomed_image)
                            result = await self.azure.extract_field(image_b64, field_name)
                            if result.value:
                                all_fields[field_name] = result.value
                                all_confidence[field_name] = result.confidence
            
            trace.add("self_critique_complete", {
                "issues_found": len(critique.issues) if critique.has_issues else 0,
            })
        
        # =================================================================
        # STAGE 8: Build Final Result
        # =================================================================
        trace.add("build_result")
        print(f"[Surgical OCR] Stage 8: Building final result...")
        
        processing_time = time.time() - start_time
        
        result = self._build_result(
            fields=all_fields,
            confidence=all_confidence,
            validation=validation,
            processing_time=processing_time,
            trace=trace,
        )
        
        print(f"[Surgical OCR] Complete! {len(result.fields)} fields in {processing_time:.1f}s")
        print(f"[Surgical OCR] Quality: {result.quality_score}/100 ({result.quality_status})")
        
        return result
    
    def _parse_extraction(
        self,
        text: str,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Parse extraction text into fields and confidence levels.
        
        Expected format:
        field_name: value [CONFIDENCE]
        
        CRITICAL RULE: Empty values (---) should NEVER have high confidence.
        An empty field means we couldn't read it, which is inherently uncertain.
        
        Returns:
            Tuple of (fields_dict, confidence_dict)
        """
        fields = {}
        confidence = {}
        
        # Empty value markers
        EMPTY_MARKERS = ["---", "[فارغ]", "[EMPTY]", "غير موجود", "[غير موجود]", "فارغ", ""]
        
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            
            # Parse field_name: value [CONFIDENCE]
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            
            field_name = parts[0].strip()
            rest = parts[1].strip()
            
            # Extract confidence level from markers
            conf = "MEDIUM"  # default
            if "[HIGH]" in rest.upper():
                conf = "HIGH"
                rest = rest.replace("[HIGH]", "").replace("[high]", "").strip()
            elif "[MEDIUM]" in rest.upper():
                conf = "MEDIUM"
                rest = rest.replace("[MEDIUM]", "").replace("[medium]", "").strip()
            elif "[LOW]" in rest.upper():
                conf = "LOW"
                rest = rest.replace("[LOW]", "").replace("[low]", "").strip()
            elif "[EMPTY]" in rest.upper():
                conf = "LOW"  # Empty fields should be LOW confidence
                rest = "---"
            
            value = rest.strip()
            
            # CRITICAL: Empty/missing values should NEVER have high confidence
            # If we couldn't find a value, we are NOT confident about it
            is_empty = any(marker in value or value == marker for marker in EMPTY_MARKERS)
            if is_empty:
                conf = "LOW"  # Force LOW confidence for empty values
                value = "---"  # Normalize empty marker
            
            # Also check for unclear markers
            if "؟" in value or "[غير واضح" in value:
                conf = "LOW"
            
            # Store
            fields[field_name] = value
            confidence[field_name] = conf
        
        return fields, confidence
    
    def _build_result(
        self,
        fields: Dict[str, str],
        confidence: Dict[str, str],
        validation: DocumentValidation,
        processing_time: float,
        trace: ProcessingTrace,
    ) -> AgenticResult:
        """Build the final AgenticResult from processed data."""
        
        # Empty value markers
        EMPTY_MARKERS = ["---", "[فارغ]", "[EMPTY]", "غير موجود", "[غير موجود]", "فارغ", ""]
        
        # Build field results
        field_results = []
        fields_needing_review = []
        confidence_summary = {"high": 0, "medium": 0, "low": 0, "empty": 0}
        
        for field_name, value in fields.items():
            conf = confidence.get(field_name, "MEDIUM")
            
            # Check for empty markers
            is_empty = any(marker in value or value == marker for marker in EMPTY_MARKERS)
            
            # CRITICAL: Empty values should ALWAYS have LOW confidence
            # and should ALWAYS need review
            if is_empty:
                conf = "LOW"
                value = "---"  # Normalize
            
            # Check for unclear markers
            has_uncertainty = "؟" in value or "[غير واضح" in value
            if has_uncertainty:
                conf = "LOW"
            
            # Determine if needs review
            needs_review = conf == "LOW" or is_empty or has_uncertainty
            
            # Get validation result for this field
            val_result = validation.field_results.get(field_name)
            
            # Calculate validation score
            # Empty fields get 20 (not 0, but not good either)
            # Invalid fields get 50
            # Valid fields get 100
            if is_empty:
                validation_score = 20
            elif val_result and not val_result.is_valid:
                validation_score = 50
            else:
                validation_score = 100
            
            # Count confidence levels
            if is_empty:
                confidence_summary["empty"] += 1
            else:
                confidence_summary[conf.lower()] += 1
            
            if needs_review:
                fields_needing_review.append(field_name)
            
            # Build review reason
            review_reason = None
            if needs_review:
                reasons = []
                if is_empty:
                    reasons.append("Empty/missing value")
                if has_uncertainty:
                    reasons.append("Uncertain reading")
                if conf == "LOW" and not is_empty and not has_uncertainty:
                    reasons.append("Low confidence")
                review_reason = "; ".join(reasons) if reasons else "Needs verification"
            
            field_results.append(FieldResult(
                field_name=field_name,
                value=value,
                confidence=conf.lower(),
                source="agentic",
                needs_review=needs_review,
                review_reason=review_reason,
                is_empty=is_empty,
                validation_score=validation_score,
            ))
        
        # Determine quality status
        quality_score = validation.overall_score
        
        if quality_score < HALLUCINATION_THRESHOLD:
            quality_status = "rejected"
        elif quality_score < MIN_QUALITY_SCORE:
            quality_status = "failed"
        elif len(validation.critical_issues) > 0:
            quality_status = "warning"
        elif len(fields_needing_review) > len(fields) * 0.3:
            quality_status = "warning"
        else:
            quality_status = "passed"
        
        # Build quality report
        quality_report = QualityReport(
            quality_score=quality_score,
            quality_status=quality_status,
            validation_issues=[
                ValidationIssue(
                    field_name=r.field_name,
                    issue_type="format",
                    severity="warning" if r.is_valid else "error",
                    message="; ".join(r.issues) if r.issues else "",
                    expected=r.expected_format,
                    actual=r.value,
                )
                for r in validation.field_results.values()
                if r.issues
            ],
            hallucination_indicators=validation.hallucination_indicators,
            field_scores={
                name: (100 if r.is_valid else 50)
                for name, r in validation.field_results.items()
            },
            duplicate_values=validation.duplicate_values,
            should_retry=quality_status in ["failed", "rejected"],
            needs_human_review=quality_status in ["warning", "failed"],
            fields_to_review=fields_needing_review,
            rejection_reasons=validation.critical_issues,
            warning_reasons=validation.warnings,
        )
        
        # Determine if hallucination detected
        hallucination_detected = (
            quality_status == "rejected" or
            len(validation.hallucination_indicators) > 0 or
            bool(validation.duplicate_values)
        )
        
        return AgenticResult(
            fields=field_results,
            raw_text="\n".join(f"{k}: {v}" for k, v in fields.items()),
            iterations_used=1,  # Surgical approach uses sections, not iterations
            processing_time_seconds=processing_time,
            confidence_summary=confidence_summary,
            fields_needing_review=fields_needing_review,
            status="success" if quality_status in ("passed", "warning") else quality_status,
            quality_score=quality_score,
            quality_status=quality_status,
            quality_report=quality_report,
            hallucination_detected=hallucination_detected,
            hallucination_indicators=validation.hallucination_indicators,
            trace=trace.steps,
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_controller(
    azure_client: Optional[AzureVisionOCR] = None,
    learning_module: Optional[LearningModule] = None,
    supabase_client=None,
) -> AgenticOCRController:
    """
    Create an agentic OCR controller with default configuration.
    
    Args:
        azure_client: Optional pre-configured Azure client
        learning_module: Optional pre-configured learning module
        supabase_client: Optional Supabase client for learning
        
    Returns:
        Configured AgenticOCRController
    """
    if not azure_client:
        from .azure_client import create_azure_client
        azure_client = create_azure_client()
    
    if not learning_module:
        from .learning import create_learning_module
        learning_module = create_learning_module(supabase_client)
    
    return AgenticOCRController(
        azure_client=azure_client,
        learning_module=learning_module,
    )


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

class SinglePassController:
    """
    Legacy single-pass controller for backward compatibility.
    Redirects to the surgical controller.
    """
    
    def __init__(self, *args, **kwargs):
        self.controller = AgenticOCRController()
    
    async def process(
        self,
        image: Image.Image,
        custom_prompt: Optional[str] = None,
    ) -> AgenticResult:
        return await self.controller.process(image)
