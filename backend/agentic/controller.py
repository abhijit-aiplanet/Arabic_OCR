"""
Agentic OCR Controller with Quality Gates

Main orchestration logic for the multi-pass, self-correcting OCR pipeline
with ANTI-HALLUCINATION quality gates at every step.

Flow:
1. Initial full-page OCR (VLM)
2. QUALITY GATE: Check for hallucination patterns
3. Analyze output for issues (LLM)
4. Estimate regions for uncertain fields (LLM)
5. Crop and re-OCR uncertain regions (VLM)
6. QUALITY GATE: Verify refinement improved quality
7. Merge original and refined results (LLM)
8. FINAL QUALITY GATE: Accept/reject final result
"""

import asyncio
import time
from typing import Optional, Dict, List, Any, Tuple
from PIL import Image

from .clients import VLMClient, LLMClient
from .cropper import RegionCropper, enhance_cropped_region
from .models import (
    AgenticResult,
    FieldResult,
    AnalysisResult,
    RegionEstimate,
    MergeResult,
    FinalFieldValue,
    QualityReport,
    ValidationIssue,
)
from .prompts import get_content_hint, build_field_reextract_prompt
from .validators import (
    validate_extraction,
    parse_extraction_to_fields,
    ValidationResult,
)
from .quality_gate import (
    evaluate_quality,
    QualityGateResult,
    QualityLevel,
    compare_extraction_quality,
    should_use_refined,
)


# =============================================================================
# QUALITY THRESHOLDS
# =============================================================================

# Minimum quality score to continue processing
MIN_QUALITY_TO_CONTINUE = 20

# Minimum quality score for final acceptance
MIN_QUALITY_TO_ACCEPT = 40

# Quality score below which we should definitely reject
REJECT_THRESHOLD = 20


# =============================================================================
# AGENTIC OCR CONTROLLER
# =============================================================================

class AgenticOCRController:
    """
    Multi-pass, self-correcting OCR controller with quality gates.
    
    Uses dual-model architecture:
    - VLM (AIN) for vision/OCR tasks
    - LLM (Qwen 2.5 or Mistral) for reasoning/orchestration
    
    Quality gates at each step ensure hallucinations are caught early.
    
    Example usage:
        controller = AgenticOCRController(vlm_client, llm_client)
        result = await controller.process(image)
        
        if result.quality_status == "rejected":
            print("Extraction failed quality checks")
        elif result.quality_status == "warning":
            print("Extraction needs human review")
        else:
            print("Extraction passed quality checks")
    """
    
    def __init__(
        self,
        vlm_client: VLMClient,
        llm_client: LLMClient,
        max_iterations: int = 3,
        min_fields_to_reexamine: int = 1,
        use_grid_fallback: bool = True,
        strict_quality: bool = True
    ):
        """
        Initialize the agentic controller.
        
        Args:
            vlm_client: Client for vision/OCR model
            llm_client: Client for reasoning model
            max_iterations: Maximum refinement iterations
            min_fields_to_reexamine: Minimum fields to warrant re-examination
            use_grid_fallback: Whether to use grid cropping if region estimation fails
            strict_quality: Use stricter quality thresholds if True
        """
        self.vlm = vlm_client
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.min_fields_to_reexamine = min_fields_to_reexamine
        self.use_grid_fallback = use_grid_fallback
        self.strict_quality = strict_quality
        self.cropper = RegionCropper()
    
    async def process(
        self,
        image: Image.Image,
        custom_initial_prompt: Optional[str] = None
    ) -> AgenticResult:
        """
        Process an image through the full agentic OCR pipeline.
        
        Args:
            image: PIL Image of the document
            custom_initial_prompt: Optional custom prompt for initial OCR
            
        Returns:
            AgenticResult with all extracted fields, quality scores, and metadata
        """
        start_time = time.time()
        trace = []  # Processing trace for debugging
        
        # =================================================================
        # STEP 1: Initial Full-Page OCR
        # =================================================================
        trace.append({"step": "initial_ocr", "timestamp": time.time()})
        
        print(f"[Agentic] Step 1: Initial full-page OCR...")
        initial_text, success = await self.vlm.extract_initial(
            image, 
            custom_prompt=custom_initial_prompt
        )
        
        if not success:
            return self._create_error_result(
                error=f"Initial OCR failed: {initial_text}",
                processing_time=time.time() - start_time,
                trace=trace
            )
        
        trace.append({
            "step": "initial_ocr_complete",
            "text_length": len(initial_text),
            "timestamp": time.time()
        })
        
        print(f"[Agentic] Initial OCR: {len(initial_text)} characters extracted")
        
        # =================================================================
        # QUALITY GATE 1: Check initial extraction quality
        # =================================================================
        trace.append({"step": "quality_gate_1", "timestamp": time.time()})
        
        print(f"[Agentic] Quality Gate 1: Checking initial extraction...")
        initial_fields = parse_extraction_to_fields(initial_text)
        initial_quality = evaluate_quality(
            initial_fields, 
            initial_text, 
            strict_mode=self.strict_quality
        )
        
        print(f"[Agentic] Initial quality score: {initial_quality.quality_score}/100")
        print(f"[Agentic] Quality level: {initial_quality.quality_level.value}")
        
        if initial_quality.rejection_reasons:
            print(f"[Agentic] Rejection reasons: {initial_quality.rejection_reasons}")
        
        trace.append({
            "step": "quality_gate_1_complete",
            "quality_score": initial_quality.quality_score,
            "quality_level": initial_quality.quality_level.value,
            "rejection_reasons": initial_quality.rejection_reasons,
            "timestamp": time.time()
        })
        
        # If quality is critically low, reject early
        if initial_quality.quality_score < REJECT_THRESHOLD:
            print(f"[Agentic] REJECTED: Quality score {initial_quality.quality_score} below threshold {REJECT_THRESHOLD}")
            return self._create_rejected_result(
                raw_text=initial_text,
                quality_result=initial_quality,
                processing_time=time.time() - start_time,
                trace=trace,
                reason="Initial extraction quality too low (likely hallucination)"
            )
        
        # Track current best extraction
        current_extraction = initial_text
        current_fields = initial_fields
        current_quality = initial_quality
        iteration = 0
        
        # =================================================================
        # ITERATION LOOP
        # =================================================================
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n[Agentic] === Iteration {iteration}/{self.max_iterations} ===")
            
            # If quality is already good, we can stop
            if current_quality.quality_score >= 70:
                print(f"[Agentic] Quality score {current_quality.quality_score} is good, skipping refinement")
                break
            
            # =============================================================
            # STEP 2: Analyze with LLM
            # =============================================================
            trace.append({"step": f"analyze_iter_{iteration}", "timestamp": time.time()})
            
            print(f"[Agentic] Step 2: Analyzing extraction for issues...")
            analysis, success = await self.llm.analyze(current_extraction)
            
            if not success or analysis is None:
                print(f"[Agentic] Analysis failed, using current extraction")
                break
            
            trace.append({
                "step": f"analysis_complete_iter_{iteration}",
                "fields_to_reexamine": len(analysis.fields_to_reexamine),
                "hallucination_detected": analysis.has_hallucination_warning(),
                "timestamp": time.time()
            })
            
            # Check for LLM-detected hallucination
            if analysis.has_hallucination_warning():
                print(f"[Agentic] LLM detected hallucination: {analysis.hallucination_warnings}")
                # Continue to try to fix, but note it
                current_quality.warning_reasons.extend(analysis.hallucination_warnings)
            
            # Check if we need to re-examine any fields
            if not analysis.has_fields_to_reexamine():
                print(f"[Agentic] All fields confident, stopping iteration")
                break
            
            if len(analysis.fields_to_reexamine) < self.min_fields_to_reexamine:
                print(f"[Agentic] Only {len(analysis.fields_to_reexamine)} uncertain fields, skipping re-examination")
                break
            
            print(f"[Agentic] Found {len(analysis.fields_to_reexamine)} fields to re-examine")
            for f in analysis.fields_to_reexamine[:5]:  # Show first 5
                print(f"  - {f.field_name}: {f.issue}")
            
            # =============================================================
            # STEP 3: Estimate Regions
            # =============================================================
            trace.append({"step": f"estimate_regions_iter_{iteration}", "timestamp": time.time()})
            
            print(f"[Agentic] Step 3: Estimating field regions...")
            fields_data = [
                {
                    "field_name": f.field_name,
                    "current_value": f.current_value,
                    "issue": f.issue,
                    "field_type": f.field_type,
                }
                for f in analysis.fields_to_reexamine
            ]
            
            regions, success = await self.llm.estimate_regions(
                fields_data,
                image.width,
                image.height
            )
            
            if not success or not regions:
                print(f"[Agentic] Region estimation failed")
                if self.use_grid_fallback:
                    print(f"[Agentic] Using grid fallback...")
                    regions = self._create_fallback_regions(
                        analysis.fields_to_reexamine,
                        image
                    )
                else:
                    break
            
            trace.append({
                "step": f"regions_estimated_iter_{iteration}",
                "num_regions": len(regions),
                "timestamp": time.time()
            })
            
            # =============================================================
            # STEP 4: Crop and Re-OCR Each Region
            # =============================================================
            trace.append({"step": f"reocr_iter_{iteration}", "timestamp": time.time()})
            
            print(f"[Agentic] Step 4: Re-OCRing {len(regions)} cropped regions...")
            refined_values = {}
            
            for region in regions:
                print(f"  - Processing region: {region.field_name}")
                
                # Crop the region
                try:
                    cropped = self.cropper.crop(
                        image,
                        region.bbox_normalized,
                        normalize=True
                    )
                    
                    # Check if crop is mostly empty (blank region detection)
                    if self._is_blank_region(cropped):
                        print(f"    Detected blank/empty region, marking as [فارغ]")
                        refined_values[region.field_name] = "[فارغ]"
                        continue
                    
                    # Enhance the crop
                    cropped = enhance_cropped_region(cropped)
                    
                    # Find previous value for context
                    previous_value = ""
                    for f in analysis.fields_to_reexamine:
                        if f.field_name == region.field_name:
                            previous_value = f.current_value
                            break
                    
                    # Re-OCR the cropped region using enhanced prompt
                    prompt = build_field_reextract_prompt(
                        field_name=region.field_name,
                        previous_value=previous_value,
                        content_hint=get_content_hint(region.field_name)
                    )
                    
                    value, success = await self.vlm.extract_field(
                        cropped,
                        region.field_name,
                        previous_value=previous_value,
                        content_hint=get_content_hint(region.field_name)
                    )
                    
                    if success:
                        refined_values[region.field_name] = value
                        display_value = value[:50] + "..." if len(value) > 50 else value
                        print(f"    Refined: {display_value}")
                    else:
                        print(f"    Failed to re-OCR")
                        
                except Exception as e:
                    print(f"    Error processing region: {e}")
            
            trace.append({
                "step": f"reocr_complete_iter_{iteration}",
                "refined_count": len(refined_values),
                "timestamp": time.time()
            })
            
            if not refined_values:
                print(f"[Agentic] No refined values obtained, stopping")
                break
            
            # =============================================================
            # QUALITY GATE 2: Check if refinement improved quality
            # =============================================================
            trace.append({"step": f"quality_gate_2_iter_{iteration}", "timestamp": time.time()})
            
            print(f"[Agentic] Quality Gate 2: Checking refinement quality...")
            
            # Build tentative refined fields
            refined_fields = current_fields.copy()
            refined_fields.update(refined_values)
            
            # Evaluate refined quality
            refined_quality = evaluate_quality(
                refined_fields,
                "",
                strict_mode=self.strict_quality
            )
            
            print(f"[Agentic] Refined quality score: {refined_quality.quality_score}/100")
            
            # Compare quality
            use_refined, explanation = should_use_refined(
                current_fields,
                refined_fields,
                min_improvement=5
            )
            
            print(f"[Agentic] Quality comparison: {explanation}")
            
            if not use_refined:
                print(f"[Agentic] Keeping original extraction (refinement didn't improve)")
                trace.append({
                    "step": f"quality_gate_2_rejected_iter_{iteration}",
                    "reason": explanation,
                    "timestamp": time.time()
                })
                # Don't update current, try another iteration or stop
                continue
            
            # =============================================================
            # STEP 5: Merge Results
            # =============================================================
            trace.append({"step": f"merge_iter_{iteration}", "timestamp": time.time()})
            
            print(f"[Agentic] Step 5: Merging results...")
            merge_result, success = await self.llm.merge(
                current_extraction,
                refined_values
            )
            
            if not success or merge_result is None:
                print(f"[Agentic] Merge failed, using previous extraction")
                break
            
            # Check merge quality
            if merge_result.has_quality_issues():
                print(f"[Agentic] Merge detected quality issues:")
                if merge_result.merge_quality_check.duplicate_values_found:
                    print(f"  - Duplicate values found")
                if merge_result.merge_quality_check.year_as_value_found:
                    print(f"  - Year values in non-date fields")
                if merge_result.merge_quality_check.suspicious_fills:
                    print(f"  - Suspicious fills: {merge_result.merge_quality_check.suspicious_fills}")
            
            trace.append({
                "step": f"merge_complete_iter_{iteration}",
                "iteration_complete": merge_result.iteration_complete,
                "quality_improved": merge_result.merge_quality_check.quality_improved,
                "timestamp": time.time()
            })
            
            # Update current extraction with merged result
            current_extraction = self._merge_result_to_text(merge_result)
            current_fields = {
                name: fv.value 
                for name, fv in merge_result.final_fields.items()
            }
            current_quality = refined_quality
            
            # Check if iteration is complete
            if merge_result.iteration_complete:
                print(f"[Agentic] Iteration complete, all fields confident")
                break
            
            if not merge_result.fields_still_uncertain:
                print(f"[Agentic] No more uncertain fields")
                break
        
        # =================================================================
        # FINAL QUALITY GATE: Accept or reject final result
        # =================================================================
        trace.append({"step": "final_quality_gate", "timestamp": time.time()})
        
        print(f"\n[Agentic] === Final Quality Gate ===")
        
        # Run final quality check
        final_quality = evaluate_quality(
            current_fields,
            current_extraction,
            strict_mode=self.strict_quality
        )
        
        print(f"[Agentic] Final quality score: {final_quality.quality_score}/100")
        print(f"[Agentic] Final quality level: {final_quality.quality_level.value}")
        
        # Determine final status
        if final_quality.quality_score < REJECT_THRESHOLD:
            quality_status = "rejected"
            print(f"[Agentic] REJECTED: Final quality too low")
        elif final_quality.quality_score < MIN_QUALITY_TO_ACCEPT:
            quality_status = "failed"
            print(f"[Agentic] FAILED: Quality below acceptance threshold")
        elif final_quality.quality_level in (QualityLevel.WARNING, QualityLevel.POOR):
            quality_status = "warning"
            print(f"[Agentic] WARNING: Quality issues detected, needs review")
        else:
            quality_status = "passed"
            print(f"[Agentic] PASSED: Quality acceptable")
        
        # =================================================================
        # BUILD FINAL RESULT
        # =================================================================
        processing_time = time.time() - start_time
        print(f"\n[Agentic] Complete! Processed in {processing_time:.1f}s with {iteration} iteration(s)")
        
        # Build result
        result = self._build_final_result(
            raw_text=initial_text,
            final_extraction=current_extraction,
            final_fields=current_fields,
            iterations=iteration,
            processing_time=processing_time,
            quality_result=final_quality,
            quality_status=quality_status,
            trace=trace
        )
        
        return result
    
    def _is_blank_region(self, image: Image.Image, threshold: float = 0.95) -> bool:
        """
        Check if a cropped region is mostly blank (empty field).
        
        Args:
            image: Cropped PIL Image
            threshold: Ratio of white pixels to consider blank
            
        Returns:
            True if region appears to be blank/empty
        """
        try:
            import numpy as np
            
            # Convert to grayscale
            gray = image.convert('L')
            pixels = np.array(gray)
            
            # Count "white" pixels (value > 240)
            white_ratio = np.sum(pixels > 240) / pixels.size
            
            return white_ratio > threshold
        except Exception:
            return False
    
    def _create_fallback_regions(
        self,
        fields_to_reexamine: List,
        image: Image.Image
    ) -> List[RegionEstimate]:
        """
        Create fallback regions when LLM estimation fails.
        
        Uses standard form layout assumptions.
        """
        # Define approximate positions for common fields
        FIELD_POSITIONS = {
            "اسم المالك": (0.4, 0.08, 0.95, 0.14),
            "الاسم": (0.4, 0.08, 0.95, 0.14),
            "رقم الهوية": (0.4, 0.12, 0.95, 0.18),
            "رقم بطاقة الأحوال": (0.4, 0.12, 0.95, 0.18),
            "تاريخ الميلاد": (0.4, 0.16, 0.95, 0.22),
            "تاريخ الإصدار": (0.4, 0.20, 0.95, 0.26),
            "تاريخ الانتهاء": (0.4, 0.24, 0.95, 0.30),
            "رقم الجوال": (0.4, 0.28, 0.95, 0.34),
            "المدينة": (0.4, 0.32, 0.95, 0.38),
            "الحي": (0.4, 0.36, 0.95, 0.42),
            "رقم اللوحة": (0.4, 0.50, 0.95, 0.56),
            "رقم رخصة القيادة": (0.4, 0.45, 0.95, 0.52),
        }
        
        regions = []
        for field in fields_to_reexamine:
            # Try to find a predefined position
            bbox = None
            for key, pos in FIELD_POSITIONS.items():
                if key in field.field_name or field.field_name in key:
                    bbox = pos
                    break
            
            if bbox is None:
                # Default to middle section
                bbox = (0.3, 0.3, 0.95, 0.6)
            
            regions.append(RegionEstimate(
                field_name=field.field_name,
                bbox_normalized=bbox,
                location_confidence="low",
                notes="Fallback position"
            ))
        
        return regions
    
    def _merge_result_to_text(self, merge_result: MergeResult) -> str:
        """Convert merge result back to text format for next iteration."""
        lines = []
        for field_name, field_value in merge_result.final_fields.items():
            confidence = field_value.confidence.upper()
            value = field_value.value
            lines.append(f"{field_name}: {value} [{confidence}]")
        return "\n".join(lines)
    
    def _create_error_result(
        self,
        error: str,
        processing_time: float,
        trace: List[Dict]
    ) -> AgenticResult:
        """Create an error result."""
        return AgenticResult(
            status="error",
            error=error,
            processing_time_seconds=processing_time,
            quality_score=0,
            quality_status="failed",
            trace=trace
        )
    
    def _create_rejected_result(
        self,
        raw_text: str,
        quality_result: QualityGateResult,
        processing_time: float,
        trace: List[Dict],
        reason: str
    ) -> AgenticResult:
        """Create a rejected result due to quality issues."""
        # Convert quality result to report
        quality_report = QualityReport(
            quality_score=quality_result.quality_score,
            quality_status="rejected",
            hallucination_indicators=quality_result.validation_result.hallucination_indicators,
            rejection_reasons=quality_result.rejection_reasons,
            warning_reasons=quality_result.warning_reasons,
            should_retry=quality_result.should_retry,
            needs_human_review=True,
            fields_to_review=quality_result.fields_to_review,
        )
        
        return AgenticResult(
            raw_text=raw_text,
            status="rejected",
            error=reason,
            processing_time_seconds=processing_time,
            quality_score=quality_result.quality_score,
            quality_status="rejected",
            quality_report=quality_report,
            hallucination_detected=True,
            hallucination_indicators=quality_result.validation_result.hallucination_indicators,
            trace=trace
        )
    
    def _build_final_result(
        self,
        raw_text: str,
        final_extraction: str,
        final_fields: Dict[str, str],
        iterations: int,
        processing_time: float,
        quality_result: QualityGateResult,
        quality_status: str,
        trace: List[Dict]
    ) -> AgenticResult:
        """Build the final AgenticResult from processed data."""
        
        # Parse fields from final extraction
        fields = []
        fields_needing_review = []
        confidence_summary = {"high": 0, "medium": 0, "low": 0, "empty": 0}
        
        for line in final_extraction.strip().split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            
            # Parse field_name: value [CONFIDENCE]
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            
            field_name = parts[0].strip()
            rest = parts[1].strip()
            
            # Extract confidence level
            confidence = "medium"
            if "[HIGH]" in rest.upper():
                confidence = "high"
                rest = rest.replace("[HIGH]", "").replace("[high]", "").strip()
            elif "[MEDIUM]" in rest.upper():
                confidence = "medium"
                rest = rest.replace("[MEDIUM]", "").replace("[medium]", "").strip()
            elif "[LOW]" in rest.upper():
                confidence = "low"
                rest = rest.replace("[LOW]", "").replace("[low]", "").strip()
            
            value = rest.strip()
            
            # Check for empty markers
            is_empty = "[فارغ]" in value or value.lower() == "empty"
            needs_review = confidence == "low" or "[غير واضح" in value
            
            # Get per-field validation score
            validation_score = quality_result.validation_result.field_scores.get(field_name, 50)
            
            if is_empty:
                confidence_summary["empty"] += 1
            else:
                confidence_summary[confidence] += 1
            
            if needs_review:
                fields_needing_review.append(field_name)
            
            fields.append(FieldResult(
                field_name=field_name,
                value=value,
                confidence=confidence,
                source="agentic",
                needs_review=needs_review,
                review_reason="Low confidence or unclear" if needs_review else None,
                is_empty=is_empty,
                validation_score=validation_score
            ))
        
        # Add fields flagged by quality gate
        for field_name in quality_result.fields_to_review:
            if field_name not in fields_needing_review:
                fields_needing_review.append(field_name)
        
        # Build quality report
        quality_report = QualityReport(
            quality_score=quality_result.quality_score,
            quality_status=quality_status,
            validation_issues=[
                ValidationIssue(
                    field_name=i.field_name,
                    issue_type=i.issue_type,
                    severity=i.severity.value if hasattr(i.severity, 'value') else str(i.severity),
                    message=i.message,
                    expected=i.expected,
                    actual=i.actual
                )
                for i in quality_result.validation_result.issues
            ],
            hallucination_indicators=quality_result.validation_result.hallucination_indicators,
            field_scores=quality_result.validation_result.field_scores,
            duplicate_values=quality_result.validation_result.duplicate_values,
            should_retry=quality_result.should_retry,
            needs_human_review=quality_result.needs_human_review,
            fields_to_review=quality_result.fields_to_review,
            rejection_reasons=quality_result.rejection_reasons,
            warning_reasons=quality_result.warning_reasons,
        )
        
        # Determine if hallucination was detected
        hallucination_detected = (
            quality_status == "rejected" or
            len(quality_result.validation_result.hallucination_indicators) > 0 or
            any("hallucination" in r.lower() for r in quality_result.rejection_reasons)
        )
        
        return AgenticResult(
            fields=fields,
            raw_text=raw_text,
            iterations_used=iterations,
            processing_time_seconds=processing_time,
            confidence_summary=confidence_summary,
            fields_needing_review=fields_needing_review,
            status="success" if quality_status in ("passed", "warning") else quality_status,
            quality_score=quality_result.quality_score,
            quality_status=quality_status,
            quality_report=quality_report,
            hallucination_detected=hallucination_detected,
            hallucination_indicators=quality_result.validation_result.hallucination_indicators,
            trace=trace
        )


# =============================================================================
# FALLBACK SINGLE-PASS CONTROLLER
# =============================================================================

class SinglePassController:
    """
    Fallback single-pass OCR controller.
    
    Used when LLM is unavailable or for simpler documents.
    Still includes basic quality checking.
    """
    
    def __init__(self, vlm_client: VLMClient, strict_quality: bool = True):
        self.vlm = vlm_client
        self.strict_quality = strict_quality
    
    async def process(
        self,
        image: Image.Image,
        custom_prompt: Optional[str] = None
    ) -> AgenticResult:
        """Process image with single-pass OCR and quality check."""
        start_time = time.time()
        
        text, success = await self.vlm.extract_initial(image, custom_prompt)
        
        if not success:
            return AgenticResult(
                status="error",
                error=text,
                processing_time_seconds=time.time() - start_time,
                quality_score=0,
                quality_status="failed"
            )
        
        # Run quality check
        fields = parse_extraction_to_fields(text)
        quality_result = evaluate_quality(fields, text, self.strict_quality)
        
        # Determine status based on quality
        if quality_result.quality_score < REJECT_THRESHOLD:
            quality_status = "rejected"
        elif quality_result.quality_score < MIN_QUALITY_TO_ACCEPT:
            quality_status = "failed"
        elif quality_result.quality_level in (QualityLevel.WARNING, QualityLevel.POOR):
            quality_status = "warning"
        else:
            quality_status = "passed"
        
        # Simple parsing
        result_fields = []
        for line in text.strip().split("\n"):
            if ":" in line:
                parts = line.split(":", 1)
                field_name = parts[0].strip()
                value = parts[1].strip() if len(parts) > 1 else ""
                
                # Remove confidence markers
                for marker in ['[HIGH]', '[MEDIUM]', '[LOW]']:
                    value = value.replace(marker, '').replace(marker.lower(), '').strip()
                
                result_fields.append(FieldResult(
                    field_name=field_name,
                    value=value,
                    confidence="medium",
                    source="single_pass",
                    needs_review=quality_status != "passed",
                    validation_score=quality_result.validation_result.field_scores.get(field_name, 50)
                ))
        
        return AgenticResult(
            fields=result_fields,
            raw_text=text,
            iterations_used=1,
            processing_time_seconds=time.time() - start_time,
            status="success" if quality_status in ("passed", "warning") else quality_status,
            quality_score=quality_result.quality_score,
            quality_status=quality_status,
            hallucination_detected=len(quality_result.validation_result.hallucination_indicators) > 0,
            hallucination_indicators=quality_result.validation_result.hallucination_indicators,
        )
