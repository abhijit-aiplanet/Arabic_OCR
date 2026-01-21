"""
Agentic OCR Controller

Main orchestration logic for the multi-pass, self-correcting OCR pipeline.

Flow:
1. Initial full-page OCR (VLM)
2. Analyze output for issues (LLM)
3. Estimate regions for uncertain fields (LLM)
4. Crop and re-OCR uncertain regions (VLM)
5. Merge original and refined results (LLM)
6. Repeat if needed, until confident or max iterations
"""

import asyncio
import time
from typing import Optional, Dict, List, Any
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
)
from .prompts import get_content_hint


# =============================================================================
# AGENTIC OCR CONTROLLER
# =============================================================================

class AgenticOCRController:
    """
    Multi-pass, self-correcting OCR controller.
    
    Uses dual-model architecture:
    - VLM (AIN) for vision/OCR tasks
    - LLM (Qwen 2.5) for reasoning/orchestration
    
    Example usage:
        controller = AgenticOCRController(vlm_client, llm_client)
        result = await controller.process(image)
    """
    
    def __init__(
        self,
        vlm_client: VLMClient,
        llm_client: LLMClient,
        max_iterations: int = 3,
        min_fields_to_reexamine: int = 1,
        use_grid_fallback: bool = True
    ):
        """
        Initialize the agentic controller.
        
        Args:
            vlm_client: Client for vision/OCR model
            llm_client: Client for reasoning model
            max_iterations: Maximum refinement iterations
            min_fields_to_reexamine: Minimum fields to warrant re-examination
            use_grid_fallback: Whether to use grid cropping if region estimation fails
        """
        self.vlm = vlm_client
        self.llm = llm_client
        self.max_iterations = max_iterations
        self.min_fields_to_reexamine = min_fields_to_reexamine
        self.use_grid_fallback = use_grid_fallback
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
            AgenticResult with all extracted fields and metadata
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
            return AgenticResult(
                status="error",
                error=f"Initial OCR failed: {initial_text}",
                processing_time_seconds=time.time() - start_time,
                trace=trace
            )
        
        trace.append({
            "step": "initial_ocr_complete",
            "text_length": len(initial_text),
            "timestamp": time.time()
        })
        
        # Track current best extraction
        current_extraction = initial_text
        iteration = 0
        
        # =================================================================
        # ITERATION LOOP
        # =================================================================
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n[Agentic] Iteration {iteration}/{self.max_iterations}")
            
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
                "timestamp": time.time()
            })
            
            # Check if we need to re-examine any fields
            if not analysis.has_fields_to_reexamine():
                print(f"[Agentic] All fields confident, stopping iteration")
                break
            
            if len(analysis.fields_to_reexamine) < self.min_fields_to_reexamine:
                print(f"[Agentic] Only {len(analysis.fields_to_reexamine)} uncertain fields, skipping re-examination")
                break
            
            print(f"[Agentic] Found {len(analysis.fields_to_reexamine)} fields to re-examine")
            
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
                    # Use grid-based cropping as fallback
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
                    
                    # Enhance the crop
                    cropped = enhance_cropped_region(cropped)
                    
                    # Find previous value for context
                    previous_value = ""
                    for f in analysis.fields_to_reexamine:
                        if f.field_name == region.field_name:
                            previous_value = f.current_value
                            break
                    
                    # Re-OCR the cropped region
                    value, success = await self.vlm.extract_field(
                        cropped,
                        region.field_name,
                        previous_value=previous_value,
                        content_hint=get_content_hint(region.field_name)
                    )
                    
                    if success:
                        refined_values[region.field_name] = value
                        print(f"    Refined: {value[:50]}..." if len(value) > 50 else f"    Refined: {value}")
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
            
            trace.append({
                "step": f"merge_complete_iter_{iteration}",
                "iteration_complete": merge_result.iteration_complete,
                "timestamp": time.time()
            })
            
            # Update current extraction with merged result
            current_extraction = self._merge_result_to_text(merge_result)
            
            # Check if iteration is complete
            if merge_result.iteration_complete:
                print(f"[Agentic] Iteration complete, all fields confident")
                break
            
            if not merge_result.fields_still_uncertain:
                print(f"[Agentic] No more uncertain fields")
                break
        
        # =================================================================
        # BUILD FINAL RESULT
        # =================================================================
        processing_time = time.time() - start_time
        print(f"\n[Agentic] Complete! Processed in {processing_time:.1f}s with {iteration} iteration(s)")
        
        # Parse the final extraction into structured fields
        result = self._build_final_result(
            raw_text=initial_text,
            final_extraction=current_extraction,
            iterations=iteration,
            processing_time=processing_time,
            trace=trace
        )
        
        return result
    
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
            "اسم المالك": (0.5, 0.08, 0.95, 0.15),
            "الاسم": (0.5, 0.08, 0.95, 0.15),
            "رقم الهوية": (0.5, 0.12, 0.95, 0.18),
            "تاريخ الميلاد": (0.5, 0.16, 0.95, 0.22),
            "رقم الجوال": (0.5, 0.20, 0.95, 0.28),
            "المدينة": (0.5, 0.24, 0.95, 0.32),
            "الحي": (0.5, 0.28, 0.95, 0.36),
            "رقم اللوحة": (0.5, 0.45, 0.95, 0.55),
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
    
    def _build_final_result(
        self,
        raw_text: str,
        final_extraction: str,
        iterations: int,
        processing_time: float,
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
                is_empty=is_empty
            ))
        
        return AgenticResult(
            fields=fields,
            raw_text=raw_text,
            iterations_used=iterations,
            processing_time_seconds=processing_time,
            confidence_summary=confidence_summary,
            fields_needing_review=fields_needing_review,
            status="success",
            trace=trace
        )


# =============================================================================
# FALLBACK SINGLE-PASS CONTROLLER
# =============================================================================

class SinglePassController:
    """
    Fallback single-pass OCR controller.
    
    Used when LLM is unavailable or for simpler documents.
    """
    
    def __init__(self, vlm_client: VLMClient):
        self.vlm = vlm_client
    
    async def process(
        self,
        image: Image.Image,
        custom_prompt: Optional[str] = None
    ) -> AgenticResult:
        """Process image with single-pass OCR."""
        start_time = time.time()
        
        text, success = await self.vlm.extract_initial(image, custom_prompt)
        
        if not success:
            return AgenticResult(
                status="error",
                error=text,
                processing_time_seconds=time.time() - start_time
            )
        
        # Simple parsing
        fields = []
        for line in text.strip().split("\n"):
            if ":" in line:
                parts = line.split(":", 1)
                fields.append(FieldResult(
                    field_name=parts[0].strip(),
                    value=parts[1].strip() if len(parts) > 1 else "",
                    confidence="medium",
                    source="single_pass",
                    needs_review=False
                ))
        
        return AgenticResult(
            fields=fields,
            raw_text=text,
            iterations_used=1,
            processing_time_seconds=time.time() - start_time,
            status="success"
        )
