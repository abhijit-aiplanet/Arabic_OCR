"""
Agentic OCR Agent with Surgical Extraction

Combines the proven surgical extraction pipeline with agent-style
reasoning trace for transparency. Uses the surgical controller
internally for extraction quality while providing step-by-step
visibility into the process.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from PIL import Image
from enum import Enum

from .azure_client import AzureVisionOCR
from .image_processor import ImageProcessor, SectionType
from .format_validator import FormatValidator
from .prompts import get_section_prompt, SECTION_PROMPTS


class AgentState(Enum):
    """Current state of the agent."""
    STARTING = "starting"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    REFLECTING = "reflecting"
    REFINING = "refining"
    VALIDATING = "validating"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ThoughtStep:
    """A single step in the agent's reasoning."""
    step_num: int
    state: str
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict] = None
    observation: Optional[str] = None
    result: Optional[Any] = None
    confidence: str = "MEDIUM"
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentTrace:
    """Complete trace of agent's reasoning and actions."""
    steps: List[ThoughtStep] = field(default_factory=list)
    final_fields: Dict[str, Any] = field(default_factory=dict)
    total_time: float = 0.0
    tool_calls: int = 0
    iterations: int = 0
    quality_score: int = 0
    
    def add_step(self, **kwargs) -> ThoughtStep:
        step = ThoughtStep(step_num=len(self.steps) + 1, **kwargs)
        self.steps.append(step)
        return step
    
    def to_dict(self) -> Dict:
        return {
            "steps": [
                {
                    "step": s.step_num,
                    "state": s.state,
                    "thought": s.thought,
                    "action": s.action,
                    "observation": s.observation[:300] if s.observation else None,
                    "confidence": s.confidence,
                }
                for s in self.steps
            ],
            "total_time": self.total_time,
            "tool_calls": self.tool_calls,
            "iterations": self.iterations,
            "quality_score": self.quality_score,
        }


class AgenticOCRAgent:
    """
    Agentic OCR using surgical extraction with reasoning trace.
    
    Uses the proven surgical pipeline internally but exposes
    a step-by-step trace for UI transparency.
    """
    
    def __init__(
        self,
        azure_client: Optional[AzureVisionOCR] = None,
        on_step: Optional[Callable[[ThoughtStep], None]] = None,
    ):
        self.azure = azure_client or AzureVisionOCR()
        self.processor = ImageProcessor()
        self.validator = FormatValidator()
        self.on_step = on_step
        
        self.fields: Dict[str, str] = {}
        self.confidence: Dict[str, str] = {}
        self.trace = AgentTrace()
    
    def _emit_step(self, step: ThoughtStep):
        """Emit a step to the callback if provided."""
        if self.on_step:
            try:
                self.on_step(step)
            except Exception as e:
                print(f"[Agent] Step callback error: {e}")
    
    def _add_step(self, state: str, thought: str, **kwargs) -> ThoughtStep:
        """Add and emit a step."""
        step = self.trace.add_step(state=state, thought=thought, **kwargs)
        self._emit_step(step)
        return step
    
    async def process(
        self,
        image: Image.Image,
        user_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process image using surgical extraction with agent trace."""
        start_time = time.time()
        self.trace = AgentTrace()
        self.fields = {}
        self.confidence = {}
        
        print("[Agent] Starting surgical agentic OCR...")
        
        # =================================================================
        # STAGE 1: PREPROCESSING
        # =================================================================
        self._add_step(
            AgentState.STARTING.value,
            "Initializing OCR pipeline. Preprocessing image for optimal extraction."
        )
        
        enhanced = self.processor.preprocess(image, contrast=2.0)
        self.trace.tool_calls += 1
        
        self._add_step(
            AgentState.OBSERVING.value,
            f"Image preprocessed: {image.size[0]}x{image.size[1]} pixels, contrast enhanced"
        )
        
        # =================================================================
        # STAGE 2: SECTION DETECTION
        # =================================================================
        self._add_step(
            AgentState.THINKING.value,
            "Detecting document sections: header, general data, address, license, vehicle, footer"
        )
        
        sections = self.processor.detect_sections(enhanced)
        self.trace.tool_calls += 1
        
        self._add_step(
            AgentState.OBSERVING.value,
            f"Detected {len(sections)} sections in document",
            observation=", ".join([s.name for s in sections])
        )
        
        # =================================================================
        # STAGE 3: SECTION-BY-SECTION EXTRACTION
        # =================================================================
        self._add_step(
            AgentState.ACTING.value,
            "Beginning section-by-section extraction with specialized prompts"
        )
        
        all_fields: Dict[str, str] = {}
        all_confidence: Dict[str, str] = {}
        unclear_fields: List[str] = []
        
        for section in sections:
            self._add_step(
                AgentState.ACTING.value,
                f"Extracting section: {section.name}",
                action="extract_section",
                action_input={"section": section.name}
            )
            
            # Crop and upscale section
            section_image = self.processor.crop_section(enhanced, section, upscale=2)
            
            # Get section-specific prompt
            section_prompt = get_section_prompt(section.section_type.value)
            
            # Extract using Azure Vision
            image_b64 = self.processor.image_to_base64(section_image)
            result = await self.azure.extract_section(image_b64, section_prompt)
            self.trace.tool_calls += 1
            
            if result.success:
                fields, conf = self._parse_extraction(result.text)
                all_fields.update(fields)
                all_confidence.update(conf)
                
                # Track unclear fields
                for field_name, c in conf.items():
                    if c in ["LOW", "MEDIUM"]:
                        unclear_fields.append(field_name)
                
                self._add_step(
                    AgentState.OBSERVING.value,
                    f"Extracted {len(fields)} fields from {section.name}",
                    observation=", ".join(fields.keys()) if fields else "No fields found"
                )
            else:
                self._add_step(
                    AgentState.OBSERVING.value,
                    f"Section {section.name}: extraction failed",
                    observation=result.error
                )
        
        # =================================================================
        # STAGE 4: ZOOM REFINEMENT
        # =================================================================
        if unclear_fields:
            self._add_step(
                AgentState.REFINING.value,
                f"Refining {len(unclear_fields)} unclear fields with zoom"
            )
            
            refined_count = 0
            for field_name in unclear_fields[:8]:  # Limit to top 8
                self._add_step(
                    AgentState.ACTING.value,
                    f"Zooming on field: {field_name}",
                    action="zoom_and_read"
                )
                
                zoom_levels = self.processor.iterative_zoom(enhanced, field_name)
                self.trace.tool_calls += 1
                
                if not zoom_levels:
                    continue
                
                # Try highest zoom
                zoomed_image, scale = zoom_levels[-1]
                
                if self.processor.is_blank_region(zoomed_image):
                    all_confidence[field_name] = "LOW"
                    continue
                
                image_b64 = self.processor.image_to_base64(zoomed_image)
                result = await self.azure.extract_field(image_b64, field_name)
                self.trace.tool_calls += 1
                
                if result.value and result.value != "---":
                    all_fields[field_name] = result.value
                    all_confidence[field_name] = result.confidence
                    refined_count += 1
                    
                    self._add_step(
                        AgentState.OBSERVING.value,
                        f"Refined {field_name}: {result.value[:30]}...",
                        confidence=result.confidence
                    )
            
            self._add_step(
                AgentState.OBSERVING.value,
                f"Zoom refinement complete: {refined_count} fields improved"
            )
        
        # =================================================================
        # STAGE 5: VALIDATION
        # =================================================================
        self._add_step(
            AgentState.VALIDATING.value,
            "Validating extracted fields against Saudi document formats"
        )
        
        validation = self.validator.validate_document(all_fields)
        self.trace.tool_calls += 1
        
        # Update confidence based on validation
        for field_name, result in validation.field_results.items():
            if not result.is_valid:
                all_confidence[field_name] = "LOW"
        
        self._add_step(
            AgentState.OBSERVING.value,
            f"Validation: {validation.overall_score}/100 score, {len(validation.critical_issues)} issues",
            observation="; ".join(validation.critical_issues[:3]) if validation.critical_issues else "No critical issues"
        )
        
        # =================================================================
        # STAGE 6: SELF-CRITIQUE
        # =================================================================
        self._add_step(
            AgentState.REFLECTING.value,
            "Running self-critique to detect potential hallucinations"
        )
        
        critique = await self.azure.self_critique(all_fields)
        self.trace.tool_calls += 1
        
        if critique.has_issues:
            self._add_step(
                AgentState.OBSERVING.value,
                f"Self-critique found {len(critique.issues)} potential issues",
                observation="; ".join([i.get("message", "") for i in critique.issues[:3]])
            )
            
            for issue in critique.issues:
                field_name = issue.get("field", "")
                if field_name in all_confidence:
                    all_confidence[field_name] = "LOW"
        else:
            self._add_step(
                AgentState.OBSERVING.value,
                "Self-critique passed: no hallucination indicators detected"
            )
        
        # =================================================================
        # STAGE 7: BUILD RESULT
        # =================================================================
        self.fields = all_fields
        self.confidence = all_confidence
        
        total_time = time.time() - start_time
        
        # Calculate quality
        total = len(all_fields)
        high_conf = sum(1 for c in all_confidence.values() if c == "HIGH")
        quality_score = int((high_conf / total * 100) if total > 0 else 0)
        
        # Boost quality score based on validation
        quality_score = max(quality_score, validation.overall_score)
        
        self.trace.total_time = total_time
        self.trace.iterations = 1
        self.trace.quality_score = quality_score
        
        self._add_step(
            AgentState.COMPLETE.value,
            f"Extraction complete: {len(all_fields)} fields, {quality_score}% quality in {total_time:.1f}s"
        )
        
        print(f"[Agent] Complete! {len(all_fields)} fields in {total_time:.1f}s")
        print(f"[Agent] Quality: {quality_score}% ({high_conf}/{total} high confidence)")
        
        return self._build_result(all_fields, all_confidence, validation)
    
    def _parse_extraction(self, text: str) -> tuple:
        """Parse extraction text into fields and confidence."""
        fields = {}
        confidence = {}
        
        EMPTY_MARKERS = ["---", "[فارغ]", "[EMPTY]", "غير موجود", ""]
        
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            
            field_name = parts[0].strip()
            rest = parts[1].strip()
            
            # Extract confidence
            conf = "MEDIUM"
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
                conf = "LOW"
                rest = "---"
            
            value = rest.strip()
            
            # Empty values get LOW confidence
            if any(m in value or value == m for m in EMPTY_MARKERS):
                conf = "LOW"
                value = "---"
            
            if "؟" in value:
                conf = "LOW"
            
            fields[field_name] = value
            confidence[field_name] = conf
        
        return fields, confidence
    
    def _build_result(
        self,
        fields: Dict[str, str],
        confidence: Dict[str, str],
        validation,
    ) -> Dict[str, Any]:
        """Build final result structure."""
        field_results = []
        fields_needing_review = []
        confidence_summary = {"high": 0, "medium": 0, "low": 0, "empty": 0}
        
        EMPTY_MARKERS = ["---", "[فارغ]", "غير موجود", ""]
        
        for field_name, value in fields.items():
            conf = confidence.get(field_name, "LOW")
            is_empty = any(m in value or value == m for m in EMPTY_MARKERS)
            
            if is_empty:
                conf = "LOW"
                value = "---"
            
            needs_review = conf == "LOW" or is_empty or "؟" in value
            
            if is_empty:
                confidence_summary["empty"] += 1
            else:
                confidence_summary[conf.lower()] += 1
            
            if needs_review:
                fields_needing_review.append(field_name)
            
            val_result = validation.field_results.get(field_name)
            validation_score = 100 if (not val_result or val_result.is_valid) else 50
            
            field_results.append({
                "field_name": field_name,
                "value": value,
                "confidence": conf.lower(),
                "needs_review": needs_review,
                "review_reason": "Needs verification" if needs_review else None,
                "is_empty": is_empty,
                "validation_score": validation_score,
            })
        
        total = len(fields)
        high = confidence_summary["high"]
        quality_score = self.trace.quality_score
        
        if quality_score < 20:
            quality_status = "rejected"
        elif quality_score < 40:
            quality_status = "failed"
        elif len(fields_needing_review) > total * 0.5:
            quality_status = "warning"
        else:
            quality_status = "passed"
        
        return {
            "fields": field_results,
            "raw_text": "\n".join(f"{k}: {v}" for k, v in fields.items()),
            "iterations_used": self.trace.iterations,
            "processing_time_seconds": self.trace.total_time,
            "confidence_summary": confidence_summary,
            "fields_needing_review": fields_needing_review,
            "status": "success" if quality_status in ("passed", "warning") else quality_status,
            "quality_score": quality_score,
            "quality_status": quality_status,
            "agent_trace": self.trace.to_dict(),
            "tool_calls": self.trace.tool_calls,
            "hallucination_detected": len(validation.hallucination_indicators) > 0,
            "hallucination_indicators": validation.hallucination_indicators,
        }


def create_agent(
    azure_client: Optional[AzureVisionOCR] = None,
    on_step: Optional[Callable] = None,
) -> AgenticOCRAgent:
    """Create an agentic OCR agent."""
    return AgenticOCRAgent(azure_client=azure_client, on_step=on_step)
