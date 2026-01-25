"""
ReAct-Style Agentic OCR Agent

The agent follows the ReAct pattern:
1. THINK: Analyze current state and decide what to do
2. ACT: Use a tool to gather information or perform action
3. OBSERVE: Process tool results
4. REFLECT: Evaluate progress and decide next step
5. REPEAT until confident or max iterations

This creates a self-improving, reasoning OCR system.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from PIL import Image
from enum import Enum

from .tools import OCRTools, ToolResult, FORM_REGIONS
from .azure_client import AzureVisionOCR


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
                    "observation": s.observation[:200] if s.observation else None,
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
    ReAct-style agent for Arabic OCR.
    
    The agent reasons about what to do, uses tools, observes results,
    and iteratively refines its extraction until confident.
    """
    
    MAX_ITERATIONS = 10
    MAX_REFINEMENT_ATTEMPTS = 3
    MIN_CONFIDENCE_THRESHOLD = 0.6  # 60% high-confidence fields
    
    def __init__(
        self,
        azure_client: Optional[AzureVisionOCR] = None,
        on_step: Optional[Callable[[ThoughtStep], None]] = None,
    ):
        """
        Initialize the agent.
        
        Args:
            azure_client: Azure OpenAI client for vision
            on_step: Callback for each reasoning step (for UI updates)
        """
        self.azure = azure_client or AzureVisionOCR()
        self.tools = OCRTools(self.azure)
        self.on_step = on_step  # Callback for real-time updates
        
        # Current state
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
    
    async def process(
        self,
        image: Image.Image,
        user_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process an image through the agentic OCR pipeline.
        
        Args:
            image: PIL Image to process
            user_context: Optional context from user corrections
            
        Returns:
            Complete extraction result with trace
        """
        start_time = time.time()
        self.trace = AgentTrace()
        self.fields = {}
        self.confidence = {}
        
        print("[Agent] Starting agentic OCR process...")
        
        # =================================================================
        # PHASE 1: INITIAL ASSESSMENT
        # =================================================================
        step = self.trace.add_step(
            state=AgentState.THINKING.value,
            thought="Starting OCR. First, I'll extract all visible text from the full document to get an overview.",
        )
        self._emit_step(step)
        
        # =================================================================
        # PHASE 2: FULL DOCUMENT EXTRACTION
        # =================================================================
        step = self.trace.add_step(
            state=AgentState.ACTING.value,
            thought="Using extract_full_document tool to get initial reading.",
            action="extract_full_document",
            action_input={"image": "full_document"},
        )
        self._emit_step(step)
        
        result = await self.tools.extract_full_document(image)
        self.trace.tool_calls += 1
        
        step = self.trace.add_step(
            state=AgentState.OBSERVING.value,
            thought=f"Full extraction complete. Got {len(result.result) if result.result else 0} fields.",
            observation=result.raw_output[:500] if result.raw_output else "No output",
            result=result.result,
        )
        self._emit_step(step)
        
        # Store initial results
        if result.result:
            for field_name, data in result.result.items():
                self.fields[field_name] = data.get("value", "---")
                self.confidence[field_name] = data.get("confidence", "LOW")
        
        # =================================================================
        # PHASE 3: REGION-BY-REGION EXTRACTION
        # =================================================================
        step = self.trace.add_step(
            state=AgentState.THINKING.value,
            thought="Now I'll extract each region separately for better accuracy.",
        )
        self._emit_step(step)
        
        for region_name in ["header", "general_data", "address", "driving_license", "footer"]:
            step = self.trace.add_step(
                state=AgentState.ACTING.value,
                thought=f"Reading {region_name} region...",
                action="read_region",
                action_input={"region": region_name},
            )
            self._emit_step(step)
            
            result = await self.tools.read_region(image, region_name)
            self.trace.tool_calls += 1
            
            # Merge results
            if result.result:
                for field_name, data in result.result.items():
                    new_conf = data.get("confidence", "LOW")
                    existing_conf = self.confidence.get(field_name, "")
                    
                    # Prefer higher confidence
                    should_update = (
                        field_name not in self.fields or
                        self._confidence_score(new_conf) > self._confidence_score(existing_conf) or
                        self.fields.get(field_name) in ["---", ""]
                    )
                    
                    if should_update and data.get("value") not in ["---", ""]:
                        self.fields[field_name] = data.get("value", "---")
                        self.confidence[field_name] = new_conf
            
            step = self.trace.add_step(
                state=AgentState.OBSERVING.value,
                thought=f"Extracted from {region_name}: {len(result.result) if result.result else 0} fields",
                observation=result.reasoning,
            )
            self._emit_step(step)
        
        # =================================================================
        # PHASE 4: REFLECTION
        # =================================================================
        step = self.trace.add_step(
            state=AgentState.REFLECTING.value,
            thought="Analyzing extraction quality and identifying fields that need refinement.",
            action="reflect",
        )
        self._emit_step(step)
        
        reflection = await self.tools.reflect(self.fields, self.confidence)
        self.trace.tool_calls += 1
        
        needs_attention = reflection.result.get("needs_attention", [])
        issues = reflection.result.get("issues", [])
        
        step = self.trace.add_step(
            state=AgentState.OBSERVING.value,
            thought=f"Reflection complete. {len(needs_attention)} fields need attention. Issues: {len(issues)}",
            observation=reflection.result.get("recommendation", ""),
        )
        self._emit_step(step)
        
        # =================================================================
        # PHASE 5: TARGETED REFINEMENT
        # =================================================================
        if needs_attention:
            step = self.trace.add_step(
                state=AgentState.REFINING.value,
                thought=f"Zooming in on unclear fields: {', '.join(needs_attention[:5])}",
            )
            self._emit_step(step)
            
            # Determine which region each field is in
            field_to_region = self._map_fields_to_regions()
            
            for field_name in needs_attention[:5]:  # Limit refinements
                region_name = field_to_region.get(field_name, "general_data")
                
                step = self.trace.add_step(
                    state=AgentState.ACTING.value,
                    thought=f"Zooming 3x on field: {field_name}",
                    action="zoom_and_read",
                    action_input={"field": field_name, "region": region_name, "zoom": 3},
                )
                self._emit_step(step)
                
                result = await self.tools.zoom_and_read(image, field_name, region_name, zoom_level=3)
                self.trace.tool_calls += 1
                
                if result.result:
                    new_value = result.result.get("value", "---")
                    new_conf = result.result.get("confidence", "LOW")
                    
                    if new_value and new_value != "---":
                        self.fields[field_name] = new_value
                        self.confidence[field_name] = new_conf
                
                step = self.trace.add_step(
                    state=AgentState.OBSERVING.value,
                    thought=f"Zoom result for {field_name}: {result.result.get('value', '---') if result.result else '---'}",
                    confidence=result.confidence,
                )
                self._emit_step(step)
        
        # =================================================================
        # PHASE 6: VALIDATION
        # =================================================================
        step = self.trace.add_step(
            state=AgentState.VALIDATING.value,
            thought="Validating all extracted fields against expected formats.",
        )
        self._emit_step(step)
        
        validation_issues = []
        for field_name, value in self.fields.items():
            result = self.tools.validate_field(field_name, value)
            if result.result and not result.result.get("valid", True):
                validation_issues.append({
                    "field": field_name,
                    "value": value,
                    "reason": result.result.get("reason", "Unknown"),
                })
                # Lower confidence for invalid fields
                self.confidence[field_name] = "LOW"
        
        self.trace.tool_calls += len(self.fields)
        
        step = self.trace.add_step(
            state=AgentState.OBSERVING.value,
            thought=f"Validation complete. Found {len(validation_issues)} issues.",
            observation=str(validation_issues) if validation_issues else "All fields valid",
        )
        self._emit_step(step)
        
        # =================================================================
        # PHASE 7: FINAL REFLECTION
        # =================================================================
        final_reflection = await self.tools.reflect(self.fields, self.confidence)
        
        step = self.trace.add_step(
            state=AgentState.COMPLETE.value,
            thought=f"Extraction complete. {len(self.fields)} fields extracted.",
            observation=final_reflection.result.get("recommendation", ""),
        )
        self._emit_step(step)
        
        # Calculate quality score
        total = len(self.fields)
        high_conf = sum(1 for c in self.confidence.values() if c == "HIGH")
        quality_score = int((high_conf / total * 100) if total > 0 else 0)
        
        # Build final result
        total_time = time.time() - start_time
        self.trace.total_time = total_time
        self.trace.iterations = 1
        self.trace.quality_score = quality_score
        self.trace.final_fields = self.fields
        
        print(f"[Agent] Complete! {len(self.fields)} fields in {total_time:.1f}s")
        print(f"[Agent] Quality: {quality_score}% ({high_conf}/{total} high confidence)")
        
        return self._build_result(validation_issues)
    
    def _confidence_score(self, conf: str) -> int:
        """Convert confidence to numeric score."""
        return {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(conf.upper(), 0)
    
    def _map_fields_to_regions(self) -> Dict[str, str]:
        """Map field names to their likely regions."""
        mapping = {}
        
        # Header fields
        for f in ["نوع_النشاط", "مدينة_مزاولة_النشاط", "تاريخ_بدء", "تاريخ_انتهاء"]:
            mapping[f] = "header"
        
        # General data fields
        for f in ["اسم_المالك", "رقم_الهوية", "مصدرها", "تاريخها", "تاريخ_الميلاد", "الحالة_الاجتماعية", "المؤهل"]:
            mapping[f] = "general_data"
        
        # Address fields
        for f in ["المدينة", "الحي", "الشارع", "رقم_المبنى", "جوال", "البريد_الإلكتروني", "فاكس"]:
            mapping[f] = "address"
        
        # License fields
        for f in ["رقمها", "تاريخ_الإصدار", "تاريخ_الانتهاء", "مصدر_الرخصة"]:
            mapping[f] = "driving_license"
        
        # Footer fields
        for f in ["اسم_مقدم_الطلب", "صفته", "توقيعه", "التاريخ"]:
            mapping[f] = "footer"
        
        return mapping
    
    def _build_result(self, validation_issues: List[Dict]) -> Dict[str, Any]:
        """Build the final result structure."""
        # Build field results
        field_results = []
        fields_needing_review = []
        confidence_summary = {"high": 0, "medium": 0, "low": 0, "empty": 0}
        
        for field_name, value in self.fields.items():
            conf = self.confidence.get(field_name, "LOW")
            is_empty = value in ["---", "[فارغ]", "غير موجود", ""]
            
            # Force LOW confidence for empty
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
            
            # Find validation issue
            val_issue = next((v for v in validation_issues if v["field"] == field_name), None)
            
            field_results.append({
                "field_name": field_name,
                "value": value,
                "confidence": conf.lower(),
                "needs_review": needs_review,
                "review_reason": val_issue["reason"] if val_issue else ("Empty value" if is_empty else None),
                "is_empty": is_empty,
                "validation_score": 100 if not val_issue else 50,
            })
        
        # Quality determination
        total = len(self.fields)
        high = confidence_summary["high"]
        quality_score = int((high / total * 100) if total > 0 else 0)
        
        if quality_score < 20:
            quality_status = "rejected"
        elif quality_score < 40:
            quality_status = "failed"
        elif len(validation_issues) > 3 or len(fields_needing_review) > total * 0.5:
            quality_status = "warning"
        else:
            quality_status = "passed"
        
        return {
            "fields": field_results,
            "raw_text": "\n".join(f"{k}: {v}" for k, v in self.fields.items()),
            "iterations_used": self.trace.iterations,
            "processing_time_seconds": self.trace.total_time,
            "confidence_summary": confidence_summary,
            "fields_needing_review": fields_needing_review,
            "status": "success" if quality_status in ("passed", "warning") else quality_status,
            "quality_score": quality_score,
            "quality_status": quality_status,
            "agent_trace": self.trace.to_dict(),
            "tool_calls": self.trace.tool_calls,
            "hallucination_detected": len(validation_issues) > 5,
            "hallucination_indicators": [v["reason"] for v in validation_issues],
        }


# =============================================================================
# FACTORY
# =============================================================================

def create_agent(
    azure_client: Optional[AzureVisionOCR] = None,
    on_step: Optional[Callable] = None,
) -> AgenticOCRAgent:
    """Create an agentic OCR agent."""
    return AgenticOCRAgent(azure_client=azure_client, on_step=on_step)
