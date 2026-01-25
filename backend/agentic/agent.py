"""
Simple Agentic OCR Agent

PRINCIPLE: Simplicity over complexity. Functionality first.

This agent sends the FULL image to the vision model with a clear prompt,
then parses the response. No over-engineered section detection or zoom.
"""

import time
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from PIL import Image
from enum import Enum
import base64
import io

from .azure_client import AzureVisionOCR
from .format_validator import FormatValidator


def log(msg: str):
    """Force-flush logging."""
    print(msg)
    sys.stdout.flush()


class AgentState(Enum):
    STARTING = "starting"
    EXTRACTING = "extracting"
    VALIDATING = "validating"
    COMPLETE = "complete"


@dataclass
class ThoughtStep:
    step_num: int
    state: str
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    confidence: str = "MEDIUM"
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentTrace:
    steps: List[ThoughtStep] = field(default_factory=list)
    total_time: float = 0.0
    tool_calls: int = 0
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
            "iterations": 1,
            "quality_score": self.quality_score,
        }


# The ONE prompt - VERY conservative, no guessing
EXTRACTION_PROMPT = """اقرأ هذا النموذج السعودي واستخرج القيم المكتوبة.

## قواعد صارمة جداً:

### الأسماء (اسم_المالك، اسم_مقدم_الطلب):
- إذا لم تستطع قراءة الاسم بوضوح تام: اكتب ---
- لا تخمن أسماء عربية
- الخطأ في الاسم خطير جداً

### نوع_النشاط:
- إذا غير واضح أو صعب القراءة: اكتب ---
- لا تخمن نوع النشاط

### الأرقام (رقم_الهوية، رقم_رخصة_القيادة):
- اكتب الأرقام التي تراها بوضوح
- إذا رقم غير واضح أضف ؟ بدل الرقم
- مثال: ١٠٣٨؟؟٦٦٨٠

### التواريخ:
- اكتب التاريخ كما تراه
- لا تصحح أو تغير الأرقام

### الحقول الفارغة:
- إذا الحقل فارغ في الصورة: ---

## الحقول المطلوبة:
اسم_المالك، رقم_الهوية، مصدرها، تاريخها، تاريخ_الميلاد، الحالة_الاجتماعية، عدد_من_يعولهم، المؤهل، المدينة، الحي، الشارع، رقم_المبنى، جوال، البريد_الإلكتروني، فاكس، نوع_النشاط، مدينة_مزاولة_النشاط، تاريخ_بدء_النشاط، تاريخ_انتهاء_الترخيص، رقم_رخصة_القيادة، تاريخ_إصدار_الرخصة، تاريخ_انتهاء_الرخصة، مصدر_الرخصة، نوع_المركبة، الموديل، اللون، رقم_اللوحة، سنة_الصنع، اسم_مقدم_الطلب، صفته، توقيعه، تاريخ_التوقيع

## التنسيق:
حقل: قيمة [HIGH/MEDIUM/LOW]

## مثال:
رقم_الهوية: ١٠٣٨٣٢٦٦٨٠ [HIGH]
اسم_المالك: --- [LOW]
المؤهل: أمي [HIGH]
جوال: ٠٥٠٧٤٧٧٩٩٨ [HIGH]
نوع_النشاط: --- [LOW]
الحي: --- [LOW]
توقيعه: [توقيع موجود] [HIGH]

ابدأ:"""


class AgenticOCRAgent:
    """Simple, effective OCR agent. Functionality first."""
    
    def __init__(
        self,
        azure_client: Optional[AzureVisionOCR] = None,
        on_step: Optional[Callable] = None,
    ):
        self.azure = azure_client or AzureVisionOCR()
        self.validator = FormatValidator()
        self.on_step = on_step
        self.trace = AgentTrace()
    
    def _emit_step(self, step: ThoughtStep):
        if self.on_step:
            try:
                self.on_step(step)
            except:
                pass
    
    def _add_step(self, state: str, thought: str, **kwargs) -> ThoughtStep:
        step = self.trace.add_step(state=state, thought=thought, **kwargs)
        self._emit_step(step)
        return step
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64."""
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    async def process(self, image: Image.Image, user_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Process image with simple, direct approach.
        
        1. Send full image to Azure Vision
        2. Parse the response
        3. Validate
        4. Done.
        """
        start_time = time.time()
        self.trace = AgentTrace()
        
        log("[Agent] Starting simple extraction...")
        
        # Step 1: Prepare
        self._add_step(
            AgentState.STARTING.value,
            f"Processing image: {image.size[0]}x{image.size[1]} pixels"
        )
        
        # Step 2: Extract everything in ONE call
        self._add_step(
            AgentState.EXTRACTING.value,
            "Sending full image to vision model for extraction",
            action="extract_all"
        )
        
        image_b64 = self._image_to_base64(image)
        
        log("[Agent] Calling Azure Vision API...")
        result = await self.azure.extract_section(image_b64, EXTRACTION_PROMPT)
        self.trace.tool_calls += 1
        
        if not result.success:
            log(f"[Agent] EXTRACTION FAILED: {result.error}")
            self._add_step(
                AgentState.COMPLETE.value,
                f"Extraction failed: {result.error}",
                observation=result.error
            )
            return self._build_empty_result(time.time() - start_time, result.error)
        
        log(f"[Agent] Got response: {len(result.text)} chars")
        
        # Step 3: Parse response
        fields, confidence = self._parse_response(result.text)
        log(f"[Agent] Parsed {len(fields)} fields")
        
        self._add_step(
            AgentState.EXTRACTING.value,
            f"Parsed {len(fields)} fields from response",
            observation=result.text[:500] if result.text else "No response"
        )
        
        # Step 4: Validate
        self._add_step(
            AgentState.VALIDATING.value,
            "Validating extracted fields"
        )
        
        validation = self.validator.validate_document(fields)
        self.trace.tool_calls += 1
        
        # Update confidence for invalid fields
        for field_name, val_result in validation.field_results.items():
            if not val_result.is_valid and field_name in confidence:
                if confidence[field_name] == "HIGH":
                    confidence[field_name] = "MEDIUM"
        
        # Calculate quality
        total = len(fields)
        non_empty = sum(1 for v in fields.values() if v != "---")
        high_conf = sum(1 for c in confidence.values() if c == "HIGH")
        
        if total > 0:
            quality_score = int((non_empty / total) * 50 + (high_conf / total) * 50)
        else:
            quality_score = 0
        
        self.trace.quality_score = quality_score
        self.trace.total_time = time.time() - start_time
        
        self._add_step(
            AgentState.COMPLETE.value,
            f"Complete: {non_empty}/{total} fields extracted, {quality_score}% quality"
        )
        
        log(f"[Agent] Done! {non_empty}/{total} fields, {quality_score}% quality in {self.trace.total_time:.1f}s")
        
        return self._build_result(fields, confidence, validation)
    
    def _normalize_value(self, field_name: str, value: str) -> str:
        """Apply minimal Saudi format rules - only for phone numbers."""
        if not value or value == "---":
            return value
        
        # Normalize phone numbers only - must be 10 digits starting with 05
        # This is the ONE normalization that consistently works
        if "جوال" in field_name or "هاتف" in field_name:
            digits = ''.join(c for c in value if c in '٠١٢٣٤٥٦٧٨٩0123456789')
            digits = digits.translate(str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩'))
            if len(digits) == 8 and not digits.startswith('٠٥'):
                digits = '٠٥٠' + digits
            elif len(digits) == 9 and digits.startswith('٥'):
                digits = '٠' + digits
            return digits if digits else value
        
        # NO other auto-corrections - let the model's raw output through
        # Auto-corrections were masking errors and adding noise
        return value
    
    def _parse_response(self, text: str) -> tuple:
        """Parse the extraction response into fields and confidence."""
        fields = {}
        confidence = {}
        
        # Known valid field names (to filter out invented fields)
        VALID_FIELDS = {
            'اسم_المالك', 'رقم_الهوية', 'مصدرها', 'تاريخها', 'تاريخ_الميلاد',
            'الحالة_الاجتماعية', 'عدد_من_يعولهم', 'المؤهل', 'المدينة', 'الحي',
            'الشارع', 'رقم_المبنى', 'جوال', 'البريد_الإلكتروني', 'فاكس',
            'نوع_النشاط', 'مدينة_مزاولة_النشاط', 'تاريخ_بدء_النشاط',
            'تاريخ_انتهاء_الترخيص', 'رقم_رخصة_القيادة', 'تاريخ_إصدار_الرخصة',
            'تاريخ_انتهاء_الرخصة', 'مصدر_الرخصة', 'نوع_المركبة', 'الموديل',
            'اللون', 'رقم_اللوحة', 'سنة_الصنع', 'اسم_مقدم_الطلب', 'صفته',
            'توقيعه', 'تاريخ_التوقيع'
        }
        
        if not text:
            return fields, confidence
        
        lines = text.strip().split("\n")
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines and headers
            if not line or line.startswith("#") or line.startswith("##"):
                continue
            
            # Must have a colon
            if ":" not in line:
                continue
            
            # Split on first colon
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            
            field_name = parts[0].strip()
            value_part = parts[1].strip()
            
            # Skip if field name looks like instruction
            if len(field_name) > 40 or "مثال" in field_name or "تنسيق" in field_name:
                continue
            
            # Filter out invented fields
            if field_name not in VALID_FIELDS:
                continue
            
            # Extract confidence marker
            conf = "MEDIUM"
            if "[HIGH]" in value_part.upper():
                conf = "HIGH"
                value_part = value_part.replace("[HIGH]", "").replace("[high]", "").strip()
            elif "[MEDIUM]" in value_part.upper():
                conf = "MEDIUM"
                value_part = value_part.replace("[MEDIUM]", "").replace("[medium]", "").strip()
            elif "[LOW]" in value_part.upper():
                conf = "LOW"
                value_part = value_part.replace("[LOW]", "").replace("[low]", "").strip()
            
            value = value_part.strip()
            
            # Normalize empty markers
            if value in ["---", "[فارغ]", "[EMPTY]", "فارغ", "غير موجود", "", "؟؟؟", "؟؟"]:
                value = "---"
                conf = "LOW"
            
            # Values with ? are partial reads - MEDIUM confidence
            if "؟" in value and conf == "HIGH":
                conf = "MEDIUM"
            
            # Apply Saudi format normalization
            value = self._normalize_value(field_name, value)
            
            # Store
            if field_name:
                fields[field_name] = value
                confidence[field_name] = conf
        
        return fields, confidence
    
    def _build_result(self, fields: Dict, confidence: Dict, validation) -> Dict[str, Any]:
        """Build the final result."""
        field_results = []
        fields_needing_review = []
        confidence_summary = {"high": 0, "medium": 0, "low": 0, "empty": 0}
        
        for field_name, value in fields.items():
            conf = confidence.get(field_name, "MEDIUM")
            is_empty = value == "---"
            
            if is_empty:
                conf = "LOW"
                confidence_summary["empty"] += 1
            else:
                confidence_summary[conf.lower()] += 1
            
            needs_review = conf == "LOW" or "؟" in value
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
        
        # Quality status
        quality_score = self.trace.quality_score
        if quality_score < 20:
            quality_status = "rejected"
        elif quality_score < 40:
            quality_status = "failed"
        elif quality_score < 60:
            quality_status = "warning"
        else:
            quality_status = "passed"
        
        return {
            "fields": field_results,
            "raw_text": "\n".join(f"{k}: {v}" for k, v in fields.items()),
            "iterations_used": 1,
            "processing_time_seconds": self.trace.total_time,
            "confidence_summary": confidence_summary,
            "fields_needing_review": fields_needing_review,
            "status": "success" if field_results else "failed",
            "quality_score": quality_score,
            "quality_status": quality_status,
            "agent_trace": self.trace.to_dict(),
            "tool_calls": self.trace.tool_calls,
            "hallucination_detected": len(validation.hallucination_indicators) > 0,
            "hallucination_indicators": validation.hallucination_indicators,
        }
    
    def _build_empty_result(self, elapsed: float, error: str) -> Dict[str, Any]:
        """Build result for failed extraction."""
        self.trace.total_time = elapsed
        self.trace.quality_score = 0
        
        return {
            "fields": [],
            "raw_text": "",
            "iterations_used": 1,
            "processing_time_seconds": elapsed,
            "confidence_summary": {"high": 0, "medium": 0, "low": 0, "empty": 0},
            "fields_needing_review": [],
            "status": "failed",
            "quality_score": 0,
            "quality_status": "failed",
            "agent_trace": self.trace.to_dict(),
            "tool_calls": self.trace.tool_calls,
            "hallucination_detected": False,
            "hallucination_indicators": [],
            "error": error,
        }


def create_agent(azure_client: Optional[AzureVisionOCR] = None, on_step: Optional[Callable] = None) -> AgenticOCRAgent:
    """Create an OCR agent."""
    return AgenticOCRAgent(azure_client=azure_client, on_step=on_step)
