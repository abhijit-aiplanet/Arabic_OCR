"""
Simple Agentic OCR Agent

PRINCIPLE: Simplicity over complexity. Functionality first.

This agent sends the FULL image to the vision model with a clear prompt,
then parses the response. No over-engineered section detection or zoom.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from PIL import Image
from enum import Enum
import base64
import io

from .azure_client import AzureVisionOCR
from .format_validator import FormatValidator


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


# The ONE prompt that works - clear, direct, comprehensive
EXTRACTION_PROMPT = """أنت خبير في قراءة النماذج الحكومية السعودية المكتوبة بخط اليد.

اقرأ هذه الصورة واستخرج كل الحقول المكتوبة.

## الحقول المطلوبة:

### بيانات عامة:
- اسم المالك
- رقم الهوية (١٠ أرقام)
- مصدرها
- تاريخها
- تاريخ الميلاد
- الحالة الاجتماعية
- عدد من يعولهم
- المؤهل

### العنوان:
- المدينة
- الحي
- الشارع
- رقم المبنى
- جوال (١٠ أرقام تبدأ بـ ٠٥)
- البريد الإلكتروني
- فاكس

### بيانات النشاط:
- نوع النشاط
- مدينة مزاولة النشاط
- تاريخ بدء النشاط
- تاريخ انتهاء الترخيص

### رخصة القيادة:
- رقمها
- تاريخ الإصدار
- تاريخ الانتهاء
- مصدرها

### بيانات المركبة:
- نوع المركبة
- الموديل
- اللون
- رقم اللوحة
- سنة الصنع

### التوقيع:
- اسم مقدم الطلب
- صفته
- توقيعه
- التاريخ

## تعليمات مهمة:
1. اقرأ كل كتابة يدوية تراها في الصورة
2. إذا رأيت قيمة لكنها غير واضحة، اكتبها مع علامة ؟
3. إذا كان الحقل فارغاً تماماً (لا توجد كتابة): اكتب ---
4. لا تخترع قيماً - اكتب فقط ما تراه

## مستويات الثقة:
- [HIGH] = قراءة واضحة ومؤكدة
- [MEDIUM] = قراءة محتملة
- [LOW] = تخمين أو فارغ

## التنسيق المطلوب:
اسم_الحقل: القيمة [مستوى_الثقة]

مثال:
المدينة: جدة [HIGH]
جوال: ٠٥٠٧٤٧٧٩٩٨ [HIGH]
اسم المالك: عبدالله محمد [MEDIUM]
رقم اللوحة: --- [LOW]

ابدأ الاستخراج الآن:"""


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
        
        print("[Agent] Starting simple extraction...")
        
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
        result = await self.azure.extract_section(image_b64, EXTRACTION_PROMPT)
        self.trace.tool_calls += 1
        
        if not result.success:
            self._add_step(
                AgentState.COMPLETE.value,
                f"Extraction failed: {result.error}",
                observation=result.error
            )
            return self._build_empty_result(time.time() - start_time, result.error)
        
        # DEBUG: Log raw response
        print(f"[Agent] Raw response length: {len(result.text) if result.text else 0}")
        print(f"[Agent] Raw response preview: {result.text[:1000] if result.text else 'EMPTY'}")
        
        # Step 3: Parse response
        fields, confidence = self._parse_response(result.text)
        print(f"[Agent] Parsed fields: {list(fields.keys())}")
        
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
        
        print(f"[Agent] Done! {non_empty}/{total} fields, {quality_score}% quality in {self.trace.total_time:.1f}s")
        
        return self._build_result(fields, confidence, validation)
    
    def _parse_response(self, text: str) -> tuple:
        """Parse the extraction response into fields and confidence."""
        fields = {}
        confidence = {}
        
        if not text:
            print("[Parser] Empty text!")
            return fields, confidence
        
        lines = text.strip().split("\n")
        print(f"[Parser] Total lines: {len(lines)}")
        
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
                print(f"[Parser] Skipping instruction line: {field_name[:30]}...")
                continue
            
            print(f"[Parser] Found field: {field_name} = {value_part[:30] if value_part else 'empty'}...")
            
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
            if value in ["---", "[فارغ]", "[EMPTY]", "فارغ", "غير موجود", ""]:
                value = "---"
                conf = "LOW"
            
            # Values with ? are partial reads - MEDIUM confidence
            if "؟" in value and conf == "HIGH":
                conf = "MEDIUM"
            
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
