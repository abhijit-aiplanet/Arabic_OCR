"""
Simple Agentic OCR Agent

SIMPLIFIED APPROACH:
1. Send FULL image to Azure (no cropping)
2. Use ONE comprehensive prompt
3. Parse results
4. Validate
5. Done

Previous over-engineering problems:
- Section cropping with fixed coordinates (wrong for different forms)
- Multiple API calls (slow, loses context)
- Complex pipelines that obscure failures

This version: Simple, transparent, effective.
"""

import time
import base64
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from PIL import Image, ImageEnhance
from enum import Enum

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
                    "observation": s.observation[:500] if s.observation else None,
                    "confidence": s.confidence,
                }
                for s in self.steps
            ],
            "total_time": self.total_time,
            "tool_calls": self.tool_calls,
            "iterations": 1,
            "quality_score": self.quality_score,
        }


# =============================================================================
# THE SINGLE COMPREHENSIVE PROMPT
# =============================================================================

FULL_EXTRACTION_PROMPT = """أنت خبير في قراءة النماذج الحكومية السعودية.

اقرأ هذه الصورة واستخرج كل الحقول المكتوبة.

## الحقول المتوقعة في النموذج السعودي:

### بيانات النشاط (أعلى الصفحة):
- نوع النشاط
- مدينة مزاولة النشاط
- تاريخ بدء النشاط
- تاريخ انتهاء الترخيص

### بيانات عامة (البيانات الشخصية):
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
- رقم الهيكل
- سنة الصنع

### التوقيع (أسفل الصفحة):
- اسم مقدم الطلب
- صفته
- التاريخ
- توقيعه

## تعليمات مهمة جداً:

1. اقرأ كل ما تراه مكتوباً بخط اليد
2. إذا رأيت كتابة لكنها غير واضحة تماماً، اكتبها مع [MEDIUM]
3. إذا كان الحقل فارغاً تماماً (لا توجد كتابة)، اكتب: ---
4. للأرقام غير الواضحة، استخدم ؟ للرقم المشكوك فيه: ٠٥٠٧؟٧٧٩٩٨

## مستويات الثقة:
- [HIGH] = واضح تماماً، متأكد ١٠٠٪
- [MEDIUM] = مقروء لكن قد يكون خطأ
- [LOW] = تخمين أو فارغ

## التنسيق المطلوب:
اسم_الحقل: القيمة [HIGH/MEDIUM/LOW]

مثال:
المدينة: جدة [HIGH]
جوال: ٠٥٠٧٤٧٧٩٩٨ [MEDIUM]
البريد الإلكتروني: --- [LOW]

## ابدأ الاستخراج الآن:
اقرأ الصورة من الأعلى إلى الأسفل واكتب كل الحقول:"""


class AgenticOCRAgent:
    """Simple agent: Full image → Single prompt → Parse → Validate."""
    
    def __init__(
        self,
        azure_client: Optional[AzureVisionOCR] = None,
        on_step: Optional[Callable] = None,
    ):
        self.azure = azure_client or AzureVisionOCR()
        self.validator = FormatValidator()
        self.on_step = on_step
        self.trace = AgentTrace()
    
    def _add_step(self, state: str, thought: str, **kwargs):
        step = self.trace.add_step(state=state, thought=thought, **kwargs)
        if self.on_step:
            self.on_step(step)
        return step
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64."""
        # Enhance contrast for better OCR
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(1.5)
        
        buffer = io.BytesIO()
        enhanced.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    async def process(self, image: Image.Image, user_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Process image with simple approach:
        1. Send full image
        2. Parse response
        3. Validate
        """
        start_time = time.time()
        self.trace = AgentTrace()
        
        print(f"[Agent] Starting simple OCR on {image.size[0]}x{image.size[1]} image")
        
        # Step 1: Start
        self._add_step(
            AgentState.STARTING.value,
            f"Processing {image.size[0]}x{image.size[1]} image"
        )
        
        # Step 2: Extract with single API call
        self._add_step(
            AgentState.EXTRACTING.value,
            "Sending full image to Azure Vision for extraction",
            action="extract_full_image"
        )
        
        image_b64 = self._image_to_base64(image)
        result = await self.azure.extract_section(image_b64, FULL_EXTRACTION_PROMPT)
        self.trace.tool_calls = 1
        
        raw_output = result.text if result.success else ""
        
        self._add_step(
            AgentState.EXTRACTING.value,
            f"Received response: {len(raw_output)} characters",
            observation=raw_output[:500] if raw_output else "No output"
        )
        
        # Step 3: Parse
        fields, confidence = self._parse_extraction(raw_output)
        
        self._add_step(
            AgentState.EXTRACTING.value,
            f"Parsed {len(fields)} fields from response"
        )
        
        # Step 4: Validate
        self._add_step(
            AgentState.VALIDATING.value,
            "Validating extracted fields"
        )
        
        validation = self.validator.validate_document(fields)
        
        # Adjust confidence based on validation
        for field_name, val_result in validation.field_results.items():
            if not val_result.is_valid and field_name in confidence:
                confidence[field_name] = "LOW"
        
        self._add_step(
            AgentState.VALIDATING.value,
            f"Validation score: {validation.overall_score}/100"
        )
        
        # Step 5: Complete
        total_time = time.time() - start_time
        
        # Calculate quality
        total = len(fields)
        high = sum(1 for c in confidence.values() if c == "HIGH")
        medium = sum(1 for c in confidence.values() if c == "MEDIUM")
        
        # Quality = high confidence + half of medium confidence
        quality_score = int(((high + medium * 0.5) / total * 100) if total > 0 else 0)
        quality_score = max(quality_score, validation.overall_score)
        
        self.trace.total_time = total_time
        self.trace.quality_score = quality_score
        
        self._add_step(
            AgentState.COMPLETE.value,
            f"Complete: {len(fields)} fields, {quality_score}% quality in {total_time:.1f}s"
        )
        
        print(f"[Agent] Complete! {len(fields)} fields in {total_time:.1f}s, quality {quality_score}%")
        
        return self._build_result(fields, confidence, validation, total_time)
    
    def _parse_extraction(self, text: str) -> tuple:
        """Parse extraction text into fields and confidence."""
        fields = {}
        confidence = {}
        
        if not text:
            return fields, confidence
        
        for line in text.strip().split("\n"):
            line = line.strip()
            
            # Skip empty lines and headers
            if not line or line.startswith("#") or line.startswith("##"):
                continue
            
            # Must have a colon
            if ":" not in line:
                continue
            
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            
            field_name = parts[0].strip()
            rest = parts[1].strip()
            
            # Skip if field name is too long (likely instruction text)
            if len(field_name) > 40:
                continue
            
            # Extract confidence
            conf = "MEDIUM"  # Default to MEDIUM, not LOW
            for marker in ["[HIGH]", "[high]"]:
                if marker in rest:
                    conf = "HIGH"
                    rest = rest.replace(marker, "").strip()
            for marker in ["[MEDIUM]", "[medium]"]:
                if marker in rest:
                    conf = "MEDIUM"
                    rest = rest.replace(marker, "").strip()
            for marker in ["[LOW]", "[low]"]:
                if marker in rest:
                    conf = "LOW"
                    rest = rest.replace(marker, "").strip()
            
            value = rest.strip()
            
            # Only truly empty = LOW
            if value in ["---", "[فارغ]", "فارغ", "[EMPTY]", ""] or not value:
                conf = "LOW"
                value = "---"
            
            # Values with ؟ are partial reads - keep MEDIUM
            if "؟" in value and conf == "HIGH":
                conf = "MEDIUM"
            
            if field_name:
                fields[field_name] = value
                confidence[field_name] = conf
        
        return fields, confidence
    
    def _build_result(
        self,
        fields: Dict[str, str],
        confidence: Dict[str, str],
        validation,
        total_time: float,
    ) -> Dict[str, Any]:
        """Build the result dictionary."""
        field_results = []
        fields_needing_review = []
        confidence_summary = {"high": 0, "medium": 0, "low": 0, "empty": 0}
        
        for field_name, value in fields.items():
            conf = confidence.get(field_name, "MEDIUM")
            is_empty = value == "---"
            
            if is_empty:
                confidence_summary["empty"] += 1
                conf = "LOW"
            else:
                confidence_summary[conf.lower()] += 1
            
            needs_review = conf == "LOW" or is_empty
            if needs_review:
                fields_needing_review.append(field_name)
            
            field_results.append({
                "field_name": field_name,
                "value": value,
                "confidence": conf.lower(),
                "needs_review": needs_review,
                "review_reason": "Low confidence" if needs_review else None,
                "is_empty": is_empty,
                "validation_score": 100,
            })
        
        quality_score = self.trace.quality_score
        
        if quality_score < 20:
            quality_status = "rejected"
        elif quality_score < 40:
            quality_status = "failed"
        elif len(fields_needing_review) > len(fields) * 0.5:
            quality_status = "warning"
        else:
            quality_status = "passed"
        
        return {
            "fields": field_results,
            "raw_text": "\n".join(f"{k}: {v}" for k, v in fields.items()),
            "iterations_used": 1,
            "processing_time_seconds": total_time,
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


def create_agent(azure_client=None, on_step=None):
    return AgenticOCRAgent(azure_client=azure_client, on_step=on_step)
