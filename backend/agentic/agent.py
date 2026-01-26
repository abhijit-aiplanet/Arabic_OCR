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


# Comprehensive Arabic OCR extraction prompt
# Works for forms, handwritten text, documents, and mixed content
EXTRACTION_PROMPT = """أنت نظام OCR متخصص في قراءة المستندات العربية. اقرأ الصورة بعناية فائقة واستخرج كل النص والبيانات.

## التعليمات:

1. اقرأ الصورة من اليمين إلى اليسار
2. استخرج كل حقل تراه مع قيمته
3. إذا كان النص مكتوب بخط اليد، اقرأه بدقة حرف بحرف
4. لا تتجاهل أي نص مرئي

## الحقول الشائعة في النماذج السعودية (استخرجها إن وجدت):

### بيانات شخصية:
اسم_المالك، الاسم_الكامل، اسم_الأب، اسم_الجد، اللقب، الجنسية، الديانة، المهنة

### بيانات الهوية:
رقم_الهوية، رقم_السجل_المدني، رقم_الإقامة، رقم_الجواز، مصدر_الهوية، تاريخ_الإصدار، تاريخ_الانتهاء

### بيانات الميلاد:
تاريخ_الميلاد، مكان_الميلاد، العمر

### الحالة الاجتماعية:
الحالة_الاجتماعية، عدد_الأبناء، عدد_من_يعولهم

### العنوان:
المنطقة، المدينة، الحي، الشارع، رقم_المبنى، الرمز_البريدي، صندوق_البريد

### الاتصال:
جوال، هاتف، فاكس، البريد_الإلكتروني

### العمل والتعليم:
المؤهل، التخصص، جهة_العمل، المسمى_الوظيفي، سنوات_الخبرة

### رخصة القيادة:
رقم_رخصة_القيادة، نوع_الرخصة، تاريخ_إصدار_الرخصة، تاريخ_انتهاء_الرخصة، مصدر_الرخصة

### بيانات المركبة:
نوع_المركبة، الماركة، الموديل، اللون، رقم_اللوحة، سنة_الصنع، رقم_الشاسيه

### بيانات مالية:
رقم_الحساب، اسم_البنك، رقم_الآيبان، المبلغ

### التوقيع:
اسم_مقدم_الطلب، صفة_مقدم_الطلب، التوقيع، تاريخ_التوقيع، الختم

## للنصوص المكتوبة بخط اليد:
- اقرأ كل سطر على حدة
- إذا كان النص غير واضح، حاول قراءته واكتب [غير واضح] بجانبه
- استخرج كل كلمة تستطيع قراءتها

## صيغة الإخراج:
اسم_الحقل: القيمة [HIGH/MEDIUM/LOW]

مثال:
الاسم_الكامل: محمد أحمد العلي [HIGH]
رقم_الهوية: ١٠٥٠٢٣٤٥٦٧ [HIGH]
جوال: ٠٥٠١٢٣٤٥٦٧ [MEDIUM]
التوقيع: [توقيع موجود] [HIGH]
الحقل_الفارغ: --- [HIGH]

## ملاحظات هامة:
- إذا الحقل فارغ اكتب: ---
- إذا يوجد توقيع اكتب: [توقيع موجود]
- إذا يوجد ختم اكتب: [ختم موجود]
- إذا يوجد صورة شخصية اكتب: [صورة شخصية موجودة]
- اكتب الأرقام كما هي (عربية أو إنجليزية)
- استخرج حتى الحقول التي ليست في القائمة أعلاه

## ابدأ الاستخراج الآن - اقرأ كل شيء في الصورة:"""


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
    
    def _normalize_and_validate(self, field_name: str, value: str, confidence: str) -> tuple:
        """Validate values and downgrade confidence if suspicious."""
        if not value or value == "---":
            return value, "LOW"
        
        # Phone numbers - must be 10 digits starting with 05
        if "جوال" in field_name or "هاتف" in field_name:
            digits = ''.join(c for c in value if c in '٠١٢٣٤٥٦٧٨٩0123456789')
            digits = digits.translate(str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩'))
            # Add 05 prefix if missing
            if len(digits) == 8:
                digits = '٠٥٠' + digits
            elif len(digits) == 9 and digits.startswith('٥'):
                digits = '٠' + digits
            # Validate: must be exactly 10 digits starting with 05
            if len(digits) != 10 or not digits.startswith('٠٥'):
                return digits if digits else value, "LOW"  # Flag suspicious
            return digits, confidence
        
        # ID numbers - must be 10 digits starting with 1
        if "هوية" in field_name or field_name == "رقم_الهوية":
            digits = ''.join(c for c in value if c in '٠١٢٣٤٥٦٧٨٩0123456789')
            digits = digits.translate(str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩'))
            # Saudi IDs are 10 digits starting with 1
            if len(digits) != 10 or not digits.startswith('١'):
                return value, "LOW"  # Flag as suspicious
            return digits, confidence
        
        # License numbers - similar to ID
        if "رخصة" in field_name and "رقم" in field_name:
            digits = ''.join(c for c in value if c in '٠١٢٣٤٥٦٧٨٩0123456789')
            if len(digits) != 10:
                return value, "LOW"  # Flag as suspicious
            return value, confidence
        
        # Education - only allow known values
        if field_name == "المؤهل":
            known = ['أمي', 'ابتدائي', 'متوسط', 'ثانوي', 'الثانوي', 'جامعي', 'دبلوم']
            if value not in known:
                return value, "LOW"  # Unknown education level
            return value, confidence
        
        # Neighborhood - if filled, should be verified
        if field_name == "الحي":
            # Common Jeddah neighborhoods for validation
            known_neighborhoods = ['الروضة', 'الحمراء', 'النعيم', 'الصفا', 'البوادي', 'الفيصلية', 'المحمدية', 'الهدى']
            if value not in known_neighborhoods and value != "---":
                return value, "MEDIUM"  # Not in known list, needs review
            return value, confidence
        
        # Activity type - high risk field
        if field_name == "نوع_النشاط":
            # If activity is filled, mark as needing review (often hallucinated)
            return value, "MEDIUM"
        
        return value, confidence
    
    def _parse_response(self, text: str) -> tuple:
        """Parse the extraction response into fields and confidence."""
        fields = {}
        confidence = {}
        
        # Words that indicate the line is an instruction, not a field
        SKIP_WORDS = {
            'مثال', 'تنسيق', 'التعليمات', 'الخطوة', 'ملاحظة', 'ملاحظات',
            'القاعدة', 'هام', 'تحذير', 'STEP', 'NOTE', 'EXAMPLE', 'FORMAT',
            'BEGIN', 'START', 'ابدأ', 'استخرج', 'اقرأ', 'نوع_المستند',
            'document_type', 'للنصوص', 'للنماذج', 'للجداول'
        }
        
        if not text:
            return fields, confidence
        
        lines = text.strip().split("\n")
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines and headers
            if not line or line.startswith("#") or line.startswith("##") or line.startswith("*"):
                continue
            
            # Skip numbered lists that look like instructions
            if line.startswith("1.") or line.startswith("2.") or line.startswith("3."):
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
            
            # Skip if field name is too long (likely instruction text)
            if len(field_name) > 50:
                continue
            
            # Skip if field name contains instruction words
            if any(skip_word in field_name for skip_word in SKIP_WORDS):
                continue
            
            # Skip common non-field patterns
            if field_name.startswith('-') or field_name.startswith('•'):
                continue
            
            # Clean up field name - remove bullets, numbers, spaces
            field_name = field_name.lstrip('-•·').strip()
            field_name = field_name.replace(' ', '_').replace('\t', '_')
            
            # Skip if field name is empty after cleanup
            if not field_name or len(field_name) < 2:
                continue
            
            # Extract confidence marker
            conf = "MEDIUM"
            value_upper = value_part.upper()
            if "[HIGH]" in value_upper:
                conf = "HIGH"
                value_part = value_part.replace("[HIGH]", "").replace("[high]", "").replace("[High]", "").strip()
            elif "[MEDIUM]" in value_upper:
                conf = "MEDIUM"
                value_part = value_part.replace("[MEDIUM]", "").replace("[medium]", "").replace("[Medium]", "").strip()
            elif "[LOW]" in value_upper:
                conf = "LOW"
                value_part = value_part.replace("[LOW]", "").replace("[low]", "").replace("[Low]", "").strip()
            
            value = value_part.strip()
            
            # Normalize empty markers
            if value.lower() in ["---", "[فارغ]", "[empty]", "فارغ", "غير موجود", "", "؟؟؟", "؟؟", "n/a", "na", "-", "--"]:
                value = "---"
                conf = "LOW"
            
            # Values with ? are partial reads - MEDIUM confidence
            if "؟" in value and conf == "HIGH":
                conf = "MEDIUM"
            
            # Apply validation and potentially downgrade confidence
            value, conf = self._normalize_and_validate(field_name, value, conf)
            
            # Store
            if field_name and value:
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
