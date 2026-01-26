"""
Agentic OCR Agent with Multi-Pass Extraction

Features:
- Document type detection (form, handwritten text, mixed)
- Multi-pass extraction with self-review
- Reasoning model optimized (high max_tokens for o4-mini)
- Minimal post-processing - let the model do the heavy lifting
"""

import time
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from PIL import Image
from enum import Enum
import base64
import io
import re

from .azure_client import AzureVisionOCR
from .format_validator import FormatValidator


def log(msg: str):
    """Force-flush logging."""
    print(msg)
    sys.stdout.flush()


class AgentState(Enum):
    STARTING = "starting"
    DETECTING = "detecting_type"
    EXTRACTING = "extracting"
    REVIEWING = "reviewing"
    REFINING = "refining"
    COMPLETE = "complete"


class DocumentType(Enum):
    FORM = "form"
    HANDWRITTEN = "handwritten"
    TABLE = "table"
    MIXED = "mixed"
    UNKNOWN = "unknown"


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
    iterations: int = 1
    
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
            "iterations": self.iterations,
            "quality_score": self.quality_score,
        }


# =============================================================================
# PROMPTS - Document Type Specific, No Phantom Fields
# =============================================================================

DETECT_TYPE_PROMPT = """انظر إلى هذه الصورة وحدد نوعها في كلمة واحدة فقط:

- form: إذا كانت نموذج رسمي مع حقول وقيم (مثل نموذج حكومي أو استمارة)
- handwritten: إذا كانت نص مكتوب بخط اليد فقط (رسالة، ملاحظات، شعر)
- table: إذا كانت جدول بصفوف وأعمدة
- mixed: إذا كانت مزيج من النصوص والنماذج

أجب بكلمة واحدة فقط: form أو handwritten أو table أو mixed"""


FORM_EXTRACTION_PROMPT = """أنت نظام OCR متخصص. اقرأ هذا النموذج واستخرج كل حقل مكتوب فيه قيمة.

المهمة:
1. انظر إلى كل حقل في النموذج
2. اقرأ اسم الحقل (العنوان المطبوع)
3. اقرأ القيمة المكتوبة بخط اليد بجانبه
4. اكتب كل حقل وقيمته

التنسيق:
اسم_الحقل: القيمة

قواعد مهمة:
- اكتب فقط الحقول التي تراها في الصورة
- لا تضف حقول غير موجودة
- اقرأ كل شيء من أعلى الصفحة إلى أسفلها
- إذا لم تستطع قراءة قيمة بوضوح، اكتب أفضل تخمين مع علامة استفهام: قيمة؟

ابدأ الاستخراج:"""


HANDWRITTEN_EXTRACTION_PROMPT = """أنت نظام OCR متخصص في قراءة النصوص المكتوبة بخط اليد.

المهمة:
اقرأ كل النص المكتوب في هذه الصورة، سطر بسطر.

التنسيق:
- اكتب كل سطر في سطر منفصل
- إذا كان هناك ترقيم أو نقاط، حافظ عليها
- إذا كلمة غير واضحة، اكتبها مع علامة استفهام: كلمة؟

اقرأ كل ما تراه:"""


TABLE_EXTRACTION_PROMPT = """أنت نظام OCR متخصص في قراءة الجداول.

المهمة:
1. اقرأ عناوين الأعمدة
2. اقرأ كل صف
3. اكتب البيانات بتنسيق واضح

التنسيق:
العمود1 | العمود2 | العمود3
القيمة1 | القيمة2 | القيمة3

اقرأ الجدول:"""


MIXED_EXTRACTION_PROMPT = """أنت نظام OCR متخصص. اقرأ كل ما هو مكتوب في هذه الصورة.

المهمة:
1. إذا وجدت نموذج مع حقول: اكتب اسم_الحقل: القيمة
2. إذا وجدت نص عادي: اكتبه كما هو سطر بسطر
3. إذا وجدت جدول: اكتب البيانات بتنسيق واضح

قواعد:
- اكتب فقط ما تراه في الصورة
- لا تضف أي شيء غير موجود
- إذا كلمة غير واضحة: اكتبها مع علامة استفهام

اقرأ كل شيء:"""


SELF_REVIEW_PROMPT = """راجع هذا الاستخراج وأجب عن الأسئلة:

## النص المستخرج:
{extracted_text}

## أسئلة المراجعة:
1. هل فاتني أي نص مرئي في الصورة؟
2. هل هناك أخطاء واضحة في القراءة؟
3. هل أضفت شيء غير موجود في الصورة؟

## إذا وجدت مشاكل:
اكتب التصحيحات بهذا التنسيق:
- تصحيح: [ما يجب تغييره]
- إضافة: [ما فاتني]
- حذف: [ما أضفته بالخطأ]

## إذا كان الاستخراج صحيح:
اكتب: الاستخراج صحيح

راجع الآن:"""


# =============================================================================
# AGENTIC OCR AGENT
# =============================================================================

class AgenticOCRAgent:
    """
    Multi-pass OCR agent with document type detection and self-review.
    
    Optimized for o4-mini reasoning model:
    - High max_tokens for reasoning
    - Minimal post-processing
    - Let the model do the heavy lifting
    """
    
    def __init__(
        self,
        azure_client: Optional[AzureVisionOCR] = None,
        on_step: Optional[Callable] = None,
        max_iterations: int = 2,
    ):
        self.azure = azure_client or AzureVisionOCR()
        self.validator = FormatValidator()
        self.on_step = on_step
        self.max_iterations = max_iterations
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
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    async def detect_document_type(self, image_b64: str) -> DocumentType:
        """Quick detection of document type before main extraction."""
        log("[Agent] Detecting document type...")
        
        result = await self.azure.extract(
            image_base64=image_b64,
            prompt=DETECT_TYPE_PROMPT,
            max_tokens=50,  # Very short response expected
        )
        
        if not result.success:
            log(f"[Agent] Type detection failed: {result.error}")
            return DocumentType.MIXED  # Default to mixed
        
        response = result.text.strip().lower()
        log(f"[Agent] Detected type: {response}")
        
        if "form" in response:
            return DocumentType.FORM
        elif "handwritten" in response or "hand" in response:
            return DocumentType.HANDWRITTEN
        elif "table" in response:
            return DocumentType.TABLE
        else:
            return DocumentType.MIXED
    
    def _get_prompt_for_type(self, doc_type: DocumentType) -> str:
        """Get the appropriate extraction prompt for document type."""
        prompts = {
            DocumentType.FORM: FORM_EXTRACTION_PROMPT,
            DocumentType.HANDWRITTEN: HANDWRITTEN_EXTRACTION_PROMPT,
            DocumentType.TABLE: TABLE_EXTRACTION_PROMPT,
            DocumentType.MIXED: MIXED_EXTRACTION_PROMPT,
            DocumentType.UNKNOWN: MIXED_EXTRACTION_PROMPT,
        }
        return prompts.get(doc_type, MIXED_EXTRACTION_PROMPT)
    
    async def process(self, image: Image.Image, user_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Multi-pass extraction process:
        1. Detect document type
        2. Extract with type-specific prompt
        3. Self-review and refine if needed
        """
        start_time = time.time()
        self.trace = AgentTrace()
        
        log("[Agent] Starting multi-pass extraction...")
        
        # Step 1: Prepare
        self._add_step(
            AgentState.STARTING.value,
            f"Processing image: {image.size[0]}x{image.size[1]} pixels"
        )
        
        image_b64 = self._image_to_base64(image)
        
        # Step 2: Detect document type
        self._add_step(
            AgentState.DETECTING.value,
            "Analyzing document to determine type"
        )
        
        doc_type = await self.detect_document_type(image_b64)
        self.trace.tool_calls += 1
        
        self._add_step(
            AgentState.DETECTING.value,
            f"Document type: {doc_type.value}",
            observation=f"Detected as {doc_type.value}"
        )
        
        # Step 3: Initial extraction with type-specific prompt
        prompt = self._get_prompt_for_type(doc_type)
        
        self._add_step(
            AgentState.EXTRACTING.value,
            f"Extracting content using {doc_type.value} prompt",
            action="extract"
        )
        
        log(f"[Agent] Pass 1: Extracting with {doc_type.value} prompt...")
        result = await self.azure.extract_section(image_b64, prompt)
        self.trace.tool_calls += 1
        
        if not result.success:
            log(f"[Agent] EXTRACTION FAILED: {result.error}")
            return self._build_empty_result(time.time() - start_time, result.error)
        
        raw_output = result.text
        log(f"[Agent] Pass 1 complete: {len(raw_output)} chars")
        log(f"[Agent] Raw output preview:\n{raw_output[:800]}")
        
        self._add_step(
            AgentState.EXTRACTING.value,
            f"Initial extraction complete: {len(raw_output)} characters",
            observation=raw_output[:500]
        )
        
        # Step 4: Self-review pass
        if self.max_iterations >= 2:
            self._add_step(
                AgentState.REVIEWING.value,
                "Self-reviewing extraction for completeness and accuracy"
            )
            
            log("[Agent] Pass 2: Self-review...")
            review_prompt = SELF_REVIEW_PROMPT.format(extracted_text=raw_output[:2000])
            
            review_result = await self.azure.extract(
                image_base64=image_b64,
                prompt=review_prompt,
                max_tokens=8000,
            )
            self.trace.tool_calls += 1
            self.trace.iterations = 2
            
            if review_result.success:
                review_text = review_result.text
                log(f"[Agent] Review result:\n{review_text[:500]}")
                
                # Check if review found issues
                if "الاستخراج صحيح" not in review_text and "صحيح" not in review_text.lower():
                    # There were corrections - append them
                    self._add_step(
                        AgentState.REFINING.value,
                        "Applying corrections from self-review",
                        observation=review_text[:300]
                    )
                    
                    # Merge corrections into raw output
                    if "تصحيح:" in review_text or "إضافة:" in review_text:
                        raw_output = raw_output + "\n\n## تصحيحات:\n" + review_text
                else:
                    self._add_step(
                        AgentState.REVIEWING.value,
                        "Self-review confirmed extraction is complete",
                        observation="No corrections needed"
                    )
        
        # Step 5: Parse the output (minimal processing)
        fields, confidence = self._parse_response(raw_output, doc_type)
        log(f"[Agent] Parsed {len(fields)} fields")
        
        # Step 6: Light validation (no modifications, just confidence adjustment)
        validation = self.validator.validate_document(fields)
        
        # Calculate quality score
        total = max(len(fields), 1)
        non_empty = sum(1 for v in fields.values() if v and v != "---")
        high_conf = sum(1 for c in confidence.values() if c == "HIGH")
        
        quality_score = int((non_empty / total) * 60 + (high_conf / total) * 40)
        self.trace.quality_score = quality_score
        self.trace.total_time = time.time() - start_time
        
        self._add_step(
            AgentState.COMPLETE.value,
            f"Complete: {len(fields)} items extracted, {quality_score}% quality in {self.trace.total_time:.1f}s"
        )
        
        log(f"[Agent] Done! {len(fields)} items, {quality_score}% quality in {self.trace.total_time:.1f}s")
        
        return self._build_result(fields, confidence, validation, raw_output, doc_type)
    
    def _parse_response(self, text: str, doc_type: DocumentType) -> tuple:
        """
        Parse extraction response with minimal filtering.
        Different parsing strategy based on document type.
        """
        fields = {}
        confidence = {}
        
        if not text:
            return fields, confidence
        
        lines = text.strip().split("\n")
        
        if doc_type == DocumentType.HANDWRITTEN:
            # For handwritten text, keep lines as-is
            line_num = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Skip review/correction sections
                if line.startswith("#") or "تصحيح" in line or "إضافة" in line:
                    continue
                
                line_num += 1
                field_name = f"سطر_{line_num}"
                
                # Check for uncertainty markers
                conf = "HIGH"
                if "؟" in line:
                    conf = "MEDIUM"
                
                fields[field_name] = line
                confidence[field_name] = conf
        
        else:
            # For forms and other types, look for field:value pattern
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and headers
                if not line or line.startswith("#"):
                    continue
                
                # Skip correction sections
                if "تصحيح:" in line or "إضافة:" in line or "حذف:" in line:
                    continue
                
                # Try to parse as field:value
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        field_name = parts[0].strip()
                        value = parts[1].strip()
                        
                        # Skip if looks like instruction
                        if len(field_name) > 60:
                            continue
                        
                        # Clean up field name
                        field_name = field_name.lstrip("-•·0123456789.").strip()
                        field_name = field_name.replace(" ", "_")
                        
                        if not field_name or len(field_name) < 2:
                            continue
                        
                        # Determine confidence
                        conf = "HIGH"
                        if "؟" in value:
                            conf = "MEDIUM"
                        if not value or value == "---":
                            conf = "LOW"
                        
                        fields[field_name] = value
                        confidence[field_name] = conf
                
                elif doc_type == DocumentType.MIXED and line:
                    # For mixed, also capture standalone lines
                    if len(line) > 5 and not any(skip in line for skip in ["المهمة", "التنسيق", "قواعد"]):
                        line_key = f"نص_{len(fields) + 1}"
                        fields[line_key] = line
                        confidence[line_key] = "MEDIUM"
        
        return fields, confidence
    
    def _build_result(self, fields: Dict, confidence: Dict, validation, raw_output: str, doc_type: DocumentType) -> Dict[str, Any]:
        """Build the final result with actual raw output."""
        field_results = []
        confidence_summary = {"high": 0, "medium": 0, "low": 0, "empty": 0}
        
        for field_name, value in fields.items():
            conf = confidence.get(field_name, "MEDIUM")
            is_empty = not value or value == "---"
            
            if is_empty:
                confidence_summary["empty"] += 1
            else:
                confidence_summary[conf.lower()] += 1
            
            needs_review = conf == "LOW" or "؟" in str(value)
            
            field_results.append({
                "field_name": field_name,
                "value": value,
                "confidence": conf.lower(),
                "needs_review": needs_review,
                "review_reason": "Needs verification" if needs_review else None,
                "is_empty": is_empty,
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
            "raw_text": raw_output,  # ACTUAL model output, not rebuilt
            "document_type": doc_type.value,
            "iterations_used": self.trace.iterations,
            "processing_time_seconds": self.trace.total_time,
            "confidence_summary": confidence_summary,
            "fields_needing_review": [f["field_name"] for f in field_results if f["needs_review"]],
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
            "document_type": "unknown",
            "iterations_used": self.trace.iterations,
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
