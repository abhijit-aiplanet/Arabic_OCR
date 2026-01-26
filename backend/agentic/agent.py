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

DETECT_TYPE_PROMPT = """Look at this image and respond with ONE word only:
- form (if it has labeled fields with values)
- handwritten (if it's handwritten text like notes, letters, poetry)
- table (if it has rows and columns)
- mixed (combination)

Reply with ONE word: form, handwritten, table, or mixed"""


FORM_EXTRACTION_PROMPT = """You are a precise OCR system. Your ONLY job is to read and transcribe what is VISIBLE in this image.

CRITICAL RULES:
1. Output ONLY the text you see - nothing else
2. DO NOT add any explanations, headers, or instructions
3. DO NOT add text that is not in the image
4. Read EVERY field from top to bottom, left to right

FORMAT:
field_name: value

For each field you see:
- Write the printed label, then colon, then the handwritten value
- If a value is unclear, write your best guess

START OUTPUT NOW (no introduction, just the fields):"""


HANDWRITTEN_EXTRACTION_PROMPT = """You are a precise OCR system. Transcribe EXACTLY what is written in this image.

CRITICAL RULES:
1. Output ONLY the handwritten text - nothing else
2. DO NOT add any titles, headers, explanations, or formatting instructions
3. DO NOT add words or lines that are not visible in the image
4. Copy each line EXACTLY as written, preserving the original text
5. Read very carefully - examine each word letter by letter

PROCESS:
- Look at line 1, read each word carefully, write it
- Look at line 2, read each word carefully, write it
- Continue for all lines

OUTPUT FORMAT:
Just the text, line by line. No numbering unless the original has numbers.

START NOW (output only the handwritten text, nothing else):"""


TABLE_EXTRACTION_PROMPT = """You are a precise OCR system. Extract the table data from this image.

CRITICAL RULES:
1. Output ONLY the table content - no explanations
2. DO NOT add text that is not in the image
3. Read each cell carefully

FORMAT:
header1 | header2 | header3
value1 | value2 | value3

START NOW:"""


MIXED_EXTRACTION_PROMPT = """You are a precise OCR system. Transcribe ALL text visible in this image.

CRITICAL RULES:
1. Output ONLY what you see - no explanations, no headers, no instructions
2. DO NOT add any text that is not in the image
3. For forms: write field_name: value
4. For regular text: write line by line exactly as shown
5. Read VERY carefully - check each word

START NOW (just the text, nothing else):"""


SELF_REVIEW_PROMPT = """Look at the image again and compare with this extraction:

{extracted_text}

CHECK:
1. Is anything MISSING from the image that wasn't extracted?
2. Is anything WRONG (misread words)?
3. Is anything EXTRA that doesn't exist in the image?

If there are problems, write corrections:
- FIX: [what to change]
- ADD: [what was missed]
- REMOVE: [what doesn't exist in image]

If extraction is correct, write: CORRECT

Check now:"""


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
                
                # Check if review found issues (English or Arabic confirmation)
                is_correct = any(word in review_text.upper() for word in ["CORRECT", "صحيح"])
                
                if not is_correct and ("FIX:" in review_text or "ADD:" in review_text or "تصحيح:" in review_text or "إضافة:" in review_text):
                    # There were corrections - append them
                    self._add_step(
                        AgentState.REFINING.value,
                        "Applying corrections from self-review",
                        observation=review_text[:300]
                    )
                    
                    # Merge corrections into raw output
                    raw_output = raw_output + "\n\n## Corrections:\n" + review_text
                else:
                    self._add_step(
                        AgentState.REVIEWING.value,
                        "Self-review confirmed extraction is complete",
                        observation="No corrections needed"
                    )
        
        # Step 5: Clean raw output of any prompt leakage
        raw_output = self._clean_raw_output(raw_output)
        
        # Step 6: Parse the output (minimal processing)
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
    
    def _clean_raw_output(self, text: str) -> str:
        """Remove any prompt leakage from raw output."""
        if not text:
            return text
        
        # Lines to completely remove (instruction lines)
        REMOVE_LINES = [
            "START NOW", "CRITICAL RULES", "OUTPUT FORMAT", "PROCESS:",
            "FORMAT:", "RULES:", "CHECK:", "Look at", "You are",
            "just the text", "nothing else", "no explanation",
            "نص عادي", "سطر بسطر", "علامة ؟", "للكلمات غير",
            "المهمة:", "التنسيق:", "قواعد:", "ابدأ الاستخراج",
        ]
        
        lines = text.split("\n")
        clean_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Skip empty lines at start
            if not stripped and not clean_lines:
                continue
            # Skip instruction lines
            if any(phrase.lower() in stripped.lower() for phrase in REMOVE_LINES):
                continue
            # Skip markdown headers that look like instructions
            if stripped.startswith("##") and any(w in stripped.lower() for w in ["correction", "تصحيح", "check"]):
                continue
            clean_lines.append(line)
        
        # Remove trailing empty lines
        while clean_lines and not clean_lines[-1].strip():
            clean_lines.pop()
        
        return "\n".join(clean_lines)
    
    def _parse_response(self, text: str, doc_type: DocumentType) -> tuple:
        """
        Parse extraction response with minimal filtering.
        Different parsing strategy based on document type.
        """
        fields = {}
        confidence = {}
        
        if not text:
            return fields, confidence
        
        # Filter out prompt leakage - common instruction phrases
        SKIP_PHRASES = [
            # English instructions
            "START NOW", "CRITICAL RULES", "OUTPUT FORMAT", "PROCESS:",
            "FORMAT:", "RULES:", "CHECK:", "Look at", "You are",
            "just the text", "nothing else", "no explanation",
            # Arabic instructions
            "المهمة", "التنسيق", "قواعد", "ابدأ", "اقرأ كل",
            "نص عادي", "سطر بسطر", "علامة",
            # Correction markers
            "FIX:", "ADD:", "REMOVE:", "Corrections:",
            "تصحيح:", "إضافة:", "حذف:",
        ]
        
        lines = text.strip().split("\n")
        
        # Clean lines - remove any that look like instructions
        clean_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip if contains instruction phrases
            if any(phrase.lower() in line.lower() for phrase in SKIP_PHRASES):
                continue
            # Skip headers and markdown
            if line.startswith("#") or line.startswith("##"):
                continue
            clean_lines.append(line)
        
        if doc_type == DocumentType.HANDWRITTEN:
            # For handwritten text, keep lines as-is
            line_num = 0
            for line in clean_lines:
                # Remove any leading numbers/bullets that model might add
                cleaned = re.sub(r'^[\d\.\)\-\*]+\s*', '', line).strip()
                if not cleaned:
                    continue
                
                line_num += 1
                field_name = f"سطر_{line_num}"
                
                # Check for uncertainty markers
                conf = "HIGH"
                if "?" in cleaned or "؟" in cleaned:
                    conf = "MEDIUM"
                
                fields[field_name] = cleaned
                confidence[field_name] = conf
        
        else:
            # For forms and other types, look for field:value pattern
            for line in clean_lines:
                # Try to parse as field:value
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        field_name = parts[0].strip()
                        value = parts[1].strip()
                        
                        # Skip if looks like instruction (too long)
                        if len(field_name) > 50:
                            continue
                        
                        # Clean up field name - remove bullets/numbers
                        field_name = re.sub(r'^[\d\.\)\-\*•·]+\s*', '', field_name).strip()
                        field_name = field_name.replace(" ", "_")
                        
                        if not field_name or len(field_name) < 2:
                            continue
                        
                        # Determine confidence
                        conf = "HIGH"
                        if "?" in value or "؟" in value:
                            conf = "MEDIUM"
                        if not value or value == "---":
                            conf = "LOW"
                        
                        fields[field_name] = value
                        confidence[field_name] = conf
                
                elif doc_type == DocumentType.MIXED and line:
                    # For mixed, also capture standalone lines (Arabic text)
                    if len(line) > 3:
                        line_key = f"نص_{len(fields) + 1}"
                        fields[line_key] = line
                        conf = "HIGH"
                        if "?" in line or "؟" in line:
                            conf = "MEDIUM"
                        confidence[line_key] = conf
        
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
