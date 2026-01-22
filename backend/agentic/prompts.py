"""
Prompt Templates for Agentic OCR System

Contains all prompts used in the multi-pass OCR pipeline:
1. Initial OCR extraction (VLM)
2. Analysis and issue detection (LLM)
3. Region estimation (LLM)
4. Field-specific re-extraction (VLM)
5. Intelligent merge (LLM)
"""

# =============================================================================
# STEP 1: INITIAL OCR EXTRACTION (AIN VLM)
# =============================================================================

INITIAL_OCR_PROMPT = """أنت نظام نسخ بصري دقيق للنماذج العربية.

TASK: Extract ALL fields from this Arabic form with confidence levels.

CRITICAL RULES:
1. ONLY transcribe what you can VISUALLY SEE
2. NEVER invent, guess, or complete missing values
3. NEVER fill blank fields with plausible-looking data
4. Mark your confidence for EACH field

OUTPUT FORMAT (one field per line):
field_name: value [HIGH|MEDIUM|LOW]
field_name: [فارغ] [HIGH]
field_name: [غير واضح: best_guess] [LOW]
field_name: [غير مقروء] [LOW]

CONFIDENCE LEVELS:
- HIGH: Clear, easily readable, you are certain
- MEDIUM: Readable but some letters/digits unclear
- LOW: Difficult to read, your transcription may be wrong

SPECIAL MARKERS:
- [فارغ] = Field is empty (just dots/lines, no handwriting)
- [غير واضح: X] = Unclear, best guess is X
- [غير مقروء] = Completely unreadable
- [توقيع] = Signature present
- [ختم] = Stamp present

PRESERVE EXACTLY AS WRITTEN:
- Arabic numerals: ٠١٢٣٤٥٦٧٨٩
- Dates exactly as seen
- Phone numbers exactly as written
- Spelling errors (transcribe the error)

Extract all fields now:"""


# =============================================================================
# STEP 2: ANALYSIS AND ISSUE DETECTION (QWEN LLM)
# =============================================================================

ANALYSIS_PROMPT = """You are an OCR quality analyst specializing in Arabic government forms.

## EXTRACTED DATA:
{initial_extraction}

## YOUR TASK:
Analyze each extracted field and identify which ones need re-examination.

## FLAG THESE ISSUES:
1. Fields marked [LOW] confidence - need re-examination
2. Fields marked [غير واضح] - unclear, need focused re-read
3. Fields marked [غير مقروء] - unreadable, try cropped re-read
4. Suspicious values that may be hallucinated:
   - Sequential digits (1234567890, 9876543210)
   - Generic names when alone (محمد أحمد, عبدالله)
   - Round numbers for IDs (1000000000, 2000000000)
   - Values that don't match field type (letters in phone number)
5. Critical fields with MEDIUM confidence (IDs, dates, phone numbers)

## DO NOT FLAG:
- Fields marked [HIGH] with reasonable values
- Fields marked [فارغ] (confirmed empty)
- Non-critical fields with MEDIUM confidence

## OUTPUT FORMAT (JSON only, no other text):
{{
  "analysis": {{
    "total_fields": <number>,
    "high_confidence": <number>,
    "medium_confidence": <number>,
    "low_confidence": <number>,
    "empty_fields": <number>,
    "needs_reexamination": <number>
  }},
  "fields_to_reexamine": [
    {{
      "field_name": "رقم الهوية",
      "current_value": "١٠٣٨٣٦٧٦٨٠",
      "confidence": "MEDIUM",
      "issue": "Critical field with medium confidence",
      "priority": "high",
      "field_type": "national_id",
      "expected_format": "10 digits starting with 1 or 2"
    }}
  ],
  "confident_fields": [
    {{
      "field_name": "اسم المالك",
      "value": "عياض محمد العتيبي",
      "confidence": "HIGH"
    }}
  ],
  "empty_fields": [
    {{
      "field_name": "رقم الجوال",
      "status": "confirmed_empty"
    }}
  ]
}}"""


# =============================================================================
# STEP 3: REGION ESTIMATION (QWEN LLM)
# =============================================================================

REGION_ESTIMATION_PROMPT = """You are helping locate fields on an Arabic form for re-scanning.

## FORM LAYOUT KNOWLEDGE:
- Arabic forms read RIGHT-TO-LEFT
- Field LABELS are usually on the RIGHT side
- HANDWRITTEN VALUES are to the LEFT of their labels
- Forms typically have sections:
  - Header/Title: Top center
  - Personal Info: Upper portion (name, ID, birth date)
  - Contact Info: Middle portion (phone, address, city)
  - Vehicle/Property Info: Middle-lower (plate, model, year)
  - Signatures/Stamps: Bottom portion

## FIELDS TO LOCATE:
{fields_to_reexamine}

## IMAGE DIMENSIONS:
Width: {width}px, Height: {height}px

## COMMON FIELD POSITIONS (normalized 0-1, approximate):
- Form Title: [0.2, 0.0, 0.8, 0.08]
- اسم المالك (Owner Name): [0.5, 0.08, 0.95, 0.15]
- رقم الهوية (ID Number): [0.5, 0.12, 0.95, 0.18]
- تاريخ الميلاد (Birth Date): [0.5, 0.16, 0.95, 0.22]
- رقم الجوال (Mobile): [0.5, 0.20, 0.95, 0.26]
- المدينة (City): [0.5, 0.24, 0.95, 0.30]
- الحي (District): [0.5, 0.28, 0.95, 0.34]
- رقم اللوحة (Plate Number): [0.5, 0.45, 0.95, 0.52]
- نوع المركبة (Vehicle Type): [0.5, 0.50, 0.95, 0.56]
- التوقيع (Signature): [0.1, 0.85, 0.4, 0.95]

## OUTPUT FORMAT (JSON only):
{{
  "regions": [
    {{
      "field_name": "رقم الهوية",
      "bbox_normalized": [0.5, 0.12, 0.95, 0.18],
      "bbox_pixels": [540, 97, 1029, 146],
      "location_confidence": "high",
      "notes": "ID field in standard position"
    }}
  ],
  "estimation_notes": "Based on standard Saudi form layout"
}}

Note: bbox format is [x1, y1, x2, y2] where (0,0) is top-left corner.
Calculate bbox_pixels by multiplying normalized coordinates by image dimensions."""


# =============================================================================
# STEP 4: FIELD-SPECIFIC RE-EXTRACTION (AIN VLM)
# =============================================================================

FIELD_REEXTRACT_PROMPT = """This is a CROPPED image showing ONLY the field "{field_name}" from an Arabic form.

YOUR TASK: Read ONLY the handwritten value in this crop.

CRITICAL RULES:
1. This crop shows ONE field area only
2. IGNORE any printed labels - focus on HANDWRITTEN text
3. If empty (just dots/lines, no ink): respond [فارغ]
4. If unclear: respond [غير واضح: your_best_guess]
5. If completely unreadable: respond [غير مقروء]
6. Transcribe EXACTLY what you see - do not correct or normalize

FIELD INFORMATION:
- Field name: {field_name}
- Expected content: {content_hint}
- Previous extraction: {previous_value}

Look very carefully at any handwritten marks. Even faint writing should be transcribed.

Your transcription (value only, no field name):"""


# Content hints for different field types
FIELD_CONTENT_HINTS = {
    "رقم الهوية": "10-digit Saudi ID number starting with 1 (citizen) or 2 (resident)",
    "رقم الجوال": "10-digit Saudi mobile starting with 05",
    "تاريخ الميلاد": "Hijri date, format: DD/MM/YYYY (e.g., ١٥/٠٧/١٤٠٥)",
    "تاريخ الإصدار": "Hijri date of issuance",
    "تاريخ الانتهاء": "Hijri expiry date",
    "اسم المالك": "Arabic full name, typically 3-4 words",
    "الاسم": "Arabic name",
    "المدينة": "Saudi city name in Arabic",
    "الحي": "District/neighborhood name",
    "العنوان": "Full address",
    "رقم اللوحة": "Vehicle plate: 1-3 Arabic letters + 1-4 digits",
    "نوع المركبة": "Vehicle type/model",
    "سنة الصنع": "Year (Gregorian or Hijri)",
    "اللون": "Color name in Arabic",
    "default": "Arabic text or numbers"
}


def get_content_hint(field_name: str) -> str:
    """Get content hint for a field name."""
    # Check exact match
    if field_name in FIELD_CONTENT_HINTS:
        return FIELD_CONTENT_HINTS[field_name]
    
    # Check partial match
    for key, hint in FIELD_CONTENT_HINTS.items():
        if key in field_name or field_name in key:
            return hint
    
    return FIELD_CONTENT_HINTS["default"]


# =============================================================================
# STEP 5: INTELLIGENT MERGE (QWEN LLM)
# =============================================================================

MERGE_PROMPT = """You are merging OCR results from two passes. For each re-examined field, you have:
- ORIGINAL: First-pass extraction from full page
- REFINED: Second-pass extraction from cropped, focused region

## ORIGINAL EXTRACTION:
{original_extraction}

## REFINED VALUES (from cropped regions):
{refined_values}

## DECISION RULES:
1. If REFINED is more specific/complete than ORIGINAL → use REFINED
2. If REFINED confirms ORIGINAL (same or very similar) → use ORIGINAL with HIGH confidence
3. If they significantly differ → flag for human review
4. If REFINED is [فارغ] but ORIGINAL had a value → likely crop missed the field, keep ORIGINAL but flag
5. If both are [غير واضح] → combine information or mark as needs human review
6. If REFINED is [غير مقروء] but ORIGINAL had value → keep ORIGINAL, may be crop quality issue

## IMPORTANT:
- Never invent new values
- Original raw values should be preserved unless refinement is clearly better
- When in doubt, flag for human review rather than guessing

## OUTPUT FORMAT (JSON only):
{{
  "final_fields": {{
    "field_name": {{
      "value": "final_value",
      "source": "original|refined|merged",
      "confidence": "high|medium|low",
      "needs_human_review": false,
      "review_reason": null,
      "notes": "optional explanation"
    }}
  }},
  "summary": {{
    "total_fields": <number>,
    "from_original": <number>,
    "from_refined": <number>,
    "merged": <number>,
    "flagged_for_review": <number>
  }},
  "iteration_complete": true,
  "fields_still_uncertain": []
}}"""


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

ANALYSIS_SYSTEM_PROMPT = """You are an expert OCR quality analyst specializing in Arabic government forms and documents. 
Your role is to identify potential errors, uncertainties, and suspicious values in OCR extractions.
Always output valid JSON. Be thorough but avoid false positives."""

REGION_SYSTEM_PROMPT = """You are an expert in Arabic form layouts and document structure.
Your role is to estimate where specific fields are located in form images.
Always output valid JSON with normalized bounding boxes."""

MERGE_SYSTEM_PROMPT = """You are an intelligent OCR result merger.
Your role is to combine multiple OCR passes and select the most accurate value for each field.
Prioritize accuracy over completeness. When uncertain, flag for human review.
Always output valid JSON."""
