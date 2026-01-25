"""
Prompt Templates for Agentic OCR System - Superior Anti-Hallucination Edition

FUNDAMENTAL PRINCIPLE: Simple prompts that ask for ONE thing produce better results
than complex prompts that ask for many things.

The key insight is that hallucination happens when the model:
1. Tries to fill in gaps with "reasonable" data
2. Gets confused by complex instructions
3. Copies values it saw once to fill other fields

SOLUTION: Two-stage approach
1. Stage 1: RAW TRANSCRIPTION - Just read what's handwritten, nothing more
2. Stage 2: STRUCTURE - Parse the raw text into fields (done by LLM, not VLM)
"""

# =============================================================================
# STAGE 1: RAW TRANSCRIPTION PROMPT (VLM)
# =============================================================================
# This is the ONLY prompt the VLM sees for initial extraction.
# It's deliberately simple to reduce hallucination.

INITIAL_OCR_PROMPT = """انظر إلى هذه الصورة. اكتب فقط النص المكتوب بخط اليد الذي تراه.

قواعد بسيطة:
١. اكتب فقط ما هو مكتوب بخط اليد (بالحبر)
٢. تجاهل النص المطبوع (العناوين والتسميات)
٣. إذا كان الحقل فارغاً (نقاط أو خطوط فقط) اكتب: ---
٤. اكتب كل قيمة في سطر منفصل

مثال للمخرجات:
محمد أحمد العتيبي
١٠٥٨٣٧٦٢٨٠
٠٥٥١٢٣٤٥٦٧
١٥/٠٧/١٤٠٥
جدة
---
---

ابدأ الآن. اكتب فقط القيم المكتوبة بخط اليد:"""

# Alternative English prompt for debugging
INITIAL_OCR_PROMPT_EN = """Look at this image. Write ONLY the handwritten text you can see.

Simple rules:
1. Write ONLY handwritten text (ink marks)
2. IGNORE printed text (headers, labels)
3. If a field is empty (dots or lines only), write: ---
4. Write each value on a separate line

Example output:
محمد أحمد العتيبي
١٠٥٨٣٧٦٢٨٠
٠٥٥١٢٣٤٥٦٧
١٥/٠٧/١٤٠٥
جدة
---
---

Start now. Write ONLY the handwritten values:"""


# =============================================================================
# STAGE 1B: VISUAL DESCRIPTION PROMPT (VLM)
# =============================================================================
# Alternative approach: Have the model describe what it sees first,
# which grounds it in the actual image content.

VISUAL_DESCRIPTION_PROMPT = """صف ما تراه في هذه الصورة.

١. ما نوع هذه الوثيقة؟
٢. كم عدد الحقول المملوءة بخط اليد؟
٣. كم عدد الحقول الفارغة؟
٤. ما اللغة المستخدمة في الكتابة اليدوية؟
٥. هل الكتابة واضحة أم غير واضحة؟

أجب بإيجاز:"""


# =============================================================================
# STAGE 1C: LINE-BY-LINE TRANSCRIPTION (VLM)
# =============================================================================
# Most focused prompt - asks model to read one line at a time

LINE_BY_LINE_PROMPT = """اقرأ النص المكتوب بخط اليد في هذه الصورة سطراً بسطر.

لكل سطر يحتوي على كتابة يدوية، اكتب:
- رقم السطر
- القيمة المكتوبة

إذا كان السطر فارغاً، اكتب: [فارغ]
إذا لم تستطع القراءة، اكتب: [؟]

مثال:
١: محمد أحمد
٢: ١٠٥٨٣٧٦٢٨٠
٣: [فارغ]
٤: ٠٥٥١٢٣٤٥٦٧

اقرأ الصورة الآن:"""


# =============================================================================
# STAGE 2: FIELD MAPPING PROMPT (LLM - NOT VLM)
# =============================================================================
# The LLM takes the raw transcription and maps it to fields.
# This separates "reading" from "understanding".

FIELD_MAPPING_PROMPT = """أنت محلل نماذج. لديك:
١. قائمة القيم المستخرجة من الصورة
٢. قائمة الحقول المتوقعة في النموذج

## القيم المستخرجة:
{raw_transcription}

## الحقول المتوقعة في النموذج السعودي:
- اسم المالك / الاسم (نص عربي)
- رقم الهوية / رقم بطاقة الأحوال (١٠ أرقام تبدأ بـ ١ أو ٢)
- رقم الجوال (١٠ أرقام تبدأ بـ ٠٥)
- تاريخ الميلاد (يوم/شهر/سنة)
- تاريخ الإصدار (يوم/شهر/سنة)
- تاريخ الانتهاء (يوم/شهر/سنة)
- المدينة (اسم مدينة)
- رقم اللوحة (حروف وأرقام)

## مهمتك:
طابق كل قيمة مستخرجة مع الحقل المناسب بناءً على الصيغة.

## قواعد صارمة:
- لا تخترع قيماً جديدة
- إذا لم تجد قيمة مناسبة للحقل، اكتب: [غير موجود]
- إذا كانت القيمة "---" أو "[فارغ]"، الحقل فارغ

## المخرجات (JSON):
{{
  "field_mapping": {{
    "اسم المالك": "القيمة أو [غير موجود] أو [فارغ]",
    "رقم الهوية": "القيمة أو [غير موجود] أو [فارغ]",
    ...
  }},
  "unmatched_values": ["قيم لم يتم مطابقتها"],
  "confidence": "high/medium/low"
}}"""


# =============================================================================
# STEP 2: ANALYSIS PROMPT (LLM)
# =============================================================================

ANALYSIS_PROMPT = """You are an OCR quality analyst. Analyze this extraction for issues.

## EXTRACTED DATA:
{initial_extraction}

## CHECK FOR THESE PROBLEMS:

1. **DUPLICATE VALUES** - Same value appearing multiple times
   - If value X appears in 2+ fields → flag ALL those fields
   
2. **FORMAT MISMATCHES**
   - ID not 10 digits starting with 1/2 → flag
   - Phone not 10 digits starting with 05 → flag
   - Date not in day/month/year format → flag

3. **SUSPICIOUS PATTERNS**
   - All fields have values (real forms have empty fields)
   - Year (1400-1450) in non-date fields
   - No Arabic names found

## OUTPUT (JSON only):
{{
  "analysis": {{
    "total_fields": <number>,
    "high_confidence": <number>,
    "medium_confidence": <number>,
    "low_confidence": <number>,
    "empty_fields": <number>,
    "needs_reexamination": <number>,
    "hallucination_detected": <boolean>,
    "hallucination_type": "<string or null>"
  }},
  "fields_to_reexamine": [
    {{
      "field_name": "رقم الهوية",
      "current_value": "١٤٠٠",
      "confidence": "LOW",
      "issue": "4-digit year used as ID (hallucination)",
      "priority": "critical",
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
  ],
  "hallucination_warnings": []
}}"""


# =============================================================================
# STEP 3: REGION ESTIMATION (LLM)
# =============================================================================

REGION_ESTIMATION_PROMPT = """Estimate bounding boxes for these fields on an Arabic form.

## FIELDS TO LOCATE:
{fields_to_reexamine}

## IMAGE DIMENSIONS:
Width: {width}px, Height: {height}px

## ARABIC FORM LAYOUT:
- Right-to-left reading
- Labels on RIGHT, values on LEFT
- Personal info at TOP
- Vehicle info in MIDDLE
- Signatures at BOTTOM

## COMMON POSITIONS (normalized 0-1):
- Name: [0.4, 0.08, 0.95, 0.14]
- ID: [0.4, 0.12, 0.95, 0.18]
- Birth date: [0.4, 0.16, 0.95, 0.22]
- Phone: [0.4, 0.28, 0.95, 0.34]
- City: [0.4, 0.32, 0.95, 0.38]
- Plate: [0.4, 0.50, 0.95, 0.56]

## OUTPUT (JSON only):
{{
  "regions": [
    {{
      "field_name": "رقم الهوية",
      "bbox_normalized": [0.4, 0.12, 0.95, 0.18],
      "bbox_pixels": [432, 97, 1026, 146],
      "location_confidence": "high",
      "notes": "Standard ID position"
    }}
  ]
}}"""


# =============================================================================
# STEP 4: FOCUSED FIELD EXTRACTION (VLM)
# =============================================================================
# Extremely simple prompt for cropped region

FIELD_REEXTRACT_PROMPT = """ما المكتوب هنا؟

إذا فارغ: ---
إذا غير واضح: ؟

اكتب القيمة فقط:"""

# Slightly more detailed version
FIELD_REEXTRACT_PROMPT_DETAILED = """هذه صورة مقصوصة لحقل "{field_name}".

اقرأ القيمة المكتوبة بخط اليد.

قواعد:
- إذا فارغ (نقاط/خطوط فقط): اكتب ---
- إذا غير واضح: اكتب ؟
- خلاف ذلك: اكتب القيمة فقط

المتوقع: {content_hint}

القيمة:"""


# Content hints for different field types
FIELD_CONTENT_HINTS = {
    "رقم الهوية": "١٠ أرقام",
    "رقم بطاقة الأحوال": "١٠ أرقام",
    "رقم الجوال": "١٠ أرقام تبدأ ٠٥",
    "الجوال": "١٠ أرقام تبدأ ٠٥",
    "تاريخ الميلاد": "يوم/شهر/سنة",
    "تاريخ الإصدار": "يوم/شهر/سنة",
    "تاريخ الانتهاء": "يوم/شهر/سنة",
    "اسم المالك": "اسم عربي",
    "الاسم": "اسم عربي",
    "المدينة": "مدينة",
    "رقم اللوحة": "حروف وأرقام",
    "default": "نص"
}

FORMAT_VALIDATION_HINTS = {
    "رقم الهوية": "١٠ أرقام تبدأ بـ ١ أو ٢",
    "رقم بطاقة الأحوال": "١٠ أرقام",
    "رقم الجوال": "١٠ أرقام تبدأ بـ ٠٥",
    "تاريخ": "يوم/شهر/سنة",
    "الاسم": "كلمات عربية",
    "default": ""
}


def get_content_hint(field_name: str) -> str:
    """Get content hint for a field name."""
    if field_name in FIELD_CONTENT_HINTS:
        return FIELD_CONTENT_HINTS[field_name]
    
    for key, hint in FIELD_CONTENT_HINTS.items():
        if key in field_name or field_name in key:
            return hint
    
    return FIELD_CONTENT_HINTS["default"]


def get_format_hint(field_name: str) -> str:
    """Get format validation hint for a field name."""
    if field_name in FORMAT_VALIDATION_HINTS:
        return FORMAT_VALIDATION_HINTS[field_name]
    
    for key, hint in FORMAT_VALIDATION_HINTS.items():
        if key in field_name or field_name in key:
            return hint
    
    return FORMAT_VALIDATION_HINTS["default"]


def build_field_reextract_prompt(
    field_name: str,
    previous_value: str = "",
    content_hint: str = None
) -> str:
    """Build field re-extraction prompt."""
    hint = content_hint or get_content_hint(field_name)
    
    return FIELD_REEXTRACT_PROMPT_DETAILED.format(
        field_name=field_name,
        content_hint=hint
    )


# =============================================================================
# STEP 5: MERGE PROMPT (LLM)
# =============================================================================

MERGE_PROMPT = """Merge OCR results from two passes.

## ORIGINAL EXTRACTION:
{original_extraction}

## REFINED VALUES:
{refined_values}

## RULES:
1. If REFINED is more specific → use REFINED
2. If REFINED confirms ORIGINAL → use ORIGINAL with HIGH confidence
3. If REFINED is "---" but ORIGINAL had value → keep ORIGINAL (crop may have missed it)
4. If both unclear → mark for human review
5. NEVER invent values

## OUTPUT (JSON only):
{{
  "final_fields": {{
    "field_name": {{
      "value": "final_value",
      "source": "original|refined|merged",
      "confidence": "high|medium|low",
      "needs_human_review": false,
      "review_reason": null
    }}
  }},
  "merge_quality_check": {{
    "duplicate_values_found": false,
    "year_as_value_found": false,
    "suspicious_fills": [],
    "quality_improved": true
  }},
  "summary": {{
    "total_fields": 0,
    "from_original": 0,
    "from_refined": 0,
    "flagged_for_review": 0
  }},
  "iteration_complete": true,
  "fields_still_uncertain": []
}}"""


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

ANALYSIS_SYSTEM_PROMPT = """You are an OCR quality analyst. Your job is to:
1. Detect hallucinations (same value in multiple fields)
2. Identify format violations
3. Flag suspicious patterns

Output valid JSON only."""

REGION_SYSTEM_PROMPT = """You estimate field locations on Arabic forms.
Arabic forms are right-to-left.
Output valid JSON with bounding boxes."""

MERGE_SYSTEM_PROMPT = """You merge OCR results intelligently.
Never invent values. When uncertain, flag for review.
Output valid JSON."""


# =============================================================================
# FINAL VALIDATION PROMPT
# =============================================================================

FINAL_VALIDATION_PROMPT = """Validate this OCR extraction.

## DATA:
{extraction}

## CHECK:
1. All values unique? (duplicates = hallucination)
2. Formats correct? (ID=10 digits, phone=05..., date=dd/mm/yyyy)
3. Arabic names present?

## OUTPUT (JSON):
{{
  "quality_score": 0-100,
  "passed": true/false,
  "issues": [],
  "recommendation": "pass|review|reject"
}}"""
