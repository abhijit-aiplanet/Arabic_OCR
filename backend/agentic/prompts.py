"""
Prompt Templates for Agentic OCR System - Anti-Hallucination Edition

Contains all prompts used in the multi-pass OCR pipeline with
EXPLICIT anti-hallucination rules to prevent:
- Value propagation (same value in multiple fields)
- Template filling (inventing plausible-looking data)
- Empty field hallucination (filling dots with fake data)
- Format confusion (years as IDs, etc.)

Pipeline Steps:
1. Initial OCR extraction (VLM)
2. Analysis and issue detection (LLM)
3. Region estimation (LLM)
4. Field-specific re-extraction (VLM)
5. Intelligent merge (LLM)
"""

# =============================================================================
# STEP 1: INITIAL OCR EXTRACTION (AIN VLM)
# =============================================================================

INITIAL_OCR_PROMPT = """أنت نظام نسخ بصري دقيق للنماذج العربية. مهمتك استخراج النص المكتوب بخط اليد فقط.

=== CRITICAL ANTI-HALLUCINATION RULES ===

⛔ FORBIDDEN BEHAVIORS:
1. NEVER copy the same value to multiple fields
2. NEVER invent or guess values - only transcribe what you SEE
3. NEVER fill empty dotted lines (....) with plausible data
4. NEVER use a year (1400, 1401, etc.) for non-date fields
5. NEVER generate sequential numbers (1234567890)

✅ REQUIRED BEHAVIORS:
1. Each field MUST have a UNIQUE value (unless truly identical in image)
2. Empty fields (dots, blank lines) = [فارغ]
3. Unreadable fields = [غير مقروء]
4. Unclear but attempted = [غير واضح: your_guess]
5. PRESERVE handwriting errors exactly as written

=== SELF-CHECK BEFORE RESPONDING ===
Before you output, verify:
□ Do any 3+ fields have the SAME value? → You made an error
□ Is a year (1400-1450) in a non-date field? → That's wrong
□ Are all numeric fields identical? → Hallucination detected
□ Did you fill empty dotted areas? → Should be [فارغ]

=== OUTPUT FORMAT ===
field_name: value [HIGH|MEDIUM|LOW]
field_name: [فارغ] [HIGH]
field_name: [غير واضح: guess] [LOW]
field_name: [غير مقروء] [LOW]

=== CONFIDENCE LEVELS ===
- HIGH: Clear handwriting, certain of value
- MEDIUM: Readable but some characters unclear
- LOW: Difficult to read, uncertain transcription

=== FIELD-SPECIFIC VALIDATION ===
- رقم الهوية (ID): Must be 10 digits starting with 1 or 2
- رقم الجوال (Mobile): Must be 10 digits starting with 05
- تاريخ (Date): Must have day/month/year format (e.g., ١٥/٠٧/١٤٠٥)
- الاسم (Name): Must be Arabic words, NOT numbers

=== EMPTY FIELD DETECTION ===
Look for these indicators of EMPTY fields:
- Dotted lines: . . . . . . . . .
- Horizontal lines: ___________
- Blank white space with no ink
- Checkbox squares without marks
→ ALL of these = [فارغ], do NOT fill with data

=== HANDWRITING FOCUS ===
Focus on ACTUAL handwritten content:
- Pen/ink marks on the form
- Filled-in boxes or circles
- Written text, not printed labels
- Signatures and stamps = [توقيع] or [ختم]

Extract all visible fields now. Remember: if in doubt, mark [غير واضح] or [غير مقروء] - never guess."""


# =============================================================================
# STEP 2: ANALYSIS AND ISSUE DETECTION (LLM)
# =============================================================================

ANALYSIS_PROMPT = """You are an OCR quality analyst specializing in Arabic government forms.
Your job is to identify extraction errors and hallucinations.

## EXTRACTED DATA TO ANALYZE:
{initial_extraction}

## YOUR TASK:
1. Identify fields with quality issues
2. DETECT HALLUCINATION PATTERNS (critical!)
3. Flag suspicious values for re-examination

## ⚠️ HALLUCINATION DETECTION (CRITICAL) ⚠️

CHECK FOR THESE PATTERNS:

1. **VALUE PROPAGATION** - Same value in multiple unrelated fields
   - If "١٤٠٠" appears in رقم الهوية, رقم اللوحة, رقم الهيكل → ALL are hallucinated
   - Action: Flag ALL fields with the duplicated value

2. **YEAR AS VALUE** - Year numbers (1400-1450) in non-date fields
   - A 4-digit Hijri year should ONLY appear in date fields
   - If رقم الجوال = "١٤٠٠" → Hallucination (phones are 10 digits starting 05)
   - If رقم الهوية = "١٤٠١" → Hallucination (IDs are 10 digits starting 1 or 2)

3. **MISSING ARABIC NAMES** - No Arabic text in name fields
   - Name fields (اسم, اسم المالك) should have Arabic words
   - If name = numbers or year → Hallucination

4. **IMPOSSIBLE VALUES** - Values that violate field rules
   - ID < 10 digits → Invalid
   - Phone not starting with 05 → Invalid
   - Plate with no Arabic letters → Invalid

5. **SUSPICIOUSLY COMPLETE** - All fields filled but form looks sparse
   - Real forms often have empty fields
   - If EVERY field has a value, check for hallucination

## FLAG THESE ISSUES:
1. Fields marked [LOW] confidence - need re-examination
2. Fields marked [غير واضح] - unclear, need focused re-read
3. Duplicate values across fields - ALL affected fields need checking
4. Year values in non-date fields - likely hallucinated
5. Format violations (wrong digit count, missing patterns)

## DO NOT FLAG:
- Fields marked [HIGH] with unique, valid values
- Fields marked [فارغ] (confirmed empty)
- Single-occurrence values that match expected format

## OUTPUT FORMAT (JSON only, no other text):
{{
  "analysis": {{
    "total_fields": <number>,
    "high_confidence": <number>,
    "medium_confidence": <number>,
    "low_confidence": <number>,
    "empty_fields": <number>,
    "needs_reexamination": <number>,
    "hallucination_detected": <boolean>,
    "hallucination_type": <string or null>
  }},
  "fields_to_reexamine": [
    {{
      "field_name": "رقم الهوية",
      "current_value": "١٤٠٠",
      "confidence": "LOW",
      "issue": "Value appears in 5 other fields (value propagation hallucination)",
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
  "hallucination_warnings": [
    "Value '١٤٠٠' appears in 6 different fields - likely model confusion",
    "No Arabic names detected despite name fields being present"
  ]
}}"""


# =============================================================================
# STEP 3: REGION ESTIMATION (LLM)
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
These are estimates - adjust based on the specific form:

Personal Information Section (typically top 30%):
- Form Title: [0.2, 0.0, 0.8, 0.08]
- اسم المالك (Owner Name): [0.4, 0.08, 0.95, 0.14]
- رقم الهوية (ID Number): [0.4, 0.12, 0.95, 0.18]
- رقم بطاقة الأحوال (ID Card): [0.4, 0.12, 0.95, 0.18]
- تاريخ الميلاد (Birth Date): [0.4, 0.16, 0.95, 0.22]
- تاريخ الإصدار (Issue Date): [0.4, 0.20, 0.95, 0.26]
- تاريخ الانتهاء (Expiry Date): [0.4, 0.24, 0.95, 0.30]

Contact Information Section (typically middle):
- رقم الجوال (Mobile): [0.4, 0.28, 0.95, 0.34]
- المدينة (City): [0.4, 0.32, 0.95, 0.38]
- الحي (District): [0.4, 0.36, 0.95, 0.42]
- العنوان (Address): [0.2, 0.40, 0.95, 0.48]

License/Vehicle Section (typically lower middle):
- رقم رخصة القيادة (License Number): [0.4, 0.45, 0.95, 0.52]
- رقم اللوحة (Plate Number): [0.4, 0.50, 0.95, 0.56]
- نوع المركبة (Vehicle Type): [0.4, 0.54, 0.95, 0.60]
- سنة الصنع (Year): [0.4, 0.58, 0.95, 0.64]

Bottom Section:
- التوقيع (Signature): [0.1, 0.80, 0.4, 0.95]
- الختم (Stamp): [0.6, 0.80, 0.9, 0.95]

## OUTPUT FORMAT (JSON only):
{{
  "regions": [
    {{
      "field_name": "رقم الهوية",
      "bbox_normalized": [0.4, 0.12, 0.95, 0.18],
      "bbox_pixels": [432, 97, 1026, 146],
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

FIELD_REEXTRACT_PROMPT = """هذه صورة مقصوصة تظهر فقط حقل "{field_name}" من نموذج عربي.

مهمتك: اقرأ فقط القيمة المكتوبة بخط اليد في هذه المنطقة المقصوصة.

=== قواعد صارمة ===

⛔ ممنوع:
- لا تخترع قيمًا - اكتب فقط ما تراه
- لا تستخدم القيمة السابقة إذا لم تستطع قراءة الجديدة
- لا تملأ المنطقة الفارغة ببيانات وهمية

✅ مطلوب:
- إذا كانت المنطقة فارغة (نقاط، خطوط، بياض): اكتب [فارغ]
- إذا كانت غير واضحة: اكتب [غير واضح: تخمينك]
- إذا كانت غير مقروءة تمامًا: اكتب [غير مقروء]
- إذا استطعت القراءة: اكتب القيمة فقط بدون اسم الحقل

=== معلومات الحقل ===
- اسم الحقل: {field_name}
- المحتوى المتوقع: {content_hint}
- القيمة السابقة (للمرجع فقط): {previous_value}

=== التحقق من الصيغة ===
{format_validation_hint}

=== إشارات الحقل الفارغ ===
انظر إلى:
- نقاط متتالية: . . . . . .
- خطوط فارغة: _______
- مساحة بيضاء بدون حبر
→ كل هذه = [فارغ]

ماذا ترى في الصورة المقصوصة؟ (القيمة فقط، بدون اسم الحقل):"""


# Content hints for different field types
FIELD_CONTENT_HINTS = {
    "رقم الهوية": "رقم مكون من 10 أرقام يبدأ بـ 1 (مواطن) أو 2 (مقيم)",
    "رقم بطاقة الأحوال": "رقم مكون من 10 أرقام يبدأ بـ 1 أو 2",
    "رقم الجوال": "رقم مكون من 10 أرقام يبدأ بـ 05",
    "الجوال": "رقم مكون من 10 أرقام يبدأ بـ 05",
    "تاريخ الميلاد": "تاريخ هجري بصيغة يوم/شهر/سنة (مثال: ١٥/٠٧/١٤٠٥)",
    "تاريخ الإصدار": "تاريخ هجري بصيغة يوم/شهر/سنة",
    "تاريخ الانتهاء": "تاريخ هجري بصيغة يوم/شهر/سنة",
    "اسم المالك": "اسم عربي كامل، عادة 3-4 كلمات",
    "الاسم": "اسم عربي",
    "المدينة": "اسم مدينة سعودية بالعربي",
    "الحي": "اسم الحي أو المنطقة",
    "العنوان": "العنوان الكامل",
    "رقم اللوحة": "لوحة السيارة: 1-3 حروف عربية + 1-4 أرقام",
    "نوع المركبة": "نوع أو موديل السيارة",
    "سنة الصنع": "سنة (ميلادية أو هجرية)",
    "اللون": "اسم اللون بالعربي",
    "رقم رخصة القيادة": "رقم الرخصة",
    "default": "نص عربي أو أرقام"
}

# Format validation hints
FORMAT_VALIDATION_HINTS = {
    "رقم الهوية": "✓ يجب أن يكون 10 أرقام بالضبط\n✓ يبدأ بـ 1 (مواطن) أو 2 (مقيم)\n✗ إذا كان 4 أرقام فقط = خطأ (ربما سنة)",
    "رقم بطاقة الأحوال": "✓ يجب أن يكون 10 أرقام بالضبط\n✓ يبدأ بـ 1 أو 2",
    "رقم الجوال": "✓ يجب أن يكون 10 أرقام بالضبط\n✓ يبدأ بـ 05\n✗ إذا كان 4 أرقام = خطأ",
    "تاريخ الميلاد": "✓ صيغة: يوم/شهر/سنة\n✓ مثال: ١/٧/١٣٨٥\n✗ سنة فقط (١٤٠٠) = ناقص",
    "تاريخ الإصدار": "✓ صيغة: يوم/شهر/سنة\n✗ سنة فقط = ناقص",
    "تاريخ الانتهاء": "✓ صيغة: يوم/شهر/سنة\n✗ سنة فقط = ناقص",
    "اسم المالك": "✓ كلمات عربية\n✗ أرقام فقط = خطأ",
    "الاسم": "✓ كلمات عربية\n✗ أرقام فقط = خطأ",
    "رقم اللوحة": "✓ حروف عربية + أرقام\n✓ مثال: أ ب ج ١٢٣٤",
    "default": ""
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


def get_format_hint(field_name: str) -> str:
    """Get format validation hint for a field name."""
    # Check exact match
    if field_name in FORMAT_VALIDATION_HINTS:
        return FORMAT_VALIDATION_HINTS[field_name]
    
    # Check partial match
    for key, hint in FORMAT_VALIDATION_HINTS.items():
        if key in field_name or field_name in key:
            return hint
    
    return FORMAT_VALIDATION_HINTS["default"]


def build_field_reextract_prompt(
    field_name: str,
    previous_value: str = "",
    content_hint: str = None
) -> str:
    """Build the complete field re-extraction prompt."""
    hint = content_hint or get_content_hint(field_name)
    format_hint = get_format_hint(field_name)
    
    return FIELD_REEXTRACT_PROMPT.format(
        field_name=field_name,
        content_hint=hint,
        previous_value=previous_value or "[لم يُستخرج سابقًا]",
        format_validation_hint=format_hint
    )


# =============================================================================
# STEP 5: INTELLIGENT MERGE (LLM)
# =============================================================================

MERGE_PROMPT = """You are merging OCR results from multiple passes.

For each re-examined field, you have:
- ORIGINAL: First-pass extraction from full page
- REFINED: Second-pass extraction from cropped, focused region

## ORIGINAL EXTRACTION:
{original_extraction}

## REFINED VALUES (from cropped regions):
{refined_values}

## MERGE DECISION RULES:

### ANTI-HALLUCINATION CHECKS FIRST:

1. **Reject Duplicate Values**
   - If REFINED has the same value as 2+ other fields → likely hallucination
   - If ORIGINAL had unique values but REFINED made them identical → keep ORIGINAL

2. **Reject Year-as-Value**
   - If REFINED is a 4-digit year (1400-1450) for a non-date field → reject it
   - Keep ORIGINAL if it was more specific

3. **Reject Suspiciously Complete**
   - If ORIGINAL had [فارغ] but REFINED has a value → verify carefully
   - Empty fields that suddenly have values may be hallucinated

### THEN APPLY QUALITY RULES:

4. **Prefer Specific Over Generic**
   - If REFINED is more specific (longer, more details) → use REFINED
   - If REFINED is just "١٤٠٠" but ORIGINAL was "١٠٣٨٣٦٧٦٨٠" → keep ORIGINAL

5. **Prefer Correct Format**
   - ID: 10 digits starting with 1 or 2
   - Phone: 10 digits starting with 05
   - Date: has day/month/year
   - Choose whichever matches the expected format

6. **Handle Uncertainty Honestly**
   - If both are unclear → mark for human review
   - Never invent a "merged" value that wasn't in either source

7. **Preserve Valid Empty**
   - If ORIGINAL was [فارغ] and REFINED is [فارغ] → field is truly empty
   - Do NOT fill empty fields with guessed values

## OUTPUT FORMAT (JSON only):
{{
  "final_fields": {{
    "field_name": {{
      "value": "final_value",
      "source": "original|refined|merged",
      "confidence": "high|medium|low",
      "needs_human_review": false,
      "review_reason": null,
      "notes": "Why this decision was made"
    }}
  }},
  "merge_quality_check": {{
    "duplicate_values_found": false,
    "year_as_value_found": false,
    "suspicious_fills": [],
    "quality_improved": true
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

Your PRIMARY job is to DETECT HALLUCINATIONS:
1. Value propagation (same value copied to multiple fields)
2. Year values in non-date fields
3. Missing Arabic content where expected
4. Suspiciously complete extractions

Your SECONDARY job is to identify fields needing re-examination.

Always output valid JSON. Be aggressive in flagging suspicious patterns - 
it's better to re-examine a good field than to miss a hallucinated one."""

REGION_SYSTEM_PROMPT = """You are an expert in Arabic form layouts and document structure.
Your role is to estimate where specific fields are located in form images.
Arabic forms are right-to-left, with labels on the right and values on the left.
Always output valid JSON with normalized bounding boxes."""

MERGE_SYSTEM_PROMPT = """You are an intelligent OCR result merger with ANTI-HALLUCINATION focus.

Your PRIMARY job is to REJECT hallucinated values:
1. Never accept a value that appears in multiple unrelated fields
2. Never accept a year (1400-1450) for non-date fields
3. Never fill empty fields with guessed values
4. Prefer the more specific/detailed value

Your SECONDARY job is to merge results intelligently.

When uncertain, flag for human review. Never guess.
Always output valid JSON."""


# =============================================================================
# VALIDATION SUMMARY PROMPT (for final quality check)
# =============================================================================

FINAL_VALIDATION_PROMPT = """Review this OCR extraction for quality issues.

## EXTRACTED DATA:
{extraction}

## QUALITY CHECKS TO PERFORM:

1. **Uniqueness Check**: Are all values unique (except truly identical fields)?
2. **Format Check**: Do values match expected formats?
   - ID: 10 digits, starts with 1 or 2
   - Phone: 10 digits, starts with 05
   - Date: day/month/year format
3. **Content Check**: Is there Arabic text where expected?
4. **Completeness Check**: Are critical fields present?

## OUTPUT FORMAT (JSON only):
{{
  "quality_score": <0-100>,
  "passed": <boolean>,
  "issues": [
    {{
      "field": "field_name",
      "issue": "description",
      "severity": "critical|error|warning|info"
    }}
  ],
  "recommendation": "pass|review|reject"
}}"""
