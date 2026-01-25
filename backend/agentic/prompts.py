"""
Surgical OCR Prompts for Arabic Government Forms

Section-specific prompts designed for maximum accuracy.
Each section gets a tailored prompt with:
- Expected fields for that section
- Format hints for each field type
- Clear instructions for empty/unclear handling

Key principles:
1. Simple prompts that ask for ONE thing
2. Section-specific context for better accuracy
3. Clear markers for empty/unclear fields
4. Learning context integration
"""

from typing import Dict, Optional


# =============================================================================
# DOCUMENT ANALYSIS PROMPT
# =============================================================================

DOCUMENT_ANALYSIS_PROMPT = """Analyze this Arabic government form image.

Describe:
1. What type of document is this?
2. How many sections do you see?
3. Which sections have handwritten content?
4. Which sections appear mostly empty?
5. Rate the handwriting clarity: واضح (clear) / متوسط (moderate) / غير واضح (poor)

Output as JSON:
{
  "document_type": "string",
  "sections_count": number,
  "sections_with_content": ["section names"],
  "sections_empty": ["section names"],
  "handwriting_quality": "واضح/متوسط/غير واضح",
  "notes": "any important observations"
}"""


# =============================================================================
# SECTION-SPECIFIC PROMPTS
# =============================================================================

SECTION_PROMPTS = {
    "header": """هذا هو قسم الرأس (HEADER) من نموذج حكومي سعودي.

الحقول المتوقعة:
- نوع النشاط (Activity Type) - نص عربي
- مدينة مزاولة النشاط (City of Activity) - اسم مدينة
- تاريخ بدء النشاط (Start Date) - يوم/شهر/سنة هجري
- تاريخ انتهاء الترخيص (License End Date) - يوم/شهر/سنة هجري

## التعليمات:
اقرأ القيم المكتوبة بخط اليد فقط (تجاهل النص المطبوع).

## التنسيق:
اسم_الحقل: القيمة [الثقة]

## قواعد الثقة:
- [HIGH] = واضح ومقروء
- [MEDIUM] = مقروء لكن غير مؤكد
- [LOW] = غير واضح، أفضل تخمين

## للحقول الفارغة:
اسم_الحقل: --- [EMPTY]

## للحقول غير الواضحة:
اسم_الحقل: ؟[أفضل تخمين] [LOW]

{learning_context}

ابدأ الاستخراج:""",

    "general_data": """هذا هو قسم البيانات العامة (بيانات عامة) من النموذج.

الحقول المتوقعة:
- اسم المالك (Owner Name) - اسم عربي كامل
- رقم الهوية (ID Number) - ١٠ أرقام تبدأ بـ ١ أو ٢
- مصدرها (ID Source) - اسم مدينة
- تاريخها (ID Date) - يوم/شهر/سنة هجري
- الحالة الاجتماعية (Marital Status) - متزوج/أعزب/مطلق/أرمل
- عدد من يعولهم (Dependents Count) - رقم
- المؤهل (Qualification) - مستوى تعليمي
- تاريخ الميلاد (Birth Date) - يوم/شهر/سنة هجري

## تنسيقات مهمة:
- رقم الهوية: ١٠ أرقام مثل ١٠٣٨٣٦٧٦٨٠
- التواريخ: مثل ٨/٧/١٤٠٤ أو 8/7/1404

## التعليمات:
اقرأ كل قيمة مكتوبة بخط اليد بعناية.
لا تخترع قيماً - إذا لم تستطع القراءة، اكتب ؟[تخمين]

{learning_context}

## التنسيق:
اسم_الحقل: القيمة [الثقة]

ابدأ الاستخراج:""",

    "address": """هذا هو قسم العنوان (العنوان) من النموذج.

الحقول المتوقعة:
- المدينة (City) - اسم مدينة سعودية
- الحي (District) - اسم حي
- الشارع (Street) - اسم شارع
- رقم المبنى (Building Number) - رقم
- جوال (Mobile) - ١٠ أرقام تبدأ بـ ٠٥
- البريد الإلكتروني (Email) - قد يكون فارغاً
- فاكس (Fax) - قد يكون فارغاً

## تنسيقات مهمة:
- رقم الجوال: ١٠ أرقام مثل ٠٥٠٧٤٧٧٩٩٨

## ملاحظة:
كثير من حقول العنوان قد تكون فارغة (---).
هذا طبيعي في النماذج السعودية.

{learning_context}

## التنسيق:
اسم_الحقل: القيمة [الثقة]

ابدأ الاستخراج:""",

    "driving_license": """هذا هو قسم رخصة القيادة (بيانات رخصة القيادة) من النموذج.

الحقول المتوقعة:
- رقمها (License Number) - ١٠ أرقام
- تاريخ الإصدار (Issue Date) - يوم/شهر/سنة هجري
- تاريخ الانتهاء (Expiry Date) - يوم/شهر/سنة هجري
- مصدرها (Source) - اسم مدينة

## تنسيقات مهمة:
- رقم الرخصة: ١٠ أرقام مثل ١٠٣٨٣٢٦٦٨٠
- التواريخ: مثل ٢٥/٧/١٤٣٧

{learning_context}

## التنسيق:
اسم_الحقل: القيمة [الثقة]

ابدأ الاستخراج:""",

    "vehicle": """هذا هو قسم المركبة (بيانات المركبة) من النموذج.

الحقول المتوقعة:
- نوع المركبة (Vehicle Type) - نص
- الموديل (Model) - سنة أو اسم
- اللون (Color) - لون بالعربي
- رقم اللوحة (Plate Number) - حروف وأرقام
- رقم الهيكل (Chassis Number) - أرقام وحروف
- سنة الصنع (Year) - ٤ أرقام
- عدد الأسطوانات (Cylinders) - رقم

## ملاحظة مهمة:
قسم المركبة غالباً يكون فارغاً بالكامل أو جزئياً.
لا تخترع قيماً - إذا كان الحقل فارغاً، اكتب ---

{learning_context}

## التنسيق:
اسم_الحقل: القيمة [الثقة]

ابدأ الاستخراج:""",

    "footer": """هذا هو قسم التذييل (Footer) من النموذج.

الحقول المتوقعة:
- اسم مقدم الطلب (Applicant Name) - اسم عربي
- صفته (Capacity) - مثل: مالك، وكيل
- توقيعه (Signature) - اكتب [توقيع] إذا موجود
- التاريخ (Date) - يوم/شهر/سنة
- الختم (Stamp) - اكتب [ختم] إذا موجود

## ملاحظة:
- التوقيع: إذا رأيت خط توقيع، اكتب [توقيع]
- الختم: إذا رأيت ختماً، اكتب [ختم]

{learning_context}

## التنسيق:
اسم_الحقل: القيمة [الثقة]

ابدأ الاستخراج:""",
}


# =============================================================================
# FIELD-LEVEL EXTRACTION PROMPTS
# =============================================================================

FIELD_ZOOM_PROMPT = """هذه صورة مقصوصة ومكبرة لحقل "{field_name}".

اقرأ القيمة المكتوبة بخط اليد فقط.

## القواعد:
- إذا فارغ (نقاط/خطوط فقط): اكتب "---"
- إذا غير واضح: اكتب "؟[أفضل تخمين]"
- خلاف ذلك: اكتب القيمة فقط

{format_hint}

## القيمة:"""


# Format hints for different field types
FIELD_FORMAT_HINTS = {
    "national_id": "التنسيق المتوقع: ١٠ أرقام تبدأ بـ ١ أو ٢",
    "phone": "التنسيق المتوقع: ١٠ أرقام تبدأ بـ ٠٥",
    "date": "التنسيق المتوقع: يوم/شهر/سنة (هجري)",
    "arabic_name": "التنسيق المتوقع: اسم عربي كامل",
    "city": "التنسيق المتوقع: اسم مدينة سعودية",
    "number": "التنسيق المتوقع: رقم أو أرقام",
    "plate": "التنسيق المتوقع: حروف عربية وأرقام",
}

# Field name to type mapping
FIELD_TYPE_MAP = {
    "رقم الهوية": "national_id",
    "رقم بطاقة الأحوال": "national_id",
    "جوال": "phone",
    "رقم الجوال": "phone",
    "تاريخ الميلاد": "date",
    "تاريخها": "date",
    "تاريخ الإصدار": "date",
    "تاريخ الانتهاء": "date",
    "التاريخ": "date",
    "اسم المالك": "arabic_name",
    "اسم مقدم الطلب": "arabic_name",
    "المدينة": "city",
    "مدينة مزاولة النشاط": "city",
    "مصدرها": "city",
    "عدد من يعولهم": "number",
    "رقم اللوحة": "plate",
    "رقمها": "national_id",  # License number same format as ID
}


# =============================================================================
# SELF-CRITIQUE PROMPT
# =============================================================================

SELF_CRITIQUE_PROMPT = """راجع هذا الاستخراج للبحث عن أخطاء:

## البيانات المستخرجة:
{extraction}

## تحقق من:
1. نفس القيمة تظهر في حقول متعددة غير مرتبطة (علامة على الهلوسة)
2. القيم لا تطابق تنسيق الحقل:
   - رقم الهوية يجب أن يكون ١٠ أرقام تبدأ بـ ١ أو ٢
   - رقم الجوال يجب أن يكون ١٠ أرقام تبدأ بـ ٠٥
   - التواريخ يجب أن تكون بتنسيق يوم/شهر/سنة
3. حقول الأسماء العربية تحتوي على أرقام فقط
4. نموذج مكتمل بشكل مشبوه (النماذج الحقيقية لها حقول فارغة)
5. قيمة سنة (١٤٠٠-١٤٥٠) في حقول غير التاريخ

## المخرجات (JSON فقط):
{{
  "has_issues": true/false,
  "issues": [
    {{"field": "اسم_الحقل", "issue": "وصف المشكلة", "severity": "high/medium/low"}}
  ],
  "fields_to_recheck": ["حقل١", "حقل٢"],
  "overall_confidence": "high/medium/low"
}}"""


# =============================================================================
# MERGE PROMPT
# =============================================================================

MERGE_PROMPT = """ادمج نتائج الاستخراج من مرورين مختلفين.

## الاستخراج الأصلي:
{original_extraction}

## القيم المُحسّنة:
{refined_values}

## القواعد:
1. إذا كانت القيمة المُحسّنة أكثر تحديداً → استخدم المُحسّنة
2. إذا تطابقت القيم → استخدم الأصلية مع ثقة HIGH
3. إذا كانت المُحسّنة "---" والأصلية لها قيمة → أبقِ الأصلية
4. إذا كلاهما غير واضح → علّم للمراجعة البشرية
5. لا تخترع قيماً أبداً

## المخرجات (JSON فقط):
{{
  "final_fields": {{
    "اسم_الحقل": {{
      "value": "القيمة النهائية",
      "source": "original|refined|merged",
      "confidence": "HIGH|MEDIUM|LOW",
      "needs_review": false
    }}
  }},
  "quality_check": {{
    "duplicates_found": false,
    "suspicious_values": [],
    "quality_improved": true
  }},
  "fields_still_uncertain": []
}}"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_section_prompt(
    section_type: str,
    learning_context: str = "",
) -> str:
    """
    Get the prompt for a specific section.
    
    Args:
        section_type: Type of section (header, general_data, etc.)
        learning_context: Optional few-shot examples from past corrections
        
    Returns:
        Formatted section prompt
    """
    prompt_template = SECTION_PROMPTS.get(section_type, SECTION_PROMPTS["general_data"])
    return prompt_template.format(learning_context=learning_context)


def get_field_prompt(
    field_name: str,
    field_type: Optional[str] = None,
) -> str:
    """
    Get the prompt for field-level extraction (zoom-in).
    
    Args:
        field_name: Name of the field
        field_type: Optional field type override
        
    Returns:
        Formatted field extraction prompt
    """
    # Determine field type
    if not field_type:
        field_type = FIELD_TYPE_MAP.get(field_name)
        
        # Try partial match
        if not field_type:
            for key, ftype in FIELD_TYPE_MAP.items():
                if key in field_name or field_name in key:
                    field_type = ftype
                    break
    
    # Get format hint
    format_hint = ""
    if field_type and field_type in FIELD_FORMAT_HINTS:
        format_hint = FIELD_FORMAT_HINTS[field_type]
    
    return FIELD_ZOOM_PROMPT.format(
        field_name=field_name,
        format_hint=format_hint,
    )


def get_format_hint(field_name: str) -> str:
    """Get format hint for a field."""
    field_type = FIELD_TYPE_MAP.get(field_name)
    
    if not field_type:
        for key, ftype in FIELD_TYPE_MAP.items():
            if key in field_name or field_name in key:
                field_type = ftype
                break
    
    return FIELD_FORMAT_HINTS.get(field_type, "")


def get_critique_prompt(extraction: str) -> str:
    """Get self-critique prompt with extraction data."""
    return SELF_CRITIQUE_PROMPT.format(extraction=extraction)


def get_merge_prompt(
    original_extraction: str,
    refined_values: str,
) -> str:
    """Get merge prompt with both extraction passes."""
    return MERGE_PROMPT.format(
        original_extraction=original_extraction,
        refined_values=refined_values,
    )


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

# Keep old function names for backward compatibility
def get_content_hint(field_name: str) -> str:
    """Legacy function - returns format hint for field."""
    return get_format_hint(field_name)


def build_field_reextract_prompt(
    field_name: str,
    previous_value: str = "",
    content_hint: str = None,
) -> str:
    """Legacy function - returns field extraction prompt."""
    return get_field_prompt(field_name)


# Legacy prompts for backward compatibility
INITIAL_OCR_PROMPT = SECTION_PROMPTS["general_data"]
FIELD_REEXTRACT_PROMPT = """ما المكتوب هنا؟

إذا فارغ: ---
إذا غير واضح: ؟

اكتب القيمة فقط:"""

ANALYSIS_PROMPT = SELF_CRITIQUE_PROMPT
ANALYSIS_SYSTEM_PROMPT = "You are an OCR quality analyst. Output valid JSON only."
REGION_SYSTEM_PROMPT = "You estimate field locations on Arabic forms. Output valid JSON."
MERGE_SYSTEM_PROMPT = "You merge OCR results intelligently. Never invent values. Output valid JSON."
REGION_ESTIMATION_PROMPT = ""  # No longer used - handled by image_processor
