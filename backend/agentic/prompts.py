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

## قاعدة أساسية:
اقرأ فقط التسميات (labels) الموجودة فعلياً في الصورة.
لا تخترع حقولاً غير موجودة.

الحقول الشائعة في هذا القسم:
- نوع النشاط - نص عربي
- مدينة مزاولة النشاط - اسم مدينة
- تاريخ بدء النشاط - يوم/شهر/سنة
- تاريخ انتهاء الترخيص - يوم/شهر/سنة

## التعليمات:
1. اقرأ التسمية المطبوعة أولاً
2. ثم اقرأ القيمة المكتوبة بخط اليد بجانبها
3. إذا كان الحقل فارغاً (نقاط فقط): --- [LOW]

## التنسيق:
التسمية_كما_في_الصورة: القيمة [الثقة]

## قواعد الثقة:
- [HIGH] = واضح تماماً، متأكد 100%
- [MEDIUM] = مقروء لكن قد يكون خطأ
- [LOW] = غير واضح أو فارغ

{learning_context}

ابدأ الاستخراج:""",

    "general_data": """هذا هو قسم البيانات العامة (بيانات عامة).

## قاعدة أساسية للأرقام:
اقرأ كل رقم على حدة، رقماً رقماً.
لا تخمن - إذا لم تستطع قراءة رقم، ضع ؟ مكانه.

الحقول الشائعة:
- اسم المالك - اسم عربي كامل
- رقم الهوية - ١٠ أرقام (تبدأ بـ ١ أو ٢)
- مصدرها - مدينة
- تاريخها - يوم/شهر/سنة
- الحالة الاجتماعية - متزوج/أعزب/مطلق/أرمل
- عدد من يعولهم - رقم
- المؤهل - مستوى تعليمي
- تاريخ الميلاد - يوم/شهر/سنة

## تعليمات خاصة للأرقام:
مثال رقم الهوية: اقرأ ١-٠-٣-٨-٣-٦-٧-٦-٨-٠
إذا لم تتأكد من رقم معين: ١٠٣٨٣؟٧٦٨٠

## تعليمات خاصة للتواريخ:
اقرأ بالترتيب: يوم / شهر / سنة
مثال: ٨/٧/١٤٠٤

## التنسيق:
الحقل: القيمة [الثقة]

## للحقول الفارغة:
الحقل: --- [LOW]

{learning_context}

ابدأ الاستخراج:""",

    "address": """هذا هو قسم العنوان (العنوان).

## قاعدة أساسية للجوال:
اقرأ كل رقم على حدة: ٠-٥-٠-٧-٤-٧-٧-٩-٩-٨
إذا لم تتأكد من رقم: ٠٥٠٧٤؟٧٩٩٨

الحقول الشائعة:
- المدينة - اسم مدينة
- الحي - اسم حي
- الشارع - اسم أو رقم
- رقم المبنى - رقم
- جوال - ١٠ أرقام تبدأ بـ ٠٥
- البريد الإلكتروني - email
- فاكس - رقم

## ملاحظة مهمة:
معظم حقول العنوان تكون فارغة عادةً.
إذا كان الحقل فارغاً: --- [LOW]
لا تخترع قيماً!

{learning_context}

## التنسيق:
الحقل: القيمة [الثقة]

ابدأ الاستخراج:""",

    "driving_license": """هذا هو قسم رخصة القيادة (بيانات رخصة القيادة).

## قاعدة أساسية:
إذا كان القسم كله فارغاً أو غير موجود:
كل الحقول: --- [LOW]

الحقول الشائعة:
- رقمها - ١٠ أرقام
- تاريخ الإصدار - يوم/شهر/سنة
- تاريخ الانتهاء - يوم/شهر/سنة
- مصدرها - مدينة

## تعليمات خاصة:
اقرأ الأرقام رقماً رقماً.
اقرأ التواريخ كاملة: يوم/شهر/سنة

{learning_context}

## التنسيق:
الحقل: القيمة [الثقة]

## للحقول الفارغة:
الحقل: --- [LOW]

ابدأ الاستخراج:""",

    "vehicle": """هذا هو قسم المركبة (بيانات المركبة).

## قاعدة أساسية مهمة جداً:
قسم المركبة غالباً يكون فارغاً تماماً!
اقرأ فقط التسميات الموجودة فعلياً في الصورة.

التسميات المحتملة (قد تختلف):
- طراز المركبة أو نوع المركبة
- الشركة الصانعة أو الموديل
- سعة المحرك أو عدد الأسطوانات
- اللون
- رقم اللوحة
- رقم الهيكل
- سنة الصنع

## تعليمات صارمة:
1. اقرأ التسمية المطبوعة كما هي في الصورة
2. إذا لم يكن هناك كتابة يدوية: --- [LOW]
3. لا تخترع حقولاً غير موجودة!

{learning_context}

## التنسيق:
التسمية_من_الصورة: القيمة [الثقة]

## إذا كان القسم فارغاً بالكامل:
المركبة: غير موجود [LOW]

ابدأ الاستخراج:""",

    "footer": """هذا هو قسم التذييل (Footer).

## قاعدة أساسية:
اقرأ فقط ما هو مكتوب فعلاً.
التوقيع والختم ليسا نصاً - فقط علامات.

الحقول الشائعة:
- اسم مقدم الطلب - اسم عربي
- صفته - مالك / وكيل / مندوب
- توقيعه - [توقيع] إذا موجود
- التاريخ - يوم/شهر/سنة
- الختم - [ختم] إذا موجود

## تعليمات خاصة للتاريخ:
انظر للتاريخ في أسفل الصفحة.
اقرأه كاملاً: يوم/شهر/سنة

## تعليمات خاصة للاسم:
اقرأ الاسم حرفاً حرفاً.
إذا لم تستطع قراءة جزء: ؟[الجزء الواضح]

{learning_context}

## التنسيق:
الحقل: القيمة [الثقة]

ابدأ الاستخراج:""",
}


# =============================================================================
# FIELD-LEVEL EXTRACTION PROMPTS
# =============================================================================

FIELD_ZOOM_PROMPT = """هذه صورة مقصوصة ومكبرة لحقل "{field_name}".

## تعليمات صارمة:
اقرأ كل حرف/رقم على حدة.
إذا لم تستطع قراءة حرف، ضع ؟ مكانه.

## القواعد:
- إذا فارغ (نقاط/خطوط فقط): ---
- إذا غير واضح جزئياً: القراءة مع ؟ للأجزاء غير الواضحة
- إذا واضح: القيمة كاملة

{format_hint}

## مثال للأرقام:
إذا رأيت ٠٥٠٧٤٧٧٩٩٨ اكتب: ٠٥٠٧٤٧٧٩٩٨
إذا رأيت رقماً غير واضح: ٠٥٠٧؟٧٧٩٩٨

## القيمة:"""


# Format hints for different field types - more specific
FIELD_FORMAT_HINTS = {
    "national_id": "١٠ أرقام - اقرأ رقماً رقماً: مثل ١-٠-٣-٨-٣-٦-٧-٦-٨-٠",
    "phone": "١٠ أرقام تبدأ بـ ٠٥ - اقرأ رقماً رقماً",
    "date": "يوم/شهر/سنة - اقرأ كل جزء: مثل ٨/٧/١٤٠٤",
    "arabic_name": "اسم عربي - اقرأ حرفاً حرفاً",
    "city": "اسم مدينة سعودية",
    "number": "رقم أو أرقام",
    "plate": "حروف عربية وأرقام - اقرأ كل حرف ورقم",
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
