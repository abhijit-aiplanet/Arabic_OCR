"""
Agentic OCR Tools

Each tool is a capability the agent can use:
- read_region: Read text from a specific region
- zoom_and_read: Zoom into unclear area and read
- validate_field: Check if a value matches expected format
- search_document: Find a specific pattern in document
- reflect: Analyze current progress and decide next action
- store_correction: Learn from user feedback

These tools follow a standard interface for the agent to use.
"""

import base64
import io
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image, ImageEnhance, ImageFilter


class ToolType(Enum):
    """Types of tools available to the agent."""
    READ_REGION = "read_region"
    ZOOM_AND_READ = "zoom_and_read"
    VALIDATE_FIELD = "validate_field"
    SEARCH_PATTERN = "search_pattern"
    REFLECT = "reflect"
    EXTRACT_FULL = "extract_full"


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool: str
    success: bool
    result: Any
    confidence: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    reasoning: str = ""
    raw_output: str = ""


@dataclass
class Region:
    """A region of the document."""
    x: float  # 0-1 normalized
    y: float
    width: float
    height: float
    name: str = ""


# =============================================================================
# PREDEFINED REGIONS FOR SAUDI FORMS
# =============================================================================

FORM_REGIONS = {
    "header": Region(0.0, 0.0, 1.0, 0.12, "Header"),
    "general_data": Region(0.0, 0.12, 1.0, 0.25, "General Data"),
    "address": Region(0.0, 0.37, 1.0, 0.15, "Address"),
    "driving_license": Region(0.0, 0.52, 0.5, 0.15, "Driving License"),
    "vehicle": Region(0.5, 0.52, 0.5, 0.15, "Vehicle"),
    "footer": Region(0.0, 0.85, 1.0, 0.15, "Footer"),
}

# Field locations within regions (relative to region)
FIELD_LOCATIONS = {
    # General Data fields
    "اسم_المالك": Region(0.0, 0.0, 0.6, 0.25, "Owner Name"),
    "رقم_الهوية": Region(0.6, 0.0, 0.4, 0.25, "ID Number"),
    "مصدرها": Region(0.0, 0.25, 0.3, 0.25, "ID Source"),
    "تاريخها": Region(0.3, 0.25, 0.3, 0.25, "ID Date"),
    "تاريخ_الميلاد": Region(0.6, 0.25, 0.4, 0.25, "Birth Date"),
    
    # Address fields
    "المدينة": Region(0.0, 0.0, 0.25, 0.5, "City"),
    "الحي": Region(0.25, 0.0, 0.25, 0.5, "District"),
    "الشارع": Region(0.5, 0.0, 0.25, 0.5, "Street"),
    "جوال": Region(0.0, 0.5, 0.4, 0.5, "Mobile"),
    
    # Footer fields
    "اسم_مقدم_الطلب": Region(0.0, 0.0, 0.4, 0.5, "Applicant Name"),
    "صفته": Region(0.4, 0.0, 0.2, 0.5, "Capacity"),
    "التاريخ": Region(0.6, 0.0, 0.2, 0.5, "Date"),
    "توقيعه": Region(0.8, 0.0, 0.2, 0.5, "Signature"),
}


# =============================================================================
# VALIDATION PATTERNS
# =============================================================================

VALIDATION_PATTERNS = {
    "national_id": {
        "pattern": r"^[١٢12][٠-٩0-9]{9}$",
        "description": "10 digits starting with 1 or 2",
        "example": "1038367680",
    },
    "phone": {
        "pattern": r"^[٠0]5[٠-٩0-9]{8}$",
        "description": "10 digits starting with 05",
        "example": "0507477998",
    },
    "hijri_date": {
        "pattern": r"^[٠-٩0-9]{1,2}/[٠-٩0-9]{1,2}/[١1][٤4][٠-٩0-9]{2}$",
        "description": "DD/MM/14XX format",
        "example": "8/7/1404",
    },
    "arabic_name": {
        "pattern": r"^[\u0600-\u06FF\s]+$",
        "description": "Arabic characters only",
        "example": "عبدالله محمد",
    },
    "city": {
        "valid_values": ["جدة", "الرياض", "مكة", "المدينة", "الدمام", "الخبر", "تبوك", "أبها", "الطائف"],
        "description": "Saudi city name",
    },
}


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

class OCRTools:
    """Collection of tools for the OCR agent."""
    
    def __init__(self, azure_client):
        """Initialize with Azure client for vision tasks."""
        self.azure = azure_client
        self.call_count = 0
        self.tool_history: List[ToolResult] = []
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def crop_region(
        self, 
        image: Image.Image, 
        region: Region,
        upscale: int = 1
    ) -> Image.Image:
        """Crop a region from the image."""
        w, h = image.size
        left = int(region.x * w)
        top = int(region.y * h)
        right = int((region.x + region.width) * w)
        bottom = int((region.y + region.height) * h)
        
        cropped = image.crop((left, top, right, bottom))
        
        if upscale > 1:
            new_size = (cropped.width * upscale, cropped.height * upscale)
            cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)
        
        return cropped
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhance image for better OCR."""
        # Convert to grayscale
        gray = image.convert("L")
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)
        
        # Sharpen
        enhanced = enhanced.filter(ImageFilter.SHARPEN)
        
        return enhanced.convert("RGB")
    
    async def read_region(
        self,
        image: Image.Image,
        region_name: str,
        custom_prompt: Optional[str] = None,
    ) -> ToolResult:
        """
        Tool: read_region
        Read text from a specific region of the document.
        """
        self.call_count += 1
        
        # Get region definition
        region = FORM_REGIONS.get(region_name)
        if not region:
            return ToolResult(
                tool="read_region",
                success=False,
                result=None,
                reasoning=f"Unknown region: {region_name}",
            )
        
        # Crop and enhance
        cropped = self.crop_region(image, region, upscale=2)
        enhanced = self.enhance_image(cropped)
        
        # Build prompt
        prompt = custom_prompt or f"""اقرأ كل النص المكتوب بخط اليد في هذه الصورة.

هذا هو قسم: {region.name}

## التعليمات:
1. اقرأ كل حقل موجود
2. اكتب التسمية المطبوعة ثم القيمة المكتوبة
3. إذا كان الحقل فارغاً اكتب: ---
4. إذا لم تستطع القراءة: ؟[تخمينك]

## التنسيق:
اسم_الحقل: القيمة [HIGH/MEDIUM/LOW]

ابدأ:"""

        # Call Azure
        image_b64 = self.image_to_base64(enhanced)
        
        try:
            result = await self.azure.extract_section(image_b64, prompt)
            
            tool_result = ToolResult(
                tool="read_region",
                success=result.success,
                result=self._parse_fields(result.text) if result.success else None,
                confidence="HIGH" if result.success else "LOW",
                reasoning=f"Read {region.name} region",
                raw_output=result.text if result.success else result.error,
            )
        except Exception as e:
            tool_result = ToolResult(
                tool="read_region",
                success=False,
                result=None,
                reasoning=f"Error: {str(e)}",
            )
        
        self.tool_history.append(tool_result)
        return tool_result
    
    async def zoom_and_read(
        self,
        image: Image.Image,
        field_name: str,
        region_name: str,
        zoom_level: int = 3,
    ) -> ToolResult:
        """
        Tool: zoom_and_read
        Zoom into a specific field area and read carefully.
        """
        self.call_count += 1
        
        # Get region
        region = FORM_REGIONS.get(region_name)
        field_region = FIELD_LOCATIONS.get(field_name)
        
        if not region:
            return ToolResult(
                tool="zoom_and_read",
                success=False,
                result=None,
                reasoning=f"Unknown region: {region_name}",
            )
        
        # Calculate absolute field position
        if field_region:
            abs_region = Region(
                x=region.x + field_region.x * region.width,
                y=region.y + field_region.y * region.height,
                width=field_region.width * region.width,
                height=field_region.height * region.height,
                name=field_name,
            )
        else:
            # Use full region if field location unknown
            abs_region = region
        
        # Crop and zoom
        cropped = self.crop_region(image, abs_region, upscale=zoom_level)
        enhanced = self.enhance_image(cropped)
        
        # Field-specific prompt
        prompt = f"""هذه صورة مكبرة لحقل: {field_name}

## مهمتك:
اقرأ القيمة المكتوبة بخط اليد فقط.

## قواعد صارمة:
- اقرأ كل حرف/رقم على حدة
- إذا فارغ: ---
- إذا غير واضح: ؟[تخمين]

## للأرقام:
اقرأها رقماً رقماً: ٠-٥-٠-٧-٤-٧-٧-٩-٩-٨

## القيمة:"""

        image_b64 = self.image_to_base64(enhanced)
        
        try:
            result = await self.azure.extract_field(image_b64, field_name)
            
            tool_result = ToolResult(
                tool="zoom_and_read",
                success=True,
                result={
                    "field": field_name,
                    "value": result.value,
                    "confidence": result.confidence,
                },
                confidence=result.confidence,
                reasoning=f"Zoomed {zoom_level}x on {field_name}",
                raw_output=result.value,
            )
        except Exception as e:
            tool_result = ToolResult(
                tool="zoom_and_read",
                success=False,
                result=None,
                reasoning=f"Error: {str(e)}",
            )
        
        self.tool_history.append(tool_result)
        return tool_result
    
    async def extract_full_document(
        self,
        image: Image.Image,
    ) -> ToolResult:
        """
        Tool: extract_full
        Extract all fields from the entire document at once.
        Useful as initial pass before targeted extraction.
        """
        self.call_count += 1
        
        enhanced = self.enhance_image(image)
        
        prompt = """أنت خبير في قراءة النماذج الحكومية السعودية المكتوبة بخط اليد.

## مهمتك:
استخرج كل الحقول المكتوبة بخط اليد من هذه الصورة.

## التعليمات:
1. ابحث عن كل تسمية مطبوعة
2. اقرأ القيمة المكتوبة بجانبها
3. إذا كان الحقل فارغاً: ---
4. إذا لم تستطع القراءة بوضوح: ؟[تخمينك]

## تعليمات خاصة للأرقام:
- رقم الهوية: ١٠ أرقام، اقرأ رقماً رقماً
- الجوال: ١٠ أرقام تبدأ بـ ٠٥
- التاريخ: يوم/شهر/سنة

## تعليمات خاصة للأسماء:
- اقرأ كل اسم حرفاً حرفاً
- الأسماء العربية تكون من عدة مقاطع

## التنسيق:
الحقل: القيمة [HIGH/MEDIUM/LOW]

ابدأ الاستخراج من أعلى الصفحة إلى أسفلها:"""

        image_b64 = self.image_to_base64(enhanced)
        
        try:
            result = await self.azure.extract_section(image_b64, prompt)
            
            fields = self._parse_fields(result.text) if result.success else {}
            
            tool_result = ToolResult(
                tool="extract_full",
                success=result.success,
                result=fields,
                confidence="MEDIUM",  # Full doc extraction is less precise
                reasoning="Full document extraction pass",
                raw_output=result.text if result.success else result.error,
            )
        except Exception as e:
            tool_result = ToolResult(
                tool="extract_full",
                success=False,
                result=None,
                reasoning=f"Error: {str(e)}",
            )
        
        self.tool_history.append(tool_result)
        return tool_result
    
    def validate_field(
        self,
        field_name: str,
        value: str,
    ) -> ToolResult:
        """
        Tool: validate_field
        Check if a field value matches expected format.
        """
        self.call_count += 1
        
        # Determine field type
        field_type = None
        if "هوية" in field_name or "رقم" in field_name:
            field_type = "national_id"
        elif "جوال" in field_name or "هاتف" in field_name:
            field_type = "phone"
        elif "تاريخ" in field_name:
            field_type = "hijri_date"
        elif "اسم" in field_name:
            field_type = "arabic_name"
        elif "مدينة" in field_name or "المدينة" in field_name:
            field_type = "city"
        
        if not field_type:
            return ToolResult(
                tool="validate_field",
                success=True,
                result={"valid": True, "reason": "No validation rule"},
                confidence="MEDIUM",
                reasoning="No specific validation for this field type",
            )
        
        validation = VALIDATION_PATTERNS.get(field_type, {})
        
        # Check pattern
        pattern = validation.get("pattern")
        valid_values = validation.get("valid_values")
        
        is_valid = False
        reason = ""
        
        if value in ["---", "", "[فارغ]"]:
            is_valid = True
            reason = "Empty field (valid)"
        elif pattern:
            # Normalize Arabic-Indic numerals
            normalized = value
            arabic_to_western = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
            normalized = normalized.translate(arabic_to_western)
            
            if re.match(pattern, normalized) or re.match(pattern, value):
                is_valid = True
                reason = f"Matches {field_type} pattern"
            else:
                is_valid = False
                reason = f"Does not match {validation.get('description', 'expected format')}"
        elif valid_values:
            if value in valid_values:
                is_valid = True
                reason = "Valid city name"
            else:
                is_valid = False
                reason = f"Unknown value. Expected one of: {', '.join(valid_values[:5])}"
        
        tool_result = ToolResult(
            tool="validate_field",
            success=True,
            result={
                "field": field_name,
                "value": value,
                "valid": is_valid,
                "reason": reason,
                "expected": validation.get("description", ""),
            },
            confidence="HIGH",
            reasoning=reason,
        )
        
        self.tool_history.append(tool_result)
        return tool_result
    
    async def reflect(
        self,
        current_fields: Dict[str, str],
        current_confidence: Dict[str, str],
    ) -> ToolResult:
        """
        Tool: reflect
        Analyze current extraction progress and decide next actions.
        """
        self.call_count += 1
        
        # Analyze what we have
        total = len(current_fields)
        high_conf = sum(1 for c in current_confidence.values() if c == "HIGH")
        low_conf = sum(1 for c in current_confidence.values() if c == "LOW")
        empty = sum(1 for v in current_fields.values() if v in ["---", ""])
        
        # Find problematic fields
        needs_attention = []
        for field, value in current_fields.items():
            conf = current_confidence.get(field, "MEDIUM")
            if conf == "LOW" and value not in ["---", ""]:
                needs_attention.append(field)
            if "؟" in value:
                needs_attention.append(field)
        
        # Check for suspicious patterns
        issues = []
        values = list(current_fields.values())
        
        # Duplicate detection
        for v in set(values):
            if v not in ["---", "", "[توقيع]", "[ختم]"] and values.count(v) > 1:
                issues.append(f"Duplicate value '{v}' in multiple fields")
        
        # Missing critical fields
        critical_fields = ["اسم_المالك", "رقم_الهوية", "جوال"]
        for cf in critical_fields:
            if cf not in current_fields or current_fields.get(cf) in ["---", ""]:
                issues.append(f"Missing critical field: {cf}")
        
        # Build recommendation
        recommendation = ""
        if not current_fields:
            recommendation = "No fields extracted. Try extract_full_document first."
        elif low_conf > total * 0.5:
            recommendation = f"Too many low-confidence fields ({low_conf}/{total}). Use zoom_and_read on: {', '.join(needs_attention[:3])}"
        elif needs_attention:
            recommendation = f"Refine unclear fields with zoom_and_read: {', '.join(needs_attention[:3])}"
        elif issues:
            recommendation = f"Issues found: {'; '.join(issues[:2])}"
        else:
            recommendation = "Extraction looks good. Proceed to validation."
        
        tool_result = ToolResult(
            tool="reflect",
            success=True,
            result={
                "total_fields": total,
                "high_confidence": high_conf,
                "low_confidence": low_conf,
                "empty_fields": empty,
                "needs_attention": needs_attention,
                "issues": issues,
                "recommendation": recommendation,
            },
            confidence="HIGH",
            reasoning=recommendation,
        )
        
        self.tool_history.append(tool_result)
        return tool_result
    
    def _parse_fields(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Parse extraction output into structured fields."""
        fields = {}
        
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            
            field_name = parts[0].strip()
            rest = parts[1].strip()
            
            # Extract confidence
            conf = "MEDIUM"
            if "[HIGH]" in rest.upper():
                conf = "HIGH"
                rest = re.sub(r'\[HIGH\]', '', rest, flags=re.IGNORECASE).strip()
            elif "[MEDIUM]" in rest.upper():
                conf = "MEDIUM"
                rest = re.sub(r'\[MEDIUM\]', '', rest, flags=re.IGNORECASE).strip()
            elif "[LOW]" in rest.upper():
                conf = "LOW"
                rest = re.sub(r'\[LOW\]', '', rest, flags=re.IGNORECASE).strip()
            
            value = rest.strip()
            
            # Force LOW for empty values
            if value in ["---", "[فارغ]", "[EMPTY]", "غير موجود", ""]:
                conf = "LOW"
                value = "---"
            
            fields[field_name] = {
                "value": value,
                "confidence": conf,
            }
        
        return fields
    
    def get_tool_summary(self) -> Dict[str, Any]:
        """Get summary of tool usage."""
        return {
            "total_calls": self.call_count,
            "history": [
                {
                    "tool": r.tool,
                    "success": r.success,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                }
                for r in self.tool_history
            ]
        }
