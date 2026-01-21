"""
Validation Framework for Saudi Government Documents (FLAG-ONLY MODE)

Provides field-specific validation for common Saudi document fields:
- National ID numbers
- Mobile phone numbers
- Hijri dates
- Vehicle plates
- Commercial registration numbers
- IBAN numbers

IMPORTANT: FLAG-ONLY MODE
=========================
This module ONLY FLAGS validation issues - it NEVER modifies the original OCR output.
Validation results are returned as metadata (is_valid, is_suspicious, reason) that
can be used by the UI to highlight fields for human review.

The original extracted values are ALWAYS preserved unchanged.

Key Features:
- Pattern-based validation
- Suspicious value detection (likely hallucinations)
- Format checking WITHOUT modification

Output Usage:
- is_valid: True if value matches expected pattern
- is_invalid: True if value doesn't match (possible OCR error)
- is_suspicious: True if value looks hallucinated
- normalized_value: ONLY for display purposes, never replaces original
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
import re


# =============================================================================
# VALIDATION RESULT
# =============================================================================

@dataclass
class ValidationResult:
    """Result of field validation."""
    is_valid: bool = False
    is_invalid: bool = False
    is_suspicious: bool = False
    is_unchecked: bool = True
    reason: Optional[str] = None
    normalized_value: Optional[str] = None
    
    @classmethod
    def valid(cls, normalized_value: Optional[str] = None, reason: str = "Matches expected pattern"):
        return cls(is_valid=True, is_unchecked=False, reason=reason, normalized_value=normalized_value)
    
    @classmethod
    def invalid(cls, reason: str):
        return cls(is_invalid=True, is_unchecked=False, reason=reason)
    
    @classmethod
    def suspicious(cls, reason: str):
        return cls(is_suspicious=True, is_unchecked=False, reason=reason)
    
    @classmethod
    def unchecked(cls, reason: str = "No validator for this field type"):
        return cls(is_unchecked=True, reason=reason)


# =============================================================================
# ARABIC-WESTERN NUMERAL CONVERSION
# =============================================================================

ARABIC_DIGITS = '٠١٢٣٤٥٦٧٨٩'
WESTERN_DIGITS = '0123456789'

def arabic_to_western(text: str) -> str:
    """Convert Arabic numerals (٠١٢...) to Western (012...)."""
    result = text
    for ar, we in zip(ARABIC_DIGITS, WESTERN_DIGITS):
        result = result.replace(ar, we)
    return result

def western_to_arabic(text: str) -> str:
    """Convert Western numerals (012...) to Arabic (٠١٢...)."""
    result = text
    for ar, we in zip(ARABIC_DIGITS, WESTERN_DIGITS):
        result = result.replace(we, ar)
    return result

def normalize_digits(text: str) -> str:
    """Normalize text to have consistent (Western) digits for validation."""
    return arabic_to_western(text)


# =============================================================================
# FIELD NAME DETECTION
# =============================================================================

# Keywords that indicate field types (in Arabic)
FIELD_TYPE_KEYWORDS = {
    "national_id": [
        "هوية", "الهوية", "رقم الهوية", "هوية وطنية", "رقم الهوية الوطنية",
        "بطاقة", "رقم البطاقة", "الإقامة", "رقم الإقامة"
    ],
    "mobile": [
        "جوال", "رقم الجوال", "هاتف", "رقم الهاتف", "موبايل", "تلفون",
        "رقم التواصل", "للتواصل"
    ],
    "hijri_date": [
        "تاريخ", "التاريخ", "تاريخ الميلاد", "تاريخ الإصدار", "تاريخ الانتهاء",
        "صلاحية", "انتهاء الصلاحية", "يوم", "شهر", "سنة"
    ],
    "vehicle_plate": [
        "لوحة", "رقم اللوحة", "لوحة السيارة", "لوحة المركبة"
    ],
    "iban": [
        "آيبان", "iban", "الحساب البنكي", "رقم الحساب"
    ],
    "commercial_registration": [
        "سجل تجاري", "رقم السجل", "السجل التجاري"
    ],
    "name": [
        "اسم", "الاسم", "اسم المالك", "صاحب", "الاسم الكامل",
        "اسم صاحب", "مقدم الطلب"
    ],
    "city": [
        "مدينة", "المدينة", "بلد", "البلد"
    ],
    "district": [
        "حي", "الحي", "منطقة"
    ],
}

def detect_field_type(field_name: str) -> Optional[str]:
    """Detect the type of field based on its name."""
    field_name_normalized = field_name.strip().lower()
    
    for field_type, keywords in FIELD_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in field_name_normalized or field_name_normalized in keyword:
                return field_type
    
    return None


# =============================================================================
# VALIDATION PATTERNS
# =============================================================================

class SaudiDocumentValidators:
    """
    Validation rules for Saudi government documents.
    
    Covers:
    - National ID (citizen starts with 1, resident starts with 2)
    - Mobile phones (05xxxxxxxx format)
    - Hijri dates (various formats)
    - Vehicle plates (Arabic letters + numbers)
    - IBAN (SA followed by 22 digits)
    - Commercial Registration (10 digits)
    """
    
    # Validation patterns
    PATTERNS = {
        # National ID: 10 digits, starts with 1 (citizen) or 2 (resident)
        "national_id": r"^[12]\d{9}$",
        
        # Mobile: 05xxxxxxxx (10 digits) or +9665xxxxxxxx (12 digits)
        "mobile": r"^(05\d{8}|\+?9665\d{8})$",
        
        # Hijri date: various formats
        # Examples: ١٤٤٥/٧/١٥, 1445/7/15, ١٥/٧/١٤٤٥, 15-07-1445 هـ
        "hijri_date": r"^(\d{1,2}[-/]\d{1,2}[-/]14\d{2}|14\d{2}[-/]\d{1,2}[-/]\d{1,2})(\s*هـ?)?$",
        
        # Vehicle plate: 1-3 Arabic letters + 1-4 digits, or reverse
        "vehicle_plate": r"^[\u0600-\u06FF]{1,3}\s*\d{1,4}$|^\d{1,4}\s*[\u0600-\u06FF]{1,3}$",
        
        # Saudi IBAN: SA + 22 characters (total 24)
        "iban": r"^SA\d{22}$",
        
        # Commercial Registration: 10 digits
        "commercial_registration": r"^\d{10}$",
    }
    
    # Suspicious patterns (likely hallucinations)
    SUSPICIOUS_PATTERNS = [
        # Sequential digits (1234567890, 0123456789)
        r"^(0?1234567890|1234567890)$",
        r"^(9876543210|0987654321)$",
        
        # Repeated digits (1111111111, 2222222222)
        r"^(\d)\1{9}$",
        
        # Test patterns
        r"^(test|example|sample)",
        
        # Placeholder-like international format when local expected
        r"^00966",  # International format when local was likely written
        
        # Very common placeholder names (likely hallucinated)
        r"^(محمد محمد|عبدالله محمد|أحمد محمد)$",
        
        # Default cities (when field was likely empty)
        r"^(الرياض|جدة|مكة)$",  # Only suspicious if nothing else in context
    ]
    
    # Common generic values that might be hallucinated
    GENERIC_VALUES = {
        "national_id": ["1000000000", "2000000000", "1234567890"],
        "mobile": ["0500000000", "0501234567", "0512345678"],
        "city": [],  # Cities are too variable to flag as suspicious
        "name": ["محمد", "أحمد", "عبدالله"],  # Single common names might be hallucinated
    }
    
    def __init__(self):
        # Compile patterns for efficiency
        self._compiled_patterns = {
            name: re.compile(pattern)
            for name, pattern in self.PATTERNS.items()
        }
        self._suspicious_compiled = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.SUSPICIOUS_PATTERNS
        ]
    
    def validate_field(self, field_name: str, value: str) -> ValidationResult:
        """
        Validate a field value based on its name.
        
        Returns:
        - valid: matches expected pattern
        - invalid: doesn't match pattern (possible OCR error)
        - suspicious: matches hallucination patterns (likely made up)
        - unchecked: no validator for this field type
        """
        if not value or value.strip() == "":
            return ValidationResult.valid(reason="Empty value (expected)")
        
        # Normalize the value (convert Arabic to Western digits)
        normalized = normalize_digits(value.strip())
        
        # First, check for suspicious patterns (likely hallucinations)
        for pattern in self._suspicious_compiled:
            if pattern.search(normalized):
                return ValidationResult.suspicious(
                    reason=f"Matches suspicious pattern (possible hallucination)"
                )
        
        # Detect field type
        field_type = detect_field_type(field_name)
        
        if field_type is None:
            return ValidationResult.unchecked(reason="Unknown field type")
        
        # Check against generic values
        if field_type in self.GENERIC_VALUES:
            if normalized in self.GENERIC_VALUES[field_type]:
                return ValidationResult.suspicious(
                    reason=f"Common placeholder value for {field_type}"
                )
        
        # Validate against pattern
        if field_type in self._compiled_patterns:
            pattern = self._compiled_patterns[field_type]
            
            # Remove common separators for matching
            clean_value = normalized.replace(" ", "").replace("-", "").replace("/", "/")
            
            if pattern.match(clean_value):
                return ValidationResult.valid(
                    normalized_value=clean_value,
                    reason=f"Valid {field_type} format"
                )
            else:
                # Check if it's close but malformed
                partial_reason = self._get_format_hint(field_type, clean_value)
                return ValidationResult.invalid(
                    reason=f"Invalid {field_type} format: {partial_reason}"
                )
        
        return ValidationResult.unchecked(reason=f"No pattern validator for {field_type}")
    
    def _get_format_hint(self, field_type: str, value: str) -> str:
        """Get a hint about the expected format."""
        hints = {
            "national_id": "Expected 10 digits starting with 1 or 2",
            "mobile": "Expected 05xxxxxxxx (10 digits)",
            "hijri_date": "Expected DD/MM/14YY or 14YY/MM/DD format",
            "vehicle_plate": "Expected Arabic letters + digits",
            "iban": "Expected SA + 22 digits",
            "commercial_registration": "Expected 10 digits",
        }
        return hints.get(field_type, "Format mismatch")
    
    def validate_national_id(self, value: str) -> ValidationResult:
        """Specifically validate Saudi National ID."""
        normalized = normalize_digits(value.strip().replace(" ", ""))
        
        if len(normalized) != 10:
            return ValidationResult.invalid(f"National ID must be 10 digits, got {len(normalized)}")
        
        if not normalized.isdigit():
            return ValidationResult.invalid("National ID must contain only digits")
        
        first_digit = normalized[0]
        if first_digit == "1":
            return ValidationResult.valid(normalized, "Valid Saudi citizen ID")
        elif first_digit == "2":
            return ValidationResult.valid(normalized, "Valid resident (Iqama) ID")
        else:
            return ValidationResult.invalid("National ID must start with 1 (citizen) or 2 (resident)")
    
    def validate_mobile(self, value: str) -> ValidationResult:
        """Specifically validate Saudi mobile number."""
        normalized = normalize_digits(value.strip().replace(" ", "").replace("-", ""))
        
        # Remove +966 or 00966 prefix
        if normalized.startswith("+966"):
            normalized = "0" + normalized[4:]
        elif normalized.startswith("00966"):
            normalized = "0" + normalized[5:]
        elif normalized.startswith("966"):
            normalized = "0" + normalized[3:]
        
        if len(normalized) != 10:
            return ValidationResult.invalid(f"Mobile must be 10 digits, got {len(normalized)}")
        
        if not normalized.startswith("05"):
            return ValidationResult.invalid("Saudi mobile must start with 05")
        
        return ValidationResult.valid(normalized, "Valid Saudi mobile number")
    
    def validate_hijri_date(self, value: str) -> ValidationResult:
        """Validate Hijri date format."""
        normalized = normalize_digits(value.strip())
        
        # Remove هـ suffix if present
        normalized = normalized.replace("هـ", "").strip()
        
        # Try different date separators
        for sep in ["/", "-", "."]:
            parts = normalized.split(sep)
            if len(parts) == 3:
                try:
                    # Check if it's YYYY/MM/DD or DD/MM/YYYY
                    if len(parts[0]) == 4 and parts[0].startswith("14"):
                        # YYYY/MM/DD format
                        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    elif len(parts[2]) == 4 and parts[2].startswith("14"):
                        # DD/MM/YYYY format
                        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                    else:
                        continue
                    
                    # Validate ranges (Hijri calendar)
                    if 1400 <= year <= 1500 and 1 <= month <= 12 and 1 <= day <= 30:
                        return ValidationResult.valid(
                            f"{year}/{month:02d}/{day:02d}",
                            "Valid Hijri date"
                        )
                except (ValueError, IndexError):
                    continue
        
        return ValidationResult.invalid("Invalid Hijri date format")


# =============================================================================
# GLOBAL VALIDATOR INSTANCE
# =============================================================================

# Create a default validator instance for convenience
default_validators = SaudiDocumentValidators()

def validate_field(field_name: str, value: str) -> ValidationResult:
    """Convenience function to validate a field using the default validators."""
    return default_validators.validate_field(field_name, value)
