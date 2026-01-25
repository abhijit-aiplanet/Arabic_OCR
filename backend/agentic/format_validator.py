"""
Format Validator for Saudi Document OCR

Validates extracted values against known Saudi document formats:
- National ID (10 digits starting with 1 or 2)
- Mobile phone (10 digits starting with 05)
- Hijri dates (DD/MM/YYYY format)
- Arabic names (Arabic characters)

Catches impossible values that indicate OCR errors.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ValidationSeverity(Enum):
    """Severity of validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of validating a single field."""
    field_name: str
    value: str
    is_valid: bool
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    issues: List[str] = field(default_factory=list)
    expected_format: Optional[str] = None
    suggestion: Optional[str] = None
    severity: ValidationSeverity = ValidationSeverity.INFO
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "value": self.value,
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "issues": self.issues,
            "expected_format": self.expected_format,
            "suggestion": self.suggestion,
            "severity": self.severity.value,
        }


@dataclass
class DocumentValidation:
    """Complete validation result for a document."""
    is_valid: bool
    overall_score: int  # 0-100
    field_results: Dict[str, ValidationResult] = field(default_factory=dict)
    duplicate_values: Dict[str, List[str]] = field(default_factory=dict)
    hallucination_indicators: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "overall_score": self.overall_score,
            "field_results": {k: v.to_dict() for k, v in self.field_results.items()},
            "duplicate_values": self.duplicate_values,
            "hallucination_indicators": self.hallucination_indicators,
            "warnings": self.warnings,
            "critical_issues": self.critical_issues,
        }


# =============================================================================
# FORMAT SPECIFICATIONS
# =============================================================================

# Arabic-Indic numerals mapping
ARABIC_TO_WESTERN = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
}

WESTERN_TO_ARABIC = {v: k for k, v in ARABIC_TO_WESTERN.items()}

# Format specifications for Saudi documents
FORMAT_SPECS = {
    # National ID
    "رقم الهوية": {
        "pattern": r"^[12]\d{9}$",
        "description": "10 digits starting with 1 or 2",
        "example": "1038367680 / ١٠٣٨٣٦٧٦٨٠",
        "min_length": 10,
        "max_length": 10,
        "field_type": "national_id",
    },
    "رقم بطاقة الأحوال": {
        "pattern": r"^[12]\d{9}$",
        "description": "10 digits starting with 1 or 2",
        "example": "1038367680",
        "min_length": 10,
        "max_length": 10,
        "field_type": "national_id",
    },
    
    # Mobile phone
    "جوال": {
        "pattern": r"^05\d{8}$",
        "description": "10 digits starting with 05",
        "example": "0507477998 / ٠٥٠٧٤٧٧٩٩٨",
        "min_length": 10,
        "max_length": 10,
        "field_type": "phone",
    },
    "رقم الجوال": {
        "pattern": r"^05\d{8}$",
        "description": "10 digits starting with 05",
        "example": "0507477998",
        "min_length": 10,
        "max_length": 10,
        "field_type": "phone",
    },
    
    # Dates (Hijri calendar)
    "تاريخ الميلاد": {
        "pattern": r"^\d{1,2}/\d{1,2}/\d{4}$",
        "description": "DD/MM/YYYY (Hijri)",
        "example": "1/7/1385 / ١/٧/١٣٨٥",
        "field_type": "date",
    },
    "تاريخها": {
        "pattern": r"^\d{1,2}/\d{1,2}/\d{4}$",
        "description": "DD/MM/YYYY (Hijri)",
        "example": "8/7/1404",
        "field_type": "date",
    },
    "تاريخ الإصدار": {
        "pattern": r"^\d{1,2}/\d{1,2}/\d{4}$",
        "description": "DD/MM/YYYY (Hijri)",
        "example": "25/5/1437",
        "field_type": "date",
    },
    "تاريخ الانتهاء": {
        "pattern": r"^\d{1,2}/\d{1,2}/\d{4}$",
        "description": "DD/MM/YYYY (Hijri)",
        "example": "25/7/1437",
        "field_type": "date",
    },
    "التاريخ": {
        "pattern": r"^\d{1,2}/\d{1,2}/\d{2,4}$",
        "description": "DD/MM/YY or DD/MM/YYYY",
        "example": "2/11/14",
        "field_type": "date",
    },
    
    # Names
    "اسم المالك": {
        "pattern": r"^[\u0600-\u06FF\s]+$",
        "description": "Arabic name (Arabic characters only)",
        "example": "عياض هديكي دعيس العتيبي",
        "field_type": "arabic_name",
    },
    "اسم مقدم الطلب": {
        "pattern": r"^[\u0600-\u06FF\s]+$",
        "description": "Arabic name",
        "example": "عبدالعزيز الهديكي",
        "field_type": "arabic_name",
    },
    
    # Cities
    "المدينة": {
        "pattern": r"^[\u0600-\u06FF\s]+$",
        "description": "Arabic city name",
        "example": "جدة / الرياض",
        "field_type": "city",
    },
    "مدينة مزاولة النشاط": {
        "pattern": r"^[\u0600-\u06FF\s]+$",
        "description": "Arabic city name",
        "example": "جدة",
        "field_type": "city",
    },
    "مصدرها": {
        "pattern": r"^[\u0600-\u06FF\s]+$",
        "description": "Arabic city name (ID source)",
        "example": "المرواحي / الرياض",
        "field_type": "city",
    },
    
    # Vehicle plate
    "رقم اللوحة": {
        "pattern": r"^[\u0600-\u06FF\d\s\-]+$",
        "description": "Arabic letters and numbers",
        "example": "أ ب ج ١٢٣٤",
        "field_type": "plate",
    },
    
    # Driving license number
    "رقمها": {
        "pattern": r"^\d{10}$",
        "description": "10 digits",
        "example": "1038326680",
        "min_length": 10,
        "max_length": 10,
        "field_type": "license_number",
    },
    
    # Dependents count
    "عدد من يعولهم": {
        "pattern": r"^\d{1,2}$",
        "description": "1-2 digits",
        "example": "1",
        "field_type": "number",
    },
}

# Known Saudi cities for validation
SAUDI_CITIES = [
    "الرياض", "جدة", "مكة", "المدينة", "الدمام", "الخبر", "الطائف",
    "بريدة", "تبوك", "خميس مشيط", "الاحساء", "القطيف", "الجبيل",
    "حائل", "نجران", "الباحة", "جازان", "سكاكا", "عرعر",
    "المرواحي", "ينبع", "القصيم", "أبها",
]

# Common marital status values
MARITAL_STATUS = ["متزوج", "أعزب", "مطلق", "أرمل"]

# Common qualification values
QUALIFICATIONS = ["أمي", "ابتدائي", "متوسط", "ثانوي", "جامعي", "دبلوم"]


# =============================================================================
# FORMAT VALIDATOR
# =============================================================================

class FormatValidator:
    """
    Validates extracted values against known Saudi document formats.
    
    Features:
    - Arabic-Indic numeral normalization
    - Format pattern matching
    - Duplicate detection (hallucination indicator)
    - Empty field handling
    - Confidence scoring
    """
    
    def __init__(self):
        """Initialize the format validator."""
        self.format_specs = FORMAT_SPECS
        self.saudi_cities = SAUDI_CITIES
    
    def normalize_arabic_digits(self, text: str) -> str:
        """Convert Arabic-Indic numerals to Western digits."""
        result = text
        for arabic, western in ARABIC_TO_WESTERN.items():
            result = result.replace(arabic, western)
        return result
    
    def normalize_for_comparison(self, text: str) -> str:
        """Normalize text for comparison (lowercase, strip, normalize digits)."""
        normalized = self.normalize_arabic_digits(text.strip())
        return normalized.replace(" ", "").replace("-", "").replace("/", "")
    
    def validate_field(
        self,
        field_name: str,
        value: str,
    ) -> ValidationResult:
        """
        Validate a single field against its expected format.
        
        Args:
            field_name: Name of the field
            value: Extracted value
            
        Returns:
            ValidationResult with validation status and details
        """
        # Handle empty values
        if not value or value in ["---", "[فارغ]", "[EMPTY]", "فارغ"]:
            return ValidationResult(
                field_name=field_name,
                value=value,
                is_valid=True,
                confidence="HIGH",
                issues=[],
                expected_format="Empty field",
            )
        
        # Get format spec for this field
        spec = self._get_format_spec(field_name)
        
        if not spec:
            # No specific format, basic validation only
            return ValidationResult(
                field_name=field_name,
                value=value,
                is_valid=True,
                confidence="MEDIUM",
                issues=[],
            )
        
        # Normalize value for pattern matching
        normalized = self.normalize_arabic_digits(value.strip())
        
        issues = []
        severity = ValidationSeverity.INFO
        
        # Check pattern
        pattern = spec.get("pattern")
        if pattern:
            if not re.match(pattern, normalized):
                issues.append(f"Format mismatch: expected {spec['description']}")
                severity = ValidationSeverity.WARNING
        
        # Check length constraints
        min_len = spec.get("min_length")
        max_len = spec.get("max_length")
        
        if min_len and len(normalized) < min_len:
            issues.append(f"Too short: expected at least {min_len} characters")
            severity = ValidationSeverity.ERROR
        
        if max_len and len(normalized) > max_len:
            issues.append(f"Too long: expected at most {max_len} characters")
            severity = ValidationSeverity.WARNING
        
        # Field-specific validation
        field_type = spec.get("field_type")
        
        if field_type == "national_id":
            id_issues = self._validate_national_id(normalized)
            issues.extend(id_issues)
            if id_issues:
                severity = ValidationSeverity.CRITICAL
        
        elif field_type == "phone":
            phone_issues = self._validate_phone(normalized)
            issues.extend(phone_issues)
            if phone_issues:
                severity = ValidationSeverity.ERROR
        
        elif field_type == "date":
            date_issues = self._validate_date(normalized)
            issues.extend(date_issues)
            if date_issues:
                severity = ValidationSeverity.WARNING
        
        elif field_type == "arabic_name":
            name_issues = self._validate_arabic_name(value)
            issues.extend(name_issues)
            if name_issues:
                severity = ValidationSeverity.ERROR
        
        elif field_type == "city":
            city_issues = self._validate_city(value)
            issues.extend(city_issues)
            # City validation is soft - unknown cities are ok
        
        # Determine confidence based on issues
        if not issues:
            confidence = "HIGH"
            is_valid = True
        elif severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]:
            confidence = "LOW"
            is_valid = False
        else:
            confidence = "MEDIUM"
            is_valid = True
        
        return ValidationResult(
            field_name=field_name,
            value=value,
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            expected_format=spec.get("description"),
            severity=severity,
        )
    
    def validate_document(
        self,
        fields: Dict[str, str],
    ) -> DocumentValidation:
        """
        Validate all fields in a document extraction.
        
        Checks for:
        - Individual field format violations
        - Duplicate values (hallucination indicator)
        - Suspicious patterns
        
        Args:
            fields: Dictionary of field_name -> value
            
        Returns:
            DocumentValidation with complete validation results
        """
        field_results = {}
        warnings = []
        critical_issues = []
        hallucination_indicators = []
        
        # Validate each field
        for field_name, value in fields.items():
            result = self.validate_field(field_name, value)
            field_results[field_name] = result
            
            if result.severity == ValidationSeverity.CRITICAL:
                critical_issues.append(f"{field_name}: {', '.join(result.issues)}")
            elif result.severity == ValidationSeverity.ERROR:
                warnings.append(f"{field_name}: {', '.join(result.issues)}")
        
        # Check for duplicate values (hallucination indicator)
        duplicates = self._find_duplicates(fields)
        if duplicates:
            hallucination_indicators.append("Duplicate values detected across fields")
            for value, field_list in duplicates.items():
                critical_issues.append(
                    f"Value '{value[:30]}...' appears in {len(field_list)} fields: {', '.join(field_list[:3])}"
                )
        
        # Check for year-as-value pattern (common hallucination)
        year_issues = self._check_year_as_value(fields)
        if year_issues:
            hallucination_indicators.extend(year_issues)
            critical_issues.extend(year_issues)
        
        # Check for suspiciously complete form
        empty_count = sum(1 for v in fields.values() if v in ["---", "[فارغ]", ""])
        if empty_count == 0 and len(fields) > 10:
            warnings.append("Suspiciously complete form - real forms usually have empty fields")
        
        # Calculate overall score
        valid_count = sum(1 for r in field_results.values() if r.is_valid)
        total_count = len(field_results) if field_results else 1
        base_score = int((valid_count / total_count) * 100)
        
        # Deduct for hallucination indicators
        penalty = len(hallucination_indicators) * 20 + len(critical_issues) * 10
        overall_score = max(0, base_score - penalty)
        
        is_valid = overall_score >= 40 and len(critical_issues) == 0
        
        return DocumentValidation(
            is_valid=is_valid,
            overall_score=overall_score,
            field_results=field_results,
            duplicate_values=duplicates,
            hallucination_indicators=hallucination_indicators,
            warnings=warnings,
            critical_issues=critical_issues,
        )
    
    def _get_format_spec(self, field_name: str) -> Optional[Dict[str, Any]]:
        """Get format specification for a field."""
        # Direct match
        if field_name in self.format_specs:
            return self.format_specs[field_name]
        
        # Partial match
        for key, spec in self.format_specs.items():
            if key in field_name or field_name in key:
                return spec
        
        return None
    
    def _validate_national_id(self, value: str) -> List[str]:
        """Validate Saudi National ID."""
        issues = []
        
        # Remove non-digits
        digits_only = re.sub(r'\D', '', value)
        
        if len(digits_only) != 10:
            issues.append(f"ID must be exactly 10 digits (got {len(digits_only)})")
        elif digits_only[0] not in ['1', '2']:
            issues.append(f"ID must start with 1 or 2 (got {digits_only[0]})")
        
        # Check for suspicious patterns
        if len(set(digits_only)) <= 2:
            issues.append("ID has suspicious repeating pattern")
        
        # Check if it looks like a year (1400-1450)
        if len(digits_only) == 4 and digits_only.startswith('14'):
            issues.append("Value appears to be a year, not an ID")
        
        return issues
    
    def _validate_phone(self, value: str) -> List[str]:
        """Validate Saudi mobile phone number."""
        issues = []
        
        # Remove non-digits
        digits_only = re.sub(r'\D', '', value)
        
        if len(digits_only) != 10:
            issues.append(f"Phone must be 10 digits (got {len(digits_only)})")
        elif not digits_only.startswith('05'):
            issues.append(f"Phone must start with 05 (got {digits_only[:2]})")
        
        return issues
    
    def _validate_date(self, value: str) -> List[str]:
        """Validate Hijri date format."""
        issues = []
        
        # Check for date separators
        if '/' not in value and '-' not in value:
            issues.append("Date should contain / or - separator")
            return issues
        
        # Try to parse
        parts = re.split(r'[/\-]', value)
        
        if len(parts) < 3:
            issues.append("Date should have day, month, and year")
            return issues
        
        try:
            day = int(parts[0])
            month = int(parts[1])
            year = int(parts[2])
            
            if day < 1 or day > 30:
                issues.append(f"Day should be 1-30 (got {day})")
            
            if month < 1 or month > 12:
                issues.append(f"Month should be 1-12 (got {month})")
            
            # Hijri year validation (reasonable range)
            if year > 100 and (year < 1300 or year > 1500):
                issues.append(f"Year appears invalid for Hijri calendar (got {year})")
                
        except ValueError:
            issues.append("Date contains non-numeric values")
        
        return issues
    
    def _validate_arabic_name(self, value: str) -> List[str]:
        """Validate Arabic name."""
        issues = []
        
        # Check for at least some Arabic characters
        arabic_chars = re.findall(r'[\u0600-\u06FF]', value)
        
        if not arabic_chars:
            issues.append("Name should contain Arabic characters")
        
        # Check for too many digits
        digits = re.findall(r'\d', value)
        if len(digits) > len(arabic_chars):
            issues.append("Name contains more digits than Arabic characters")
        
        return issues
    
    def _validate_city(self, value: str) -> List[str]:
        """Validate city name."""
        issues = []
        
        # Soft validation - just check if it looks like a city name
        if re.match(r'^\d+$', value):
            issues.append("City should not be a number")
        
        return issues
    
    def _find_duplicates(self, fields: Dict[str, str]) -> Dict[str, List[str]]:
        """Find duplicate values across fields (hallucination indicator)."""
        duplicates = {}
        
        # Normalize values for comparison
        value_to_fields: Dict[str, List[str]] = {}
        
        for field_name, value in fields.items():
            if not value or value in ["---", "[فارغ]", ""]:
                continue
            
            normalized = self.normalize_for_comparison(value)
            
            # Skip very short values (could legitimately repeat)
            if len(normalized) < 3:
                continue
            
            if normalized not in value_to_fields:
                value_to_fields[normalized] = []
            value_to_fields[normalized].append(field_name)
        
        # Find fields with same value
        for normalized, field_list in value_to_fields.items():
            if len(field_list) >= 2:
                # Find the original value
                original_value = None
                for field in field_list:
                    if field in fields:
                        original_value = fields[field]
                        break
                
                if original_value:
                    duplicates[original_value] = field_list
        
        return duplicates
    
    def _check_year_as_value(self, fields: Dict[str, str]) -> List[str]:
        """Check for year values in non-date fields (common hallucination)."""
        issues = []
        year_pattern = re.compile(r'^14\d{2}$')  # Hijri years 1400-1499
        
        non_date_fields = [
            "رقم الهوية", "رقم الجوال", "جوال", "اسم المالك",
            "المدينة", "الحي", "رقم اللوحة",
        ]
        
        for field_name, value in fields.items():
            if not value:
                continue
            
            normalized = self.normalize_arabic_digits(value.strip())
            
            # Check if this looks like a year
            if year_pattern.match(normalized):
                # Check if it's in a non-date field
                is_date_field = any(
                    date_term in field_name 
                    for date_term in ["تاريخ", "سنة", "عام"]
                )
                
                if not is_date_field:
                    for non_date in non_date_fields:
                        if non_date in field_name or field_name in non_date:
                            issues.append(
                                f"Year '{value}' found in non-date field '{field_name}' - possible hallucination"
                            )
                            break
        
        return issues


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_extraction(fields: Dict[str, str]) -> DocumentValidation:
    """
    Convenience function to validate an extraction.
    
    Args:
        fields: Dictionary of field_name -> value
        
    Returns:
        DocumentValidation result
    """
    validator = FormatValidator()
    return validator.validate_document(fields)


def get_field_format_hint(field_name: str) -> str:
    """Get format hint for a field name."""
    for key, spec in FORMAT_SPECS.items():
        if key in field_name or field_name in key:
            return spec.get("description", "")
    return ""
