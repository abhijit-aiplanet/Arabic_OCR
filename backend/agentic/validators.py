"""
Field Validators for Arabic OCR Anti-Hallucination System

This module provides comprehensive validation to detect and prevent:
- Value propagation (same value copied to multiple fields)
- Format mismatches (wrong data type for field)
- Hallucinated values (invented data not in the image)
- Empty field misclassification

Saudi-specific validation rules for:
- National ID (رقم الهوية)
- Mobile numbers (رقم الجوال)
- Hijri dates (تاريخ)
- Vehicle plates (رقم اللوحة)
- Arabic names (الاسم)
"""

import re
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter


# =============================================================================
# CONSTANTS AND PATTERNS
# =============================================================================

# Arabic digits mapping
ARABIC_TO_WESTERN = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
}

WESTERN_TO_ARABIC = {v: k for k, v in ARABIC_TO_WESTERN.items()}

# Arabic letters range
ARABIC_LETTERS = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')

# Field type patterns
PATTERNS = {
    # Saudi National ID: 10 digits, starts with 1 (citizen) or 2 (resident)
    'national_id': re.compile(r'^[12][0-9٠-٩]{9}$'),
    
    # Saudi Mobile: 10 digits, starts with 05
    'mobile': re.compile(r'^0[5٥][0-9٠-٩]{8}$'),
    
    # Hijri Date: Various formats
    'date_full': re.compile(r'^[0-9٠-٩]{1,2}[/\-\.][0-9٠-٩]{1,2}[/\-\.][0-9٠-٩]{4}$'),
    'date_short': re.compile(r'^[0-9٠-٩]{1,2}[/\-\.][0-9٠-٩]{1,2}[/\-\.][0-9٠-٩]{2}$'),
    
    # Year only (for hallucination detection)
    'year_only': re.compile(r'^[0-9٠-٩]{4}$'),
    'hijri_year': re.compile(r'^1[34][0-9٠-٩]{2}$'),  # 1300-1499 Hijri
    
    # Vehicle Plate: Arabic letters + digits
    'plate': re.compile(r'^[\u0600-\u06FF]{1,3}\s*[0-9٠-٩]{1,4}$'),
    'plate_reverse': re.compile(r'^[0-9٠-٩]{1,4}\s*[\u0600-\u06FF]{1,3}$'),
    
    # Arabic name: 2-6 Arabic words
    'arabic_name': re.compile(r'^[\u0600-\u06FF\s]{4,100}$'),
}

# Field name to type mapping (Arabic field names)
FIELD_TYPE_MAP = {
    # National ID variants
    'رقم الهوية': 'national_id',
    'رقم بطاقة الأحوال': 'national_id',
    'رقم الهوية الوطنية': 'national_id',
    'هوية': 'national_id',
    
    # Mobile variants
    'رقم الجوال': 'mobile',
    'الجوال': 'mobile',
    'رقم الهاتف': 'mobile',
    'الهاتف': 'mobile',
    'جوال': 'mobile',
    
    # Date variants
    'تاريخ الميلاد': 'date',
    'تاريخ الإصدار': 'date',
    'تاريخ الانتهاء': 'date',
    'تاريخ التسجيل': 'date',
    'تاريخ': 'date',
    
    # Name variants
    'اسم المالك': 'name',
    'الاسم': 'name',
    'اسم': 'name',
    'الاسم الكامل': 'name',
    
    # Vehicle plate variants
    'رقم اللوحة': 'plate',
    'اللوحة': 'plate',
    'لوحة المركبة': 'plate',
    
    # City (less strict validation)
    'المدينة': 'city',
    'مدينة': 'city',
    
    # Generic numeric
    'رقم': 'numeric',
}

# Known Saudi cities for validation
SAUDI_CITIES = {
    'الرياض', 'جدة', 'مكة', 'المدينة', 'الدمام', 'الخبر', 'الطائف',
    'تبوك', 'بريدة', 'خميس مشيط', 'حائل', 'نجران', 'جازان', 'أبها',
    'ينبع', 'القطيف', 'الجبيل', 'الأحساء', 'عرعر', 'سكاكا'
}

# Special markers that indicate valid empty/unclear values
VALID_MARKERS = {
    '[فارغ]', '[غير واضح', '[غير مقروء]', '[توقيع]', '[ختم]',
    'فارغ', 'غير واضح', 'غير مقروء', 'empty', 'unclear', 'unreadable'
}


# =============================================================================
# VALIDATION RESULT TYPES
# =============================================================================

class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"           # Minor observation
    WARNING = "warning"     # Suspicious but not blocking
    ERROR = "error"         # Definite problem
    CRITICAL = "critical"   # Likely hallucination, should reject


@dataclass
class ValidationIssue:
    """A single validation issue found."""
    field_name: str
    issue_type: str
    severity: ValidationSeverity
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "issue_type": self.issue_type,
            "severity": self.severity.value,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass
class ValidationResult:
    """Complete validation result for an extraction."""
    is_valid: bool
    quality_score: int  # 0-100
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    field_scores: Dict[str, int] = field(default_factory=dict)
    duplicate_values: Dict[str, List[str]] = field(default_factory=dict)
    hallucination_indicators: List[str] = field(default_factory=list)
    
    def has_critical_issues(self) -> bool:
        return any(i.severity == ValidationSeverity.CRITICAL for i in self.issues)
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == severity]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "quality_score": self.quality_score,
            "issues": [i.to_dict() for i in self.issues],
            "warnings": self.warnings,
            "field_scores": self.field_scores,
            "duplicate_values": self.duplicate_values,
            "hallucination_indicators": self.hallucination_indicators,
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_arabic_digits(text: str) -> str:
    """Convert Arabic digits to Western digits for validation."""
    if not text:
        return text
    result = text
    for arabic, western in ARABIC_TO_WESTERN.items():
        result = result.replace(arabic, western)
    return result


def extract_digits(text: str) -> str:
    """Extract only digits from text."""
    normalized = normalize_arabic_digits(text)
    return ''.join(c for c in normalized if c.isdigit())


def is_valid_marker(value: str) -> bool:
    """Check if value is a valid empty/unclear marker."""
    if not value:
        return False
    value_lower = value.strip().lower()
    for marker in VALID_MARKERS:
        if marker in value or marker.lower() in value_lower:
            return True
    return False


def get_field_type(field_name: str) -> Optional[str]:
    """Determine the field type from its name."""
    # Exact match
    if field_name in FIELD_TYPE_MAP:
        return FIELD_TYPE_MAP[field_name]
    
    # Partial match
    for key, ftype in FIELD_TYPE_MAP.items():
        if key in field_name or field_name in key:
            return ftype
    
    # Keyword-based detection
    name_lower = field_name.lower()
    if 'هوية' in field_name or 'id' in name_lower:
        return 'national_id'
    if 'جوال' in field_name or 'هاتف' in field_name or 'mobile' in name_lower or 'phone' in name_lower:
        return 'mobile'
    if 'تاريخ' in field_name or 'date' in name_lower:
        return 'date'
    if 'اسم' in field_name or 'name' in name_lower:
        return 'name'
    if 'لوحة' in field_name or 'plate' in name_lower:
        return 'plate'
    
    return None


def count_arabic_words(text: str) -> int:
    """Count Arabic words in text."""
    if not text:
        return 0
    words = ARABIC_LETTERS.findall(text)
    return len(words)


def is_hijri_year(value: str) -> bool:
    """Check if value looks like a Hijri year (1300-1450)."""
    digits = extract_digits(value)
    if len(digits) == 4:
        try:
            year = int(digits)
            return 1300 <= year <= 1450
        except ValueError:
            pass
    return False


def is_gregorian_year(value: str) -> bool:
    """Check if value looks like a Gregorian year (1900-2100)."""
    digits = extract_digits(value)
    if len(digits) == 4:
        try:
            year = int(digits)
            return 1900 <= year <= 2100
        except ValueError:
            pass
    return False


# =============================================================================
# FIELD FORMAT VALIDATORS
# =============================================================================

def validate_national_id(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Saudi National ID.
    
    Rules:
    - Exactly 10 digits
    - Starts with 1 (citizen) or 2 (resident)
    """
    if is_valid_marker(value):
        return True, None
    
    digits = extract_digits(value)
    
    if len(digits) == 0:
        return False, "No digits found in ID field"
    
    if len(digits) != 10:
        return False, f"ID must be 10 digits, found {len(digits)}"
    
    if digits[0] not in ('1', '2'):
        return False, f"ID must start with 1 or 2, found {digits[0]}"
    
    # Check for suspicious patterns
    if len(set(digits)) <= 2:
        return False, "ID has too few unique digits (likely hallucination)"
    
    if digits == '1234567890' or digits == '0987654321':
        return False, "Sequential digits (likely hallucination)"
    
    return True, None


def validate_mobile(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Saudi Mobile Number.
    
    Rules:
    - Exactly 10 digits
    - Starts with 05
    """
    if is_valid_marker(value):
        return True, None
    
    digits = extract_digits(value)
    
    if len(digits) == 0:
        return False, "No digits found in mobile field"
    
    if len(digits) != 10:
        return False, f"Mobile must be 10 digits, found {len(digits)}"
    
    if not digits.startswith('05'):
        return False, f"Mobile must start with 05, found {digits[:2]}"
    
    # Check for suspicious patterns
    if len(set(digits)) <= 2:
        return False, "Mobile has too few unique digits (likely hallucination)"
    
    return True, None


def validate_date(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Hijri or Gregorian date.
    
    Accepts:
    - DD/MM/YYYY
    - DD-MM-YYYY
    - DD.MM.YYYY
    - Just year (but flags as incomplete)
    """
    if is_valid_marker(value):
        return True, None
    
    # Check for full date format
    if PATTERNS['date_full'].match(value) or PATTERNS['date_short'].match(value):
        # Extract components
        parts = re.split(r'[/\-\.]', normalize_arabic_digits(value))
        if len(parts) >= 3:
            try:
                day = int(parts[0])
                month = int(parts[1])
                year = int(parts[2])
                
                # Basic range validation
                if not (1 <= day <= 30):
                    return False, f"Invalid day: {day}"
                if not (1 <= month <= 12):
                    return False, f"Invalid month: {month}"
                
                # Year should be Hijri (1300-1450) or Gregorian (1900-2100)
                if year < 100:
                    year += 1400 if year < 50 else 1300  # Assume Hijri short year
                
                if not ((1300 <= year <= 1450) or (1900 <= year <= 2100)):
                    return False, f"Year {year} out of valid range"
                    
            except ValueError:
                pass
        return True, None
    
    # Check if it's just a year (incomplete but not invalid)
    if PATTERNS['year_only'].match(normalize_arabic_digits(value)):
        return True, "Date appears to be year-only (incomplete)"
    
    # Check if it contains any date-like pattern
    if re.search(r'[0-9٠-٩]{1,4}[/\-\.][0-9٠-٩]{1,4}', value):
        return True, "Partial date format detected"
    
    return False, f"Does not match date format: {value}"


def validate_name(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Arabic name.
    
    Rules:
    - Contains Arabic characters
    - Typically 2-6 words
    - Not just numbers
    """
    if is_valid_marker(value):
        return True, None
    
    # Check for Arabic content
    arabic_words = count_arabic_words(value)
    
    if arabic_words == 0:
        return False, "No Arabic text found in name field"
    
    if arabic_words == 1 and len(value) < 3:
        return False, "Name too short (single short word)"
    
    # Check it's not just numbers
    digits = extract_digits(value)
    if len(digits) > len(value) / 2:
        return False, "Name field contains mostly digits"
    
    return True, None


def validate_plate(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate Saudi vehicle plate.
    
    Rules:
    - 1-3 Arabic letters
    - 1-4 digits
    - Letters and digits can be in either order
    """
    if is_valid_marker(value):
        return True, None
    
    # Check for plate patterns
    if PATTERNS['plate'].match(value) or PATTERNS['plate_reverse'].match(value):
        return True, None
    
    # Check for partial patterns
    has_arabic = bool(ARABIC_LETTERS.search(value))
    has_digits = bool(re.search(r'[0-9٠-٩]', value))
    
    if has_arabic and has_digits:
        return True, "Partial plate format detected"
    
    if has_digits and not has_arabic:
        return False, "Plate missing Arabic letters"
    
    return False, f"Does not match plate format: {value}"


def validate_city(value: str) -> Tuple[bool, Optional[str]]:
    """
    Validate city name.
    
    Rules:
    - Should be Arabic text
    - Preferably a known Saudi city
    """
    if is_valid_marker(value):
        return True, None
    
    # Check for Arabic content
    if not ARABIC_LETTERS.search(value):
        return False, "No Arabic text in city field"
    
    # Check against known cities
    normalized = value.strip()
    if normalized in SAUDI_CITIES:
        return True, None
    
    # Partial match
    for city in SAUDI_CITIES:
        if city in normalized or normalized in city:
            return True, None
    
    # Still valid if it's Arabic text (might be unknown city)
    return True, "City not in known list (may still be valid)"


# =============================================================================
# FIELD VALIDATOR DISPATCHER
# =============================================================================

def validate_field(field_name: str, value: str) -> Tuple[bool, Optional[str], int]:
    """
    Validate a single field based on its type.
    
    Returns:
        Tuple of (is_valid, error_message, confidence_score)
        confidence_score: 0-100, higher = more confident in validity
    """
    if not value or not value.strip():
        return True, None, 50  # Empty is acceptable
    
    if is_valid_marker(value):
        return True, None, 80  # Valid marker = good
    
    field_type = get_field_type(field_name)
    
    if field_type == 'national_id':
        valid, msg = validate_national_id(value)
        return valid, msg, 90 if valid else 20
    
    elif field_type == 'mobile':
        valid, msg = validate_mobile(value)
        return valid, msg, 90 if valid else 20
    
    elif field_type == 'date':
        valid, msg = validate_date(value)
        return valid, msg, 85 if valid else 30
    
    elif field_type == 'name':
        valid, msg = validate_name(value)
        return valid, msg, 85 if valid else 30
    
    elif field_type == 'plate':
        valid, msg = validate_plate(value)
        return valid, msg, 85 if valid else 30
    
    elif field_type == 'city':
        valid, msg = validate_city(value)
        return valid, msg, 80 if valid else 40
    
    else:
        # Unknown field type - basic validation
        return True, None, 60


# =============================================================================
# DUPLICATE VALUE DETECTION
# =============================================================================

def detect_duplicate_values(
    fields: Dict[str, str],
    threshold: int = 3
) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Detect duplicate values that appear in multiple unrelated fields.
    
    This is a key hallucination indicator - when the model can't read
    the form, it often copies one value to all fields.
    
    Args:
        fields: Dict of field_name -> value
        threshold: Minimum occurrences to flag as duplicate
    
    Returns:
        Tuple of (duplicate_map, warning_messages)
    """
    duplicates: Dict[str, List[str]] = {}
    warnings: List[str] = []
    
    # Skip markers and empty values
    value_to_fields: Dict[str, List[str]] = {}
    for field_name, value in fields.items():
        if not value or is_valid_marker(value):
            continue
        
        # Normalize value for comparison
        normalized = normalize_arabic_digits(value.strip().lower())
        
        if normalized not in value_to_fields:
            value_to_fields[normalized] = []
        value_to_fields[normalized].append(field_name)
    
    # Find duplicates
    for value, field_names in value_to_fields.items():
        if len(field_names) >= threshold:
            duplicates[value] = field_names
            warnings.append(
                f"CRITICAL: Value '{value}' appears in {len(field_names)} fields: "
                f"{', '.join(field_names)}"
            )
    
    # Check for year values in non-date fields
    for value, field_names in value_to_fields.items():
        if is_hijri_year(value) or is_gregorian_year(value):
            non_date_fields = [
                f for f in field_names 
                if get_field_type(f) not in ('date', None)
            ]
            if non_date_fields:
                warnings.append(
                    f"SUSPICIOUS: Year value '{value}' in non-date fields: "
                    f"{', '.join(non_date_fields)}"
                )
    
    return duplicates, warnings


# =============================================================================
# SEMANTIC SANITY CHECKS
# =============================================================================

def check_semantic_sanity(fields: Dict[str, str]) -> List[ValidationIssue]:
    """
    Check for semantic inconsistencies that indicate hallucination.
    
    Rules:
    - ID and phone should be different
    - Date fields should have date-like values
    - Name fields should have Arabic text
    - Numeric fields shouldn't have the same value
    """
    issues: List[ValidationIssue] = []
    
    # Extract key fields
    id_value = None
    phone_value = None
    date_values: List[Tuple[str, str]] = []
    name_values: List[Tuple[str, str]] = []
    numeric_values: List[Tuple[str, str]] = []
    
    for field_name, value in fields.items():
        if is_valid_marker(value) or not value:
            continue
            
        field_type = get_field_type(field_name)
        
        if field_type == 'national_id':
            id_value = (field_name, value)
        elif field_type == 'mobile':
            phone_value = (field_name, value)
        elif field_type == 'date':
            date_values.append((field_name, value))
        elif field_type == 'name':
            name_values.append((field_name, value))
        
        # Track all numeric values
        digits = extract_digits(value)
        if len(digits) >= 4:
            numeric_values.append((field_name, digits))
    
    # Check: ID and phone should be different
    if id_value and phone_value:
        id_digits = extract_digits(id_value[1])
        phone_digits = extract_digits(phone_value[1])
        if id_digits == phone_digits:
            issues.append(ValidationIssue(
                field_name=f"{id_value[0]}, {phone_value[0]}",
                issue_type="same_value_different_fields",
                severity=ValidationSeverity.CRITICAL,
                message="ID and phone number have identical values (hallucination)",
                expected="Different values",
                actual=id_digits
            ))
    
    # Check: Date fields should not contain just a year
    for field_name, value in date_values:
        if PATTERNS['year_only'].match(normalize_arabic_digits(value)):
            # Check if this year appears in other non-date fields
            year_digits = extract_digits(value)
            for other_name, other_digits in numeric_values:
                if other_name != field_name and other_digits == year_digits:
                    issues.append(ValidationIssue(
                        field_name=field_name,
                        issue_type="year_propagation",
                        severity=ValidationSeverity.ERROR,
                        message=f"Date field has year-only value that matches other fields",
                        expected="Full date (DD/MM/YYYY)",
                        actual=value
                    ))
                    break
    
    # Check: Name fields should have Arabic text, not numbers
    for field_name, value in name_values:
        digits = extract_digits(value)
        arabic_count = count_arabic_words(value)
        if len(digits) > arabic_count * 3:  # More digits than expected
            issues.append(ValidationIssue(
                field_name=field_name,
                issue_type="name_has_numbers",
                severity=ValidationSeverity.ERROR,
                message="Name field contains mostly numbers",
                expected="Arabic text",
                actual=value
            ))
    
    # Check: Multiple numeric fields with same value
    if len(numeric_values) >= 3:
        value_counts = Counter(v for _, v in numeric_values)
        for value, count in value_counts.items():
            if count >= 3:
                affected_fields = [f for f, v in numeric_values if v == value]
                issues.append(ValidationIssue(
                    field_name=", ".join(affected_fields),
                    issue_type="numeric_value_propagation",
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Same numeric value in {count} different fields (hallucination)",
                    expected="Different values per field",
                    actual=value
                ))
    
    return issues


# =============================================================================
# HALLUCINATION PATTERN DETECTION
# =============================================================================

def detect_hallucination_patterns(
    fields: Dict[str, str],
    raw_text: str = ""
) -> List[str]:
    """
    Detect common hallucination patterns.
    
    Patterns:
    1. Value propagation - same value in many fields
    2. Year-as-value - year numbers used for non-date fields
    3. Template filling - generic placeholder-like values
    4. No Arabic content - all values are numbers
    5. Sequential numbers - 1234567890 patterns
    """
    indicators: List[str] = []
    
    # Count various metrics
    total_fields = len(fields)
    non_empty_fields = sum(1 for v in fields.values() if v and not is_valid_marker(v))
    
    if non_empty_fields == 0:
        indicators.append("NO_CONTENT: All fields are empty or marked unclear")
        return indicators
    
    # Collect all non-empty values
    values = [v.strip() for v in fields.values() if v and not is_valid_marker(v)]
    unique_values = set(normalize_arabic_digits(v.lower()) for v in values)
    
    # Pattern 1: Value diversity check
    diversity_ratio = len(unique_values) / len(values) if values else 0
    if diversity_ratio < 0.5:
        indicators.append(
            f"LOW_DIVERSITY: Only {len(unique_values)}/{len(values)} unique values "
            f"({diversity_ratio:.0%} diversity)"
        )
    
    # Pattern 2: Year-as-value
    year_count = 0
    non_date_year_fields = []
    for field_name, value in fields.items():
        if value and (is_hijri_year(value) or is_gregorian_year(value)):
            year_count += 1
            if get_field_type(field_name) != 'date':
                non_date_year_fields.append(field_name)
    
    if non_date_year_fields and len(non_date_year_fields) >= 2:
        indicators.append(
            f"YEAR_AS_VALUE: Year values in {len(non_date_year_fields)} non-date fields: "
            f"{', '.join(non_date_year_fields[:5])}"
        )
    
    # Pattern 3: No Arabic names when expected
    name_fields = [
        (f, v) for f, v in fields.items() 
        if get_field_type(f) == 'name' and v and not is_valid_marker(v)
    ]
    if name_fields:
        arabic_names = sum(1 for _, v in name_fields if count_arabic_words(v) >= 2)
        if arabic_names == 0:
            indicators.append(
                f"NO_ARABIC_NAMES: {len(name_fields)} name fields but no Arabic names found"
            )
    
    # Pattern 4: Sequential/round numbers
    for field_name, value in fields.items():
        digits = extract_digits(value)
        if digits in ('1234567890', '0987654321', '1111111111', '0000000000'):
            indicators.append(f"SEQUENTIAL_NUMBER: Field '{field_name}' has sequential digits")
        if digits and len(digits) >= 4 and len(set(digits)) == 1:
            indicators.append(f"REPEATED_DIGIT: Field '{field_name}' has all same digits: {digits}")
    
    # Pattern 5: All numeric (no Arabic text at all)
    total_arabic_words = sum(count_arabic_words(v) for v in values)
    if total_arabic_words < 2 and non_empty_fields >= 5:
        indicators.append(
            f"NO_ARABIC_TEXT: Only {total_arabic_words} Arabic words in {non_empty_fields} fields"
        )
    
    return indicators


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def validate_extraction(
    fields: Dict[str, str],
    raw_text: str = "",
    strict_mode: bool = True
) -> ValidationResult:
    """
    Comprehensive validation of OCR extraction.
    
    Args:
        fields: Dict of field_name -> extracted_value
        raw_text: Original raw OCR text (for additional context)
        strict_mode: If True, use stricter thresholds
    
    Returns:
        ValidationResult with quality score and all issues
    """
    issues: List[ValidationIssue] = []
    warnings: List[str] = []
    field_scores: Dict[str, int] = {}
    
    # Phase 1: Validate each field individually
    for field_name, value in fields.items():
        is_valid, error_msg, score = validate_field(field_name, value)
        field_scores[field_name] = score
        
        if not is_valid and error_msg:
            issues.append(ValidationIssue(
                field_name=field_name,
                issue_type="format_violation",
                severity=ValidationSeverity.ERROR,
                message=error_msg,
                actual=value
            ))
    
    # Phase 2: Detect duplicate values
    duplicates, dup_warnings = detect_duplicate_values(
        fields,
        threshold=3 if strict_mode else 4
    )
    warnings.extend(dup_warnings)
    
    for value, field_names in duplicates.items():
        issues.append(ValidationIssue(
            field_name=", ".join(field_names),
            issue_type="duplicate_value",
            severity=ValidationSeverity.CRITICAL,
            message=f"Same value '{value}' in {len(field_names)} fields",
            actual=value
        ))
    
    # Phase 3: Semantic sanity checks
    sanity_issues = check_semantic_sanity(fields)
    issues.extend(sanity_issues)
    
    # Phase 4: Hallucination pattern detection
    hallucination_indicators = detect_hallucination_patterns(fields, raw_text)
    for indicator in hallucination_indicators:
        warnings.append(f"HALLUCINATION_INDICATOR: {indicator}")
    
    # Calculate quality score
    quality_score = calculate_quality_score(
        fields=fields,
        issues=issues,
        duplicates=duplicates,
        hallucination_indicators=hallucination_indicators,
        field_scores=field_scores
    )
    
    # Determine overall validity
    is_valid = quality_score >= (40 if strict_mode else 30)
    
    return ValidationResult(
        is_valid=is_valid,
        quality_score=quality_score,
        issues=issues,
        warnings=warnings,
        field_scores=field_scores,
        duplicate_values=duplicates,
        hallucination_indicators=hallucination_indicators
    )


def calculate_quality_score(
    fields: Dict[str, str],
    issues: List[ValidationIssue],
    duplicates: Dict[str, List[str]],
    hallucination_indicators: List[str],
    field_scores: Dict[str, int]
) -> int:
    """
    Calculate overall quality score (0-100).
    
    Scoring:
    - Start at 100
    - Deduct for critical issues: -30 each
    - Deduct for errors: -15 each
    - Deduct for warnings: -5 each
    - Deduct for low diversity: -20
    - Deduct for hallucination indicators: -10 each
    - Bonus for good field scores: +5 for each >80
    """
    score = 100
    
    # Deduct for issues by severity
    for issue in issues:
        if issue.severity == ValidationSeverity.CRITICAL:
            score -= 30
        elif issue.severity == ValidationSeverity.ERROR:
            score -= 15
        elif issue.severity == ValidationSeverity.WARNING:
            score -= 5
    
    # Deduct for duplicates (value propagation)
    total_duplicate_fields = sum(len(fields) for fields in duplicates.values())
    if total_duplicate_fields > 0:
        dup_ratio = total_duplicate_fields / len(fields) if fields else 0
        if dup_ratio > 0.5:
            score -= 30  # More than half fields have duplicate values
        elif dup_ratio > 0.3:
            score -= 20
        elif dup_ratio > 0.1:
            score -= 10
    
    # Deduct for hallucination indicators
    score -= len(hallucination_indicators) * 10
    
    # Bonus for good individual field scores
    good_fields = sum(1 for s in field_scores.values() if s >= 80)
    score += good_fields * 2  # Small bonus
    
    # Ensure score is in valid range
    return max(0, min(100, score))


# =============================================================================
# EXTRACTION PARSER
# =============================================================================

def parse_extraction_to_fields(text: str) -> Dict[str, str]:
    """
    Parse raw OCR extraction text into field dictionary.
    
    Expected format:
        field_name: value [CONFIDENCE]
        field_name: [فارغ] [HIGH]
    """
    fields: Dict[str, str] = {}
    
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue
        
        # Split on first colon
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue
        
        field_name = parts[0].strip()
        rest = parts[1].strip()
        
        # Remove confidence markers
        for marker in ['[HIGH]', '[MEDIUM]', '[LOW]', '[high]', '[medium]', '[low]']:
            rest = rest.replace(marker, '')
        
        value = rest.strip()
        fields[field_name] = value
    
    return fields


def validate_raw_extraction(
    raw_text: str,
    strict_mode: bool = True
) -> ValidationResult:
    """
    Validate raw OCR extraction text.
    
    Convenience function that parses text and validates.
    """
    fields = parse_extraction_to_fields(raw_text)
    return validate_extraction(fields, raw_text, strict_mode)
