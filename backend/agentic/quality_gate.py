"""
Quality Gate for Arabic OCR System

This module provides quality control that can:
- REJECT extractions that are clearly hallucinated
- WARN about suspicious extractions
- PASS good extractions through

The goal is to "fail closed" - if we're not confident in the extraction,
we should reject it rather than return garbage to the user.

Quality Score Thresholds:
- 70-100: PASS (good quality)
- 40-69: WARN (needs review)
- 0-39: REJECT (likely hallucination)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from .validators import (
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    validate_extraction,
    parse_extraction_to_fields,
    detect_hallucination_patterns,
    detect_duplicate_values,
    is_valid_marker,
    extract_digits,
    count_arabic_words,
    is_hijri_year,
    get_field_type,
)


# =============================================================================
# QUALITY GATE THRESHOLDS
# =============================================================================

class QualityLevel(Enum):
    """Quality level classification."""
    EXCELLENT = "excellent"  # 85-100: High confidence, minimal issues
    GOOD = "good"           # 70-84: Acceptable, minor issues
    WARNING = "warning"     # 40-69: Needs review, moderate issues
    POOR = "poor"           # 20-39: Likely problems, should warn user
    REJECT = "reject"       # 0-19: Definite hallucination, should not use


# Thresholds
THRESHOLDS = {
    "excellent": 85,
    "good": 70,
    "warning": 40,
    "poor": 20,
    "reject": 0,
}


@dataclass
class QualityGateResult:
    """Result of quality gate evaluation."""
    
    # Core decision
    passed: bool                    # Whether extraction should be used
    quality_level: QualityLevel
    quality_score: int              # 0-100
    
    # Detailed results
    validation_result: ValidationResult
    
    # Decision factors
    rejection_reasons: List[str] = field(default_factory=list)
    warning_reasons: List[str] = field(default_factory=list)
    
    # Recommendations
    should_retry: bool = False      # Should try re-extraction
    needs_human_review: bool = False
    fields_to_review: List[str] = field(default_factory=list)
    
    # Metadata
    checks_performed: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "quality_level": self.quality_level.value,
            "quality_score": self.quality_score,
            "rejection_reasons": self.rejection_reasons,
            "warning_reasons": self.warning_reasons,
            "should_retry": self.should_retry,
            "needs_human_review": self.needs_human_review,
            "fields_to_review": self.fields_to_review,
            "checks_performed": self.checks_performed,
            "validation_details": self.validation_result.to_dict(),
        }


# =============================================================================
# HARD REJECTION CHECKS
# =============================================================================

def check_value_propagation(
    fields: Dict[str, str],
    threshold: float = 0.5
) -> Tuple[bool, Optional[str]]:
    """
    Check if same value appears in too many fields.
    
    This is the #1 hallucination pattern - model copies one value everywhere.
    
    Args:
        fields: Field name -> value mapping
        threshold: Ratio of fields with same value to trigger rejection
    
    Returns:
        Tuple of (should_reject, reason)
    """
    if not fields:
        return False, None
    
    # Get non-empty, non-marker values
    values = []
    for field_name, value in fields.items():
        if value and not is_valid_marker(value):
            # Normalize for comparison
            normalized = value.strip().lower()
            values.append((field_name, normalized))
    
    if len(values) < 3:
        return False, None
    
    # Count occurrences
    from collections import Counter
    value_counts = Counter(v for _, v in values)
    
    # Find most common value
    if value_counts:
        most_common_value, count = value_counts.most_common(1)[0]
        ratio = count / len(values)
        
        if ratio >= threshold:
            affected_fields = [f for f, v in values if v == most_common_value]
            return True, (
                f"Value propagation: '{most_common_value[:30]}...' appears in "
                f"{count}/{len(values)} fields ({ratio:.0%}): {', '.join(affected_fields[:5])}"
            )
    
    return False, None


def check_all_years(
    fields: Dict[str, str],
    threshold: int = 3
) -> Tuple[bool, Optional[str]]:
    """
    Check if year values are used for non-date fields.
    
    Common hallucination: model sees a date and uses the year for everything.
    
    Args:
        fields: Field name -> value mapping
        threshold: Number of non-date fields with years to trigger rejection
    
    Returns:
        Tuple of (should_reject, reason)
    """
    year_in_non_date = []
    
    for field_name, value in fields.items():
        if not value or is_valid_marker(value):
            continue
        
        field_type = get_field_type(field_name)
        
        # Skip date fields - years are expected there
        if field_type == 'date':
            continue
        
        # Check if value is just a year
        digits = extract_digits(value)
        if len(digits) == 4 and is_hijri_year(value):
            year_in_non_date.append((field_name, digits))
    
    if len(year_in_non_date) >= threshold:
        fields_str = ", ".join(f"{f}={v}" for f, v in year_in_non_date[:5])
        return True, (
            f"Year values in {len(year_in_non_date)} non-date fields: {fields_str}"
        )
    
    return False, None


def check_no_arabic_content(
    fields: Dict[str, str],
    min_arabic_words: int = 2
) -> Tuple[bool, Optional[str]]:
    """
    Check if extraction has Arabic content.
    
    Arabic forms should have Arabic text - if there's none, something is wrong.
    
    Args:
        fields: Field name -> value mapping
        min_arabic_words: Minimum Arabic words expected
    
    Returns:
        Tuple of (should_reject, reason)
    """
    total_arabic_words = 0
    
    for value in fields.values():
        if value and not is_valid_marker(value):
            total_arabic_words += count_arabic_words(value)
    
    if total_arabic_words < min_arabic_words:
        return True, (
            f"No Arabic content: Only {total_arabic_words} Arabic words found "
            f"(expected at least {min_arabic_words})"
        )
    
    return False, None


def check_all_identical_numeric(
    fields: Dict[str, str]
) -> Tuple[bool, Optional[str]]:
    """
    Check if all numeric fields have the same value.
    
    This catches patterns like all fields = "1400".
    
    Returns:
        Tuple of (should_reject, reason)
    """
    numeric_values = []
    
    for field_name, value in fields.items():
        if not value or is_valid_marker(value):
            continue
        
        digits = extract_digits(value)
        if len(digits) >= 4:
            numeric_values.append((field_name, digits))
    
    if len(numeric_values) < 3:
        return False, None
    
    # Check if all same
    unique_digits = set(d for _, d in numeric_values)
    if len(unique_digits) == 1:
        common_value = numeric_values[0][1]
        return True, (
            f"All {len(numeric_values)} numeric fields have identical value: {common_value}"
        )
    
    return False, None


def check_suspicious_patterns(
    fields: Dict[str, str]
) -> Tuple[bool, Optional[str]]:
    """
    Check for obviously fake/sequential numbers.
    
    Returns:
        Tuple of (should_reject, reason)
    """
    suspicious_patterns = [
        '1234567890', '0987654321', '1111111111', '2222222222',
        '0000000000', '9999999999', '1234512345', '1000000000',
        '2000000000', '3000000000',
    ]
    
    found_patterns = []
    
    for field_name, value in fields.items():
        if not value:
            continue
        
        digits = extract_digits(value)
        
        for pattern in suspicious_patterns:
            if pattern in digits or digits == pattern:
                found_patterns.append(f"{field_name}={digits}")
                break
        
        # Check for all same digit
        if len(digits) >= 6 and len(set(digits)) == 1:
            found_patterns.append(f"{field_name}={digits}")
    
    if len(found_patterns) >= 2:
        return True, (
            f"Suspicious sequential/repeated numbers: {', '.join(found_patterns[:5])}"
        )
    
    return False, None


# =============================================================================
# WARNING CHECKS
# =============================================================================

def check_low_diversity(
    fields: Dict[str, str],
    warning_threshold: float = 0.4,
    reject_threshold: float = 0.25
) -> Tuple[bool, bool, Optional[str]]:
    """
    Check value diversity ratio.
    
    Returns:
        Tuple of (should_reject, should_warn, reason)
    """
    values = []
    for value in fields.values():
        if value and not is_valid_marker(value):
            values.append(value.strip().lower())
    
    if len(values) < 3:
        return False, False, None
    
    unique_count = len(set(values))
    diversity = unique_count / len(values)
    
    reason = f"Low value diversity: {unique_count}/{len(values)} unique ({diversity:.0%})"
    
    if diversity < reject_threshold:
        return True, True, reason
    elif diversity < warning_threshold:
        return False, True, reason
    
    return False, False, None


def check_missing_critical_fields(
    fields: Dict[str, str],
    expected_types: List[str] = None
) -> Tuple[bool, Optional[str]]:
    """
    Check if critical field types are missing values.
    
    Returns:
        Tuple of (should_warn, reason)
    """
    if expected_types is None:
        expected_types = ['name', 'national_id']  # Typically required
    
    missing_types = []
    
    for expected_type in expected_types:
        found = False
        for field_name, value in fields.items():
            if get_field_type(field_name) == expected_type:
                if value and not is_valid_marker(value):
                    found = True
                    break
        
        if not found:
            missing_types.append(expected_type)
    
    if missing_types:
        return True, f"Missing critical field types: {', '.join(missing_types)}"
    
    return False, None


def check_format_violations(
    validation_result: ValidationResult,
    warning_threshold: int = 3,
    reject_threshold: int = 5
) -> Tuple[bool, bool, Optional[str]]:
    """
    Check number of format violations.
    
    Returns:
        Tuple of (should_reject, should_warn, reason)
    """
    format_issues = [
        i for i in validation_result.issues
        if i.issue_type == "format_violation"
    ]
    
    count = len(format_issues)
    
    if count >= reject_threshold:
        fields = [i.field_name for i in format_issues[:5]]
        return True, True, f"{count} format violations: {', '.join(fields)}"
    elif count >= warning_threshold:
        fields = [i.field_name for i in format_issues[:5]]
        return False, True, f"{count} format violations: {', '.join(fields)}"
    
    return False, False, None


# =============================================================================
# MAIN QUALITY GATE FUNCTION
# =============================================================================

def evaluate_quality(
    fields: Dict[str, str],
    raw_text: str = "",
    strict_mode: bool = True
) -> QualityGateResult:
    """
    Main quality gate evaluation.
    
    Performs all checks and returns a comprehensive quality assessment.
    
    Args:
        fields: Dict of field_name -> value
        raw_text: Original raw extraction text
        strict_mode: Use stricter thresholds if True
    
    Returns:
        QualityGateResult with pass/fail decision and details
    """
    checks_performed = []
    rejection_reasons = []
    warning_reasons = []
    fields_to_review = []
    
    # Run validation first
    validation_result = validate_extraction(fields, raw_text, strict_mode)
    checks_performed.append("field_validation")
    
    # Hard rejection checks
    
    # 1. Value propagation (most critical)
    reject, reason = check_value_propagation(
        fields, 
        threshold=0.4 if strict_mode else 0.5
    )
    checks_performed.append("value_propagation")
    if reject:
        rejection_reasons.append(reason)
    
    # 2. Years in non-date fields
    reject, reason = check_all_years(
        fields,
        threshold=2 if strict_mode else 3
    )
    checks_performed.append("year_in_non_date")
    if reject:
        rejection_reasons.append(reason)
    
    # 3. No Arabic content
    reject, reason = check_no_arabic_content(
        fields,
        min_arabic_words=2 if strict_mode else 1
    )
    checks_performed.append("arabic_content")
    if reject:
        rejection_reasons.append(reason)
    
    # 4. All identical numeric values
    reject, reason = check_all_identical_numeric(fields)
    checks_performed.append("identical_numeric")
    if reject:
        rejection_reasons.append(reason)
    
    # 5. Suspicious patterns
    reject, reason = check_suspicious_patterns(fields)
    checks_performed.append("suspicious_patterns")
    if reject:
        rejection_reasons.append(reason)
    
    # Warning checks
    
    # 6. Low diversity
    reject, warn, reason = check_low_diversity(
        fields,
        warning_threshold=0.5 if strict_mode else 0.4,
        reject_threshold=0.25 if strict_mode else 0.2
    )
    checks_performed.append("value_diversity")
    if reject:
        rejection_reasons.append(reason)
    elif warn:
        warning_reasons.append(reason)
    
    # 7. Format violations
    reject, warn, reason = check_format_violations(
        validation_result,
        warning_threshold=2 if strict_mode else 3,
        reject_threshold=4 if strict_mode else 5
    )
    checks_performed.append("format_violations")
    if reject:
        rejection_reasons.append(reason)
    elif warn:
        warning_reasons.append(reason)
    
    # 8. Missing critical fields
    warn, reason = check_missing_critical_fields(fields)
    checks_performed.append("critical_fields")
    if warn:
        warning_reasons.append(reason)
    
    # Add hallucination indicators to warnings
    for indicator in validation_result.hallucination_indicators:
        warning_reasons.append(f"Hallucination indicator: {indicator}")
    
    # Collect fields needing review
    for issue in validation_result.issues:
        if issue.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL):
            if issue.field_name not in fields_to_review:
                fields_to_review.append(issue.field_name)
    
    # Calculate final quality score
    # Start with validation score, then apply penalties
    quality_score = validation_result.quality_score
    
    # Additional penalties for rejection reasons
    quality_score -= len(rejection_reasons) * 20
    quality_score -= len(warning_reasons) * 5
    
    # Ensure in range
    quality_score = max(0, min(100, quality_score))
    
    # Determine quality level
    if quality_score >= THRESHOLDS["excellent"]:
        quality_level = QualityLevel.EXCELLENT
    elif quality_score >= THRESHOLDS["good"]:
        quality_level = QualityLevel.GOOD
    elif quality_score >= THRESHOLDS["warning"]:
        quality_level = QualityLevel.WARNING
    elif quality_score >= THRESHOLDS["poor"]:
        quality_level = QualityLevel.POOR
    else:
        quality_level = QualityLevel.REJECT
    
    # Final decision
    passed = len(rejection_reasons) == 0 and quality_score >= THRESHOLDS["warning"]
    should_retry = quality_level in (QualityLevel.REJECT, QualityLevel.POOR)
    needs_human_review = quality_level in (QualityLevel.WARNING, QualityLevel.POOR)
    
    return QualityGateResult(
        passed=passed,
        quality_level=quality_level,
        quality_score=quality_score,
        validation_result=validation_result,
        rejection_reasons=rejection_reasons,
        warning_reasons=warning_reasons,
        should_retry=should_retry,
        needs_human_review=needs_human_review,
        fields_to_review=fields_to_review,
        checks_performed=checks_performed,
    )


def evaluate_raw_extraction(
    raw_text: str,
    strict_mode: bool = True
) -> QualityGateResult:
    """
    Convenience function to evaluate raw OCR text.
    
    Parses the text into fields and runs quality gate.
    """
    fields = parse_extraction_to_fields(raw_text)
    return evaluate_quality(fields, raw_text, strict_mode)


# =============================================================================
# QUALITY IMPROVEMENT SUGGESTIONS
# =============================================================================

def get_improvement_suggestions(
    result: QualityGateResult
) -> List[str]:
    """
    Get suggestions for improving extraction quality.
    
    Based on the issues found, suggest what could be done differently.
    """
    suggestions = []
    
    if result.quality_level == QualityLevel.REJECT:
        suggestions.append(
            "Extraction quality is too low. Consider: "
            "1) Better image quality, 2) Different prompting, 3) Manual entry"
        )
    
    # Based on specific issues
    for reason in result.rejection_reasons:
        if "Value propagation" in reason:
            suggestions.append(
                "Model is copying one value to all fields. "
                "This suggests the handwriting is not being read. "
                "Try: higher resolution image, different angle, or cleaner scan."
            )
        elif "Year values" in reason:
            suggestions.append(
                "Model is using date years for non-date fields. "
                "This suggests confusion about field types. "
                "Try: a template that explicitly marks field boundaries."
            )
        elif "No Arabic content" in reason:
            suggestions.append(
                "No Arabic text detected. "
                "Check: Is the document actually in Arabic? Is the image readable?"
            )
    
    if result.needs_human_review:
        suggestions.append(
            f"Fields needing review: {', '.join(result.fields_to_review)}"
        )
    
    return suggestions


# =============================================================================
# ITERATIVE QUALITY CHECK
# =============================================================================

def compare_extraction_quality(
    extraction_v1: Dict[str, str],
    extraction_v2: Dict[str, str]
) -> Tuple[int, str]:
    """
    Compare quality between two extractions (e.g., before and after refinement).
    
    Returns:
        Tuple of (quality_delta, explanation)
        Positive delta means v2 is better
    """
    result_v1 = evaluate_quality(extraction_v1, strict_mode=False)
    result_v2 = evaluate_quality(extraction_v2, strict_mode=False)
    
    delta = result_v2.quality_score - result_v1.quality_score
    
    if delta > 10:
        explanation = f"Refinement improved quality significantly (+{delta} points)"
    elif delta > 0:
        explanation = f"Refinement improved quality slightly (+{delta} points)"
    elif delta == 0:
        explanation = "No change in quality"
    elif delta > -10:
        explanation = f"Refinement degraded quality slightly ({delta} points)"
    else:
        explanation = f"Refinement degraded quality significantly ({delta} points)"
    
    return delta, explanation


def should_use_refined(
    original: Dict[str, str],
    refined: Dict[str, str],
    min_improvement: int = 5
) -> Tuple[bool, str]:
    """
    Decide whether to use refined extraction over original.
    
    Only use refined if it's actually better.
    
    Returns:
        Tuple of (use_refined, explanation)
    """
    delta, explanation = compare_extraction_quality(original, refined)
    
    if delta >= min_improvement:
        return True, f"Using refined: {explanation}"
    else:
        return False, f"Keeping original: {explanation}"
