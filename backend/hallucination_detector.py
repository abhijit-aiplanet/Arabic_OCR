"""
Hallucination Detection for Arabic OCR (FLAG-ONLY MODE)

Detects likely hallucinated values in OCR output.

IMPORTANT: FLAG-ONLY MODE
=========================
This module ONLY FLAGS suspicious values - it NEVER modifies the original OCR output.
All detection results are returned as metadata (suspicion_score, reasons) that can be
used by the UI to highlight fields for human review.

The original extracted values are ALWAYS preserved unchanged.

Vision Language Models (VLMs) like AIN tend to "complete" forms when uncertain,
generating plausible-looking but incorrect values. This module identifies
suspicious patterns that indicate hallucination.

Key Detection Strategies:
1. Pattern-based detection (sequential digits, repeated patterns)
2. Generic value detection (common placeholder names/values)
3. Format anomaly detection (international format when local expected)
4. Cross-field consistency checking
5. Statistical anomaly detection (too-perfect formatting)

Output Usage:
- is_suspicious: Boolean flag for UI highlighting
- suspicion_score: 0.0-1.0 score for severity
- reasons: Human-readable explanations for review
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple
import re


# =============================================================================
# HALLUCINATION RESULT
# =============================================================================

@dataclass
class HallucinationResult:
    """Result of hallucination detection for a single field."""
    is_suspicious: bool = False
    suspicion_score: float = 0.0  # 0.0 = definitely real, 1.0 = definitely hallucinated
    reasons: List[str] = field(default_factory=list)
    field_name: str = ""
    value: str = ""


@dataclass
class DocumentHallucinationReport:
    """Hallucination report for entire document."""
    overall_suspicion_score: float = 0.0
    suspicious_fields: List[HallucinationResult] = field(default_factory=list)
    total_fields: int = 0
    high_suspicion_count: int = 0  # > 0.7
    medium_suspicion_count: int = 0  # 0.4-0.7
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# HALLUCINATION PATTERNS
# =============================================================================

class HallucinationPatterns:
    """Known patterns that indicate hallucinated content."""
    
    # Sequential digit patterns (very common in hallucination)
    SEQUENTIAL_PATTERNS = [
        r"^0?123456789\d*$",      # 0123456789, 123456789x
        r"^987654321\d*$",        # 9876543210
        r"^1234567890$",          # Exact 10 digits ascending
        r"^0987654321$",          # Exact 10 digits descending
        r"^(\d)\1+$",             # All same digit (1111111111)
    ]
    
    # Too-perfect patterns (suspiciously well-formatted)
    TOO_PERFECT_PATTERNS = [
        r"^00966\d{9}$",          # International format with exact digits
        r"^\+966\d{9}$",          # International with plus
        r"^SA\d{22}$",            # Perfect IBAN format (might be template)
    ]
    
    # Generic Arabic names (common LLM completions)
    GENERIC_ARABIC_NAMES = [
        "محمد أحمد",
        "أحمد محمد",
        "عبدالله محمد",
        "محمد عبدالله",
        "عبدالرحمن",
        "فهد",
        "سالم",
        "خالد أحمد",
        "سعود",
        "عمر أحمد",
    ]
    
    # Generic city names (default completions)
    GENERIC_CITIES = [
        "الرياض",
        "جدة",
        "مكة المكرمة",
        "المدينة المنورة",
    ]
    
    # Generic district names
    GENERIC_DISTRICTS = [
        "حي الخير",
        "حي النسيم",
        "حي العليا",
        "حي السلام",
        "حي الملك فهد",
    ]
    
    # Placeholder-like values
    PLACEHOLDER_VALUES = [
        "غير محدد",
        "لا يوجد",
        "غير متوفر",
        "-",
        "...",
        "N/A",
        "n/a",
        "none",
        "null",
    ]
    
    # Test/example patterns
    TEST_PATTERNS = [
        r"test",
        r"example",
        r"sample",
        r"demo",
        r"placeholder",
        r"xxx+",
        r"yyy+",
    ]


# =============================================================================
# HALLUCINATION DETECTOR
# =============================================================================

class HallucinationDetector:
    """
    Detects likely hallucinated values in OCR output.
    
    Uses multiple detection strategies:
    1. Pattern matching against known hallucination patterns
    2. Statistical analysis of value characteristics
    3. Cross-field consistency checking
    4. Field-type specific validation
    """
    
    def __init__(self):
        # Compile patterns for efficiency
        self._sequential_patterns = [
            re.compile(p) for p in HallucinationPatterns.SEQUENTIAL_PATTERNS
        ]
        self._too_perfect_patterns = [
            re.compile(p) for p in HallucinationPatterns.TOO_PERFECT_PATTERNS
        ]
        self._test_patterns = [
            re.compile(p, re.IGNORECASE) for p in HallucinationPatterns.TEST_PATTERNS
        ]
        
        # Convert lists to sets for O(1) lookup
        self._generic_names = set(HallucinationPatterns.GENERIC_ARABIC_NAMES)
        self._generic_cities = set(HallucinationPatterns.GENERIC_CITIES)
        self._generic_districts = set(HallucinationPatterns.GENERIC_DISTRICTS)
        self._placeholder_values = set(HallucinationPatterns.PLACEHOLDER_VALUES)
    
    def detect(self, field_name: str, value: str) -> HallucinationResult:
        """
        Detect if a value is likely hallucinated.
        
        Args:
            field_name: Name of the field
            value: The extracted value
            
        Returns:
            HallucinationResult with suspicion score and reasons
        """
        if not value or value.strip() == "":
            return HallucinationResult(
                is_suspicious=False,
                suspicion_score=0.0,
                field_name=field_name,
                value=value
            )
        
        reasons = []
        suspicion_score = 0.0
        
        value_clean = value.strip()
        value_normalized = self._normalize_value(value_clean)
        
        # Check 1: Sequential digit patterns (high confidence)
        seq_score, seq_reason = self._check_sequential_patterns(value_normalized)
        if seq_score > 0:
            reasons.append(seq_reason)
            suspicion_score += seq_score * 0.4  # Weight: 40%
        
        # Check 2: Too-perfect formatting (medium confidence)
        perf_score, perf_reason = self._check_too_perfect(value_normalized)
        if perf_score > 0:
            reasons.append(perf_reason)
            suspicion_score += perf_score * 0.2  # Weight: 20%
        
        # Check 3: Generic values (field-type specific)
        gen_score, gen_reason = self._check_generic_values(field_name, value_clean)
        if gen_score > 0:
            reasons.append(gen_reason)
            suspicion_score += gen_score * 0.25  # Weight: 25%
        
        # Check 4: Test/placeholder patterns
        test_score, test_reason = self._check_test_patterns(value_normalized)
        if test_score > 0:
            reasons.append(test_reason)
            suspicion_score += test_score * 0.15  # Weight: 15%
        
        # Cap at 1.0
        suspicion_score = min(1.0, suspicion_score)
        
        return HallucinationResult(
            is_suspicious=suspicion_score > 0.5,
            suspicion_score=round(suspicion_score, 3),
            reasons=reasons,
            field_name=field_name,
            value=value
        )
    
    def detect_document(
        self,
        fields: Dict[str, str],
        check_consistency: bool = True
    ) -> DocumentHallucinationReport:
        """
        Detect hallucinations across an entire document.
        
        Args:
            fields: Dictionary of field_name -> value
            check_consistency: Whether to check cross-field consistency
            
        Returns:
            DocumentHallucinationReport with overall analysis
        """
        suspicious_fields = []
        high_count = 0
        medium_count = 0
        warnings = []
        
        for field_name, value in fields.items():
            result = self.detect(field_name, value)
            
            if result.suspicion_score > 0.3:  # Only track somewhat suspicious
                suspicious_fields.append(result)
            
            if result.suspicion_score > 0.7:
                high_count += 1
            elif result.suspicion_score > 0.4:
                medium_count += 1
        
        # Cross-field consistency checks
        if check_consistency:
            consistency_warnings = self._check_document_consistency(fields)
            warnings.extend(consistency_warnings)
        
        # Calculate overall score
        total_fields = len(fields)
        if total_fields > 0:
            # Weight by suspicion level
            total_suspicion = sum(f.suspicion_score for f in suspicious_fields)
            overall_score = total_suspicion / total_fields
        else:
            overall_score = 0.0
        
        # Add warning if too many suspicious fields
        if high_count >= 3:
            warnings.append(f"⚠️ High hallucination risk: {high_count} fields highly suspicious")
        elif high_count + medium_count >= 5:
            warnings.append(f"⚠️ Multiple suspicious fields detected ({high_count} high, {medium_count} medium)")
        
        return DocumentHallucinationReport(
            overall_suspicion_score=round(overall_score, 3),
            suspicious_fields=suspicious_fields,
            total_fields=total_fields,
            high_suspicion_count=high_count,
            medium_suspicion_count=medium_count,
            warnings=warnings
        )
    
    def _normalize_value(self, value: str) -> str:
        """Normalize value for pattern matching."""
        # Convert Arabic to Western digits
        arabic_digits = '٠١٢٣٤٥٦٧٨٩'
        western_digits = '0123456789'
        result = value
        for ar, we in zip(arabic_digits, western_digits):
            result = result.replace(ar, we)
        
        # Remove common separators
        result = result.replace(" ", "").replace("-", "").replace("/", "")
        
        return result.lower()
    
    def _check_sequential_patterns(self, value: str) -> Tuple[float, str]:
        """Check for sequential digit patterns."""
        for pattern in self._sequential_patterns:
            if pattern.match(value):
                return 1.0, "Sequential digit pattern (likely placeholder)"
        
        # Check for partial sequences (like 12345 or 98765)
        digits_only = ''.join(c for c in value if c.isdigit())
        if len(digits_only) >= 5:
            # Check for ascending sequence
            for i in range(len(digits_only) - 4):
                segment = digits_only[i:i+5]
                if segment in "0123456789":
                    return 0.7, "Contains ascending digit sequence"
                if segment in "9876543210":
                    return 0.7, "Contains descending digit sequence"
        
        return 0.0, ""
    
    def _check_too_perfect(self, value: str) -> Tuple[float, str]:
        """Check for suspiciously well-formatted values."""
        for pattern in self._too_perfect_patterns:
            if pattern.match(value):
                return 0.6, "Too-perfect formatting (possible template completion)"
        
        return 0.0, ""
    
    def _check_generic_values(self, field_name: str, value: str) -> Tuple[float, str]:
        """Check for generic placeholder values based on field type."""
        field_lower = field_name.lower()
        value_clean = value.strip()
        
        # Check placeholder values first
        if value_clean in self._placeholder_values:
            return 0.8, "Placeholder value detected"
        
        # Name fields
        if any(k in field_lower for k in ["اسم", "مالك", "صاحب", "name"]):
            if value_clean in self._generic_names:
                return 0.6, f"Generic Arabic name: {value_clean}"
            # Check if it's a very short name (likely incomplete)
            if len(value_clean) <= 3 and value_clean in ["محمد", "أحمد", "علي", "عمر"]:
                return 0.5, "Very short generic name"
        
        # City fields
        if any(k in field_lower for k in ["مدينة", "بلد", "city"]):
            if value_clean in self._generic_cities:
                # Lower suspicion for cities (they might be legitimate)
                return 0.3, f"Common city name (may need verification): {value_clean}"
        
        # District fields
        if any(k in field_lower for k in ["حي", "منطقة", "district"]):
            if value_clean in self._generic_districts:
                return 0.5, f"Generic district name: {value_clean}"
        
        return 0.0, ""
    
    def _check_test_patterns(self, value: str) -> Tuple[float, str]:
        """Check for test/example patterns."""
        for pattern in self._test_patterns:
            if pattern.search(value):
                return 0.9, "Test/example pattern detected"
        
        return 0.0, ""
    
    def _check_document_consistency(self, fields: Dict[str, str]) -> List[str]:
        """Check for cross-field consistency issues."""
        warnings = []
        
        # Check if multiple fields have the same value
        value_counts: Dict[str, List[str]] = {}
        for field_name, value in fields.items():
            if value and value.strip():
                clean_value = value.strip()
                if clean_value not in value_counts:
                    value_counts[clean_value] = []
                value_counts[clean_value].append(field_name)
        
        # Flag if same value appears in multiple unrelated fields
        for value, field_names in value_counts.items():
            if len(field_names) >= 3:
                warnings.append(
                    f"Same value '{value}' appears in {len(field_names)} fields: {', '.join(field_names)}"
                )
        
        # Check for suspiciously complete documents (all fields filled)
        total_fields = len(fields)
        filled_fields = sum(1 for v in fields.values() if v and v.strip() and v.strip() not in self._placeholder_values)
        
        if total_fields > 5 and filled_fields == total_fields:
            warnings.append("All fields are filled - verify that blank fields weren't hallucinated")
        
        return warnings


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global detector instance
_default_detector = HallucinationDetector()

def is_suspicious(field_name: str, value: str) -> bool:
    """Quick check if a value is suspicious."""
    result = _default_detector.detect(field_name, value)
    return result.is_suspicious

def get_suspicion_score(field_name: str, value: str) -> float:
    """Get suspicion score for a value."""
    result = _default_detector.detect(field_name, value)
    return result.suspicion_score

def analyze_document(fields: Dict[str, str]) -> DocumentHallucinationReport:
    """Analyze entire document for hallucinations."""
    return _default_detector.detect_document(fields)
