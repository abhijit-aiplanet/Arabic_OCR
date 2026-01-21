"""
Confidence Scoring System for Arabic OCR

Provides confidence scoring, calibration, and routing decisions for OCR extractions.

Key Features:
- Token-level confidence from model output
- Field-level confidence aggregation
- Validation-based confidence adjustment
- Routing decisions (auto-accept, flag, require review)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal, Tuple
from enum import Enum
import re
import math


# =============================================================================
# CONFIDENCE LEVELS AND THRESHOLDS
# =============================================================================

class ConfidenceLevel(Enum):
    """Confidence level categories for routing decisions."""
    HIGH = "high"           # > 0.85 - Auto-accept
    MEDIUM = "medium"       # 0.60-0.85 - Flag for optional review
    LOW = "low"             # 0.40-0.60 - Require human verification
    VERY_LOW = "very_low"   # < 0.40 - Mark as unreliable
    EMPTY = "empty"         # Field is blank (high confidence of emptiness)
    UNREADABLE = "unreadable"  # Cannot be read at all


@dataclass
class ConfidenceThresholds:
    """Configurable thresholds for confidence routing."""
    auto_accept: float = 0.85      # High confidence - accept automatically
    flag_review: float = 0.60      # Medium - flag for optional review
    require_review: float = 0.40   # Low - require human verification
    reject: float = 0.20           # Below this - mark as unreadable
    
    def get_level(self, score: float) -> ConfidenceLevel:
        """Get confidence level for a given score."""
        if score >= self.auto_accept:
            return ConfidenceLevel.HIGH
        elif score >= self.flag_review:
            return ConfidenceLevel.MEDIUM
        elif score >= self.require_review:
            return ConfidenceLevel.LOW
        elif score >= self.reject:
            return ConfidenceLevel.VERY_LOW
        else:
            return ConfidenceLevel.UNREADABLE


# Default thresholds
DEFAULT_THRESHOLDS = ConfidenceThresholds()


# =============================================================================
# CONFIDENCE SCORING DATA STRUCTURES
# =============================================================================

@dataclass
class TokenConfidence:
    """Confidence score for a single token."""
    token: str
    probability: float
    is_arabic: bool = False
    is_digit: bool = False


@dataclass
class FieldConfidence:
    """Comprehensive confidence information for a single field."""
    field_name: str
    value: str
    
    # Core confidence scores
    base_confidence: float          # Raw confidence from model
    adjusted_confidence: float      # After validation adjustments
    final_confidence: float         # Final score after all factors
    
    # Confidence level and routing
    level: ConfidenceLevel
    needs_review: bool
    auto_accepted: bool
    
    # Token-level details (if available)
    token_confidences: List[TokenConfidence] = field(default_factory=list)
    min_token_confidence: float = 0.0
    avg_token_confidence: float = 0.0
    
    # Validation impact
    validation_adjustment: float = 0.0
    validation_reason: Optional[str] = None
    
    # Special markers
    is_empty: bool = False
    is_unclear: bool = False
    is_unreadable: bool = False
    unclear_guess: Optional[str] = None


@dataclass
class DocumentConfidence:
    """Confidence summary for an entire document."""
    overall_confidence: float
    field_confidences: List[FieldConfidence]
    
    # Summary statistics
    total_fields: int = 0
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0
    empty_count: int = 0
    unreadable_count: int = 0
    
    # Review routing
    needs_any_review: bool = False
    fields_requiring_review: List[str] = field(default_factory=list)
    review_reason: Optional[str] = None


# =============================================================================
# CONFIDENCE CALCULATOR
# =============================================================================

class ConfidenceCalculator:
    """
    Calculates and adjusts confidence scores for OCR extractions.
    
    Features:
    - Token-level confidence aggregation
    - Validation-based adjustment
    - Field-type specific calibration
    - Hallucination detection penalty
    """
    
    def __init__(
        self,
        thresholds: Optional[ConfidenceThresholds] = None,
        validators: Optional[Any] = None
    ):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.validators = validators
    
    def calculate_field_confidence(
        self,
        field_name: str,
        value: str,
        token_probs: Optional[List[float]] = None,
        is_empty_marker: bool = False,
        is_unclear_marker: bool = False,
        is_unreadable_marker: bool = False,
        unclear_guess: Optional[str] = None
    ) -> FieldConfidence:
        """
        Calculate comprehensive confidence for a single field.
        
        Args:
            field_name: Name of the field
            value: Extracted value
            token_probs: List of token probabilities from model (if available)
            is_empty_marker: Whether the value contains empty marker [فارغ]
            is_unclear_marker: Whether the value contains unclear marker
            is_unreadable_marker: Whether the value contains unreadable marker
            unclear_guess: The guessed value if unclear
            
        Returns:
            FieldConfidence with all scores and routing decisions
        """
        
        # Handle special markers first
        if is_empty_marker:
            return FieldConfidence(
                field_name=field_name,
                value="",
                base_confidence=1.0,
                adjusted_confidence=1.0,
                final_confidence=1.0,
                level=ConfidenceLevel.EMPTY,
                needs_review=False,
                auto_accepted=True,
                is_empty=True
            )
        
        if is_unreadable_marker:
            return FieldConfidence(
                field_name=field_name,
                value="",
                base_confidence=0.0,
                adjusted_confidence=0.0,
                final_confidence=0.0,
                level=ConfidenceLevel.UNREADABLE,
                needs_review=True,
                auto_accepted=False,
                is_unreadable=True
            )
        
        if is_unclear_marker:
            return FieldConfidence(
                field_name=field_name,
                value=unclear_guess or "",
                base_confidence=0.5,
                adjusted_confidence=0.5,
                final_confidence=0.5,
                level=ConfidenceLevel.MEDIUM,
                needs_review=True,
                auto_accepted=False,
                is_unclear=True,
                unclear_guess=unclear_guess
            )
        
        # Calculate base confidence from token probabilities
        if token_probs and len(token_probs) > 0:
            base_confidence = self._aggregate_token_confidence(token_probs)
            min_token = min(token_probs)
            avg_token = sum(token_probs) / len(token_probs)
        else:
            # Default confidence when no token probs available
            base_confidence = 0.75
            min_token = 0.75
            avg_token = 0.75
        
        # Apply validation adjustments
        adjusted_confidence = base_confidence
        validation_adjustment = 0.0
        validation_reason = None
        
        if self.validators:
            validation_result = self.validators.validate_field(field_name, value)
            
            if validation_result.is_valid:
                # Boost confidence for valid patterns
                validation_adjustment = 0.10
                validation_reason = "Matches expected pattern"
            elif validation_result.is_suspicious:
                # Significant penalty for suspicious (likely hallucinated) values
                validation_adjustment = -0.30
                validation_reason = f"Suspicious: {validation_result.reason}"
            elif validation_result.is_invalid:
                # Moderate penalty for invalid format
                validation_adjustment = -0.15
                validation_reason = f"Invalid format: {validation_result.reason}"
        
        adjusted_confidence = max(0.0, min(1.0, base_confidence + validation_adjustment))
        
        # Apply field-type specific calibration
        final_confidence = self._calibrate_for_field_type(field_name, adjusted_confidence, value)
        
        # Determine confidence level and routing
        level = self.thresholds.get_level(final_confidence)
        needs_review = level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW, ConfidenceLevel.UNREADABLE]
        auto_accepted = level == ConfidenceLevel.HIGH
        
        return FieldConfidence(
            field_name=field_name,
            value=value,
            base_confidence=base_confidence,
            adjusted_confidence=adjusted_confidence,
            final_confidence=final_confidence,
            level=level,
            needs_review=needs_review,
            auto_accepted=auto_accepted,
            min_token_confidence=min_token,
            avg_token_confidence=avg_token,
            validation_adjustment=validation_adjustment,
            validation_reason=validation_reason
        )
    
    def calculate_document_confidence(
        self,
        field_confidences: List[FieldConfidence]
    ) -> DocumentConfidence:
        """
        Calculate overall confidence for a document from field confidences.
        """
        if not field_confidences:
            return DocumentConfidence(
                overall_confidence=0.0,
                field_confidences=[],
                total_fields=0,
                needs_any_review=False
            )
        
        # Count by level
        high_count = sum(1 for f in field_confidences if f.level == ConfidenceLevel.HIGH)
        medium_count = sum(1 for f in field_confidences if f.level == ConfidenceLevel.MEDIUM)
        low_count = sum(1 for f in field_confidences if f.level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW])
        empty_count = sum(1 for f in field_confidences if f.level == ConfidenceLevel.EMPTY)
        unreadable_count = sum(1 for f in field_confidences if f.level == ConfidenceLevel.UNREADABLE)
        
        # Calculate overall confidence (weighted average, excluding empty fields)
        non_empty_fields = [f for f in field_confidences if not f.is_empty]
        if non_empty_fields:
            overall_confidence = sum(f.final_confidence for f in non_empty_fields) / len(non_empty_fields)
        else:
            overall_confidence = 1.0  # All fields are empty, high confidence of that
        
        # Determine review needs
        fields_requiring_review = [f.field_name for f in field_confidences if f.needs_review]
        needs_any_review = len(fields_requiring_review) > 0
        
        review_reason = None
        if needs_any_review:
            review_reason = f"{len(fields_requiring_review)} field(s) require human review"
        
        return DocumentConfidence(
            overall_confidence=overall_confidence,
            field_confidences=field_confidences,
            total_fields=len(field_confidences),
            high_confidence_count=high_count,
            medium_confidence_count=medium_count,
            low_confidence_count=low_count,
            empty_count=empty_count,
            unreadable_count=unreadable_count,
            needs_any_review=needs_any_review,
            fields_requiring_review=fields_requiring_review,
            review_reason=review_reason
        )
    
    def _aggregate_token_confidence(self, token_probs: List[float]) -> float:
        """
        Aggregate token-level probabilities into a single confidence score.
        
        Uses geometric mean to penalize low-confidence tokens more heavily.
        """
        if not token_probs:
            return 0.0
        
        # Geometric mean (more sensitive to low values)
        log_sum = sum(math.log(max(p, 1e-10)) for p in token_probs)
        geometric_mean = math.exp(log_sum / len(token_probs))
        
        # Also consider minimum token confidence
        min_prob = min(token_probs)
        
        # Weighted combination: geometric mean + penalty for very low tokens
        confidence = geometric_mean * 0.7 + min_prob * 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def _calibrate_for_field_type(
        self,
        field_name: str,
        confidence: float,
        value: str
    ) -> float:
        """
        Apply field-type specific calibration.
        
        Some fields are harder to OCR correctly (e.g., handwritten numbers).
        """
        field_name_lower = field_name.lower()
        
        # Numeric fields - slightly lower confidence (digits can be misread)
        if any(keyword in field_name_lower for keyword in ['رقم', 'هوية', 'هاتف', 'جوال', 'لوحة']):
            # If value is mostly digits, slightly reduce confidence
            digit_ratio = sum(1 for c in value if c.isdigit() or c in '٠١٢٣٤٥٦٧٨٩') / max(len(value), 1)
            if digit_ratio > 0.7:
                confidence *= 0.95  # Small reduction for digit-heavy fields
        
        # Date fields - check format consistency
        if any(keyword in field_name_lower for keyword in ['تاريخ', 'ميلاد', 'إصدار', 'انتهاء']):
            # Slight reduction for dates (format variations)
            confidence *= 0.97
        
        # Name fields - generally more reliable
        if any(keyword in field_name_lower for keyword in ['اسم', 'مالك', 'صاحب']):
            # Names are usually more readable
            confidence = min(confidence * 1.02, 1.0)
        
        return confidence


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def confidence_to_dict(conf: FieldConfidence) -> Dict[str, Any]:
    """Convert FieldConfidence to dictionary for JSON serialization."""
    return {
        "field_name": conf.field_name,
        "value": conf.value,
        "confidence": {
            "base": round(conf.base_confidence, 3),
            "adjusted": round(conf.adjusted_confidence, 3),
            "final": round(conf.final_confidence, 3),
        },
        "level": conf.level.value,
        "needs_review": conf.needs_review,
        "auto_accepted": conf.auto_accepted,
        "is_empty": conf.is_empty,
        "is_unclear": conf.is_unclear,
        "is_unreadable": conf.is_unreadable,
        "validation": {
            "adjustment": round(conf.validation_adjustment, 3),
            "reason": conf.validation_reason,
        } if conf.validation_adjustment != 0 else None,
    }


def document_confidence_to_dict(doc_conf: DocumentConfidence) -> Dict[str, Any]:
    """Convert DocumentConfidence to dictionary for JSON serialization."""
    return {
        "overall_confidence": round(doc_conf.overall_confidence, 3),
        "total_fields": doc_conf.total_fields,
        "summary": {
            "high": doc_conf.high_confidence_count,
            "medium": doc_conf.medium_confidence_count,
            "low": doc_conf.low_confidence_count,
            "empty": doc_conf.empty_count,
            "unreadable": doc_conf.unreadable_count,
        },
        "needs_review": doc_conf.needs_any_review,
        "fields_requiring_review": doc_conf.fields_requiring_review,
        "review_reason": doc_conf.review_reason,
        "fields": [confidence_to_dict(f) for f in doc_conf.field_confidences],
    }


def get_confidence_color(level: ConfidenceLevel) -> str:
    """Get a color code for displaying confidence level."""
    colors = {
        ConfidenceLevel.HIGH: "#22c55e",      # Green
        ConfidenceLevel.MEDIUM: "#eab308",    # Yellow
        ConfidenceLevel.LOW: "#f97316",       # Orange
        ConfidenceLevel.VERY_LOW: "#ef4444",  # Red
        ConfidenceLevel.EMPTY: "#9ca3af",     # Gray
        ConfidenceLevel.UNREADABLE: "#dc2626", # Dark Red
    }
    return colors.get(level, "#6b7280")


def get_confidence_label(level: ConfidenceLevel) -> str:
    """Get a human-readable label for confidence level."""
    labels = {
        ConfidenceLevel.HIGH: "High Confidence",
        ConfidenceLevel.MEDIUM: "Medium Confidence",
        ConfidenceLevel.LOW: "Low Confidence - Review Needed",
        ConfidenceLevel.VERY_LOW: "Very Low - Likely Error",
        ConfidenceLevel.EMPTY: "Empty Field",
        ConfidenceLevel.UNREADABLE: "Unreadable",
    }
    return labels.get(level, "Unknown")


def get_confidence_label_arabic(level: ConfidenceLevel) -> str:
    """Get Arabic label for confidence level."""
    labels = {
        ConfidenceLevel.HIGH: "ثقة عالية",
        ConfidenceLevel.MEDIUM: "ثقة متوسطة",
        ConfidenceLevel.LOW: "ثقة منخفضة - يحتاج مراجعة",
        ConfidenceLevel.VERY_LOW: "ثقة ضعيفة جداً",
        ConfidenceLevel.EMPTY: "حقل فارغ",
        ConfidenceLevel.UNREADABLE: "غير مقروء",
    }
    return labels.get(level, "غير معروف")
