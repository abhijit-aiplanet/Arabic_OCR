"""
Pydantic Models for Agentic OCR System

Data structures for:
- Analysis results from LLM
- Region estimates for cropping
- Merge results from multiple OCR passes
- Final agentic OCR output
- Quality scoring and validation results
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class ConfidenceLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FieldSource(Enum):
    ORIGINAL = "original"
    REFINED = "refined"
    MERGED = "merged"


class QualityStatus(Enum):
    """Quality status for extraction results."""
    PASSED = "passed"           # Good quality, can use
    WARNING = "warning"         # Usable but has issues
    FAILED = "failed"           # Quality too low, should not use
    REJECTED = "rejected"       # Definite hallucination detected


class ValidationSeverity(Enum):
    """Severity of validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# VALIDATION MODELS
# =============================================================================

@dataclass
class ValidationIssue:
    """A single validation issue found during quality checking."""
    field_name: str
    issue_type: str
    severity: str  # "info", "warning", "error", "critical"
    message: str
    expected: Optional[str] = None
    actual: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationIssue":
        return cls(
            field_name=data.get("field_name", ""),
            issue_type=data.get("issue_type", ""),
            severity=data.get("severity", "warning"),
            message=data.get("message", ""),
            expected=data.get("expected"),
            actual=data.get("actual"),
        )


@dataclass
class QualityReport:
    """Quality assessment report for an extraction."""
    quality_score: int  # 0-100
    quality_status: str  # "passed", "warning", "failed", "rejected"
    
    # Issue breakdown
    validation_issues: List[ValidationIssue] = field(default_factory=list)
    hallucination_indicators: List[str] = field(default_factory=list)
    
    # Field-level scores
    field_scores: Dict[str, int] = field(default_factory=dict)
    
    # Duplicate detection
    duplicate_values: Dict[str, List[str]] = field(default_factory=dict)
    
    # Recommendations
    should_retry: bool = False
    needs_human_review: bool = False
    fields_to_review: List[str] = field(default_factory=list)
    
    # Summary
    rejection_reasons: List[str] = field(default_factory=list)
    warning_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "quality_score": self.quality_score,
            "quality_status": self.quality_status,
            "validation_issues": [i.to_dict() for i in self.validation_issues],
            "hallucination_indicators": self.hallucination_indicators,
            "field_scores": self.field_scores,
            "duplicate_values": self.duplicate_values,
            "should_retry": self.should_retry,
            "needs_human_review": self.needs_human_review,
            "fields_to_review": self.fields_to_review,
            "rejection_reasons": self.rejection_reasons,
            "warning_reasons": self.warning_reasons,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualityReport":
        return cls(
            quality_score=data.get("quality_score", 0),
            quality_status=data.get("quality_status", "failed"),
            validation_issues=[
                ValidationIssue.from_dict(i) 
                for i in data.get("validation_issues", [])
            ],
            hallucination_indicators=data.get("hallucination_indicators", []),
            field_scores=data.get("field_scores", {}),
            duplicate_values=data.get("duplicate_values", {}),
            should_retry=data.get("should_retry", False),
            needs_human_review=data.get("needs_human_review", False),
            fields_to_review=data.get("fields_to_review", []),
            rejection_reasons=data.get("rejection_reasons", []),
            warning_reasons=data.get("warning_reasons", []),
        )


# =============================================================================
# FIELD MODELS
# =============================================================================

@dataclass
class FieldToReexamine:
    """Field identified for re-examination."""
    field_name: str
    current_value: str
    confidence: str
    issue: str
    priority: str = "medium"
    field_type: Optional[str] = None
    expected_format: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldToReexamine":
        return cls(
            field_name=data.get("field_name", ""),
            current_value=data.get("current_value", ""),
            confidence=data.get("confidence", "LOW"),
            issue=data.get("issue", ""),
            priority=data.get("priority", "medium"),
            field_type=data.get("field_type"),
            expected_format=data.get("expected_format"),
        )


@dataclass
class ConfidentField:
    """Field with high confidence extraction."""
    field_name: str
    value: str
    confidence: str = "HIGH"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfidentField":
        return cls(
            field_name=data.get("field_name", ""),
            value=data.get("value", ""),
            confidence=data.get("confidence", "HIGH"),
        )


@dataclass
class EmptyField:
    """Field confirmed as empty."""
    field_name: str
    status: str = "confirmed_empty"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmptyField":
        return cls(
            field_name=data.get("field_name", ""),
            status=data.get("status", "confirmed_empty"),
        )


# =============================================================================
# ANALYSIS RESULT (from LLM)
# =============================================================================

@dataclass
class AnalysisStats:
    """Statistics from OCR analysis."""
    total_fields: int = 0
    high_confidence: int = 0
    medium_confidence: int = 0
    low_confidence: int = 0
    empty_fields: int = 0
    needs_reexamination: int = 0
    hallucination_detected: bool = False
    hallucination_type: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisStats":
        return cls(
            total_fields=data.get("total_fields", 0),
            high_confidence=data.get("high_confidence", 0),
            medium_confidence=data.get("medium_confidence", 0),
            low_confidence=data.get("low_confidence", 0),
            empty_fields=data.get("empty_fields", 0),
            needs_reexamination=data.get("needs_reexamination", 0),
            hallucination_detected=data.get("hallucination_detected", False),
            hallucination_type=data.get("hallucination_type"),
        )


@dataclass
class AnalysisResult:
    """Complete analysis result from LLM."""
    analysis: AnalysisStats
    fields_to_reexamine: List[FieldToReexamine] = field(default_factory=list)
    confident_fields: List[ConfidentField] = field(default_factory=list)
    empty_fields: List[EmptyField] = field(default_factory=list)
    hallucination_warnings: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        analysis_data = data.get("analysis", {})
        
        return cls(
            analysis=AnalysisStats.from_dict(analysis_data),
            fields_to_reexamine=[
                FieldToReexamine.from_dict(f)
                for f in data.get("fields_to_reexamine", [])
            ],
            confident_fields=[
                ConfidentField.from_dict(f)
                for f in data.get("confident_fields", [])
            ],
            empty_fields=[
                EmptyField.from_dict(f)
                for f in data.get("empty_fields", [])
            ],
            hallucination_warnings=data.get("hallucination_warnings", []),
        )
    
    def has_fields_to_reexamine(self) -> bool:
        """Check if there are fields needing re-examination."""
        return len(self.fields_to_reexamine) > 0
    
    def has_hallucination_warning(self) -> bool:
        """Check if hallucination was detected."""
        return (
            self.analysis.hallucination_detected or 
            len(self.hallucination_warnings) > 0
        )


# =============================================================================
# REGION ESTIMATE (from LLM)
# =============================================================================

@dataclass
class RegionEstimate:
    """Estimated bounding box for a field region."""
    field_name: str
    bbox_normalized: tuple  # (x1, y1, x2, y2) normalized 0-1
    bbox_pixels: Optional[tuple] = None  # (x1, y1, x2, y2) in pixels
    location_confidence: str = "medium"
    notes: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegionEstimate":
        bbox_norm = data.get("bbox_normalized", [0, 0, 1, 1])
        bbox_px = data.get("bbox_pixels")
        
        return cls(
            field_name=data.get("field_name", ""),
            bbox_normalized=tuple(bbox_norm) if isinstance(bbox_norm, list) else bbox_norm,
            bbox_pixels=tuple(bbox_px) if isinstance(bbox_px, list) else bbox_px,
            location_confidence=data.get("location_confidence", "medium"),
            notes=data.get("notes"),
        )


# =============================================================================
# MERGE RESULT (from LLM)
# =============================================================================

@dataclass
class FinalFieldValue:
    """Final merged value for a field."""
    value: str
    source: str  # "original", "refined", "merged"
    confidence: str  # "high", "medium", "low"
    needs_human_review: bool = False
    review_reason: Optional[str] = None
    notes: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinalFieldValue":
        return cls(
            value=data.get("value", ""),
            source=data.get("source", "original"),
            confidence=data.get("confidence", "medium"),
            needs_human_review=data.get("needs_human_review", False),
            review_reason=data.get("review_reason"),
            notes=data.get("notes"),
        )


@dataclass
class MergeSummary:
    """Summary of merge operation."""
    total_fields: int = 0
    from_original: int = 0
    from_refined: int = 0
    merged: int = 0
    flagged_for_review: int = 0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MergeSummary":
        return cls(
            total_fields=data.get("total_fields", 0),
            from_original=data.get("from_original", 0),
            from_refined=data.get("from_refined", 0),
            merged=data.get("merged", 0),
            flagged_for_review=data.get("flagged_for_review", 0),
        )


@dataclass
class MergeQualityCheck:
    """Quality check results from merge operation."""
    duplicate_values_found: bool = False
    year_as_value_found: bool = False
    suspicious_fills: List[str] = field(default_factory=list)
    quality_improved: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MergeQualityCheck":
        return cls(
            duplicate_values_found=data.get("duplicate_values_found", False),
            year_as_value_found=data.get("year_as_value_found", False),
            suspicious_fills=data.get("suspicious_fills", []),
            quality_improved=data.get("quality_improved", True),
        )


@dataclass
class MergeResult:
    """Complete merge result from LLM."""
    final_fields: Dict[str, FinalFieldValue] = field(default_factory=dict)
    summary: MergeSummary = field(default_factory=MergeSummary)
    merge_quality_check: MergeQualityCheck = field(default_factory=MergeQualityCheck)
    iteration_complete: bool = True
    fields_still_uncertain: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MergeResult":
        final_fields = {}
        for name, field_data in data.get("final_fields", {}).items():
            final_fields[name] = FinalFieldValue.from_dict(field_data)
        
        return cls(
            final_fields=final_fields,
            summary=MergeSummary.from_dict(data.get("summary", {})),
            merge_quality_check=MergeQualityCheck.from_dict(
                data.get("merge_quality_check", {})
            ),
            iteration_complete=data.get("iteration_complete", True),
            fields_still_uncertain=data.get("fields_still_uncertain", []),
        )
    
    def has_quality_issues(self) -> bool:
        """Check if merge found quality issues."""
        return (
            self.merge_quality_check.duplicate_values_found or
            self.merge_quality_check.year_as_value_found or
            len(self.merge_quality_check.suspicious_fills) > 0
        )


# =============================================================================
# FINAL AGENTIC OCR RESULT
# =============================================================================

@dataclass
class FieldResult:
    """Final result for a single field."""
    field_name: str
    value: str
    confidence: str
    source: str
    needs_review: bool = False
    review_reason: Optional[str] = None
    is_empty: bool = False
    validation_score: int = 100  # Per-field quality score
    raw_extractions: Dict[str, str] = field(default_factory=dict)  # iteration -> value


@dataclass
class AgenticResult:
    """Complete result from agentic OCR pipeline."""
    # Field-level results
    fields: List[FieldResult] = field(default_factory=list)
    
    # Raw text from initial extraction
    raw_text: str = ""
    
    # Processing metadata
    iterations_used: int = 1
    processing_time_seconds: float = 0.0
    
    # Confidence summary
    confidence_summary: Dict[str, int] = field(default_factory=dict)
    
    # Fields needing human review
    fields_needing_review: List[str] = field(default_factory=list)
    
    # Status
    status: str = "success"
    error: Optional[str] = None
    
    # Quality assessment (NEW)
    quality_score: int = 100  # 0-100, overall extraction quality
    quality_status: str = "passed"  # "passed", "warning", "failed", "rejected"
    quality_report: Optional[QualityReport] = None
    
    # Hallucination detection (NEW)
    hallucination_detected: bool = False
    hallucination_indicators: List[str] = field(default_factory=list)
    
    # Processing trace (for debugging)
    trace: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "fields": [
                {
                    "field_name": f.field_name,
                    "value": f.value,
                    "confidence": f.confidence,
                    "source": f.source,
                    "needs_review": f.needs_review,
                    "review_reason": f.review_reason,
                    "is_empty": f.is_empty,
                    "validation_score": f.validation_score,
                }
                for f in self.fields
            ],
            "raw_text": self.raw_text,
            "iterations_used": self.iterations_used,
            "processing_time_seconds": self.processing_time_seconds,
            "confidence_summary": self.confidence_summary,
            "fields_needing_review": self.fields_needing_review,
            "status": self.status,
            "error": self.error,
            "quality_score": self.quality_score,
            "quality_status": self.quality_status,
            "quality_report": self.quality_report.to_dict() if self.quality_report else None,
            "hallucination_detected": self.hallucination_detected,
            "hallucination_indicators": self.hallucination_indicators,
        }
    
    def get_field_value(self, field_name: str) -> Optional[str]:
        """Get value for a specific field."""
        for f in self.fields:
            if f.field_name == field_name:
                return f.value
        return None
    
    def get_fields_dict(self) -> Dict[str, str]:
        """Get all fields as a simple dict."""
        return {f.field_name: f.value for f in self.fields}
    
    def is_usable(self) -> bool:
        """Check if result is usable (not rejected)."""
        return self.quality_status in ("passed", "warning")
    
    def needs_review(self) -> bool:
        """Check if result needs human review."""
        return (
            self.quality_status == "warning" or
            len(self.fields_needing_review) > 0 or
            self.hallucination_detected
        )
