"""
Pydantic Models for Agentic OCR System

Data structures for:
- Analysis results from LLM
- Region estimates for cropping
- Merge results from multiple OCR passes
- Final agentic OCR output
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisStats":
        return cls(
            total_fields=data.get("total_fields", 0),
            high_confidence=data.get("high_confidence", 0),
            medium_confidence=data.get("medium_confidence", 0),
            low_confidence=data.get("low_confidence", 0),
            empty_fields=data.get("empty_fields", 0),
            needs_reexamination=data.get("needs_reexamination", 0),
        )


@dataclass
class AnalysisResult:
    """Complete analysis result from LLM."""
    analysis: AnalysisStats
    fields_to_reexamine: List[FieldToReexamine] = field(default_factory=list)
    confident_fields: List[ConfidentField] = field(default_factory=list)
    empty_fields: List[EmptyField] = field(default_factory=list)
    
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
        )
    
    def has_fields_to_reexamine(self) -> bool:
        """Check if there are fields needing re-examination."""
        return len(self.fields_to_reexamine) > 0


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
class MergeResult:
    """Complete merge result from LLM."""
    final_fields: Dict[str, FinalFieldValue] = field(default_factory=dict)
    summary: MergeSummary = field(default_factory=MergeSummary)
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
            iteration_complete=data.get("iteration_complete", True),
            fields_still_uncertain=data.get("fields_still_uncertain", []),
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
