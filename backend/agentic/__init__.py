"""
Agentic OCR Package - Anti-Hallucination Edition

Multi-pass, self-correcting OCR system using dual-model architecture:
- AIN VLM for vision/OCR tasks
- LLM for reasoning/orchestration (Mistral API or self-hosted)

KEY FEATURES:
- Quality gates at every step to catch hallucinations early
- Field format validation (Saudi IDs, phones, dates)
- Duplicate value detection (same value in multiple fields)
- Automatic rejection of low-quality extractions

Modules:
- controller: Main AgenticOCRController class with quality gates
- clients: VLMClient, LLMClient, and MistralLLMClient for model APIs
- prompts: Anti-hallucination prompt templates
- validators: Field format validators and sanity checks
- quality_gate: Quality scoring and acceptance/rejection logic
- cropper: Region cropping utilities
- models: Pydantic models for data structures
"""

from .controller import AgenticOCRController, SinglePassController
from .clients import VLMClient, LLMClient, MistralLLMClient, create_mistral_client
from .models import (
    AgenticResult,
    FieldResult,
    AnalysisResult,
    RegionEstimate,
    MergeResult,
    QualityReport,
    ValidationIssue,
)
from .validators import (
    validate_extraction,
    validate_raw_extraction,
    parse_extraction_to_fields,
    ValidationResult,
)
from .quality_gate import (
    evaluate_quality,
    evaluate_raw_extraction as evaluate_raw_quality,
    QualityGateResult,
    QualityLevel,
)

__all__ = [
    # Controllers
    "AgenticOCRController",
    "SinglePassController",
    
    # Clients
    "VLMClient",
    "LLMClient",
    "MistralLLMClient",
    "create_mistral_client",
    
    # Result models
    "AgenticResult",
    "FieldResult",
    "AnalysisResult",
    "RegionEstimate",
    "MergeResult",
    "QualityReport",
    "ValidationIssue",
    
    # Validation
    "validate_extraction",
    "validate_raw_extraction",
    "parse_extraction_to_fields",
    "ValidationResult",
    
    # Quality gate
    "evaluate_quality",
    "evaluate_raw_quality",
    "QualityGateResult",
    "QualityLevel",
]

__version__ = "2.0.0"  # Major version bump for anti-hallucination system
