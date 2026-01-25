"""
Agentic OCR Package - Surgical Precision Edition

Multi-pass, self-correcting OCR system using Azure OpenAI GPT-4o-mini Vision.

KEY FEATURES:
- Surgical section-by-section extraction
- Iterative zoom-in refinement for unclear fields
- Format validation for Saudi documents (IDs, phones, dates)
- Self-critique for hallucination detection
- Continuous learning from user corrections

Architecture:
- controller: AgenticOCRController - surgical OCR pipeline
- azure_client: AzureVisionOCR - GPT-4o-mini vision client
- image_processor: Image preprocessing and section detection
- format_validator: Saudi document format validation
- learning: Correction storage and few-shot learning
- prompts: Section-specific extraction prompts
- models: Pydantic models for data structures
"""

# New surgical OCR imports
from .controller import AgenticOCRController, SinglePassController, create_controller
from .azure_client import AzureVisionOCR, create_azure_client
from .image_processor import ImageProcessor, Section, SectionType
from .format_validator import FormatValidator, validate_extraction as validate_document
from .learning import LearningModule, create_learning_module

# Legacy imports for backward compatibility
from .models import (
    AgenticResult,
    FieldResult,
    AnalysisResult,
    RegionEstimate,
    MergeResult,
    QualityReport,
    ValidationIssue,
)

# Keep old validator imports for compatibility
try:
    from .validators import (
        validate_extraction,
        validate_raw_extraction,
        parse_extraction_to_fields,
        ValidationResult,
    )
except ImportError:
    # New system uses format_validator
    validate_extraction = validate_document
    validate_raw_extraction = None
    parse_extraction_to_fields = None
    ValidationResult = None

try:
    from .quality_gate import (
        evaluate_quality,
        evaluate_raw_extraction as evaluate_raw_quality,
        QualityGateResult,
        QualityLevel,
    )
except ImportError:
    evaluate_quality = None
    evaluate_raw_quality = None
    QualityGateResult = None
    QualityLevel = None

__all__ = [
    # New surgical OCR
    "AgenticOCRController",
    "SinglePassController",
    "create_controller",
    "AzureVisionOCR",
    "create_azure_client",
    "ImageProcessor",
    "Section",
    "SectionType",
    "FormatValidator",
    "validate_document",
    "LearningModule",
    "create_learning_module",
    
    # Result models
    "AgenticResult",
    "FieldResult",
    "AnalysisResult",
    "RegionEstimate",
    "MergeResult",
    "QualityReport",
    "ValidationIssue",
    
    # Legacy (for backward compatibility)
    "validate_extraction",
    "validate_raw_extraction",
    "parse_extraction_to_fields",
    "ValidationResult",
    "evaluate_quality",
    "evaluate_raw_quality",
    "QualityGateResult",
    "QualityLevel",
]

__version__ = "3.0.0"  # Major version: Surgical OCR with Azure OpenAI
