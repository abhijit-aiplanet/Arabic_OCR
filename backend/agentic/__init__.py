"""
Agentic OCR Package

Multi-pass, self-correcting OCR system using dual-model architecture:
- AIN VLM for vision/OCR tasks
- LLM for reasoning/orchestration (Mistral API or self-hosted)

Modules:
- controller: Main AgenticOCRController class
- clients: VLMClient, LLMClient, and MistralLLMClient for model APIs
- prompts: All prompt templates for the agentic flow
- cropper: Region cropping utilities
- models: Pydantic models for data structures
"""

from .controller import AgenticOCRController
from .clients import VLMClient, LLMClient, MistralLLMClient, create_mistral_client
from .models import (
    AgenticResult,
    FieldResult,
    AnalysisResult,
    RegionEstimate,
    MergeResult,
)

__all__ = [
    "AgenticOCRController",
    "VLMClient",
    "LLMClient",
    "MistralLLMClient",
    "create_mistral_client",
    "AgenticResult",
    "FieldResult",
    "AnalysisResult",
    "RegionEstimate",
    "MergeResult",
]

__version__ = "1.1.0"
