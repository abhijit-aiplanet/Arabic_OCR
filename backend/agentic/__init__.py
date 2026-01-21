"""
Agentic OCR Package

Multi-pass, self-correcting OCR system using dual-model architecture:
- AIN VLM for vision/OCR tasks
- Qwen 2.5 LLM for reasoning/orchestration

Modules:
- controller: Main AgenticOCRController class
- clients: VLMClient and LLMClient for model APIs
- prompts: All prompt templates for the agentic flow
- cropper: Region cropping utilities
- models: Pydantic models for data structures
"""

from .controller import AgenticOCRController
from .clients import VLMClient, LLMClient
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
    "AgenticResult",
    "FieldResult",
    "AnalysisResult",
    "RegionEstimate",
    "MergeResult",
]

__version__ = "1.0.0"
