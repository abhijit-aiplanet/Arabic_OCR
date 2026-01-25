"""
Learning Module for Continuous OCR Improvement

Stores user corrections and uses them as few-shot examples
to improve future extractions.

Features:
- Store corrections in Supabase
- Generate few-shot context from past corrections
- Build field-specific knowledge
- Track correction patterns
"""

import os
from typing import Optional, Dict, List, Any
from datetime import datetime
from dataclasses import dataclass, field
import json


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Correction:
    """A single user correction."""
    id: Optional[str] = None
    user_id: str = ""
    ocr_history_id: Optional[str] = None
    field_name: str = ""
    field_type: Optional[str] = None
    original_value: str = ""
    corrected_value: str = ""
    image_context: Optional[str] = None  # Description of what the field looked like
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "ocr_history_id": self.ocr_history_id,
            "field_name": self.field_name,
            "field_type": self.field_type,
            "original_value": self.original_value,
            "corrected_value": self.corrected_value,
            "image_context": self.image_context,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Correction":
        return cls(
            id=data.get("id"),
            user_id=data.get("user_id", ""),
            ocr_history_id=data.get("ocr_history_id"),
            field_name=data.get("field_name", ""),
            field_type=data.get("field_type"),
            original_value=data.get("original_value", ""),
            corrected_value=data.get("corrected_value", ""),
            image_context=data.get("image_context"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
        )


@dataclass
class LearningContext:
    """Context generated from past corrections."""
    few_shot_examples: str
    field_hints: Dict[str, str]
    common_mistakes: List[str]
    
    def to_prompt(self) -> str:
        """Convert to prompt format."""
        parts = []
        
        if self.few_shot_examples:
            parts.append("## From past corrections:\n" + self.few_shot_examples)
        
        if self.common_mistakes:
            parts.append("## Common mistakes to avoid:\n" + "\n".join(f"- {m}" for m in self.common_mistakes))
        
        return "\n\n".join(parts)


# =============================================================================
# LEARNING MODULE
# =============================================================================

class LearningModule:
    """
    Manages learning from user corrections.
    
    Features:
    - Store corrections in Supabase
    - Generate few-shot examples for prompts
    - Track common mistakes per field type
    - Build knowledge base over time
    """
    
    def __init__(self, supabase_client=None):
        """
        Initialize the learning module.
        
        Args:
            supabase_client: Optional Supabase client. If None, will use in-memory storage.
        """
        self.db = supabase_client
        self._memory_corrections: List[Correction] = []  # Fallback storage
        self._table_name = "ocr_corrections"
    
    async def record_correction(
        self,
        user_id: str,
        field_name: str,
        original_value: str,
        corrected_value: str,
        field_type: Optional[str] = None,
        ocr_history_id: Optional[str] = None,
        image_context: Optional[str] = None,
    ) -> bool:
        """
        Store a user correction for future learning.
        
        Args:
            user_id: ID of the user making the correction
            field_name: Name of the corrected field
            original_value: Original OCR value
            corrected_value: User's corrected value
            field_type: Type of field (e.g., "national_id", "phone")
            ocr_history_id: Optional reference to OCR history entry
            image_context: Optional description of field appearance
            
        Returns:
            True if correction was stored successfully
        """
        correction = Correction(
            user_id=user_id,
            field_name=field_name,
            field_type=field_type or self._infer_field_type(field_name),
            original_value=original_value,
            corrected_value=corrected_value,
            ocr_history_id=ocr_history_id,
            image_context=image_context,
            created_at=datetime.utcnow(),
        )
        
        if self.db:
            try:
                result = self.db.table(self._table_name).insert(
                    correction.to_dict()
                ).execute()
                return bool(result.data)
            except Exception as e:
                print(f"[Learning] Failed to store correction: {e}")
                # Fall back to memory
                self._memory_corrections.append(correction)
                return True
        else:
            # Use in-memory storage
            self._memory_corrections.append(correction)
            return True
    
    async def get_few_shot_examples(
        self,
        field_type: Optional[str] = None,
        field_name: Optional[str] = None,
        limit: int = 5,
    ) -> str:
        """
        Get few-shot examples from past corrections.
        
        Args:
            field_type: Optional filter by field type
            field_name: Optional filter by field name
            limit: Maximum number of examples to return
            
        Returns:
            Formatted string of few-shot examples
        """
        corrections = await self._get_corrections(
            field_type=field_type,
            field_name=field_name,
            limit=limit,
        )
        
        if not corrections:
            return ""
        
        examples = []
        for c in corrections:
            if c.original_value != c.corrected_value:
                examples.append(
                    f"- الخطأ: '{c.original_value}' → الصحيح: '{c.corrected_value}'"
                )
        
        return "\n".join(examples)
    
    async def get_context(
        self,
        field_type: Optional[str] = None,
        field_names: Optional[List[str]] = None,
    ) -> LearningContext:
        """
        Get full learning context for extraction.
        
        Args:
            field_type: Optional filter by field type
            field_names: Optional list of field names to get context for
            
        Returns:
            LearningContext with examples and hints
        """
        # Get recent corrections
        corrections = await self._get_corrections(limit=20)
        
        # Generate few-shot examples
        few_shot = await self.get_few_shot_examples(field_type=field_type, limit=5)
        
        # Build field-specific hints
        field_hints = {}
        if field_names:
            for name in field_names:
                hint = await self._get_field_hint(name)
                if hint:
                    field_hints[name] = hint
        
        # Identify common mistakes
        common_mistakes = self._analyze_common_mistakes(corrections)
        
        return LearningContext(
            few_shot_examples=few_shot,
            field_hints=field_hints,
            common_mistakes=common_mistakes,
        )
    
    async def get_correction_stats(self) -> Dict[str, Any]:
        """Get statistics about corrections."""
        corrections = await self._get_corrections(limit=1000)
        
        if not corrections:
            return {"total": 0}
        
        # Count by field type
        by_type: Dict[str, int] = {}
        for c in corrections:
            ft = c.field_type or "unknown"
            by_type[ft] = by_type.get(ft, 0) + 1
        
        # Most corrected fields
        by_field: Dict[str, int] = {}
        for c in corrections:
            by_field[c.field_name] = by_field.get(c.field_name, 0) + 1
        
        top_fields = sorted(by_field.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "total": len(corrections),
            "by_type": by_type,
            "top_corrected_fields": top_fields,
        }
    
    async def _get_corrections(
        self,
        field_type: Optional[str] = None,
        field_name: Optional[str] = None,
        limit: int = 20,
    ) -> List[Correction]:
        """Get corrections from storage."""
        if self.db:
            try:
                query = self.db.table(self._table_name).select("*")
                
                if field_type:
                    query = query.eq("field_type", field_type)
                
                if field_name:
                    query = query.eq("field_name", field_name)
                
                result = query.order("created_at", desc=True).limit(limit).execute()
                
                return [Correction.from_dict(d) for d in result.data]
                
            except Exception as e:
                print(f"[Learning] Failed to get corrections: {e}")
                # Fall back to memory
                return self._filter_memory_corrections(field_type, field_name, limit)
        else:
            return self._filter_memory_corrections(field_type, field_name, limit)
    
    def _filter_memory_corrections(
        self,
        field_type: Optional[str],
        field_name: Optional[str],
        limit: int,
    ) -> List[Correction]:
        """Filter in-memory corrections."""
        corrections = self._memory_corrections
        
        if field_type:
            corrections = [c for c in corrections if c.field_type == field_type]
        
        if field_name:
            corrections = [c for c in corrections if c.field_name == field_name]
        
        # Sort by created_at descending
        corrections = sorted(
            corrections,
            key=lambda c: c.created_at or datetime.min,
            reverse=True,
        )
        
        return corrections[:limit]
    
    async def _get_field_hint(self, field_name: str) -> Optional[str]:
        """Get accumulated hints for a specific field."""
        corrections = await self._get_corrections(field_name=field_name, limit=10)
        
        if not corrections:
            return None
        
        # Analyze correction patterns
        patterns = []
        
        for c in corrections:
            if c.original_value and c.corrected_value:
                # Look for common patterns
                if len(c.original_value) != len(c.corrected_value):
                    patterns.append("Check for missing/extra characters")
                
                # Check for digit/letter confusion
                orig_digits = sum(1 for ch in c.original_value if ch.isdigit())
                corr_digits = sum(1 for ch in c.corrected_value if ch.isdigit())
                if orig_digits != corr_digits:
                    patterns.append("Verify number of digits")
        
        if patterns:
            return ", ".join(set(patterns))
        
        return None
    
    def _analyze_common_mistakes(self, corrections: List[Correction]) -> List[str]:
        """Analyze corrections to find common mistake patterns."""
        mistakes = []
        
        if not corrections:
            return mistakes
        
        # Count mistake types
        digit_errors = 0
        length_errors = 0
        format_errors = 0
        
        for c in corrections:
            if not c.original_value or not c.corrected_value:
                continue
            
            # Check for digit count difference
            orig_digits = sum(1 for ch in c.original_value if ch.isdigit())
            corr_digits = sum(1 for ch in c.corrected_value if ch.isdigit())
            if orig_digits != corr_digits:
                digit_errors += 1
            
            # Check for length difference
            if len(c.original_value) != len(c.corrected_value):
                length_errors += 1
            
            # Check for format markers
            if "[" in c.original_value or "]" in c.original_value:
                format_errors += 1
        
        # Generate hints based on frequency
        total = len(corrections)
        
        if digit_errors > total * 0.3:
            mistakes.append("Pay attention to digit count")
        
        if length_errors > total * 0.3:
            mistakes.append("Check value length carefully")
        
        if format_errors > total * 0.2:
            mistakes.append("Remove format markers from values")
        
        return mistakes
    
    def _infer_field_type(self, field_name: str) -> Optional[str]:
        """Infer field type from field name."""
        type_mapping = {
            "الهوية": "national_id",
            "بطاقة الأحوال": "national_id",
            "جوال": "phone",
            "تاريخ": "date",
            "الميلاد": "date",
            "الإصدار": "date",
            "الانتهاء": "date",
            "اسم": "arabic_name",
            "المدينة": "city",
            "الحي": "city",
            "مصدر": "city",
            "اللوحة": "plate",
        }
        
        for key, field_type in type_mapping.items():
            if key in field_name:
                return field_type
        
        return None


# =============================================================================
# DATABASE SCHEMA
# =============================================================================

# SQL to create the ocr_corrections table in Supabase:
OCR_CORRECTIONS_SCHEMA = """
-- Create ocr_corrections table for learning from user feedback
CREATE TABLE IF NOT EXISTS ocr_corrections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    ocr_history_id UUID REFERENCES ocr_history(id) ON DELETE SET NULL,
    field_name TEXT NOT NULL,
    field_type TEXT,
    original_value TEXT,
    corrected_value TEXT NOT NULL,
    image_context TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_corrections_user ON ocr_corrections(user_id);
CREATE INDEX IF NOT EXISTS idx_corrections_field ON ocr_corrections(field_name);
CREATE INDEX IF NOT EXISTS idx_corrections_type ON ocr_corrections(field_type);
CREATE INDEX IF NOT EXISTS idx_corrections_created ON ocr_corrections(created_at DESC);

-- Enable Row Level Security (optional)
ALTER TABLE ocr_corrections ENABLE ROW LEVEL SECURITY;

-- Policy to allow users to see their own corrections
CREATE POLICY "Users can view own corrections" ON ocr_corrections
    FOR SELECT USING (auth.uid()::text = user_id);

-- Policy to allow users to insert corrections
CREATE POLICY "Users can insert corrections" ON ocr_corrections
    FOR INSERT WITH CHECK (auth.uid()::text = user_id);
"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_learning_module(supabase_client=None) -> LearningModule:
    """
    Create a learning module instance.
    
    Args:
        supabase_client: Optional Supabase client
        
    Returns:
        LearningModule instance
    """
    return LearningModule(supabase_client)


async def record_correction(
    learning_module: LearningModule,
    user_id: str,
    field_name: str,
    original_value: str,
    corrected_value: str,
    **kwargs,
) -> bool:
    """
    Convenience function to record a correction.
    
    Args:
        learning_module: LearningModule instance
        user_id: User ID
        field_name: Field name
        original_value: Original value
        corrected_value: Corrected value
        **kwargs: Additional arguments passed to record_correction
        
    Returns:
        True if successful
    """
    return await learning_module.record_correction(
        user_id=user_id,
        field_name=field_name,
        original_value=original_value,
        corrected_value=corrected_value,
        **kwargs,
    )
