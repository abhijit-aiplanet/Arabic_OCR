"""
Image Processor for Surgical OCR

Handles:
- Image preprocessing (grayscale, contrast enhancement)
- Section detection for Arabic government forms
- Smart cropping with upscaling for unclear fields
- Iterative zoom-in refinement

Inspired by how humans read forms:
1. Get an overview of the document
2. Identify logical sections
3. Process each section with enhanced focus
4. Zoom in on unclear handwriting
"""

from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SectionType(Enum):
    """Types of sections in Saudi government forms."""
    HEADER = "header"
    GENERAL_DATA = "general_data"
    ADDRESS = "address"
    DRIVING_LICENSE = "driving_license"
    VEHICLE = "vehicle"
    FOOTER = "footer"


@dataclass
class Section:
    """Represents a section of the document."""
    section_type: SectionType
    name: str
    box: Tuple[int, int, int, int]  # (left, top, right, bottom) in pixels
    box_normalized: Tuple[float, float, float, float]  # normalized 0-1
    expected_fields: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_type": self.section_type.value,
            "name": self.name,
            "box": self.box,
            "box_normalized": self.box_normalized,
            "expected_fields": self.expected_fields,
        }


@dataclass
class FieldLocation:
    """Estimated location of a field within the document."""
    field_name: str
    box: Tuple[int, int, int, int]  # (left, top, right, bottom) in pixels
    box_normalized: Tuple[float, float, float, float]
    section: SectionType
    confidence: str  # "high", "medium", "low"


# =============================================================================
# SECTION DEFINITIONS FOR ARABIC GOVERNMENT FORMS
# =============================================================================

# Standard section layouts (normalized coordinates 0-1)
# Based on typical Saudi government form layout
SECTION_LAYOUTS = {
    SectionType.HEADER: {
        "name": "Header",
        "box_normalized": (0.0, 0.0, 1.0, 0.18),
        "expected_fields": [
            "نوع النشاط",
            "مدينة مزاولة النشاط",
            "تاريخ بدء النشاط",
            "تاريخ انتهاء الترخيص",
        ]
    },
    SectionType.GENERAL_DATA: {
        "name": "General Data (بيانات عامة)",
        "box_normalized": (0.0, 0.15, 1.0, 0.38),
        "expected_fields": [
            "اسم المالك",
            "رقم الهوية",
            "مصدرها",
            "تاريخها",
            "الحالة الاجتماعية",
            "عدد من يعولهم",
            "المؤهل",
            "تاريخ الميلاد",
        ]
    },
    SectionType.ADDRESS: {
        "name": "Address (العنوان)",
        "box_normalized": (0.0, 0.35, 1.0, 0.55),
        "expected_fields": [
            "المدينة",
            "الحي",
            "الشارع",
            "رقم المبنى",
            "جوال",
            "البريد الإلكتروني",
            "فاكس",
        ]
    },
    SectionType.DRIVING_LICENSE: {
        "name": "Driving License (رخصة القيادة)",
        "box_normalized": (0.0, 0.52, 1.0, 0.65),
        "expected_fields": [
            "رقمها",
            "تاريخ الإصدار",
            "تاريخ الانتهاء",
            "مصدرها",
        ]
    },
    SectionType.VEHICLE: {
        "name": "Vehicle (المركبة)",
        "box_normalized": (0.0, 0.62, 1.0, 0.88),
        "expected_fields": [
            "نوع المركبة",
            "الموديل",
            "اللون",
            "رقم اللوحة",
            "رقم الهيكل",
            "سنة الصنع",
            "عدد الأسطوانات",
        ]
    },
    SectionType.FOOTER: {
        "name": "Footer (التوقيع)",
        "box_normalized": (0.0, 0.85, 1.0, 1.0),
        "expected_fields": [
            "اسم مقدم الطلب",
            "صفته",
            "توقيعه",
            "التاريخ",
            "الختم",
        ]
    },
}

# Common field positions (for zoom-in refinement)
# Normalized coordinates (x1, y1, x2, y2)
FIELD_POSITIONS = {
    # General Data fields
    "اسم المالك": (0.35, 0.10, 0.95, 0.16),
    "الاسم": (0.35, 0.10, 0.95, 0.16),
    "رقم الهوية": (0.60, 0.14, 0.95, 0.20),
    "رقم بطاقة الأحوال": (0.60, 0.14, 0.95, 0.20),
    "مصدرها": (0.35, 0.14, 0.55, 0.20),
    "تاريخها": (0.15, 0.14, 0.35, 0.20),
    "الحالة الاجتماعية": (0.60, 0.18, 0.80, 0.24),
    "عدد من يعولهم": (0.35, 0.18, 0.55, 0.24),
    "المؤهل": (0.15, 0.08, 0.35, 0.14),
    "تاريخ الميلاد": (0.35, 0.08, 0.60, 0.14),
    
    # Address fields
    "المدينة": (0.70, 0.36, 0.95, 0.42),
    "الحي": (0.35, 0.36, 0.65, 0.42),
    "الشارع": (0.15, 0.36, 0.35, 0.42),
    "رقم المبنى": (0.70, 0.40, 0.95, 0.46),
    "جوال": (0.70, 0.44, 0.95, 0.50),
    "رقم الجوال": (0.70, 0.44, 0.95, 0.50),
    "البريد الإلكتروني": (0.35, 0.44, 0.65, 0.50),
    "فاكس": (0.15, 0.44, 0.35, 0.50),
    
    # Driving License fields
    "رقمها": (0.70, 0.54, 0.95, 0.60),
    "رقم رخصة القيادة": (0.70, 0.54, 0.95, 0.60),
    "تاريخ الإصدار": (0.50, 0.54, 0.70, 0.60),
    "تاريخ الانتهاء": (0.30, 0.54, 0.50, 0.60),
    
    # Vehicle fields
    "نوع المركبة": (0.70, 0.64, 0.95, 0.70),
    "الموديل": (0.50, 0.64, 0.70, 0.70),
    "اللون": (0.30, 0.64, 0.50, 0.70),
    "رقم اللوحة": (0.70, 0.70, 0.95, 0.76),
    "رقم الهيكل": (0.35, 0.70, 0.65, 0.76),
    "سنة الصنع": (0.15, 0.70, 0.35, 0.76),
    
    # Footer fields
    "اسم مقدم الطلب": (0.60, 0.88, 0.95, 0.95),
    "صفته": (0.40, 0.88, 0.60, 0.95),
    "التاريخ": (0.20, 0.88, 0.40, 0.95),
}


# =============================================================================
# IMAGE PROCESSOR
# =============================================================================

class ImageProcessor:
    """
    Handles image preprocessing and section detection for surgical OCR.
    
    Key features:
    - Grayscale conversion for noise reduction
    - Contrast enhancement (1.8-2.2x) for handwriting
    - Section detection based on form layout
    - Smart cropping with upscaling
    - Iterative zoom for unclear fields
    """
    
    def __init__(
        self,
        default_contrast: float = 2.0,
        default_upscale: int = 2,
        max_upscale: int = 4,
    ):
        """
        Initialize the image processor.
        
        Args:
            default_contrast: Default contrast enhancement factor
            default_upscale: Default upscale factor for cropped regions
            max_upscale: Maximum upscale factor for zoom refinement
        """
        self.default_contrast = default_contrast
        self.default_upscale = default_upscale
        self.max_upscale = max_upscale
    
    def preprocess(
        self,
        image: Image.Image,
        contrast: Optional[float] = None,
        to_grayscale: bool = True,
    ) -> Image.Image:
        """
        Preprocess image for OCR.
        
        Steps:
        1. Convert to grayscale (reduces noise, improves contrast)
        2. Enhance contrast (makes handwriting more visible)
        
        Args:
            image: PIL Image to preprocess
            contrast: Contrast enhancement factor (default: 2.0)
            to_grayscale: Whether to convert to grayscale
            
        Returns:
            Preprocessed PIL Image
        """
        contrast = contrast or self.default_contrast
        
        # Convert to grayscale
        if to_grayscale:
            processed = image.convert('L')
        else:
            processed = image.copy()
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(processed)
        processed = enhancer.enhance(contrast)
        
        return processed
    
    def detect_sections(self, image: Image.Image) -> List[Section]:
        """
        Detect logical sections in the document.
        
        Uses predefined layouts for Saudi government forms.
        Sections are detected based on typical form structure.
        
        Args:
            image: PIL Image of the document
            
        Returns:
            List of Section objects with pixel coordinates
        """
        width, height = image.size
        sections = []
        
        for section_type, layout in SECTION_LAYOUTS.items():
            # Convert normalized to pixel coordinates
            x1_norm, y1_norm, x2_norm, y2_norm = layout["box_normalized"]
            
            box = (
                int(x1_norm * width),
                int(y1_norm * height),
                int(x2_norm * width),
                int(y2_norm * height),
            )
            
            section = Section(
                section_type=section_type,
                name=layout["name"],
                box=box,
                box_normalized=layout["box_normalized"],
                expected_fields=layout["expected_fields"],
            )
            sections.append(section)
        
        return sections
    
    def crop_section(
        self,
        image: Image.Image,
        section: Section,
        upscale: Optional[int] = None,
        enhance_contrast: bool = True,
    ) -> Image.Image:
        """
        Crop a section from the image with optional upscaling.
        
        Args:
            image: Full document image
            section: Section to crop
            upscale: Upscale factor (default: 2)
            enhance_contrast: Whether to enhance contrast
            
        Returns:
            Cropped and processed image
        """
        upscale = upscale or self.default_upscale
        
        # Crop the section
        cropped = image.crop(section.box)
        
        # Upscale for better OCR accuracy
        if upscale > 1:
            new_size = (cropped.width * upscale, cropped.height * upscale)
            cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)
        
        # Enhance contrast
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(cropped)
            cropped = enhancer.enhance(self.default_contrast)
        
        return cropped
    
    def crop_field(
        self,
        image: Image.Image,
        field_name: str,
        box_normalized: Optional[Tuple[float, float, float, float]] = None,
        upscale: int = 3,
        margin: int = 10,
    ) -> Optional[Image.Image]:
        """
        Crop a specific field area for zoom-in refinement.
        
        Args:
            image: Full document image (preferably preprocessed)
            field_name: Name of the field to crop
            box_normalized: Optional custom bounding box (normalized 0-1)
            upscale: Upscale factor for the crop
            margin: Pixel margin to add around the field
            
        Returns:
            Cropped field image or None if field not found
        """
        # Get field position
        if box_normalized:
            x1_norm, y1_norm, x2_norm, y2_norm = box_normalized
        elif field_name in FIELD_POSITIONS:
            x1_norm, y1_norm, x2_norm, y2_norm = FIELD_POSITIONS[field_name]
        else:
            # Try partial match
            for key, pos in FIELD_POSITIONS.items():
                if key in field_name or field_name in key:
                    x1_norm, y1_norm, x2_norm, y2_norm = pos
                    break
            else:
                return None
        
        width, height = image.size
        
        # Convert to pixels with margin
        box = (
            max(0, int(x1_norm * width) - margin),
            max(0, int(y1_norm * height) - margin),
            min(width, int(x2_norm * width) + margin),
            min(height, int(y2_norm * height) + margin),
        )
        
        # Crop and upscale
        cropped = image.crop(box)
        
        if upscale > 1:
            new_size = (cropped.width * upscale, cropped.height * upscale)
            cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)
        
        return cropped
    
    def iterative_zoom(
        self,
        image: Image.Image,
        field_name: str,
        box_normalized: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[Tuple[Image.Image, int]]:
        """
        Generate multiple zoom levels for a field.
        
        Used when initial extraction is unclear - provides
        progressively zoomed images for refinement.
        
        Args:
            image: Full document image
            field_name: Name of the field
            box_normalized: Optional custom bounding box
            
        Returns:
            List of (image, scale) tuples at different zoom levels
        """
        levels = []
        
        for scale in [2, 3, 4]:
            cropped = self.crop_field(
                image, 
                field_name, 
                box_normalized,
                upscale=scale,
            )
            
            if cropped:
                # Apply extra contrast at higher zoom levels
                if scale >= 3:
                    enhancer = ImageEnhance.Contrast(cropped)
                    cropped = enhancer.enhance(2.2)
                
                levels.append((cropped, scale))
        
        return levels
    
    def estimate_field_location(
        self,
        field_name: str,
        image_width: int,
        image_height: int,
    ) -> Optional[FieldLocation]:
        """
        Estimate the location of a field based on predefined positions.
        
        Args:
            field_name: Name of the field
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            FieldLocation or None if not found
        """
        # Direct match
        if field_name in FIELD_POSITIONS:
            box_norm = FIELD_POSITIONS[field_name]
        else:
            # Try partial match
            box_norm = None
            for key, pos in FIELD_POSITIONS.items():
                if key in field_name or field_name in key:
                    box_norm = pos
                    break
        
        if not box_norm:
            return None
        
        x1_norm, y1_norm, x2_norm, y2_norm = box_norm
        box = (
            int(x1_norm * image_width),
            int(y1_norm * image_height),
            int(x2_norm * image_width),
            int(y2_norm * image_height),
        )
        
        # Determine section
        section = self._get_section_for_field(field_name)
        
        return FieldLocation(
            field_name=field_name,
            box=box,
            box_normalized=box_norm,
            section=section,
            confidence="medium",
        )
    
    def _get_section_for_field(self, field_name: str) -> SectionType:
        """Determine which section a field belongs to."""
        for section_type, layout in SECTION_LAYOUTS.items():
            for expected_field in layout["expected_fields"]:
                if expected_field in field_name or field_name in expected_field:
                    return section_type
        return SectionType.GENERAL_DATA  # Default
    
    def is_blank_region(
        self,
        image: Image.Image,
        threshold: float = 0.95,
    ) -> bool:
        """
        Check if a cropped region is mostly blank (empty field).
        
        Args:
            image: Cropped PIL Image
            threshold: Ratio of white pixels to consider blank
            
        Returns:
            True if region appears to be blank/empty
        """
        try:
            import numpy as np
            
            # Convert to grayscale
            gray = image.convert('L')
            pixels = np.array(gray)
            
            # Count "white" pixels (value > 240)
            white_ratio = np.sum(pixels > 240) / pixels.size
            
            return white_ratio > threshold
        except ImportError:
            # Fallback without numpy
            gray = image.convert('L')
            pixels = list(gray.getdata())
            white_count = sum(1 for p in pixels if p > 240)
            return (white_count / len(pixels)) > threshold
    
    def image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def base64_to_image(self, base64_string: str) -> Image.Image:
        """Convert base64 string to PIL Image."""
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_section_for_fields(fields: List[str]) -> Dict[str, SectionType]:
    """Map fields to their sections."""
    processor = ImageProcessor()
    return {
        field: processor._get_section_for_field(field)
        for field in fields
    }


def get_all_expected_fields() -> List[str]:
    """Get all expected fields across all sections."""
    fields = []
    for layout in SECTION_LAYOUTS.values():
        fields.extend(layout["expected_fields"])
    return fields
