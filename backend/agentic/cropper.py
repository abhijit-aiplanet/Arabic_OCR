"""
Region Cropper for Agentic OCR System

Provides utilities for:
- Cropping image regions based on bounding boxes
- Smart padding to include context
- Grid-based fallback cropping
- Image preprocessing for cropped regions
"""

from typing import Tuple, List, Optional
from PIL import Image
import io


# =============================================================================
# REGION CROPPER
# =============================================================================

class RegionCropper:
    """
    Crop image regions for focused OCR re-extraction.
    
    Features:
    - Smart padding to include surrounding context
    - Minimum size enforcement for OCR quality
    - Aspect ratio preservation options
    - Grid-based fallback for when region estimation fails
    """
    
    def __init__(
        self,
        default_padding: float = 0.15,  # 15% padding
        min_size: int = 150,  # Minimum dimension in pixels
        max_aspect_ratio: float = 5.0  # Max width/height ratio
    ):
        self.default_padding = default_padding
        self.min_size = min_size
        self.max_aspect_ratio = max_aspect_ratio
    
    def crop(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
        padding: Optional[float] = None,
        normalize: bool = True
    ) -> Image.Image:
        """
        Crop a region from the image with smart padding.
        
        Args:
            image: Source PIL Image
            bbox: Bounding box as (x1, y1, x2, y2)
                  If normalize=True: values are 0-1 normalized
                  If normalize=False: values are pixel coordinates
            padding: Padding factor (0.15 = 15% padding on each side)
            normalize: Whether bbox values are normalized (0-1) or pixels
            
        Returns:
            Cropped PIL Image
        """
        width, height = image.size
        padding = padding if padding is not None else self.default_padding
        
        x1, y1, x2, y2 = bbox
        
        # Convert normalized to pixels if needed
        if normalize:
            x1 = int(x1 * width)
            y1 = int(y1 * height)
            x2 = int(x2 * width)
            y2 = int(y2 * height)
        
        # Calculate padding in pixels
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = int(box_width * padding)
        pad_y = int(box_height * padding)
        
        # Apply padding with boundary checks
        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(width, x2 + pad_x)
        y2_padded = min(height, y2 + pad_y)
        
        # Ensure minimum size
        crop_width = x2_padded - x1_padded
        crop_height = y2_padded - y1_padded
        
        if crop_width < self.min_size:
            expand = (self.min_size - crop_width) // 2
            x1_padded = max(0, x1_padded - expand)
            x2_padded = min(width, x2_padded + expand)
        
        if crop_height < self.min_size:
            expand = (self.min_size - crop_height) // 2
            y1_padded = max(0, y1_padded - expand)
            y2_padded = min(height, y2_padded + expand)
        
        # Ensure reasonable aspect ratio
        final_width = x2_padded - x1_padded
        final_height = y2_padded - y1_padded
        
        if final_width > 0 and final_height > 0:
            aspect = final_width / final_height
            if aspect > self.max_aspect_ratio:
                # Too wide, expand height
                target_height = int(final_width / self.max_aspect_ratio)
                expand = (target_height - final_height) // 2
                y1_padded = max(0, y1_padded - expand)
                y2_padded = min(height, y2_padded + expand)
            elif aspect < 1 / self.max_aspect_ratio:
                # Too tall, expand width
                target_width = int(final_height / self.max_aspect_ratio)
                expand = (target_width - final_width) // 2
                x1_padded = max(0, x1_padded - expand)
                x2_padded = min(width, x2_padded + expand)
        
        # Perform crop
        cropped = image.crop((x1_padded, y1_padded, x2_padded, y2_padded))
        
        return cropped
    
    def crop_from_estimate(
        self,
        image: Image.Image,
        region_estimate: "RegionEstimate"
    ) -> Image.Image:
        """
        Crop using a RegionEstimate object.
        
        Args:
            image: Source PIL Image
            region_estimate: RegionEstimate object with bbox_normalized
            
        Returns:
            Cropped PIL Image
        """
        bbox = region_estimate.bbox_normalized
        return self.crop(image, bbox, normalize=True)
    
    def create_grid_regions(
        self,
        image: Image.Image,
        rows: int = 3,
        cols: int = 2,
        overlap: float = 0.1
    ) -> List[Tuple[str, Image.Image, Tuple[float, float, float, float]]]:
        """
        Create grid-based crop regions as fallback.
        
        Useful when LLM region estimation fails.
        
        Args:
            image: Source PIL Image
            rows: Number of rows in grid
            cols: Number of columns in grid
            overlap: Overlap between adjacent cells (0.1 = 10%)
            
        Returns:
            List of (region_name, cropped_image, bbox_normalized)
        """
        width, height = image.size
        
        cell_width = 1.0 / cols
        cell_height = 1.0 / rows
        
        overlap_x = cell_width * overlap
        overlap_y = cell_height * overlap
        
        regions = []
        
        for row in range(rows):
            for col in range(cols):
                # Calculate normalized bbox with overlap
                x1 = max(0, col * cell_width - overlap_x)
                y1 = max(0, row * cell_height - overlap_y)
                x2 = min(1.0, (col + 1) * cell_width + overlap_x)
                y2 = min(1.0, (row + 1) * cell_height + overlap_y)
                
                bbox = (x1, y1, x2, y2)
                
                # Crop the region
                cropped = self.crop(image, bbox, padding=0, normalize=True)
                
                # Create region name
                region_name = f"grid_{row}_{col}"
                
                regions.append((region_name, cropped, bbox))
        
        return regions
    
    def create_row_regions(
        self,
        image: Image.Image,
        num_rows: int = 6,
        overlap: float = 0.15
    ) -> List[Tuple[str, Image.Image, Tuple[float, float, float, float]]]:
        """
        Create horizontal row-based crop regions.
        
        Good for forms with horizontal field layout.
        
        Args:
            image: Source PIL Image
            num_rows: Number of horizontal rows
            overlap: Overlap between adjacent rows
            
        Returns:
            List of (region_name, cropped_image, bbox_normalized)
        """
        row_height = 1.0 / num_rows
        overlap_y = row_height * overlap
        
        regions = []
        
        for row in range(num_rows):
            y1 = max(0, row * row_height - overlap_y)
            y2 = min(1.0, (row + 1) * row_height + overlap_y)
            
            bbox = (0.0, y1, 1.0, y2)
            cropped = self.crop(image, bbox, padding=0, normalize=True)
            
            region_name = f"row_{row}"
            regions.append((region_name, cropped, bbox))
        
        return regions


# =============================================================================
# PREPROCESSING FOR CROPPED REGIONS
# =============================================================================

def enhance_cropped_region(image: Image.Image) -> Image.Image:
    """
    Apply light preprocessing to cropped region for better OCR.
    
    This is gentler than full-page preprocessing since crops are already focused.
    
    Args:
        image: Cropped PIL Image
        
    Returns:
        Enhanced PIL Image
    """
    # Ensure RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Light sharpening for text clarity
    from PIL import ImageFilter, ImageEnhance
    
    # Slight contrast boost
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)  # 10% contrast increase
    
    # Very light sharpening
    image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=3))
    
    return image


def resize_for_ocr(
    image: Image.Image,
    min_dimension: int = 300,
    max_dimension: int = 1500
) -> Image.Image:
    """
    Resize cropped region to optimal size for OCR.
    
    Args:
        image: Cropped PIL Image
        min_dimension: Minimum dimension target
        max_dimension: Maximum dimension target
        
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    
    # Check if resize needed
    min_current = min(width, height)
    max_current = max(width, height)
    
    if min_current < min_dimension:
        # Upscale
        scale = min_dimension / min_current
    elif max_current > max_dimension:
        # Downscale
        scale = max_dimension / max_current
    else:
        return image  # No resize needed
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Use LANCZOS for quality
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
