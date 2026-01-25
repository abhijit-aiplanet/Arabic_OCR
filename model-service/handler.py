"""
RunPod Handler for AIN Vision Language Model OCR
This service runs on RunPod GPU instances and processes OCR requests
Build trigger: 2026-01-22-v2 (Fix hallucination: reduce max_tokens from 8192 to 2048)
"""

import runpod
import torch
import os
from PIL import Image, ImageEnhance, ImageFilter
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from transformers import Qwen2VLProcessor, Qwen2VLImageProcessor
import traceback
import io
import base64
import numpy as np
import cv2
from scipy import ndimage

# =============================================================================
# MODEL CONFIGURATION - Uses Network Volume for model caching
# =============================================================================
MODEL_ID = "MBZUAI/AIN"

# Network Volume path (standard RunPod mount point)
NETWORK_VOLUME_PATH = "/runpod-volume"
NETWORK_VOLUME_CACHE = os.path.join(NETWORK_VOLUME_PATH, "huggingface")

# Fallback to container cache if network volume not available
CONTAINER_CACHE = "/app/hf_cache"


def get_cache_directory():
    """
    Determine the best cache directory to use.
    
    Priority:
    1. Network Volume (/runpod-volume/huggingface) - persistent across builds
    2. Container cache (/app/hf_cache) - fallback
    """
    if os.path.exists(NETWORK_VOLUME_PATH):
        # Network volume is mounted
        os.makedirs(NETWORK_VOLUME_CACHE, exist_ok=True)
        print(f"✅ Network Volume detected at {NETWORK_VOLUME_PATH}")
        print(f"   Using cache directory: {NETWORK_VOLUME_CACHE}")
        return NETWORK_VOLUME_CACHE
    else:
        # Fallback to container cache
        os.makedirs(CONTAINER_CACHE, exist_ok=True)
        print(f"⚠️ Network Volume not mounted at {NETWORK_VOLUME_PATH}")
        print(f"   Using container cache: {CONTAINER_CACHE}")
        return CONTAINER_CACHE


# Initialize cache directory
HF_CACHE_DIR = get_cache_directory()

# Set environment variables for HuggingFace
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = HF_CACHE_DIR

# =============================================================================
# RESOLUTION SETTINGS - OPTIMIZED FOR HANDWRITING RECOGNITION
# =============================================================================
# Higher resolution is critical for handwriting - thin strokes get lost at low res
# AIN model uses visual tokens, more pixels = more visual information

# Base resolution settings
MIN_PIXELS = 256 * 28 * 28  # 200,704 - Minimum for any image

# Resolution presets for different document types
RESOLUTION_PRESETS = {
    # Standard printed text - can use lower resolution
    "printed": 1280 * 28 * 28,     # 1,003,520 pixels
    
    # Handwriting-heavy documents - NEED higher resolution
    "handwriting": 1680 * 28 * 28,  # 1,317,120 pixels (~30% more)
    
    # Complex forms with small fields - highest resolution
    "complex_form": 2016 * 28 * 28, # 1,580,544 pixels (~57% more)
    
    # Default - use handwriting preset for Arabic forms
    "default": 1680 * 28 * 28,      # 1,317,120 pixels
}

# Default MAX_PIXELS - increased for better handwriting recognition
MAX_PIXELS = RESOLUTION_PRESETS["default"]  # 1,317,120 pixels

# Maximum tokens for generation - CONSERVATIVE to prevent hallucination
# Note: 8192 caused EXTREME repetition (171+ repeats) and 5+ min generation time
# 1024 is sufficient for transcription and prevents most hallucinations
DEFAULT_MAX_TOKENS = 1024  # Further reduced for anti-hallucination

# Transcription mode uses even fewer tokens (just raw values)
TRANSCRIPTION_MAX_TOKENS = 512

def get_resolution_for_document_type(doc_type: str = "default") -> int:
    """Get appropriate MAX_PIXELS for document type."""
    return RESOLUTION_PRESETS.get(doc_type, RESOLUTION_PRESETS["default"])

# Global model and processor
model = None
processor = None


def check_flash_attn_available():
    """Check if flash_attn package is installed and usable."""
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def load_model():
    """Load the Arabic VLM model and processor with optimized attention for GPU.
    
    Model is cached on Network Volume for fast cold starts.
    First run: Downloads model (~30GB) to Network Volume - takes 5-10 minutes
    Subsequent runs: Loads from cache - takes ~30-60 seconds
    """
    global model, processor
    
    if model is not None and processor is not None:
        return
    
    print("=" * 60)
    print("🔄 Loading Arabic VLM model (MBZUAI/AIN)")
    print("=" * 60)
    print(f"   Cache directory: {HF_CACHE_DIR}")
    
    # Check if using Network Volume
    using_network_volume = HF_CACHE_DIR.startswith(NETWORK_VOLUME_PATH)
    if using_network_volume:
        print(f"   ✅ Using Network Volume for persistent caching")
    else:
        print(f"   ⚠️ Using container cache (not persistent)")
    
    # Check cache status
    if os.path.exists(HF_CACHE_DIR):
        try:
            cache_size = sum(os.path.getsize(os.path.join(dirpath, filename)) 
                            for dirpath, dirnames, filenames in os.walk(HF_CACHE_DIR) 
                            for filename in filenames) / (1024**3)
            print(f"   Cache size: {cache_size:.2f} GB")
            
            if cache_size > 20:
                print(f"   ✅ Model appears to be cached (fast load expected)")
            else:
                print(f"   ⚠️ Cache is small - model may need to download")
        except Exception as e:
            print(f"   Could not calculate cache size: {e}")
    else:
        print("   📥 Cache directory empty - model will be downloaded")
        print("   ⏳ First run will take 5-10 minutes to download ~30GB")
        os.makedirs(HF_CACHE_DIR, exist_ok=True)
    
    try:
        # Use GPU if available
        if torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = torch.bfloat16  # Use bfloat16 for A100 optimization
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
            # Check for Flash Attention 2 support
            gpu_name = torch.cuda.get_device_name(0).lower()
            supports_flash_attn_hw = any(x in gpu_name for x in ['a100', 'a10', 'h100', 'l40', 'rtx 40', 'rtx 30'])
            flash_attn_installed = check_flash_attn_available()
            
            if supports_flash_attn_hw and flash_attn_installed:
                print("🚀 Flash Attention 2 available! Enabling for maximum performance...")
                attn_implementation = "flash_attention_2"
            elif supports_flash_attn_hw:
                print("⚠️ GPU supports Flash Attention 2, but flash_attn package not installed")
                print("   Using SDPA (Scaled Dot Product Attention) as fallback - still fast!")
                attn_implementation = "sdpa"
            else:
                print("ℹ️ Using SDPA (Scaled Dot Product Attention)")
                attn_implementation = "sdpa"
        else:
            device_map = "cpu"
            torch_dtype = torch.float32
            attn_implementation = "eager"
            print("⚠️ Using CPU (not recommended)")
        
        # Load model with Flash Attention 2 for A100
        # Uses pre-downloaded model from HF_CACHE_DIR (set during Docker build)
        print(f"🔧 Loading model with attention implementation: {attn_implementation}")
        loaded_model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            cache_dir=HF_CACHE_DIR,  # Use pre-downloaded model
            local_files_only=False,   # Allow download as fallback if cache missing
        )
        
        # Load processor
        try:
            loaded_processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
                cache_dir=HF_CACHE_DIR,  # Use pre-downloaded processor
            )
            print("✅ Processor loaded successfully (standard method)")
        except ValueError as e:
            if "size must contain 'shortest_edge' and 'longest_edge' keys" in str(e):
                print("⚠️ Standard processor loading failed, trying manual construction...")
                # Manually construct processor
                tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_ID, 
                    trust_remote_code=True,
                    cache_dir=HF_CACHE_DIR,
                )
                
                image_processor = Qwen2VLImageProcessor(
                    size={"shortest_edge": 224, "longest_edge": 1120},
                    do_resize=True,
                    do_rescale=True,
                    do_normalize=True,
                )
                
                loaded_processor = Qwen2VLProcessor(
                    image_processor=image_processor,
                    tokenizer=tokenizer,
                )
                print("✅ Processor loaded successfully (manual construction)")
            else:
                raise
        
        model = loaded_model
        processor = loaded_processor
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model loaded successfully!")
        print(f"   Total parameters: {total_params / 1e9:.2f}B")
        print(f"   Attention: {attn_implementation}")
        print(f"   Dtype: {torch_dtype}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        raise


def _clean_repetition_loops(text: str, max_repeats: int = 3) -> str:
    """
    AGGRESSIVE cleanup for repetition loops.
    Triggers on 3+ repeats (reduced from 5) for better hallucination prevention.
    """
    if not text or len(text) < 100:
        return text
    
    original_len = len(text)
    lines = text.split('\n')
    
    # Check for line-level repetition
    for pattern_length in range(1, 5):  # Check 1-4 line patterns
        if len(lines) < pattern_length * (max_repeats + 1):
            continue
        
        for start_idx in range(len(lines) - pattern_length * max_repeats):
            pattern = lines[start_idx:start_idx + pattern_length]
            pattern_str = '\n'.join(pattern)
            
            # Skip very long patterns
            if len(pattern_str) > 150:
                continue
            
            # Skip empty patterns
            if not pattern_str.strip():
                continue
            
            repeat_count = 1
            check_idx = start_idx + pattern_length
            
            while check_idx + pattern_length <= len(lines):
                check_pattern = lines[check_idx:check_idx + pattern_length]
                check_str = '\n'.join(check_pattern)
                
                if check_str == pattern_str:
                    repeat_count += 1
                    check_idx += pattern_length
                else:
                    break
            
            if repeat_count >= max_repeats:
                print(f"⚠️ Detected repetition: {repeat_count} repeats of pattern")
                # Keep only first occurrence
                truncated_lines = lines[:start_idx + pattern_length]
                result = '\n'.join(truncated_lines).strip()
                print(f"📉 Reduced from {original_len} to {len(result)} characters")
                return result
    
    # Check for word-level repetition (same word 10+ times)
    words = text.split()
    if words:
        from collections import Counter
        word_counts = Counter(words)
        for word, count in word_counts.most_common(3):
            if count > 10 and len(word) > 2:
                print(f"⚠️ Word '{word}' repeated {count} times - possible hallucination")
    
    return text


def _detect_value_propagation(text: str) -> dict:
    """
    Detect if the same value appears multiple times (hallucination indicator).
    Returns info about detected issues.
    """
    issues = {
        "has_propagation": False,
        "propagated_values": [],
        "year_as_value": False,
        "year_fields": []
    }
    
    if not text:
        return issues
    
    # Parse field:value pairs
    lines = [l.strip() for l in text.split('\n') if ':' in l]
    values = []
    
    for line in lines:
        parts = line.split(':', 1)
        if len(parts) == 2:
            field = parts[0].strip()
            value = parts[1].strip()
            # Remove confidence markers
            for marker in ['[HIGH]', '[MEDIUM]', '[LOW]', '[فارغ]', '[غير', '---']:
                if marker in value:
                    value = value.split(marker)[0].strip()
            if value:
                values.append((field, value))
    
    if len(values) < 2:
        return issues
    
    # Check for value propagation
    from collections import Counter
    value_counts = Counter(v for _, v in values)
    
    for value, count in value_counts.most_common(3):
        if count >= 3 and len(value) > 2:
            issues["has_propagation"] = True
            affected_fields = [f for f, v in values if v == value]
            issues["propagated_values"].append({
                "value": value,
                "count": count,
                "fields": affected_fields[:5]
            })
    
    # Check for year values (1300-1500) in non-date fields
    for field, value in values:
        # Extract digits
        digits = ''.join(c for c in value if c.isdigit() or c in '٠١٢٣٤٥٦٧٨٩')
        for ar, en in [('٠','0'),('١','1'),('٢','2'),('٣','3'),('٤','4'),
                       ('٥','5'),('٦','6'),('٧','7'),('٨','8'),('٩','9')]:
            digits = digits.replace(ar, en)
        
        if len(digits) == 4:
            try:
                year = int(digits)
                if 1300 <= year <= 1500:
                    # Check if this is a date field
                    is_date_field = any(d in field.lower() for d in ['تاريخ', 'date', 'ميلاد', 'إصدار', 'انتهاء'])
                    if not is_date_field:
                        issues["year_as_value"] = True
                        issues["year_fields"].append({"field": field, "value": value})
            except ValueError:
                pass
    
    return issues


def preprocess_image(image: Image.Image, optimize_for_handwriting: bool = True) -> Image.Image:
    """
    ENHANCED preprocessing for Arabic OCR with HANDWRITING optimization.
    
    Key improvements for handwriting:
    - Preserves thin strokes (critical for Arabic handwriting)
    - Better contrast for faint handwriting
    - Gentle noise reduction that doesn't blur strokes
    - Higher effective resolution maintenance
    
    Args:
        image: Input PIL Image
        optimize_for_handwriting: If True, use gentler processing to preserve thin strokes
    """
    try:
        print(f"📸 Original image size: {image.size}, mode: {image.mode}")
        original_pixels = image.size[0] * image.size[1]
        
        # Convert to RGB
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        elif image.mode == 'L':
            image = image.convert('RGB')
        
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # =================================================================
        # STEP 1: Detect if this is a form with handwriting
        # =================================================================
        # Check for form-like structure (grid lines, boxes)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        is_form = lines is not None and len(lines) > 5
        
        if is_form:
            print("📋 Detected form-like document structure")
        
        # =================================================================
        # STEP 2: Auto-deskewing (gentle)
        # =================================================================
        if lines is not None and len(lines) > 10:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) - 90
                if -45 < angle < 45:
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                # Only correct significant skew (>2 degrees)
                if abs(median_angle) > 2:
                    print(f"✓ Rotation detected: {median_angle:.2f}°, correcting...")
                    image = image.rotate(median_angle, resample=Image.BICUBIC, expand=True, fillcolor='white')
                    img_array = np.array(image)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    print(f"✓ Image alignment good ({median_angle:.2f}°), no rotation needed")
        
        # =================================================================
        # STEP 3: Smart resizing - HIGHER limit for handwriting
        # =================================================================
        # Increased from 1600 to 2000 to preserve handwriting detail
        max_dimension = 2000 if optimize_for_handwriting else 1600
        width, height = image.size
        
        if width > max_dimension or height > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            # Use LANCZOS for best quality downscaling
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"✓ Resized: {image.size[0]}x{image.size[1]}")
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # =================================================================
        # STEP 4: GENTLE contrast enhancement for handwriting
        # =================================================================
        # Use lower CLAHE clip limit to avoid over-enhancing noise
        img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_lab)
        contrast_std = gray.std()
        
        # Gentler CLAHE for handwriting to preserve stroke integrity
        if optimize_for_handwriting:
            # Lower clip limits to avoid artifacts
            if contrast_std < 40:
                clip_limit = 2.5
                print(f"✓ Low contrast detected (std={contrast_std:.1f}), applying gentle CLAHE")
            elif contrast_std < 60:
                clip_limit = 2.0
                print(f"✓ Medium contrast (std={contrast_std:.1f}), applying mild CLAHE")
            else:
                clip_limit = 1.5
                print(f"✓ Good contrast detected (std={contrast_std:.1f}), minimal CLAHE")
        else:
            clip_limit = 3.0 if contrast_std < 40 else 2.5 if contrast_std < 60 else 2.0
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        img_clahe = cv2.merge([l_clahe, a, b])
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)
        image = Image.fromarray(img_rgb)
        
        # =================================================================
        # STEP 5: Gamma correction for brightness normalization
        # =================================================================
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        brightness_avg = gray.mean()
        
        if brightness_avg < 100:
            gamma = 0.85  # Slightly gentler than before
            print(f"✓ Image is dark (brightness={brightness_avg:.0f}), applying gamma correction")
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            img_array = cv2.LUT(img_array, table)
            image = Image.fromarray(img_array)
        elif brightness_avg > 180:
            gamma = 1.15  # Slightly gentler
            print(f"✓ Image is bright (brightness={brightness_avg:.0f}), applying gamma correction")
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            img_array = cv2.LUT(img_array, table)
            image = Image.fromarray(img_array)
        else:
            print(f"✓ Brightness is good ({brightness_avg:.0f}), no gamma needed")
        
        # =================================================================
        # STEP 6: GENTLE sharpening for handwriting
        # =================================================================
        # Important: Over-sharpening can create artifacts that confuse OCR
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_score = laplacian.var()
        
        if optimize_for_handwriting:
            # Gentler sharpening for handwriting - preserve thin strokes
            if sharpness_score < 50:
                # Very blurry - apply moderate sharpening
                radius, percent = 1.5, 100  # Reduced from 2.0, 150
                print(f"✓ Image is blurry (sharpness={sharpness_score:.0f}), applying moderate sharpen")
            elif sharpness_score < 200:
                # Slightly blurry - gentle sharpening
                radius, percent = 1.0, 80   # Reduced from 1.5, 120
                print(f"✓ Image is slightly soft (sharpness={sharpness_score:.0f}), applying gentle sharpen")
            else:
                # Sharp enough - minimal processing
                radius, percent = 0.5, 50   # Reduced from 0.5, 80
                print(f"✓ Image is sharp (sharpness={sharpness_score:.0f}), minimal sharpen")
        else:
            # Standard sharpening for printed text
            if sharpness_score < 50:
                radius, percent = 2.0, 150
            elif sharpness_score < 200:
                radius, percent = 1.5, 120
            else:
                radius, percent = 0.5, 80
        
        # Higher threshold (3) to avoid sharpening noise
        image = image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=3))
        
        final_pixels = image.size[0] * image.size[1]
        print(f"✅ Preprocessing complete: {image.size[0]}x{image.size[1]} (optimized for {'handwriting' if optimize_for_handwriting else 'standard'})")
        
        return image
        
    except Exception as e:
        print(f"⚠️ Preprocessing failed: {str(e)}")
        return image


def _compute_token_confidences(generated_ids_trimmed, scores, tokenizer) -> dict:
    """Compute per-token confidence for generated tokens."""
    try:
        if generated_ids_trimmed is None:
            token_ids = []
        elif isinstance(generated_ids_trimmed, torch.Tensor):
            if generated_ids_trimmed.dim() == 2:
                token_ids = generated_ids_trimmed[0].tolist()
            elif generated_ids_trimmed.dim() == 1:
                token_ids = generated_ids_trimmed.tolist()
            else:
                token_ids = []
        elif isinstance(generated_ids_trimmed, list):
            token_ids = generated_ids_trimmed
        else:
            token_ids = []
            
        token_confidences = []
        
        if scores is None or isinstance(scores, (int, float)):
            scores = []
        if isinstance(scores, tuple):
            scores = list(scores)

        if not scores or not token_ids:
            return {
                "overall_token_confidence": None,
                "token_confidences": [],
                "word_confidences": [],
                "line_confidences": [],
            }

        steps = min(len(scores), len(token_ids))
        for i in range(steps):
            logits = scores[i]
            tid = token_ids[i]
            probs = torch.softmax(logits, dim=-1)
            p = float(probs[0, tid].detach().cpu().item())
            token_confidences.append(p)

        overall = float(np.mean(token_confidences)) if token_confidences else None

        # Word confidences
        word_confidences = []
        current_word = ""
        current_scores = []

        def flush_word():
            nonlocal current_word, current_scores
            w = current_word.strip()
            if w:
                word_confidences.append({
                    "word": w,
                    "confidence": float(np.mean(current_scores)) if current_scores else None,
                })
            current_word = ""
            current_scores = []

        for tid, conf in zip(token_ids[:steps], token_confidences[:steps]):
            piece = tokenizer.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if piece is None:
                piece = ""
            if current_word and piece[:1].isspace():
                flush_word()
            current_word += piece
            current_scores.append(conf)

        flush_word()

        return {
            "overall_token_confidence": overall,
            "token_confidences": token_confidences,
            "word_confidences": word_confidences,
            "line_confidences": [],
        }
    except Exception as e:
        print(f"⚠️ Failed to compute confidences: {str(e)}")
        return {
            "overall_token_confidence": None,
            "token_confidences": [],
            "word_confidences": [],
            "line_confidences": [],
        }


def extract_text_from_image(
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS
) -> dict:
    """Extract text from image using Arabic VLM model."""
    try:
        if model is None or processor is None:
            raise RuntimeError("Model not loaded")
        
        print("🔧 Preprocessing image...")
        image = preprocess_image(image)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                    },
                    {
                        "type": "text",
                        "text": prompt
                    },
                ],
            }
        ]
        
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        print(f"🤖 Generating with max_new_tokens={max_new_tokens}")
        
        eos_token_id = processor.tokenizer.eos_token_id
        pad_token_id = processor.tokenizer.pad_token_id
        
        with torch.no_grad():
            # Use torch.cuda.amp for mixed precision inference (faster on A100)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                # =============================================================
                # DETERMINISTIC GENERATION SETTINGS FOR ANTI-HALLUCINATION
                # =============================================================
                # Key principles:
                # 1. do_sample=False - No random sampling
                # 2. temperature=0.0 - Deterministic (greedy)
                # 3. num_beams=1 - Simple greedy (beams can introduce variation)
                # 4. repetition_penalty=1.05 - Gentle penalty (too high hurts accuracy)
                # 5. no_repeat_ngram_size=0 - Disable (can hurt Arabic text)
                # =============================================================
                
                # =============================================================
                # SUPERIOR ANTI-HALLUCINATION GENERATION SETTINGS
                # =============================================================
                # The key insight: VLMs hallucinate when they try to "complete"
                # text they can't read. We prevent this by:
                # 1. Very low max_tokens (forces brevity)
                # 2. High repetition penalty (prevents loops)
                # 3. Strict EOS enforcement
                # =============================================================
                
                # Calculate effective max tokens (cap at reasonable limit)
                effective_max_tokens = min(max_new_tokens, 1024)
                print(f"📊 Effective max_tokens: {effective_max_tokens}")
                
                generation = model.generate(
                    **inputs,
                    max_new_tokens=effective_max_tokens,
                    
                    # ==========================================
                    # DETERMINISTIC SETTINGS (STRICT)
                    # ==========================================
                    do_sample=False,        # No random sampling
                    temperature=0.0,        # Fully deterministic
                    num_beams=1,            # Greedy decoding
                    
                    # ==========================================
                    # ANTI-HALLUCINATION SETTINGS (AGGRESSIVE)
                    # ==========================================
                    repetition_penalty=1.15,  # Higher penalty to prevent loops
                    no_repeat_ngram_size=4,   # Prevent repeating 4-grams
                    
                    # ==========================================
                    # LENGTH CONTROL (STRICT)
                    # ==========================================
                    min_new_tokens=1,       # Must generate something
                    max_length=None,        # Use max_new_tokens instead
                    early_stopping=True,    # Stop at EOS
                    
                    # ==========================================
                    # TOKEN HANDLING
                    # ==========================================
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    
                    # ==========================================
                    # OUTPUT CONFIGURATION
                    # ==========================================
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            generated_ids = generation.sequences
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        print(f"📊 Generated {generated_ids.shape[1] - inputs.input_ids.shape[1]} new tokens")
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        result = output_text[0] if output_text else ""
        result = result.strip()
        result = _clean_repetition_loops(result, max_repeats=5)
        
        if not result or len(result) < 3:
            print(f"⚠️ Very short or empty result: '{result}'")
            return {
                "text": "No text extracted",
                "token_confidence": {
                    "overall_token_confidence": None,
                    "token_confidences": [],
                    "word_confidences": [],
                    "line_confidences": [],
                },
            }
        
        print(f"✅ Extracted {len(result)} characters")
        
        scores_to_pass = generation.scores if hasattr(generation, "scores") else []
        ids_to_pass = generated_ids_trimmed[0] if isinstance(generated_ids_trimmed, list) else generated_ids_trimmed
        
        token_conf = _compute_token_confidences(ids_to_pass, scores_to_pass, processor.tokenizer)
        
        return {
            "text": result,
            "token_confidence": token_conf,
        }
        
    except Exception as e:
        error_msg = f"Error during extraction: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        raise RuntimeError(error_msg)


def _check_output_sanity(text: str) -> dict:
    """
    COMPREHENSIVE sanity check on VLM output to detect hallucinations.
    
    Checks for:
    1. Value propagation (same value appearing multiple times)
    2. Year-only values in non-date fields
    3. No Arabic content where expected
    4. Suspicious sequential/repeated numbers
    5. Impossibly short output
    6. All empty markers (nothing read)
    
    Returns:
        dict with 'passed', 'warnings', 'quality_score', and 'should_retry'
    """
    result = {
        "passed": True,
        "warnings": [],
        "quality_score": 100,
        "should_retry": False,
        "hallucination_type": None
    }
    
    if not text or len(text) < 10:
        result["quality_score"] = 50
        result["warnings"].append("OUTPUT_EMPTY: Very short or empty output")
        return result
    
    # Use the detailed detection function
    propagation_info = _detect_value_propagation(text)
    
    # Check 1: Value propagation (CRITICAL)
    if propagation_info["has_propagation"]:
        for pv in propagation_info["propagated_values"]:
            result["warnings"].append(
                f"VALUE_PROPAGATION: '{pv['value'][:30]}' appears in {pv['count']} fields: "
                f"{', '.join(pv['fields'][:3])}"
            )
            result["quality_score"] -= 35
            result["hallucination_type"] = "value_propagation"
    
    # Check 2: Year as value (CRITICAL)
    if propagation_info["year_as_value"]:
        for yf in propagation_info["year_fields"][:3]:
            result["warnings"].append(
                f"YEAR_AS_VALUE: Field '{yf['field']}' has year value '{yf['value']}'"
            )
        result["quality_score"] -= 25
        if not result["hallucination_type"]:
            result["hallucination_type"] = "year_as_value"
    
    # Parse for additional checks
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    field_lines = [l for l in lines if ':' in l]
    
    # Check 3: Value diversity
    if len(field_lines) >= 3:
        values = []
        for line in field_lines:
            parts = line.split(':', 1)
            if len(parts) == 2:
                value = parts[1].strip()
                for marker in ['[HIGH]', '[MEDIUM]', '[LOW]', '[فارغ]', '[غير', '---', '؟']:
                    value = value.replace(marker, '').strip()
                if value:
                    values.append(value)
        
        if values:
            unique_count = len(set(values))
            diversity = unique_count / len(values)
            
            if diversity < 0.3:
                result["warnings"].append(
                    f"LOW_DIVERSITY: Only {unique_count}/{len(values)} unique values ({diversity:.0%})"
                )
                result["quality_score"] -= 20
                result["hallucination_type"] = result["hallucination_type"] or "low_diversity"
    
    # Check 4: Arabic content
    has_arabic = any('\u0600' <= c <= '\u06FF' for c in text)
    if not has_arabic and len(text) > 50:
        result["warnings"].append("NO_ARABIC: No Arabic text detected in output")
        result["quality_score"] -= 15
    
    # Check 5: Suspicious patterns
    suspicious = ['1234567890', '0987654321', '1111111111', '0000000000', '9999999999']
    digits_in_text = ''.join(c for c in text if c.isdigit())
    for pattern in suspicious:
        if pattern in digits_in_text:
            result["warnings"].append(f"SUSPICIOUS_NUMBER: Sequential pattern '{pattern}' detected")
            result["quality_score"] -= 20
            break
    
    # Check 6: All empty markers
    empty_markers = ['[فارغ]', '---', '[غير مقروء]', '؟']
    non_empty_values = 0
    for line in field_lines:
        parts = line.split(':', 1)
        if len(parts) == 2:
            value = parts[1].strip()
            if value and not any(m in value for m in empty_markers):
                non_empty_values += 1
    
    if len(field_lines) >= 3 and non_empty_values == 0:
        result["warnings"].append("ALL_EMPTY: All fields marked as empty/unreadable")
        result["quality_score"] -= 10
    
    # Final determination
    result["quality_score"] = max(0, result["quality_score"])
    result["passed"] = result["quality_score"] >= 40 and result["hallucination_type"] is None
    result["should_retry"] = result["quality_score"] < 30
    
    if result["warnings"]:
        print(f"⚠️ Output sanity check:")
        for w in result["warnings"]:
            print(f"   - {w}")
        print(f"   Quality score: {result['quality_score']}/100")
        if result["hallucination_type"]:
            print(f"   🚨 Hallucination type: {result['hallucination_type']}")
    else:
        print(f"✅ Output sanity check passed (score: {result['quality_score']}/100)")
    
    return result


def handler(job):
    """RunPod handler function with output sanity checking."""
    try:
        job_input = job.get("input", {})
        
        image_b64 = job_input.get("image")
        prompt = job_input.get("prompt")
        max_new_tokens = job_input.get("max_new_tokens", DEFAULT_MAX_TOKENS)
        min_pixels = job_input.get("min_pixels", MIN_PIXELS)
        max_pixels = job_input.get("max_pixels", MAX_PIXELS)
        
        if not image_b64:
            return {"error": "No image provided"}
        
        if not prompt:
            return {"error": "No prompt provided"}
        
        try:
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
        except Exception as e:
            return {"error": f"Invalid image data: {str(e)}"}
        
        print(f"📝 Processing with prompt length: {len(prompt)}")
        extracted = extract_text_from_image(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )

        extracted_text = extracted.get("text", "") if isinstance(extracted, dict) else str(extracted)
        token_confidence = extracted.get("token_confidence") if isinstance(extracted, dict) else None

        # Run output sanity check
        sanity_result = _check_output_sanity(extracted_text)
        
        print(f"✅ Returning {len(extracted_text)} characters")
        print(f"📊 Output quality score: {sanity_result['quality_score']}/100")
        
        return {
            "text": extracted_text,
            "token_confidence": token_confidence,
            "status": "success",
            "output_quality": {
                "passed": sanity_result["passed"],
                "score": sanity_result["quality_score"],
                "warnings": sanity_result["warnings"]
            }
        }
        
    except Exception as e:
        print(f"❌ Handler error: {str(e)}")
        traceback.print_exc()
        return {"error": str(e), "status": "failed"}


# Load model on startup
print("🚀 Initializing RunPod handler with Flash Attention 2...")
load_model()
print("✅ Handler ready!")

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
