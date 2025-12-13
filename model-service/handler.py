"""
RunPod Handler for AIN Vision Language Model OCR
This service runs on RunPod GPU instances and processes OCR requests
Build trigger: 2025-01-11
"""

import runpod
import torch
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

# Model configuration
MODEL_ID = "MBZUAI/AIN"

# Image resolution settings - Balanced for quality and speed
MIN_PIXELS = 256 * 28 * 28  # 200,704 - Keep same for small images
MAX_PIXELS = 1280 * 28 * 28  # 1,003,520 - Reduced for faster processing

# Maximum tokens for generation - Balanced for speed vs capacity
DEFAULT_MAX_TOKENS = 4096  # Reduced from 8192 for 2x faster generation

# Global model and processor
model = None
processor = None


def load_model():
    """Load the Arabic VLM model and processor."""
    global model, processor
    
    if model is not None and processor is not None:
        return
    
    print("🔄 Loading Arabic VLM model on RunPod...")
    
    try:
        # Use GPU if available
        if torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = "auto"
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            device_map = "cpu"
            torch_dtype = torch.float32
            print("⚠️ Using CPU (not recommended)")
        
        # Load model
        loaded_model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        
        # Load processor
        try:
            loaded_processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
            )
            print("✅ Processor loaded successfully (standard method)")
        except ValueError as e:
            if "size must contain 'shortest_edge' and 'longest_edge' keys" in str(e):
                print("⚠️ Standard processor loading failed, trying manual construction...")
                # Manually construct processor
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
                
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
        
        print("✅ Model loaded successfully and ready for inference!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        raise


def _clean_repetition_loops(text: str, max_repeats: int = 5) -> str:
    """
    Conservative cleanup for excessive repetition loops.
    Only triggers on EXTREME repetition (5+ repeats) to preserve accuracy.
    
    Args:
        text: The text to clean
        max_repeats: Maximum repeats before truncation (default: 5, more lenient)
        
    Returns:
        Cleaned text with extreme repetitions removed
    """
    if not text or len(text) < 200:
        return text
    
    # Split into lines to detect line-level repetitions
    lines = text.split('\n')
    
    # Only check for EXACT repeating patterns of 2-3 lines
    # More conservative than before (was 2-5 lines)
    for pattern_length in range(2, 4):
        if len(lines) < pattern_length * (max_repeats + 1):
            continue
        
        for start_idx in range(len(lines) - pattern_length * max_repeats):
            # Get the pattern
            pattern = lines[start_idx:start_idx + pattern_length]
            pattern_str = '\n'.join(pattern)
            
            # Only trigger if pattern is short and repetitive (likely a loop, not real data)
            if len(pattern_str) > 100:  # Skip long patterns (likely legitimate data)
                continue
            
            # Count consecutive repeats
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
            
            # Only truncate on EXTREME repetition (5+ times)
            if repeat_count > max_repeats:
                print(f"⚠️ Detected EXTREME repetition: {repeat_count} repeats of {pattern_length}-line pattern")
                print(f"🔧 Truncating (keeping first {max_repeats} occurrences)")
                
                truncated_lines = lines[:start_idx + pattern_length * max_repeats]
                result = '\n'.join(truncated_lines).strip()
                
                print(f"📉 Reduced from {len(text)} to {len(result)} characters")
                return result
    
    return text


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    ENHANCED preprocessing for Arabic OCR with VLM best practices.
    Uses CLAHE, proper Laplacian detection, gamma correction, and adaptive techniques.
    
    Args:
        image: PIL Image to preprocess
        
    Returns:
        Preprocessed PIL Image (optimized for VLM processing)
    """
    try:
        print(f"📸 Original image size: {image.size}, mode: {image.mode}")
        original_pixels = image.size[0] * image.size[1]
        
        # Convert to RGB first (required for VLM)
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
            print(f"✓ Converted to RGB")
        elif image.mode == 'L':
            image = image.convert('RGB')
            print(f"✓ Converted grayscale to RGB")
        
        # 1. AUTO-DESKEWING - Correct rotated documents
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect rotation using edge detection and Hough transform
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None and len(lines) > 10:
            # Calculate dominant angle
            angles = []
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) - 90
                # Filter out extreme angles
                if -45 < angle < 45:
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                
                # Only rotate if angle > 2 degrees (avoid unnecessary rotation)
                if abs(median_angle) > 2:
                    print(f"✓ Rotation detected: {median_angle:.2f}°, correcting...")
                    image = image.rotate(median_angle, resample=Image.BICUBIC, expand=True, fillcolor='white')
                    img_array = np.array(image)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    print(f"✓ Image alignment good ({median_angle:.2f}°), no rotation needed")
        else:
            print(f"✓ Could not detect rotation (insufficient edges), skipping deskew")
        
        # 2. SMART RESIZING - Do this early to speed up subsequent operations
        max_dimension = 1600
        width, height = image.size
        
        if width > max_dimension or height > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Use LANCZOS for high-quality downscaling
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"✓ Resized: {image.size[0]}x{image.size[1]} (was {width}x{height})")
            
            # Update arrays after resize
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 3. CLAHE CONTRAST ENHANCEMENT - Better than simple contrast
        # Convert to LAB color space for better contrast enhancement
        img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_lab)
        
        # Calculate contrast level to determine CLAHE parameters
        contrast_std = gray.std()
        
        if contrast_std < 40:  # Low contrast
            clip_limit = 3.0
            tile_size = (8, 8)
            print(f"✓ Low contrast detected (std={contrast_std:.1f}), applying strong CLAHE")
        elif contrast_std < 60:  # Medium contrast
            clip_limit = 2.5
            tile_size = (8, 8)
            print(f"✓ Medium contrast detected (std={contrast_std:.1f}), applying moderate CLAHE")
        else:  # Good contrast
            clip_limit = 2.0
            tile_size = (8, 8)
            print(f"✓ Good contrast detected (std={contrast_std:.1f}), applying mild CLAHE")
        
        # Apply CLAHE to L channel only
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l_clahe = clahe.apply(l)
        
        # Merge back and convert to RGB
        img_clahe = cv2.merge([l_clahe, a, b])
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)
        image = Image.fromarray(img_rgb)
        
        # 4. GAMMA CORRECTION - Better than linear brightness adjustment
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        brightness_avg = gray.mean()
        
        if brightness_avg < 100:  # Dark image
            gamma = 0.8  # Brighten (gamma < 1)
            print(f"✓ Dark image detected (avg={brightness_avg:.1f}), applying gamma correction (γ={gamma})")
            
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            img_array = cv2.LUT(img_array, table)
            image = Image.fromarray(img_array)
            
        elif brightness_avg > 180:  # Bright image
            gamma = 1.2  # Darken (gamma > 1)
            print(f"✓ Bright image detected (avg={brightness_avg:.1f}), applying gamma correction (γ={gamma})")
            
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            img_array = cv2.LUT(img_array, table)
            image = Image.fromarray(img_array)
        else:
            print(f"✓ Good brightness (avg={brightness_avg:.1f}), no gamma correction needed")
        
        # 5. PROPER SHARPNESS DETECTION using Laplacian
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate Laplacian variance (proper edge-based blur detection)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_score = laplacian.var()
        
        # 6. ADAPTIVE UNSHARP MASK - Better than simple sharpness enhancement
        if sharpness_score < 50:  # Very blurry
            radius = 2.0
            percent = 150
            threshold = 3
            print(f"✓ Very blurry image detected (Laplacian var={sharpness_score:.1f}), strong unsharp mask")
        elif sharpness_score < 200:  # Slightly blurry
            radius = 1.5
            percent = 120
            threshold = 3
            print(f"✓ Slightly blurry image detected (Laplacian var={sharpness_score:.1f}), moderate unsharp mask")
        else:  # Sharp enough
            radius = 0.5
            percent = 80
            threshold = 3
            print(f"✓ Sharp image (Laplacian var={sharpness_score:.1f}), minimal unsharp mask")
        
        image = image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
        
        final_pixels = image.size[0] * image.size[1]
        reduction = ((original_pixels - final_pixels) / original_pixels) * 100 if original_pixels > final_pixels else 0
        print(f"✅ Enhanced preprocessing complete: {image.size[0]}x{image.size[1]} ({reduction:.1f}% size reduction)")
        print(f"📊 Final image stats: {final_pixels:,} pixels ({final_pixels/1000000:.2f}MP)")
        
        return image
        
    except Exception as e:
        print(f"⚠️ Image preprocessing failed, using original: {str(e)}")
        import traceback
        traceback.print_exc()
        # If preprocessing fails, return original image
        return image


def extract_text_from_image(
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS
) -> str:
    """
    Extract text from image using Arabic VLM model.
    
    Args:
        image: PIL Image to process
        prompt: Prompt for text extraction
        max_new_tokens: Maximum tokens to generate
        min_pixels: Minimum image resolution
        max_pixels: Maximum image resolution
        
    Returns:
        Extracted text as string
    """
    try:
        if model is None or processor is None:
            raise RuntimeError("Model not loaded")
        
        # Preprocess image for better quality
        print("🔧 Preprocessing image...")
        image = preprocess_image(image)
        
        # Prepare messages
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
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process vision information
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        # Generate output with CONSERVATIVE anti-repetition penalties
        # Balanced for accuracy + loop prevention
        print(f"🤖 Generating with max_new_tokens={max_new_tokens}, max_pixels={max_pixels}")
        print(f"🎯 Using greedy decoding with CONSERVATIVE penalties (accuracy-focused)")
        
        # Get tokenizer tokens for better control
        eos_token_id = processor.tokenizer.eos_token_id
        pad_token_id = processor.tokenizer.pad_token_id
        print(f"🔧 Token IDs - EOS: {eos_token_id}, PAD: {pad_token_id}")
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Pure greedy decoding - deterministic
                repetition_penalty=1.1,  # CONSERVATIVE penalty (was 1.2, now less aggressive)
                no_repeat_ngram_size=4,  # Block 4-grams (was 3, now more flexible)
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        
        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        print(f"📊 Input tokens: {inputs.input_ids.shape[1]}")
        print(f"📊 Generated tokens: {generated_ids.shape[1]}")
        print(f"📊 New tokens generated: {generated_ids.shape[1] - inputs.input_ids.shape[1]}")
        print(f"🔍 First 20 generated token IDs: {generated_ids_trimmed[0][:20].tolist() if len(generated_ids_trimmed[0]) > 0 else 'None'}")
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        print(f"📝 Decoded output length: {len(output_text[0]) if output_text else 0} characters")
        print(f"📝 Output preview: {output_text[0][:200] if output_text and output_text[0] else 'EMPTY'}")
        
        result = output_text[0] if output_text else ""
        result = result.strip()
        
        # Conservative post-processing: Only clean EXTREME repetition loops
        # More lenient than before (5+ repeats vs 3+) to preserve accuracy
        result = _clean_repetition_loops(result, max_repeats=5)
        
        # Better validation - but more lenient (was < 5, now < 3)
        # Sometimes short but valid text like "نعم" or "لا" should be preserved
        if not result or len(result) < 3:
            print(f"⚠️ Very short or empty result: '{result}' ({len(result)} chars)")
            print("   This might indicate:")
            print("   - Model couldn't extract text from complex layout")
            print("   - Image quality issues")
            print("   - Model needs more tokens to generate properly")
            return "No text extracted"
        
        print(f"✅ Successfully extracted {len(result)} characters")
        return result
        
    except Exception as e:
        error_msg = f"Error during text extraction: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        raise RuntimeError(error_msg)


def handler(job):
    """
    RunPod handler function.
    
    Expected input format:
    {
        "input": {
            "image": "base64_encoded_image",
            "prompt": "text extraction prompt",
            "max_new_tokens": 2048,
            "min_pixels": 200704,
            "max_pixels": 1003520
        }
    }
    
    Returns:
    {
        "text": "extracted text"
    }
    """
    try:
        # Get job input
        job_input = job.get("input", {})
        print(f"🔍 Received job input keys: {list(job_input.keys())}")
        
        # Extract parameters
        image_b64 = job_input.get("image")
        prompt = job_input.get("prompt")
        max_new_tokens = job_input.get("max_new_tokens", DEFAULT_MAX_TOKENS)
        min_pixels = job_input.get("min_pixels", MIN_PIXELS)
        max_pixels = job_input.get("max_pixels", MAX_PIXELS)
        
        print(f"📥 Image received: {bool(image_b64)}, Prompt received: {bool(prompt)}")
        
        # Validate input
        if not image_b64:
            print("❌ No image provided")
            return {"error": "No image provided"}
        
        if not prompt:
            print("❌ No prompt provided")
            return {"error": "No prompt provided"}
        
        # Decode image from base64
        try:
            image_data = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
        except Exception as e:
            return {"error": f"Invalid image data: {str(e)}"}
        
        # Process OCR
        print(f"📝 Processing OCR with prompt length: {len(prompt)}")
        extracted_text = extract_text_from_image(
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        
        print(f"✅ Extracted text length: {len(extracted_text)}")
        print(f"📤 Returning: {extracted_text[:100]}...")
        
        result = {
            "text": extracted_text,
            "status": "success"
        }
        print(f"📦 Full result: {result}")
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Handler error: {error_msg}")
        traceback.print_exc()
        return {
            "error": error_msg,
            "status": "failed"
        }


# Load model on startup
print("🚀 Initializing RunPod handler...")
load_model()
print("✅ Handler ready!")

# Start RunPod serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})

