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
    OPTIMIZED preprocessing for Arabic OCR - smaller, cleaner, easier for model.
    Focuses on making images lightweight and high-contrast for better text extraction.
    
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
        
        # 1. SMART RESIZING - More aggressive but quality-preserving
        # Target 1600px max (down from 2048) - still excellent for OCR, but 40% fewer pixels
        max_dimension = 1600
        width, height = image.size
        
        # Always resize if larger than target, even slightly
        if width > max_dimension or height > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Use LANCZOS for high-quality downscaling (preserves text clarity)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"✓ Resized: {image.size[0]}x{image.size[1]} (was {width}x{height})")
        
        # 2. ADAPTIVE CONTRAST - Stronger for low-contrast images
        # Calculate image contrast level
        import numpy as np
        img_array = np.array(image.convert('L'))  # Temporary grayscale for analysis
        contrast_std = img_array.std()
        
        # Adaptive contrast enhancement based on image quality
        if contrast_std < 40:  # Low contrast image
            contrast_factor = 1.4  # Strong enhancement
            print(f"✓ Low contrast detected (std={contrast_std:.1f}), applying strong enhancement")
        elif contrast_std < 60:  # Medium contrast
            contrast_factor = 1.25  # Moderate enhancement
            print(f"✓ Medium contrast detected (std={contrast_std:.1f}), applying moderate enhancement")
        else:  # Good contrast
            contrast_factor = 1.15  # Mild enhancement
            print(f"✓ Good contrast detected (std={contrast_std:.1f}), applying mild enhancement")
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        
        # 3. ADAPTIVE SHARPNESS - Helps with blurry text
        # Check for blur using Laplacian variance
        laplacian_var = img_array.var()
        
        if laplacian_var < 100:  # Blurry image
            sharpness_factor = 1.5  # Strong sharpening
            print(f"✓ Blur detected (var={laplacian_var:.1f}), applying strong sharpening")
        elif laplacian_var < 300:  # Slightly blurry
            sharpness_factor = 1.3  # Moderate sharpening
            print(f"✓ Slight blur detected (var={laplacian_var:.1f}), applying moderate sharpening")
        else:  # Sharp image
            sharpness_factor = 1.1  # Minimal sharpening
            print(f"✓ Sharp image (var={laplacian_var:.1f}), applying minimal sharpening")
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness_factor)
        
        # 4. SUBTLE BRIGHTNESS NORMALIZATION (helps with dark/light images)
        # Calculate average brightness
        brightness_avg = img_array.mean()
        
        if brightness_avg < 100:  # Dark image
            brightness_factor = 1.2
            print(f"✓ Dark image detected (avg={brightness_avg:.1f}), brightening")
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)
        elif brightness_avg > 180:  # Very bright image
            brightness_factor = 0.9
            print(f"✓ Bright image detected (avg={brightness_avg:.1f}), dimming slightly")
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)
        else:
            print(f"✓ Good brightness (avg={brightness_avg:.1f}), no adjustment needed")
        
        final_pixels = image.size[0] * image.size[1]
        reduction = ((original_pixels - final_pixels) / original_pixels) * 100 if original_pixels > final_pixels else 0
        print(f"✅ Preprocessing complete: {image.size[0]}x{image.size[1]} ({reduction:.1f}% size reduction)")
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

