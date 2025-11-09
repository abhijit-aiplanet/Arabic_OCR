"""
RunPod Handler for AIN Vision Language Model OCR
This service runs on RunPod GPU instances and processes OCR requests
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

# Image resolution settings - Increased for better handling of large images with lots of text
MIN_PIXELS = 256 * 28 * 28  # 200,704 - Keep same for small images
MAX_PIXELS = 2560 * 28 * 28  # 2,007,040 - 2x increase for large images

# Maximum tokens for generation - Increased for longer documents
DEFAULT_MAX_TOKENS = 8192  # 4x increase from 2048

# Global model and processor
model = None
processor = None


def load_model():
    """Load the AIN VLM model and processor."""
    global model, processor
    
    if model is not None and processor is not None:
        return
    
    print("🔄 Loading AIN VLM model on RunPod...")
    
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


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess image for better OCR quality.
    Applies enhancement techniques while preserving text quality.
    
    Args:
        image: PIL Image to preprocess
        
    Returns:
        Preprocessed PIL Image
    """
    try:
        print(f"📸 Original image size: {image.size}, mode: {image.mode}")
        
        # Convert to RGB if needed
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
            print(f"✓ Converted to RGB")
        
        # 1. Resize if image is too large (max 4K resolution for efficiency)
        max_dimension = 4096
        width, height = image.size
        if width > max_dimension or height > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"✓ Resized to: {image.size}")
        
        # 2. Enhance contrast for better text visibility
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)  # 30% contrast increase
        print(f"✓ Enhanced contrast")
        
        # 3. Enhance sharpness for clearer text edges
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)  # 50% sharpness increase
        print(f"✓ Enhanced sharpness")
        
        # 4. Slight brightness adjustment for better readability
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)  # 10% brightness increase
        print(f"✓ Adjusted brightness")
        
        # 5. Apply subtle unsharp mask for text clarity (but not too aggressive)
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        print(f"✓ Applied unsharp mask")
        
        print(f"✅ Image preprocessing complete: {image.size}")
        return image
        
    except Exception as e:
        print(f"⚠️ Image preprocessing failed, using original: {str(e)}")
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
    Extract text from image using AIN VLM model.
    
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
        
        # Generate output with optimized parameters for long documents
        print(f"🤖 Generating with max_new_tokens={max_new_tokens}, max_pixels={max_pixels}")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,  # Enable sampling for better quality
                temperature=0.3,  # Low temperature for focused, deterministic output
                top_p=0.9,  # Nucleus sampling for quality
                top_k=50,  # Limit to top 50 tokens
                repetition_penalty=1.1,  # Slight penalty to avoid repetition
                num_beams=1,  # Greedy decoding for speed (can increase to 3-5 for quality)
            )
        
        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        result = output_text[0] if output_text else ""
        return result.strip() if result else "No text extracted"
        
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
        max_new_tokens = job_input.get("max_new_tokens", 2048)
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

