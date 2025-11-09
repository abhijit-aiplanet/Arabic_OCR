"""
RunPod Handler for AIN Vision Language Model OCR
This service runs on RunPod GPU instances and processes OCR requests
"""

import runpod
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from transformers import Qwen2VLProcessor, Qwen2VLImageProcessor
import traceback
import io
import base64

# Model configuration
MODEL_ID = "MBZUAI/AIN"

# Image resolution settings
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

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


def extract_text_from_image(
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 2048,
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
        
        # Generate output
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
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

