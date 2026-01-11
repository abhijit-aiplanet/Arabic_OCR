"""
RunPod Handler for AIN Vision Language Model OCR
This service runs on RunPod GPU instances and processes OCR requests
Build trigger: 2025-01-11-v2 (Flash Attention 2 + Improved Structured Extraction)
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
    """Load the Arabic VLM model and processor with Flash Attention 2 for A100 GPU."""
    global model, processor
    
    if model is not None and processor is not None:
        return
    
    print("🔄 Loading Arabic VLM model on RunPod...")
    
    try:
        # Use GPU if available
        if torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = torch.bfloat16  # Use bfloat16 for A100 optimization
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
            # Check for Flash Attention 2 support
            gpu_name = torch.cuda.get_device_name(0).lower()
            supports_flash_attn = any(x in gpu_name for x in ['a100', 'a10', 'h100', 'l40', 'rtx 40', 'rtx 30'])
            
            if supports_flash_attn:
                print("🚀 Flash Attention 2 supported! Enabling for maximum performance...")
                attn_implementation = "flash_attention_2"
            else:
                print("⚠️ Flash Attention 2 not supported on this GPU, using default attention")
                attn_implementation = "sdpa"  # Scaled dot product attention fallback
        else:
            device_map = "cpu"
            torch_dtype = torch.float32
            attn_implementation = "eager"
            print("⚠️ Using CPU (not recommended)")
        
        # Load model with Flash Attention 2 for A100
        print(f"🔧 Loading model with attention implementation: {attn_implementation}")
        loaded_model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation=attn_implementation,  # Flash Attention 2 for A100!
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


def _clean_repetition_loops(text: str, max_repeats: int = 5) -> str:
    """
    Conservative cleanup for excessive repetition loops.
    Only triggers on EXTREME repetition (5+ repeats) to preserve accuracy.
    """
    if not text or len(text) < 200:
        return text
    
    lines = text.split('\n')
    
    for pattern_length in range(2, 4):
        if len(lines) < pattern_length * (max_repeats + 1):
            continue
        
        for start_idx in range(len(lines) - pattern_length * max_repeats):
            pattern = lines[start_idx:start_idx + pattern_length]
            pattern_str = '\n'.join(pattern)
            
            if len(pattern_str) > 100:
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
            
            if repeat_count > max_repeats:
                print(f"⚠️ Detected EXTREME repetition: {repeat_count} repeats")
                truncated_lines = lines[:start_idx + pattern_length * max_repeats]
                result = '\n'.join(truncated_lines).strip()
                print(f"📉 Reduced from {len(text)} to {len(result)} characters")
                return result
    
    return text


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    ENHANCED preprocessing for Arabic OCR with VLM best practices.
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
        
        # Auto-deskewing
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        if lines is not None and len(lines) > 10:
            angles = []
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) - 90
                if -45 < angle < 45:
                    angles.append(angle)
            
            if angles:
                median_angle = np.median(angles)
                if abs(median_angle) > 2:
                    print(f"✓ Rotation detected: {median_angle:.2f}°, correcting...")
                    image = image.rotate(median_angle, resample=Image.BICUBIC, expand=True, fillcolor='white')
                    img_array = np.array(image)
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Smart resizing
        max_dimension = 1600
        width, height = image.size
        
        if width > max_dimension or height > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"✓ Resized: {image.size[0]}x{image.size[1]}")
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # CLAHE contrast enhancement
        img_lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_lab)
        contrast_std = gray.std()
        
        clip_limit = 3.0 if contrast_std < 40 else 2.5 if contrast_std < 60 else 2.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        img_clahe = cv2.merge([l_clahe, a, b])
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)
        image = Image.fromarray(img_rgb)
        
        # Gamma correction
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        brightness_avg = gray.mean()
        
        if brightness_avg < 100:
            gamma = 0.8
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            img_array = cv2.LUT(img_array, table)
            image = Image.fromarray(img_array)
        elif brightness_avg > 180:
            gamma = 1.2
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            img_array = cv2.LUT(img_array, table)
            image = Image.fromarray(img_array)
        
        # Sharpness detection and enhancement
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_score = laplacian.var()
        
        if sharpness_score < 50:
            radius, percent = 2.0, 150
        elif sharpness_score < 200:
            radius, percent = 1.5, 120
        else:
            radius, percent = 0.5, 80
        
        image = image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=3))
        
        final_pixels = image.size[0] * image.size[1]
        print(f"✅ Preprocessing complete: {image.size[0]}x{image.size[1]}")
        
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
                generation = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=4,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
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


def handler(job):
    """RunPod handler function."""
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

        print(f"✅ Returning {len(extracted_text)} characters")
        
        return {
            "text": extracted_text,
            "token_confidence": token_confidence,
            "status": "success"
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
