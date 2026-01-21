"""
RunPod Handler for Qwen 2.5 Reasoning LLM

This service provides reasoning capabilities for the agentic OCR system:
- Analyzing OCR output for issues
- Estimating region coordinates for uncertain fields
- Merging original and refined OCR results
- Making intelligent decisions about final output

Build trigger: 2025-01-08-v1
"""

import runpod
import torch
import os
import json
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model selection (match download_model.py)
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"

# For 72B AWQ quantized (uncomment if using)
# MODEL_ID = "Qwen/Qwen2.5-72B-Instruct-AWQ"

# For development/testing
# MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# HuggingFace cache directory
HF_CACHE_DIR = os.environ.get('HF_HOME', '/app/hf_cache')
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR

# Generation settings
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.1  # Low for consistent reasoning
DEFAULT_TOP_P = 0.9

# Global model and tokenizer
model = None
tokenizer = None


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model():
    """Load the Qwen 2.5 model and tokenizer."""
    global model, tokenizer
    
    if model is not None and tokenizer is not None:
        print("Model already loaded")
        return
    
    print(f"=" * 60)
    print(f"Loading model: {MODEL_ID}")
    print(f"Cache directory: {HF_CACHE_DIR}")
    print(f"=" * 60)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: CUDA not available, using CPU (very slow)")
    
    try:
        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=HF_CACHE_DIR,
            trust_remote_code=True
        )
        
        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with automatic device mapping
        print("\nLoading model...")
        
        # Try to load with bitsandbytes 4-bit quantization for memory efficiency
        try:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                cache_dir=HF_CACHE_DIR,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            print("Model loaded with 4-bit quantization")
            
        except Exception as quant_error:
            print(f"4-bit quantization failed: {quant_error}")
            print("Falling back to auto dtype...")
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                cache_dir=HF_CACHE_DIR,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype="auto",
            )
        
        # Print memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"\nGPU Memory: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")
        
        print(f"\n{'=' * 60}")
        print("Model loaded successfully!")
        print(f"{'=' * 60}\n")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        raise


# =============================================================================
# TEXT GENERATION
# =============================================================================

def generate_response(
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    system_prompt: str = None
) -> dict:
    """
    Generate a response from the LLM.
    
    Args:
        prompt: User prompt/question
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = deterministic)
        top_p: Top-p sampling parameter
        system_prompt: Optional system prompt
        
    Returns:
        dict with 'text', 'tokens_generated', 'status'
    """
    global model, tokenizer
    
    if model is None or tokenizer is None:
        return {
            "text": "",
            "status": "error",
            "error": "Model not loaded"
        }
    
    try:
        # Build messages
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=8192  # Leave room for generation
        )
        
        # Move to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        input_length = inputs["input_ids"].shape[1]
        print(f"Input tokens: {input_length}")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated_ids = outputs[0][input_length:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        tokens_generated = len(generated_ids)
        print(f"Generated tokens: {tokens_generated}")
        
        return {
            "text": response_text.strip(),
            "tokens_generated": tokens_generated,
            "status": "success"
        }
        
    except Exception as e:
        print(f"Generation error: {e}")
        traceback.print_exc()
        return {
            "text": "",
            "status": "error",
            "error": str(e)
        }


def parse_json_response(text: str) -> dict:
    """
    Parse JSON from LLM response.
    
    Handles common issues:
    - JSON wrapped in markdown code blocks
    - Trailing text after JSON
    """
    # Remove markdown code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            text = text[start:end]
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end]
    
    # Try to find JSON object
    text = text.strip()
    
    # Find first { and last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Text was: {text[:500]}...")
        return None


# =============================================================================
# RUNPOD HANDLER
# =============================================================================

def handler(job):
    """
    RunPod handler for LLM inference.
    
    Expected input:
    {
        "prompt": "Your prompt here",
        "max_tokens": 2048,
        "temperature": 0.1,
        "top_p": 0.9,
        "system_prompt": "Optional system prompt",
        "parse_json": true  # If true, attempt to parse response as JSON
    }
    
    Returns:
    {
        "text": "Generated response",
        "json": {...} or null,  # Parsed JSON if parse_json=true
        "tokens_generated": 150,
        "status": "success"
    }
    """
    job_input = job.get("input", {})
    
    # Extract parameters
    prompt = job_input.get("prompt", "")
    max_tokens = job_input.get("max_tokens", DEFAULT_MAX_TOKENS)
    temperature = job_input.get("temperature", DEFAULT_TEMPERATURE)
    top_p = job_input.get("top_p", DEFAULT_TOP_P)
    system_prompt = job_input.get("system_prompt", None)
    parse_json = job_input.get("parse_json", False)
    
    if not prompt:
        return {
            "status": "error",
            "error": "No prompt provided"
        }
    
    print(f"\n{'=' * 40}")
    print(f"Processing LLM request")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Max tokens: {max_tokens}")
    print(f"Temperature: {temperature}")
    print(f"{'=' * 40}\n")
    
    # Generate response
    result = generate_response(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        system_prompt=system_prompt
    )
    
    # Parse JSON if requested
    if parse_json and result["status"] == "success":
        parsed = parse_json_response(result["text"])
        result["json"] = parsed
    
    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Qwen 2.5 Reasoning LLM Service for Agentic OCR")
    print("=" * 60 + "\n")
    
    # Load model on startup
    load_model()
    
    # Start RunPod handler
    print("Starting RunPod handler...")
    runpod.serverless.start({"handler": handler})
