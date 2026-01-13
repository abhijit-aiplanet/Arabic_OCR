#!/usr/bin/env python3
"""
Pre-download AIN model during Docker build for faster cold starts.

SIMPLE APPROACH: Just use from_pretrained() - it only downloads what's needed!
This is the standard HuggingFace way and only gets the necessary files.
"""

import os

# Set cache directory BEFORE importing transformers
os.environ["HF_HOME"] = "/app/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/app/hf_cache"

print("=" * 60)
print("📥 PRE-DOWNLOADING AIN MODEL FOR FASTER COLD STARTS")
print("=" * 60)

MODEL_ID = "MBZUAI/AIN"
CACHE_DIR = "/app/hf_cache"

print(f"Model: {MODEL_ID}")
print(f"Cache directory: {CACHE_DIR}")
print()

# Create cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

try:
    # =================================================================
    # STEP 1: Download model weights using from_pretrained
    # This ONLY downloads the files needed for inference!
    # =================================================================
    print("Step 1/2: Downloading model weights...")
    print("(This downloads ONLY necessary safetensors, not fp32/GGUF)")
    print()
    
    from transformers import Qwen2VLForConditionalGeneration
    
    # Load on CPU during build (no GPU in Docker build)
    # This triggers download of only the required files
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    
    # Free memory immediately
    del model
    print("✅ Model weights downloaded!")
    print()
    
    # =================================================================
    # STEP 2: Download processor/tokenizer
    # =================================================================
    print("Step 2/2: Downloading processor/tokenizer...")
    
    from transformers import AutoProcessor, AutoTokenizer
    
    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
        del processor
    except Exception as e:
        print(f"AutoProcessor failed ({e}), trying AutoTokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
        del tokenizer
    
    print("✅ Processor/tokenizer downloaded!")
    print()
    
    # =================================================================
    # Verify download size
    # =================================================================
    print("📊 Verifying downloaded files:")
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(CACHE_DIR):
        for f in files:
            filepath = os.path.join(root, f)
            try:
                size = os.path.getsize(filepath)
                total_size += size
                file_count += 1
            except:
                pass
    
    size_gb = total_size / (1024**3)
    print(f"   Files: {file_count}")
    print(f"   Total size: {size_gb:.2f} GB")
    print()
    print("✅ MODEL PRE-DOWNLOAD COMPLETE!")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
