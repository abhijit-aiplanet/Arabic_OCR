#!/usr/bin/env python3
"""
Pre-download AIN model during Docker build for faster cold starts.
This script is run during Docker build, NOT at runtime.

OPTIMIZED: Only downloads necessary files (~7-8GB instead of 61GB)
"""

import os

# Set cache directory
os.environ["HF_HOME"] = "/app/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/app/hf_cache"

from huggingface_hub import snapshot_download

MODEL_ID = "MBZUAI/AIN"
CACHE_DIR = "/app/hf_cache"

print("=" * 60)
print("📥 PRE-DOWNLOADING AIN MODEL FOR FASTER COLD STARTS")
print("=" * 60)
print(f"Model: {MODEL_ID}")
print(f"Cache directory: {CACHE_DIR}")
print()

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

print("Downloading ONLY necessary model files...")
print("Skipping: fp32 weights, GGUF, GGML, docs (saves ~50GB)")
print()

try:
    # Download only necessary files - skip large unnecessary formats
    local_dir = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        resume_download=True,
        # Only download essential files for inference
        ignore_patterns=[
            # Skip fp32 weights (we use bf16/fp16)
            "*fp32*",
            "*FP32*",
            "*.fp32.*",
            # Skip alternative formats
            "*.gguf",
            "*.ggml", 
            "*.ggmlv3",
            "*.bin.index.json",  # Old format index
            # Skip consolidated/merged checkpoints if separate shards exist
            "consolidated*",
            "pytorch_model.bin",  # Old single-file format (use safetensors)
            # Skip documentation and non-essential files
            "*.md",
            "*.txt",
            "*.pdf",
            # Skip original/training checkpoints
            "original/",
            "training/",
        ],
    )
    
    print()
    print(f"✅ Model downloaded to: {local_dir}")
    
    # Verify files exist
    print()
    print("📊 Verifying downloaded files:")
    total_size = 0
    file_count = 0
    safetensor_count = 0
    for root, dirs, files in os.walk(CACHE_DIR):
        for f in files:
            filepath = os.path.join(root, f)
            size = os.path.getsize(filepath)
            total_size += size
            file_count += 1
            if f.endswith('.safetensors'):
                safetensor_count += 1
    
    size_gb = total_size / (1024**3)
    print(f"   Files: {file_count}")
    print(f"   Safetensor shards: {safetensor_count}")
    print(f"   Total size: {size_gb:.2f} GB")
    
    # Warn if size is still too large
    if size_gb > 15:
        print()
        print(f"⚠️  WARNING: Download size ({size_gb:.1f}GB) larger than expected (~7-8GB)")
        print("   This may include unnecessary files")
    
    print()
    print("✅ MODEL PRE-DOWNLOAD COMPLETE!")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ ERROR downloading model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
