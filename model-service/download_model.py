#!/usr/bin/env python3
"""
Pre-download AIN model during Docker build for faster cold starts.
This script is run during Docker build, NOT at runtime.
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

print("Downloading model files from HuggingFace Hub...")
print("This may take several minutes (~7GB)...")
print()

try:
    # Download all model files
    local_dir = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        resume_download=True,
    )
    
    print()
    print(f"✅ Model downloaded to: {local_dir}")
    
    # Verify files exist
    print()
    print("📊 Verifying downloaded files:")
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(CACHE_DIR):
        for f in files:
            filepath = os.path.join(root, f)
            size = os.path.getsize(filepath)
            total_size += size
            file_count += 1
    
    print(f"   Files: {file_count}")
    print(f"   Total size: {total_size / (1024**3):.2f} GB")
    print()
    print("✅ MODEL PRE-DOWNLOAD COMPLETE!")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ ERROR downloading model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
