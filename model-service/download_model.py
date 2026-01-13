#!/usr/bin/env python3
"""
Pre-download AIN model during Docker build for faster cold starts.

Uses huggingface_hub to ONLY download files - does NOT load model into memory!
"""

import os

# Set cache directory BEFORE importing
os.environ["HF_HOME"] = "/app/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/app/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/app/hf_cache"

from huggingface_hub import hf_hub_download, list_repo_files

MODEL_ID = "MBZUAI/AIN"
CACHE_DIR = "/app/hf_cache"

print("=" * 60)
print("📥 PRE-DOWNLOADING AIN MODEL (download only, no loading)")
print("=" * 60)
print(f"Model: {MODEL_ID}")
print(f"Cache directory: {CACHE_DIR}")
print()

os.makedirs(CACHE_DIR, exist_ok=True)

try:
    # List all files in the repo
    print("Fetching file list from HuggingFace...")
    all_files = list_repo_files(MODEL_ID)
    
    # Filter to only essential files for inference
    # We need: config, safetensors weights, tokenizer
    essential_patterns = [
        '.json',           # config.json, tokenizer_config.json, etc.
        '.safetensors',    # model weights (preferred format)
        'tokenizer',       # tokenizer files
        '.model',          # sentencepiece model if any
        '.txt',            # vocab.txt, merges.txt
    ]
    
    # Skip patterns (large unnecessary files)
    skip_patterns = [
        '.gguf',
        '.ggml', 
        '.bin',            # Skip .bin, use .safetensors instead
        '.h5',
        '.msgpack',
        'pytorch_model',   # Skip old format
        'tf_model',
        'flax_model',
        'onnx',
        '.md',             # Skip docs
        '.git',
    ]
    
    # Determine which files to download
    files_to_download = []
    for f in all_files:
        f_lower = f.lower()
        
        # Skip if matches skip pattern
        should_skip = any(skip in f_lower for skip in skip_patterns)
        if should_skip:
            continue
            
        # Include if matches essential pattern
        should_include = any(pattern in f_lower for pattern in essential_patterns)
        if should_include:
            files_to_download.append(f)
    
    print(f"Found {len(all_files)} total files in repo")
    print(f"Downloading {len(files_to_download)} essential files...")
    print()
    
    # Download each file
    downloaded_size = 0
    for i, filename in enumerate(files_to_download, 1):
        print(f"[{i}/{len(files_to_download)}] {filename}")
        try:
            filepath = hf_hub_download(
                repo_id=MODEL_ID,
                filename=filename,
                cache_dir=CACHE_DIR,
            )
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                downloaded_size += size
                print(f"         ✓ {size / (1024**2):.1f} MB")
        except Exception as e:
            print(f"         ✗ Failed: {e}")
    
    print()
    print("=" * 60)
    print(f"✅ Download complete!")
    print(f"   Files: {len(files_to_download)}")
    print(f"   Size: {downloaded_size / (1024**3):.2f} GB")
    print("=" * 60)
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
