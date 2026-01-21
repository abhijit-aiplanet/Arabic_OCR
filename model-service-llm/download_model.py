"""
Pre-download Qwen 2.5 model during Docker build for fast cold starts.

This script downloads only the necessary model files without loading into memory.
Uses huggingface_hub for efficient downloading.
"""

import os
from huggingface_hub import snapshot_download

# Model configuration
# Using Qwen 2.5 32B-Instruct as it fits better on most GPUs
# For 72B, you need 48GB+ VRAM or use AWQ quantization
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"

# Alternative: Use AWQ quantized 72B model (fits in ~24GB)
# MODEL_ID = "Qwen/Qwen2.5-72B-Instruct-AWQ"

# For testing/development, you can use smaller model:
# MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

CACHE_DIR = os.environ.get('HF_HOME', '/app/hf_cache')

def download_model():
    """Download model files to cache directory."""
    print(f"=" * 60)
    print(f"Downloading model: {MODEL_ID}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"=" * 60)
    
    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Download model with smart file filtering
    # Ignore unnecessary files to reduce download size
    ignore_patterns = [
        "*.md",
        "*.txt",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.gif",
        "LICENSE*",
        "NOTICE*",
        ".gitattributes",
        "original/",  # Skip original fp32 weights if AWQ version
        "*.gguf",     # Skip GGUF format
        "*.ggml",     # Skip GGML format
    ]
    
    print(f"\nStarting download...")
    print(f"This may take a while for large models.\n")
    
    try:
        local_path = snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=CACHE_DIR,
            ignore_patterns=ignore_patterns,
            resume_download=True,
        )
        
        print(f"\n{'=' * 60}")
        print(f"Download complete!")
        print(f"Model saved to: {local_path}")
        print(f"{'=' * 60}")
        
        # List downloaded files
        print(f"\nDownloaded files:")
        total_size = 0
        for root, dirs, files in os.walk(local_path):
            for file in files:
                filepath = os.path.join(root, file)
                size = os.path.getsize(filepath)
                total_size += size
                if size > 1024 * 1024:  # Only show files > 1MB
                    print(f"  {file}: {size / (1024*1024*1024):.2f} GB")
        
        print(f"\nTotal size: {total_size / (1024*1024*1024):.2f} GB")
        
    except Exception as e:
        print(f"\nError downloading model: {e}")
        print(f"The model will be downloaded at runtime (slower cold start)")
        raise

if __name__ == "__main__":
    download_model()
