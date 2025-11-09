"""
Test script for the RunPod handler
Run this locally to test the handler before deploying
"""

import base64
from PIL import Image
import io
import sys

def test_handler():
    """Test the handler with a sample image"""
    
    print("🧪 Testing RunPod Handler...\n")
    
    # Create a simple test image with text
    test_image = Image.new('RGB', (400, 100), color='white')
    
    # Save to bytes
    buffered = io.BytesIO()
    test_image.save(buffered, format="PNG")
    image_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    # Create test job
    test_job = {
        "input": {
            "image": image_b64,
            "prompt": "Extract all text from this image.",
            "max_new_tokens": 512,
            "min_pixels": 200704,
            "max_pixels": 1003520
        }
    }
    
    print("📝 Test Job Created")
    print(f"   Image size: {len(image_b64)} bytes")
    print(f"   Prompt: {test_job['input']['prompt'][:50]}...")
    print()
    
    # Import and test handler
    try:
        from handler import handler
        print("✅ Handler imported successfully\n")
        
        print("🚀 Running inference...")
        result = handler(test_job)
        
        print("\n📊 Result:")
        print(f"   Status: {result.get('status', 'unknown')}")
        
        if 'text' in result:
            print(f"   Extracted Text: {result['text'][:100]}...")
        
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
        
        print("\n✅ Test completed!")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_handler()

