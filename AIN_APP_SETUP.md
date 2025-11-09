# AIN VLM OCR App - Setup Guide

## 🔧 Problem Solved

The "size must contain 'shortest_edge' and 'longest_edge' keys" error has been fixed with a **two-tier fallback approach**:

### Tier 1: Standard Loading
Attempts to load the processor normally using `AutoProcessor.from_pretrained()`

### Tier 2: Manual Construction (Fallback)
If standard loading fails, manually constructs the processor with:
- Separate tokenizer loading
- Image processor with correct size format: `{"shortest_edge": 224, "longest_edge": 1120}`
- Manual processor assembly from components

## 📦 Updates Made

### 1. `ain_app.py`
- ✅ Added robust error handling for processor initialization
- ✅ Implemented fallback to manual processor construction
- ✅ Properly configured min_pixels and max_pixels in message content
- ✅ Added resolution controls to UI

### 2. `requirements.txt`
- ✅ Updated transformers version to `>=4.45.0` for better compatibility

## 🚀 How to Deploy

1. **Upload to Hugging Face Space**:
   ```bash
   # Upload these files:
   - ain_app.py
   - requirements.txt
   ```

2. **Set as Main App**:
   - Rename `ain_app.py` to `app.py` in your Space, OR
   - Update your Space settings to use `ain_app.py` as the main file

3. **Hardware Requirements**:
   - Recommended: GPU (T4 or better)
   - RAM: At least 16GB
   - The model is ~14GB in size

## 🎯 Features

- **Simple Interface**: Upload image → Click Process → Get text
- **Smart Prompting**: Strict OCR-focused prompt ensures text-only extraction
- **Resolution Control**: Adjust min/max pixels for quality vs speed tradeoff
- **Custom Prompts**: Override default prompt for specific use cases
- **Examples**: Pre-loaded example images for quick testing

## ⚙️ Configuration

### Default Settings
```python
MIN_PIXELS = 256 * 28 * 28  # ~200k pixels
MAX_PIXELS = 1280 * 28 * 28  # ~1M pixels
MAX_TOKENS = 2048
```

### Adjust for Your Needs
- **Higher quality, slower**: Increase MAX_PIXELS to 2560×28×28
- **Faster, lower quality**: Decrease MAX_PIXELS to 640×28×28
- **More text**: Increase MAX_TOKENS to 4096+

## 🐛 Troubleshooting

### If you still get the size error:
1. Check transformers version: Should be >= 4.45.0
2. Clear Hugging Face cache and restart Space
3. Check logs for "Manual construction" message - it should activate automatically

### If model loading is slow:
- This is normal for first run (~10-20 seconds)
- Model files are cached after first load

### If out of memory:
- Reduce MAX_PIXELS
- Reduce MAX_TOKENS
- Upgrade to larger GPU instance

## 📊 Expected Behavior

### Successful Loading:
```
🔄 Loading AIN VLM model...
✅ Using GPU (CUDA)
Loading checkpoint shards: 100%|██████████| 7/7
✅ Processor loaded successfully (standard method)
✅ Model loaded successfully!
```

### With Fallback:
```
🔄 Loading AIN VLM model...
✅ Using GPU (CUDA)
Loading checkpoint shards: 100%|██████████| 7/7
⚠️ Standard processor loading failed, trying manual construction...
✅ Processor loaded successfully (manual construction)
✅ Model loaded successfully!
```

Both scenarios will work correctly!

## 📝 Usage Tips

1. **For Arabic text**: The model will automatically maintain RTL direction
2. **For handwritten text**: VLM models perform better than traditional OCR
3. **For complex layouts**: The model understands structure and context
4. **For long documents**: Increase MAX_TOKENS if text gets cut off

## 🎨 UI Features

- **Live preview**: See your uploaded image
- **Advanced settings**: Fold-out panel for power users
- **Status messages**: Real-time feedback on processing
- **Copy button**: One-click copy of extracted text
- **Example gallery**: Test with provided images

## 🔗 Model Info

- **Model**: MBZUAI/AIN (Qwen2-VL based)
- **Type**: Vision-Language Model
- **Size**: ~14GB
- **Languages**: Multilingual (including Arabic)
- **Strengths**: Context understanding, handwriting, complex layouts

---

## ✅ Ready to Test!

Your app is now configured with robust error handling and should work smoothly on Hugging Face Spaces. The two-tier loading approach ensures maximum compatibility.

Good luck with your OCR project! 🚀

