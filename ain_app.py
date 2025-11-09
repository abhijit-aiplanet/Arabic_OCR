import spaces
import gradio as gr
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from transformers import Qwen2VLProcessor, Qwen2VLImageProcessor
import traceback
import json
import os

# ========================================
# AIN VLM MODEL FOR OCR
# ========================================

# Model configuration
MODEL_ID = "MBZUAI/AIN"

# Image resolution settings for the processor
# The default range for the number of visual tokens per image in the model is 4-16384
# These settings balance speed and memory usage
MIN_PIXELS = 256 * 28 * 28  # Minimum resolution
MAX_PIXELS = 1280 * 28 * 28  # Maximum resolution

# Global model and processor
model = None
processor = None

# Strict OCR-focused prompt
OCR_PROMPT = """Extract all text from this image exactly as it appears. 

Requirements:
1. Extract ONLY the text content - do not describe, analyze, or interpret the image
2. Maintain the original text structure, layout, and formatting
3. Preserve line breaks, paragraphs, and spacing as they appear
4. Do not translate the text - keep it in its original language
5. Do not add any explanations, descriptions, or additional commentary
6. If there are tables, maintain their structure
7. If there are headers, titles, or sections, preserve their hierarchy

Output only the extracted text, nothing else."""


def ensure_model_loaded():
    """Lazily load the AIN VLM model and processor."""
    global model, processor
    
    if model is not None and processor is not None:
        return
    
    print("🔄 Loading AIN VLM model...")
    
    try:
        # Determine device and dtype
        if torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = "auto"
            print("✅ Using GPU (CUDA)")
        else:
            device_map = "cpu"
            torch_dtype = torch.float32
            print("✅ Using CPU")
        
        # Load model
        loaded_model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        
        # Load processor with proper configuration
        # Manual construction to avoid size parameter issues
        try:
            # First, try the standard way
            loaded_processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                trust_remote_code=True,
            )
            print("✅ Processor loaded successfully (standard method)")
        except ValueError as e:
            if "size must contain 'shortest_edge' and 'longest_edge' keys" in str(e):
                print("⚠️ Standard processor loading failed, trying manual construction...")
                # Manually construct processor with correct size format
                try:
                    # Load tokenizer separately
                    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
                    
                    # Create image processor with correct size format
                    image_processor = Qwen2VLImageProcessor(
                        size={"shortest_edge": 224, "longest_edge": 1120},  # Valid format
                        do_resize=True,
                        do_rescale=True,
                        do_normalize=True,
                    )
                    
                    # Create processor from components
                    loaded_processor = Qwen2VLProcessor(
                        image_processor=image_processor,
                        tokenizer=tokenizer,
                    )
                    print("✅ Processor loaded successfully (manual construction)")
                except Exception as manual_error:
                    print(f"❌ Manual construction also failed: {manual_error}")
                    raise
            else:
                raise
        
        model = loaded_model
        processor = loaded_processor
        
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        raise


@spaces.GPU()
def extract_text_from_image(
    image: Image.Image, 
    custom_prompt: str = None, 
    max_new_tokens: int = 2048,
    min_pixels: int = None,
    max_pixels: int = None
) -> str:
    """
    Extract text from image using AIN VLM model.
    
    Args:
        image: PIL Image to process
        custom_prompt: Optional custom prompt (uses default OCR prompt if None)
        max_new_tokens: Maximum tokens to generate
        min_pixels: Minimum image resolution (optional)
        max_pixels: Maximum image resolution (optional)
        
    Returns:
        Extracted text as string
    """
    try:
        # Ensure model is loaded
        ensure_model_loaded()
        
        if model is None or processor is None:
            return "❌ Error: Model not loaded. Please refresh and try again."
        
        # Use custom prompt or default OCR prompt
        prompt_to_use = custom_prompt if custom_prompt and custom_prompt.strip() else OCR_PROMPT
        
        # Use custom resolution settings if provided, otherwise use defaults
        min_pix = min_pixels if min_pixels else MIN_PIXELS
        max_pix = max_pixels if max_pixels else MAX_PIXELS
        
        # Prepare messages in the format expected by the model
        # Include min_pixels and max_pixels in the image content for proper resizing
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": min_pix,
                        "max_pixels": max_pix,
                    },
                    {
                        "type": "text",
                        "text": prompt_to_use
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
                do_sample=False,  # Greedy decoding for consistency
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
        error_msg = f"❌ Error during text extraction: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg


def create_gradio_interface():
    """Create the Gradio interface for AIN OCR."""
    
    # Custom CSS for better UI
    css = """
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .header-text {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 30px;
    }
    
    .process-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 1.1em !important;
        padding: 12px 24px !important;
        width: 100% !important;
        margin-top: 10px !important;
    }
    
    .process-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2) !important;
    }
    
    /* Larger font for extracted text */
    .output-textbox textarea {
        font-size: 20px !important;
        line-height: 2.0 !important;
        font-family: 'Segoe UI', 'Tahoma', 'Traditional Arabic', 'Arabic Typesetting', sans-serif !important;
        padding: 24px !important;
        direction: auto !important;
        text-align: start !important;
    }
    
    .output-textbox {
        background: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Better Arabic text support */
    .output-textbox textarea[dir="rtl"] {
        text-align: right !important;
        direction: rtl !important;
    }
    
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
    }
    
    /* Status box styling */
    .status-box {
        background: #f0f4f8;
        border: 1px solid #d0dae6;
        border-radius: 6px;
        padding: 12px;
        margin-top: 10px;
        text-align: center;
        font-size: 14px;
    }
    
    /* Better spacing for rows and columns */
    .gradio-container {
        gap: 20px !important;
    }
    
    .contain {
        gap: 15px !important;
    }
    
    /* Image preview styling */
    .image-preview {
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Accordion styling */
    .accordion {
        background: #f8f9fa;
        border-radius: 8px;
        margin-top: 15px;
        padding: 5px;
    }
    
    /* Clear button */
    button[variant="secondary"] {
        width: 100% !important;
        margin-top: 10px !important;
    }
    
    /* Label styling */
    label {
        font-weight: 600 !important;
        margin-bottom: 8px !important;
    }
    
    /* Better component spacing */
    .gr-form {
        gap: 12px !important;
    }
    
    /* Example images styling */
    .gr-examples {
        margin-top: 15px;
    }
    """
    
    with gr.Blocks(theme=gr.themes.Soft(), css=css, title="AIN VLM OCR") as demo:
        
        # Header
        gr.HTML("""
        <div class="header-text">
            <h1>🔍 AIN VLM - Vision Language Model OCR</h1>
            <p style="font-size: 1.1em; color: #6b7280; margin-top: 10px;">
                Advanced OCR using Vision Language Model (VLM) for accurate text extraction
            </p>
            <p style="font-size: 0.95em; color: #9ca3af; margin-top: 8px;">
                Powered by <strong>MBZUAI/AIN</strong> - Specialized for understanding and extracting text from images
            </p>
        </div>
        """)
        
        # Info box
        gr.Markdown("""
        <div class="info-box">
        <strong>ℹ️ How it works:</strong> Upload an image containing text, click "Process Image", and get the extracted text.
        The VLM model intelligently understands context and can handle handwritten text better than traditional OCR models.
        </div>
        """)
        
        # Main interface
        with gr.Row(equal_height=False):
            # Left column - Input
            with gr.Column(scale=1, min_width=400):
                # Image input
                image_input = gr.Image(
                    label="📸 Upload Image",
                    type="pil",
                    height=400,
                    elem_classes=["image-preview"]
                )
                
                # Advanced settings
                with gr.Accordion("⚙️ Advanced Settings", open=False, elem_classes=["accordion"]):
                    custom_prompt = gr.Textbox(
                        label="Custom Prompt (Optional)",
                        placeholder="Leave empty to use default OCR prompt...",
                        lines=3,
                        info="Customize the prompt if you want specific extraction behavior"
                    )
                    
                    max_tokens = gr.Slider(
                        minimum=512,
                        maximum=4096,
                        value=2048,
                        step=128,
                        label="Max Tokens",
                        info="Maximum length of extracted text"
                    )
                    
                    gr.Markdown("**📐 Image Resolution Settings**")
                    gr.Markdown("*Controls visual token range (4-16384) - balance quality vs speed*")
                    
                    with gr.Row():
                        min_pixels_input = gr.Number(
                            value=MIN_PIXELS,
                            label="Min Pixels",
                            info=f"Default: {MIN_PIXELS:,} (~{MIN_PIXELS//1000}k)",
                            precision=0
                        )
                        max_pixels_input = gr.Number(
                            value=MAX_PIXELS,
                            label="Max Pixels",
                            info=f"Default: {MAX_PIXELS:,} (~{MAX_PIXELS//1000}k)",
                            precision=0
                        )
                    
                    show_prompt_btn = gr.Button("👁️ Show Default Prompt", size="sm", variant="secondary")
                
                # Process button
                process_btn = gr.Button(
                    "🚀 Process Image",
                    variant="primary",
                    elem_classes=["process-button"],
                    size="lg"
                )
                
                # Clear button
                clear_btn = gr.Button("🗑️ Clear All", variant="secondary", size="lg")
            
            # Right column - Output
            with gr.Column(scale=1, min_width=500):
                # Text output with larger font
                text_output = gr.Textbox(
                    label="📝 Extracted Text",
                    placeholder="Extracted text will appear here...",
                    lines=18,
                    max_lines=22,
                    show_copy_button=True,
                    interactive=False,
                    elem_classes=["output-textbox"],
                    container=True,
                )
                
                # Status/info
                status_output = gr.Markdown(
                    value="✨ *Ready to process images*",
                    elem_classes=["status-box"]
                )
        
        # Examples section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📚 Example Images")
                gr.Markdown("*Click on any example below to load it*")
                gr.Examples(
                    examples=[
                        ["image/app/1762329983969.png"],
                        ["image/app/1762330009302.png"],
                        ["image/app/1762330020168.png"],
                    ],
                    inputs=image_input,
                    label="",
                    examples_per_page=3
                )
        
        # Default prompt display
        default_prompt_display = gr.Textbox(
            label="Default OCR Prompt",
            value=OCR_PROMPT,
            lines=10,
            visible=False,
            interactive=False
        )
        
        # Event handlers
        def process_image_handler(image, custom_prompt_text, max_tokens_value, min_pix, max_pix):
            """Handle image processing."""
            if image is None:
                return "", "⚠️ Please upload an image first."
            
            try:
                status = "⏳ Processing image..."
                extracted_text = extract_text_from_image(
                    image,
                    custom_prompt=custom_prompt_text,
                    max_new_tokens=int(max_tokens_value),
                    min_pixels=int(min_pix) if min_pix else None,
                    max_pixels=int(max_pix) if max_pix else None
                )
                
                if extracted_text and not extracted_text.startswith("❌"):
                    status = f"✅ Text extracted successfully! ({len(extracted_text)} characters)"
                else:
                    status = "⚠️ No text extracted or error occurred."
                
                return extracted_text, status
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                return error_msg, "❌ Processing failed."
        
        def clear_all_handler():
            """Clear all inputs and outputs."""
            return None, "", "", "✨ Ready to process images"
        
        def toggle_prompt_display(current_visible):
            """Toggle the visibility of the default prompt."""
            return gr.update(visible=not current_visible)
        
        # Wire up events
        process_btn.click(
            process_image_handler,
            inputs=[image_input, custom_prompt, max_tokens, min_pixels_input, max_pixels_input],
            outputs=[text_output, status_output]
        )
        
        clear_btn.click(
            clear_all_handler,
            outputs=[image_input, text_output, custom_prompt, status_output]
        )
        
        # Show/hide default prompt
        show_prompt_btn.click(
            lambda: gr.update(visible=True),
            outputs=[default_prompt_display]
        )
    
    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_gradio_interface()
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )

