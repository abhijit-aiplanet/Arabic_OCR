import spaces
import json
import math
import os
import traceback
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import gradio as gr
import requests
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFont
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoProcessor

print(torch.__version__)

# Constants
MIN_PIXELS = 3136
MAX_PIXELS = 11289600
IMAGE_FACTOR = 28

# Prompts
prompt = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

# Utility functions
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 3136,
    max_pixels: int = 11289600,
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = round_by_factor(height / beta, factor)
        w_bar = round_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = round_by_factor(height * beta, factor)
        w_bar = round_by_factor(width * beta, factor)
    return h_bar, w_bar


def fetch_image(image_input, min_pixels: int = None, max_pixels: int = None):
    """Fetch and process an image"""
    if isinstance(image_input, str):
        if image_input.startswith(("http://", "https://")):
            response = requests.get(image_input)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')
    else:
        raise ValueError(f"Invalid image input type: {type(image_input)}")
    
    if min_pixels is not None or max_pixels is not None:
        min_pixels = min_pixels or MIN_PIXELS
        max_pixels = max_pixels or MAX_PIXELS
        height, width = smart_resize(
            image.height, 
            image.width, 
            factor=IMAGE_FACTOR,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        image = image.resize((width, height), Image.LANCZOS)
    
    return image


def load_images_from_pdf(pdf_path: str) -> List[Image.Image]:
    """Load images from PDF file"""
    images = []
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            # Convert page to image
            mat = fitz.Matrix(2.0, 2.0)  # Increase resolution
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("ppm")
            image = Image.open(BytesIO(img_data)).convert('RGB')
            images.append(image)
        pdf_document.close()
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []
    return images


def draw_layout_on_image(image: Image.Image, layout_data: List[Dict]) -> Image.Image:
    """Draw layout bounding boxes on image"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Colors for different categories
    colors = {
        'Caption': '#FF6B6B',
        'Footnote': '#4ECDC4', 
        'Formula': '#45B7D1',
        'List-item': '#96CEB4',
        'Page-footer': '#FFEAA7',
        'Page-header': '#DDA0DD',
        'Picture': '#FFD93D',
        'Section-header': '#6C5CE7',
        'Table': '#FD79A8',
        'Text': '#74B9FF',
        'Title': '#E17055'
    }
    
    try:
        # Load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
        
        for item in layout_data:
            if 'bbox' in item and 'category' in item:
                bbox = item['bbox']
                category = item['category']
                color = colors.get(category, '#000000')
                
                # Draw rectangle
                draw.rectangle(bbox, outline=color, width=2)
                
                # Draw label
                label = category
                label_bbox = draw.textbbox((0, 0), label, font=font)
                label_width = label_bbox[2] - label_bbox[0]
                label_height = label_bbox[3] - label_bbox[1]
                
                # Position label above the box
                label_x = bbox[0]
                label_y = max(0, bbox[1] - label_height - 2)
                
                # Draw background for label
                draw.rectangle(
                    [label_x, label_y, label_x + label_width + 4, label_y + label_height + 2],
                    fill=color
                )
                
                # Draw text
                draw.text((label_x + 2, label_y + 1), label, fill='white', font=font)
                
    except Exception as e:
        print(f"Error drawing layout: {e}")
    
    return img_copy


def layoutjson2md(image: Image.Image, layout_data: List[Dict], text_key: str = 'text', no_page_hf: bool = False) -> str:
    """Convert layout JSON to markdown format"""
    markdown_lines = []
    
    if not no_page_hf:
        markdown_lines.append("# Document Content\n")
    
    try:
        # Sort items by reading order (top to bottom, left to right)
        sorted_items = sorted(layout_data, key=lambda x: (x.get('bbox', [0, 0, 0, 0])[1], x.get('bbox', [0, 0, 0, 0])[0]))
        
        for item in sorted_items:
            category = item.get('category', '')
            text = item.get(text_key, '')
            
            if not text:
                continue
                
            if category == 'Title':
                markdown_lines.append(f"# {text}\n")
            elif category == 'Section-header':
                markdown_lines.append(f"## {text}\n")
            elif category == 'Text':
                markdown_lines.append(f"{text}\n")
            elif category == 'List-item':
                markdown_lines.append(f"- {text}\n")
            elif category == 'Table':
                # If text is already HTML, keep it as is
                if text.strip().startswith('<'):
                    markdown_lines.append(f"{text}\n")
                else:
                    markdown_lines.append(f"**Table:** {text}\n")
            elif category == 'Formula':
                # If text is LaTeX, format it properly
                if text.strip().startswith('$') or '\\' in text:
                    markdown_lines.append(f"$$\n{text}\n$$\n")
                else:
                    markdown_lines.append(f"**Formula:** {text}\n")
            elif category == 'Caption':
                markdown_lines.append(f"*{text}*\n")
            elif category == 'Footnote':
                markdown_lines.append(f"^{text}^\n")
            elif category in ['Page-header', 'Page-footer']:
                # Skip headers and footers in main content
                continue
            else:
                markdown_lines.append(f"{text}\n")
            
            markdown_lines.append("")  # Add spacing
            
    except Exception as e:
        print(f"Error converting to markdown: {e}")
        return str(layout_data)
    
    return "\n".join(markdown_lines)

# Initialize model and processor at script level
model_id = "rednote-hilab/dots.ocr"
model_path = "./models/dots-ocr-local"
snapshot_download(
    repo_id=model_id,
    local_dir=model_path,
    local_dir_use_symlinks=False, # Recommended to set to False to avoid symlink issues
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    model_path, 
    trust_remote_code=True
)

# Global state variables
device = "cuda" if torch.cuda.is_available() else "cpu"

# PDF handling state
pdf_cache = {
    "images": [],
    "current_page": 0,
    "total_pages": 0,
    "file_type": None,
    "is_parsed": False,
    "results": []
}

# Processing state
processing_results = {
    'original_image': None,
    'processed_image': None,
    'layout_result': None,
    'markdown_content': None,
    'raw_output': None,
}
@spaces.gpu
def inference(image: Image.Image, prompt: str, max_new_tokens: int = 24000) -> str:
    """Run inference on an image with the given prompt"""
    try:
        if model is None or processor is None:
            raise RuntimeError("Model not loaded. Please check model initialization.")
        
        # Prepare messages in the expected format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {"type": "text", "text": prompt}
                ]
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
        inputs = inputs.to(device)
        
        # Generate output
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1
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
        
        return output_text[0] if output_text else ""
        
    except Exception as e:
        print(f"Error during inference: {e}")
        traceback.print_exc()
        return f"Error during inference: {str(e)}"


def process_image(
    image: Image.Image, 
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None
) -> Dict[str, Any]:
    """Process a single image with the specified prompt mode"""
    try:
        # Resize image if needed
        if min_pixels is not None or max_pixels is not None:
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        
        # Run inference with the default prompt
        raw_output = inference(image, prompt)
        
        # Process results based on prompt mode
        result = {
            'original_image': image,
            'raw_output': raw_output,
            'processed_image': image,
            'layout_result': None,
            'markdown_content': None
        }
        
        # Try to parse JSON and create visualizations (since we're doing layout analysis)
        try:
            # Try to parse JSON output
            layout_data = json.loads(raw_output)
            result['layout_result'] = layout_data
            
            # Create visualization with bounding boxes
            try:
                processed_image = draw_layout_on_image(image, layout_data)
                result['processed_image'] = processed_image
            except Exception as e:
                print(f"Error drawing layout: {e}")
                result['processed_image'] = image
            
            # Generate markdown from layout data
            try:
                markdown_content = layoutjson2md(image, layout_data, text_key='text')
                result['markdown_content'] = markdown_content
            except Exception as e:
                print(f"Error generating markdown: {e}")
                result['markdown_content'] = raw_output
            
        except json.JSONDecodeError:
            print("Failed to parse JSON output, using raw output")
            result['markdown_content'] = raw_output
        
        return result
        
    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()
        return {
            'original_image': image,
            'raw_output': f"Error processing image: {str(e)}",
            'processed_image': image,
            'layout_result': None,
            'markdown_content': f"Error processing image: {str(e)}"
        }


def load_file_for_preview(file_path: str) -> Tuple[Optional[Image.Image], str]:
    """Load file for preview (supports PDF and images)"""
    global pdf_cache
    
    if not file_path or not os.path.exists(file_path):
        return None, "No file selected"
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.pdf':
            # Load PDF pages
            images = load_images_from_pdf(file_path)
            if not images:
                return None, "Failed to load PDF"
            
            pdf_cache.update({
                "images": images,
                "current_page": 0,
                "total_pages": len(images),
                "file_type": "pdf",
                "is_parsed": False,
                "results": []
            })
            
            return images[0], f"Page 1 / {len(images)}"
            
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            # Load single image
            image = Image.open(file_path).convert('RGB')
            
            pdf_cache.update({
                "images": [image],
                "current_page": 0,
                "total_pages": 1,
                "file_type": "image",
                "is_parsed": False,
                "results": []
            })
            
            return image, "Page 1 / 1"
        else:
            return None, f"Unsupported file format: {file_ext}"
            
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, f"Error loading file: {str(e)}"


def turn_page(direction: str) -> Tuple[Optional[Image.Image], str, str]:
    """Navigate through PDF pages"""
    global pdf_cache
    
    if not pdf_cache["images"]:
        return None, "No file loaded", "No results yet"
    
    if direction == "prev":
        pdf_cache["current_page"] = max(0, pdf_cache["current_page"] - 1)
    elif direction == "next":
        pdf_cache["current_page"] = min(
            pdf_cache["total_pages"] - 1, 
            pdf_cache["current_page"] + 1
        )
    
    index = pdf_cache["current_page"]
    current_image = pdf_cache["images"][index]
    page_info = f"Page {index + 1} / {pdf_cache['total_pages']}"
    
    # Get results for current page if available
    current_result = ""
    if (pdf_cache["is_parsed"] and 
        index < len(pdf_cache["results"]) and 
        pdf_cache["results"][index]):
        result = pdf_cache["results"][index]
        if result.get('markdown_content'):
            current_result = result['markdown_content']
        else:
            current_result = result.get('raw_output', 'No content available')
    else:
        current_result = "Page not processed yet"
    
    return current_image, page_info, current_result


def create_gradio_interface():
    """Create the Gradio interface"""
    
    # Custom CSS
    css = """
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .header-text {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    
    .process-button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: bold !important;
    }
    
    .process-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    
    .info-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .page-info {
        text-align: center;
        padding: 8px 16px;
        background: #e9ecef;
        border-radius: 20px;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .model-status {
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
    }
    
    .status-loading {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .status-ready {
        background: #d1edff;
        color: #0c5460;
        border: 1px solid #b8daff;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    """
    
    with gr.Blocks(theme=gr.themes.Soft(), css=css, title="Dots.OCR Demo") as demo:
        
        # Header
        gr.HTML("""
        <div class="header-text">
            <h1>🔍 Dots.OCR Hugging Face Demo</h1>
            <p>Advanced OCR and Document Layout Analysis powered by Hugging Face Transformers</p>
        </div>
        """)
        
        # Model status
        model_status = gr.HTML(
            '<div class="model-status status-loading">🔄 Initializing model...</div>',
            elem_id="model_status"
        )
        
        # Main interface
        with gr.Row():
            # Left column - Input and controls
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Input")
                
                # File input
                file_input = gr.File(
                    label="Upload Image or PDF",
                    file_types=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".pdf"],
                    type="filepath"
                )
                
                # Image preview
                image_preview = gr.Image(
                    label="Preview",
                    type="pil",
                    interactive=False,
                    height=300
                )
                
                # Page navigation for PDFs
                with gr.Row():
                    prev_page_btn = gr.Button("◀ Previous", size="sm")
                    page_info = gr.HTML('<div class="page-info">No file loaded</div>')
                    next_page_btn = gr.Button("Next ▶", size="sm")
                
                gr.Markdown("### ⚙️ Settings")
                                
                # Advanced settings
                with gr.Accordion("Advanced Settings", open=False):
                    max_new_tokens = gr.Slider(
                        minimum=1000,
                        maximum=32000,
                        value=24000,
                        step=1000,
                        label="Max New Tokens",
                        info="Maximum number of tokens to generate"
                    )
                    
                    min_pixels = gr.Number(
                        value=MIN_PIXELS,
                        label="Min Pixels",
                        info="Minimum image resolution"
                    )
                    
                    max_pixels = gr.Number(
                        value=MAX_PIXELS,
                        label="Max Pixels", 
                        info="Maximum image resolution"
                    )
                
                # Process button
                process_btn = gr.Button(
                    "🚀 Process Document",
                    variant="primary",
                    elem_classes=["process-button"],
                    size="lg"
                )
                
                # Clear button
                clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
            
            # Right column - Results
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")
                
                # Results tabs
                with gr.Tabs():
                    # Processed image tab
                    with gr.Tab("🖼️ Processed Image"):
                        processed_image = gr.Image(
                            label="Image with Layout Detection",
                            type="pil",
                            interactive=False,
                            height=500
                        )
                    
                    # Markdown output tab  
                    with gr.Tab("📝 Extracted Content"):
                        markdown_output = gr.Markdown(
                            value="Click 'Process Document' to see extracted content...",
                            height=500
                        )
                    
                    # Raw output tab
                    with gr.Tab("🔧 Raw Output"):
                        raw_output = gr.Textbox(
                            label="Raw Model Output",
                            lines=20,
                            max_lines=30,
                            interactive=False
                        )
                    
                    # JSON layout tab
                    with gr.Tab("📋 Layout JSON"):
                        json_output = gr.JSON(
                            label="Layout Analysis Results",
                            value=None
                        )
        
        # Event handlers
        def load_model_on_startup():
            """Load model when the interface starts"""
            try:
                # Model is already loaded at script level
                return '<div class="model-status status-ready">✅ Model loaded successfully!</div>'
            except Exception as e:
                return f'<div class="model-status status-error">❌ Error: {str(e)}</div>'
        
        def process_document(file_path, max_tokens, min_pix, max_pix):
            """Process the uploaded document"""
            global pdf_cache
            
            try:
                if not file_path:
                    return (
                        None, 
                        "Please upload a file first.", 
                        "No file uploaded",
                        None,
                        '<div class="model-status status-error">❌ No file uploaded</div>'
                    )
                
                if model is None:
                    return (
                        None,
                        "Model not loaded. Please refresh the page and try again.",
                        "Model not loaded",
                        None,
                        '<div class="model-status status-error">❌ Model not loaded</div>'
                    )
                
                # Load and preview file
                image, page_info = load_file_for_preview(file_path)
                if image is None:
                    return (
                        None,
                        page_info,
                        "Failed to load file",
                        None,
                        '<div class="model-status status-error">❌ Failed to load file</div>'
                    )
                
                # Process the image(s)
                if pdf_cache["file_type"] == "pdf":
                    # Process all pages for PDF
                    all_results = []
                    all_markdown = []
                    
                    for i, img in enumerate(pdf_cache["images"]):
                        result = process_image(
                            img, 
                            min_pixels=int(min_pix) if min_pix else None,
                            max_pixels=int(max_pix) if max_pix else None
                        )
                        all_results.append(result)
                        if result.get('markdown_content'):
                            all_markdown.append(f"## Page {i+1}\n\n{result['markdown_content']}")
                    
                    pdf_cache["results"] = all_results
                    pdf_cache["is_parsed"] = True
                    
                    # Show results for first page
                    first_result = all_results[0]
                    combined_markdown = "\n\n---\n\n".join(all_markdown)
                    
                    return (
                        first_result['processed_image'],
                        combined_markdown,
                        first_result['raw_output'],
                        first_result['layout_result'],
                        '<div class="model-status status-ready">✅ Processing completed!</div>'
                    )
                else:
                    # Process single image
                    result = process_image(
                        image,
                        min_pixels=int(min_pix) if min_pix else None,
                        max_pixels=int(max_pix) if max_pix else None
                    )
                    
                    pdf_cache["results"] = [result]
                    pdf_cache["is_parsed"] = True
                    
                    return (
                        result['processed_image'],
                        result['markdown_content'] or "No content extracted",
                        result['raw_output'],
                        result['layout_result'],
                        '<div class="model-status status-ready">✅ Processing completed!</div>'
                    )
                    
            except Exception as e:
                error_msg = f"Error processing document: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                return (
                    None,
                    error_msg,
                    error_msg, 
                    None,
                    f'<div class="model-status status-error">❌ {error_msg}</div>'
                )
        
        def handle_file_upload(file_path):
            """Handle file upload and show preview"""
            if not file_path:
                return None, "No file loaded"
            
            image, page_info = load_file_for_preview(file_path)
            return image, page_info
        
        def handle_page_turn(direction):
            """Handle page navigation"""
            image, page_info, result = turn_page(direction)
            return image, page_info, result
        
        def clear_all():
            """Clear all data and reset interface"""
            global pdf_cache, processing_results
            
            pdf_cache = {
                "images": [],
                "current_page": 0, 
                "total_pages": 0,
                "file_type": None,
                "is_parsed": False,
                "results": []
            }
            processing_results = {
                'original_image': None,
                'processed_image': None,
                'layout_result': None,
                'markdown_content': None,
                'raw_output': None,
            }
            
            return (
                None,  # file_input
                None,  # image_preview
                "No file loaded",  # page_info
                None,  # processed_image
                "Click 'Process Document' to see extracted content...",  # markdown_output
                "",  # raw_output
                None,  # json_output
                '<div class="model-status status-ready">✅ Interface cleared</div>'  # model_status
            )
        
        # Wire up event handlers
        demo.load(load_model_on_startup, outputs=[model_status])
        
        file_input.change(
            handle_file_upload,
            inputs=[file_input],
            outputs=[image_preview, page_info]
        )
        
        prev_page_btn.click(
            lambda: handle_page_turn("prev"),
            outputs=[image_preview, page_info, markdown_output]
        )
        
        next_page_btn.click(
            lambda: handle_page_turn("next"), 
            outputs=[image_preview, page_info, markdown_output]
        )
        
        process_btn.click(
            process_document,
            inputs=[file_input, max_new_tokens, min_pixels, max_pixels],
            outputs=[processed_image, markdown_output, raw_output, json_output, model_status]
        )
        
        clear_btn.click(
            clear_all,
            outputs=[
                file_input, image_preview, page_info, processed_image, 
                markdown_output, raw_output, json_output, model_status
            ]
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
