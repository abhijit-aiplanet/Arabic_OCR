"""
Structured Form Extraction Module

This module provides robust extraction of key-value pairs from Arabic forms.
Uses a hybrid approach:
1. Natural language prompt for VLM (better compliance)
2. Intelligent parsing to extract structured data from natural text

The key insight is that VLMs are much better at natural descriptions than strict
output formats. So we ask for natural output and parse it ourselves.
"""

import re
from typing import Dict, List, Any, Optional, Tuple


# =============================================================================
# PROMPTS
# =============================================================================

# Natural extraction prompt - works much better than strict formats
NATURAL_EXTRACTION_PROMPT = """أنت مساعد متخصص في قراءة النماذج العربية. اقرأ هذه الصورة واستخرج جميع المعلومات.

Read this Arabic form image carefully and extract ALL information you can see.

INSTRUCTIONS:
1. List every field label you see and its corresponding value
2. For each field, write: "field_name: value" (use Arabic labels as they appear)
3. If a field is empty or unclear, write: "field_name: [فارغ]"
4. Include ALL visible fields - don't skip any
5. If there are checkboxes, write: "checkbox_name: ☑" or "checkbox_name: ☐"
6. If there are sections or headers, mention them
7. For tables, describe row by row
8. Keep the original Arabic text - don't translate

IMPORTANT: Extract EVERY piece of text you can see. Don't summarize.

Start extracting now:"""


# Alternative prompt for forms with many fields
COMPREHENSIVE_EXTRACTION_PROMPT = """You are an Arabic form reader. Your task is to extract ALL fields and values from this form image.

اقرأ النموذج واستخرج كل المعلومات المكتوبة

Output format for each field:
• field_label: field_value

Rules:
- Extract EVERY field you see, no matter how small
- Write Arabic text exactly as shown
- For empty fields write: field_name: -
- For checkmarks write: [✓] or [  ]
- Include headers/section titles
- Number your fields if the form has numbers

Now read and extract everything:"""


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def extract_fields_from_text(text: str) -> List[Dict[str, str]]:
    """
    Extract field-value pairs from natural VLM output.
    Handles multiple formats:
    - label: value
    - label : value (with spaces)
    - [FIELD] label [VALUE] value
    - • label: value (bullet points)
    - 1. label: value (numbered)
    """
    if not text or not text.strip():
        return []
    
    fields = []
    lines = text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
        
        # Skip common garbage patterns
        garbage_patterns = [
            "here's the", "here is the", "i'll extract", "let me",
            "the form contains", "based on", "extracted data", "form data",
            "instructions", "output format", "rules:", "now read"
        ]
        if any(pat in line.lower() for pat in garbage_patterns):
            i += 1
            continue
        
        # Pattern 1: [FIELD] label / [VALUE] value (handle 3-line format)
        if line.upper().startswith('[FIELD]'):
            label = line[7:].strip()
            value = ""
            
            # Check if [VALUE] is inline
            if '[VALUE]' in label.upper():
                parts = re.split(r'\[VALUE\]', label, flags=re.IGNORECASE)
                label = parts[0].strip()
                value = parts[1].strip() if len(parts) > 1 else ""
            else:
                # Look for [VALUE] on next lines
                j = i + 1
                while j < len(lines) and j <= i + 3:  # Look up to 3 lines ahead
                    next_line = lines[j].strip()
                    
                    if next_line.upper().startswith('[VALUE]'):
                        value_part = next_line[7:].strip()
                        if value_part and value_part != '-':
                            value = value_part
                            i = j
                            break
                        else:
                            # Value might be on the NEXT line
                            if j + 1 < len(lines):
                                potential_value = lines[j + 1].strip()
                                if (potential_value and 
                                    not potential_value.upper().startswith('[FIELD]') and
                                    not potential_value.upper().startswith('[VALUE]')):
                                    value = potential_value
                                    i = j + 1
                                    break
                            i = j
                            break
                    elif next_line.upper().startswith('[FIELD]'):
                        # Next field found, current value is empty
                        break
                    j += 1
            
            if label and len(label) > 1:
                fields.append({
                    "label": clean_label(label),
                    "value": clean_value(value)
                })
            i += 1
            continue
        
        # Pattern 2: Bullet point or numbered list
        list_match = re.match(r'^[•\-\*\d]+[\.\)]\s*(.+)$', line)
        if list_match:
            content = list_match.group(1)
            if ':' in content:
                label, value = split_label_value(content)
                if label:
                    fields.append({
                        "label": clean_label(label),
                        "value": clean_value(value)
                    })
            i += 1
            continue
        
        # Pattern 3: Simple "label: value" format (most common)
        if ':' in line:
            label, value = split_label_value(line)
            if label and is_valid_label(label):
                fields.append({
                    "label": clean_label(label),
                    "value": clean_value(value)
                })
                i += 1
                continue
        
        # Pattern 4: FIELD:/VALUE: format
        if line.upper().startswith('FIELD:'):
            label = line[6:].strip()
            value = ""
            
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.upper().startswith('VALUE:'):
                    value = next_line[6:].strip()
                    if not value and i + 2 < len(lines):
                        potential = lines[i + 2].strip()
                        if not potential.upper().startswith('FIELD:'):
                            value = potential
                            i += 1
                    i += 1
            
            if label and len(label) > 1:
                fields.append({
                    "label": clean_label(label),
                    "value": clean_value(value)
                })
            i += 1
            continue
        
        i += 1
    
    return fields


def split_label_value(text: str) -> Tuple[str, str]:
    """Split a line by colon into label and value."""
    # Find the first colon that's likely a separator (not part of time, etc.)
    # Arabic colon is ، but we mainly use :
    
    colon_idx = -1
    
    # Look for colon followed by space or at a reasonable position
    for idx, char in enumerate(text):
        if char == ':':
            # Check it's not a time pattern like 12:30
            before = text[:idx].strip()
            after = text[idx+1:].strip() if idx + 1 < len(text) else ""
            
            # If before ends with digit and after starts with digit, likely time/ratio
            if before and after and before[-1].isdigit() and after and after[0].isdigit():
                continue
            
            colon_idx = idx
            break
    
    if colon_idx == -1:
        return ("", "")
    
    label = text[:colon_idx].strip()
    value = text[colon_idx + 1:].strip()
    
    return (label, value)


def is_valid_label(label: str) -> bool:
    """Check if a label is valid (not garbage)."""
    if not label or len(label) < 2:
        return False
    
    if len(label) > 100:
        return False
    
    # Reject if mostly numbers
    if label.replace(' ', '').isdigit():
        return False
    
    # Reject URLs
    if label.startswith('http') or label.startswith('www'):
        return False
    
    # Reject common non-label patterns
    reject_patterns = [
        'here', 'extracted', 'form data', 'instructions',
        'output', 'rules', 'now read', 'example'
    ]
    if any(pat in label.lower() for pat in reject_patterns):
        return False
    
    return True


def clean_label(label: str) -> str:
    """Clean up a field label."""
    # Remove leading/trailing special characters
    label = re.sub(r'^[\-•\*\d\.\)\]]+\s*', '', label)
    label = re.sub(r'\s*[\:\-•\*]+$', '', label)
    return label.strip()


def clean_value(value: str) -> str:
    """Clean up a field value."""
    if not value:
        return ""
    
    # Normalize empty indicators
    empty_indicators = ['-', '—', '[فارغ]', '[empty]', 'empty', 'فارغ', 'n/a', 'none']
    if value.lower().strip() in empty_indicators:
        return ""
    
    return value.strip()


def infer_field_type(label: str, value: str) -> str:
    """Infer field type from label and value."""
    label_lower = label.lower()
    
    # Date patterns
    date_keywords = ['date', 'تاريخ', 'يوم', 'شهر', 'سنة', 'الميلاد', 'التسجيل', 'الإصدار', 'الانتهاء', 'birth']
    if any(kw in label_lower or kw in label for kw in date_keywords):
        return 'date'
    
    # Check if value looks like a date
    if value and re.match(r'^\d{1,4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,4}', value.strip()):
        return 'date'
    
    # Number patterns
    number_keywords = ['number', 'رقم', 'هاتف', 'جوال', 'هوية', 'جواز', 'عدد', 'كمية', 'سعر', 'مبلغ', 'id', 'phone']
    if any(kw in label_lower or kw in label for kw in number_keywords):
        return 'number'
    
    # Check if value looks like a number
    if value and re.match(r'^[\d\s\-\+\.٠-٩]+$', value.replace(' ', '')):
        return 'number'
    
    # Checkbox patterns
    if value in ['☑', '☐', '✓', '✗', 'نعم', 'لا', 'yes', 'no', 'checked', 'unchecked']:
        return 'checkbox'
    
    return 'text'


def parse_to_structured_format(
    raw_text: str,
    include_raw: bool = True
) -> Dict[str, Any]:
    """
    Parse VLM output into structured format with sections.
    
    Returns:
        {
            "form_title": str | None,
            "sections": [{"name": str | None, "fields": [...]}],
            "tables": [],
            "checkboxes": [],
            "raw_text": str
        }
    """
    fields = extract_fields_from_text(raw_text)
    
    # Separate checkboxes from regular fields
    regular_fields = []
    checkboxes = []
    
    for field in fields:
        field_type = infer_field_type(field["label"], field["value"])
        
        if field_type == 'checkbox' or field["value"] in ['☑', '☐', '✓', '✗']:
            checkboxes.append({
                "label": field["label"],
                "checked": field["value"] in ['☑', '✓', 'نعم', 'yes', 'checked']
            })
        else:
            regular_fields.append({
                "label": field["label"],
                "value": field["value"],
                "type": field_type
            })
    
    result = {
        "form_title": None,
        "sections": [{"name": None, "fields": regular_fields}] if regular_fields else [],
        "tables": [],
        "checkboxes": checkboxes
    }
    
    if include_raw:
        result["raw_text"] = raw_text
    
    return result


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def get_extraction_prompt(template: Optional[Dict] = None) -> str:
    """
    Get the best extraction prompt, optionally enhanced with template info.
    """
    base_prompt = NATURAL_EXTRACTION_PROMPT
    
    if not template:
        return base_prompt
    
    # If template has expected fields, add them as hints
    field_schema = template.get("sections", {})
    if not field_schema:
        return base_prompt
    
    expected_fields = []
    sections_info = field_schema.get("sections", [])
    
    for section in sections_info:
        section_name = section.get("name", "")
        fields = section.get("fields", [])
        
        if section_name:
            expected_fields.append(f"\n[{section_name}]")
        
        for field in fields:
            label = field.get("label", "")
            if label:
                expected_fields.append(f"  - {label}")
    
    if not expected_fields:
        return base_prompt
    
    enhanced_prompt = f"""{base_prompt}

EXPECTED FIELDS (look for these specifically):
{chr(10).join(expected_fields)}

Extract these and any other fields you find:"""
    
    return enhanced_prompt
