/**
 * Structured Parser for Form Extraction
 * Handles parsing and validation of structured OCR output
 */

export type FieldType = 'text' | 'date' | 'number' | 'checkbox' | 'unknown'

export interface ExtractedField {
  label: string
  value: string
  type: FieldType
  confidence?: number
}

export interface ExtractedSection {
  name: string | null
  fields: ExtractedField[]
}

export interface ExtractedTable {
  headers: string[]
  rows: string[][]
}

export interface ExtractedCheckbox {
  label: string
  checked: boolean
}

export interface StructuredExtraction {
  form_title?: string | null
  sections: ExtractedSection[]
  tables: ExtractedTable[]
  checkboxes: ExtractedCheckbox[]
  raw_text: string
}

export interface ParseResult {
  success: boolean
  data: StructuredExtraction | null
  error?: string
}

/**
 * Try to parse JSON from text that might contain markdown code blocks
 */
function extractJsonFromText(text: string): string | null {
  if (!text || !text.trim()) return null
  
  let cleaned = text.trim()
  
  // Remove markdown code blocks
  if (cleaned.startsWith('```json')) {
    cleaned = cleaned.slice(7)
  } else if (cleaned.startsWith('```')) {
    cleaned = cleaned.slice(3)
  }
  if (cleaned.endsWith('```')) {
    cleaned = cleaned.slice(0, -3)
  }
  cleaned = cleaned.trim()
  
  // Find JSON boundaries
  const startIdx = cleaned.indexOf('{')
  const endIdx = cleaned.lastIndexOf('}')
  
  if (startIdx === -1 || endIdx === -1 || startIdx >= endIdx) {
    return null
  }
  
  return cleaned.slice(startIdx, endIdx + 1)
}

/**
 * Infer field type from label and value
 */
function inferFieldType(label: string, value: string): FieldType {
  const labelLower = label.toLowerCase()
  const valueLower = value.toLowerCase()
  
  // Date patterns
  if (/تاريخ|date|التاريخ|الميلاد|birth/i.test(label)) {
    return 'date'
  }
  if (/\d{1,4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,4}/.test(value)) {
    return 'date'
  }
  
  // Number patterns  
  if (/رقم|number|هاتف|phone|جوال|mobile|هوية|id|سجل/i.test(label)) {
    return 'number'
  }
  if (/^\d+$/.test(value.replace(/[\s\-]/g, ''))) {
    return 'number'
  }
  
  // Checkbox patterns
  if (/نعم|لا|yes|no|☑|☐|checked|unchecked/i.test(valueLower)) {
    return 'checkbox'
  }
  
  return 'text'
}

/**
 * Parse [FIELD]/[VALUE] format from VLM output
 * Handles multiple line formats:
 * - [FIELD] label \n [VALUE] value
 * - [FIELD] label \n [VALUE] \n value (value on separate line!)
 * - FIELD: label \n VALUE: value
 */
function parseFieldValueFormat(text: string): { 
  fields: ExtractedField[], 
  sections: string[], 
  checkboxes: ExtractedCheckbox[], 
  formTitle: string | null 
} {
  const fields: ExtractedField[] = []
  const checkboxes: ExtractedCheckbox[] = []
  const sections: string[] = []
  let formTitle: string | null = null
  let currentSection: string | null = null
  
  const lines = text.split('\n').map(l => l.trim()).filter(l => l)
  
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    const upperLine = line.toUpperCase()
    
    // Handle [FIELD] format
    if (line.startsWith('[FIELD]') || upperLine.startsWith('[FIELD]')) {
      let fieldLabel = line.slice(7).trim()
      let value = ''
      
      // Look ahead for [VALUE]
      if (i + 1 < lines.length) {
        const nextLine = lines[i + 1].trim()
        const nextUpper = nextLine.toUpperCase()
        
        if (nextLine.startsWith('[VALUE]') || nextUpper.startsWith('[VALUE]')) {
          // Get value after [VALUE] tag
          value = nextLine.slice(7).trim()
          i++
          
          // CRITICAL FIX: If [VALUE] line is empty or just has the tag,
          // the actual value might be on the NEXT line
          if (!value || value === '-' || value.toUpperCase() === 'EMPTY') {
            if (i + 1 < lines.length) {
              const potentialValue = lines[i + 1].trim()
              // Check if next line is NOT a new field/section marker
              if (!potentialValue.startsWith('[FIELD]') && 
                  !potentialValue.toUpperCase().startsWith('[FIELD]') &&
                  !potentialValue.startsWith('[SECTION]') &&
                  !potentialValue.startsWith('[VALUE]') &&
                  !potentialValue.toUpperCase().startsWith('FIELD:') &&
                  potentialValue.length > 0) {
                value = potentialValue
                i++
              }
            }
          }
        }
      }
      
      // Also check if label contains [VALUE] inline (e.g., "[FIELD] label [VALUE] value")
      if (fieldLabel.includes('[VALUE]') || fieldLabel.toUpperCase().includes('[VALUE]')) {
        const parts = fieldLabel.split(/\[VALUE\]/i)
        fieldLabel = parts[0].trim()
        if (parts[1]) {
          value = parts[1].trim()
        }
      }
      
      if (fieldLabel && fieldLabel.length > 1) {
        if (value === '-') value = ''
        
        fields.push({
          label: fieldLabel,
          value: value,
          type: inferFieldType(fieldLabel, value)
        })
      }
      
      i++
      continue
    }
    
    // Handle FIELD: format (legacy)
    if (upperLine.startsWith('FIELD:')) {
      let fieldLabel = line.slice(6).trim()
      let value = ''
      
      if (i + 1 < lines.length) {
        const nextLine = lines[i + 1].trim()
        if (nextLine.toUpperCase().startsWith('VALUE:')) {
          value = nextLine.slice(6).trim()
          i++
          
          // Check for value on next line
          if (!value && i + 1 < lines.length) {
            const potentialValue = lines[i + 1].trim()
            if (!potentialValue.toUpperCase().startsWith('FIELD:') &&
                !potentialValue.toUpperCase().startsWith('SECTION:')) {
              value = potentialValue
              i++
            }
          }
        }
      }
      
      if (fieldLabel && fieldLabel.length > 1) {
        if (value === '-' || value.toUpperCase() === 'EMPTY') value = ''
        
        fields.push({
          label: fieldLabel,
          value: value,
          type: inferFieldType(fieldLabel, value)
        })
      }
      
      i++
      continue
    }
    
    // Handle [SECTION] format
    if (line.startsWith('[SECTION]') || upperLine.startsWith('[SECTION]') || 
        upperLine.startsWith('SECTION:')) {
      const marker = line.startsWith('[SECTION]') ? 9 : 8
      currentSection = line.slice(marker).trim()
      if (currentSection && !sections.includes(currentSection)) {
        sections.push(currentSection)
      }
      i++
      continue
    }
    
    // Handle inline "Label: Value" format (common in Arabic forms)
    if (line.includes(':') && !line.startsWith('[') && !line.startsWith('http')) {
      const colonIdx = line.indexOf(':')
      const label = line.slice(0, colonIdx).trim()
      const value = line.slice(colonIdx + 1).trim()
      
      // Validate - label should be reasonably short
      if (label && label.length > 1 && label.length < 80 && value) {
        fields.push({
          label: label,
          value: value === '-' ? '' : value,
          type: inferFieldType(label, value)
        })
      }
    }
    
    i++
  }
  
  return { fields, sections, checkboxes, formTitle }
}

/**
 * Parse simple key-value patterns as fallback
 */
function parseFallbackKeyValue(text: string): ExtractedField[] {
  const fields: ExtractedField[] = []
  const lines = text.split('\n')
  
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#') || trimmed.startsWith('//')) continue
    
    // Look for colon-separated patterns
    const colonMatch = trimmed.match(/^([^:]{2,50}):\s*(.+)$/)
    if (colonMatch) {
      const label = colonMatch[1].trim()
      const value = colonMatch[2].trim()
      
      if (label && value && value !== '-') {
        fields.push({
          label,
          value,
          type: inferFieldType(label, value)
        })
      }
    }
  }
  
  return fields
}

/**
 * Main parser function - attempts multiple parsing strategies
 */
export function parseStructuredOutput(rawText: string): ParseResult {
  if (!rawText || !rawText.trim()) {
    return { success: false, data: null, error: 'Empty input' }
  }
  
  const text = rawText.trim()
  
  // Strategy 1: Try [FIELD]/[VALUE] format first (our structured format)
  const fieldValueResult = parseFieldValueFormat(text)
  if (fieldValueResult.fields.length > 0) {
    // Group by sections if available
    const sections: ExtractedSection[] = fieldValueResult.sections.length > 0
      ? fieldValueResult.sections.map(name => ({
          name,
          fields: fieldValueResult.fields.filter((_, idx) => idx < fieldValueResult.fields.length)
        }))
      : [{ name: null, fields: fieldValueResult.fields }]
    
    return {
      success: true,
      data: {
        form_title: fieldValueResult.formTitle,
        sections,
        tables: [],
        checkboxes: fieldValueResult.checkboxes,
        raw_text: rawText
      }
    }
  }
  
  // Strategy 2: Try JSON parsing
  const jsonStr = extractJsonFromText(text)
  if (jsonStr) {
    try {
      const parsed = JSON.parse(jsonStr)
      if (parsed.sections && Array.isArray(parsed.sections)) {
        return {
          success: true,
          data: {
            form_title: parsed.form_title || null,
            sections: parsed.sections,
            tables: parsed.tables || [],
            checkboxes: parsed.checkboxes || [],
            raw_text: rawText
          }
        }
      }
    } catch {
      // JSON parsing failed, continue to fallback
    }
  }
  
  // Strategy 3: Fallback key-value parsing
  const fallbackFields = parseFallbackKeyValue(text)
  if (fallbackFields.length > 0) {
    return {
      success: true,
      data: {
        form_title: null,
        sections: [{ name: null, fields: fallbackFields }],
        tables: [],
        checkboxes: [],
        raw_text: rawText
      }
    }
  }
  
  // No structured data found
  return {
    success: false,
    data: {
      form_title: null,
      sections: [],
      tables: [],
      checkboxes: [],
      raw_text: rawText
    },
    error: 'Could not parse structured data'
  }
}

/**
 * Convert structured data to plain text
 */
export function structuredToPlainText(data: StructuredExtraction): string {
  const lines: string[] = []
  
  if (data.form_title) {
    lines.push(data.form_title)
    lines.push('='.repeat(40))
    lines.push('')
  }
  
  for (const section of data.sections) {
    if (section.name) {
      lines.push(`\n--- ${section.name} ---\n`)
    }
    
    for (const field of section.fields) {
      if (field.value) {
        lines.push(`${field.label}: ${field.value}`)
      }
    }
  }
  
  for (const cb of data.checkboxes) {
    lines.push(`[${cb.checked ? '✓' : ' '}] ${cb.label}`)
  }
  
  return lines.join('\n')
}

/**
 * Convert structured data to CSV
 */
export function structuredToCSV(data: StructuredExtraction): string {
  const rows: string[] = ['Field,Value,Type,Section']
  
  for (const section of data.sections) {
    for (const field of section.fields) {
      const escapedLabel = `"${field.label.replace(/"/g, '""')}"`
      const escapedValue = `"${field.value.replace(/"/g, '""')}"`
      const sectionName = section.name || 'General'
      rows.push(`${escapedLabel},${escapedValue},${field.type},"${sectionName}"`)
    }
  }
  
  return rows.join('\n')
}

/**
 * Convert structured data to JSON
 */
export function structuredToJSON(data: StructuredExtraction): string {
  return JSON.stringify(data, null, 2)
}

/**
 * Merge extracted data with a template
 */
export function mergeWithTemplate(
  extracted: StructuredExtraction,
  template: { sections?: Array<{ name: string; fields: Array<{ label: string; type?: string }> }> }
): StructuredExtraction {
  if (!template.sections) return extracted
  
  const mergedSections: ExtractedSection[] = template.sections.map(templateSection => {
    const fields: ExtractedField[] = templateSection.fields.map(templateField => {
      // Find matching extracted field
      const match = extracted.sections
        .flatMap(s => s.fields)
        .find(f => 
          f.label.toLowerCase().includes(templateField.label.toLowerCase()) ||
          templateField.label.toLowerCase().includes(f.label.toLowerCase())
        )
      
      return {
        label: templateField.label,
        value: match?.value || '',
        type: (templateField.type as FieldType) || match?.type || 'text'
      }
    })
    
    return {
      name: templateSection.name,
      fields
    }
  })
  
  return {
    ...extracted,
    sections: mergedSections
  }
}
