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
 * Common garbage patterns to filter out
 */
const GARBAGE_PATTERNS = [
  "here's the extracted data",
  "here is the extracted",
  "i'll extract",
  "let me extract",
  "the form contains",
  "based on the image",
  "from the form",
  "extracted data:",
  "form data:",
]

/**
 * Parse [FIELD]/[VALUE] or FIELD:/VALUE: format from VLM output
 */
function parseFieldValueFormat(text: string): { fields: ExtractedField[], sections: string[], checkboxes: ExtractedCheckbox[], formTitle: string | null } {
  const fields: ExtractedField[] = []
  const checkboxes: ExtractedCheckbox[] = []
  const sections: string[] = []
  let formTitle: string | null = null
  let currentSection: string | null = null
  
  // Filter out garbage intro lines
  let lines = text.split('\n')
  let started = false
  const filteredLines: string[] = []
  
  for (const line of lines) {
    const lineLower = line.toLowerCase().trim()
    
    if (!started) {
      const isGarbage = GARBAGE_PATTERNS.some(pat => lineLower.includes(pat))
      // Start when we see [FIELD] or FIELD: or Arabic text with colon
      if (lineLower.includes('[field]') || lineLower.startsWith('field:') || (/[\u0600-\u06FF]/.test(line) && line.includes(':'))) {
        started = true
        filteredLines.push(line)
      } else if (!isGarbage && line.trim()) {
        // Check if line looks like a field-value pair
        if (line.includes(':') && line.length > 3) {
          started = true
          filteredLines.push(line)
        }
      }
    } else {
      filteredLines.push(line)
    }
  }
  
  lines = filteredLines
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim()
    if (!line) continue
    
    const upperLine = line.toUpperCase()
    
    // NEW FORMAT: [FIELD] label / [VALUE] value
    if (line.startsWith('[FIELD]') || upperLine.startsWith('[FIELD]')) {
      const fieldLabel = line.slice(7).trim()
      let value = ''
      
      if (i + 1 < lines.length) {
        const nextLine = lines[i + 1].trim()
        if (nextLine.startsWith('[VALUE]') || nextLine.toUpperCase().startsWith('[VALUE]')) {
          value = nextLine.slice(7).trim()
          if (value === '-' || value.toUpperCase() === 'EMPTY') {
            value = ''
          }
          i++
        }
      }
      
      // Filter garbage labels
      if (fieldLabel && fieldLabel.length > 1 && !GARBAGE_PATTERNS.some(pat => fieldLabel.toLowerCase().includes(pat))) {
        fields.push({
          label: fieldLabel,
          value: value,
          type: inferFieldType(fieldLabel, value)
        })
      }
      continue
    }
    
    // Check for section header
    if (upperLine.startsWith('SECTION:') || upperLine.startsWith('[SECTION]')) {
      const marker = upperLine.startsWith('[SECTION]') ? '[SECTION]' : 'SECTION:'
      currentSection = line.slice(marker.length).trim()
      if (currentSection && !sections.includes(currentSection)) {
        sections.push(currentSection)
      }
      continue
    }
    
    // Check for form title
    if (upperLine.startsWith('TITLE:') || upperLine.startsWith('FORM_TITLE:')) {
      formTitle = line.split(':').slice(1).join(':').trim()
      continue
    }
    
    // Legacy FIELD:/VALUE: format
    if (upperLine.startsWith('FIELD:')) {
      const fieldLabel = line.slice(6).trim()
      let value = ''
      
      if (i + 1 < lines.length) {
        const nextLine = lines[i + 1].trim()
        if (nextLine.toUpperCase().startsWith('VALUE:')) {
          value = nextLine.slice(6).trim()
          if (value.toUpperCase() === 'EMPTY' || value === '-') {
            value = ''
          }
          i++
        }
      }
      
      const upperValue = value.toUpperCase()
      if (['CHECKED', 'UNCHECKED', '☑', '☐', 'نعم', 'لا'].includes(upperValue)) {
        checkboxes.push({
          label: fieldLabel,
          checked: ['CHECKED', '☑', 'نعم'].includes(upperValue)
        })
      } else if (fieldLabel && fieldLabel.length > 1) {
        fields.push({
          label: fieldLabel,
          value: value,
          type: inferFieldType(fieldLabel, value)
        })
      }
      continue
    }
  }
  
  return { fields, sections, checkboxes, formTitle }
}

/**
 * Infer field type from label and value
 */
function inferFieldType(label: string, value: string): FieldType {
  const labelLower = label.toLowerCase()
  const arabicLabel = label
  
  // Date patterns
  const dateKeywords = ['date', 'تاريخ', 'يوم', 'شهر', 'سنة', 'الميلاد', 'التسجيل', 'الإصدار', 'الانتهاء']
  if (dateKeywords.some(kw => labelLower.includes(kw) || arabicLabel.includes(kw))) {
    return 'date'
  }
  
  // Number patterns
  const numberKeywords = ['number', 'رقم', 'هاتف', 'جوال', 'هوية', 'جواز', 'عدد', 'كمية', 'سعر', 'مبلغ']
  if (numberKeywords.some(kw => labelLower.includes(kw) || arabicLabel.includes(kw))) {
    return 'number'
  }
  
  // Check if value looks like a number
  if (/^[\d٠-٩\s\-\+\.]+$/.test(value.replace(/\s/g, ''))) {
    return 'number'
  }
  
  // Check if value looks like a date
  if (/^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}$/.test(value.trim())) {
    return 'date'
  }
  
  return 'text'
}

/**
 * Parse raw text output from VLM into structured extraction
 */
export function parseStructuredOutput(rawText: string): ParseResult {
  if (!rawText || !rawText.trim()) {
    return {
      success: false,
      data: null,
      error: 'Empty input text'
    }
  }
  
  // First try FIELD:/VALUE: format (our preferred format)
  if (rawText.toUpperCase().includes('FIELD:')) {
    const parsed = parseFieldValueFormat(rawText)
    
    if (parsed.fields.length > 0 || parsed.checkboxes.length > 0) {
      // Group fields by section (if any)
      const sectionsMap = new Map<string, ExtractedField[]>()
      
      for (const field of parsed.fields) {
        const sectionName = 'General' // All in general for now from FIELD/VALUE format
        if (!sectionsMap.has(sectionName)) {
          sectionsMap.set(sectionName, [])
        }
        sectionsMap.get(sectionName)!.push(field)
      }
      
      const sections: ExtractedSection[] = Array.from(sectionsMap.entries()).map(([name, fields]) => ({
        name: name === 'General' ? null : name,
        fields
      }))
      
      return {
        success: true,
        data: {
          form_title: parsed.formTitle,
          sections,
          tables: [],
          checkboxes: parsed.checkboxes,
          raw_text: rawText
        }
      }
    }
  }
  
  // Try JSON format
  const jsonStr = extractJsonFromText(rawText)
  
  if (jsonStr) {
    try {
      const parsed = JSON.parse(jsonStr)
      
      // Check if it's a valid structured response (not empty skeleton)
      const hasContent = (
        (Array.isArray(parsed.sections) && parsed.sections.some((s: any) => 
          Array.isArray(s.fields) && s.fields.length > 0
        )) ||
        (Array.isArray(parsed.tables) && parsed.tables.length > 0) ||
        (Array.isArray(parsed.checkboxes) && parsed.checkboxes.length > 0)
      )
      
      if (hasContent) {
        // Validate and transform the parsed data
        const sections: ExtractedSection[] = []
        
        if (Array.isArray(parsed.sections)) {
          for (const section of parsed.sections) {
            const fields: ExtractedField[] = []
            
            if (Array.isArray(section.fields)) {
              for (const field of section.fields) {
                const label = String(field.label || '').trim()
                const value = String(field.value || '').trim()
                const type = inferFieldType(label, value)
                
                if (label) {
                  fields.push({ label, value, type })
                }
              }
            }
            
            if (fields.length > 0) {
              sections.push({
                name: section.name || null,
                fields
              })
            }
          }
        }
        
        // Parse tables
        const tables: ExtractedTable[] = []
        if (Array.isArray(parsed.tables)) {
          for (const table of parsed.tables) {
            if (Array.isArray(table.headers) && Array.isArray(table.rows)) {
              tables.push({
                headers: table.headers.map((h: any) => String(h || '')),
                rows: table.rows.map((row: any) => 
                  Array.isArray(row) ? row.map((cell: any) => String(cell || '')) : []
                )
              })
            }
          }
        }
        
        // Parse checkboxes
        const checkboxes: ExtractedCheckbox[] = []
        if (Array.isArray(parsed.checkboxes)) {
          for (const cb of parsed.checkboxes) {
            const label = String(cb.label || '').trim()
            if (label) {
              checkboxes.push({
                label,
                checked: Boolean(cb.checked)
              })
            }
          }
        }
        
        if (sections.length > 0 || tables.length > 0 || checkboxes.length > 0) {
          return {
            success: true,
            data: {
              form_title: parsed.form_title || null,
              sections,
              tables,
              checkboxes,
              raw_text: rawText
            }
          }
        }
      }
    } catch (e) {
      console.error('JSON parse error:', e)
    }
  }
  
  // Fallback to key-value parsing
  return parseFallbackKeyValue(rawText)
}

/**
 * Fallback parser for when JSON parsing fails
 * Extracts key-value pairs from plain text
 */
function parseFallbackKeyValue(text: string): ParseResult {
  const lines = text.split('\n').map(l => l.trim()).filter(Boolean)
  const fields: ExtractedField[] = []
  
  for (const line of lines) {
    // Try to match "Label: Value" pattern
    const colonMatch = line.match(/^(.+?):\s*(.*)$/)
    if (colonMatch) {
      const label = colonMatch[1].trim()
      const value = colonMatch[2].trim()
      
      // Skip if it looks like a section header (label only, no value, ends with colon in original)
      if (label && (value || !line.endsWith(':'))) {
        const type = inferFieldType(label, value)
        fields.push({ label, value, type })
      }
      continue
    }
    
    // Try to match space-separated pattern "Label    Value"
    const spaceMatch = line.match(/^(\S+)\s{2,}(.+)$/)
    if (spaceMatch) {
      const label = spaceMatch[1].trim()
      const value = spaceMatch[2].trim()
      const type = inferFieldType(label, value)
      fields.push({ label, value, type })
    }
  }
  
  if (fields.length === 0) {
    return {
      success: false,
      data: {
        form_title: null,
        sections: [],
        tables: [],
        checkboxes: [],
        raw_text: text
      },
      error: 'Could not parse structured data from text'
    }
  }
  
  return {
    success: true,
    data: {
      form_title: null,
      sections: [{
        name: null,
        fields
      }],
      tables: [],
      checkboxes: [],
      raw_text: text
    }
  }
}

/**
 * Convert structured extraction to plain text format
 */
export function structuredToPlainText(data: StructuredExtraction): string {
  const lines: string[] = []
  
  if (data.form_title) {
    lines.push(`=== ${data.form_title} ===`)
    lines.push('')
  }
  
  for (const section of data.sections) {
    if (section.name) {
      lines.push(`--- ${section.name} ---`)
    }
    
    for (const field of section.fields) {
      lines.push(`${field.label}: ${field.value || '[empty]'}`)
    }
    
    if (section.fields.length > 0) {
      lines.push('')
    }
  }
  
  for (const checkbox of data.checkboxes) {
    const mark = checkbox.checked ? '☑' : '☐'
    lines.push(`${mark} ${checkbox.label}`)
  }
  
  for (const table of data.tables) {
    if (table.headers.length > 0) {
      lines.push('')
      lines.push(table.headers.join(' | '))
      lines.push(table.headers.map(() => '---').join(' | '))
      for (const row of table.rows) {
        lines.push(row.join(' | '))
      }
    }
  }
  
  return lines.join('\n')
}

/**
 * Convert structured extraction to CSV format
 */
export function structuredToCSV(data: StructuredExtraction): string {
  const rows: string[][] = []
  
  // Header row
  rows.push(['Section', 'Label', 'Value', 'Type'])
  
  for (const section of data.sections) {
    const sectionName = section.name || 'General'
    for (const field of section.fields) {
      rows.push([
        sectionName,
        field.label,
        field.value,
        field.type
      ])
    }
  }
  
  // Add checkboxes
  for (const cb of data.checkboxes) {
    rows.push([
      'Checkboxes',
      cb.label,
      cb.checked ? 'Yes' : 'No',
      'checkbox'
    ])
  }
  
  // Convert to CSV string
  return rows.map(row => 
    row.map(cell => {
      // Escape quotes and wrap in quotes if contains comma or newline
      const escaped = cell.replace(/"/g, '""')
      return /[,\n"]/.test(cell) ? `"${escaped}"` : escaped
    }).join(',')
  ).join('\n')
}

/**
 * Convert structured extraction to JSON string
 */
export function structuredToJSON(data: StructuredExtraction): string {
  const exportData = {
    form_title: data.form_title,
    fields: data.sections.flatMap(s => 
      s.fields.map(f => ({
        section: s.name,
        label: f.label,
        value: f.value,
        type: f.type
      }))
    ),
    tables: data.tables,
    checkboxes: data.checkboxes
  }
  
  return JSON.stringify(exportData, null, 2)
}

/**
 * Merge template field schema with extracted data
 * Fills in expected fields that weren't extracted
 */
export function mergeWithTemplate(
  extracted: StructuredExtraction,
  templateSchema: { sections: Array<{ name: string; fields: Array<{ label: string; type: string }> }> }
): StructuredExtraction {
  const mergedSections: ExtractedSection[] = []
  
  for (const templateSection of templateSchema.sections) {
    const existingSection = extracted.sections.find(s => 
      s.name === templateSection.name || 
      (s.name === null && templateSection.name === '')
    )
    
    const mergedFields: ExtractedField[] = []
    
    for (const templateField of templateSection.fields) {
      const existingField = existingSection?.fields.find(f => 
        f.label === templateField.label ||
        f.label.includes(templateField.label) ||
        templateField.label.includes(f.label)
      )
      
      if (existingField) {
        mergedFields.push(existingField)
      } else {
        // Add placeholder for expected but missing field
        mergedFields.push({
          label: templateField.label,
          value: '',
          type: templateField.type as FieldType
        })
      }
    }
    
    // Add any extra fields that weren't in template
    if (existingSection) {
      for (const field of existingSection.fields) {
        if (!mergedFields.some(f => f.label === field.label)) {
          mergedFields.push(field)
        }
      }
    }
    
    mergedSections.push({
      name: templateSection.name || null,
      fields: mergedFields
    })
  }
  
  // Add sections that weren't in template
  for (const section of extracted.sections) {
    if (!mergedSections.some(s => s.name === section.name)) {
      mergedSections.push(section)
    }
  }
  
  return {
    ...extracted,
    sections: mergedSections
  }
}
