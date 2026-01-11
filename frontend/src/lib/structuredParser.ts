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
 * Common garbage patterns to filter out from VLM output
 */
const GARBAGE_PATTERNS = [
  "here's the extracted", "here is the extracted", "i'll extract",
  "let me extract", "the form contains", "based on the image",
  "from the form", "extracted data:", "form data:", "your task",
  "begin extraction", "extract everything", "output format",
  "instructions:", "important:", "rules:", "critical",
  "field patterns", "start extracting"
]

/**
 * Normalize empty field values
 */
function normalizeEmptyValue(value: string): string {
  if (!value) return ""
  
  value = value.trim()
  
  // Common empty patterns in Arabic forms
  const emptyPatterns = [
    '-', '—', '−',
    '............', '...........', '..........', '.........', '........', '.......', '......', '.....', '....', '...',
    '____', '___', '__',
    '/ / ١٤هـ', '/ / ١٤', '/ /',
    '[فارغ]', '[empty]', 'فارغ', 'empty', 'n/a', 'N/A', 'none', 'None',
  ]
  
  if (emptyPatterns.includes(value)) return ""
  
  // Value is mostly dots or dashes
  if (/^[\.\-_\s/]+$/.test(value)) return ""
  
  // Just a date placeholder
  if (/^[\s/]*١٤هـ?[\s/]*$/.test(value)) return ""
  
  return value
}

/**
 * Parse natural VLM output into structured fields.
 * Handles multiple formats:
 * - Natural "label: value" format (primary)
 * - [FIELD]/[VALUE] format with multi-line support
 * - FIELD:/VALUE: format (legacy)
 * - Bullet/numbered lists
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
  
  // Filter garbage intro lines
  let rawLines = text.split('\n')
  let started = false
  const filteredLines: string[] = []
  
  for (const line of rawLines) {
    const lineLower = line.toLowerCase().trim()
    
    if (!started) {
      const isGarbage = GARBAGE_PATTERNS.some(pat => lineLower.includes(pat))
      const hasArabic = /[\u0600-\u06FF]/.test(line)
      const hasColon = line.includes(':')
      const hasFieldMarker = lineLower.includes('[field]') || lineLower.startsWith('field:')
      const hasBullet = /^[•\-\*١٢٣٤٥٦٧٨٩٠\d]/.test(line.trim())
      
      if (hasFieldMarker || (hasArabic && hasColon) || hasBullet) {
        started = true
        filteredLines.push(line)
      } else if (!isGarbage && line.trim() && (hasColon || hasBullet)) {
        started = true
        filteredLines.push(line)
      }
    } else {
      filteredLines.push(line)
    }
  }
  
  const lines = filteredLines.map(l => l.trim())
  
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (!line) {
      i++
      continue
    }
    
    const upperLine = line.toUpperCase()
    
    // Pattern 0: Arabic section headers [قسم] or "بيانات ..."
    const sectionMatch = line.match(/^\[قسم\]\s*(.+)$/)
    if (sectionMatch) {
      currentSection = sectionMatch[1].trim()
      if (currentSection && !sections.includes(currentSection)) {
        sections.push(currentSection)
      }
      i++
      continue
    }
    
    // Detect section headers like "بيانات عامة" or "بيانات المركبة"
    if (/^بيانات\s+[\u0600-\u06FF]+/.test(line) && !line.includes(':')) {
      currentSection = line.trim()
      if (currentSection && !sections.includes(currentSection)) {
        sections.push(currentSection)
      }
      i++
      continue
    }
    
    // Detect checklist/requirements sections: "طلبات الإصدار الجديد :" or "طلبات التجديد :"
    const checklistMatch = line.match(/^(طلبات|متطلبات|شروط)\s+[\u0600-\u06FF\s]+\s*:?\s*$/)
    if (checklistMatch) {
      currentSection = line.replace(':', '').trim()
      if (currentSection && !sections.includes(currentSection)) {
        sections.push(currentSection)
      }
      i++
      continue
    }
    
    // Detect [قائمة] list marker
    const listMarkerMatch = line.match(/^\[قائمة\]\s*(.+)$/)
    if (listMarkerMatch) {
      currentSection = listMarkerMatch[1].trim()
      if (currentSection && !sections.includes(currentSection)) {
        sections.push(currentSection)
      }
      i++
      continue
    }
    
    // Pattern 1: [FIELD] with multi-line value support
    if (line.startsWith('[FIELD]') || upperLine.startsWith('[FIELD]')) {
      let fieldLabel = line.slice(7).trim()
      let value = ''
      
      // Check for inline [VALUE]
      if (fieldLabel.toUpperCase().includes('[VALUE]')) {
        const parts = fieldLabel.split(/\[VALUE\]/i)
        fieldLabel = parts[0].trim()
        value = parts[1]?.trim() || ''
      } else {
        // Look for [VALUE] on next lines (up to 3 lines ahead)
        let j = i + 1
        while (j < lines.length && j <= i + 3) {
          const nextLine = lines[j]
          
          if (nextLine.toUpperCase().startsWith('[VALUE]')) {
            const valueContent = nextLine.slice(7).trim()
            
            if (valueContent && valueContent !== '-') {
              value = valueContent
              i = j
              break
            } else {
              // CRITICAL: Check the line AFTER [VALUE] for the actual value
              if (j + 1 < lines.length) {
                const potential = lines[j + 1]
                if (potential && 
                    !potential.toUpperCase().startsWith('[FIELD]') &&
                    !potential.toUpperCase().startsWith('[VALUE]') &&
                    !potential.toUpperCase().startsWith('FIELD:') &&
                    !potential.toUpperCase().startsWith('[SECTION]')) {
                  value = potential
                  i = j + 1
                  break
                }
              }
              i = j
              break
            }
          } else if (nextLine.toUpperCase().startsWith('[FIELD]')) {
            break
          }
          j++
        }
      }
      
      if (fieldLabel && fieldLabel.length > 1) {
        const cleanLabel = fieldLabel.replace(/^[•\-\*\d\.\)]+\s*/, '').trim()
        if (!GARBAGE_PATTERNS.some(pat => cleanLabel.toLowerCase().includes(pat))) {
          fields.push({
            label: cleanLabel,
            value: value === '-' ? '' : value,
            type: inferFieldType(cleanLabel, value)
          })
        }
      }
      i++
      continue
    }
    
    // Pattern 2: Section headers
    if (upperLine.startsWith('SECTION:') || upperLine.startsWith('[SECTION]')) {
      const markerLen = upperLine.startsWith('[SECTION]') ? 9 : 8
      currentSection = line.slice(markerLen).trim()
      if (currentSection && !sections.includes(currentSection)) {
        sections.push(currentSection)
      }
      i++
      continue
    }
    
    // Pattern 3: Bullet points and numbered lists (Western + Arabic numerals)
    const bulletMatch = line.match(/^[•\-\*]\s*(.+)$/)
    // Match both Western (1, 2, 3) and Arabic (١، ٢، ٣) numerals with various separators
    const numberMatch = line.match(/^[\d١٢٣٤٥٦٧٨٩٠]+[\.\)\-–]\s*(.+)$/)
    
    if (bulletMatch || numberMatch) {
      let content = (bulletMatch || numberMatch)![1].trim()
      // Remove trailing period/dot
      content = content.replace(/\s*\.\s*$/, '')
      
      if (content.includes(':')) {
        // This is a field:value pair within a list
        const colonIdx = findMeaningfulColon(content)
        if (colonIdx > 0) {
          const label = content.slice(0, colonIdx).trim()
          const value = content.slice(colonIdx + 1).trim()
          if (label && label.length > 1 && label.length < 80) {
            fields.push({
              label: label,
              value: normalizeEmptyValue(value),
              type: inferFieldType(label, value)
            })
          }
        }
      } else {
        // This is a standalone list item (checklist/requirements)
        if (content && content.length > 2 && content.length < 200) {
          fields.push({
            label: content,
            value: '',  // No value for checklist items
            type: 'text'  // Could add 'list_item' type if needed
          })
        }
      }
      i++
      continue
    }
    
    // Pattern 4: Legacy FIELD:/VALUE: format
    if (upperLine.startsWith('FIELD:')) {
      let fieldLabel = line.slice(6).trim()
      let value = ''
      
      if (i + 1 < lines.length) {
        const nextLine = lines[i + 1]
        if (nextLine.toUpperCase().startsWith('VALUE:')) {
          value = nextLine.slice(6).trim()
          if (!value && i + 2 < lines.length) {
            const potential = lines[i + 2]
            if (!potential.toUpperCase().startsWith('FIELD:')) {
              value = potential
              i++
            }
          }
          i++
        }
      }
      
      if (fieldLabel && fieldLabel.length > 1) {
        const checkboxValues = ['☑', '☐', '✓', '✗', 'نعم', 'لا', 'yes', 'no', 'checked', 'unchecked']
        if (checkboxValues.includes(value.toLowerCase())) {
          checkboxes.push({
            label: fieldLabel,
            checked: ['☑', '✓', 'نعم', 'yes', 'checked'].includes(value.toLowerCase())
          })
        } else {
          fields.push({
            label: fieldLabel,
            value: value === '-' || value.toUpperCase() === 'EMPTY' ? '' : value,
            type: inferFieldType(fieldLabel, value)
          })
        }
      }
      i++
      continue
    }
    
    // Pattern 5: Natural "label: value" format - HANDLES MULTIPLE FIELDS PER LINE
    if (line.includes(':') && !line.startsWith('[') && !line.startsWith('http')) {
      // Split by common table separators (multiple spaces, tabs, pipes)
      const segments = line.split(/\s{3,}|\t|\|/)
      
      for (const segment of segments) {
        const trimmedSegment = segment.trim()
        if (!trimmedSegment || !trimmedSegment.includes(':')) continue
        
        const colonIdx = findMeaningfulColon(trimmedSegment)
        
        if (colonIdx > 0) {
          let label = trimmedSegment.slice(0, colonIdx).trim()
          const value = trimmedSegment.slice(colonIdx + 1).trim()
          
          // Clean label
          label = label.replace(/^[•\-\*\d\.\)]+\s*/, '').trim()
          
          if (label && 
              label.length > 1 && 
              label.length < 80 && 
              !/^\d+$/.test(label) &&
              !GARBAGE_PATTERNS.some(pat => label.toLowerCase().includes(pat))) {
            
            // Normalize empty values
            const normalizedValue = normalizeEmptyValue(value)
            
            const checkboxValues = ['☑', '☐', '✓', '✗', 'نعم', 'لا', 'yes', 'no']
            if (checkboxValues.includes(value.toLowerCase())) {
              checkboxes.push({
                label: label,
                checked: ['☑', '✓', 'نعم', 'yes'].includes(value.toLowerCase())
              })
            } else {
              fields.push({
                label: label,
                value: normalizedValue,
                type: inferFieldType(label, value)
              })
            }
          }
        }
      }
    }
    
    i++
  }
  
  return { fields, sections, checkboxes, formTitle }
}

/**
 * Find the first meaningful colon in a line (not a time separator)
 */
function findMeaningfulColon(text: string): number {
  for (let idx = 0; idx < text.length; idx++) {
    if (text[idx] === ':') {
      const before = text.slice(0, idx).trim()
      const after = text.slice(idx + 1).trim()
      // Skip time patterns like 12:30
      if (before && after && /\d$/.test(before) && /^\d/.test(after)) {
        continue
      }
      return idx
    }
  }
  return -1
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
