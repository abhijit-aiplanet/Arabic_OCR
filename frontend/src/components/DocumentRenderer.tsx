'use client'

import { useState, useMemo } from 'react'
import { 
  Copy, Check, Download, FileText, Eye, Code,
  ChevronDown, ChevronUp, Maximize2, Minimize2,
  Table2, FormInput, List
} from 'lucide-react'
import toast from 'react-hot-toast'

interface DocumentRendererProps {
  rawText: string
  imagePreview?: string | null
  confidence?: number
  className?: string
}

type ViewMode = 'document' | 'structured' | 'raw'

export default function DocumentRenderer({
  rawText,
  imagePreview,
  confidence,
  className = ''
}: DocumentRendererProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('document')
  const [copied, setCopied] = useState(false)
  const [showImage, setShowImage] = useState(true)

  // Parse the raw text into a document structure
  const documentStructure = useMemo(() => {
    return parseToDocument(rawText)
  }, [rawText])

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(rawText)
      setCopied(true)
      toast.success('Copied!')
      setTimeout(() => setCopied(false), 2000)
    } catch {
      toast.error('Failed to copy')
    }
  }

  const handleDownload = () => {
    const blob = new Blob([rawText], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'extracted-text.txt'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    toast.success('Downloaded!')
  }

  if (!rawText) {
    return (
      <div className={`bg-white rounded-2xl border border-gray-100 p-8 text-center ${className}`}>
        <FileText className="w-12 h-12 text-gray-300 mx-auto mb-3" />
        <p className="text-gray-500">No text extracted yet</p>
      </div>
    )
  }

  return (
    <div className={`bg-white rounded-2xl border border-gray-100 overflow-hidden flex flex-col ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-100 bg-gray-50/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h3 className="font-medium text-gray-900">Output</h3>
            {confidence && (
              <span className={`text-xs px-2 py-1 rounded-full ${
                confidence >= 80 ? 'bg-emerald-100 text-emerald-700' :
                confidence >= 60 ? 'bg-amber-100 text-amber-700' :
                'bg-red-100 text-red-700'
              }`}>
                {confidence}% confidence
              </span>
            )}
          </div>
          
          <div className="flex items-center gap-1">
            {/* View Mode Toggle */}
            <div className="flex items-center bg-white border border-gray-200 rounded-lg p-0.5 mr-2">
              <button
                onClick={() => setViewMode('document')}
                className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors flex items-center gap-1 ${
                  viewMode === 'document' 
                    ? 'bg-gray-900 text-white' 
                    : 'text-gray-600 hover:text-gray-900'
                }`}
                title="Document View"
              >
                <Eye className="w-3.5 h-3.5" />
                Document
              </button>
              <button
                onClick={() => setViewMode('structured')}
                className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors flex items-center gap-1 ${
                  viewMode === 'structured' 
                    ? 'bg-gray-900 text-white' 
                    : 'text-gray-600 hover:text-gray-900'
                }`}
                title="Structured View"
              >
                <List className="w-3.5 h-3.5" />
                Fields
              </button>
              <button
                onClick={() => setViewMode('raw')}
                className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors flex items-center gap-1 ${
                  viewMode === 'raw' 
                    ? 'bg-gray-900 text-white' 
                    : 'text-gray-600 hover:text-gray-900'
                }`}
                title="Raw Text"
              >
                <Code className="w-3.5 h-3.5" />
                Raw
              </button>
            </div>
            
            {/* Actions */}
            <button
              onClick={handleCopy}
              className={`p-2 rounded-lg transition-colors ${
                copied ? 'text-emerald-600 bg-emerald-50' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
              }`}
              title="Copy"
            >
              {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
            </button>
            <button
              onClick={handleDownload}
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
              title="Download"
            >
              <Download className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {viewMode === 'document' && (
          <DocumentView structure={documentStructure} />
        )}
        {viewMode === 'structured' && (
          <StructuredView structure={documentStructure} />
        )}
        {viewMode === 'raw' && (
          <RawView text={rawText} />
        )}
      </div>
    </div>
  )
}

// Document structure types
interface DocumentLine {
  type: 'header' | 'field' | 'text' | 'table-row' | 'empty' | 'list-item'
  content: string
  label?: string
  value?: string
  level?: number
}

interface DocumentStructure {
  lines: DocumentLine[]
  fields: Array<{ label: string; value: string }>
  hasStructure: boolean
}

// Parse raw text into document structure
function parseToDocument(text: string): DocumentStructure {
  const lines: DocumentLine[] = []
  const fields: Array<{ label: string; value: string }> = []
  
  const rawLines = text.split('\n')
  let hasStructure = false
  
  let currentField: { label: string; value: string } | null = null
  
  for (let i = 0; i < rawLines.length; i++) {
    const line = rawLines[i]
    const trimmed = line.trim()
    
    // Empty line
    if (!trimmed) {
      lines.push({ type: 'empty', content: '' })
      continue
    }
    
    // [FIELD] marker
    if (trimmed.startsWith('[FIELD]') || trimmed.toUpperCase().startsWith('[FIELD]')) {
      hasStructure = true
      const labelPart = trimmed.slice(7).trim()
      
      // Check if [VALUE] is inline
      if (labelPart.includes('[VALUE]') || labelPart.toUpperCase().includes('[VALUE]')) {
        const parts = labelPart.split(/\[VALUE\]/i)
        const label = parts[0].trim()
        const value = parts[1]?.trim() || ''
        
        if (label) {
          lines.push({ type: 'field', content: line, label, value })
          fields.push({ label, value })
        }
      } else {
        currentField = { label: labelPart, value: '' }
      }
      continue
    }
    
    // [VALUE] marker
    if (trimmed.startsWith('[VALUE]') || trimmed.toUpperCase().startsWith('[VALUE]')) {
      hasStructure = true
      const valuePart = trimmed.slice(7).trim()
      
      if (currentField) {
        // If value is empty, check next line
        if (!valuePart || valuePart === '-') {
          // Look at next non-empty line
          let nextIdx = i + 1
          while (nextIdx < rawLines.length && !rawLines[nextIdx].trim()) {
            nextIdx++
          }
          if (nextIdx < rawLines.length) {
            const nextLine = rawLines[nextIdx].trim()
            if (!nextLine.startsWith('[FIELD]') && 
                !nextLine.toUpperCase().startsWith('[FIELD]') &&
                !nextLine.startsWith('[VALUE]')) {
              currentField.value = nextLine
              i = nextIdx
            }
          }
        } else {
          currentField.value = valuePart
        }
        
        lines.push({ 
          type: 'field', 
          content: line, 
          label: currentField.label, 
          value: currentField.value 
        })
        fields.push({ ...currentField })
        currentField = null
      }
      continue
    }
    
    // Header detection (all caps or starts with specific patterns)
    if (isLikelyHeader(trimmed)) {
      lines.push({ type: 'header', content: trimmed, level: 1 })
      continue
    }
    
    // Colon-separated field
    if (trimmed.includes(':') && !trimmed.startsWith('http')) {
      const colonIdx = trimmed.indexOf(':')
      const potentialLabel = trimmed.slice(0, colonIdx).trim()
      const potentialValue = trimmed.slice(colonIdx + 1).trim()
      
      if (potentialLabel.length > 1 && potentialLabel.length < 60 && potentialValue) {
        hasStructure = true
        lines.push({ 
          type: 'field', 
          content: trimmed, 
          label: potentialLabel, 
          value: potentialValue 
        })
        fields.push({ label: potentialLabel, value: potentialValue })
        continue
      }
    }
    
    // List item (Western and Arabic numerals)
    if (trimmed.match(/^[-•*]\s/) || trimmed.match(/^[\d١٢٣٤٥٦٧٨٩٠]+[.)\-–]\s*/)) {
      // Clean the content - remove the list marker for cleaner display
      const cleanContent = trimmed.replace(/^[-•*]\s*/, '').replace(/^[\d١٢٣٤٥٦٧٨٩٠]+[.)\-–]\s*/, '')
      lines.push({ type: 'list-item', content: cleanContent || trimmed })
      continue
    }
    
    // Regular text
    lines.push({ type: 'text', content: trimmed })
  }
  
  return { lines, fields, hasStructure }
}

function isLikelyHeader(text: string): boolean {
  // Arabic section headers often have specific patterns
  if (/^(بيانات|معلومات|تفاصيل|ملاحظات|المطلوب|طلبات|متطلبات|شروط)/i.test(text)) return true
  // Headers ending with colon (like "طلبات الإصدار الجديد :")
  if (/^[\u0600-\u06FF\s]+\s*:\s*$/.test(text) && text.length < 50) return true
  // Short lines with no punctuation
  if (text.length < 40 && !text.includes(':') && !text.includes('.')) {
    // Check if mostly uppercase (for English) or starts with specific Arabic words
    if (text === text.toUpperCase() && /[A-Z]/.test(text)) return true
  }
  return false
}

// Document View - Renders text preserving structure
function DocumentView({ structure }: { structure: DocumentStructure }) {
  return (
    <div className="p-6 max-w-none" dir="auto">
      <div className="prose prose-sm max-w-none" style={{ direction: 'rtl', textAlign: 'right' }}>
        {structure.lines.map((line, idx) => {
          switch (line.type) {
            case 'header':
              return (
                <h3 key={idx} className="text-base font-bold text-gray-900 mt-6 mb-3 pb-2 border-b border-gray-200">
                  {line.content}
                </h3>
              )
            
            case 'field':
              return (
                <div key={idx} className="flex items-start gap-3 py-2 border-b border-gray-50 hover:bg-gray-50 rounded px-2 -mx-2">
                  <span className="text-gray-600 font-medium min-w-0 flex-shrink-0">
                    {line.label}:
                  </span>
                  <span className={`flex-1 ${line.value ? 'text-gray-900 font-semibold' : 'text-gray-300 italic'}`}>
                    {line.value || 'غير محدد'}
                  </span>
                </div>
              )
            
            case 'list-item':
              return (
                <div key={idx} className="flex items-start gap-2 py-1 pr-4">
                  <span className="text-gray-400">•</span>
                  <span className="text-gray-800">{line.content.replace(/^[-•*]\s*/, '').replace(/^\d+[.)]\s*/, '')}</span>
                </div>
              )
            
            case 'empty':
              return <div key={idx} className="h-3" />
            
            default:
              return (
                <p key={idx} className="text-gray-800 leading-relaxed my-2">
                  {line.content}
                </p>
              )
          }
        })}
      </div>
    </div>
  )
}

// Structured View - Shows fields in a table
function StructuredView({ structure }: { structure: DocumentStructure }) {
  if (structure.fields.length === 0) {
    return (
      <div className="p-8 text-center">
        <FormInput className="w-10 h-10 text-gray-300 mx-auto mb-3" />
        <p className="text-gray-500 text-sm">No structured fields detected</p>
        <p className="text-gray-400 text-xs mt-1">Try the Document view for plain text</p>
      </div>
    )
  }

  const filledFields = structure.fields.filter(f => f.value && f.value !== '-')
  
  return (
    <div className="p-4">
      <div className="mb-4 flex items-center justify-between">
        <span className="text-sm text-gray-600">
          <span className="font-semibold text-gray-900">{filledFields.length}</span> of {structure.fields.length} fields filled
        </span>
      </div>
      
      <div className="space-y-2">
        {structure.fields.map((field, idx) => (
          <div 
            key={idx}
            className={`rounded-xl p-4 ${
              field.value && field.value !== '-'
                ? 'bg-white border border-gray-200 shadow-sm'
                : 'bg-gray-50 border border-dashed border-gray-200'
            }`}
            dir="auto"
          >
            <div className="text-xs font-medium text-gray-500 mb-1">
              {field.label}
            </div>
            <div className={`text-base ${
              field.value && field.value !== '-'
                ? 'text-gray-900 font-medium'
                : 'text-gray-300 italic'
            }`}>
              {field.value && field.value !== '-' ? field.value : 'Not filled'}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// Raw View - Shows raw text
function RawView({ text }: { text: string }) {
  return (
    <div className="p-4">
      <pre 
        className="whitespace-pre-wrap text-sm text-gray-800 font-mono bg-gray-50 rounded-xl p-4 overflow-auto"
        dir="auto"
        style={{ direction: 'rtl', textAlign: 'right' }}
      >
        {text}
      </pre>
    </div>
  )
}
