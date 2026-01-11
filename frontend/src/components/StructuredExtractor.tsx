'use client'

import { useState, useMemo, useEffect } from 'react'
import { 
  Copy, Check, Download, FileJson, FileSpreadsheet, 
  ChevronDown, ChevronUp, Edit2, Save, X, ZoomIn, ZoomOut,
  FormInput, CheckSquare, FileText, AlertCircle,
  Maximize2, Minimize2, Eye, List, Code, Table2
} from 'lucide-react'
import toast from 'react-hot-toast'
import type { StructuredExtraction, ExtractedSection, ExtractedField, ExtractedTable, ExtractedCheckbox } from '@/lib/structuredParser'
import { structuredToCSV, structuredToJSON, structuredToPlainText, parseStructuredOutput } from '@/lib/structuredParser'

interface StructuredExtractorProps {
  imagePreview: string | null
  structuredData: StructuredExtraction | null
  isProcessing: boolean
  parsingSuccessful: boolean
  onFieldEdit?: (sectionIndex: number, fieldIndex: number, newValue: string) => void
  onSaveAsTemplate?: () => void
  rawText?: string
}

type ViewMode = 'document' | 'cards' | 'table'

export default function StructuredExtractor({
  imagePreview,
  structuredData,
  isProcessing,
  parsingSuccessful,
  onFieldEdit,
  onSaveAsTemplate,
  rawText
}: StructuredExtractorProps) {
  const [imageZoom, setImageZoom] = useState(1)
  const [editingField, setEditingField] = useState<{ section: number; field: number } | null>(null)
  const [editValue, setEditValue] = useState('')
  const [copied, setCopied] = useState(false)
  const [showEmptyFields, setShowEmptyFields] = useState(true)
  const [fullscreenImage, setFullscreenImage] = useState(false)
  const [viewMode, setViewMode] = useState<ViewMode>('document')

  // Re-parse the raw text to get better results if structuredData is poor
  const parsedData = useMemo(() => {
    if (structuredData && structuredData.sections.length > 0) {
      // Check if we have filled fields
      const filledCount = structuredData.sections.reduce((sum, s) => 
        sum + s.fields.filter(f => f.value && f.value.trim()).length, 0
      )
      
      // If we have good data, use it
      if (filledCount > 0) {
        return structuredData
      }
    }
    
    // Try re-parsing from raw_text
    const textToParse = rawText || structuredData?.raw_text
    if (textToParse) {
      const result = parseStructuredOutput(textToParse)
      if (result.success && result.data) {
        return result.data
      }
    }
    
    return structuredData
  }, [structuredData, rawText])

  const totalFields = useMemo(() => {
    if (!parsedData) return 0
    return parsedData.sections.reduce((sum, s) => sum + s.fields.length, 0) +
           parsedData.checkboxes.length
  }, [parsedData])

  const filledFields = useMemo(() => {
    if (!parsedData) return 0
    return parsedData.sections.reduce((sum, s) => 
      sum + s.fields.filter(f => f.value && f.value.trim() && f.value !== '-').length, 0
    ) + parsedData.checkboxes.length
  }, [parsedData])

  const allFields = useMemo(() => {
    if (!parsedData) return []
    return parsedData.sections.flatMap((section, sIdx) => 
      section.fields.map((field, fIdx) => ({
        ...field,
        sectionName: section.name || 'General',
        sectionIndex: sIdx,
        fieldIndex: fIdx
      }))
    )
  }, [parsedData])

  const visibleFields = useMemo(() => {
    if (showEmptyFields) return allFields
    return allFields.filter(f => f.value && f.value.trim() && f.value !== '-')
  }, [allFields, showEmptyFields])

  const handleCopy = async () => {
    if (!parsedData) return
    try {
      const text = structuredToPlainText(parsedData)
      await navigator.clipboard.writeText(text)
      setCopied(true)
      toast.success('Copied!')
      setTimeout(() => setCopied(false), 2000)
    } catch {
      toast.error('Failed to copy')
    }
  }

  const handleExportJSON = () => {
    if (!parsedData) return
    const json = structuredToJSON(parsedData)
    downloadFile(json, 'extracted-data.json', 'application/json')
    toast.success('JSON exported!')
  }

  const handleExportCSV = () => {
    if (!parsedData) return
    const csv = structuredToCSV(parsedData)
    downloadFile(csv, 'extracted-data.csv', 'text/csv')
    toast.success('CSV exported!')
  }

  const downloadFile = (content: string, filename: string, type: string) => {
    const blob = new Blob([content], { type })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const startEditing = (sectionIndex: number, fieldIndex: number, currentValue: string) => {
    setEditingField({ section: sectionIndex, field: fieldIndex })
    setEditValue(currentValue)
  }

  const cancelEditing = () => {
    setEditingField(null)
    setEditValue('')
  }

  const saveEditing = () => {
    if (editingField && onFieldEdit) {
      onFieldEdit(editingField.section, editingField.field, editValue)
    }
    setEditingField(null)
    setEditValue('')
  }

  // Loading state
  if (isProcessing) {
    return (
      <div className="bg-white rounded-2xl border border-gray-100 h-full flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="relative w-16 h-16 mx-auto mb-4">
            <div className="w-16 h-16 border-2 border-gray-100 rounded-full"></div>
            <div className="absolute inset-0 w-16 h-16 border-2 border-transparent border-t-gray-900 rounded-full animate-spin"></div>
          </div>
          <p className="text-base font-medium text-gray-900">Extracting Form Data</p>
          <p className="text-sm text-gray-500 mt-1">Analyzing structure...</p>
        </div>
      </div>
    )
  }

  // Empty state
  if (!parsedData && !imagePreview) {
    return (
      <div className="bg-white rounded-2xl border border-gray-100 h-full flex items-center justify-center min-h-[400px]">
        <div className="text-center px-6">
          <FormInput className="w-12 h-12 text-gray-300 mx-auto mb-4" />
          <h3 className="font-semibold text-gray-900 mb-1">Structured Extraction</h3>
          <p className="text-sm text-gray-500 max-w-[250px]">
            Upload a form to extract key-value pairs
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-2xl border border-gray-100 overflow-hidden flex flex-col h-full">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-100 bg-gray-50/50 flex-shrink-0">
        <div className="flex items-center justify-between mb-2">
          <div>
            <h3 className="font-semibold text-gray-900">Extracted Form Data</h3>
            {parsedData && (
              <p className="text-xs text-gray-500">
                <span className="font-medium text-gray-900">{filledFields}</span> of {totalFields} fields filled
              </p>
            )}
          </div>
          
          <div className="flex items-center gap-1">
            <button
              onClick={handleCopy}
              className={`p-1.5 rounded-lg transition-colors ${
                copied ? 'text-emerald-600 bg-emerald-50' : 'text-gray-500 hover:bg-gray-100'
              }`}
              title="Copy"
            >
              {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
            </button>
            <button
              onClick={handleExportJSON}
              className="p-1.5 text-gray-500 hover:bg-gray-100 rounded-lg"
              title="Export JSON"
            >
              <FileJson className="w-4 h-4" />
            </button>
            <button
              onClick={handleExportCSV}
              className="p-1.5 text-gray-500 hover:bg-gray-100 rounded-lg"
              title="Export CSV"
            >
              <FileSpreadsheet className="w-4 h-4" />
            </button>
            {onSaveAsTemplate && (
              <button
                onClick={onSaveAsTemplate}
                className="ml-2 px-3 py-1.5 text-xs font-medium text-white bg-gray-900 hover:bg-gray-800 rounded-lg"
              >
                Save as Template
              </button>
            )}
          </div>
        </div>
        
        {/* View Mode Toggle */}
        {parsedData && (
          <div className="flex items-center gap-2">
            <div className="flex items-center bg-white border border-gray-200 rounded-lg p-0.5">
              <button
                onClick={() => setViewMode('document')}
                className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors flex items-center gap-1 ${
                  viewMode === 'document' ? 'bg-gray-900 text-white' : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Eye className="w-3 h-3" />
                Document
              </button>
              <button
                onClick={() => setViewMode('cards')}
                className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors flex items-center gap-1 ${
                  viewMode === 'cards' ? 'bg-gray-900 text-white' : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <List className="w-3 h-3" />
                Cards
              </button>
              <button
                onClick={() => setViewMode('table')}
                className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors flex items-center gap-1 ${
                  viewMode === 'table' ? 'bg-gray-900 text-white' : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Table2 className="w-3 h-3" />
                Table
              </button>
            </div>
            
            <button
              onClick={() => setShowEmptyFields(!showEmptyFields)}
              className={`px-2.5 py-1 text-xs font-medium rounded-lg transition-colors ${
                showEmptyFields ? 'bg-gray-100 text-gray-600' : 'bg-gray-900 text-white'
              }`}
            >
              {showEmptyFields ? 'Hide empty' : 'Show all'}
            </button>
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto min-h-0">
        {parsedData ? (
          <>
            {viewMode === 'document' && (
              <DocumentStyleView 
                data={parsedData} 
                showEmpty={showEmptyFields}
              />
            )}
            {viewMode === 'cards' && (
              <CardsView 
                fields={visibleFields}
                editingField={editingField}
                editValue={editValue}
                onStartEdit={startEditing}
                onCancelEdit={cancelEditing}
                onSaveEdit={saveEditing}
                onEditValueChange={setEditValue}
                canEdit={!!onFieldEdit}
              />
            )}
            {viewMode === 'table' && (
              <TableView 
                fields={visibleFields}
                editingField={editingField}
                editValue={editValue}
                onStartEdit={startEditing}
                onCancelEdit={cancelEditing}
                onSaveEdit={saveEditing}
                onEditValueChange={setEditValue}
                canEdit={!!onFieldEdit}
              />
            )}
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center p-8">
            <p className="text-gray-500">Process an image to see extracted data</p>
          </div>
        )}
      </div>
    </div>
  )
}

// Document Style View - Renders like a document
function DocumentStyleView({ 
  data, 
  showEmpty 
}: { 
  data: StructuredExtraction
  showEmpty: boolean 
}) {
  const visibleSections = data.sections.map(section => ({
    ...section,
    fields: showEmpty 
      ? section.fields 
      : section.fields.filter(f => f.value && f.value.trim() && f.value !== '-')
  })).filter(s => s.fields.length > 0)

  if (visibleSections.length === 0) {
    return (
      <div className="p-8 text-center">
        <FormInput className="w-10 h-10 text-gray-300 mx-auto mb-3" />
        <p className="text-gray-500 text-sm">No fields extracted</p>
      </div>
    )
  }

  return (
    <div className="p-6" dir="auto" style={{ direction: 'rtl', textAlign: 'right' }}>
      {data.form_title && (
        <h2 className="text-lg font-bold text-gray-900 mb-4 pb-2 border-b-2 border-gray-900">
          {data.form_title}
        </h2>
      )}
      
      {visibleSections.map((section, sIdx) => (
        <div key={sIdx} className="mb-6">
          {section.name && (
            <h3 className="text-sm font-bold text-gray-700 mb-3 pb-1 border-b border-gray-200 flex items-center gap-2">
              <span className="w-1.5 h-1.5 bg-gray-900 rounded-full" />
              {section.name}
            </h3>
          )}
          
          <div className="space-y-1">
            {section.fields.map((field, fIdx) => {
              const hasValue = field.value && field.value.trim() && field.value !== '-'
              
              return (
                <div 
                  key={fIdx}
                  className={`flex items-start gap-3 py-2 px-3 rounded-lg transition-colors ${
                    hasValue ? 'hover:bg-gray-50' : 'opacity-50'
                  }`}
                >
                  <span className="text-gray-600 min-w-0 text-sm">
                    {field.label}:
                  </span>
                  <span className={`flex-1 text-sm ${
                    hasValue ? 'text-gray-900 font-semibold' : 'text-gray-300 italic'
                  }`}>
                    {hasValue ? field.value : 'غير محدد'}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      ))}
      
      {/* Checkboxes */}
      {data.checkboxes.length > 0 && (
        <div className="mt-6 pt-4 border-t border-gray-200">
          <h3 className="text-sm font-bold text-gray-700 mb-3">الخيارات</h3>
          <div className="space-y-2">
            {data.checkboxes.map((cb, idx) => (
              <div key={idx} className="flex items-center gap-3">
                <div className={`w-5 h-5 rounded flex items-center justify-center ${
                  cb.checked ? 'bg-emerald-500' : 'border-2 border-gray-300'
                }`}>
                  {cb.checked && <Check className="w-3 h-3 text-white" />}
                </div>
                <span className="text-sm text-gray-700">{cb.label}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// Cards View
function CardsView({
  fields,
  editingField,
  editValue,
  onStartEdit,
  onCancelEdit,
  onSaveEdit,
  onEditValueChange,
  canEdit
}: {
  fields: Array<ExtractedField & { sectionName: string; sectionIndex: number; fieldIndex: number }>
  editingField: { section: number; field: number } | null
  editValue: string
  onStartEdit: (sectionIndex: number, fieldIndex: number, value: string) => void
  onCancelEdit: () => void
  onSaveEdit: () => void
  onEditValueChange: (value: string) => void
  canEdit: boolean
}) {
  if (fields.length === 0) {
    return (
      <div className="p-8 text-center">
        <FormInput className="w-10 h-10 text-gray-300 mx-auto mb-3" />
        <p className="text-gray-500 text-sm">No fields to display</p>
      </div>
    )
  }

  return (
    <div className="p-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
      {fields.map((field, idx) => {
        const isEditing = editingField?.section === field.sectionIndex && editingField?.field === field.fieldIndex
        const hasValue = field.value && field.value.trim() && field.value !== '-'
        
        return (
          <div 
            key={idx}
            className={`rounded-xl p-4 ${
              hasValue 
                ? 'bg-white border border-gray-200 shadow-sm' 
                : 'bg-gray-50 border border-dashed border-gray-200'
            }`}
            dir="auto"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-medium text-gray-500">{field.label}</span>
              <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                field.type === 'date' ? 'bg-blue-50 text-blue-600' :
                field.type === 'number' ? 'bg-green-50 text-green-600' :
                'bg-gray-50 text-gray-500'
              }`}>
                {field.type}
              </span>
            </div>
            
            {isEditing ? (
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={editValue}
                  onChange={(e) => onEditValueChange(e.target.value)}
                  className="flex-1 px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-gray-900"
                  autoFocus
                  dir="auto"
                />
                <button onClick={onSaveEdit} className="p-1 text-emerald-600 hover:bg-emerald-50 rounded">
                  <Save className="w-4 h-4" />
                </button>
                <button onClick={onCancelEdit} className="p-1 text-gray-400 hover:bg-gray-100 rounded">
                  <X className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <div className="group flex items-start gap-2">
                <p className={`flex-1 text-sm ${
                  hasValue ? 'text-gray-900 font-medium' : 'text-gray-300 italic'
                }`}>
                  {hasValue ? field.value : 'Not filled'}
                </p>
                {canEdit && (
                  <button
                    onClick={() => onStartEdit(field.sectionIndex, field.fieldIndex, field.value)}
                    className="p-1 text-gray-300 hover:text-gray-600 hover:bg-gray-100 rounded opacity-0 group-hover:opacity-100 transition-all"
                  >
                    <Edit2 className="w-3 h-3" />
                  </button>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

// Table View
function TableView({
  fields,
  editingField,
  editValue,
  onStartEdit,
  onCancelEdit,
  onSaveEdit,
  onEditValueChange,
  canEdit
}: {
  fields: Array<ExtractedField & { sectionName: string; sectionIndex: number; fieldIndex: number }>
  editingField: { section: number; field: number } | null
  editValue: string
  onStartEdit: (sectionIndex: number, fieldIndex: number, value: string) => void
  onCancelEdit: () => void
  onSaveEdit: () => void
  onEditValueChange: (value: string) => void
  canEdit: boolean
}) {
  if (fields.length === 0) {
    return (
      <div className="p-8 text-center">
        <Table2 className="w-10 h-10 text-gray-300 mx-auto mb-3" />
        <p className="text-gray-500 text-sm">No fields to display</p>
      </div>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="bg-gray-50 sticky top-0">
          <tr>
            <th className="px-4 py-2 text-right font-medium text-gray-600 border-b">Field</th>
            <th className="px-4 py-2 text-right font-medium text-gray-600 border-b">Value</th>
            <th className="px-4 py-2 text-center font-medium text-gray-600 border-b w-16">Type</th>
            {canEdit && <th className="px-4 py-2 w-10 border-b"></th>}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {fields.map((field, idx) => {
            const isEditing = editingField?.section === field.sectionIndex && editingField?.field === field.fieldIndex
            const hasValue = field.value && field.value.trim() && field.value !== '-'
            
            return (
              <tr key={idx} className={hasValue ? 'hover:bg-gray-50' : 'bg-gray-50/30'}>
                <td className="px-4 py-2 font-medium text-gray-900" dir="auto">{field.label}</td>
                <td className="px-4 py-2" dir="auto">
                  {isEditing ? (
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={editValue}
                        onChange={(e) => onEditValueChange(e.target.value)}
                        className="flex-1 px-2 py-1 text-sm border rounded focus:outline-none focus:ring-2 focus:ring-gray-900"
                        autoFocus
                        dir="auto"
                      />
                      <button onClick={onSaveEdit} className="p-1 text-emerald-600"><Save className="w-4 h-4" /></button>
                      <button onClick={onCancelEdit} className="p-1 text-gray-400"><X className="w-4 h-4" /></button>
                    </div>
                  ) : (
                    <span className={hasValue ? 'text-gray-900' : 'text-gray-300 italic'}>
                      {hasValue ? field.value : 'Not filled'}
                    </span>
                  )}
                </td>
                <td className="px-4 py-2 text-center">
                  <span className={`text-xs px-2 py-0.5 rounded ${
                    field.type === 'date' ? 'bg-blue-50 text-blue-600' :
                    field.type === 'number' ? 'bg-green-50 text-green-600' :
                    'bg-gray-100 text-gray-600'
                  }`}>
                    {field.type}
                  </span>
                </td>
                {canEdit && (
                  <td className="px-4 py-2">
                    {!isEditing && (
                      <button
                        onClick={() => onStartEdit(field.sectionIndex, field.fieldIndex, field.value)}
                        className="p-1 text-gray-400 hover:text-gray-600"
                      >
                        <Edit2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </td>
                )}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
