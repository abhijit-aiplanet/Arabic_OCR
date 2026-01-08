'use client'

import { useState, useMemo } from 'react'
import { 
  Copy, Check, Download, FileJson, FileSpreadsheet, 
  ChevronDown, ChevronUp, Edit2, Save, X, ZoomIn, ZoomOut,
  Table, FormInput, CheckSquare, FileText, AlertCircle
} from 'lucide-react'
import toast from 'react-hot-toast'
import type { StructuredExtraction, ExtractedSection, ExtractedField, ExtractedTable, ExtractedCheckbox } from '@/lib/structuredParser'
import { structuredToCSV, structuredToJSON, structuredToPlainText } from '@/lib/structuredParser'

interface StructuredExtractorProps {
  imagePreview: string | null
  structuredData: StructuredExtraction | null
  isProcessing: boolean
  parsingSuccessful: boolean
  onFieldEdit?: (sectionIndex: number, fieldIndex: number, newValue: string) => void
  onSaveAsTemplate?: () => void
}

export default function StructuredExtractor({
  imagePreview,
  structuredData,
  isProcessing,
  parsingSuccessful,
  onFieldEdit,
  onSaveAsTemplate
}: StructuredExtractorProps) {
  const [imageZoom, setImageZoom] = useState(1)
  const [expandedSections, setExpandedSections] = useState<Set<number>>(new Set([0]))
  const [editingField, setEditingField] = useState<{ section: number; field: number } | null>(null)
  const [editValue, setEditValue] = useState('')
  const [copied, setCopied] = useState(false)

  const totalFields = useMemo(() => {
    if (!structuredData) return 0
    return structuredData.sections.reduce((sum, s) => sum + s.fields.length, 0) +
           structuredData.checkboxes.length
  }, [structuredData])

  const filledFields = useMemo(() => {
    if (!structuredData) return 0
    return structuredData.sections.reduce((sum, s) => 
      sum + s.fields.filter(f => f.value.trim()).length, 0
    ) + structuredData.checkboxes.length
  }, [structuredData])

  const toggleSection = (index: number) => {
    const newExpanded = new Set(expandedSections)
    if (newExpanded.has(index)) {
      newExpanded.delete(index)
    } else {
      newExpanded.add(index)
    }
    setExpandedSections(newExpanded)
  }

  const handleCopy = async () => {
    if (!structuredData) return
    try {
      const text = structuredToPlainText(structuredData)
      await navigator.clipboard.writeText(text)
      setCopied(true)
      toast.success('Copied to clipboard!')
      setTimeout(() => setCopied(false), 2000)
    } catch {
      toast.error('Failed to copy')
    }
  }

  const handleExportJSON = () => {
    if (!structuredData) return
    const json = structuredToJSON(structuredData)
    downloadFile(json, 'extracted-data.json', 'application/json')
    toast.success('JSON exported!')
  }

  const handleExportCSV = () => {
    if (!structuredData) return
    const csv = structuredToCSV(structuredData)
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
      <div className="bg-white rounded-2xl border border-gray-100 h-full flex items-center justify-center min-h-[500px]">
        <div className="text-center">
          <div className="relative w-16 h-16 mx-auto mb-4">
            <div className="w-16 h-16 border-2 border-gray-200 rounded-full"></div>
            <div className="absolute inset-0 w-16 h-16 border-2 border-transparent border-t-gray-900 rounded-full animate-spin"></div>
          </div>
          <p className="text-base font-medium text-gray-900">Extracting form data...</p>
          <p className="text-sm text-gray-500 mt-1">Analyzing structure and fields</p>
        </div>
      </div>
    )
  }

  // Empty state
  if (!structuredData && !imagePreview) {
    return (
      <div className="bg-white rounded-2xl border border-gray-100 h-full flex items-center justify-center min-h-[500px]">
        <div className="text-center">
          <div className="w-16 h-16 bg-gray-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <FormInput className="w-8 h-8 text-gray-400" />
          </div>
          <p className="text-base font-medium text-gray-900">Structured Extraction</p>
          <p className="text-sm text-gray-500 mt-1 max-w-[250px] mx-auto">
            Upload a form image to extract key-value pairs and structured data
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-2xl border border-gray-100 overflow-hidden h-full flex flex-col">
      {/* Header */}
      <div className="px-5 py-4 border-b border-gray-100 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-gray-900">
              {structuredData?.form_title || 'Extracted Form Data'}
            </h2>
            {structuredData && (
              <p className="text-xs text-gray-500 mt-0.5">
                {filledFields} of {totalFields} fields extracted
                {!parsingSuccessful && (
                  <span className="text-amber-600 ml-2">
                    (Fallback parsing)
                  </span>
                )}
              </p>
            )}
          </div>
          
          {/* Actions */}
          {structuredData && (
            <div className="flex items-center gap-1">
              <button
                onClick={handleCopy}
                className={`p-2 rounded-lg transition-colors ${
                  copied ? 'text-emerald-600 bg-emerald-50' : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                }`}
                title="Copy as text"
              >
                {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
              </button>
              <button
                onClick={handleExportJSON}
                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-50 rounded-lg transition-colors"
                title="Export JSON"
              >
                <FileJson className="w-4 h-4" />
              </button>
              <button
                onClick={handleExportCSV}
                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-50 rounded-lg transition-colors"
                title="Export CSV"
              >
                <FileSpreadsheet className="w-4 h-4" />
              </button>
              {onSaveAsTemplate && (
                <button
                  onClick={onSaveAsTemplate}
                  className="ml-2 px-3 py-1.5 text-xs font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                >
                  Save as Template
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Content - Side by Side */}
      <div className="flex-1 flex flex-col lg:flex-row min-h-0 overflow-hidden">
        {/* Left Panel - Image */}
        {imagePreview && (
          <div className="lg:w-1/2 border-b lg:border-b-0 lg:border-r border-gray-100 flex flex-col">
            {/* Image Zoom Controls */}
            <div className="px-4 py-2 border-b border-gray-50 flex items-center justify-between bg-gray-50/50">
              <span className="text-xs font-medium text-gray-500">Original Form</span>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setImageZoom(Math.max(0.5, imageZoom - 0.25))}
                  className="p-1 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded transition-colors"
                  disabled={imageZoom <= 0.5}
                >
                  <ZoomOut className="w-4 h-4" />
                </button>
                <span className="text-xs text-gray-500 w-12 text-center">
                  {Math.round(imageZoom * 100)}%
                </span>
                <button
                  onClick={() => setImageZoom(Math.min(3, imageZoom + 0.25))}
                  className="p-1 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded transition-colors"
                  disabled={imageZoom >= 3}
                >
                  <ZoomIn className="w-4 h-4" />
                </button>
              </div>
            </div>
            
            {/* Image Container */}
            <div className="flex-1 overflow-auto p-4 bg-gray-50/30">
              <div 
                className="transition-transform origin-top-left"
                style={{ transform: `scale(${imageZoom})` }}
              >
                <img
                  src={imagePreview}
                  alt="Form preview"
                  className="max-w-full rounded-lg shadow-sm border border-gray-200"
                />
              </div>
            </div>
          </div>
        )}

        {/* Right Panel - Extracted Data */}
        <div className={`flex-1 flex flex-col min-h-0 ${!imagePreview ? 'w-full' : 'lg:w-1/2'}`}>
          {structuredData ? (
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
              {/* Sections */}
              {structuredData.sections.map((section, sectionIndex) => (
                <SectionCard
                  key={sectionIndex}
                  section={section}
                  sectionIndex={sectionIndex}
                  isExpanded={expandedSections.has(sectionIndex)}
                  onToggle={() => toggleSection(sectionIndex)}
                  editingField={editingField}
                  editValue={editValue}
                  onStartEdit={(fieldIndex, value) => startEditing(sectionIndex, fieldIndex, value)}
                  onCancelEdit={cancelEditing}
                  onSaveEdit={saveEditing}
                  onEditValueChange={setEditValue}
                  canEdit={!!onFieldEdit}
                />
              ))}

              {/* Checkboxes */}
              {structuredData.checkboxes.length > 0 && (
                <div className="bg-gray-50 rounded-xl overflow-hidden">
                  <div className="px-4 py-3 bg-gray-100/80 border-b border-gray-200/50 flex items-center gap-2">
                    <CheckSquare className="w-4 h-4 text-gray-500" />
                    <span className="text-sm font-medium text-gray-700">Checkboxes</span>
                  </div>
                  <div className="p-4 space-y-2">
                    {structuredData.checkboxes.map((cb, idx) => (
                      <CheckboxItem key={idx} checkbox={cb} />
                    ))}
                  </div>
                </div>
              )}

              {/* Tables */}
              {structuredData.tables.map((table, idx) => (
                <TableCard key={idx} table={table} index={idx} />
              ))}

              {/* Parsing Warning */}
              {!parsingSuccessful && (
                <div className="flex items-start gap-3 p-4 bg-amber-50 rounded-xl border border-amber-100">
                  <AlertCircle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-amber-800">Fallback Parsing Used</p>
                    <p className="text-xs text-amber-600 mt-1">
                      The structured JSON could not be parsed. Data was extracted using pattern matching, which may be less accurate.
                    </p>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center p-4">
              <div className="text-center">
                <FileText className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                <p className="text-sm text-gray-500">No structured data extracted yet</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Section Card Component
function SectionCard({
  section,
  sectionIndex,
  isExpanded,
  onToggle,
  editingField,
  editValue,
  onStartEdit,
  onCancelEdit,
  onSaveEdit,
  onEditValueChange,
  canEdit
}: {
  section: ExtractedSection
  sectionIndex: number
  isExpanded: boolean
  onToggle: () => void
  editingField: { section: number; field: number } | null
  editValue: string
  onStartEdit: (fieldIndex: number, value: string) => void
  onCancelEdit: () => void
  onSaveEdit: () => void
  onEditValueChange: (value: string) => void
  canEdit: boolean
}) {
  const filledCount = section.fields.filter(f => f.value.trim()).length
  
  return (
    <div className="bg-gray-50 rounded-xl overflow-hidden">
      {/* Section Header */}
      <button
        onClick={onToggle}
        className="w-full px-4 py-3 bg-gray-100/80 border-b border-gray-200/50 flex items-center justify-between hover:bg-gray-100 transition-colors"
      >
        <div className="flex items-center gap-2">
          <FormInput className="w-4 h-4 text-gray-500" />
          <span className="text-sm font-medium text-gray-700">
            {section.name || 'General Fields'}
          </span>
          <span className="text-xs text-gray-400">
            ({filledCount}/{section.fields.length})
          </span>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-4 h-4 text-gray-400" />
        ) : (
          <ChevronDown className="w-4 h-4 text-gray-400" />
        )}
      </button>

      {/* Fields */}
      {isExpanded && (
        <div className="divide-y divide-gray-100">
          {section.fields.map((field, fieldIndex) => {
            const isEditing = editingField?.section === sectionIndex && editingField?.field === fieldIndex
            
            return (
              <div 
                key={fieldIndex} 
                className={`px-4 py-3 flex items-start gap-3 ${
                  !field.value.trim() ? 'bg-red-50/30' : ''
                }`}
                dir="auto"
              >
                <div className="flex-1 min-w-0">
                  <div className="text-xs font-medium text-gray-500 mb-1">
                    {field.label}
                    <span className="text-gray-300 ml-1">({field.type})</span>
                  </div>
                  
                  {isEditing ? (
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={editValue}
                        onChange={(e) => onEditValueChange(e.target.value)}
                        className="flex-1 px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent"
                        autoFocus
                        dir="auto"
                      />
                      <button
                        onClick={onSaveEdit}
                        className="p-1 text-emerald-600 hover:bg-emerald-50 rounded"
                      >
                        <Save className="w-4 h-4" />
                      </button>
                      <button
                        onClick={onCancelEdit}
                        className="p-1 text-gray-400 hover:bg-gray-100 rounded"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 group">
                      <span className={`text-sm ${
                        field.value.trim() 
                          ? 'text-gray-900 font-medium' 
                          : 'text-gray-300 italic'
                      }`}>
                        {field.value.trim() || '[empty]'}
                      </span>
                      {canEdit && (
                        <button
                          onClick={() => onStartEdit(fieldIndex, field.value)}
                          className="p-1 text-gray-300 hover:text-gray-600 opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          <Edit2 className="w-3 h-3" />
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// Checkbox Item Component
function CheckboxItem({ checkbox }: { checkbox: ExtractedCheckbox }) {
  return (
    <div className="flex items-center gap-2" dir="auto">
      <div className={`w-4 h-4 rounded border ${
        checkbox.checked 
          ? 'bg-emerald-500 border-emerald-500' 
          : 'bg-white border-gray-300'
      } flex items-center justify-center`}>
        {checkbox.checked && <Check className="w-3 h-3 text-white" />}
      </div>
      <span className="text-sm text-gray-700">{checkbox.label}</span>
    </div>
  )
}

// Table Card Component
function TableCard({ table, index }: { table: ExtractedTable; index: number }) {
  if (table.headers.length === 0) return null
  
  return (
    <div className="bg-gray-50 rounded-xl overflow-hidden">
      <div className="px-4 py-3 bg-gray-100/80 border-b border-gray-200/50 flex items-center gap-2">
        <Table className="w-4 h-4 text-gray-500" />
        <span className="text-sm font-medium text-gray-700">Table {index + 1}</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm" dir="auto">
          <thead>
            <tr className="bg-gray-100/50">
              {table.headers.map((header, idx) => (
                <th 
                  key={idx} 
                  className="px-4 py-2 text-right font-medium text-gray-700 border-b border-gray-200/50 first:text-left"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {table.rows.map((row, rowIdx) => (
              <tr key={rowIdx} className="hover:bg-gray-50 transition-colors">
                {table.headers.map((_, colIdx) => (
                  <td 
                    key={colIdx} 
                    className="px-4 py-2 text-right text-gray-900 first:text-left first:font-medium"
                  >
                    {row[colIdx] || ''}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
