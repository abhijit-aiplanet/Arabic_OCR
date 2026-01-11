'use client'

import { useState, useMemo } from 'react'
import { 
  Copy, Check, Download, FileJson, FileSpreadsheet, 
  ChevronDown, ChevronUp, Edit2, Save, X, ZoomIn, ZoomOut,
  Table, FormInput, CheckSquare, FileText, AlertCircle,
  Maximize2, Minimize2, Eye, EyeOff
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
  const [editingField, setEditingField] = useState<{ section: number; field: number } | null>(null)
  const [editValue, setEditValue] = useState('')
  const [copied, setCopied] = useState(false)
  const [showEmptyFields, setShowEmptyFields] = useState(true)
  const [fullscreenImage, setFullscreenImage] = useState(false)
  const [viewMode, setViewMode] = useState<'cards' | 'table'>('cards')

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

  // All fields flattened for table view
  const allFields = useMemo(() => {
    if (!structuredData) return []
    return structuredData.sections.flatMap((section, sIdx) => 
      section.fields.map((field, fIdx) => ({
        ...field,
        sectionName: section.name || 'General',
        sectionIndex: sIdx,
        fieldIndex: fIdx
      }))
    )
  }, [structuredData])

  const visibleFields = useMemo(() => {
    if (showEmptyFields) return allFields
    return allFields.filter(f => f.value.trim())
  }, [allFields, showEmptyFields])

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
          <div className="relative w-20 h-20 mx-auto mb-6">
            <div className="w-20 h-20 border-3 border-gray-100 rounded-full"></div>
            <div className="absolute inset-0 w-20 h-20 border-3 border-transparent border-t-gray-900 rounded-full animate-spin"></div>
            <FormInput className="absolute inset-0 m-auto w-8 h-8 text-gray-400" />
          </div>
          <p className="text-lg font-semibold text-gray-900">Extracting Form Data</p>
          <p className="text-sm text-gray-500 mt-2 max-w-xs mx-auto">
            Analyzing document structure and extracting field values...
          </p>
        </div>
      </div>
    )
  }

  // Empty state - no data and no image
  if (!structuredData && !imagePreview) {
    return (
      <div className="bg-white rounded-2xl border border-gray-100 h-full flex items-center justify-center min-h-[500px]">
        <div className="text-center px-6">
          <div className="w-20 h-20 bg-gradient-to-br from-gray-50 to-gray-100 rounded-2xl flex items-center justify-center mx-auto mb-6">
            <FormInput className="w-10 h-10 text-gray-400" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Structured Extraction</h3>
          <p className="text-sm text-gray-500 max-w-[280px] mx-auto leading-relaxed">
            Upload a form image to automatically extract labels, values, and structured data
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-2xl border border-gray-100 overflow-hidden h-full flex flex-col">
      {/* Header */}
      <div className="px-5 py-4 border-b border-gray-100 flex-shrink-0 bg-white">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">
              {structuredData?.form_title || 'Extracted Form Data'}
            </h2>
            {structuredData && (
              <div className="flex items-center gap-3 mt-1">
                <span className="text-sm text-gray-500">
                  <span className="font-medium text-gray-900">{filledFields}</span> of {totalFields} fields filled
                </span>
                {!parsingSuccessful && (
                  <span className="text-xs px-2 py-0.5 bg-amber-100 text-amber-700 rounded-full">
                    Fallback parsing
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
        
        {/* Toolbar */}
        {structuredData && (
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {/* View Mode Toggle */}
              <div className="flex items-center bg-gray-100 rounded-lg p-0.5">
                <button
                  onClick={() => setViewMode('cards')}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                    viewMode === 'cards' 
                      ? 'bg-white text-gray-900 shadow-sm' 
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  Cards
                </button>
                <button
                  onClick={() => setViewMode('table')}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                    viewMode === 'table' 
                      ? 'bg-white text-gray-900 shadow-sm' 
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  Table
                </button>
              </div>
              
              {/* Show/Hide Empty */}
              <button
                onClick={() => setShowEmptyFields(!showEmptyFields)}
                className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                  showEmptyFields 
                    ? 'bg-gray-100 text-gray-700' 
                    : 'bg-gray-900 text-white'
                }`}
              >
                {showEmptyFields ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
                {showEmptyFields ? 'Showing empty' : 'Hiding empty'}
              </button>
            </div>
            
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
                  className="ml-2 px-3 py-1.5 text-xs font-medium text-white bg-gray-900 hover:bg-gray-800 rounded-lg transition-colors"
                >
                  Save as Template
                </button>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Content - Side by Side */}
      <div className="flex-1 flex flex-col lg:flex-row min-h-0 overflow-hidden">
        {/* Left Panel - Image */}
        {imagePreview && (
          <div className={`${fullscreenImage ? 'fixed inset-0 z-50 bg-black' : 'lg:w-2/5 border-b lg:border-b-0 lg:border-r border-gray-100'} flex flex-col`}>
            {/* Image Controls */}
            <div className={`px-4 py-2 flex items-center justify-between ${fullscreenImage ? 'bg-black/80' : 'bg-gray-50 border-b border-gray-100'}`}>
              <span className={`text-xs font-medium ${fullscreenImage ? 'text-white' : 'text-gray-600'}`}>
                Original Document
              </span>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setImageZoom(Math.max(0.25, imageZoom - 0.25))}
                  className={`p-1.5 rounded transition-colors ${fullscreenImage ? 'text-white hover:bg-white/10' : 'text-gray-500 hover:bg-gray-100'}`}
                >
                  <ZoomOut className="w-4 h-4" />
                </button>
                <span className={`text-xs w-14 text-center ${fullscreenImage ? 'text-white' : 'text-gray-500'}`}>
                  {Math.round(imageZoom * 100)}%
                </span>
                <button
                  onClick={() => setImageZoom(Math.min(4, imageZoom + 0.25))}
                  className={`p-1.5 rounded transition-colors ${fullscreenImage ? 'text-white hover:bg-white/10' : 'text-gray-500 hover:bg-gray-100'}`}
                >
                  <ZoomIn className="w-4 h-4" />
                </button>
                <div className={`w-px h-4 mx-1 ${fullscreenImage ? 'bg-white/20' : 'bg-gray-200'}`} />
                <button
                  onClick={() => setFullscreenImage(!fullscreenImage)}
                  className={`p-1.5 rounded transition-colors ${fullscreenImage ? 'text-white hover:bg-white/10' : 'text-gray-500 hover:bg-gray-100'}`}
                >
                  {fullscreenImage ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
                </button>
              </div>
            </div>
            
            {/* Image Container */}
            <div className={`flex-1 overflow-auto ${fullscreenImage ? 'bg-black p-8' : 'p-4 bg-gray-50/50'}`}>
              <div 
                className="transition-transform origin-center inline-block"
                style={{ transform: `scale(${imageZoom})` }}
              >
                <img
                  src={imagePreview}
                  alt="Form preview"
                  className="max-w-full rounded-lg shadow-lg"
                />
              </div>
            </div>
          </div>
        )}

        {/* Right Panel - Extracted Data */}
        <div className={`flex-1 flex flex-col min-h-0 ${!imagePreview ? 'w-full' : 'lg:w-3/5'}`}>
          {structuredData ? (
            <div className="flex-1 overflow-y-auto">
              {viewMode === 'cards' ? (
                <CardsView 
                  structuredData={structuredData}
                  showEmptyFields={showEmptyFields}
                  editingField={editingField}
                  editValue={editValue}
                  onStartEdit={startEditing}
                  onCancelEdit={cancelEditing}
                  onSaveEdit={saveEditing}
                  onEditValueChange={setEditValue}
                  canEdit={!!onFieldEdit}
                />
              ) : (
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

              {/* Checkboxes */}
              {structuredData.checkboxes.length > 0 && (
                <div className="p-4 border-t border-gray-100">
                  <h3 className="text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
                    <CheckSquare className="w-4 h-4" />
                    Checkboxes
                  </h3>
                  <div className="grid grid-cols-2 gap-2">
                    {structuredData.checkboxes.map((cb, idx) => (
                      <CheckboxItem key={idx} checkbox={cb} />
                    ))}
                  </div>
                </div>
              )}

              {/* Tables */}
              {structuredData.tables.map((table, idx) => (
                <div key={idx} className="p-4 border-t border-gray-100">
                  <TableCard table={table} index={idx} />
                </div>
              ))}

              {/* Parsing Warning */}
              {!parsingSuccessful && (
                <div className="m-4 flex items-start gap-3 p-4 bg-amber-50 rounded-xl border border-amber-100">
                  <AlertCircle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-amber-800">Using Fallback Parsing</p>
                    <p className="text-xs text-amber-600 mt-1">
                      The structured format could not be fully parsed. Some fields may be missing or inaccurate.
                    </p>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center p-8">
              <div className="text-center">
                <FileText className="w-16 h-16 text-gray-200 mx-auto mb-4" />
                <p className="text-gray-500">Click "Extract Form Data" to begin</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Cards View Component
function CardsView({
  structuredData,
  showEmptyFields,
  editingField,
  editValue,
  onStartEdit,
  onCancelEdit,
  onSaveEdit,
  onEditValueChange,
  canEdit
}: {
  structuredData: StructuredExtraction
  showEmptyFields: boolean
  editingField: { section: number; field: number } | null
  editValue: string
  onStartEdit: (sectionIndex: number, fieldIndex: number, value: string) => void
  onCancelEdit: () => void
  onSaveEdit: () => void
  onEditValueChange: (value: string) => void
  canEdit: boolean
}) {
  return (
    <div className="p-4 space-y-4">
      {structuredData.sections.map((section, sectionIndex) => {
        const visibleFields = showEmptyFields 
          ? section.fields 
          : section.fields.filter(f => f.value.trim())
        
        if (visibleFields.length === 0) return null
        
        return (
          <div key={sectionIndex} className="space-y-3">
            {/* Section Header */}
            <div className="flex items-center gap-2">
              <div className="w-1 h-5 bg-gray-900 rounded-full" />
              <h3 className="text-sm font-semibold text-gray-900">
                {section.name || 'General Fields'}
              </h3>
              <span className="text-xs text-gray-400">
                {section.fields.filter(f => f.value.trim()).length}/{section.fields.length}
              </span>
            </div>
            
            {/* Fields Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {visibleFields.map((field, fieldIndex) => {
                const actualFieldIndex = section.fields.indexOf(field)
                const isEditing = editingField?.section === sectionIndex && editingField?.field === actualFieldIndex
                const isEmpty = !field.value.trim()
                
                return (
                  <div 
                    key={fieldIndex} 
                    className={`rounded-xl p-4 transition-all ${
                      isEmpty 
                        ? 'bg-gray-50 border border-dashed border-gray-200' 
                        : 'bg-white border border-gray-200 shadow-sm hover:shadow-md'
                    }`}
                    dir="auto"
                  >
                    {/* Field Label */}
                    <div className="flex items-start justify-between mb-2">
                      <span className="text-xs font-medium text-gray-500 leading-tight">
                        {field.label}
                      </span>
                      <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                        field.type === 'date' ? 'bg-blue-50 text-blue-600' :
                        field.type === 'number' ? 'bg-green-50 text-green-600' :
                        'bg-gray-50 text-gray-500'
                      }`}>
                        {field.type}
                      </span>
                    </div>
                    
                    {/* Field Value */}
                    {isEditing ? (
                      <div className="flex items-center gap-2">
                        <input
                          type="text"
                          value={editValue}
                          onChange={(e) => onEditValueChange(e.target.value)}
                          className="flex-1 px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent"
                          autoFocus
                          dir="auto"
                        />
                        <button
                          onClick={onSaveEdit}
                          className="p-2 text-white bg-gray-900 hover:bg-gray-800 rounded-lg"
                        >
                          <Save className="w-4 h-4" />
                        </button>
                        <button
                          onClick={onCancelEdit}
                          className="p-2 text-gray-500 hover:bg-gray-100 rounded-lg"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    ) : (
                      <div className="group flex items-start gap-2">
                        <p className={`flex-1 text-base leading-relaxed ${
                          isEmpty 
                            ? 'text-gray-300 italic text-sm' 
                            : 'text-gray-900 font-medium'
                        }`}>
                          {isEmpty ? 'Not filled' : field.value}
                        </p>
                        {canEdit && (
                          <button
                            onClick={() => onStartEdit(sectionIndex, actualFieldIndex, field.value)}
                            className="p-1.5 text-gray-300 hover:text-gray-600 hover:bg-gray-100 rounded opacity-0 group-hover:opacity-100 transition-all"
                          >
                            <Edit2 className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        )
      })}
    </div>
  )
}

// Table View Component
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
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="bg-gray-50 sticky top-0">
          <tr>
            <th className="px-4 py-3 text-left font-medium text-gray-600 border-b border-gray-200">Field</th>
            <th className="px-4 py-3 text-left font-medium text-gray-600 border-b border-gray-200">Value</th>
            <th className="px-4 py-3 text-left font-medium text-gray-600 border-b border-gray-200 w-20">Type</th>
            {canEdit && <th className="px-4 py-3 w-12 border-b border-gray-200"></th>}
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-100">
          {fields.map((field, idx) => {
            const isEditing = editingField?.section === field.sectionIndex && editingField?.field === field.fieldIndex
            const isEmpty = !field.value.trim()
            
            return (
              <tr key={idx} className={`${isEmpty ? 'bg-gray-50/50' : 'hover:bg-gray-50'} transition-colors`}>
                <td className="px-4 py-3" dir="auto">
                  <span className="font-medium text-gray-900">{field.label}</span>
                  <span className="text-xs text-gray-400 ml-2">{field.sectionName}</span>
                </td>
                <td className="px-4 py-3" dir="auto">
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
                    <span className={isEmpty ? 'text-gray-300 italic' : 'text-gray-900'}>
                      {isEmpty ? 'Not filled' : field.value}
                    </span>
                  )}
                </td>
                <td className="px-4 py-3">
                  <span className={`text-xs px-2 py-1 rounded ${
                    field.type === 'date' ? 'bg-blue-50 text-blue-600' :
                    field.type === 'number' ? 'bg-green-50 text-green-600' :
                    'bg-gray-100 text-gray-600'
                  }`}>
                    {field.type}
                  </span>
                </td>
                {canEdit && (
                  <td className="px-4 py-3">
                    {!isEditing && (
                      <button
                        onClick={() => onStartEdit(field.sectionIndex, field.fieldIndex, field.value)}
                        className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded"
                      >
                        <Edit2 className="w-4 h-4" />
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

// Checkbox Item Component
function CheckboxItem({ checkbox }: { checkbox: ExtractedCheckbox }) {
  return (
    <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg" dir="auto">
      <div className={`w-5 h-5 rounded flex items-center justify-center ${
        checkbox.checked 
          ? 'bg-emerald-500' 
          : 'bg-white border-2 border-gray-300'
      }`}>
        {checkbox.checked && <Check className="w-3.5 h-3.5 text-white" />}
      </div>
      <span className="text-sm text-gray-700 flex-1">{checkbox.label}</span>
    </div>
  )
}

// Table Card Component
function TableCard({ table, index }: { table: ExtractedTable; index: number }) {
  if (table.headers.length === 0) return null
  
  return (
    <div>
      <h3 className="text-sm font-medium text-gray-700 mb-3 flex items-center gap-2">
        <Table className="w-4 h-4" />
        Table {index + 1}
      </h3>
      <div className="overflow-x-auto rounded-lg border border-gray-200">
        <table className="w-full text-sm" dir="auto">
          <thead>
            <tr className="bg-gray-50">
              {table.headers.map((header, idx) => (
                <th 
                  key={idx} 
                  className="px-4 py-2 text-right font-medium text-gray-700 border-b border-gray-200 first:text-left"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {table.rows.map((row, rowIdx) => (
              <tr key={rowIdx} className="hover:bg-gray-50">
                {table.headers.map((_, colIdx) => (
                  <td 
                    key={colIdx} 
                    className="px-4 py-2 text-right text-gray-900 first:text-left"
                  >
                    {row[colIdx] || '-'}
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
