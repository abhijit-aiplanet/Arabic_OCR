'use client'

import { useMemo, useState } from 'react'
import { Upload, Save, X, FileDown, Plus, Trash2, ChevronDown, ChevronUp, GripVertical } from 'lucide-react'
import toast from 'react-hot-toast'
import type { ContentType, CreateTemplateRequest, OCRTemplate } from '@/lib/api'

type BuilderMode = 'create' | 'import'
type FieldType = 'text' | 'number' | 'date' | 'checkbox'

interface FieldDefinition {
  id: string
  label: string
  type: FieldType
  required: boolean
}

interface SectionDefinition {
  id: string
  name: string
  fields: FieldDefinition[]
  isExpanded: boolean
}

interface TemplateBuilderProps {
  onCancel: () => void
  onCreate: (payload: CreateTemplateRequest) => Promise<void>
  initialFieldSchema?: { sections: Array<{ name: string; fields: Array<{ label: string; type: string }> }> }
}

const CONTENT_TYPES: Array<Exclude<ContentType, 'auto'>> = [
  'form',
  'document',
  'receipt',
  'invoice',
  'table',
  'id_card',
  'certificate',
  'handwritten',
  'mixed',
  'unknown'
]

const FIELD_TYPES: { value: FieldType; label: string }[] = [
  { value: 'text', label: 'Text' },
  { value: 'number', label: 'Number' },
  { value: 'date', label: 'Date' },
  { value: 'checkbox', label: 'Checkbox' }
]

function generateId() {
  return Math.random().toString(36).substring(2, 9)
}

export default function TemplateBuilder({ onCancel, onCreate, initialFieldSchema }: TemplateBuilderProps) {
  const [mode, setMode] = useState<BuilderMode>('create')
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [contentType, setContentType] = useState<Exclude<ContentType, 'auto'>>('form')
  const [language, setLanguage] = useState<'ar' | 'en' | 'mixed'>('ar')
  const [customPrompt, setCustomPrompt] = useState('')
  const [keywords, setKeywords] = useState('')
  const [isPublic, setIsPublic] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  
  // Field schema state
  const [sections, setSections] = useState<SectionDefinition[]>(() => {
    if (initialFieldSchema?.sections) {
      return initialFieldSchema.sections.map(s => ({
        id: generateId(),
        name: s.name || '',
        isExpanded: true,
        fields: s.fields.map(f => ({
          id: generateId(),
          label: f.label,
          type: (f.type || 'text') as FieldType,
          required: false
        }))
      }))
    }
    return [{
      id: generateId(),
      name: '',
      isExpanded: true,
      fields: []
    }]
  })
  const [showFieldSchema, setShowFieldSchema] = useState(contentType === 'form' || !!initialFieldSchema)

  const keywordsList = useMemo(() => {
    return keywords.split(',').map((k) => k.trim()).filter(Boolean)
  }, [keywords])

  // Build field schema for API
  const fieldSchema = useMemo(() => {
    const schemaSections = sections
      .filter(s => s.fields.length > 0)
      .map(s => ({
        name: s.name.trim() || 'General',
        fields: s.fields.map(f => ({
          label: f.label,
          type: f.type,
          required: f.required
        }))
      }))
    
    return schemaSections.length > 0 ? { sections: schemaSections } : null
  }, [sections])

  // Section management
  const addSection = () => {
    setSections([...sections, {
      id: generateId(),
      name: '',
      isExpanded: true,
      fields: []
    }])
  }

  const removeSection = (sectionId: string) => {
    if (sections.length <= 1) {
      toast.error('At least one section is required')
      return
    }
    setSections(sections.filter(s => s.id !== sectionId))
  }

  const updateSection = (sectionId: string, updates: Partial<SectionDefinition>) => {
    setSections(sections.map(s => s.id === sectionId ? { ...s, ...updates } : s))
  }

  const toggleSection = (sectionId: string) => {
    setSections(sections.map(s => s.id === sectionId ? { ...s, isExpanded: !s.isExpanded } : s))
  }

  // Field management
  const addField = (sectionId: string) => {
    setSections(sections.map(s => {
      if (s.id !== sectionId) return s
      return {
        ...s,
        fields: [...s.fields, {
          id: generateId(),
          label: '',
          type: 'text' as FieldType,
          required: false
        }]
      }
    }))
  }

  const removeField = (sectionId: string, fieldId: string) => {
    setSections(sections.map(s => {
      if (s.id !== sectionId) return s
      return {
        ...s,
        fields: s.fields.filter(f => f.id !== fieldId)
      }
    }))
  }

  const updateField = (sectionId: string, fieldId: string, updates: Partial<FieldDefinition>) => {
    setSections(sections.map(s => {
      if (s.id !== sectionId) return s
      return {
        ...s,
        fields: s.fields.map(f => f.id === fieldId ? { ...f, ...updates } : f)
      }
    }))
  }

  const handleImportJson = async (file: File) => {
    try {
      const text = await file.text()
      const data = JSON.parse(text) as Partial<OCRTemplate> & Partial<CreateTemplateRequest>

      setName(String(data.name || '').trim())
      setDescription(String(data.description || '').trim())
      setContentType((data.content_type as any) || 'form')
      setLanguage((data.language as any) || 'ar')
      setCustomPrompt(String(data.custom_prompt || '').trim())
      setKeywords((Array.isArray(data.keywords) ? data.keywords.join(', ') : String(data.keywords || '')).trim())
      setIsPublic(Boolean(data.is_public ?? false))

      // Import field schema if present
      const importedSections = data.sections as any
      if (importedSections?.sections && Array.isArray(importedSections.sections)) {
        setSections(importedSections.sections.map((s: any) => ({
          id: generateId(),
          name: s.name || '',
          isExpanded: true,
          fields: (s.fields || []).map((f: any) => ({
            id: generateId(),
            label: f.label || '',
            type: (f.type || 'text') as FieldType,
            required: Boolean(f.required)
          }))
        })))
        setShowFieldSchema(true)
      }

      toast.success('Template JSON imported')
    } catch (e: any) {
      toast.error(e?.message || 'Failed to import template JSON')
    }
  }

  const handleDownloadJson = () => {
    const payload: CreateTemplateRequest = {
      name: name.trim() || 'Untitled Template',
      description: description.trim() || undefined,
      content_type: contentType,
      language,
      custom_prompt: customPrompt.trim() || undefined,
      keywords: keywordsList.length ? keywordsList : undefined,
      is_public: isPublic,
      sections: fieldSchema || undefined
    }

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${payload.name.replace(/\s+/g, '_')}.template.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const handleSave = async () => {
    if (!name.trim()) {
      toast.error('Template name is required')
      return
    }

    // Validate field schema if enabled
    if (showFieldSchema) {
      const hasEmptyLabels = sections.some(s => 
        s.fields.some(f => !f.label.trim())
      )
      if (hasEmptyLabels) {
        toast.error('All fields must have a label')
        return
      }
    }

    setIsSaving(true)
    try {
      await onCreate({
        name: name.trim(),
        description: description.trim() || undefined,
        content_type: contentType,
        language,
        custom_prompt: customPrompt.trim() || undefined,
        keywords: keywordsList.length ? keywordsList : undefined,
        is_public: isPublic,
        sections: showFieldSchema ? fieldSchema || undefined : undefined
      })
      toast.success('Template created')
      onCancel()
    } catch (e: any) {
      toast.error(e?.message || 'Failed to create template')
    } finally {
      setIsSaving(false)
    }
  }

  const totalFields = sections.reduce((sum, s) => sum + s.fields.length, 0)

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden max-h-[90vh] flex flex-col">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between flex-shrink-0">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Template Builder</h3>
          <p className="text-sm text-gray-500">Define expected fields for structured form extraction</p>
        </div>
        <button onClick={onCancel} className="p-2 hover:bg-gray-100 rounded-lg">
          <X className="w-5 h-5 text-gray-500" />
        </button>
      </div>

      {/* Mode Toggle */}
      <div className="px-6 py-3 border-b border-gray-100 bg-gray-50 flex items-center gap-2 flex-shrink-0">
        <button
          onClick={() => setMode('create')}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            mode === 'create' ? 'bg-gray-900 text-white' : 'bg-white text-gray-700 border border-gray-200 hover:bg-gray-50'
          }`}
        >
          Create
        </button>
        <button
          onClick={() => setMode('import')}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            mode === 'import' ? 'bg-gray-900 text-white' : 'bg-white text-gray-700 border border-gray-200 hover:bg-gray-50'
          }`}
        >
          Import JSON
        </button>

        <div className="ml-auto">
          <button
            onClick={handleDownloadJson}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium bg-white text-gray-700 border border-gray-200 hover:bg-gray-50 transition-colors"
          >
            <FileDown className="w-4 h-4" />
            Export
          </button>
        </div>
      </div>

      {/* Content - Scrollable */}
      <div className="flex-1 overflow-y-auto p-6 space-y-5">
        {/* Import Section */}
        {mode === 'import' && (
          <div className="border border-dashed border-gray-300 rounded-lg p-4 bg-gray-50">
            <label className="block text-sm font-medium text-gray-700 mb-2">Import template JSON</label>
            <input
              type="file"
              accept="application/json"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) void handleImportJson(f)
              }}
              className="block w-full text-sm text-gray-700"
            />
          </div>
        )}

        {/* Basic Info */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Template Name *</label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent"
              placeholder="e.g., Saudi ID Card"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Content Type</label>
            <select
              value={contentType}
              onChange={(e) => {
                const newType = e.target.value as any
                setContentType(newType)
                if (newType === 'form' || newType === 'id_card') {
                  setShowFieldSchema(true)
                }
              }}
              className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent"
            >
              {CONTENT_TYPES.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
          <input
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent"
            placeholder="Brief description of this template"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Language</label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value as any)}
              className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent"
            >
              <option value="ar">Arabic</option>
              <option value="en">English</option>
              <option value="mixed">Mixed</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Keywords</label>
            <input
              value={keywords}
              onChange={(e) => setKeywords(e.target.value)}
              className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent"
              placeholder="comma-separated"
            />
          </div>
        </div>

        {/* Field Schema Toggle */}
        <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
          <div>
            <p className="text-sm font-medium text-gray-900">Field Schema</p>
            <p className="text-xs text-gray-500">Define expected fields for better extraction accuracy</p>
          </div>
          <button
            onClick={() => setShowFieldSchema(!showFieldSchema)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
              showFieldSchema ? 'bg-gray-900' : 'bg-gray-200'
            }`}
          >
            <span
              className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                showFieldSchema ? 'translate-x-6' : 'translate-x-1'
              }`}
            />
          </button>
        </div>

        {/* Field Schema Builder */}
        {showFieldSchema && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <p className="text-sm font-medium text-gray-700">
                Sections & Fields
                <span className="text-gray-400 font-normal ml-2">({totalFields} fields)</span>
              </p>
              <button
                onClick={addSection}
                className="flex items-center gap-1 px-2 py-1 text-xs font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition-colors"
              >
                <Plus className="w-3 h-3" />
                Add Section
              </button>
            </div>

            {sections.map((section, sectionIndex) => (
              <div key={section.id} className="border border-gray-200 rounded-lg overflow-hidden">
                {/* Section Header */}
                <div className="px-4 py-3 bg-gray-50 flex items-center gap-3">
                  <button
                    onClick={() => toggleSection(section.id)}
                    className="p-1 text-gray-400 hover:text-gray-600"
                  >
                    {section.isExpanded ? (
                      <ChevronUp className="w-4 h-4" />
                    ) : (
                      <ChevronDown className="w-4 h-4" />
                    )}
                  </button>
                  <input
                    value={section.name}
                    onChange={(e) => updateSection(section.id, { name: e.target.value })}
                    className="flex-1 px-2 py-1 text-sm border border-gray-200 rounded focus:outline-none focus:ring-1 focus:ring-gray-900"
                    placeholder={`Section ${sectionIndex + 1} name (optional)`}
                  />
                  <span className="text-xs text-gray-400">{section.fields.length} fields</span>
                  {sections.length > 1 && (
                    <button
                      onClick={() => removeSection(section.id)}
                      className="p-1 text-gray-400 hover:text-red-500 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>

                {/* Section Fields */}
                {section.isExpanded && (
                  <div className="p-4 space-y-2">
                    {section.fields.map((field, fieldIndex) => (
                      <div key={field.id} className="flex items-center gap-2 group">
                        <GripVertical className="w-4 h-4 text-gray-300 flex-shrink-0" />
                        <input
                          value={field.label}
                          onChange={(e) => updateField(section.id, field.id, { label: e.target.value })}
                          className="flex-1 px-2 py-1.5 text-sm border border-gray-200 rounded focus:outline-none focus:ring-1 focus:ring-gray-900"
                          placeholder="Field label (Arabic or English)"
                          dir="auto"
                        />
                        <select
                          value={field.type}
                          onChange={(e) => updateField(section.id, field.id, { type: e.target.value as FieldType })}
                          className="px-2 py-1.5 text-sm border border-gray-200 rounded focus:outline-none focus:ring-1 focus:ring-gray-900"
                        >
                          {FIELD_TYPES.map(t => (
                            <option key={t.value} value={t.value}>{t.label}</option>
                          ))}
                        </select>
                        <label className="flex items-center gap-1 text-xs text-gray-500">
                          <input
                            type="checkbox"
                            checked={field.required}
                            onChange={(e) => updateField(section.id, field.id, { required: e.target.checked })}
                            className="w-3 h-3"
                          />
                          Required
                        </label>
                        <button
                          onClick={() => removeField(section.id, field.id)}
                          className="p-1 text-gray-300 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-all"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    ))}
                    
                    <button
                      onClick={() => addField(section.id)}
                      className="w-full py-2 border border-dashed border-gray-200 rounded-lg text-sm text-gray-500 hover:text-gray-700 hover:border-gray-300 hover:bg-gray-50 transition-colors flex items-center justify-center gap-1"
                    >
                      <Plus className="w-4 h-4" />
                      Add Field
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Custom Prompt (Optional) */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Custom Prompt <span className="text-gray-400 font-normal">(optional override)</span>
          </label>
          <textarea
            value={customPrompt}
            onChange={(e) => setCustomPrompt(e.target.value)}
            className="w-full min-h-[100px] px-3 py-2 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent text-sm"
            placeholder="Leave empty to use auto-generated prompt based on field schema..."
          />
        </div>

        {/* Options */}
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer">
            <input
              type="checkbox"
              checked={isPublic}
              onChange={(e) => setIsPublic(e.target.checked)}
              className="w-4 h-4 rounded border-gray-300"
            />
            Make template public
          </label>
        </div>
      </div>

      {/* Footer */}
      <div className="px-6 py-4 border-t border-gray-200 bg-gray-50 flex items-center justify-end gap-3 flex-shrink-0">
        <button
          onClick={onCancel}
          className="px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
        >
          Cancel
        </button>
        <button
          onClick={handleSave}
          disabled={isSaving}
          className="px-4 py-2 text-sm font-medium bg-gray-900 text-white rounded-lg hover:bg-gray-800 disabled:opacity-50 transition-colors flex items-center gap-2"
        >
          {isSaving ? (
            <>
              <Upload className="w-4 h-4 animate-pulse" />
              Saving...
            </>
          ) : (
            <>
              <Save className="w-4 h-4" />
              Save Template
            </>
          )}
        </button>
      </div>
    </div>
  )
}
