'use client'

import { useMemo, useState } from 'react'
import { Upload, Save, X, FileDown } from 'lucide-react'
import toast from 'react-hot-toast'
import type { ContentType, CreateTemplateRequest, OCRTemplate } from '@/lib/api'

type BuilderMode = 'create' | 'import'

interface TemplateBuilderProps {
  onCancel: () => void
  onCreate: (payload: CreateTemplateRequest) => Promise<void>
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

export default function TemplateBuilder({ onCancel, onCreate }: TemplateBuilderProps) {
  const [mode, setMode] = useState<BuilderMode>('create')
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [contentType, setContentType] = useState<Exclude<ContentType, 'auto'>>('form')
  const [language, setLanguage] = useState<'ar' | 'en' | 'mixed'>('ar')
  const [customPrompt, setCustomPrompt] = useState('')
  const [keywords, setKeywords] = useState('')
  const [isPublic, setIsPublic] = useState(false)
  const [isSaving, setIsSaving] = useState(false)

  const keywordsList = useMemo(() => {
    return keywords
      .split(',')
      .map((k) => k.trim())
      .filter(Boolean)
  }, [keywords])

  const handleImportJson = async (file: File) => {
    try {
      const text = await file.text()
      const data = JSON.parse(text) as Partial<OCRTemplate> & Partial<CreateTemplateRequest>

      setName(String(data.name || '').trim())
      setDescription(String(data.description || '').trim())
      setContentType((data.content_type as any) || (data.content_type ? (data.content_type as any) : 'form'))
      setLanguage((data.language as any) || 'ar')
      setCustomPrompt(String((data.custom_prompt as any) || data.custom_prompt || '').trim())
      setKeywords((Array.isArray(data.keywords) ? data.keywords.join(', ') : String(data.keywords || '')).trim())
      setIsPublic(Boolean(data.is_public ?? false))

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
      is_public: isPublic
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
    if (!customPrompt.trim()) {
      toast.error('Custom prompt is required (keep it short but specific)')
      return
    }

    setIsSaving(true)
    try {
      await onCreate({
        name: name.trim(),
        description: description.trim() || undefined,
        content_type: contentType,
        language,
        custom_prompt: customPrompt.trim(),
        keywords: keywordsList.length ? keywordsList : undefined,
        is_public: isPublic
      })
      toast.success('Template created')
      onCancel()
    } catch (e: any) {
      toast.error(e?.message || 'Failed to create template')
    } finally {
      setIsSaving(false)
    }
  }

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Template Builder</h3>
          <p className="text-sm text-gray-600">Create or import a prompt template for consistent OCR output</p>
        </div>
        <button onClick={onCancel} className="p-2 hover:bg-gray-100 rounded-lg">
          <X className="w-5 h-5 text-gray-500" />
        </button>
      </div>

      <div className="px-6 py-4 border-b border-gray-200 bg-gray-50 flex items-center gap-2">
        <button
          onClick={() => setMode('create')}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
            mode === 'create' ? 'bg-blue-600 text-white' : 'bg-white text-gray-700 border border-gray-200'
          }`}
        >
          Create
        </button>
        <button
          onClick={() => setMode('import')}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium ${
            mode === 'import' ? 'bg-blue-600 text-white' : 'bg-white text-gray-700 border border-gray-200'
          }`}
        >
          Import JSON
        </button>

        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={handleDownloadJson}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium bg-white text-gray-700 border border-gray-200 hover:bg-gray-50"
            title="Download current template as JSON"
          >
            <FileDown className="w-4 h-4" />
            Export JSON
          </button>
        </div>
      </div>

      <div className="p-6 space-y-5">
        {mode === 'import' && (
          <div className="border border-dashed border-gray-300 rounded-lg p-4 bg-gray-50">
            <label className="block text-sm font-medium text-gray-700 mb-2">Import a template JSON file</label>
            <input
              type="file"
              accept="application/json"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) void handleImportJson(f)
              }}
              className="block w-full text-sm text-gray-700"
            />
            <p className="text-xs text-gray-500 mt-2">
              Tip: you can share templates between users by exporting/importing JSON.
            </p>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
              placeholder="e.g., Saudi Form v1"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Content Type</label>
            <select
              value={contentType}
              onChange={(e) => setContentType(e.target.value as any)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
            >
              {CONTENT_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Description (optional)</label>
          <input
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
            placeholder="Short description"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Language</label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value as any)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
            >
              <option value="ar">ar</option>
              <option value="en">en</option>
              <option value="mixed">mixed</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Keywords (optional)</label>
            <input
              value={keywords}
              onChange={(e) => setKeywords(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
              placeholder="comma-separated, e.g., هوية, نموذج, وزارة"
            />
          </div>
        </div>

        <div className="flex items-center gap-2">
          <input
            id="isPublic"
            type="checkbox"
            checked={isPublic}
            onChange={(e) => setIsPublic(e.target.checked)}
            className="h-4 w-4"
          />
          <label htmlFor="isPublic" className="text-sm text-gray-700">
            Public (shareable)
          </label>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Custom Prompt</label>
          <textarea
            value={customPrompt}
            onChange={(e) => setCustomPrompt(e.target.value)}
            className="w-full min-h-[160px] px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
            placeholder="Write a short prompt that forces the exact structure you want..."
          />
          <p className="text-xs text-gray-500 mt-2">
            Keep it simple: label/value rules, table separator rules, checkbox rules.
          </p>
        </div>

        <div className="flex items-center justify-end gap-2 pt-2">
          <button
            onClick={onCancel}
            className="px-4 py-2 rounded-lg border border-gray-200 text-gray-700 hover:bg-gray-50"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="px-4 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
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
    </div>
  )
}


