'use client'

import { useEffect, useMemo, useState } from 'react'
import toast from 'react-hot-toast'
import { Plus, RefreshCw, Trash2 } from 'lucide-react'
import type { ContentType, OCRTemplate } from '@/lib/api'
import { createTemplate, deleteTemplate, fetchPublicTemplates, fetchTemplates } from '@/lib/api'
import TemplateBuilder from '@/components/TemplateBuilder'

interface TemplateSelectorProps {
  authToken: string | null
  selectedTemplate: OCRTemplate | null
  onSelectTemplate: (t: OCRTemplate | null) => void
  contentTypeOverride: ContentType
  onContentTypeOverrideChange: (t: ContentType) => void
}

const CONTENT_TYPE_OPTIONS: ContentType[] = [
  'auto',
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

export default function TemplateSelector({
  authToken,
  selectedTemplate,
  onSelectTemplate,
  contentTypeOverride,
  onContentTypeOverrideChange
}: TemplateSelectorProps) {
  const [templates, setTemplates] = useState<OCRTemplate[]>([])
  const [publicTemplates, setPublicTemplates] = useState<OCRTemplate[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [showBuilder, setShowBuilder] = useState(false)

  const allTemplates = useMemo(() => {
    const dedup = new Map<string, OCRTemplate>()
    for (const t of templates) dedup.set(t.id, t)
    for (const t of publicTemplates) dedup.set(t.id, t)
    return Array.from(dedup.values()).sort((a, b) => (b.usage_count || 0) - (a.usage_count || 0))
  }, [templates, publicTemplates])

  const load = async () => {
    setIsLoading(true)
    try {
      if (authToken) {
        const [mine, pub] = await Promise.all([fetchTemplates(authToken), fetchPublicTemplates()])
        setTemplates(mine)
        setPublicTemplates(pub)
      } else {
        const pub = await fetchPublicTemplates()
        setPublicTemplates(pub)
        setTemplates([])
      }
    } catch (e: any) {
      toast.error(e?.message || 'Failed to load templates')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    void load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authToken])

  const handleDelete = async (templateId: string) => {
    if (!authToken) return
    if (!confirm('Delete this template?')) return
    try {
      await deleteTemplate(templateId, authToken)
      if (selectedTemplate?.id === templateId) onSelectTemplate(null)
      toast.success('Template deleted')
      await load()
    } catch (e: any) {
      toast.error(e?.message || 'Failed to delete template')
    }
  }

  if (showBuilder) {
    return (
      <TemplateBuilder
        onCancel={() => setShowBuilder(false)}
        onCreate={async (payload) => {
          if (!authToken) {
            throw new Error('Please sign in to create templates')
          }
          await createTemplate(payload, authToken)
          await load()
        }}
      />
    )
  }

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Templates & Content Type</h3>
          <p className="text-sm text-gray-600">Optional: pick a template or force a content type (Auto is safest)</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => void load()}
            disabled={isLoading}
            className="p-2 hover:bg-gray-100 rounded-lg disabled:opacity-50"
            title="Refresh"
          >
            <RefreshCw className={`w-5 h-5 text-gray-500 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
          <button
            onClick={() => setShowBuilder(true)}
            className="flex items-center gap-2 px-3 py-2 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-colors"
          >
            <Plus className="w-4 h-4" />
            New
          </button>
        </div>
      </div>

      <div className="p-6 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Content type</label>
            <select
              value={contentTypeOverride}
              onChange={(e) => onContentTypeOverrideChange(e.target.value as ContentType)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
            >
              {CONTENT_TYPE_OPTIONS.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-500 mt-1">
              If you choose a specific type, we’ll generate a matching prompt even if the file isn’t a form.
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Template</label>
            <select
              value={selectedTemplate?.id || ''}
              onChange={(e) => {
                const id = e.target.value
                const t = allTemplates.find((x) => x.id === id) || null
                onSelectTemplate(t)
              }}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
            >
              <option value="">(No template)</option>
              {allTemplates.map((t) => (
                <option key={t.id} value={t.id}>
                  {t.name} — {t.content_type}{t.is_public ? ' (public)' : ''}
                </option>
              ))}
            </select>
            <p className="text-xs text-gray-500 mt-1">
              Template prompt overrides content-type prompt. Your manual prompt in Advanced Settings still wins.
            </p>
          </div>
        </div>

        {selectedTemplate && (
          <div className="border border-gray-200 rounded-lg p-4 bg-gray-50">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="font-semibold text-gray-900">{selectedTemplate.name}</div>
                <div className="text-sm text-gray-600">
                  type: <span className="font-medium">{selectedTemplate.content_type}</span> · language:{' '}
                  <span className="font-medium">{selectedTemplate.language}</span>
                  {selectedTemplate.is_public ? ' · public' : ''}
                </div>
                {selectedTemplate.description && (
                  <div className="text-sm text-gray-600 mt-1">{selectedTemplate.description}</div>
                )}
              </div>
              {authToken && selectedTemplate.user_id && (
                <button
                  onClick={() => void handleDelete(selectedTemplate.id)}
                  className="p-2 hover:bg-red-50 rounded-lg"
                  title="Delete template"
                >
                  <Trash2 className="w-5 h-5 text-red-600" />
                </button>
              )}
            </div>
            {selectedTemplate.custom_prompt && (
              <pre className="mt-3 whitespace-pre-wrap text-xs text-gray-800 bg-white border border-gray-200 rounded-lg p-3">
                {selectedTemplate.custom_prompt}
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  )
}


