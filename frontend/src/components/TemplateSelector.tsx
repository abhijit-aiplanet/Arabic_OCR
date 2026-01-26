'use client'

import { useEffect, useMemo, useState } from 'react'
import toast from 'react-hot-toast'
import { Plus, RefreshCw, Trash2, FileText, ChevronDown, ChevronUp, Zap } from 'lucide-react'
import type { ContentType, OCRTemplate } from '@/lib/api'
import { createTemplate, deleteTemplate, fetchPublicTemplates, fetchTemplates } from '@/lib/api'
import TemplateBuilder from '@/components/TemplateBuilder'

interface TemplateSelectorProps {
  authToken: string | null
  selectedTemplate: OCRTemplate | null
  onSelectTemplate: (t: OCRTemplate | null) => void
}

export default function TemplateSelector({
  authToken,
  selectedTemplate,
  onSelectTemplate,
}: TemplateSelectorProps) {
  const [templates, setTemplates] = useState<OCRTemplate[]>([])
  const [publicTemplates, setPublicTemplates] = useState<OCRTemplate[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [showBuilder, setShowBuilder] = useState(false)
  const [showDetails, setShowDetails] = useState(false)

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
      console.warn('Failed to load templates:', e)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    void load()
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

  const getFieldCount = (template: OCRTemplate) => {
    const sections = template.sections as any
    if (!sections?.sections) return 0
    return sections.sections.reduce((sum: number, s: any) => 
      sum + (s.fields?.length || 0), 0
    )
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
    <div className="bg-white rounded-2xl border border-gray-100 overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b border-gray-100">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-purple-100 rounded-lg flex items-center justify-center">
              <Zap className="w-5 h-5 text-purple-600" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-gray-900">Agentic OCR</h3>
              <p className="text-xs text-gray-500">Multi-pass AI extraction with self-correction</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => void load()}
              disabled={isLoading}
              className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-50 rounded-lg disabled:opacity-50 transition-colors"
              title="Refresh templates"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={() => setShowBuilder(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors"
            >
              <Plus className="w-4 h-4" />
              New Template
            </button>
          </div>
        </div>
      </div>

      <div className="p-5 space-y-4">
        {/* Info Banner */}
        <div className="p-3 bg-purple-50 border border-purple-100 rounded-lg">
          <div className="flex items-start gap-2">
            <Zap className="w-4 h-4 text-purple-600 mt-0.5 flex-shrink-0" />
            <div className="text-xs text-purple-700">
              <p className="font-medium">Multi-pass Self-Correcting OCR</p>
              <p className="mt-0.5 text-purple-600">
                Uses AI reasoning to identify and re-examine uncertain fields. 
                Achieves highest accuracy on handwritten Arabic forms.
              </p>
            </div>
          </div>
        </div>

        {/* Template Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1.5">Template (Optional)</label>
          <select
            value={selectedTemplate?.id || ''}
            onChange={(e) => {
              const id = e.target.value
              const t = allTemplates.find((x) => x.id === id) || null
              onSelectTemplate(t)
            }}
            className="w-full px-3 py-2.5 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-sm"
          >
            <option value="">No template (auto-detect fields)</option>
            {allTemplates.map((t) => {
              const fieldCount = getFieldCount(t)
              return (
                <option key={t.id} value={t.id}>
                  {t.name} ({t.content_type}){fieldCount > 0 ? ` - ${fieldCount} fields` : ''}{t.is_public ? ' [public]' : ''}
                </option>
              )
            })}
          </select>
          <p className="mt-1.5 text-xs text-gray-500">
            Templates help guide extraction for specific document types
          </p>
        </div>

        {/* Selected Template Details */}
        {selectedTemplate && (
          <div className="border border-gray-100 rounded-xl overflow-hidden">
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="w-full px-4 py-3 bg-gray-50 flex items-center justify-between hover:bg-gray-100 transition-colors"
            >
              <div className="flex items-center gap-3">
                <div className="text-left">
                  <p className="font-medium text-gray-900 text-sm">{selectedTemplate.name}</p>
                  <p className="text-xs text-gray-500">
                    {selectedTemplate.content_type} | {selectedTemplate.language}
                    {getFieldCount(selectedTemplate) > 0 && ` | ${getFieldCount(selectedTemplate)} fields`}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {authToken && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      void handleDelete(selectedTemplate.id)
                    }}
                    className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                    title="Delete template"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
                {showDetails ? (
                  <ChevronUp className="w-4 h-4 text-gray-400" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-gray-400" />
                )}
              </div>
            </button>

            {showDetails && (
              <div className="p-4 space-y-3 border-t border-gray-100">
                {selectedTemplate.description && (
                  <p className="text-sm text-gray-600">{selectedTemplate.description}</p>
                )}
                
                {(() => {
                  const sections = selectedTemplate.sections as any
                  if (!sections?.sections?.length) return null
                  
                  return (
                    <div className="space-y-2">
                      <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">Expected Fields</p>
                      {sections.sections.map((section: any, idx: number) => (
                        <div key={idx} className="bg-gray-50 rounded-lg p-3">
                          <p className="text-xs font-medium text-gray-700 mb-2">
                            {section.name || 'General'}
                          </p>
                          <div className="flex flex-wrap gap-1">
                            {section.fields?.map((field: any, fidx: number) => (
                              <span 
                                key={fidx}
                                className="px-2 py-0.5 bg-white border border-gray-200 rounded text-xs text-gray-600"
                              >
                                {field.label}
                              </span>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )
                })()}

                {selectedTemplate.custom_prompt && (
                  <div>
                    <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Custom Prompt</p>
                    <pre className="whitespace-pre-wrap text-xs text-gray-700 bg-gray-50 rounded-lg p-3 max-h-32 overflow-y-auto">
                      {selectedTemplate.custom_prompt}
                    </pre>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
