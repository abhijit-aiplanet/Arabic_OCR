'use client'

import { useMemo, useState } from 'react'
import ExtractedText from '@/components/ExtractedText'
import type { ContentType, OCRConfidence } from '@/lib/api'
import { detectContentTypeFromText } from '@/lib/contentHeuristics'
import ConfidencePanel from '@/components/ConfidencePanel'
import { Eye, FileText, Sparkles } from 'lucide-react'

interface UniversalRendererProps {
  text: string
  isProcessing: boolean
  onTextEdit?: (newText: string) => void
  isEditable?: boolean
  preferredType?: ContentType // user override (auto allowed)
  confidence?: OCRConfidence | null
}

type ViewMode = 'smart' | 'raw'

export default function UniversalRenderer({
  text,
  isProcessing,
  onTextEdit,
  isEditable = true,
  preferredType = 'auto',
  confidence = null
}: UniversalRendererProps) {
  const [view, setView] = useState<ViewMode>('smart')
  const [showWordConfidence, setShowWordConfidence] = useState(false)

  const detectedType = useMemo(() => detectContentTypeFromText(text), [text])
  const effectiveType: Exclude<ContentType, 'auto'> =
    preferredType !== 'auto' ? (preferredType as any) : detectedType

  const smartView = useMemo(() => {
    if (!text || !text.trim()) return null

    if (effectiveType === 'table') {
      const table = tryParseTable(text)
      if (table) return <TableView headers={table.headers} rows={table.rows} />
    }

    if (effectiveType === 'form' || effectiveType === 'id_card') {
      const form = tryParseLabelValue(text)
      // Guardrail: if it doesn't really look like a form, fall back
      if (form && form.items.length >= 3) {
        return <FormView title={effectiveType === 'id_card' ? 'ID Card' : 'Form'} items={form.items} />
      }
    }

    if (effectiveType === 'receipt' || effectiveType === 'invoice') {
      return (
        <div className="bg-gradient-to-br from-slate-50 to-gray-50 rounded-xl p-5 border border-gray-200" dir="auto">
          <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">Receipt/Invoice</div>
          <pre className="whitespace-pre-wrap font-sans text-sm text-gray-900 leading-relaxed">{text}</pre>
        </div>
      )
    }

    if (effectiveType === 'document' || effectiveType === 'handwritten' || effectiveType === 'mixed' || effectiveType === 'unknown') {
      return (
        <div className="bg-gradient-to-br from-slate-50 to-gray-50 rounded-xl p-5 border border-gray-200" dir="auto">
          <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
            {effectiveType === 'document' ? 'Document' : effectiveType === 'handwritten' ? 'Handwritten' : effectiveType === 'mixed' ? 'Mixed Content' : 'Text'}
          </div>
          <pre className="whitespace-pre-wrap font-sans text-sm text-gray-900 leading-relaxed">{text}</pre>
        </div>
      )
    }

    return null
  }, [effectiveType, text])

  return (
    <div className="bg-white rounded-2xl shadow-lg border border-gray-100 h-full flex flex-col overflow-hidden">
      {/* Header Section */}
      <div className="px-6 py-4 border-b border-gray-100 bg-gradient-to-r from-white to-gray-50">
        <div className="flex items-start justify-between gap-4">
          {/* Left: Title + Confidence */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2">
              <Sparkles className="w-5 h-5 text-blue-600" />
              <h2 className="text-lg font-bold text-gray-900">Output</h2>
            </div>
            <ConfidencePanel confidence={confidence} />
          </div>

          {/* Right: View Toggle */}
          {text && !isProcessing && (
            <div className="flex flex-col items-end gap-2">
              {/* Toggle Button Group */}
              <div className="inline-flex rounded-lg border border-gray-200 bg-gray-100 p-1">
                <button
                  onClick={() => setView('smart')}
                  className={`inline-flex items-center gap-1.5 px-4 py-2 text-sm font-medium rounded-md transition-all duration-200 ${
                    view === 'smart' 
                      ? 'bg-white text-blue-700 shadow-sm' 
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <Eye className="w-4 h-4" />
                  Smart
                </button>
                <button
                  onClick={() => setView('raw')}
                  className={`inline-flex items-center gap-1.5 px-4 py-2 text-sm font-medium rounded-md transition-all duration-200 ${
                    view === 'raw' 
                      ? 'bg-white text-blue-700 shadow-sm' 
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <FileText className="w-4 h-4" />
                  Raw
                </button>
              </div>
              
              {/* Word Confidence Toggle */}
              {confidence?.per_word?.length ? (
                <button
                  onClick={() => setShowWordConfidence((v) => !v)}
                  className={`text-xs font-medium px-3 py-1.5 rounded-full transition-all ${
                    showWordConfidence 
                      ? 'bg-blue-100 text-blue-700' 
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  {showWordConfidence ? '✓ Word confidence' : 'Word confidence'}
                </button>
              ) : null}
            </div>
          )}
        </div>
      </div>

      {/* Content Section */}
      <div className="flex-1 p-6 overflow-auto">
        {view === 'raw' ? (
          <ExtractedText text={text} isProcessing={isProcessing} onTextEdit={onTextEdit} isEditable={isEditable} />
        ) : isProcessing ? (
          <ExtractedText text={text} isProcessing={isProcessing} onTextEdit={onTextEdit} isEditable={isEditable} />
        ) : text ? (
          showWordConfidence && confidence?.per_word?.length ? (
            <WordConfidenceView text={text} perWord={confidence.per_word} />
          ) : (
            smartView || <ExtractedText text={text} isProcessing={false} onTextEdit={onTextEdit} isEditable={isEditable} />
          )
        ) : (
          <ExtractedText text={text} isProcessing={isProcessing} onTextEdit={onTextEdit} isEditable={isEditable} />
        )}
      </div>
    </div>
  )
}

function tryParseLabelValue(text: string): { items: Array<{ label: string; value: string }> } | null {
  const lines = text.split('\n').map((l) => l.trim()).filter(Boolean)
  const items: Array<{ label: string; value: string }> = []
  
  let currentSection = ''
  
  for (const line of lines) {
    // Pattern 1: label: value (traditional form field)
    const labelValue = line.match(/^(.+?):\s*(.+)$/)
    if (labelValue) {
      const label = labelValue[1].trim()
      const value = labelValue[2].trim()
      if (label.length >= 1 && value.length >= 1) {
        items.push({ label, value })
        continue
      }
    }
    
    // Pattern 2: Section header (ends with colon, no value)
    const sectionHeader = line.match(/^(.+?):\s*\.?\s*$/)
    if (sectionHeader) {
      currentSection = sectionHeader[1].trim()
      continue
    }
    
    // Pattern 3: Numbered/bulleted list items under a section
    const listItem = line.match(/^[\d٠-٩]+\s*[-–—]\s*(.+)$/)
    if (listItem && currentSection) {
      const value = listItem[1].trim()
      items.push({ 
        label: currentSection, 
        value: value
      })
      continue
    }
    
    // Pattern 4: Plain key-value separated by spaces (like "العنم ختم")
    const spaceSeparated = line.match(/^(\S+)\s{2,}(.+)$/)
    if (spaceSeparated) {
      const label = spaceSeparated[1].trim()
      const value = spaceSeparated[2].trim()
      if (label.length >= 1 && value.length >= 1) {
        items.push({ label, value })
      }
    }
  }
  
  if (items.length === 0) return null
  return { items }
}

function tryParseTable(text: string): { headers: string[]; rows: string[][] } | null {
  const lines = text.split('\n').map((l) => l.trim()).filter(Boolean)
  const rows = lines
    .map((line) => line.split(/[|│]/).map((c) => c.trim()).filter(Boolean))
    .filter((r) => r.length >= 2)

  if (rows.length < 2) return null

  // Basic consistency check: most rows should have similar column counts
  const counts = rows.map((r) => r.length)
  const modeCount = mostCommon(counts)
  const consistent = rows.filter((r) => r.length === modeCount).length / rows.length
  if (consistent < 0.6) return null

  const headers = rows[0]
  const body = rows.slice(1)
  return { headers, rows: body }
}

function mostCommon(nums: number[]): number {
  const m = new Map<number, number>()
  for (const n of nums) m.set(n, (m.get(n) || 0) + 1)
  let best = nums[0] || 0
  let bestCount = 0
  // Avoid iterating MapIterator directly to keep compatibility with lower TS targets (Vercel builds)
  m.forEach((v, k) => {
    if (v > bestCount) {
      best = k
      bestCount = v
    }
  })
  return best
}

function FormView({ title, items }: { title: string; items: Array<{ label: string; value: string }> }) {
  return (
    <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
      <div className="px-5 py-3 bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-gray-200">
        <div className="text-xs font-semibold text-blue-700 uppercase tracking-wider">{title}</div>
      </div>
      <div className="divide-y divide-gray-100" dir="auto">
        {items.map((it, idx) => (
          <div key={idx} className="flex items-start gap-4 px-5 py-3 hover:bg-gray-50 transition-colors">
            <div className="text-sm font-medium text-gray-500 min-w-[120px] pt-0.5">{it.label}</div>
            <div className="text-sm text-gray-900 flex-1 whitespace-pre-wrap font-medium">
              {it.value ? it.value : <span className="text-gray-300 italic font-normal">[empty]</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function TableView({ headers, rows }: { headers: string[]; rows: string[][] }) {
  return (
    <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm">
      <div className="px-5 py-3 bg-gradient-to-r from-emerald-50 to-teal-50 border-b border-gray-200">
        <div className="text-xs font-semibold text-emerald-700 uppercase tracking-wider">Table</div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm" dir="auto">
          <thead>
            <tr className="bg-gray-50">
              {headers.map((h, idx) => (
                <th key={idx} className="px-4 py-3 text-right font-semibold text-gray-700 border-b border-gray-200 first:text-left">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {rows.map((r, i) => (
              <tr key={i} className="hover:bg-gray-50 transition-colors">
                {headers.map((_, j) => (
                  <td key={j} className="px-4 py-3 text-right text-gray-900 first:text-left first:font-medium">
                    {r[j] ?? ''}
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

function WordConfidenceView({
  text,
  perWord
}: {
  text: string
  perWord: Array<{ word: string; confidence: number | null }>
}) {
  const tokens = text.split(/(\s+)/)
  let idx = 0

  return (
    <div className="bg-gradient-to-br from-slate-50 to-gray-50 rounded-xl p-6 border border-gray-200" dir="auto">
      <div className="flex items-center justify-between mb-4">
        <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Word-level Confidence</div>
        <div className="flex items-center gap-3 text-xs">
          <span className="inline-flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-emerald-200"></span>
            <span className="text-gray-600">High</span>
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-amber-200"></span>
            <span className="text-gray-600">Medium</span>
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-orange-200"></span>
            <span className="text-gray-600">Low-Med</span>
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-red-200"></span>
            <span className="text-gray-600">Low</span>
          </span>
        </div>
      </div>
      <div className="leading-loose text-gray-900 text-base">
        {tokens.map((tok, i) => {
          if (!tok.trim()) return <span key={i}>{tok}</span>
          const c = perWord[idx]?.confidence ?? null
          idx += 1
          const cls = confidenceClass(c)
          const title = c === null ? 'confidence: —' : `confidence: ${Math.round(c * 100)}%`
          return (
            <span key={i} className={`px-1 py-0.5 rounded-md ${cls} transition-colors cursor-default`} title={title}>
              {tok}
            </span>
          )
        })}
      </div>
    </div>
  )
}

function confidenceClass(c: number | null) {
  if (c === null || c === undefined || Number.isNaN(c)) return 'bg-gray-100'
  if (c >= 0.9) return 'bg-emerald-100 hover:bg-emerald-200'
  if (c >= 0.75) return 'bg-amber-100 hover:bg-amber-200'
  if (c >= 0.6) return 'bg-orange-100 hover:bg-orange-200'
  return 'bg-red-100 hover:bg-red-200'
}


