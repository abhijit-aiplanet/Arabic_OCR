'use client'

import { useMemo, useState } from 'react'
import ExtractedText from '@/components/ExtractedText'
import type { ContentType, OCRConfidence } from '@/lib/api'
import { detectContentTypeFromText } from '@/lib/contentHeuristics'
import ConfidencePanel from '@/components/ConfidencePanel'
import { Eye, FileText } from 'lucide-react'

interface UniversalRendererProps {
  text: string
  isProcessing: boolean
  onTextEdit?: (newText: string) => void
  isEditable?: boolean
  preferredType?: ContentType
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
      if (form && form.items.length >= 3) {
        return <FormView title={effectiveType === 'id_card' ? 'ID Card' : 'Form'} items={form.items} />
      }
    }

    if (effectiveType === 'receipt' || effectiveType === 'invoice') {
      return (
        <div className="bg-gray-50 rounded-xl p-5" dir="auto">
          <div className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-3">Receipt/Invoice</div>
          <pre className="whitespace-pre-wrap font-sans text-sm text-gray-900 leading-relaxed">{text}</pre>
        </div>
      )
    }

    if (effectiveType === 'document' || effectiveType === 'handwritten' || effectiveType === 'mixed' || effectiveType === 'unknown') {
      return (
        <div className="bg-gray-50 rounded-xl p-5" dir="auto">
          <pre className="whitespace-pre-wrap font-sans text-sm text-gray-900 leading-relaxed">{text}</pre>
        </div>
      )
    }

    return null
  }, [effectiveType, text])

  return (
    <div className="bg-white rounded-2xl border border-gray-100 h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b border-gray-100">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-gray-900">Output</h2>

          {/* View Toggle */}
          {text && !isProcessing && (
            <div className="flex items-center gap-2">
              <div className="inline-flex rounded-lg bg-gray-100 p-0.5">
                <button
                  onClick={() => setView('smart')}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                    view === 'smart' 
                      ? 'bg-white text-gray-900 shadow-sm' 
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <Eye className="w-3.5 h-3.5" />
                  Smart
                </button>
                <button
                  onClick={() => setView('raw')}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                    view === 'raw' 
                      ? 'bg-white text-gray-900 shadow-sm' 
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <FileText className="w-3.5 h-3.5" />
                  Raw
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Confidence Panel */}
        {text && !isProcessing && confidence && (
          <div className="mt-3">
            <ConfidencePanel confidence={confidence} />
          </div>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 p-5 overflow-auto">
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

      {/* Word Confidence Toggle (if available) */}
      {text && !isProcessing && confidence?.per_word?.length ? (
        <div className="px-5 py-3 border-t border-gray-100">
          <button
            onClick={() => setShowWordConfidence((v) => !v)}
            className={`text-xs font-medium px-3 py-1.5 rounded-lg transition-all ${
              showWordConfidence 
                ? 'bg-gray-900 text-white' 
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {showWordConfidence ? 'Hide word confidence' : 'Show word confidence'}
          </button>
        </div>
      ) : null}
    </div>
  )
}

function tryParseLabelValue(text: string): { items: Array<{ label: string; value: string }> } | null {
  const lines = text.split('\n').map((l) => l.trim()).filter(Boolean)
  const items: Array<{ label: string; value: string }> = []
  
  let currentSection = ''
  
  for (const line of lines) {
    const labelValue = line.match(/^(.+?):\s*(.+)$/)
    if (labelValue) {
      const label = labelValue[1].trim()
      const value = labelValue[2].trim()
      if (label.length >= 1 && value.length >= 1) {
        items.push({ label, value })
        continue
      }
    }
    
    const sectionHeader = line.match(/^(.+?):\s*\.?\s*$/)
    if (sectionHeader) {
      currentSection = sectionHeader[1].trim()
      continue
    }
    
    const listItem = line.match(/^[\d٠-٩]+\s*[-–—]\s*(.+)$/)
    if (listItem && currentSection) {
      const value = listItem[1].trim()
      items.push({ 
        label: currentSection, 
        value: value
      })
      continue
    }
    
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
    <div className="bg-gray-50 rounded-xl overflow-hidden">
      <div className="px-5 py-3 bg-gray-100 border-b border-gray-200">
        <div className="text-xs font-medium text-gray-500 uppercase tracking-wider">{title}</div>
      </div>
      <div className="divide-y divide-gray-100" dir="auto">
        {items.map((it, idx) => (
          <div key={idx} className="flex items-start gap-4 px-5 py-3">
            <div className="text-sm text-gray-500 min-w-[100px] pt-0.5">{it.label}</div>
            <div className="text-sm text-gray-900 flex-1 whitespace-pre-wrap font-medium">
              {it.value ? it.value : <span className="text-gray-300">[empty]</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function TableView({ headers, rows }: { headers: string[]; rows: string[][] }) {
  return (
    <div className="bg-gray-50 rounded-xl overflow-hidden">
      <div className="px-5 py-3 bg-gray-100 border-b border-gray-200">
        <div className="text-xs font-medium text-gray-500 uppercase tracking-wider">Table</div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm" dir="auto">
          <thead>
            <tr className="bg-gray-100">
              {headers.map((h, idx) => (
                <th key={idx} className="px-4 py-3 text-right font-medium text-gray-700 border-b border-gray-200 first:text-left">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {rows.map((r, i) => (
              <tr key={i} className="hover:bg-gray-100/50 transition-colors">
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
    <div className="bg-gray-50 rounded-xl p-5" dir="auto">
      <div className="flex items-center justify-between mb-4">
        <div className="text-xs font-medium text-gray-400 uppercase tracking-wider">Word Confidence</div>
        <div className="flex items-center gap-3 text-xs">
          <span className="inline-flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded bg-emerald-200"></span>
            <span className="text-gray-500">High</span>
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded bg-amber-200"></span>
            <span className="text-gray-500">Medium</span>
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded bg-red-200"></span>
            <span className="text-gray-500">Low</span>
          </span>
        </div>
      </div>
      <div className="leading-loose text-gray-900 text-sm">
        {tokens.map((tok, i) => {
          if (!tok.trim()) return <span key={i}>{tok}</span>
          const c = perWord[idx]?.confidence ?? null
          idx += 1
          const cls = confidenceClass(c)
          const title = c === null ? 'confidence: —' : `confidence: ${Math.round(c * 100)}%`
          return (
            <span key={i} className={`px-1 py-0.5 rounded ${cls} transition-colors cursor-default`} title={title}>
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
