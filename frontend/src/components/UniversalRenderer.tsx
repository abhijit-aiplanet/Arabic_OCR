'use client'

import { useMemo, useState } from 'react'
import ExtractedText from '@/components/ExtractedText'
import type { ContentType } from '@/lib/api'
import { detectContentTypeFromText } from '@/lib/contentHeuristics'

interface UniversalRendererProps {
  text: string
  isProcessing: boolean
  onTextEdit?: (newText: string) => void
  isEditable?: boolean
  preferredType?: ContentType // user override (auto allowed)
}

type ViewMode = 'smart' | 'raw'

export default function UniversalRenderer({
  text,
  isProcessing,
  onTextEdit,
  isEditable = true,
  preferredType = 'auto'
}: UniversalRendererProps) {
  const [view, setView] = useState<ViewMode>('smart')

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
        return <FormView title={effectiveType === 'id_card' ? 'ID Card (structured)' : 'Form (structured)'} items={form.items} />
      }
    }

    if (effectiveType === 'receipt' || effectiveType === 'invoice') {
      // MVP: still raw, but with a better wrapper
      return (
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200" dir="auto">
          <div className="text-sm font-semibold text-gray-800 mb-2">Receipt/Invoice (raw)</div>
          <pre className="whitespace-pre-wrap font-sans text-sm text-gray-900">{text}</pre>
        </div>
      )
    }

    if (effectiveType === 'document' || effectiveType === 'handwritten' || effectiveType === 'mixed' || effectiveType === 'unknown') {
      // For non-forms, raw is the best default representation.
      return (
        <div className="bg-gray-50 rounded-lg p-4 border border-gray-200" dir="auto">
          <div className="text-sm font-semibold text-gray-800 mb-2">
            {effectiveType === 'document' ? 'Document' : effectiveType === 'handwritten' ? 'Handwritten' : effectiveType === 'mixed' ? 'Mixed' : 'Text'}
          </div>
          <pre className="whitespace-pre-wrap font-sans text-sm text-gray-900">{text}</pre>
        </div>
      )
    }

    return null
  }, [effectiveType, text])

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200 h-full">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Output</h2>
          <div className="text-xs text-gray-600">
            type: <span className="font-medium">{effectiveType}</span> · detected: <span className="font-medium">{detectedType}</span>
          </div>
        </div>

        {text && !isProcessing && (
          <div className="flex items-center gap-2">
            <button
              onClick={() => setView('smart')}
              className={`px-3 py-1.5 text-sm font-medium rounded-lg border transition-colors ${
                view === 'smart' ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'
              }`}
            >
              Smart view
            </button>
            <button
              onClick={() => setView('raw')}
              className={`px-3 py-1.5 text-sm font-medium rounded-lg border transition-colors ${
                view === 'raw' ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'
              }`}
            >
              Raw text
            </button>
          </div>
        )}
      </div>

      <div className="relative min-h-[400px] max-h-[600px] overflow-auto">
        {view === 'raw' ? (
          <ExtractedText text={text} isProcessing={isProcessing} onTextEdit={onTextEdit} isEditable={isEditable} />
        ) : isProcessing ? (
          <ExtractedText text={text} isProcessing={isProcessing} onTextEdit={onTextEdit} isEditable={isEditable} />
        ) : text ? (
          smartView || <ExtractedText text={text} isProcessing={false} onTextEdit={onTextEdit} isEditable={isEditable} />
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
  for (const line of lines) {
    const m = line.match(/^(.+?):\s*(.*)$/)
    if (m) {
      const label = m[1].trim()
      const value = (m[2] || '').trim()
      if (label.length >= 1) items.push({ label, value })
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
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
      <div className="px-4 py-3 bg-gray-50 border-b border-gray-200 text-sm font-semibold text-gray-900">{title}</div>
      <div className="p-4 space-y-2" dir="auto">
        {items.map((it, idx) => (
          <div key={idx} className="grid grid-cols-1 md:grid-cols-3 gap-2 p-2 rounded hover:bg-gray-50">
            <div className="text-sm font-medium text-gray-700 md:col-span-1">{it.label}</div>
            <div className="text-sm text-gray-900 md:col-span-2 whitespace-pre-wrap">
              {it.value ? it.value : <span className="text-gray-400 italic">[فارغ]</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function TableView({ headers, rows }: { headers: string[]; rows: string[][] }) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
      <div className="px-4 py-3 bg-gray-50 border-b border-gray-200 text-sm font-semibold text-gray-900">Table (structured)</div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm" dir="auto">
          <thead className="bg-gray-100">
            <tr>
              {headers.map((h, idx) => (
                <th key={idx} className="px-3 py-2 text-right font-semibold text-gray-700 border-b border-gray-200">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="hover:bg-gray-50">
                {headers.map((_, j) => (
                  <td key={j} className="px-3 py-2 text-right text-gray-900 border-b border-gray-100">
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


