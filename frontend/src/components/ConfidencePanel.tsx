'use client'

import { useMemo, useState } from 'react'
import type { OCRConfidence } from '@/lib/api'
import { ChevronDown, ChevronUp, AlertTriangle, CheckCircle, Info } from 'lucide-react'

function pct(x?: number | null) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  return `${Math.round(x * 100)}%`
}

function levelConfig(level?: OCRConfidence['confidence_level']) {
  switch (level) {
    case 'high':
      return { bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200', icon: CheckCircle }
    case 'medium':
      return { bg: 'bg-amber-50', text: 'text-amber-700', border: 'border-amber-200', icon: Info }
    case 'low_medium':
      return { bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200', icon: AlertTriangle }
    case 'low':
      return { bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200', icon: AlertTriangle }
    default:
      return { bg: 'bg-gray-50', text: 'text-gray-600', border: 'border-gray-200', icon: Info }
  }
}

export default function ConfidencePanel({
  confidence,
  className = ''
}: {
  confidence?: OCRConfidence | null
  className?: string
}) {
  const [open, setOpen] = useState(false)

  const overall = confidence?.overall_confidence
  const sources = confidence?.confidence_sources || {}
  const config = levelConfig(confidence?.confidence_level)
  const Icon = config.icon

  if (!confidence) return null

  return (
    <div className={className}>
      {/* Compact Badge */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full border text-sm font-semibold transition-all hover:shadow-sm ${config.bg} ${config.text} ${config.border}`}
      >
        <Icon className="w-4 h-4" />
        <span>{pct(overall)}</span>
        {open ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
      </button>

      {/* Expanded Details */}
      {open && (
        <div className="mt-3 border border-gray-200 rounded-xl bg-white shadow-sm overflow-hidden">
          {/* Metrics Grid */}
          <div className="grid grid-cols-5 divide-x divide-gray-100">
            <MetricCell label="Overall" value={pct(overall)} highlight />
            <MetricCell label="Image" value={pct(sources.image_quality ?? null)} />
            <MetricCell label="Model" value={pct(sources.token_logits ?? null)} />
            <MetricCell label="Text" value={pct(sources.text_quality ?? null)} />
            <MetricCell label="Level" value={confidence.confidence_level || '—'} />
          </div>

          {/* Recommendations */}
          {(confidence.recommendations?.length || 0) > 0 && (
            <div className="px-4 py-3 bg-gray-50 border-t border-gray-100">
              <ul className="text-sm text-gray-700 space-y-1">
                {confidence.recommendations?.map((r, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <span className="text-gray-400 mt-0.5">•</span>
                    <span>{r}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function MetricCell({ label, value, highlight = false }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={`px-3 py-3 text-center ${highlight ? 'bg-blue-50' : 'bg-white'}`}>
      <div className="text-xs text-gray-500 font-medium uppercase tracking-wider">{label}</div>
      <div className={`text-lg font-bold mt-0.5 ${highlight ? 'text-blue-700' : 'text-gray-900'}`}>{value}</div>
    </div>
  )
}


