'use client'

import { useState } from 'react'
import type { OCRConfidence } from '@/lib/api'
import { ChevronDown, ChevronUp, AlertTriangle, CheckCircle, Info } from 'lucide-react'

function pct(x?: number | null) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  return `${Math.round(x * 100)}%`
}

function levelConfig(level?: OCRConfidence['confidence_level']) {
  switch (level) {
    case 'high':
      return { bg: 'bg-emerald-50', text: 'text-emerald-700', icon: CheckCircle }
    case 'medium':
      return { bg: 'bg-amber-50', text: 'text-amber-700', icon: Info }
    case 'low_medium':
      return { bg: 'bg-orange-50', text: 'text-orange-700', icon: AlertTriangle }
    case 'low':
      return { bg: 'bg-red-50', text: 'text-red-700', icon: AlertTriangle }
    default:
      return { bg: 'bg-gray-50', text: 'text-gray-600', icon: Info }
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
        className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${config.bg} ${config.text}`}
      >
        <Icon className="w-4 h-4" />
        <span>{pct(overall)} confidence</span>
        {open ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
      </button>

      {/* Expanded Details */}
      {open && (
        <div className="mt-3 border border-gray-100 rounded-xl bg-white overflow-hidden">
          {/* Metrics Grid */}
          <div className="grid grid-cols-4 divide-x divide-gray-100">
            <MetricCell label="Overall" value={pct(overall)} highlight />
            <MetricCell label="Image" value={pct(sources.image_quality ?? null)} />
            <MetricCell label="Model" value={pct(sources.token_logits ?? null)} />
            <MetricCell label="Text" value={pct(sources.text_quality ?? null)} />
          </div>

          {/* Recommendations */}
          {(confidence.recommendations?.length || 0) > 0 && (
            <div className="px-4 py-3 bg-gray-50 border-t border-gray-100">
              <ul className="text-xs text-gray-600 space-y-1">
                {confidence.recommendations?.map((r, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <span className="text-gray-400">•</span>
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
    <div className={`px-3 py-3 text-center ${highlight ? 'bg-gray-50' : 'bg-white'}`}>
      <div className="text-xs text-gray-400 uppercase tracking-wide">{label}</div>
      <div className={`text-base font-semibold mt-0.5 ${highlight ? 'text-gray-900' : 'text-gray-700'}`}>{value}</div>
    </div>
  )
}
