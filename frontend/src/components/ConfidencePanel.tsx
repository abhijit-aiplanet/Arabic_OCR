'use client'

import { useMemo, useState } from 'react'
import type { OCRConfidence } from '@/lib/api'

function pct(x?: number | null) {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  return `${Math.round(x * 100)}%`
}

function levelColor(level?: OCRConfidence['confidence_level']) {
  switch (level) {
    case 'high':
      return 'bg-green-100 text-green-800 border-green-200'
    case 'medium':
      return 'bg-yellow-100 text-yellow-800 border-yellow-200'
    case 'low_medium':
      return 'bg-orange-100 text-orange-800 border-orange-200'
    case 'low':
      return 'bg-red-100 text-red-800 border-red-200'
    default:
      return 'bg-gray-100 text-gray-700 border-gray-200'
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

  const badge = useMemo(() => {
    if (!confidence) return null
    return (
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full border text-sm font-semibold transition-colors ${levelColor(
          confidence.confidence_level
        )}`}
        title="Click for confidence breakdown"
      >
        Confidence: {pct(overall)}
        <span className="text-xs font-medium opacity-80">{open ? 'Hide' : 'Details'}</span>
      </button>
    )
  }, [confidence, open, overall])

  if (!confidence) return null

  return (
    <div className={className}>
      <div className="flex items-center justify-between gap-3">
        {badge}
        {confidence.warnings?.length ? (
          <div className="text-xs text-gray-600">
            <span className="font-semibold">Warnings:</span> {confidence.warnings.slice(0, 2).join(' · ')}
            {confidence.warnings.length > 2 ? '…' : ''}
          </div>
        ) : null}
      </div>

      {open && (
        <div className="mt-3 border border-gray-200 rounded-lg bg-white p-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Metric label="Overall" value={pct(overall)} />
            <Metric label="Image quality" value={pct(sources.image_quality ?? null)} />
            <Metric label="Model logits" value={pct(sources.token_logits ?? null)} />
            <Metric label="Text quality" value={pct(sources.text_quality ?? null)} />
            <Metric label="Level" value={confidence.confidence_level} />
          </div>

          {(confidence.recommendations?.length || 0) > 0 && (
            <div className="mt-4">
              <div className="text-sm font-semibold text-gray-900 mb-1">Recommendations</div>
              <ul className="list-disc pl-5 text-sm text-gray-700">
                {confidence.recommendations?.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-gray-200 rounded-lg p-3 bg-gray-50">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-lg font-bold text-gray-900">{value}</div>
    </div>
  )
}


