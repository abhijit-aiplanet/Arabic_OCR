'use client'

import { useState } from 'react'
import { ZoomIn, ZoomOut, RotateCw, Copy, Check, Download, ChevronLeft, ChevronRight } from 'lucide-react'
import type { AgenticOCRResponse } from '@/lib/api'

interface PageResultViewProps {
  imageUrl: string  // Base64 data URL or blob URL
  result: AgenticOCRResponse | null
  isProcessing: boolean
  pageLabel?: string
  onPrevPage?: () => void
  onNextPage?: () => void
  hasPrev?: boolean
  hasNext?: boolean
}

export default function PageResultView({
  imageUrl,
  result,
  isProcessing,
  pageLabel,
  onPrevPage,
  onNextPage,
  hasPrev = false,
  hasNext = false,
}: PageResultViewProps) {
  const [zoom, setZoom] = useState(1)
  const [copied, setCopied] = useState(false)

  const handleZoomIn = () => setZoom(prev => Math.min(prev + 0.25, 3))
  const handleZoomOut = () => setZoom(prev => Math.max(prev - 0.25, 0.5))
  const handleResetZoom = () => setZoom(1)

  const handleCopy = async () => {
    if (!result?.raw_text) return
    try {
      await navigator.clipboard.writeText(result.raw_text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }

  const handleDownload = () => {
    if (!result?.raw_text) return
    const blob = new Blob([result.raw_text], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `ocr-result-${pageLabel || 'page'}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <div className="bg-white rounded-2xl border border-gray-100 overflow-hidden">
      {/* Header */}
      <div className="px-5 py-3 border-b border-gray-100 flex items-center justify-between">
        <div className="flex items-center gap-3">
          {pageLabel && (
            <span className="text-sm font-medium text-gray-900">{pageLabel}</span>
          )}
          {result && (
            <div className="flex items-center gap-2">
              <span className={`text-xs px-2 py-0.5 rounded-full ${
                result.quality_status === 'passed' ? 'bg-emerald-100 text-emerald-700' :
                result.quality_status === 'warning' ? 'bg-amber-100 text-amber-700' :
                'bg-red-100 text-red-700'
              }`}>
                {result.quality_score}% quality
              </span>
              <span className="text-xs text-gray-500">
                {result.processing_time_seconds.toFixed(1)}s
              </span>
            </div>
          )}
        </div>
        
        {/* Navigation */}
        {(hasPrev || hasNext) && (
          <div className="flex items-center gap-1">
            <button
              onClick={onPrevPage}
              disabled={!hasPrev}
              className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-50 rounded-lg disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={onNextPage}
              disabled={!hasNext}
              className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-50 rounded-lg disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>

      {/* Content - Side by Side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x divide-gray-100">
        {/* Left - Original Image */}
        <div className="p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Original</span>
            <div className="flex items-center gap-1">
              <button
                onClick={handleZoomOut}
                className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-50 rounded transition-colors"
                title="Zoom out"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <span className="text-xs text-gray-500 w-12 text-center">{Math.round(zoom * 100)}%</span>
              <button
                onClick={handleZoomIn}
                className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-50 rounded transition-colors"
                title="Zoom in"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                onClick={handleResetZoom}
                className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-50 rounded transition-colors"
                title="Reset zoom"
              >
                <RotateCw className="w-4 h-4" />
              </button>
            </div>
          </div>
          
          <div className="border border-gray-200 rounded-xl overflow-auto bg-gray-50" style={{ maxHeight: '500px' }}>
            <div 
              className="transition-transform duration-200 origin-top-left"
              style={{ transform: `scale(${zoom})` }}
            >
              <img
                src={imageUrl}
                alt="Original document"
                className="w-full h-auto"
              />
            </div>
          </div>
        </div>

        {/* Right - OCR Output */}
        <div className="p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Extracted Data</span>
            {result && (
              <div className="flex items-center gap-1">
                <button
                  onClick={handleCopy}
                  className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-50 rounded transition-colors"
                  title="Copy to clipboard"
                >
                  {copied ? <Check className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
                </button>
                <button
                  onClick={handleDownload}
                  className="p-1.5 text-gray-400 hover:text-gray-600 hover:bg-gray-50 rounded transition-colors"
                  title="Download as text"
                >
                  <Download className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>

          <div className="border border-gray-200 rounded-xl overflow-auto bg-gray-50" style={{ maxHeight: '500px' }}>
            {isProcessing ? (
              <div className="flex flex-col items-center justify-center py-16">
                <div className="w-10 h-10 border-2 border-purple-200 border-t-purple-600 rounded-full animate-spin mb-4" />
                <p className="text-sm text-gray-600">Processing...</p>
                <p className="text-xs text-gray-400 mt-1">This may take 30-90 seconds</p>
              </div>
            ) : result ? (
              <div className="p-4 space-y-4">
                {/* Confidence Summary */}
                <div className="flex items-center gap-4 text-xs pb-3 border-b border-gray-200">
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full bg-emerald-500" />
                    <span className="text-gray-600">{result.confidence_summary.high} high</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full bg-amber-500" />
                    <span className="text-gray-600">{result.confidence_summary.medium} medium</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full bg-red-500" />
                    <span className="text-gray-600">{result.confidence_summary.low} low</span>
                  </div>
                </div>

                {/* Fields */}
                <div className="space-y-2">
                  {result.fields.map((field, idx) => (
                    <div
                      key={idx}
                      className={`p-3 rounded-lg border ${
                        field.confidence === 'high' 
                          ? 'border-emerald-200 bg-emerald-50' 
                          : field.confidence === 'medium'
                          ? 'border-amber-200 bg-amber-50'
                          : 'border-gray-200 bg-gray-50'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <p className="text-xs font-medium text-gray-500 mb-1">
                            {field.field_name}
                          </p>
                          <p className={`text-sm ${field.is_empty ? 'text-gray-400 italic' : 'text-gray-900'}`} dir="auto">
                            {field.is_empty ? '(empty)' : field.value}
                          </p>
                        </div>
                        <div className={`w-2 h-2 rounded-full flex-shrink-0 mt-1 ${
                          field.confidence === 'high' ? 'bg-emerald-500' :
                          field.confidence === 'medium' ? 'bg-amber-500' :
                          'bg-red-500'
                        }`} />
                      </div>
                      {field.needs_review && field.review_reason && (
                        <p className="text-xs text-amber-600 mt-2">
                          {field.review_reason}
                        </p>
                      )}
                    </div>
                  ))}
                </div>

                {/* Warnings */}
                {result.hallucination_detected && (
                  <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                    <p className="text-sm text-red-700 font-medium">
                      Potential inaccuracies detected
                    </p>
                    <p className="text-xs text-red-600 mt-1">
                      Please verify the extracted data carefully.
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-16 text-gray-400">
                <p className="text-sm">No results yet</p>
                <p className="text-xs mt-1">Process the file to see extracted data</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
