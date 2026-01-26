'use client'

import { useState } from 'react'
import { ZoomIn, ZoomOut, RotateCw, Copy, Check, Download, ChevronLeft, ChevronRight, Palette } from 'lucide-react'
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
  const [showConfidence, setShowConfidence] = useState(false)

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

  // Get confidence color for a line based on field data
  const getLineConfidenceStyle = (line: string) => {
    if (!showConfidence || !result?.fields) return {}
    
    // Find matching field
    const field = result.fields.find(f => line.includes(f.field_name))
    if (!field) return {}
    
    if (field.confidence === 'high') {
      return { backgroundColor: 'rgba(16, 185, 129, 0.15)', borderLeft: '3px solid rgb(16, 185, 129)' }
    } else if (field.confidence === 'medium') {
      return { backgroundColor: 'rgba(245, 158, 11, 0.15)', borderLeft: '3px solid rgb(245, 158, 11)' }
    } else {
      return { backgroundColor: 'rgba(239, 68, 68, 0.15)', borderLeft: '3px solid rgb(239, 68, 68)' }
    }
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
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Extracted Text</span>
            {result && (
              <div className="flex items-center gap-1">
                {/* Confidence Toggle */}
                <button
                  onClick={() => setShowConfidence(!showConfidence)}
                  className={`p-1.5 rounded transition-colors ${
                    showConfidence 
                      ? 'text-purple-600 bg-purple-50' 
                      : 'text-gray-400 hover:text-gray-600 hover:bg-gray-50'
                  }`}
                  title={showConfidence ? 'Hide confidence colors' : 'Show confidence colors'}
                >
                  <Palette className="w-4 h-4" />
                </button>
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

          <div className="border border-gray-200 rounded-xl overflow-auto bg-white" style={{ maxHeight: '500px' }}>
            {isProcessing ? (
              <div className="flex flex-col items-center justify-center py-16">
                <div className="w-10 h-10 border-2 border-purple-200 border-t-purple-600 rounded-full animate-spin mb-4" />
                <p className="text-sm text-gray-600">Processing...</p>
                <p className="text-xs text-gray-400 mt-1">This may take 30-90 seconds</p>
              </div>
            ) : result ? (
              <div className="p-4">
                {/* Confidence Legend (only show when toggle is on) */}
                {showConfidence && (
                  <div className="flex items-center gap-4 text-xs pb-3 mb-3 border-b border-gray-200">
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
                )}

                {/* Raw Text Output */}
                <div 
                  className="font-mono text-sm leading-relaxed whitespace-pre-wrap"
                  dir="auto"
                  style={{ fontFamily: "'Noto Sans Arabic', 'Arial', monospace" }}
                >
                  {result.raw_text ? (
                    result.raw_text.split('\n').map((line, idx) => (
                      <div 
                        key={idx} 
                        className={`py-1 px-2 -mx-2 rounded ${showConfidence ? 'my-0.5' : ''}`}
                        style={getLineConfidenceStyle(line)}
                      >
                        {line || '\u00A0'}
                      </div>
                    ))
                  ) : (
                    <div className="text-gray-400 italic">No text extracted</div>
                  )}
                </div>

                {/* Warnings */}
                {result.hallucination_detected && (
                  <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
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
