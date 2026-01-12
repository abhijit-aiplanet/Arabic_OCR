'use client'

import { useState, useEffect } from 'react'
import { PDFPageResult, QueueStatus } from '@/lib/api'
import { ChevronLeft, ChevronRight, Download, FileText, CheckCircle, XCircle, Clock } from 'lucide-react'
import UniversalRenderer from '@/components/UniversalRenderer'

interface PDFProcessorProps {
  totalPages: number
  results: PDFPageResult[]
  isProcessing: boolean
  queueStatus?: QueueStatus | null
  elapsedTime?: number
  selectedPage?: number
  onPageSelect?: (page: number) => void
}

export default function PDFProcessor({
  totalPages,
  results,
  isProcessing,
  queueStatus,
  elapsedTime = 0,
  selectedPage,
  onPageSelect
}: PDFProcessorProps) {
  const [currentPage, setCurrentPage] = useState(1)
  const currentResult = results.find(r => r.page_number === currentPage)
  
  // Handle page selection - internal or external
  const handlePageClick = (pageNum: number) => {
    setCurrentPage(pageNum)
    if (onPageSelect) {
      onPageSelect(pageNum)
    }
  }

  // Auto-navigate to the latest completed page
  useEffect(() => {
    if (results.length > 0) {
      const latestPage = results[results.length - 1].page_number
      if (latestPage > currentPage) {
        handlePageClick(latestPage)
      }
    }
  }, [results])
  
  // Sync with external selectedPage if provided
  useEffect(() => {
    if (selectedPage !== undefined && selectedPage > 0 && selectedPage !== currentPage) {
      setCurrentPage(selectedPage)
    }
  }, [selectedPage])

  const handleDownloadMarkdown = () => {
    const markdown = results
      .filter(r => r.status === 'success')
      .sort((a, b) => a.page_number - b.page_number)
      .map(r => `# Page ${r.page_number}\n\n${r.extracted_text}\n\n---\n`)
      .join('\n')

    const blob = new Blob([markdown], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'ocr-results.md'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const successCount = results.filter(r => r.status === 'success').length

  return (
    <div className="bg-white rounded-2xl border border-gray-100 overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b border-gray-100">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-gray-900">
              Results {isProcessing && <span className="text-gray-400 font-normal">(Processing...)</span>}
            </h2>
            <p className="text-sm text-gray-500 mt-0.5">
            {results.length} of {totalPages} pages completed · {successCount} successful
          </p>
        </div>
          <button
            onClick={handleDownloadMarkdown}
            disabled={results.length === 0}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-gray-900 hover:bg-gray-800 rounded-lg transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Download className="w-4 h-4" />
            Download All
          </button>
        </div>
        
        {/* ETA Display for PDF Processing */}
        {isProcessing && queueStatus && (
          <div className="mt-3 bg-blue-50 border border-blue-100 rounded-lg p-3">
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-2 text-blue-900">
                <Clock className="w-4 h-4" />
                <span className="font-medium">
                  {Math.floor(elapsedTime / 60)}:{String(elapsedTime % 60).padStart(2, '0')} elapsed
                </span>
              </div>
              <span className="text-blue-700">
                Est: {queueStatus.estimated_wait_display}
              </span>
            </div>
            <div className="mt-2 text-xs text-blue-600">
              {queueStatus.message}
              {queueStatus.queue_length > 0 && (
                <span className={`ml-2 px-1.5 py-0.5 rounded ${
                  queueStatus.status === 'very_high_load' ? 'bg-red-100 text-red-700' :
                  queueStatus.status === 'high_load' ? 'bg-amber-100 text-amber-700' :
                  'bg-blue-100 text-blue-700'
                }`}>
                  {queueStatus.queue_length} in queue
                </span>
              )}
            </div>
            <div className="mt-2 w-full bg-blue-200 rounded-full h-1">
              <div 
                className="bg-blue-600 h-1 rounded-full transition-all duration-500"
                style={{ width: `${(results.length / totalPages) * 100}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Page Thumbnails */}
      <div className="border-b border-gray-100 bg-gray-50 px-5 py-4 overflow-x-auto">
        <div className="flex gap-3">
          {Array.from({ length: totalPages }, (_, i) => i + 1).map(pageNum => {
            const result = results.find(r => r.page_number === pageNum)
            const isCompleted = !!result
            const isSuccess = result?.status === 'success'
            const isCurrent = pageNum === currentPage
            const isCurrentlyProcessing = !isCompleted && results.length >= pageNum - 1

            return (
              <button
                key={pageNum}
                onClick={() => handlePageClick(pageNum)}
                disabled={!isCompleted}
                className={`relative flex-shrink-0 w-16 h-20 rounded-lg border-2 transition-all overflow-hidden ${
                  isCurrent
                    ? 'border-gray-900 shadow-md'
                    : isCompleted
                    ? 'border-gray-200 hover:border-gray-300'
                    : 'border-gray-100 opacity-50 cursor-not-allowed'
                }`}
              >
                {result?.page_image ? (
                  <img
                    src={`data:image/png;base64,${result.page_image}`}
                    alt={`Page ${pageNum}`}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center bg-gray-100">
                    {isCurrentlyProcessing ? (
                      <div className="w-4 h-4 border-2 border-gray-300 border-t-gray-600 rounded-full animate-spin" />
                    ) : (
                      <span className="text-xs text-gray-400">{pageNum}</span>
                    )}
                  </div>
                )}
                
                {/* Status Badge */}
                {isCompleted && (
                  <div className={`absolute top-1 right-1 w-4 h-4 rounded-full flex items-center justify-center ${
                    isSuccess ? 'bg-emerald-500' : 'bg-red-500'
                  }`}>
                    {isSuccess ? (
                      <CheckCircle className="w-3 h-3 text-white" />
                    ) : (
                      <XCircle className="w-3 h-3 text-white" />
                    )}
                  </div>
                )}
              </button>
            )
          })}
        </div>
      </div>

      {/* Main Content */}
      {currentResult ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-5">
          {/* Left: Original PDF Page */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-gray-900">
                Page {currentPage}
              </h3>
              {currentResult.status === 'success' ? (
                <span className="inline-flex items-center gap-1 text-xs font-medium text-emerald-700 bg-emerald-50 px-2 py-1 rounded-full">
                  <CheckCircle className="w-3 h-3" />
                  Success
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 text-xs font-medium text-red-700 bg-red-50 px-2 py-1 rounded-full">
                  <XCircle className="w-3 h-3" />
                  Error
                </span>
              )}
            </div>
            
            {currentResult.page_image && (
              <div className="border border-gray-200 rounded-xl overflow-hidden bg-gray-50">
                <img
                  src={`data:image/png;base64,${currentResult.page_image}`}
                  alt={`Page ${currentPage}`}
                  className="w-full h-auto"
                />
              </div>
            )}
          </div>

          {/* Right: OCR Output */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-gray-900">
              Extracted Text
            </h3>
            
            <div className="min-h-[400px]">
              {currentResult.status === 'success' ? (
                <UniversalRenderer
                  text={currentResult.extracted_text}
                  isProcessing={false}
                  isEditable={false}
                  confidence={currentResult.confidence || null}
                />
              ) : (
                <div className="bg-red-50 border border-red-100 rounded-xl p-4 min-h-[400px]">
                  <div className="flex items-start gap-3">
                    <XCircle className="w-5 h-5 text-red-500 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium text-red-900">Failed to process this page</p>
                      <p className="text-sm text-red-600 mt-1">{currentResult.error || 'Unknown error'}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="p-12 text-center">
          <div className="relative w-12 h-12 mx-auto mb-4">
            <div className="w-12 h-12 border-2 border-gray-200 rounded-full"></div>
            <div className="absolute inset-0 w-12 h-12 border-2 border-transparent border-t-gray-900 rounded-full animate-spin"></div>
          </div>
          <p className="text-sm font-medium text-gray-900">Processing page {currentPage}...</p>
          <p className="text-xs text-gray-500 mt-1">This may take a moment</p>
        </div>
      )}

      {/* Navigation */}
      {totalPages > 1 && (
        <div className="border-t border-gray-100 px-5 py-4 flex items-center justify-between">
          <button
            onClick={() => handlePageClick(Math.max(1, currentPage - 1))}
            disabled={currentPage === 1}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded-lg transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <ChevronLeft className="w-4 h-4" />
            Previous
          </button>

          <span className="text-sm text-gray-500">
            {currentPage} of {totalPages}
          </span>

          <button
            onClick={() => handlePageClick(Math.min(totalPages, currentPage + 1))}
            disabled={currentPage === totalPages}
            className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded-lg transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Next
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  )
}
