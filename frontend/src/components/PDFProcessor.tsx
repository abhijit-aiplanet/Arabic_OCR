'use client'

import { useState, useEffect } from 'react'
import { PDFPageResult } from '@/lib/api'
import { ChevronLeft, ChevronRight, Download, FileText } from 'lucide-react'
import UniversalRenderer from '@/components/UniversalRenderer'

interface PDFProcessorProps {
  totalPages: number
  results: PDFPageResult[]
  isProcessing: boolean
}

export default function PDFProcessor({
  totalPages,
  results,
  isProcessing
}: PDFProcessorProps) {
  const [currentPage, setCurrentPage] = useState(1)
  const currentResult = results.find(r => r.page_number === currentPage)

  // Auto-navigate to the latest completed page
  useEffect(() => {
    if (results.length > 0) {
      const latestPage = results[results.length - 1].page_number
      if (latestPage > currentPage) {
        setCurrentPage(latestPage)
      }
    }
  }, [results])

  const handleDownloadMarkdown = () => {
    // Combine all successful results
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

  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FileText className="w-6 h-6 text-white" />
            <div>
              <h2 className="text-xl font-bold text-white">
                OCR Results {isProcessing && '(Processing...)'}
              </h2>
              <p className="text-blue-100 text-sm">
                Total Pages: {totalPages} | Completed: {results.length} | Successful: {results.filter(r => r.status === 'success').length}
              </p>
            </div>
          </div>
          <button
            onClick={handleDownloadMarkdown}
            disabled={results.length === 0}
            className="bg-white/20 hover:bg-white/30 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Download className="w-4 h-4" />
            Download Markdown
          </button>
        </div>
      </div>

      {/* Page Thumbnails */}
      <div className="border-b border-gray-200 bg-gray-50 px-6 py-4 overflow-x-auto">
        <div className="flex gap-3">
          {Array.from({ length: totalPages }, (_, i) => i + 1).map(pageNum => {
            const result = results.find(r => r.page_number === pageNum)
            const isCompleted = !!result
            const isSuccess = result?.status === 'success'
            const isCurrent = pageNum === currentPage
            const isProcessing = !isCompleted && results.length >= pageNum - 1

            return (
              <button
                key={pageNum}
                onClick={() => setCurrentPage(pageNum)}
                disabled={!isCompleted}
                className={`relative flex-shrink-0 w-20 h-24 rounded-lg border-2 transition-all ${
                  isCurrent
                    ? 'border-blue-500 shadow-lg scale-105'
                    : isCompleted
                    ? 'border-green-300 hover:border-green-500'
                    : 'border-gray-300'
                } ${!isCompleted && 'opacity-50 cursor-not-allowed'}`}
              >
                {result?.page_image ? (
                  <img
                    src={`data:image/png;base64,${result.page_image}`}
                    alt={`Page ${pageNum}`}
                    className="w-full h-full object-cover rounded"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center bg-gray-100 rounded">
                    {isProcessing ? (
                      <div className="animate-spin w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full" />
                    ) : (
                      <span className="text-gray-400 text-sm">Page {pageNum}</span>
                    )}
                  </div>
                )}
                
                {/* Status Badge */}
                {isCompleted && (
                  <div className={`absolute -top-2 -right-2 px-2 py-0.5 rounded-full text-xs font-bold ${
                    isSuccess ? 'bg-green-500 text-white' : 'bg-red-500 text-white'
                  }`}>
                    {isSuccess ? '✓' : '✗'}
                  </div>
                )}
                
                {/* Page Number */}
                <div className="absolute bottom-0 left-0 right-0 bg-black/70 text-white text-xs py-1 text-center rounded-b">
                  Page {pageNum}
                </div>
              </button>
            )
          })}
        </div>
      </div>

      {/* Main Content */}
      {currentResult ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6">
          {/* Left: Original PDF Page */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-gray-900">
                Original PDF - Page {currentPage}
              </h3>
              {currentResult.status === 'success' && (
                <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm font-medium">
                  SUCCESS
                </span>
              )}
              {currentResult.status === 'error' && (
                <span className="bg-red-100 text-red-800 px-3 py-1 rounded-full text-sm font-medium">
                  ERROR
                </span>
              )}
            </div>
            
            {currentResult.page_image && (
              <div className="border border-gray-300 rounded-lg overflow-hidden bg-gray-50">
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
            <h3 className="text-lg font-semibold text-gray-900">
              OCR Output - Page {currentPage}
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
                <div className="border border-gray-300 rounded-lg bg-white p-4 min-h-[400px] max-h-[600px] overflow-y-auto text-red-600">
                  <p className="font-semibold">Error processing this page:</p>
                  <p className="mt-2">{currentResult.error || 'Unknown error'}</p>
                </div>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="p-12 text-center text-gray-500">
          <div className="animate-pulse">
            <div className="w-16 h-16 bg-blue-100 rounded-full mx-auto mb-4 flex items-center justify-center">
              <div className="animate-spin w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full" />
            </div>
            <p className="text-lg font-medium">Processing page {currentPage}...</p>
            <p className="text-sm mt-2">Please wait while we extract the text</p>
          </div>
        </div>
      )}

      {/* Navigation */}
      {totalPages > 1 && (
        <div className="border-t border-gray-200 px-6 py-4 flex items-center justify-between bg-gray-50">
          <button
            onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
            disabled={currentPage === 1}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-blue-600"
          >
            <ChevronLeft className="w-4 h-4" />
            Previous
          </button>

          <span className="text-gray-700 font-medium">
            Page {currentPage} of {totalPages}
          </span>

          <button
            onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
            disabled={currentPage === totalPages}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-blue-600"
          >
            Next
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  )
}

