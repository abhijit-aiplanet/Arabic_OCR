'use client'

import { useState, useEffect } from 'react'
import { Clock, FileText, CheckCircle, XCircle, Download, Trash2, ChevronDown, ChevronUp } from 'lucide-react'
import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface OCRHistoryItem {
  id: string
  user_id: string
  file_name: string
  file_type: string
  file_size: number
  total_pages: number
  extracted_text: string
  status: string
  error_message?: string
  processing_time: number
  settings: any
  blob_url?: string
  created_at: string
}

interface OCRHistoryProps {
  authToken: string | null
  onSelectItem?: (item: OCRHistoryItem) => void
}

export default function OCRHistory({ authToken, onSelectItem }: OCRHistoryProps) {
  const [history, setHistory] = useState<OCRHistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expandedId, setExpandedId] = useState<string | null>(null)

  useEffect(() => {
    if (authToken) {
      fetchHistory()
    }
  }, [authToken])

  const fetchHistory = async () => {
    if (!authToken) return

    setLoading(true)
    setError(null)

    try {
      const response = await axios.get(`${API_URL}/api/history`, {
        headers: {
          'Authorization': `Bearer ${authToken}`
        }
      })

      setHistory(response.data.history)
    } catch (err: any) {
      console.error('Failed to fetch history:', err)
      setError(err.response?.data?.detail || 'Failed to load history')
    } finally {
      setLoading(false)
    }
  }

  const downloadAsMarkdown = (item: OCRHistoryItem) => {
    const markdown = `# ${item.file_name}\n\n**Date:** ${new Date(item.created_at).toLocaleString()}\n**Status:** ${item.status}\n**Processing Time:** ${item.processing_time.toFixed(2)}s\n**Pages:** ${item.total_pages}\n\n## Extracted Text\n\n${item.extracted_text}\n`
    
    const blob = new Blob([markdown], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${item.file_name.replace(/\.[^/.]+$/, '')}_ocr.md`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const formatDate = (dateString: string): string => {
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`
    return date.toLocaleDateString()
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6 bg-red-50 border border-red-200 rounded-lg">
        <p className="text-red-600 font-semibold">Error loading history</p>
        <p className="text-red-500 text-sm mt-1">{error}</p>
        <button 
          onClick={fetchHistory}
          className="mt-3 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
        >
          Retry
        </button>
      </div>
    )
  }

  if (history.length === 0) {
    return (
      <div className="p-8 text-center bg-gray-50 rounded-lg border border-gray-200">
        <FileText className="w-12 h-12 text-gray-400 mx-auto mb-3" />
        <p className="text-gray-600 font-semibold">No OCR history yet</p>
        <p className="text-gray-500 text-sm mt-1">
          Your processed files will appear here
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">
          Recent OCR History
        </h3>
        <span className="text-sm text-gray-500">{history.length} items</span>
      </div>

      <div className="space-y-3">
        {history.map((item) => (
          <div
            key={item.id}
            className="bg-white rounded-lg border border-gray-200 hover:border-blue-300 transition-all shadow-sm overflow-hidden"
          >
            {/* Header */}
            <div 
              className="p-4 cursor-pointer"
              onClick={() => setExpandedId(expandedId === item.id ? null : item.id)}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3 flex-1">
                  {/* Icon */}
                  <div className={`p-2 rounded-lg ${
                    item.status === 'success' 
                      ? 'bg-green-100' 
                      : 'bg-red-100'
                  }`}>
                    {item.status === 'success' ? (
                      <CheckCircle className="w-5 h-5 text-green-600" />
                    ) : (
                      <XCircle className="w-5 h-5 text-red-600" />
                    )}
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <p className="font-semibold text-gray-900 truncate">
                      {item.file_name}
                    </p>
                    <div className="flex items-center gap-2 mt-1 text-xs text-gray-500">
                      <Clock className="w-3 h-3" />
                      <span>{formatDate(item.created_at)}</span>
                      <span>•</span>
                      <span>{formatFileSize(item.file_size)}</span>
                      <span>•</span>
                      <span>{item.processing_time.toFixed(1)}s</span>
                      {item.total_pages > 1 && (
                        <>
                          <span>•</span>
                          <span>{item.total_pages} pages</span>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Expand icon */}
                  <div>
                    {expandedId === item.id ? (
                      <ChevronUp className="w-5 h-5 text-gray-400" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-gray-400" />
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Expanded Content */}
            {expandedId === item.id && (
              <div className="px-4 pb-4 border-t border-gray-100">
                <div className="mt-3 space-y-3">
                  {/* Status and Error */}
                  {item.status === 'error' && item.error_message && (
                    <div className="p-3 bg-red-50 rounded-lg">
                      <p className="text-sm text-red-600 font-semibold">Error:</p>
                      <p className="text-sm text-red-500 mt-1">{item.error_message}</p>
                    </div>
                  )}

                  {/* Extracted Text Preview */}
                  {item.extracted_text && (
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <p className="text-xs font-semibold text-gray-700 mb-2">Extracted Text:</p>
                      <p className="text-sm text-gray-600 line-clamp-3">
                        {item.extracted_text}
                      </p>
                    </div>
                  )}

                  {/* Actions */}
                  <div className="flex gap-2">
                    {item.status === 'success' && item.extracted_text && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          downloadAsMarkdown(item)
                        }}
                        className="flex items-center gap-2 px-3 py-2 bg-blue-50 text-blue-600 rounded-lg hover:bg-blue-100 transition-colors text-sm font-semibold"
                      >
                        <Download className="w-4 h-4" />
                        Download
                      </button>
                    )}
                    {item.blob_url && (
                      <a
                        href={item.blob_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-2 px-3 py-2 bg-gray-50 text-gray-600 rounded-lg hover:bg-gray-100 transition-colors text-sm font-semibold"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <FileText className="w-4 h-4" />
                        View File
                      </a>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

