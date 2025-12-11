'use client'

import { useState, useEffect } from 'react'
import { Clock, FileText, CheckCircle, XCircle, Download, Trash2, ChevronDown, ChevronUp, Edit2, Save, X as XIcon, ArrowLeft } from 'lucide-react'
import axios from 'axios'
import { updateHistoryText } from '@/lib/api'
import toast from 'react-hot-toast'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface OCRHistoryItem {
  id: string
  user_id: string
  file_name: string
  file_type: string
  file_size: number
  total_pages: number
  extracted_text: string
  edited_text?: string
  edited_at?: string
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
  const [selectedItem, setSelectedItem] = useState<OCRHistoryItem | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editText, setEditText] = useState('')

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
    const textToDownload = item.edited_text || item.extracted_text
    const markdown = `# ${item.file_name}\n\n**Date:** ${new Date(item.created_at).toLocaleString()}\n**Status:** ${item.status}\n**Processing Time:** ${item.processing_time.toFixed(2)}s\n**Pages:** ${item.total_pages}\n${item.edited_at ? `**Last Edited:** ${new Date(item.edited_at).toLocaleString()}\n` : ''}\n## ${item.edited_text ? 'Edited' : 'Extracted'} Text\n\n${textToDownload}\n`
    
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

  const handleEdit = (item: OCRHistoryItem) => {
    setEditingId(item.id)
    setEditText(item.edited_text || item.extracted_text)
  }

  const handleSaveEdit = async (item: OCRHistoryItem) => {
    if (!authToken) {
      toast.error('Authentication required')
      return
    }

    try {
      await updateHistoryText(item.id, editText, authToken)
      
      // Update local state
      setHistory(history.map(h => 
        h.id === item.id 
          ? { ...h, edited_text: editText, edited_at: new Date().toISOString() }
          : h
      ))
      
      setEditingId(null)
      toast.success('Changes saved!')
    } catch (err: any) {
      console.error('Failed to save edit:', err)
      toast.error(err.message || 'Failed to save changes')
    }
  }

  const handleCancelEdit = () => {
    setEditingId(null)
    setEditText('')
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

  // If item is selected, show detail view
  if (selectedItem) {
    const displayText = selectedItem.edited_text || selectedItem.extracted_text

    return (
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4 sticky top-0 z-10">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => {
                  setSelectedItem(null)
                  setEditingId(null)
                }}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <ArrowLeft className="w-5 h-5 text-gray-600" />
              </button>
              <div>
                <h2 className="text-xl font-bold text-gray-900">{selectedItem.file_name}</h2>
                <div className="flex items-center gap-2 mt-1 text-sm text-gray-500">
                  <Clock className="w-4 h-4" />
                  <span>{formatDate(selectedItem.created_at)}</span>
                  {selectedItem.edited_at && (
                    <>
                      <span>•</span>
                      <span>Edited {formatDate(selectedItem.edited_at)}</span>
                    </>
                  )}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {editingId !== selectedItem.id && (
                <>
                  {selectedItem.status === 'success' && (
                    <>
                      <button
                        onClick={() => handleEdit(selectedItem)}
                        className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-gray-600 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                      >
                        <Edit2 className="w-4 h-4" />
                        Edit
                      </button>
                      <button
                        onClick={() => downloadAsMarkdown(selectedItem)}
                        className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-colors"
                      >
                        <Download className="w-4 h-4" />
                        Download
                      </button>
                    </>
                  )}
                  {selectedItem.blob_url && (
                    <a
                      href={selectedItem.blob_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-gray-600 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                    >
                      <FileText className="w-4 h-4" />
                      View File
                    </a>
                  )}
                </>
              )}
              {editingId === selectedItem.id && (
                <>
                  <button
                    onClick={() => handleCancelEdit()}
                    className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-gray-600 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                  >
                    <XIcon className="w-4 h-4" />
                    Cancel
                  </button>
                  <button
                    onClick={() => handleSaveEdit(selectedItem)}
                    className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
                  >
                    <Save className="w-4 h-4" />
                    Save Changes
                  </button>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="max-w-5xl mx-auto px-6 py-8">
          {/* Metadata */}
          <div className="bg-white rounded-lg border border-gray-200 p-6 mb-6">
            <div className="grid grid-cols-4 gap-4">
              <div>
                <p className="text-xs text-gray-500 mb-1">Status</p>
                <div className="flex items-center gap-2">
                  {selectedItem.status === 'success' ? (
                    <CheckCircle className="w-4 h-4 text-green-600" />
                  ) : (
                    <XCircle className="w-4 h-4 text-red-600" />
                  )}
                  <span className="text-sm font-semibold capitalize">{selectedItem.status}</span>
                </div>
              </div>
              <div>
                <p className="text-xs text-gray-500 mb-1">File Size</p>
                <p className="text-sm font-semibold">{formatFileSize(selectedItem.file_size)}</p>
              </div>
              <div>
                <p className="text-xs text-gray-500 mb-1">Processing Time</p>
                <p className="text-sm font-semibold">{selectedItem.processing_time.toFixed(2)}s</p>
              </div>
              <div>
                <p className="text-xs text-gray-500 mb-1">Pages</p>
                <p className="text-sm font-semibold">{selectedItem.total_pages}</p>
              </div>
            </div>
          </div>

          {/* Error message if failed */}
          {selectedItem.status === 'error' && selectedItem.error_message && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-6 mb-6">
              <p className="text-sm font-semibold text-red-600 mb-2">Error:</p>
              <p className="text-sm text-red-500">{selectedItem.error_message}</p>
            </div>
          )}

          {/* Extracted Text */}
          {selectedItem.extracted_text && (
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  {selectedItem.edited_text ? 'Edited Text' : 'Extracted Text'}
                </h3>
                <div className="text-sm text-gray-500">
                  {displayText.length} characters • {displayText.split(/\s+/).filter(Boolean).length} words
                </div>
              </div>

              {editingId === selectedItem.id ? (
                <textarea
                  value={editText}
                  onChange={(e) => setEditText(e.target.value)}
                  className="w-full min-h-[500px] p-4 text-base text-gray-900 bg-gray-50 border-2 border-blue-300 rounded-lg focus:outline-none focus:border-blue-500 resize-y font-sans leading-relaxed"
                  placeholder="Edit your text here..."
                  dir="auto"
                />
              ) : (
                <pre className="whitespace-pre-wrap font-sans text-base leading-relaxed text-gray-900 text-left bg-gray-50 p-6 rounded-lg" dir="auto">
                  {displayText}
                </pre>
              )}
            </div>
          )}
        </div>
      </div>
    )
  }

  // List view
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
            onClick={() => setSelectedItem(item)}
            className="bg-white rounded-lg border border-gray-200 hover:border-blue-300 hover:shadow-md cursor-pointer transition-all p-4"
          >
            <div className="flex items-start gap-3">
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
                <p className="text-sm text-gray-600 line-clamp-2 mt-1">
                  {item.edited_text || item.extracted_text || item.error_message || 'No text'}
                </p>
                <div className="flex items-center gap-2 mt-2 text-xs text-gray-500">
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
                  {item.edited_at && (
                    <>
                      <span>•</span>
                      <span className="text-blue-600">Edited</span>
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

