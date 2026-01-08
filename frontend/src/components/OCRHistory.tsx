'use client'

import { useState, useEffect } from 'react'
import { Clock, FileText, CheckCircle, XCircle, Download, Trash2, Edit2, Save, X as XIcon, ArrowLeft, File, Image, Calendar, Timer, Layers, Search, Filter, ChevronRight } from 'lucide-react'
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
  getToken: () => Promise<string | null>
  onSelectItem?: (item: OCRHistoryItem) => void
}

export default function OCRHistory({ getToken, onSelectItem }: OCRHistoryProps) {
  const [history, setHistory] = useState<OCRHistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedItem, setSelectedItem] = useState<OCRHistoryItem | null>(null)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editText, setEditText] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [filterType, setFilterType] = useState<'all' | 'image' | 'pdf'>('all')

  useEffect(() => {
    fetchHistory()
  }, [])

  const fetchHistory = async () => {
    setLoading(true)
    setError(null)

    try {
      const authToken = await getToken()
      if (!authToken) {
        setError('Authentication required. Please sign in.')
        setLoading(false)
        return
      }

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
    const markdown = `# ${item.file_name}\n\n**Date:** ${new Date(item.created_at).toLocaleString()}\n**Status:** ${item.status}\n**Processing Time:** ${item.processing_time?.toFixed(2) || 0}s\n**Pages:** ${item.total_pages || 1}\n${item.edited_at ? `**Last Edited:** ${new Date(item.edited_at).toLocaleString()}\n` : ''}\n## ${item.edited_text ? 'Edited' : 'Extracted'} Text\n\n${textToDownload}\n`
    
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
    try {
      const authToken = await getToken()
      if (!authToken) {
        toast.error('Authentication required. Please sign in.')
        return
      }

      await updateHistoryText(item.id, editText, authToken)
      
      // Update local state
      setHistory(history.map(h => 
        h.id === item.id 
          ? { ...h, edited_text: editText, edited_at: new Date().toISOString() }
          : h
      ))
      
      if (selectedItem?.id === item.id) {
        setSelectedItem({ ...selectedItem, edited_text: editText, edited_at: new Date().toISOString() })
      }
      
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
    if (!bytes || bytes === 0) return '-'
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

  const formatFullDate = (dateString: string): string => {
    return new Date(dateString).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const formatProcessingTime = (seconds: number): string => {
    if (!seconds || seconds === 0) return '-'
    if (seconds < 60) return `${seconds.toFixed(1)}s`
    const mins = Math.floor(seconds / 60)
    const secs = (seconds % 60).toFixed(0)
    return `${mins}m ${secs}s`
  }

  // Filter history based on search and type
  const filteredHistory = history.filter(item => {
    const matchesSearch = searchQuery === '' || 
      item.file_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (item.extracted_text || '').toLowerCase().includes(searchQuery.toLowerCase())
    
    const matchesType = filterType === 'all' || item.file_type === filterType

    return matchesSearch && matchesType
  })

  // Detail view when an item is selected
  if (selectedItem) {
    const displayText = selectedItem.edited_text || selectedItem.extracted_text

    return (
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 px-6 py-4 sticky top-0 z-10">
          <div className="flex items-center justify-between max-w-7xl mx-auto">
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
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${selectedItem.file_type === 'pdf' ? 'bg-red-100' : 'bg-blue-100'}`}>
                  {selectedItem.file_type === 'pdf' ? (
                    <File className="w-5 h-5 text-red-600" />
                  ) : (
                    <Image className="w-5 h-5 text-blue-600" />
                  )}
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-gray-900 truncate max-w-md">{selectedItem.file_name}</h2>
                  <p className="text-sm text-gray-500">{formatFullDate(selectedItem.created_at)}</p>
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
                        className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                      >
                        <Edit2 className="w-4 h-4" />
                        Edit
                      </button>
                      <button
                        onClick={() => downloadAsMarkdown(selectedItem)}
                        className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-gray-900 hover:bg-gray-800 rounded-lg transition-colors"
                      >
                        <Download className="w-4 h-4" />
                        Download
                      </button>
                    </>
                  )}
                </>
              )}
              {editingId === selectedItem.id && (
                <>
                  <button
                    onClick={() => handleCancelEdit()}
                    className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                  >
                    <XIcon className="w-4 h-4" />
                    Cancel
                  </button>
                  <button
                    onClick={() => handleSaveEdit(selectedItem)}
                    className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-gray-900 hover:bg-gray-800 rounded-lg transition-colors"
                  >
                    <Save className="w-4 h-4" />
                    Save
                  </button>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="max-w-7xl mx-auto px-6 py-6">
          {/* Stats Cards */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg ${selectedItem.status === 'success' ? 'bg-emerald-100' : 'bg-red-100'}`}>
                  {selectedItem.status === 'success' ? (
                    <CheckCircle className="w-5 h-5 text-emerald-600" />
                  ) : (
                    <XCircle className="w-5 h-5 text-red-600" />
                  )}
                </div>
                <div>
                  <p className="text-xs text-gray-500 uppercase tracking-wide">Status</p>
                  <p className="text-sm font-semibold text-gray-900 capitalize">{selectedItem.status}</p>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-purple-100">
                  <Layers className="w-5 h-5 text-purple-600" />
                </div>
                <div>
                  <p className="text-xs text-gray-500 uppercase tracking-wide">File Size</p>
                  <p className="text-sm font-semibold text-gray-900">{formatFileSize(selectedItem.file_size)}</p>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-amber-100">
                  <Timer className="w-5 h-5 text-amber-600" />
                </div>
                <div>
                  <p className="text-xs text-gray-500 uppercase tracking-wide">Processing Time</p>
                  <p className="text-sm font-semibold text-gray-900">{formatProcessingTime(selectedItem.processing_time)}</p>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-xl border border-gray-200 p-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-blue-100">
                  <FileText className="w-5 h-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-xs text-gray-500 uppercase tracking-wide">Pages</p>
                  <p className="text-sm font-semibold text-gray-900">{selectedItem.total_pages || 1}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Error message if failed */}
          {selectedItem.status === 'error' && selectedItem.error_message && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-6">
              <div className="flex items-start gap-3">
                <XCircle className="w-5 h-5 text-red-500 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-red-800">Error Details</p>
                  <p className="text-sm text-red-600 mt-1">{selectedItem.error_message}</p>
                </div>
              </div>
            </div>
          )}

          {/* Two Column Layout */}
          {selectedItem.extracted_text && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left: Original File Viewer */}
              {selectedItem.blob_url && (
                <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
                  <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
                    <h3 className="text-sm font-semibold text-gray-900">Original File</h3>
                  </div>
                  <div className="p-4">
                    {selectedItem.file_type === 'pdf' ? (
                      <div className="bg-gray-100 rounded-lg overflow-hidden">
                        <iframe
                          src={selectedItem.blob_url}
                          className="w-full h-[550px] border-0"
                          title="PDF Viewer"
                        />
                      </div>
                    ) : (
                      <div className="bg-gray-100 rounded-lg overflow-hidden flex items-center justify-center">
                        <img
                          src={selectedItem.blob_url}
                          alt={selectedItem.file_name}
                          className="max-w-full h-auto max-h-[550px] object-contain"
                        />
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Right: Extracted Text */}
              <div className={`bg-white rounded-xl border border-gray-200 overflow-hidden ${!selectedItem.blob_url ? 'lg:col-span-2' : ''}`}>
                <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-gray-900">
                    {selectedItem.edited_text ? 'Edited Text' : 'Extracted Text'}
                  </h3>
                  <span className="text-xs text-gray-500">
                    {displayText.length.toLocaleString()} chars
                  </span>
                </div>
                <div className="p-4">
                  {editingId === selectedItem.id ? (
                    <textarea
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                      className="w-full min-h-[500px] p-4 text-sm text-gray-900 bg-gray-50 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent resize-y font-mono leading-relaxed"
                      placeholder="Edit your text here..."
                      dir="auto"
                    />
                  ) : (
                    <div className="min-h-[500px] max-h-[550px] overflow-y-auto">
                      <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-gray-900 text-left bg-gray-50 p-4 rounded-lg" dir="auto">
                        {displayText}
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Full width text if no blob_url */}
          {selectedItem.extracted_text && !selectedItem.blob_url && (
            <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-gray-900">
                  {selectedItem.edited_text ? 'Edited Text' : 'Extracted Text'}
                </h3>
                <span className="text-xs text-gray-500">
                  {displayText.length.toLocaleString()} chars
                </span>
              </div>
              <div className="p-4">
                {editingId === selectedItem.id ? (
                  <textarea
                    value={editText}
                    onChange={(e) => setEditText(e.target.value)}
                    className="w-full min-h-[400px] p-4 text-sm text-gray-900 bg-gray-50 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent resize-y font-mono leading-relaxed"
                    placeholder="Edit your text here..."
                    dir="auto"
                  />
                ) : (
                  <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-gray-900 text-left bg-gray-50 p-4 rounded-lg" dir="auto">
                    {displayText}
                  </pre>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    )
  }

  // Loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-10 w-10 border-2 border-gray-300 border-t-gray-900 mx-auto"></div>
          <p className="mt-4 text-sm text-gray-600">Loading history...</p>
        </div>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className="max-w-md mx-auto mt-12">
        <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-center">
          <XCircle className="w-10 h-10 text-red-400 mx-auto mb-3" />
          <p className="text-red-800 font-medium">Error loading history</p>
          <p className="text-red-600 text-sm mt-1">{error}</p>
          <button 
            onClick={fetchHistory}
            className="mt-4 px-4 py-2 bg-red-600 text-white text-sm font-medium rounded-lg hover:bg-red-700 transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    )
  }

  // Empty state
  if (history.length === 0) {
    return (
      <div className="max-w-md mx-auto mt-12">
        <div className="text-center py-12 px-6 bg-gray-50 rounded-xl border border-gray-200">
          <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-4">
            <FileText className="w-8 h-8 text-gray-400" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-1">No history yet</h3>
          <p className="text-sm text-gray-600">
            Your processed documents will appear here
          </p>
        </div>
      </div>
    )
  }

  // List view
  return (
    <div className="space-y-6">
      {/* Search and Filter Bar */}
      <div className="flex items-center gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search files..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-10 pr-4 py-2.5 bg-white border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent"
          />
        </div>
        <div className="flex items-center gap-1 bg-gray-100 p-1 rounded-lg">
          <button
            onClick={() => setFilterType('all')}
            className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
              filterType === 'all' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            All
          </button>
          <button
            onClick={() => setFilterType('image')}
            className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
              filterType === 'image' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            Images
          </button>
          <button
            onClick={() => setFilterType('pdf')}
            className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
              filterType === 'pdf' ? 'bg-white text-gray-900 shadow-sm' : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            PDFs
          </button>
        </div>
      </div>

      {/* Results count */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-gray-600">
          {filteredHistory.length} {filteredHistory.length === 1 ? 'document' : 'documents'}
        </p>
      </div>

      {/* History List */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden divide-y divide-gray-100">
        {filteredHistory.map((item) => (
          <div
            key={item.id}
            onClick={() => setSelectedItem(item)}
            className="flex items-center gap-4 p-4 hover:bg-gray-50 cursor-pointer transition-colors group"
          >
            {/* File Type Icon */}
            <div className={`flex-shrink-0 p-2.5 rounded-lg ${
              item.file_type === 'pdf' ? 'bg-red-50' : 'bg-blue-50'
            }`}>
              {item.file_type === 'pdf' ? (
                <File className={`w-5 h-5 ${item.file_type === 'pdf' ? 'text-red-500' : 'text-blue-500'}`} />
              ) : (
                <Image className="w-5 h-5 text-blue-500" />
              )}
            </div>

            {/* File Info */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <p className="font-medium text-gray-900 truncate">
                  {item.file_name}
                </p>
                {item.edited_at && (
                  <span className="flex-shrink-0 px-2 py-0.5 text-xs font-medium text-blue-700 bg-blue-50 rounded-full">
                    Edited
                  </span>
                )}
              </div>
              <p className="text-sm text-gray-500 truncate mt-0.5">
                {(item.edited_text || item.extracted_text || item.error_message || 'No text').slice(0, 80)}
                {(item.edited_text || item.extracted_text || '').length > 80 ? '...' : ''}
              </p>
            </div>

            {/* Meta Info */}
            <div className="flex-shrink-0 flex items-center gap-6 text-sm text-gray-500">
              <div className="flex items-center gap-1.5" title="Pages">
                <FileText className="w-3.5 h-3.5" />
                <span>{item.total_pages || 1}</span>
              </div>
              <div className="flex items-center gap-1.5" title="Processing Time">
                <Timer className="w-3.5 h-3.5" />
                <span>{formatProcessingTime(item.processing_time)}</span>
              </div>
              <div className="flex items-center gap-1.5 w-20" title="Date">
                <Calendar className="w-3.5 h-3.5" />
                <span>{formatDate(item.created_at)}</span>
              </div>
            </div>

            {/* Status */}
            <div className="flex-shrink-0">
              {item.status === 'success' ? (
                <div className="p-1.5 bg-emerald-100 rounded-full">
                  <CheckCircle className="w-4 h-4 text-emerald-600" />
                </div>
              ) : (
                <div className="p-1.5 bg-red-100 rounded-full">
                  <XCircle className="w-4 h-4 text-red-600" />
                </div>
              )}
            </div>

            {/* Arrow */}
            <ChevronRight className="w-4 h-4 text-gray-400 group-hover:text-gray-600 transition-colors" />
          </div>
        ))}
      </div>

      {/* No results from filter */}
      {filteredHistory.length === 0 && history.length > 0 && (
        <div className="text-center py-12">
          <p className="text-gray-600">No documents match your search</p>
          <button
            onClick={() => {
              setSearchQuery('')
              setFilterType('all')
            }}
            className="mt-2 text-sm text-blue-600 hover:text-blue-700 font-medium"
          >
            Clear filters
          </button>
        </div>
      )}
    </div>
  )
}
