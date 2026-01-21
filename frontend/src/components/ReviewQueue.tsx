'use client'

import { useState, useEffect, useCallback } from 'react'
import { 
  CheckCircle, 
  XCircle, 
  AlertTriangle, 
  Edit3, 
  SkipForward,
  Eye,
  RefreshCw,
  Filter,
  ChevronDown,
  ChevronUp
} from 'lucide-react'

// =============================================================================
// TYPES
// =============================================================================

interface ReviewQueueItem {
  id: string
  ocr_history_id?: string
  field_name: string
  extracted_value?: string
  confidence_score?: number
  confidence_level?: string
  review_reason?: string
  validation_status?: string
  is_suspicious: boolean
  suspicion_score?: number
  full_image_url?: string
  status: string
  created_at?: string
}

interface ReviewQueueStats {
  total: number
  pending_count: number
  approved_count: number
  corrected_count: number
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function getConfidenceColor(level?: string): string {
  switch (level) {
    case 'high':
      return 'bg-emerald-100 text-emerald-700'
    case 'medium':
      return 'bg-amber-100 text-amber-700'
    case 'low':
      return 'bg-orange-100 text-orange-700'
    case 'very_low':
    case 'unreadable':
      return 'bg-red-100 text-red-700'
    case 'empty':
      return 'bg-gray-100 text-gray-600'
    default:
      return 'bg-gray-100 text-gray-600'
  }
}

function formatConfidence(score?: number): string {
  if (score === undefined || score === null) return '—'
  return `${Math.round(score * 100)}%`
}

function formatDate(dateStr?: string): string {
  if (!dateStr) return '—'
  try {
    const date = new Date(dateStr)
    return date.toLocaleDateString('ar-SA', { 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  } catch {
    return dateStr
  }
}

// =============================================================================
// REVIEW QUEUE COMPONENT
// =============================================================================

export default function ReviewQueue({ 
  apiBase = '/api',
  onReviewComplete
}: { 
  apiBase?: string
  onReviewComplete?: () => void
}) {
  const [items, setItems] = useState<ReviewQueueItem[]>([])
  const [stats, setStats] = useState<ReviewQueueStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [statusFilter, setStatusFilter] = useState<string>('pending')
  const [expandedItem, setExpandedItem] = useState<string | null>(null)
  const [correctionValue, setCorrectionValue] = useState<string>('')
  const [processingId, setProcessingId] = useState<string | null>(null)

  // Fetch review queue items
  const fetchQueue = useCallback(async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch(
        `${apiBase}/review-queue?status=${statusFilter}&limit=50`,
        { credentials: 'include' }
      )
      
      if (!response.ok) {
        throw new Error('Failed to fetch review queue')
      }
      
      const data = await response.json()
      setItems(data.items || [])
      setStats({
        total: data.total,
        pending_count: data.pending_count,
        approved_count: data.approved_count,
        corrected_count: data.corrected_count
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [apiBase, statusFilter])

  useEffect(() => {
    fetchQueue()
  }, [fetchQueue])

  // Handle review actions
  const handleAction = async (
    itemId: string, 
    action: 'approve' | 'correct' | 'reject' | 'skip',
    correctedValue?: string
  ) => {
    setProcessingId(itemId)
    
    try {
      const endpoint = action === 'skip' 
        ? `${apiBase}/review-queue/batch`
        : `${apiBase}/review-queue/${itemId}/${action}`
      
      const body = action === 'skip'
        ? JSON.stringify({ actions: [{ id: itemId, action: 'skip' }] })
        : action === 'correct'
          ? JSON.stringify({ action: 'correct', corrected_value: correctedValue })
          : JSON.stringify({ action })
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body
      })
      
      if (!response.ok) {
        throw new Error(`Failed to ${action} item`)
      }
      
      // Remove item from local state
      setItems(prev => prev.filter(item => item.id !== itemId))
      
      // Update stats
      if (stats) {
        setStats({
          ...stats,
          pending_count: Math.max(0, stats.pending_count - 1),
          approved_count: action === 'approve' ? stats.approved_count + 1 : stats.approved_count,
          corrected_count: action === 'correct' ? stats.corrected_count + 1 : stats.corrected_count
        })
      }
      
      // Clear expanded state
      if (expandedItem === itemId) {
        setExpandedItem(null)
        setCorrectionValue('')
      }
      
      onReviewComplete?.()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Action failed')
    } finally {
      setProcessingId(null)
    }
  }

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!expandedItem || processingId) return
      
      switch (e.key) {
        case 'a':
        case 'A':
          if (!e.ctrlKey && !e.metaKey) {
            e.preventDefault()
            handleAction(expandedItem, 'approve')
          }
          break
        case 'r':
        case 'R':
          if (!e.ctrlKey && !e.metaKey) {
            e.preventDefault()
            handleAction(expandedItem, 'reject')
          }
          break
        case 's':
        case 'S':
          if (!e.ctrlKey && !e.metaKey) {
            e.preventDefault()
            handleAction(expandedItem, 'skip')
          }
          break
        case 'Escape':
          setExpandedItem(null)
          setCorrectionValue('')
          break
      }
    }
    
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [expandedItem, processingId])

  // ==========================================================================
  // RENDER
  // ==========================================================================

  if (loading && items.length === 0) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="w-6 h-6 animate-spin text-gray-400" />
        <span className="ml-2 text-gray-500">Loading review queue...</span>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h3 className="font-semibold text-gray-900">Review Queue</h3>
            {stats && (
              <div className="flex items-center gap-2 text-sm">
                <span className="px-2 py-0.5 bg-amber-100 text-amber-700 rounded-full">
                  {stats.pending_count} pending
                </span>
                <span className="px-2 py-0.5 bg-emerald-100 text-emerald-700 rounded-full">
                  {stats.approved_count} approved
                </span>
              </div>
            )}
          </div>
          
          <div className="flex items-center gap-2">
            {/* Filter Dropdown */}
            <div className="relative">
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="appearance-none pl-8 pr-8 py-1.5 text-sm border border-gray-200 rounded-lg bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="pending">Pending</option>
                <option value="approved">Approved</option>
                <option value="corrected">Corrected</option>
                <option value="rejected">Rejected</option>
                <option value="all">All</option>
              </select>
              <Filter className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            </div>
            
            {/* Refresh Button */}
            <button
              onClick={fetchQueue}
              disabled={loading}
              className="p-1.5 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>
        
        {/* Keyboard Shortcuts Help */}
        <div className="mt-2 flex items-center gap-4 text-xs text-gray-500">
          <span>Shortcuts:</span>
          <span><kbd className="px-1.5 py-0.5 bg-gray-200 rounded text-gray-600">A</kbd> Approve</span>
          <span><kbd className="px-1.5 py-0.5 bg-gray-200 rounded text-gray-600">R</kbd> Reject</span>
          <span><kbd className="px-1.5 py-0.5 bg-gray-200 rounded text-gray-600">S</kbd> Skip</span>
          <span><kbd className="px-1.5 py-0.5 bg-gray-200 rounded text-gray-600">Esc</kbd> Close</span>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="px-4 py-3 bg-red-50 border-b border-red-100">
          <div className="flex items-center gap-2 text-red-700">
            <AlertTriangle className="w-4 h-4" />
            <span className="text-sm">{error}</span>
            <button 
              onClick={() => setError(null)}
              className="ml-auto text-red-500 hover:text-red-700"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* Empty State */}
      {items.length === 0 && !loading && (
        <div className="px-4 py-12 text-center">
          <CheckCircle className="w-12 h-12 text-emerald-400 mx-auto mb-3" />
          <h4 className="font-medium text-gray-900">No items to review</h4>
          <p className="text-sm text-gray-500 mt-1">
            {statusFilter === 'pending' 
              ? 'All extractions have been reviewed!'
              : `No ${statusFilter} items found.`}
          </p>
        </div>
      )}

      {/* Queue Items */}
      <div className="divide-y divide-gray-100">
        {items.map((item) => (
          <ReviewQueueCard
            key={item.id}
            item={item}
            isExpanded={expandedItem === item.id}
            isProcessing={processingId === item.id}
            correctionValue={correctionValue}
            onToggleExpand={() => {
              if (expandedItem === item.id) {
                setExpandedItem(null)
                setCorrectionValue('')
              } else {
                setExpandedItem(item.id)
                setCorrectionValue(item.extracted_value || '')
              }
            }}
            onCorrectionChange={setCorrectionValue}
            onAction={(action) => handleAction(item.id, action, correctionValue)}
          />
        ))}
      </div>
    </div>
  )
}

// =============================================================================
// REVIEW QUEUE CARD COMPONENT
// =============================================================================

function ReviewQueueCard({
  item,
  isExpanded,
  isProcessing,
  correctionValue,
  onToggleExpand,
  onCorrectionChange,
  onAction
}: {
  item: ReviewQueueItem
  isExpanded: boolean
  isProcessing: boolean
  correctionValue: string
  onToggleExpand: () => void
  onCorrectionChange: (value: string) => void
  onAction: (action: 'approve' | 'correct' | 'reject' | 'skip') => void
}) {
  return (
    <div className={`transition-colors ${isExpanded ? 'bg-blue-50/50' : 'hover:bg-gray-50'}`}>
      {/* Compact Row */}
      <div 
        className="px-4 py-3 flex items-center gap-4 cursor-pointer"
        onClick={onToggleExpand}
      >
        {/* Confidence Badge */}
        <div className={`px-2 py-1 rounded text-xs font-medium ${getConfidenceColor(item.confidence_level)}`}>
          {formatConfidence(item.confidence_score)}
        </div>
        
        {/* Field Name */}
        <div className="flex-1 min-w-0">
          <div className="font-medium text-gray-900 truncate">{item.field_name}</div>
          <div className="text-sm text-gray-500 truncate" dir="rtl">
            {item.extracted_value || <span className="text-gray-400">[فارغ]</span>}
          </div>
        </div>
        
        {/* Suspicious Badge */}
        {item.is_suspicious && (
          <div className="px-2 py-1 bg-red-100 text-red-700 rounded text-xs font-medium flex items-center gap-1">
            <AlertTriangle className="w-3 h-3" />
            Suspicious
          </div>
        )}
        
        {/* Reason */}
        {item.review_reason && (
          <div className="text-xs text-gray-500 max-w-[200px] truncate">
            {item.review_reason}
          </div>
        )}
        
        {/* Expand Icon */}
        {isExpanded ? (
          <ChevronUp className="w-4 h-4 text-gray-400" />
        ) : (
          <ChevronDown className="w-4 h-4 text-gray-400" />
        )}
      </div>

      {/* Expanded Details */}
      {isExpanded && (
        <div className="px-4 pb-4 space-y-4">
          {/* Image Preview (if available) */}
          {item.full_image_url && (
            <div className="relative">
              <a 
                href={item.full_image_url} 
                target="_blank" 
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-sm text-blue-600 hover:text-blue-700"
              >
                <Eye className="w-4 h-4" />
                View Original Image
              </a>
            </div>
          )}
          
          {/* Correction Input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Correct Value (if needed)
            </label>
            <input
              type="text"
              value={correctionValue}
              onChange={(e) => onCorrectionChange(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              dir="rtl"
              placeholder="Enter corrected value..."
            />
          </div>
          
          {/* Validation Info */}
          {item.validation_status && item.validation_status !== 'unchecked' && (
            <div className={`text-sm px-3 py-2 rounded ${
              item.validation_status === 'valid' 
                ? 'bg-emerald-50 text-emerald-700'
                : item.validation_status === 'suspicious'
                  ? 'bg-red-50 text-red-700'
                  : 'bg-amber-50 text-amber-700'
            }`}>
              <span className="font-medium">Validation:</span> {item.validation_status}
            </div>
          )}
          
          {/* Action Buttons */}
          <div className="flex items-center gap-2">
            {/* Approve Button */}
            <button
              onClick={() => onAction('approve')}
              disabled={isProcessing}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <CheckCircle className="w-4 h-4" />
              Approve
            </button>
            
            {/* Correct Button */}
            <button
              onClick={() => onAction('correct')}
              disabled={isProcessing || !correctionValue || correctionValue === item.extracted_value}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <Edit3 className="w-4 h-4" />
              Correct
            </button>
            
            {/* Reject Button */}
            <button
              onClick={() => onAction('reject')}
              disabled={isProcessing}
              className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <XCircle className="w-4 h-4" />
              Reject
            </button>
            
            {/* Skip Button */}
            <button
              onClick={() => onAction('skip')}
              disabled={isProcessing}
              className="px-4 py-2 text-gray-600 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <SkipForward className="w-4 h-4" />
            </button>
          </div>
          
          {/* Processing Indicator */}
          {isProcessing && (
            <div className="flex items-center justify-center gap-2 text-gray-500">
              <RefreshCw className="w-4 h-4 animate-spin" />
              Processing...
            </div>
          )}
        </div>
      )}
    </div>
  )
}
