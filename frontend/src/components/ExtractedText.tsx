'use client'

import { useState, useEffect } from 'react'
import { Copy, Check, FileText, Edit2, Save, X } from 'lucide-react'
import toast from 'react-hot-toast'

interface ExtractedTextProps {
  text: string
  isProcessing: boolean
  onTextEdit?: (newText: string) => void
  isEditable?: boolean
}

export default function ExtractedText({ 
  text, 
  isProcessing, 
  onTextEdit,
  isEditable = true 
}: ExtractedTextProps) {
  const [copied, setCopied] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editedText, setEditedText] = useState(text)

  useEffect(() => {
    if (!isEditing) {
      setEditedText(text)
    }
  }, [text, isEditing])

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(isEditing ? editedText : text)
      setCopied(true)
      toast.success('Copied!')
      setTimeout(() => setCopied(false), 2000)
    } catch (error) {
      toast.error('Failed to copy')
    }
  }

  const handleEdit = () => {
    setEditedText(text)
    setIsEditing(true)
  }

  const handleSave = () => {
    if (onTextEdit) {
      onTextEdit(editedText)
      setIsEditing(false)
      toast.success('Saved!')
    }
  }

  const handleCancel = () => {
    setEditedText(text)
    setIsEditing(false)
  }

  return (
    <div className="h-full flex flex-col">
      {/* Actions */}
      {text && !isProcessing && (
        <div className="flex items-center justify-end gap-1 mb-3">
          {isEditing ? (
            <>
              <button
                onClick={handleCancel}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <X className="w-3.5 h-3.5" />
                Cancel
              </button>
              <button
                onClick={handleSave}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-gray-900 hover:bg-gray-800 rounded-lg transition-colors"
              >
                <Save className="w-3.5 h-3.5" />
                Save
              </button>
            </>
          ) : (
            <>
              {isEditable && onTextEdit && (
                <button
                  onClick={handleEdit}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <Edit2 className="w-3.5 h-3.5" />
                  Edit
                </button>
              )}
              <button
                onClick={handleCopy}
                className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                  copied 
                    ? 'text-emerald-700 bg-emerald-50' 
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                {copied ? 'Copied' : 'Copy'}
              </button>
            </>
          )}
        </div>
      )}

      {/* Content */}
      <div className="flex-1 min-h-0">
        {isProcessing ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="relative w-12 h-12 mx-auto mb-4">
                <div className="w-12 h-12 border-2 border-gray-200 rounded-full"></div>
                <div className="absolute inset-0 w-12 h-12 border-2 border-transparent border-t-gray-900 rounded-full animate-spin"></div>
              </div>
              <p className="text-sm font-medium text-gray-900">Processing...</p>
              <p className="text-xs text-gray-500 mt-1">This may take a moment</p>
            </div>
          </div>
        ) : text ? (
          <div className="space-y-2">
            {isEditing ? (
              <textarea
                value={editedText}
                onChange={(e) => setEditedText(e.target.value)}
                className="w-full min-h-[300px] p-4 text-sm text-gray-900 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-gray-900 focus:border-transparent resize-y font-sans leading-relaxed"
                placeholder="Edit your text here..."
                dir="auto"
              />
            ) : (
              <div className="bg-gray-50 rounded-xl p-5" dir="auto">
                <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-gray-900">
                  {text}
                </pre>
              </div>
            )}
            
            {/* Stats */}
            <div className="flex items-center gap-3 text-xs text-gray-400 px-1">
              <span>{(isEditing ? editedText : text).length.toLocaleString()} characters</span>
              <span className="w-1 h-1 rounded-full bg-gray-300"></span>
              <span>{(isEditing ? editedText : text).split(/\s+/).filter(Boolean).length.toLocaleString()} words</span>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="w-14 h-14 bg-gray-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
                <FileText className="w-6 h-6 text-gray-400" />
              </div>
              <p className="text-sm font-medium text-gray-900">No output yet</p>
              <p className="text-xs text-gray-500 mt-1 max-w-[200px] mx-auto">
                Upload a file and click Process to extract text
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
