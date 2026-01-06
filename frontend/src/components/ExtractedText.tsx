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

  // Sync editedText when text prop changes
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
      {/* Header with actions */}
      {text && !isProcessing && (
        <div className="flex items-center justify-between mb-4 pb-3 border-b border-gray-100">
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-gray-400" />
            <span className="text-sm font-medium text-gray-700">Extracted Text</span>
            {isEditing && (
              <span className="text-xs font-medium text-blue-600 bg-blue-50 px-2 py-0.5 rounded-full">Editing</span>
            )}
          </div>
          
          <div className="flex items-center gap-1">
            {isEditing ? (
              <>
                <button
                  onClick={handleCancel}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <X className="w-3.5 h-3.5" />
                  Cancel
                </button>
                <button
                  onClick={handleSave}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors shadow-sm"
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
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
                  >
                    <Edit2 className="w-3.5 h-3.5" />
                    Edit
                  </button>
                )}
                <button
                  onClick={handleCopy}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-lg transition-colors ${
                    copied 
                      ? 'text-emerald-600 bg-emerald-50' 
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                  {copied ? 'Copied' : 'Copy'}
                </button>
              </>
            )}
          </div>
        </div>
      )}

      {/* Content */}
      <div className="flex-1 min-h-0">
        {isProcessing ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="relative">
                <div className="w-16 h-16 border-4 border-blue-100 rounded-full"></div>
                <div className="absolute inset-0 w-16 h-16 border-4 border-transparent border-t-blue-600 rounded-full animate-spin"></div>
              </div>
              <p className="text-gray-700 mt-5 font-medium">Processing image...</p>
              <p className="text-sm text-gray-500 mt-1">This may take a few moments</p>
            </div>
          </div>
        ) : text ? (
          <div className="space-y-3">
            {isEditing ? (
              <textarea
                value={editedText}
                onChange={(e) => setEditedText(e.target.value)}
                className="w-full min-h-[300px] bg-white rounded-xl p-5 border-2 border-blue-200 font-sans text-base leading-relaxed text-gray-900 focus:outline-none focus:border-blue-400 focus:ring-4 focus:ring-blue-50 resize-y transition-all"
                placeholder="Edit your text here..."
                dir="auto"
              />
            ) : (
              <div className="bg-gradient-to-br from-slate-50 to-gray-50 rounded-xl p-5 border border-gray-200" dir="auto">
                <pre className="whitespace-pre-wrap font-sans text-base leading-relaxed text-gray-900">
                  {text}
                </pre>
              </div>
            )}
            
            {/* Stats */}
            <div className="flex items-center justify-between text-xs text-gray-400 px-1">
              <span>{(isEditing ? editedText : text).length} characters</span>
              <span>{(isEditing ? editedText : text).split(/\s+/).filter(Boolean).length} words</span>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="w-20 h-20 mx-auto mb-5 rounded-2xl bg-gradient-to-br from-gray-100 to-gray-50 flex items-center justify-center">
                <FileText className="w-10 h-10 text-gray-300" />
              </div>
              <p className="text-lg font-semibold text-gray-700">No text extracted yet</p>
              <p className="text-sm text-gray-500 mt-1.5 max-w-xs mx-auto">
                Upload an image or PDF and click "Process" to extract text
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

