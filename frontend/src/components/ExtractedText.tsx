'use client'

import { useState } from 'react'
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

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(isEditing ? editedText : text)
      setCopied(true)
      toast.success('Text copied to clipboard!')
      setTimeout(() => setCopied(false), 2000)
    } catch (error) {
      toast.error('Failed to copy text')
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
      toast.success('Changes saved!')
    }
  }

  const handleCancel = () => {
    setEditedText(text)
    setIsEditing(false)
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200 h-full">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
          <FileText className="w-5 h-5 text-blue-600" />
          Extracted Text
          {isEditing && <span className="text-sm font-normal text-blue-600">(Editing)</span>}
        </h2>
        
        {text && !isProcessing && (
          <div className="flex items-center gap-2">
            {isEditing ? (
              <>
                <button
                  onClick={handleCancel}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-600 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <X className="w-4 h-4" />
                  Cancel
                </button>
                <button
                  onClick={handleSave}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
                >
                  <Save className="w-4 h-4" />
                  Save
                </button>
              </>
            ) : (
              <>
                {isEditable && onTextEdit && (
                  <button
                    onClick={handleEdit}
                    className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-600 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                  >
                    <Edit2 className="w-4 h-4" />
                    Edit
                  </button>
                )}
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-colors"
                >
                  {copied ? (
                    <>
                      <Check className="w-4 h-4" />
                      Copied
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4" />
                      Copy
                    </>
                  )}
                </button>
              </>
            )}
          </div>
        )}
      </div>

      <div className="relative min-h-[400px] max-h-[600px] overflow-auto">
        {isProcessing ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
              <p className="text-gray-600 mt-4">Processing image...</p>
              <p className="text-sm text-gray-500 mt-2">This may take a few moments</p>
            </div>
          </div>
        ) : text ? (
          <div className="space-y-2">
            {isEditing ? (
              <textarea
                value={editedText}
                onChange={(e) => setEditedText(e.target.value)}
                className="w-full min-h-[400px] bg-white rounded-lg p-6 border-2 border-blue-300 font-sans text-base leading-relaxed text-gray-900 text-left focus:outline-none focus:border-blue-500 resize-y"
                placeholder="Edit your text here..."
              />
            ) : (
              <div className="bg-gray-50 rounded-lg p-6 border border-gray-200">
                <pre className="whitespace-pre-wrap font-sans text-base leading-relaxed text-gray-900 text-left">
                  {text}
                </pre>
              </div>
            )}
            
            <div className="flex items-center justify-between text-xs text-gray-500 px-2">
              <span>{(isEditing ? editedText : text).length} characters</span>
              <span>{(isEditing ? editedText : text).split(/\s+/).filter(Boolean).length} words</span>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center h-64">
            <div className="text-center text-gray-500">
              <FileText className="w-16 h-16 mx-auto mb-4 text-gray-300" />
              <p className="text-lg font-medium">No text extracted yet</p>
              <p className="text-sm mt-2">Upload an image and click "Process Image" to begin</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

