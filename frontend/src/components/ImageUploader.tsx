'use client'

import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, Image as ImageIcon, FileText, X, File } from 'lucide-react'
import Image from 'next/image'

export interface FileWithPreview {
  file: File
  id: string
  preview?: string
  pageCount?: number
  status: 'pending' | 'processing' | 'complete' | 'error'
  error?: string
}

interface ImageUploaderProps {
  onFilesSelect: (files: FileWithPreview[]) => void
  selectedFiles: FileWithPreview[]
  onRemoveFile: (id: string) => void
  onClearAll: () => void
  isProcessing: boolean
  maxFiles?: number
  maxPdfPages?: number
}

export default function ImageUploader({
  onFilesSelect,
  selectedFiles,
  onRemoveFile,
  onClearAll,
  isProcessing,
  maxFiles = 20,
  maxPdfPages = 30,
}: ImageUploaderProps) {
  
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: FileWithPreview[] = acceptedFiles.map(file => {
      const id = `${file.name}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
      
      // Create preview for images
      let preview: string | undefined
      if (file.type.startsWith('image/')) {
        preview = URL.createObjectURL(file)
      }
      
      return {
        file,
        id,
        preview,
        status: 'pending' as const,
      }
    })
    
    // Limit total files
    const totalFiles = selectedFiles.length + newFiles.length
    if (totalFiles > maxFiles) {
      const allowedNew = maxFiles - selectedFiles.length
      if (allowedNew <= 0) {
        return // Already at max
      }
      onFilesSelect([...selectedFiles, ...newFiles.slice(0, allowedNew)])
    } else {
      onFilesSelect([...selectedFiles, ...newFiles])
    }
  }, [selectedFiles, onFilesSelect, maxFiles])

  const getAcceptTypes = () => {
    return {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'],
      'application/pdf': ['.pdf']
    }
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: getAcceptTypes(),
    multiple: true,
    disabled: isProcessing,
    maxFiles: maxFiles - selectedFiles.length,
  })

  const totalPages = selectedFiles.reduce((sum, f) => {
    if (f.file.type === 'application/pdf') {
      return sum + (f.pageCount || 1)
    }
    return sum + 1
  }, 0)

  const pdfCount = selectedFiles.filter(f => f.file.type === 'application/pdf').length
  const imageCount = selectedFiles.length - pdfCount

  return (
    <div className="bg-white rounded-2xl border border-gray-100 overflow-hidden">
      <div className="px-5 py-4 border-b border-gray-100">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-gray-900">Upload Files</h2>
          {selectedFiles.length > 0 && (
            <div className="flex items-center gap-3">
              <span className="text-sm text-gray-500">
                {selectedFiles.length} file{selectedFiles.length !== 1 ? 's' : ''}
                {totalPages > selectedFiles.length && ` (${totalPages} pages)`}
              </span>
              {!isProcessing && (
                <button
                  onClick={onClearAll}
                  className="text-sm text-gray-500 hover:text-red-600 transition-colors"
                >
                  Clear all
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      <div className="p-5 space-y-4">
        {/* Drop Zone */}
        <div
          {...getRootProps()}
          className={`
            relative border-2 border-dashed rounded-xl p-6 text-center cursor-pointer
            transition-all duration-200
            ${isDragActive 
              ? 'border-purple-400 bg-purple-50' 
              : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
            }
            ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}
          `}
        >
          <input {...getInputProps()} />
          
          <div className="space-y-3">
            <div className="flex justify-center">
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                isDragActive ? 'bg-purple-100' : 'bg-gray-100'
              }`}>
                <Upload className={`w-5 h-5 ${isDragActive ? 'text-purple-500' : 'text-gray-400'}`} />
              </div>
            </div>
            <div>
              <p className="font-medium text-gray-900">
                {isDragActive 
                  ? 'Drop files here' 
                  : 'Drop files or click to browse'
                }
              </p>
              <p className="text-sm text-gray-500 mt-1">
                Images (PNG, JPG) or PDFs up to {maxFiles} files, {maxPdfPages} pages per PDF
              </p>
            </div>
          </div>
        </div>

        {/* File List */}
        {selectedFiles.length > 0 && (
          <div className="space-y-2">
            {selectedFiles.map((fileItem) => (
              <div
                key={fileItem.id}
                className={`flex items-center gap-3 p-3 rounded-xl border transition-all ${
                  fileItem.status === 'processing' 
                    ? 'border-purple-200 bg-purple-50' 
                    : fileItem.status === 'complete'
                    ? 'border-emerald-200 bg-emerald-50'
                    : fileItem.status === 'error'
                    ? 'border-red-200 bg-red-50'
                    : 'border-gray-100 bg-gray-50'
                }`}
              >
                {/* Thumbnail */}
                <div className="w-12 h-12 rounded-lg overflow-hidden bg-white border border-gray-200 flex-shrink-0">
                  {fileItem.preview ? (
                    <img
                      src={fileItem.preview}
                      alt={fileItem.file.name}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center bg-red-50">
                      <FileText className="w-5 h-5 text-red-500" />
                    </div>
                  )}
                </div>

                {/* File Info */}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 truncate">
                    {fileItem.file.name}
                  </p>
                  <p className="text-xs text-gray-500">
                    {(fileItem.file.size / 1024 / 1024).toFixed(2)} MB
                    {fileItem.file.type === 'application/pdf' && fileItem.pageCount && (
                      <span className="ml-2">{fileItem.pageCount} pages</span>
                    )}
                  </p>
                </div>

                {/* Status */}
                <div className="flex items-center gap-2">
                  {fileItem.status === 'processing' && (
                    <div className="w-5 h-5 border-2 border-purple-200 border-t-purple-600 rounded-full animate-spin" />
                  )}
                  {fileItem.status === 'complete' && (
                    <div className="w-5 h-5 rounded-full bg-emerald-500 flex items-center justify-center">
                      <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                  )}
                  {fileItem.status === 'error' && (
                    <div className="w-5 h-5 rounded-full bg-red-500 flex items-center justify-center">
                      <X className="w-3 h-3 text-white" />
                    </div>
                  )}
                  
                  {/* Remove Button */}
                  {!isProcessing && (
                    <button
                      onClick={() => onRemoveFile(fileItem.id)}
                      className="p-1.5 text-gray-400 hover:text-red-500 hover:bg-white rounded-lg transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Summary */}
        {selectedFiles.length > 0 && (
          <div className="flex items-center justify-between text-sm text-gray-500 pt-2 border-t border-gray-100">
            <div className="flex items-center gap-4">
              {imageCount > 0 && (
                <span className="flex items-center gap-1.5">
                  <ImageIcon className="w-4 h-4" />
                  {imageCount} image{imageCount !== 1 ? 's' : ''}
                </span>
              )}
              {pdfCount > 0 && (
                <span className="flex items-center gap-1.5">
                  <FileText className="w-4 h-4" />
                  {pdfCount} PDF{pdfCount !== 1 ? 's' : ''}
                </span>
              )}
            </div>
            <span>
              {totalPages} total page{totalPages !== 1 ? 's' : ''} to process
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
