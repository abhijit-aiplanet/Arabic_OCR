'use client'

import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, Image as ImageIcon, FileText } from 'lucide-react'
import Image from 'next/image'

interface ImageUploaderProps {
  onImageSelect: (file: File) => void
  imagePreview: string | null
  isProcessing: boolean
  acceptPDF?: boolean
}

export default function ImageUploader({
  onImageSelect,
  imagePreview,
  isProcessing,
  acceptPDF = true,
}: ImageUploaderProps) {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onImageSelect(acceptedFiles[0])
    }
  }, [onImageSelect])

  // Construct accept types based on acceptPDF prop
  const getAcceptTypes = () => {
    const types: Record<string, string[]> = {
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']
    }
    
    if (acceptPDF) {
      types['application/pdf'] = ['.pdf']
    }
    
    return types
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: getAcceptTypes(),
    multiple: false,
    disabled: isProcessing,
  })

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
      <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
        {acceptPDF ? (
          <>
            <FileText className="w-5 h-5 text-blue-600" />
            Upload Image or PDF
          </>
        ) : (
          <>
            <ImageIcon className="w-5 h-5 text-blue-600" />
            Upload Image
          </>
        )}
      </h2>

      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          transition-all duration-200
          ${isDragActive 
            ? 'border-blue-500 bg-blue-50' 
            : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
          }
          ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />

        {imagePreview ? (
          <div className="space-y-4">
            <div className="relative w-full h-64 rounded-lg overflow-hidden">
              <Image
                src={imagePreview}
                alt="Preview"
                fill
                className="object-contain"
              />
            </div>
            {!isProcessing && (
              <p className="text-sm text-gray-600">
                Click or drag to replace image
              </p>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            <div className="flex justify-center">
              <div className="p-4 bg-blue-100 rounded-full">
                <Upload className="w-8 h-8 text-blue-600" />
              </div>
            </div>
            <div>
              <p className="text-lg font-medium text-gray-900">
                {isDragActive 
                  ? (acceptPDF ? 'Drop file here' : 'Drop image here')
                  : (acceptPDF ? 'Drag & drop an image or PDF' : 'Drag & drop an image')
                }
              </p>
              <p className="text-sm text-gray-600 mt-1">
                or click to browse
              </p>
            </div>
            <p className="text-xs text-gray-500">
              {acceptPDF 
                ? 'Supports: Images (PNG, JPG, etc.) and PDF documents'
                : 'Supports: PNG, JPG, JPEG, GIF, WebP, BMP'
              }
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

