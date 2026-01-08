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
  selectedFile?: File | null
}

export default function ImageUploader({
  onImageSelect,
  imagePreview,
  isProcessing,
  acceptPDF = true,
  selectedFile = null,
}: ImageUploaderProps) {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onImageSelect(acceptedFiles[0])
    }
  }, [onImageSelect])

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
    <div className="bg-white rounded-2xl border border-gray-100 overflow-hidden">
      <div className="px-5 py-4 border-b border-gray-100">
        <h2 className="text-base font-semibold text-gray-900">Upload</h2>
      </div>

      <div className="p-5">
        <div
          {...getRootProps()}
          className={`
            relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer
            transition-all duration-200
            ${isDragActive 
              ? 'border-gray-400 bg-gray-50' 
              : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
            }
            ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}
          `}
        >
          <input {...getInputProps()} />

          {imagePreview ? (
            <div className="space-y-4">
              <div className="relative w-full h-56 rounded-lg overflow-hidden bg-gray-100">
                <Image
                  src={imagePreview}
                  alt="Preview"
                  fill
                  className="object-contain"
                />
              </div>
              {!isProcessing && (
                <p className="text-sm text-gray-500">
                  Drop a new file to replace
                </p>
              )}
            </div>
          ) : selectedFile?.type === 'application/pdf' ? (
            <div className="space-y-4">
              <div className="flex flex-col items-center justify-center py-6">
                <div className="w-14 h-14 bg-red-50 rounded-2xl flex items-center justify-center mb-4">
                  <FileText className="w-7 h-7 text-red-500" />
                </div>
                <p className="font-medium text-gray-900">{selectedFile.name}</p>
                <p className="text-sm text-gray-500 mt-1">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              {!isProcessing && (
                <p className="text-sm text-gray-500">
                  Drop a new file to replace
                </p>
              )}
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex justify-center">
                <div className="w-14 h-14 bg-gray-100 rounded-2xl flex items-center justify-center">
                  <Upload className="w-6 h-6 text-gray-400" />
                </div>
              </div>
              <div>
                <p className="font-medium text-gray-900">
                  {isDragActive 
                    ? 'Drop file here' 
                    : 'Drop file or click to browse'
                  }
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  {acceptPDF 
                    ? 'PNG, JPG, PDF up to 20MB'
                    : 'PNG, JPG, WebP, BMP'
                  }
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
