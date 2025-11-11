'use client'

import { useState } from 'react'
import ImageUploader from '@/components/ImageUploader'
import ExtractedText from '@/components/ExtractedText'
import AdvancedSettings from '@/components/AdvancedSettings'
import { processOCR } from '@/lib/api'
import toast from 'react-hot-toast'
import { FileText, Sparkles } from 'lucide-react'

interface OCRSettings {
  customPrompt: string
  maxTokens: number
  minPixels: number
  maxPixels: number
}

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [extractedText, setExtractedText] = useState<string>('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [settings, setSettings] = useState<OCRSettings>({
    customPrompt: '',
    maxTokens: 4096,  // Balanced for speed and capacity
    minPixels: 200704,
    maxPixels: 1003520,  // Reduced for faster processing
  })

  const handleImageSelect = (file: File) => {
    setSelectedImage(file)
    
    // Create preview
    const reader = new FileReader()
    reader.onloadend = () => {
      setImagePreview(reader.result as string)
    }
    reader.readAsDataURL(file)
    
    // Clear previous results
    setExtractedText('')
  }

  const handleProcess = async () => {
    if (!selectedImage) {
      toast.error('Please select an image first')
      return
    }

    setIsProcessing(true)
    const loadingToast = toast.loading('Processing image...')

    try {
      const result = await processOCR(selectedImage, settings)
      
      console.log('🎯 Result received in page:', result)
      console.log('🎯 Result status:', result.status)
      console.log('🎯 Result extracted_text:', result.extracted_text)
      
      if (result.status === 'success') {
        console.log('✅ Setting extracted text:', result.extracted_text)
        setExtractedText(result.extracted_text)
        toast.success('Text extracted successfully!', { id: loadingToast })
      } else {
        console.log('❌ Status not success:', result.status)
        throw new Error(result.error || 'Failed to extract text')
      }
    } catch (error) {
      console.error('OCR Error:', error)
      toast.error(
        error instanceof Error ? error.message : 'Failed to process image',
        { id: loadingToast }
      )
    } finally {
      setIsProcessing(false)
    }
  }

  const handleClear = () => {
    setSelectedImage(null)
    setImagePreview(null)
    setExtractedText('')
    setSettings({
      customPrompt: '',
      maxTokens: 4096,  // Balanced for speed and capacity
      minPixels: 200704,
      maxPixels: 1003520,  // Reduced for faster processing
    })
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
              <FileText className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                AIN VLM - Vision Language Model OCR
              </h1>
              <p className="text-gray-600 mt-1">
                Advanced OCR using Vision Language Model for accurate text extraction
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Info Banner */}
        <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-6 rounded-r-lg">
          <div className="flex items-start">
            <Sparkles className="w-5 h-5 text-blue-500 mt-0.5 mr-3" />
            <div>
              <p className="text-sm font-medium text-blue-900">
                How it works
              </p>
              <p className="text-sm text-blue-700 mt-1">
                Upload an image containing text, click "Process Image", and get the extracted text.
                The VLM model intelligently understands context and can handle handwritten text better than traditional OCR models.
              </p>
            </div>
          </div>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column - Input */}
          <div className="space-y-6">
            <ImageUploader
              onImageSelect={handleImageSelect}
              imagePreview={imagePreview}
              isProcessing={isProcessing}
            />

            <AdvancedSettings
              settings={settings}
              onSettingsChange={setSettings}
            />

            {/* Action Buttons */}
            <div className="space-y-3">
              <button
                onClick={handleProcess}
                disabled={!selectedImage || isProcessing}
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-6 rounded-lg font-semibold text-lg shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
              >
                {isProcessing ? (
                  <span className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Processing...
                  </span>
                ) : (
                  '🚀 Process Image'
                )}
              </button>

              <button
                onClick={handleClear}
                disabled={isProcessing}
                className="w-full bg-gray-200 text-gray-700 py-3 px-6 rounded-lg font-semibold hover:bg-gray-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                🗑️ Clear All
              </button>
            </div>
          </div>

          {/* Right Column - Output */}
          <div>
            <ExtractedText
              text={extractedText}
              isProcessing={isProcessing}
            />
          </div>
        </div>
      </div>
    </main>
  )
}

