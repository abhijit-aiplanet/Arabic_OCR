'use client'

import { useState } from 'react'
import ImageUploader from '@/components/ImageUploader'
import ExtractedText from '@/components/ExtractedText'
import PDFProcessor from '@/components/PDFProcessor'
import AdvancedSettings from '@/components/AdvancedSettings'
import { processOCR, processPDFOCR, PDFPageResult } from '@/lib/api'
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

  // PDF-specific state
  const [isPDF, setIsPDF] = useState(false)
  const [pdfTotalPages, setPdfTotalPages] = useState(0)
  const [pdfResults, setPdfResults] = useState<PDFPageResult[]>([])
  const [pdfProcessedCount, setPdfProcessedCount] = useState(0)

  const handleImageSelect = (file: File) => {
    setSelectedImage(file)
    
    // Check if PDF
    const isPDFFile = file.type === 'application/pdf'
    setIsPDF(isPDFFile)
    
    // Create preview for images only
    if (!isPDFFile) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result as string)
      }
      reader.readAsDataURL(file)
    } else {
      setImagePreview(null)
    }
    
    // Clear previous results
    setExtractedText('')
    setPdfResults([])
    setPdfTotalPages(0)
    setPdfProcessedCount(0)
  }

  const handleProcess = async () => {
    if (!selectedImage) {
      toast.error('Please select a file first')
      return
    }

    setIsProcessing(true)

    try {
      if (isPDF) {
        // Process PDF
        const loadingToast = toast.loading('Processing PDF...')
        
        await processPDFOCR(
          selectedImage,
          settings,
          (pageResult: PDFPageResult) => {
            // Called when each page completes
            setPdfResults(prev => [...prev, pageResult])
            setPdfProcessedCount(prev => prev + 1)
            
            if (pageResult.status === 'success') {
              toast.success(`Page ${pageResult.page_number} completed!`, {
                duration: 2000
              })
            } else {
              toast.error(`Page ${pageResult.page_number} failed: ${pageResult.error}`, {
                duration: 3000
              })
            }
          },
          (totalPages: number) => {
            // Called when metadata is received
            setPdfTotalPages(totalPages)
            toast.success(`PDF loaded: ${totalPages} pages`, {
              id: loadingToast,
              duration: 2000
            })
          }
        )
        
        toast.success('PDF processing complete!', {
          id: loadingToast
        })
      } else {
        // Process single image
        const loadingToast = toast.loading('Processing image...')
        
        const result = await processOCR(selectedImage, settings)
        
        if (result.status === 'success') {
          setExtractedText(result.extracted_text)
          toast.success('Text extracted successfully!', { id: loadingToast })
        } else {
          throw new Error(result.error || 'Failed to extract text')
        }
      }
    } catch (error) {
      console.error('OCR Error:', error)
      toast.error(
        error instanceof Error ? error.message : 'Failed to process file'
      )
    } finally {
      setIsProcessing(false)
    }
  }

  const handleClear = () => {
    setSelectedImage(null)
    setImagePreview(null)
    setExtractedText('')
    setIsPDF(false)
    setPdfResults([])
    setPdfTotalPages(0)
    setPdfProcessedCount(0)
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
                Arabic Vision Language Model OCR
              </h1>
              <p className="text-gray-600 mt-1">
                Advanced OCR for images and PDFs using Vision Language Model
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
                Upload an image or PDF document, click "Process", and get the extracted text.
                For PDFs, each page is processed sequentially and results are shown as they complete.
                The VLM model intelligently understands context and can handle handwritten text better than traditional OCR models.
              </p>
            </div>
          </div>
        </div>

        {/* Main Content - Conditional Layout */}
        {isPDF && pdfTotalPages > 0 ? (
          // PDF Processing View - Full Width
          <div className="space-y-6">
            {/* Upload and Controls - Collapsed */}
            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <FileText className="w-6 h-6 text-blue-600" />
                  <div>
                    <h3 className="font-semibold text-gray-900">
                      {selectedImage?.name}
                    </h3>
                    <p className="text-sm text-gray-600">
                      {isPDF ? `PDF Document - ${pdfTotalPages} pages` : 'Image File'}
                    </p>
                  </div>
                </div>
                <button
                  onClick={handleClear}
                  disabled={isProcessing}
                  className="bg-gray-200 text-gray-700 px-4 py-2 rounded-lg font-semibold hover:bg-gray-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  🗑️ Clear
                </button>
              </div>
            </div>

            {/* PDF Results */}
            <PDFProcessor
              totalPages={pdfTotalPages}
              results={pdfResults}
              isProcessing={isProcessing}
            />
          </div>
        ) : (
          // Image Processing View - Two Columns
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left Column - Input */}
            <div className="space-y-6">
            <ImageUploader
              onImageSelect={handleImageSelect}
              imagePreview={imagePreview}
              isProcessing={isProcessing}
              acceptPDF={true}
              selectedFile={selectedImage}
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
                      {isPDF ? `Processing PDF (${pdfProcessedCount}/${pdfTotalPages || '?'})...` : 'Processing...'}
                    </span>
                  ) : (
                    `🚀 Process ${isPDF ? 'PDF' : 'Image'}`
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
        )}
      </div>
    </main>
  )
}

