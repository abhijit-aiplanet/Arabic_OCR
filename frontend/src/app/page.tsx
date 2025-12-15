'use client'

import { useState, useEffect } from 'react'
import ImageUploader from '@/components/ImageUploader'
import ExtractedText from '@/components/ExtractedText'
import PDFProcessor from '@/components/PDFProcessor'
import AdvancedSettings from '@/components/AdvancedSettings'
import OCRHistory from '@/components/OCRHistory'
import TemplateSelector from '@/components/TemplateSelector'
import UniversalRenderer from '@/components/UniversalRenderer'
import { processOCR, processPDFOCR, PDFPageResult, updateHistoryText, type ContentType, type OCRTemplate, type OCRConfidence } from '@/lib/api'
import { getEffectivePrompt } from '@/lib/promptGenerator'
import toast from 'react-hot-toast'
import { FileText, Sparkles, Lock, History, X } from 'lucide-react'
import { SignedIn, SignedOut, SignInButton, UserButton, useAuth } from '@clerk/nextjs'

interface OCRSettings {
  customPrompt: string
  maxTokens: number
  minPixels: number
  maxPixels: number
}

export default function Home() {
  const { getToken } = useAuth()
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [extractedText, setExtractedText] = useState<string>('')
  const [extractedConfidence, setExtractedConfidence] = useState<OCRConfidence | null>(null)
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

  // History state
  const [showHistory, setShowHistory] = useState(false)
  const [authToken, setAuthToken] = useState<string | null>(null)
  const [currentHistoryId, setCurrentHistoryId] = useState<string | null>(null)

  // Template + content type routing
  const [contentTypeOverride, setContentTypeOverride] = useState<ContentType>('auto')
  const [selectedTemplate, setSelectedTemplate] = useState<OCRTemplate | null>(null)

  // Get auth token on mount
  useEffect(() => {
    const fetchToken = async () => {
      const token = await getToken()
      setAuthToken(token)
    }
    fetchToken()
  }, [getToken])

  // Handle live text edit
  const handleLiveTextEdit = async (newText: string) => {
    if (!authToken || !currentHistoryId) {
      toast.error('Unable to save changes')
      return
    }

    try {
      await updateHistoryText(currentHistoryId, newText, authToken)
      setExtractedText(newText)
      toast.success('Changes saved to history!')
    } catch (error: any) {
      toast.error(error.message || 'Failed to save changes')
    }
  }

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
    setExtractedConfidence(null)
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
      // Get auth token from Clerk
      const token = await getToken()

      const effective = getEffectivePrompt({
        userCustomPrompt: settings.customPrompt,
        contentType: contentTypeOverride,
        template: selectedTemplate
      })

      const settingsForRequest = {
        ...settings,
        // If user typed a custom prompt, it remains. Otherwise use template/content-type prompt.
        customPrompt: effective.prompt
      }
      
      if (isPDF) {
        // Process PDF
        const loadingToast = toast.loading('Processing PDF...')
        
        await processPDFOCR(
          selectedImage,
          settingsForRequest,
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
          },
          token
        )
        
        toast.success('PDF processing complete!', {
          id: loadingToast
        })
      } else {
        // Process single image
        const loadingToast = toast.loading('Processing image...')
        
        const result = await processOCR(selectedImage, settingsForRequest, token)
        
        if (result.status === 'success') {
          setExtractedText(result.extracted_text)
          setExtractedConfidence(result.confidence || null)
          // Note: We'll get history ID from backend response once we add it
          // For now, clear it so user can edit after processing
          setCurrentHistoryId(null)
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
    setExtractedConfidence(null)
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
    setSelectedTemplate(null)
    setContentTypeOverride('auto')
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
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
            
            {/* Authentication UI */}
            <div className="flex items-center gap-4">
              <SignedIn>
                <button
                  onClick={() => setShowHistory(!showHistory)}
                  className="flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg font-semibold hover:bg-gray-200 transition-colors"
                >
                  <History className="w-4 h-4" />
                  History
                </button>
                <UserButton 
                  afterSignOutUrl="/"
                  appearance={{
                    elements: {
                      avatarBox: "w-10 h-10"
                    }
                  }}
                />
              </SignedIn>
              <SignedOut>
                <SignInButton mode="modal">
                  <button className="flex items-center gap-2 px-6 py-2.5 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold hover:shadow-lg transform hover:-translate-y-0.5 transition-all duration-200">
                    <Lock className="w-4 h-4" />
                    Sign In
                  </button>
                </SignInButton>
              </SignedOut>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Container */}
      <div className="flex">
        {/* Main Content */}
        <div className="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Signed Out View */}
          <SignedOut>
          <div className="flex items-center justify-center min-h-[60vh]">
            <div className="text-center max-w-md">
              <div className="p-4 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full w-20 h-20 mx-auto mb-6 flex items-center justify-center">
                <Lock className="w-10 h-10 text-white" />
              </div>
              <h2 className="text-3xl font-bold text-gray-900 mb-4">
                Sign In Required
              </h2>
              <p className="text-gray-600 mb-8">
                Please sign in to access the Arabic OCR service. Extract text from images and PDFs with high accuracy using our advanced Vision Language Model.
              </p>
              <SignInButton mode="modal">
                <button className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg font-semibold text-lg hover:shadow-xl transform hover:-translate-y-0.5 transition-all duration-200">
                  Sign In to Continue
                </button>
              </SignInButton>
              <div className="mt-8 p-4 bg-blue-50 rounded-lg">
                <p className="text-sm text-gray-700">
                  <strong>Features:</strong> Upload images or PDFs • Extract Arabic text • Handwritten & typed text support • Page-by-page PDF processing
                </p>
              </div>
            </div>
          </div>
        </SignedOut>

        {/* Signed In View */}
        <SignedIn>
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

              {/* Template Selector - TEMPORARILY HIDDEN */}
              {false && (
                <TemplateSelector
                  authToken={authToken}
                  selectedTemplate={selectedTemplate}
                  onSelectTemplate={setSelectedTemplate}
                  contentTypeOverride={contentTypeOverride}
                  onContentTypeOverrideChange={setContentTypeOverride}
                />
              )}

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
              <UniversalRenderer
                text={extractedText}
                isProcessing={isProcessing}
                onTextEdit={handleLiveTextEdit}
                isEditable={true}
                preferredType={contentTypeOverride}
                confidence={extractedConfidence}
              />
            </div>
          </div>
        )}
        </SignedIn>
        </div>

        {/* History Full-Screen Modal */}
        <SignedIn>
          {showHistory && (
            <div className="fixed inset-0 bg-white z-50 overflow-y-auto">
              {/* Header */}
              <div className="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 z-10 shadow-sm">
                <div className="max-w-7xl mx-auto flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
                      <History className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <h2 className="text-2xl font-bold text-gray-900">OCR History</h2>
                      <p className="text-sm text-gray-600">View and manage all your processed files</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setShowHistory(false)}
                    className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
                  >
                    <X className="w-6 h-6 text-gray-500" />
                  </button>
                </div>
              </div>

              {/* Content */}
              <div className="max-w-7xl mx-auto px-6 py-8">
                <OCRHistory getToken={getToken} />
              </div>
            </div>
          )}
        </SignedIn>
      </div>
    </main>
  )
}

