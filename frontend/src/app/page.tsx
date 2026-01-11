'use client'

import { useState, useEffect } from 'react'
import ImageUploader from '@/components/ImageUploader'
import PDFProcessor from '@/components/PDFProcessor'
import AdvancedSettings from '@/components/AdvancedSettings'
import OCRHistory from '@/components/OCRHistory'
import TemplateSelector from '@/components/TemplateSelector'
import UniversalRenderer from '@/components/UniversalRenderer'
import StructuredExtractor from '@/components/StructuredExtractor'
import TemplateBuilder from '@/components/TemplateBuilder'
import { 
  processOCR, 
  processPDFOCR, 
  processStructuredOCR,
  processStructuredPDFOCR,
  createTemplate,
  PDFPageResult, 
  StructuredPDFPageResult,
  updateHistoryText, 
  type ContentType, 
  type OCRTemplate, 
  type OCRConfidence,
  type StructuredExtractionData,
  type CreateTemplateRequest
} from '@/lib/api'
import { getEffectivePrompt } from '@/lib/promptGenerator'
import { parseStructuredOutput, type StructuredExtraction } from '@/lib/structuredParser'
import toast from 'react-hot-toast'
import { FileText, History, X, Trash2, ArrowRight, Zap, Shield, Clock } from 'lucide-react'
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
    maxTokens: 4096,
    minPixels: 200704,
    maxPixels: 1003520,
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
  
  // Structured extraction mode
  const [structuredMode, setStructuredMode] = useState(false)
  const [structuredData, setStructuredData] = useState<StructuredExtraction | null>(null)
  const [parsingSuccessful, setParsingSuccessful] = useState(false)
  
  // Save as template modal
  const [showSaveTemplateModal, setShowSaveTemplateModal] = useState(false)

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

  // Handle structured field edit
  const handleStructuredFieldEdit = (sectionIndex: number, fieldIndex: number, newValue: string) => {
    if (!structuredData) return
    
    const newSections = [...structuredData.sections]
    if (newSections[sectionIndex] && newSections[sectionIndex].fields[fieldIndex]) {
      newSections[sectionIndex].fields[fieldIndex] = {
        ...newSections[sectionIndex].fields[fieldIndex],
        value: newValue
      }
      setStructuredData({
        ...structuredData,
        sections: newSections
      })
      toast.success('Field updated')
    }
  }

  // Handle save as template
  const handleSaveAsTemplate = () => {
    if (!structuredData || structuredData.sections.length === 0) {
      toast.error('No structured data to save as template')
      return
    }
    setShowSaveTemplateModal(true)
  }

  // Get initial field schema from structured data
  const getInitialFieldSchema = () => {
    if (!structuredData) return undefined
    return {
      sections: structuredData.sections.map(s => ({
        name: s.name || 'General',
        fields: s.fields.map(f => ({
          label: f.label,
          type: f.type || 'text'
        }))
      }))
    }
  }

  const handleImageSelect = (file: File) => {
    setSelectedImage(file)
    
    const isPDFFile = file.type === 'application/pdf'
    setIsPDF(isPDFFile)
    
    if (!isPDFFile) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setImagePreview(reader.result as string)
      }
      reader.readAsDataURL(file)
    } else {
      setImagePreview(null)
    }
    
    setExtractedText('')
    setExtractedConfidence(null)
    setStructuredData(null)
    setParsingSuccessful(false)
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
      const token = await getToken()
      
      if (isPDF && structuredMode) {
        // Structured PDF extraction
        const loadingToast = toast.loading('Extracting form data from PDF...')
        
        await processStructuredPDFOCR(
          selectedImage,
          {
            maxTokens: settings.maxTokens,
            minPixels: settings.minPixels,
            maxPixels: settings.maxPixels,
            templateId: selectedTemplate?.id || null
          },
          (pageResult: StructuredPDFPageResult) => {
            // Convert structured result to standard PDF result for display
            const pdfResult: PDFPageResult = {
              page_number: pageResult.page_number,
              extracted_text: pageResult.raw_text || '',
              status: pageResult.status,
              error: pageResult.error,
              page_image: pageResult.page_image,
              confidence: pageResult.confidence
            }
            setPdfResults(prev => [...prev, pdfResult])
            setPdfProcessedCount(prev => prev + 1)
            
            // Store structured data for the current page
            if (pageResult.structured_data && pageResult.status === 'success') {
              setStructuredData(prev => {
                // Merge structured data from all pages
                if (!prev) {
                  return {
                    form_title: pageResult.structured_data?.form_title || null,
                    sections: pageResult.structured_data?.sections || [],
                    tables: pageResult.structured_data?.tables || [],
                    checkboxes: pageResult.structured_data?.checkboxes || [],
                    raw_text: pageResult.raw_text || ''
                  }
                }
                return {
                  ...prev,
                  sections: [
                    ...prev.sections,
                    ...(pageResult.structured_data?.sections || []).map(s => ({
                      ...s,
                      name: s.name ? `Page ${pageResult.page_number}: ${s.name}` : `Page ${pageResult.page_number}`
                    }))
                  ],
                  tables: [...prev.tables, ...(pageResult.structured_data?.tables || [])],
                  checkboxes: [...prev.checkboxes, ...(pageResult.structured_data?.checkboxes || [])],
                  raw_text: prev.raw_text + '\n\n' + (pageResult.raw_text || '')
                }
              })
              setParsingSuccessful(pageResult.parsing_successful || false)
              toast.success(`Page ${pageResult.page_number} extracted!`, { duration: 2000 })
            } else if (pageResult.status === 'error') {
              toast.error(`Page ${pageResult.page_number} failed: ${pageResult.error}`, { duration: 3000 })
            }
          },
          (totalPages: number) => {
            setPdfTotalPages(totalPages)
            toast.success(`PDF loaded: ${totalPages} pages`, { id: loadingToast, duration: 2000 })
          },
          token
        )
        
        toast.success('PDF extraction complete!', { id: loadingToast })
      } else if (isPDF) {
        // Standard PDF processing
        const effective = getEffectivePrompt({
          userCustomPrompt: settings.customPrompt,
          contentType: contentTypeOverride,
          template: selectedTemplate
        })

        const settingsForRequest = {
          ...settings,
          customPrompt: effective.prompt
        }
        
        const loadingToast = toast.loading('Processing PDF...')
        
        await processPDFOCR(
          selectedImage,
          settingsForRequest,
          (pageResult: PDFPageResult) => {
            setPdfResults(prev => [...prev, pageResult])
            setPdfProcessedCount(prev => prev + 1)
            
            if (pageResult.status === 'success') {
              toast.success(`Page ${pageResult.page_number} completed!`, { duration: 2000 })
            } else {
              toast.error(`Page ${pageResult.page_number} failed: ${pageResult.error}`, { duration: 3000 })
            }
          },
          (totalPages: number) => {
            setPdfTotalPages(totalPages)
            toast.success(`PDF loaded: ${totalPages} pages`, { id: loadingToast, duration: 2000 })
          },
          token
        )
        
        toast.success('PDF processing complete!', { id: loadingToast })
      } else if (structuredMode) {
        // Structured extraction mode
        const loadingToast = toast.loading('Extracting form data...')
        
        const result = await processStructuredOCR(
          selectedImage,
          {
            maxTokens: settings.maxTokens,
            minPixels: settings.minPixels,
            maxPixels: settings.maxPixels,
            templateId: selectedTemplate?.id || null
          },
          token
        )
        
        if (result.status === 'success') {
          setExtractedText(result.raw_text)
          setExtractedConfidence(result.confidence || null)
          setParsingSuccessful(result.parsing_successful)
          
          // Convert API response to local structured data format
          if (result.structured_data) {
            setStructuredData({
              form_title: result.structured_data.form_title,
              sections: result.structured_data.sections.map(s => ({
                name: s.name,
                fields: s.fields.map(f => ({
                  label: f.label,
                  value: f.value,
                  type: f.type as any
                }))
              })),
              tables: result.structured_data.tables || [],
              checkboxes: result.structured_data.checkboxes || [],
              raw_text: result.raw_text
            })
          } else {
            // Try client-side parsing as fallback
            const parsed = parseStructuredOutput(result.raw_text)
            if (parsed.success && parsed.data) {
              setStructuredData(parsed.data)
              setParsingSuccessful(true)
            } else {
              setStructuredData(null)
            }
          }
          
          setCurrentHistoryId(null)
          toast.success('Form data extracted!', { id: loadingToast })
        } else {
          throw new Error(result.error || 'Failed to extract form data')
        }
      } else {
        // Standard OCR mode
        const effective = getEffectivePrompt({
          userCustomPrompt: settings.customPrompt,
          contentType: contentTypeOverride,
          template: selectedTemplate
        })

        const settingsForRequest = {
          ...settings,
          customPrompt: effective.prompt
        }
        
        const loadingToast = toast.loading('Processing image...')
        
        const result = await processOCR(selectedImage, settingsForRequest, token)
        
        if (result.status === 'success') {
          setExtractedText(result.extracted_text)
          setExtractedConfidence(result.confidence || null)
          setStructuredData(null)
          setCurrentHistoryId(null)
          toast.success('Text extracted successfully!', { id: loadingToast })
        } else {
          throw new Error(result.error || 'Failed to extract text')
        }
      }
    } catch (error) {
      console.error('OCR Error:', error)
      toast.error(error instanceof Error ? error.message : 'Failed to process file')
    } finally {
      setIsProcessing(false)
    }
  }

  const handleClear = () => {
    setSelectedImage(null)
    setImagePreview(null)
    setExtractedText('')
    setExtractedConfidence(null)
    setStructuredData(null)
    setParsingSuccessful(false)
    setIsPDF(false)
    setPdfResults([])
    setPdfTotalPages(0)
    setPdfProcessedCount(0)
    setSettings({
      customPrompt: '',
      maxTokens: 4096,
      minPixels: 200704,
      maxPixels: 1003520,
    })
    setSelectedTemplate(null)
    setContentTypeOverride('auto')
  }

  return (
    <main className="min-h-screen bg-[#fafafa]">
      {/* Header */}
      <header className="bg-white border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-black rounded-lg flex items-center justify-center">
              <FileText className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-semibold text-gray-900 tracking-tight">Arabic OCR</span>
          </div>
          
          {/* Navigation */}
          <div className="flex items-center gap-3">
            <SignedIn>
              <button
                onClick={() => setShowHistory(!showHistory)}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded-lg transition-colors"
              >
                <History className="w-4 h-4" />
                History
              </button>
              <div className="w-px h-6 bg-gray-200" />
              <UserButton 
                afterSignOutUrl="/"
                appearance={{
                  elements: {
                    avatarBox: "w-9 h-9"
                  }
                }}
              />
            </SignedIn>
            <SignedOut>
              <SignInButton mode="modal">
                <button className="flex items-center gap-2 px-5 py-2 bg-black text-white text-sm font-medium rounded-lg hover:bg-gray-800 transition-colors">
                  Sign In
                </button>
              </SignInButton>
            </SignedOut>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Signed Out View */}
        <SignedOut>
          <div className="flex flex-col items-center justify-center min-h-[70vh] text-center">
            {/* Hero */}
            <div className="mb-8">
              <div className="w-16 h-16 bg-black rounded-2xl flex items-center justify-center mx-auto mb-6">
                <FileText className="w-8 h-8 text-white" />
              </div>
              <h1 className="text-4xl font-bold text-gray-900 tracking-tight mb-3">
                Arabic OCR
              </h1>
              <p className="text-lg text-gray-500 max-w-md mx-auto">
                Extract Arabic text from images and PDFs with precision using Vision Language Models
              </p>
            </div>

            {/* CTA */}
            <SignInButton mode="modal">
              <button className="flex items-center gap-2 px-8 py-3.5 bg-black text-white font-medium rounded-xl hover:bg-gray-800 transition-colors mb-12">
                Get Started
                <ArrowRight className="w-4 h-4" />
              </button>
            </SignInButton>

            {/* Features */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-3xl">
              <div className="p-6 bg-white rounded-2xl border border-gray-100">
                <div className="w-10 h-10 bg-gray-100 rounded-xl flex items-center justify-center mb-4">
                  <Zap className="w-5 h-5 text-gray-700" />
                </div>
                <h3 className="font-semibold text-gray-900 mb-1">Fast Processing</h3>
                <p className="text-sm text-gray-500">Extract text in seconds with optimized AI models</p>
              </div>
              <div className="p-6 bg-white rounded-2xl border border-gray-100">
                <div className="w-10 h-10 bg-gray-100 rounded-xl flex items-center justify-center mb-4">
                  <Shield className="w-5 h-5 text-gray-700" />
                </div>
                <h3 className="font-semibold text-gray-900 mb-1">High Accuracy</h3>
                <p className="text-sm text-gray-500">Handles handwritten and printed text with precision</p>
              </div>
              <div className="p-6 bg-white rounded-2xl border border-gray-100">
                <div className="w-10 h-10 bg-gray-100 rounded-xl flex items-center justify-center mb-4">
                  <Clock className="w-5 h-5 text-gray-700" />
                </div>
                <h3 className="font-semibold text-gray-900 mb-1">History Saved</h3>
                <p className="text-sm text-gray-500">Access all your processed documents anytime</p>
              </div>
            </div>
          </div>
        </SignedOut>

        {/* Signed In View */}
        <SignedIn>
          {/* Main Content - Conditional Layout */}
          {isPDF && pdfTotalPages > 0 ? (
            // PDF Processing View
            <div className="space-y-6">
              {/* Upload and Controls - Collapsed */}
              <div className="bg-white rounded-2xl p-5 border border-gray-100">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-red-50 rounded-xl flex items-center justify-center">
                      <FileText className="w-5 h-5 text-red-600" />
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">{selectedImage?.name}</h3>
                      <p className="text-sm text-gray-500">
                        {pdfTotalPages} pages
                        {structuredMode && <span className="ml-2 text-xs bg-gray-100 px-2 py-0.5 rounded">Structured Mode</span>}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={handleClear}
                    disabled={isProcessing}
                    className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded-lg transition-colors disabled:opacity-50"
                  >
                    <Trash2 className="w-4 h-4" />
                    Clear
                  </button>
                </div>
              </div>

              {/* PDF Results - Show structured data panel when in structured mode */}
              {structuredMode && structuredData ? (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Left - PDF pages */}
                  <PDFProcessor
                    totalPages={pdfTotalPages}
                    results={pdfResults}
                    isProcessing={isProcessing}
                  />
                  {/* Right - Aggregated Structured Data */}
                  <StructuredExtractor
                    imagePreview={null}
                    structuredData={structuredData}
                    isProcessing={isProcessing}
                    parsingSuccessful={parsingSuccessful}
                    onFieldEdit={handleStructuredFieldEdit}
                    onSaveAsTemplate={handleSaveAsTemplate}
                  />
                </div>
              ) : (
                <PDFProcessor
                  totalPages={pdfTotalPages}
                  results={pdfResults}
                  isProcessing={isProcessing}
                />
              )}
            </div>
          ) : (
            // Image Processing View - Two Columns
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left Column - Input */}
              <div className="space-y-5">
                <ImageUploader
                  onImageSelect={handleImageSelect}
                  imagePreview={imagePreview}
                  isProcessing={isProcessing}
                  acceptPDF={true}
                  selectedFile={selectedImage}
                />

                {/* Template Selector */}
                <TemplateSelector
                  authToken={authToken}
                  selectedTemplate={selectedTemplate}
                  onSelectTemplate={setSelectedTemplate}
                  contentTypeOverride={contentTypeOverride}
                  onContentTypeOverrideChange={setContentTypeOverride}
                  structuredMode={structuredMode}
                  onStructuredModeChange={setStructuredMode}
                />

                {/* Advanced Settings - only show for standard mode */}
                {!structuredMode && (
                  <AdvancedSettings
                    settings={settings}
                    onSettingsChange={setSettings}
                  />
                )}

                {/* Action Buttons */}
                <div className="space-y-3">
                  <button
                    onClick={handleProcess}
                    disabled={!selectedImage || isProcessing}
                    className="w-full bg-black text-white py-3.5 px-6 rounded-xl font-medium text-base hover:bg-gray-800 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    {isProcessing ? (
                      <span className="flex items-center justify-center gap-2">
                        <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        {isPDF ? `Processing (${pdfProcessedCount}/${pdfTotalPages || '?'})` : 'Processing...'}
                      </span>
                    ) : (
                      structuredMode ? 'Extract Form Data' : `Process ${isPDF ? 'PDF' : 'Image'}`
                    )}
                  </button>

                  {selectedImage && (
                    <button
                      onClick={handleClear}
                      disabled={isProcessing}
                      className="w-full flex items-center justify-center gap-2 py-3 px-6 text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-50 rounded-xl transition-colors disabled:opacity-50"
                    >
                      <Trash2 className="w-4 h-4" />
                      Clear
                    </button>
                  )}
                </div>
              </div>

              {/* Right Column - Output */}
              <div>
                {structuredMode ? (
                  <StructuredExtractor
                    imagePreview={imagePreview}
                    structuredData={structuredData}
                    isProcessing={isProcessing}
                    parsingSuccessful={parsingSuccessful}
                    onFieldEdit={handleStructuredFieldEdit}
                    onSaveAsTemplate={structuredData ? handleSaveAsTemplate : undefined}
                  />
                ) : (
                  <UniversalRenderer
                    text={extractedText}
                    isProcessing={isProcessing}
                    onTextEdit={handleLiveTextEdit}
                    isEditable={true}
                    preferredType={contentTypeOverride}
                    confidence={extractedConfidence}
                  />
                )}
              </div>
            </div>
          )}
        </SignedIn>
      </div>

      {/* History Modal */}
      <SignedIn>
        {showHistory && (
          <div className="fixed inset-0 bg-white z-50 overflow-y-auto">
            {/* Header */}
            <div className="sticky top-0 bg-white border-b border-gray-100 z-10">
              <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 bg-black rounded-lg flex items-center justify-center">
                    <History className="w-5 h-5 text-white" />
                  </div>
                  <span className="text-xl font-semibold text-gray-900 tracking-tight">History</span>
                </div>
                <button
                  onClick={() => setShowHistory(false)}
                  className="w-9 h-9 flex items-center justify-center hover:bg-gray-50 rounded-lg transition-colors"
                >
                  <X className="w-5 h-5 text-gray-500" />
                </button>
              </div>
            </div>

            {/* Content */}
            <div className="max-w-7xl mx-auto px-6 py-8">
              <OCRHistory getToken={getToken} />
            </div>
          </div>
        )}

        {/* Save as Template Modal */}
        {showSaveTemplateModal && (
          <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
            <div className="w-full max-w-2xl max-h-[90vh] overflow-y-auto">
              <TemplateBuilder
                onCancel={() => setShowSaveTemplateModal(false)}
                onCreate={async (payload) => {
                  if (!authToken) {
                    throw new Error('Please sign in to create templates')
                  }
                  await createTemplate(payload, authToken)
                  setShowSaveTemplateModal(false)
                  toast.success('Template saved!')
                }}
                initialFieldSchema={getInitialFieldSchema()}
              />
            </div>
          </div>
        )}
      </SignedIn>
    </main>
  )
}
