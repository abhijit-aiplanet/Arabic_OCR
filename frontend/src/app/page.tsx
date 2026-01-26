'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import ImageUploader, { FileWithPreview } from '@/components/ImageUploader'
import TemplateSelector from '@/components/TemplateSelector'
import PageResultView from '@/components/PageResultView'
import { 
  processAgenticOCR,
  processAgenticOCRBase64,
  splitPdfToImages,
  getQueueStatus,
  type OCRTemplate,
  type AgenticOCRResponse,
  type QueueStatus,
  type PdfPageImage,
} from '@/lib/api'
import toast from 'react-hot-toast'
import { FileText, History, X, Download, ChevronLeft, ChevronRight, Zap } from 'lucide-react'
import { SignedIn, SignedOut, SignInButton, UserButton, useAuth } from '@clerk/nextjs'

// Result for a single page (image or PDF page)
interface PageResult {
  id: string
  fileId: string
  fileName: string
  pageNumber: number
  totalPages: number
  imageUrl: string  // data URL for display
  imageBase64?: string  // raw base64 for API
  result: AgenticOCRResponse | null
  status: 'pending' | 'processing' | 'complete' | 'error'
  error?: string
}

// Helper to get PDF page count client-side
async function getPdfPageCount(file: File): Promise<number> {
  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.onload = async (e) => {
      try {
        const arrayBuffer = e.target?.result as ArrayBuffer
        const bytes = new Uint8Array(arrayBuffer)
        
        // Simple PDF page count by counting /Type /Page occurrences
        // More reliable: look for /Count in the document catalog
        let text = ''
        for (let i = 0; i < Math.min(bytes.length, 50000); i++) {
          text += String.fromCharCode(bytes[i])
        }
        
        // Look for /Count N pattern in PDF
        const countMatch = text.match(/\/Count\s+(\d+)/g)
        if (countMatch && countMatch.length > 0) {
          // Get the largest count (usually the root Pages object)
          const counts = countMatch.map(m => parseInt(m.replace('/Count', '').trim()))
          const maxCount = Math.max(...counts)
          if (maxCount > 0 && maxCount < 1000) {
            resolve(maxCount)
            return
          }
        }
        
        // Fallback: count page objects
        const pageMatches = text.match(/\/Type\s*\/Page[^s]/g)
        if (pageMatches) {
          resolve(pageMatches.length)
          return
        }
        
        resolve(1) // Default to 1 if we can't determine
      } catch {
        resolve(1)
      }
    }
    reader.onerror = () => resolve(1)
    reader.readAsArrayBuffer(file)
  })
}

export default function Home() {
  const { getToken } = useAuth()
  
  // File management
  const [files, setFiles] = useState<FileWithPreview[]>([])
  const [authToken, setAuthToken] = useState<string | null>(null)
  
  // Processing state
  const [isProcessing, setIsProcessing] = useState(false)
  const [currentProcessingId, setCurrentProcessingId] = useState<string | null>(null)
  const [processingStartTime, setProcessingStartTime] = useState<number | null>(null)
  const [elapsedTime, setElapsedTime] = useState<number>(0)
  const [queueStatus, setQueueStatus] = useState<QueueStatus | null>(null)
  
  // Results
  const [pageResults, setPageResults] = useState<PageResult[]>([])
  const [selectedResultIndex, setSelectedResultIndex] = useState<number>(0)
  
  // Template
  const [selectedTemplate, setSelectedTemplate] = useState<OCRTemplate | null>(null)
  
  // History
  const [showHistory, setShowHistory] = useState(false)

  // Get fresh token - call this before each API request
  const getFreshToken = useCallback(async (): Promise<string | null> => {
    try {
      const token = await getToken({ skipCache: true })
      setAuthToken(token)
      return token
    } catch (e) {
      console.error('Failed to get token:', e)
      return null
    }
  }, [getToken])

  // Get auth token on mount
  useEffect(() => {
    getFreshToken()
  }, [getFreshToken])
  
  // Track elapsed time during processing
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null
    
    if (isProcessing && processingStartTime) {
      interval = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - processingStartTime) / 1000))
      }, 1000)
    } else {
      setElapsedTime(0)
    }
    
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isProcessing, processingStartTime])

  // Handle file selection - also get PDF page counts
  const handleFilesSelect = useCallback(async (newFiles: FileWithPreview[]) => {
    // Get page counts for PDFs
    const filesWithPageCounts = await Promise.all(
      newFiles.map(async (f) => {
        if (f.file.type === 'application/pdf' && !f.pageCount) {
          const pageCount = await getPdfPageCount(f.file)
          return { ...f, pageCount }
        }
        return f
      })
    )
    setFiles(filesWithPageCounts)
  }, [])

  const handleRemoveFile = useCallback((id: string) => {
    setFiles(prev => prev.filter(f => f.id !== id))
    setPageResults(prev => prev.filter(r => r.fileId !== id))
  }, [])

  const handleClearAll = useCallback(() => {
    setFiles([])
    setPageResults([])
    setSelectedResultIndex(0)
  }, [])

  // Process all files
  const handleProcess = async () => {
    if (files.length === 0) {
      toast.error('Please select files first')
      return
    }

    // Get fresh token at start
    let token = await getFreshToken()
    if (!token) {
      toast.error('Please sign in to use OCR features')
      return
    }

    setIsProcessing(true)
    setProcessingStartTime(Date.now())
    setPageResults([])
    setSelectedResultIndex(0)

    try {
      // Fetch queue status
      const status = await getQueueStatus('image', token)
      setQueueStatus(status)

      // First, expand all files into pages
      const allPages: PageResult[] = []
      
      for (const fileItem of files) {
        if (fileItem.file.type === 'application/pdf') {
          // Refresh token before PDF split (can take time)
          token = await getFreshToken()
          if (!token) {
            toast.error('Session expired. Please sign in again.')
            return
          }
          
          // Split PDF into pages
          toast.loading(`Loading PDF: ${fileItem.file.name}...`, { id: fileItem.id })
          
          try {
            const splitResult = await splitPdfToImages(fileItem.file, 30, token)
            
            // Update file with page count
            setFiles(prev => prev.map(f => 
              f.id === fileItem.id 
                ? { ...f, pageCount: splitResult.total_pages }
                : f
            ))
            
            // Add each page as a result entry
            for (const page of splitResult.pages) {
              allPages.push({
                id: `${fileItem.id}-page-${page.page_number}`,
                fileId: fileItem.id,
                fileName: fileItem.file.name,
                pageNumber: page.page_number,
                totalPages: splitResult.total_pages,
                imageUrl: `data:image/png;base64,${page.image_base64}`,
                imageBase64: page.image_base64,
                result: null,
                status: 'pending',
              })
            }
            
            toast.success(`PDF loaded: ${splitResult.total_pages} pages`, { id: fileItem.id })
          } catch (err: any) {
            toast.error(`Failed to load PDF: ${err.message}`, { id: fileItem.id })
            setFiles(prev => prev.map(f => 
              f.id === fileItem.id 
                ? { ...f, status: 'error', error: err.message }
                : f
            ))
          }
        } else {
          // Regular image
          allPages.push({
            id: `${fileItem.id}-page-1`,
            fileId: fileItem.id,
            fileName: fileItem.file.name,
            pageNumber: 1,
            totalPages: 1,
            imageUrl: fileItem.preview || URL.createObjectURL(fileItem.file),
            result: null,
            status: 'pending',
          })
        }
      }

      // Set initial results
      setPageResults(allPages)

      // Parallel processing with configurable concurrency
      const PARALLEL_WORKERS = 3  // Process 3 pages simultaneously
      
      // Process a single page
      const processPage = async (page: PageResult, pageIndex: number) => {
        // Get fresh token for this worker
        const workerToken = await getFreshToken()
        if (!workerToken) {
          throw new Error('Session expired')
        }
        
        // Update status to processing
        setPageResults(prev => prev.map(p => 
          p.id === page.id ? { ...p, status: 'processing' } : p
        ))

        try {
          let result: AgenticOCRResponse

          if (page.imageBase64) {
            // PDF page - use base64
            result = await processAgenticOCRBase64(
              page.imageBase64,
              `${page.fileName}-page-${page.pageNumber}.png`,
              { maxIterations: 2 },
              workerToken
            )
          } else {
            // Regular image file
            const file = files.find(f => f.id === page.fileId)?.file
            if (!file) throw new Error('File not found')
            
            result = await processAgenticOCR(file, { maxIterations: 2 }, workerToken)
          }

          // Update with result
          setPageResults(prev => prev.map(p => 
            p.id === page.id 
              ? { ...p, status: 'complete', result }
              : p
          ))
          
          toast.success(
            `Page ${pageIndex + 1}: ${result.fields.length} items extracted`,
            { duration: 2000 }
          )
          
          return { success: true, page }

        } catch (err: any) {
          // Retry once with fresh token
          try {
            const retryToken = await getFreshToken()
            if (!retryToken) throw err
            
            let result: AgenticOCRResponse
            if (page.imageBase64) {
              result = await processAgenticOCRBase64(
                page.imageBase64,
                `${page.fileName}-page-${page.pageNumber}.png`,
                { maxIterations: 2 },
                retryToken
              )
            } else {
              const file = files.find(f => f.id === page.fileId)?.file
              if (!file) throw new Error('File not found')
              result = await processAgenticOCR(file, { maxIterations: 2 }, retryToken)
            }
            
            setPageResults(prev => prev.map(p => 
              p.id === page.id ? { ...p, status: 'complete', result } : p
            ))
            return { success: true, page }
          } catch (retryErr: any) {
            setPageResults(prev => prev.map(p => 
              p.id === page.id 
                ? { ...p, status: 'error', error: retryErr.message }
                : p
            ))
            toast.error(`Page ${pageIndex + 1} failed: ${retryErr.message}`)
            return { success: false, page, error: retryErr.message }
          }
        }
      }

      // Process pages in parallel batches
      const processBatch = async (batch: PageResult[], startIndex: number) => {
        setCurrentProcessingId(batch[0]?.id || null)
        setSelectedResultIndex(startIndex)
        
        const results = await Promise.allSettled(
          batch.map((page, i) => processPage(page, startIndex + i))
        )
        return results
      }

      // Split into batches and process
      for (let i = 0; i < allPages.length; i += PARALLEL_WORKERS) {
        const batch = allPages.slice(i, i + PARALLEL_WORKERS)
        await processBatch(batch, i)
      }

      // Update all file statuses to complete
      setFiles(prev => prev.map(f => ({ ...f, status: 'complete' as const })))

      // Count results
      const completed = allPages.filter(p => {
        const result = pageResults.find(r => r.id === p.id)
        return result?.status === 'complete'
      }).length

      toast.success(`Processing complete! ${completed}/${allPages.length} pages extracted`)

    } catch (err: any) {
      toast.error(err.message || 'Processing failed')
    } finally {
      setIsProcessing(false)
      setProcessingStartTime(null)
      setCurrentProcessingId(null)
      setQueueStatus(null)
    }
  }

  // Download all results
  const handleDownloadAll = () => {
    const results = pageResults.filter(p => p.result)
    if (results.length === 0) {
      toast.error('No results to download')
      return
    }

    const output = results.map(p => {
      const header = `# ${p.fileName}${p.totalPages > 1 ? ` - Page ${p.pageNumber}/${p.totalPages}` : ''}`
      const fields = p.result?.fields.map(f => 
        `${f.field_name}: ${f.value} [${f.confidence}]`
      ).join('\n') || ''
      return `${header}\n\n${fields}`
    }).join('\n\n---\n\n')

    const blob = new Blob([output], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'ocr-results.md'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    
    toast.success('Results downloaded!')
  }

  const selectedResult = pageResults[selectedResultIndex]
  const completedCount = pageResults.filter(p => p.status === 'complete').length
  const totalCount = pageResults.length

  return (
    <main className="min-h-screen bg-[#fafafa]">
      {/* Header */}
      <header className="bg-white border-b border-gray-100">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-black rounded-lg flex items-center justify-center">
              <FileText className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-semibold text-gray-900 tracking-tight">Arabic OCR</span>
          </div>
          
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
              <UserButton afterSignOutUrl="/" />
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
        <SignedOut>
          {/* Landing page for signed out users */}
          <div className="flex flex-col items-center justify-center min-h-[70vh] text-center">
            <div className="w-16 h-16 bg-black rounded-2xl flex items-center justify-center mx-auto mb-6">
              <FileText className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-gray-900 tracking-tight mb-3">
              Arabic OCR
            </h1>
            <p className="text-lg text-gray-500 max-w-md mx-auto mb-8">
              Extract Arabic text from images and PDFs with AI-powered multi-pass extraction
            </p>
            <SignInButton mode="modal">
              <button className="px-8 py-3.5 bg-black text-white font-medium rounded-xl hover:bg-gray-800 transition-colors">
                Get Started
              </button>
            </SignInButton>
          </div>
        </SignedOut>

        <SignedIn>
          {/* Processing Status Banner */}
          {isProcessing && (
            <div className="sticky top-0 z-50 mb-6">
              <div className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white rounded-2xl p-4 shadow-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center">
                      <Zap className="w-5 h-5 text-white animate-pulse" />
                    </div>
                    <div>
                      <div className="font-semibold">
                        Processing {completedCount}/{totalCount} pages
                      </div>
                      <div className="text-sm text-purple-100">
                        {queueStatus?.message || 'Agentic OCR in progress...'}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-mono font-bold">
                      {Math.floor(elapsedTime / 60)}:{String(elapsedTime % 60).padStart(2, '0')}
                    </div>
                  </div>
                </div>
                <div className="mt-3">
                  <div className="w-full bg-white/20 rounded-full h-2">
                    <div 
                      className="bg-white h-2 rounded-full transition-all duration-500"
                      style={{ width: `${totalCount > 0 ? (completedCount / totalCount) * 100 : 0}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column - Upload & Settings */}
            <div className="space-y-5">
              <ImageUploader
                onFilesSelect={handleFilesSelect}
                selectedFiles={files}
                onRemoveFile={handleRemoveFile}
                onClearAll={handleClearAll}
                isProcessing={isProcessing}
                maxFiles={20}
                maxPdfPages={30}
              />

              <TemplateSelector
                authToken={authToken}
                selectedTemplate={selectedTemplate}
                onSelectTemplate={setSelectedTemplate}
              />

              {/* Action Buttons */}
              <div className="space-y-3">
                <button
                  onClick={handleProcess}
                  disabled={files.length === 0 || isProcessing}
                  className="w-full bg-black text-white py-3.5 px-6 rounded-xl font-medium text-base hover:bg-gray-800 transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {isProcessing ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4" />
                      Process All Files
                    </>
                  )}
                </button>

                {pageResults.length > 0 && !isProcessing && (
                  <button
                    onClick={handleDownloadAll}
                    className="w-full flex items-center justify-center gap-2 py-3 px-6 text-sm font-medium text-gray-600 hover:text-gray-900 bg-white border border-gray-200 hover:border-gray-300 rounded-xl transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    Download All Results
                  </button>
                )}
              </div>
            </div>

            {/* Right Column - Results (2 columns wide) */}
            <div className="lg:col-span-2">
              {pageResults.length > 0 ? (
                <div className="space-y-4">
                  {/* Page Navigation */}
                  {pageResults.length > 1 && (
                    <div className="bg-white rounded-xl border border-gray-100 p-3">
                      <div className="flex items-center gap-2 overflow-x-auto pb-1">
                        {pageResults.map((page, idx) => (
                          <button
                            key={page.id}
                            onClick={() => setSelectedResultIndex(idx)}
                            className={`flex-shrink-0 px-3 py-1.5 text-sm font-medium rounded-lg transition-colors ${
                              selectedResultIndex === idx
                                ? 'bg-black text-white'
                                : page.status === 'complete'
                                ? 'bg-emerald-50 text-emerald-700 hover:bg-emerald-100'
                                : page.status === 'error'
                                ? 'bg-red-50 text-red-700 hover:bg-red-100'
                                : page.status === 'processing'
                                ? 'bg-purple-50 text-purple-700'
                                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                            }`}
                          >
                            {page.totalPages > 1 
                              ? `${page.fileName.slice(0, 15)}... P${page.pageNumber}`
                              : page.fileName.slice(0, 20)
                            }
                            {page.status === 'processing' && (
                              <span className="ml-1.5 inline-block w-2 h-2 bg-purple-500 rounded-full animate-pulse" />
                            )}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Selected Result View */}
                  {selectedResult && (
                    <PageResultView
                      imageUrl={selectedResult.imageUrl}
                      result={selectedResult.result}
                      isProcessing={selectedResult.status === 'processing'}
                      pageLabel={
                        selectedResult.totalPages > 1
                          ? `${selectedResult.fileName} - Page ${selectedResult.pageNumber}/${selectedResult.totalPages}`
                          : selectedResult.fileName
                      }
                      onPrevPage={() => setSelectedResultIndex(Math.max(0, selectedResultIndex - 1))}
                      onNextPage={() => setSelectedResultIndex(Math.min(pageResults.length - 1, selectedResultIndex + 1))}
                      hasPrev={selectedResultIndex > 0}
                      hasNext={selectedResultIndex < pageResults.length - 1}
                    />
                  )}
                </div>
              ) : (
                /* Empty State */
                <div className="bg-white rounded-2xl border border-gray-100 p-12 text-center">
                  <div className="w-16 h-16 bg-gray-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <FileText className="w-8 h-8 text-gray-400" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">No files uploaded</h3>
                  <p className="text-gray-500 max-w-sm mx-auto">
                    Upload images or PDFs to extract Arabic text using AI-powered multi-pass OCR
                  </p>
                </div>
              )}
            </div>
          </div>
        </SignedIn>
      </div>

      {/* History Modal Placeholder */}
      {showHistory && (
        <div className="fixed inset-0 bg-white z-50 overflow-y-auto">
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
          <div className="max-w-7xl mx-auto px-6 py-8">
            <p className="text-gray-500">History feature coming soon...</p>
          </div>
        </div>
      )}
    </main>
  )
}
