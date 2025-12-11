import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface OCRSettings {
  customPrompt: string
  maxTokens: number
  minPixels: number
  maxPixels: number
}

interface OCRResponse {
  extracted_text: string
  status: string
  error?: string
}

export interface PDFPageResult {
  page_number: number
  extracted_text: string
  status: string
  error?: string
  page_image?: string  // Base64 encoded
}

export interface PDFStreamMessage {
  type: 'metadata' | 'page_result' | 'complete' | 'error'
  total_pages?: number
  page_number?: number
  status?: string
  extracted_text?: string
  error?: string
  page_image?: string
}

export async function processOCR(
  imageFile: File,
  settings: OCRSettings,
  authToken?: string | null
): Promise<OCRResponse> {
  try {
    const formData = new FormData()
    formData.append('file', imageFile)
    
    if (settings.customPrompt) {
      formData.append('custom_prompt', settings.customPrompt)
    }
    
    formData.append('max_new_tokens', settings.maxTokens.toString())
    formData.append('min_pixels', settings.minPixels.toString())
    formData.append('max_pixels', settings.maxPixels.toString())

    const headers: Record<string, string> = {
      'Content-Type': 'multipart/form-data',
    }

    // Add auth token if provided
    if (authToken) {
      headers['Authorization'] = `Bearer ${authToken}`
    }

    const response = await axios.post<OCRResponse>(
      `${API_URL}/api/ocr`,
      formData,
      {
        headers,
        timeout: 120000, // 2 minutes timeout
      }
    )

    console.log('🔍 API Response:', response.data)
    console.log('🔍 Response status:', response.data.status)
    console.log('🔍 Response extracted_text:', response.data.extracted_text)

    return response.data
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const message = error.response?.data?.detail || error.message
      throw new Error(message)
    }
    throw error
  }
}

export async function getDefaultPrompt(): Promise<string> {
  try {
    const response = await axios.get(`${API_URL}/api/prompt`)
    return response.data.default_prompt
  } catch (error) {
    console.error('Failed to fetch default prompt:', error)
    return ''
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await axios.get(`${API_URL}/health`)
    return response.data.status === 'healthy'
  } catch (error) {
    console.error('Health check failed:', error)
    return false
  }
}

export async function processPDFOCR(
  pdfFile: File,
  settings: OCRSettings,
  onPageComplete: (result: PDFPageResult) => void,
  onMetadata: (totalPages: number) => void,
  authToken?: string | null
): Promise<void> {
  try {
    const formData = new FormData()
    formData.append('file', pdfFile)
    
    if (settings.customPrompt) {
      formData.append('custom_prompt', settings.customPrompt)
    }
    
    formData.append('max_new_tokens', settings.maxTokens.toString())
    formData.append('min_pixels', settings.minPixels.toString())
    formData.append('max_pixels', settings.maxPixels.toString())

    const headers: Record<string, string> = {}

    // Add auth token if provided
    if (authToken) {
      headers['Authorization'] = `Bearer ${authToken}`
    }

    const response = await fetch(`${API_URL}/api/ocr-pdf`, {
      method: 'POST',
      body: formData,
      headers,
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const reader = response.body?.getReader()
    const decoder = new TextDecoder()

    if (!reader) {
      throw new Error('No response body')
    }

    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      
      // Process complete lines
      const lines = buffer.split('\n')
      buffer = lines.pop() || '' // Keep the incomplete line in buffer

      for (const line of lines) {
        if (line.trim()) {
          try {
            const message: PDFStreamMessage = JSON.parse(line)
            
            if (message.type === 'metadata' && message.total_pages) {
              onMetadata(message.total_pages)
            } else if (message.type === 'page_result') {
              const pageResult: PDFPageResult = {
                page_number: message.page_number!,
                extracted_text: message.extracted_text || '',
                status: message.status || 'error',
                error: message.error,
                page_image: message.page_image
              }
              onPageComplete(pageResult)
            } else if (message.type === 'error') {
              throw new Error(message.error || 'Unknown error')
            }
          } catch (e) {
            console.error('Error parsing line:', line, e)
          }
        }
      }
    }
  } catch (error) {
    if (error instanceof Error) {
      throw error
    }
    throw new Error('Failed to process PDF')
  }
}

