import axios from 'axios'
import type { FieldType } from './structuredParser'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// =============================================================================
// QUEUE STATUS TYPES
// =============================================================================

export interface QueueStatus {
  queue_length: number
  workers_running: number
  workers_total: number
  estimated_wait_seconds: number
  estimated_wait_display: string
  avg_processing_time: number
  status: 'low_load' | 'moderate_load' | 'high_load' | 'very_high_load' | 'unknown'
  message: string
}

// =============================================================================
// QUEUE STATUS API
// =============================================================================

export async function getQueueStatus(
  operationType: 'image' | 'pdf_page' | 'structured' = 'image',
  authToken?: string | null
): Promise<QueueStatus> {
  try {
    const headers: Record<string, string> = {}
    if (authToken) {
      headers['Authorization'] = `Bearer ${authToken}`
    }

    const response = await axios.get<QueueStatus>(
      `${API_URL}/api/queue-status?operation_type=${operationType}`,
      { headers, timeout: 10000 }
    )
    return response.data
  } catch (error) {
    console.error('Failed to get queue status:', error)
    // Return safe defaults
    return {
      queue_length: 0,
      workers_running: 0,
      workers_total: 3,
      estimated_wait_seconds: 30,
      estimated_wait_display: '~30 seconds',
      avg_processing_time: 20,
      status: 'unknown',
      message: 'Processing will begin shortly'
    }
  }
}

// =============================================================================
// OCR TYPES
// =============================================================================

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
  confidence?: OCRConfidence
}

export type ContentType =
  | 'auto'
  | 'form'
  | 'document'
  | 'receipt'
  | 'invoice'
  | 'table'
  | 'id_card'
  | 'certificate'
  | 'handwritten'
  | 'mixed'
  | 'unknown'

export interface OCRTemplate {
  id: string
  user_id: string
  name: string
  description?: string | null
  content_type: Exclude<ContentType, 'auto'>
  language: 'ar' | 'en' | 'mixed'
  custom_prompt?: string | null
  sections?: any
  tables?: any
  keywords?: string[] | null
  is_public: boolean
  usage_count: number
  example_image_url?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export interface CreateTemplateRequest {
  name: string
  description?: string
  content_type: Exclude<ContentType, 'auto'>
  language?: 'ar' | 'en' | 'mixed'
  custom_prompt?: string
  sections?: any
  tables?: any
  keywords?: string[]
  is_public?: boolean
  example_image_url?: string
}

export interface UpdateTemplateRequest {
  name?: string
  description?: string
  content_type?: Exclude<ContentType, 'auto'>
  language?: 'ar' | 'en' | 'mixed'
  custom_prompt?: string
  sections?: any
  tables?: any
  keywords?: string[]
  is_public?: boolean
  example_image_url?: string
}

export interface PDFPageResult {
  page_number: number
  extracted_text: string
  status: string
  error?: string
  page_image?: string  // Base64 encoded
  confidence?: OCRConfidence
}

export interface PDFStreamMessage {
  type: 'metadata' | 'page_result' | 'complete' | 'error'
  total_pages?: number
  page_number?: number
  status?: string
  extracted_text?: string
  error?: string
  page_image?: string
  confidence?: OCRConfidence
}

export interface OCRConfidence {
  overall_confidence: number
  confidence_level: 'high' | 'medium' | 'low_medium' | 'low'
  confidence_sources?: {
    image_quality?: number | null
    token_logits?: number | null
    text_quality?: number | null
  }
  image_quality?: any
  text_quality?: any
  per_word?: Array<{ word: string; confidence: number | null }>
  per_line?: Array<any>
  warnings?: string[]
  recommendations?: string[]
}

// Structured Extraction Types
export interface ExtractedField {
  label: string
  value: string
  type: FieldType
}

export interface ExtractedSection {
  name: string | null
  fields: ExtractedField[]
}

export interface ExtractedTable {
  headers: string[]
  rows: string[][]
}

export interface ExtractedCheckbox {
  label: string
  checked: boolean
}

export interface StructuredExtractionData {
  form_title?: string | null
  sections: ExtractedSection[]
  tables: ExtractedTable[]
  checkboxes: ExtractedCheckbox[]
}

export interface StructuredOCRResponse {
  raw_text: string
  structured_data: StructuredExtractionData | null
  status: string
  error?: string
  confidence?: OCRConfidence
  parsing_successful: boolean
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
        timeout: 600000, // 10 minutes timeout (handles cold starts)
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
                page_image: message.page_image,
                confidence: message.confidence
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

export async function updateHistoryText(
  historyId: string,
  editedText: string,
  authToken: string | null
): Promise<void> {
  if (!authToken) {
    throw new Error('Authentication required')
  }

  try {
    await axios.patch(
      `${API_URL}/api/history/${historyId}`,
      { edited_text: editedText },
      {
        headers: {
          'Authorization': `Bearer ${authToken}`,
          'Content-Type': 'application/json'
        }
      }
    )
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const message = error.response?.data?.detail || error.message
      throw new Error(message)
    }
    throw error
  }
}

export async function fetchTemplates(authToken: string | null): Promise<OCRTemplate[]> {
  if (!authToken) {
    throw new Error('Authentication required')
  }

  try {
    const response = await axios.get<OCRTemplate[]>(`${API_URL}/api/templates`, {
      headers: {
        Authorization: `Bearer ${authToken}`
      }
    })
    return response.data
  } catch (error) {
    // Templates are optional; fail open so the app doesn't look "down" due to templates.
    console.warn('Failed to fetch user templates:', error)
    return []
  }
}

export async function fetchPublicTemplates(): Promise<OCRTemplate[]> {
  try {
    const response = await axios.get<OCRTemplate[]>(`${API_URL}/api/templates/public`)
    return response.data
  } catch (error) {
    // Public templates are optional; fail open.
    console.warn('Failed to fetch public templates:', error)
    return []
  }
}

export async function createTemplate(
  payload: CreateTemplateRequest,
  authToken: string | null
): Promise<OCRTemplate> {
  if (!authToken) {
    throw new Error('Authentication required')
  }

  try {
    const response = await axios.post<OCRTemplate>(`${API_URL}/api/templates`, payload, {
      headers: {
        Authorization: `Bearer ${authToken}`,
        'Content-Type': 'application/json'
      }
    })
    return response.data
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const message = (error.response?.data as any)?.detail || error.message
      throw new Error(message)
    }
    throw error
  }
}

export async function updateTemplate(
  templateId: string,
  payload: UpdateTemplateRequest,
  authToken: string | null
): Promise<OCRTemplate> {
  if (!authToken) {
    throw new Error('Authentication required')
  }

  try {
    const response = await axios.patch<OCRTemplate>(`${API_URL}/api/templates/${templateId}`, payload, {
      headers: {
        Authorization: `Bearer ${authToken}`,
        'Content-Type': 'application/json'
      }
    })
    return response.data
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const message = (error.response?.data as any)?.detail || error.message
      throw new Error(message)
    }
    throw error
  }
}

export async function deleteTemplate(templateId: string, authToken: string | null): Promise<void> {
  if (!authToken) {
    throw new Error('Authentication required')
  }

  try {
    await axios.delete(`${API_URL}/api/templates/${templateId}`, {
      headers: {
        Authorization: `Bearer ${authToken}`
      }
    })
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const message = (error.response?.data as any)?.detail || error.message
      throw new Error(message)
    }
    throw error
  }
}

// Structured OCR Extraction (Image)
export async function processStructuredOCR(
  imageFile: File,
  settings: {
    maxTokens: number
    minPixels: number
    maxPixels: number
    templateId?: string | null
  },
  authToken?: string | null
): Promise<StructuredOCRResponse> {
  try {
    const formData = new FormData()
    formData.append('file', imageFile)
    formData.append('max_new_tokens', settings.maxTokens.toString())
    formData.append('min_pixels', settings.minPixels.toString())
    formData.append('max_pixels', settings.maxPixels.toString())
    
    if (settings.templateId) {
      formData.append('template_id', settings.templateId)
    }

    const headers: Record<string, string> = {}
    if (authToken) {
      headers['Authorization'] = `Bearer ${authToken}`
    }

    const response = await axios.post<StructuredOCRResponse>(
      `${API_URL}/api/ocr-structured`,
      formData,
      {
        headers,
        timeout: 600000, // 10 minutes timeout
      }
    )

    return response.data
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const message = error.response?.data?.detail || error.message
      throw new Error(message)
    }
    throw error
  }
}

// Structured PDF Page Result
export interface StructuredPDFPageResult {
  page_number: number
  raw_text?: string
  structured_data?: StructuredExtractionData
  parsing_successful?: boolean
  status: string
  error?: string
  page_image?: string
  confidence?: OCRConfidence
}

// Structured PDF Stream Message
export interface StructuredPDFStreamMessage {
  type: 'metadata' | 'page_result' | 'complete' | 'error'
  total_pages?: number
  page_number?: number
  status?: string
  raw_text?: string
  structured_data?: StructuredExtractionData
  parsing_successful?: boolean
  error?: string
  page_image?: string
  confidence?: OCRConfidence
  successful_pages?: number
  error_pages?: number
}

// Structured PDF OCR Extraction
export async function processStructuredPDFOCR(
  pdfFile: File,
  settings: {
    maxTokens: number
    minPixels: number
    maxPixels: number
    templateId?: string | null
  },
  onPageComplete: (result: StructuredPDFPageResult) => void,
  onMetadata: (totalPages: number) => void,
  authToken?: string | null
): Promise<void> {
  try {
    const formData = new FormData()
    formData.append('file', pdfFile)
    formData.append('max_new_tokens', settings.maxTokens.toString())
    formData.append('min_pixels', settings.minPixels.toString())
    formData.append('max_pixels', settings.maxPixels.toString())
    
    if (settings.templateId) {
      formData.append('template_id', settings.templateId)
    }

    const headers: Record<string, string> = {}
    if (authToken) {
      headers['Authorization'] = `Bearer ${authToken}`
    }

    const response = await fetch(`${API_URL}/api/ocr-pdf-structured`, {
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
      
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.trim()) {
          try {
            const message: StructuredPDFStreamMessage = JSON.parse(line)
            
            if (message.type === 'metadata' && message.total_pages) {
              onMetadata(message.total_pages)
            } else if (message.type === 'page_result') {
              const pageResult: StructuredPDFPageResult = {
                page_number: message.page_number!,
                raw_text: message.raw_text,
                structured_data: message.structured_data,
                parsing_successful: message.parsing_successful,
                status: message.status || 'error',
                error: message.error,
                page_image: message.page_image,
                confidence: message.confidence
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

