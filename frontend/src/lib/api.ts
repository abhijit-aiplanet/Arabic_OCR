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

export async function processOCR(
  imageFile: File,
  settings: OCRSettings
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

    const response = await axios.post<OCRResponse>(
      `${API_URL}/api/ocr`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 120000, // 2 minutes timeout
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

