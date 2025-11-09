'use client'

import { useState } from 'react'
import { Settings, ChevronDown, ChevronUp, Eye } from 'lucide-react'

interface AdvancedSettingsProps {
  settings: {
    customPrompt: string
    maxTokens: number
    minPixels: number
    maxPixels: number
  }
  onSettingsChange: (settings: any) => void
}

const DEFAULT_PROMPT = `Extract all text from this image exactly as it appears. 

Requirements:
1. Extract ONLY the text content - do not describe, analyze, or interpret the image
2. Maintain the original text structure, layout, and formatting
3. Preserve line breaks, paragraphs, and spacing as they appear
4. Do not translate the text - keep it in its original language
5. Do not add any explanations, descriptions, or additional commentary
6. If there are tables, maintain their structure
7. If there are headers, titles, or sections, preserve their hierarchy

Output only the extracted text, nothing else.`

export default function AdvancedSettings({
  settings,
  onSettingsChange,
}: AdvancedSettingsProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [showDefaultPrompt, setShowDefaultPrompt] = useState(false)

  const handleChange = (field: string, value: any) => {
    onSettingsChange({
      ...settings,
      [field]: value,
    })
  }

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Settings className="w-5 h-5 text-gray-600" />
          <h2 className="text-lg font-semibold text-gray-900">
            Advanced Settings
          </h2>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-gray-600" />
        ) : (
          <ChevronDown className="w-5 h-5 text-gray-600" />
        )}
      </button>

      {isExpanded && (
        <div className="px-6 pb-6 space-y-4 border-t border-gray-200 pt-4">
          {/* Custom Prompt */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-gray-700">
                Custom Prompt (Optional)
              </label>
              <button
                onClick={() => setShowDefaultPrompt(!showDefaultPrompt)}
                className="text-xs text-blue-600 hover:text-blue-700 flex items-center gap-1"
              >
                <Eye className="w-3 h-3" />
                {showDefaultPrompt ? 'Hide' : 'Show'} Default
              </button>
            </div>
            <textarea
              value={settings.customPrompt}
              onChange={(e) => handleChange('customPrompt', e.target.value)}
              placeholder="Leave empty to use default OCR prompt..."
              rows={3}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-sm"
            />
            <p className="text-xs text-gray-500 mt-1">
              Customize the prompt if you want specific extraction behavior
            </p>

            {showDefaultPrompt && (
              <div className="mt-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
                <p className="text-xs font-medium text-gray-700 mb-2">Default Prompt:</p>
                <pre className="text-xs text-gray-600 whitespace-pre-wrap">
                  {DEFAULT_PROMPT}
                </pre>
              </div>
            )}
          </div>

          {/* Max Tokens */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Max Tokens: {settings.maxTokens.toLocaleString()}
            </label>
            <input
              type="range"
              min="1024"
              max="12288"
              step="256"
              value={settings.maxTokens}
              onChange={(e) => handleChange('maxTokens', parseInt(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>1,024</span>
              <span>8,192 (default)</span>
              <span>12,288</span>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              Maximum length of extracted text (increased for longer documents)
            </p>
          </div>

          {/* Image Resolution */}
          <div>
            <p className="block text-sm font-medium text-gray-700 mb-3">
              📐 Image Resolution Settings
            </p>
            <p className="text-xs text-gray-500 mb-3">
              Controls visual token range (4-16384) - balance quality vs speed
            </p>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Min Pixels
                </label>
                <input
                  type="number"
                  value={settings.minPixels}
                  onChange={(e) => handleChange('minPixels', parseInt(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Default: 200,704
                </p>
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Max Pixels
                </label>
                <input
                  type="number"
                  value={settings.maxPixels}
                  onChange={(e) => handleChange('maxPixels', parseInt(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                />
                <p className="text-xs text-gray-500 mt-1">
                  Default: 2,007,040 (2x increase for large images)
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

