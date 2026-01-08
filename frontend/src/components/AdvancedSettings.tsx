'use client'

import { useState } from 'react'
import { Settings, ChevronDown, ChevronUp, Eye, EyeOff } from 'lucide-react'

interface AdvancedSettingsProps {
  settings: {
    customPrompt: string
    maxTokens: number
    minPixels: number
    maxPixels: number
  }
  onSettingsChange: (settings: any) => void
}

const DEFAULT_PROMPT = `Extract all Arabic text from this image with maximum accuracy.

CRITICAL: This image contains ARABIC text that may be:
- Handwritten Arabic text
- Typed/printed Arabic text
- Small Arabic annotations or notes
- Arabic text in forms, tables, or documents

Your task: Find and extract EVERY piece of Arabic text visible in the image, no matter how small.

Requirements:
1. Extract ALL Arabic text - handwritten, typed, printed, or annotations
2. Accuracy is CRITICAL - extract Arabic text exactly as it appears
3. Do NOT miss any Arabic text, even small notes or annotations
4. Maintain the original text structure, layout, and formatting
5. Preserve line breaks, paragraphs, and spacing as they appear
6. Do NOT translate - keep all text in Arabic
7. Do NOT add descriptions, interpretations, or commentary
8. If there are tables or forms, maintain their structure
9. If there are headers, titles, or sections, preserve their hierarchy
10. Focus on ACCURACY over speed

Output only the extracted Arabic text, nothing else.`

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
    <div className="bg-white rounded-2xl border border-gray-100 overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-5 py-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Settings className="w-5 h-5 text-gray-400" />
          <span className="text-base font-semibold text-gray-900">Settings</span>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-gray-400" />
        ) : (
          <ChevronDown className="w-5 h-5 text-gray-400" />
        )}
      </button>

      {isExpanded && (
        <div className="px-5 pb-5 space-y-5 border-t border-gray-100 pt-5">
          {/* Custom Prompt */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-gray-700">
                Custom Prompt
              </label>
              <button
                onClick={() => setShowDefaultPrompt(!showDefaultPrompt)}
                className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-700 transition-colors"
              >
                {showDefaultPrompt ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                {showDefaultPrompt ? 'Hide default' : 'View default'}
              </button>
            </div>
            <textarea
              value={settings.customPrompt}
              onChange={(e) => handleChange('customPrompt', e.target.value)}
              placeholder="Leave empty for default OCR prompt..."
              rows={3}
              className="w-full px-3.5 py-2.5 bg-gray-50 border border-gray-200 rounded-xl focus:ring-2 focus:ring-gray-900 focus:border-transparent resize-none text-sm text-gray-900 placeholder-gray-400 transition-all"
            />

            {showDefaultPrompt && (
              <div className="mt-3 p-4 bg-gray-50 rounded-xl border border-gray-200">
                <p className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Default Prompt</p>
                <pre className="text-xs text-gray-600 whitespace-pre-wrap leading-relaxed">
                  {DEFAULT_PROMPT}
                </pre>
              </div>
            )}
          </div>

          {/* Max Tokens */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-gray-700">
                Max Tokens
              </label>
              <span className="text-sm font-mono text-gray-500">{settings.maxTokens.toLocaleString()}</span>
            </div>
            <input
              type="range"
              min="1024"
              max="8192"
              step="256"
              value={settings.maxTokens}
              onChange={(e) => handleChange('maxTokens', parseInt(e.target.value))}
              className="w-full h-1.5 bg-gray-200 rounded-full appearance-none cursor-pointer accent-gray-900"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1.5">
              <span>1,024</span>
              <span>4,096</span>
              <span>8,192</span>
            </div>
          </div>

          {/* Image Resolution */}
          <div>
            <label className="text-sm font-medium text-gray-700 mb-3 block">
              Resolution Settings
            </label>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-gray-500 mb-1.5 block">
                  Min Pixels
                </label>
                <input
                  type="number"
                  value={settings.minPixels}
                  onChange={(e) => handleChange('minPixels', parseInt(e.target.value))}
                  className="w-full px-3 py-2 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent text-sm text-gray-900 font-mono"
                />
              </div>

              <div>
                <label className="text-xs text-gray-500 mb-1.5 block">
                  Max Pixels
                </label>
                <input
                  type="number"
                  value={settings.maxPixels}
                  onChange={(e) => handleChange('maxPixels', parseInt(e.target.value))}
                  className="w-full px-3 py-2 bg-gray-50 border border-gray-200 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-transparent text-sm text-gray-900 font-mono"
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
