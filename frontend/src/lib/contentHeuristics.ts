import type { ContentType } from '@/lib/api'

export function detectContentTypeFromText(text: string): Exclude<ContentType, 'auto'> {
  const t = (text || '').trim()
  if (!t) return 'unknown'

  const colonCount = (t.match(/:/g) || []).length
  const hasCheckbox = t.includes('☑') || t.includes('☐')
  const hasSectionSep = /━{3,}/.test(t)
  const hasTableSep = t.includes('|') || t.includes('│')

  // Receipt-like keywords (Arabic)
  const receiptKw = /(الإجمالي|المجموع|ضريبة|فاتورة|إيصال|المبلغ|المبلغ الإجمالي|المجموع الكلي|السعر|الكمية)/.test(t)

  if (hasTableSep && receiptKw) return 'invoice'
  if (receiptKw) return 'receipt'
  if (hasTableSep) return 'table'

  // Form-like: many label:value pairs OR checkboxes OR strong section separators
  if (colonCount >= 3 || hasCheckbox || hasSectionSep) return 'form'

  // Document-like: longer text, fewer colons
  if (t.length > 250 && colonCount <= 1) return 'document'

  return 'unknown'
}


