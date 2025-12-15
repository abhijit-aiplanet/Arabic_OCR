import type { ContentType } from '@/lib/api'

export function detectContentTypeFromText(text: string): Exclude<ContentType, 'auto'> {
  const t = (text || '').trim()
  if (!t) return 'unknown'

  const colonCount = (t.match(/:/g) || []).length
  const hasCheckbox = t.includes('☑') || t.includes('☐')
  const hasSectionSep = /━{3,}/.test(t)
  const hasTableSep = t.includes('|') || t.includes('│')
  
  // Arabic form indicators
  const hasNumberedLists = /^\s*[\d٠-٩]+\s*[-–—]\s*/m.test(t) // Lines starting with numbers + dash
  const hasFormKeywords = /(طلب|استمارة|نموذج|بيانات|معلومات|توقيع|التاريخ|الاسم|العنوان|الصفة)/.test(t)
  const hasFieldStructure = /(:\s*$|:\s*\.)/m.test(t) // Colons at end of line or followed by period

  // Receipt-like keywords (Arabic)
  const receiptKw = /(الإجمالي|المجموع|ضريبة|فاتورة|إيصال|المبلغ|المبلغ الإجمالي|المجموع الكلي|السعر|الكمية)/.test(t)

  if (hasTableSep && receiptKw) return 'invoice'
  if (receiptKw) return 'receipt'
  if (hasTableSep) return 'table'

  // Form-like: many label:value pairs OR checkboxes OR section separators OR Arabic form patterns
  if (colonCount >= 3 || hasCheckbox || hasSectionSep || (hasNumberedLists && (colonCount >= 2 || hasFormKeywords || hasFieldStructure))) {
    return 'form'
  }

  // Document-like: longer text, fewer colons
  if (t.length > 250 && colonCount <= 1) return 'document'

  return 'unknown'
}


