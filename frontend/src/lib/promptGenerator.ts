import type { ContentType, OCRTemplate } from '@/lib/api'

function baseRules() {
  return `General rules:
- Extract Arabic text exactly as it appears (no translation).
- Preserve reading order and layout as much as possible.
- Keep line breaks and spacing meaningful.
- Do not add commentary. Output only the extracted content.`
}

export function generatePromptForContentType(contentType: ContentType): string {
  switch (contentType) {
    case 'form':
      return `You are extracting text from an Arabic FORM (typed + handwritten).

Goal: preserve the original form structure so fields stay connected to their values.

Form rules:
- Keep section headers and separators.
- For each field, output: Label: Value
- If a field is empty, output: Label: [فارغ]
- Preserve tables using a clear grid with | separators.
- Preserve checkboxes as: ☑ (checked) or ☐ (unchecked)

${baseRules()}`

    case 'table':
      return `You are extracting an Arabic TABLE.

Rules:
- Preserve all rows and columns.
- Use | to separate columns.
- Keep headers as the first row.
- Do not merge columns.

${baseRules()}`

    case 'receipt':
    case 'invoice':
      return `You are extracting an Arabic receipt/invoice.

Rules:
- Preserve itemized lines (item, quantity, price).
- Preserve totals/subtotals clearly.
- Keep dates, numbers, and currency as-is.
- If there is a table, use | separators.

${baseRules()}`

    case 'document':
      return `You are extracting an Arabic document (paragraph text).

Rules:
- Preserve paragraphs, headings, and bullet points.
- Keep line breaks where they reflect structure.

${baseRules()}`

    case 'id_card':
      return `You are extracting an Arabic ID card.

Rules:
- Output key fields as Label: Value (e.g. الاسم, رقم الهوية, تاريخ الميلاد, الجنسية).
- Preserve the exact values.

${baseRules()}`

    case 'certificate':
      return `You are extracting an Arabic certificate.

Rules:
- Preserve title vs body vs signatures/stamps text if readable.
- Keep hierarchy and line breaks.

${baseRules()}`

    case 'handwritten':
      return `You are extracting handwritten Arabic text.

Rules:
- Preserve line breaks as written.
- If a word is unclear, keep your best guess; if impossible, use [غير واضح].

${baseRules()}`

    case 'mixed':
      return `You are extracting Arabic content that may include mixed structures (form + table + paragraphs).

Rules:
- Use clear separators between sections.
- Preserve tables with |.
- Preserve fields as Label: Value when applicable.

${baseRules()}`

    case 'unknown':
    case 'auto':
    default:
      return `Extract all Arabic text with maximum accuracy.

Rules:
- Preserve structure and layout as much as possible (forms, tables, paragraphs).
- Keep line breaks and spacing.

${baseRules()}`
  }
}

export function getEffectivePrompt(opts: {
  userCustomPrompt?: string
  contentType: ContentType
  template?: OCRTemplate | null
}): { prompt: string; source: 'user' | 'template' | 'content_type' } {
  const userPrompt = (opts.userCustomPrompt || '').trim()
  if (userPrompt) return { prompt: userPrompt, source: 'user' }

  const templatePrompt = (opts.template?.custom_prompt || '').trim()
  if (templatePrompt) return { prompt: templatePrompt, source: 'template' }

  return { prompt: generatePromptForContentType(opts.contentType), source: 'content_type' }
}


