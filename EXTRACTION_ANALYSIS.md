# Structured Extraction Analysis - Arabic Documents

## Document Types Identified

From analyzing real user test documents, we've identified these common patterns:

### Type 1: Table-Style Forms
- Multiple fields per row in table cells
- Sections in bordered boxes
- Empty placeholders: `............`, `/ / ١٤هـ`

### Type 2: Checklist/Requirements Documents
- Numbered lists (١، ٢، ٣... or 1, 2, 3...)
- Section headers ending with `:` like `طلبات الإصدار الجديد :`
- Items without values (standalone list items)

### Type 3: Signature Areas
- Fields: اسم مقدم الطلب، صفته، توقيعه، التاريخ، الختم
- May contain handwritten signatures/stamps

### Type 4: Mixed Documents
- Combination of forms, lists, and signatures
- Handwritten annotations overlaid on printed forms

---

## Form Structure Analysis (Type 1 - Table Form)

The uploaded form is a complex Saudi Arabian vehicle/business registration form with:

### 1. **Multiple Fields Per Line (CRITICAL ISSUE)**
```
| رقم التشغيل : ............ | تاريخ الإصدار : / / ١٤هـ | تاريخ الانتهاء : / / ١٤هـ |
```
Many rows contain 2-3 separate field:value pairs horizontally.

### 2. **Section Headers in Bordered Boxes**
- بيانات عامة (General Data)
- عنوان المراسلات (Correspondence Address)
- بيانات رخصة القيادة (Driving License Data)
- بيانات المركبة (Vehicle Data)

### 3. **Empty Fields with Placeholders**
Fields show `............` or `/ / ١٤هـ` as placeholders, making it hard to distinguish empty from filled.

### 4. **Mixed Content**
- Printed Arabic labels
- Handwritten Arabic values
- Numbers in both Arabic (٠١٢٣٤٥٦٧٨٩) and Western (0123456789) numerals
- Dates in Islamic calendar format (هـ)

### 5. **Table-Like Row Structure**
The form is essentially a table where each row may contain multiple label:value cells.

---

## Why Current Extraction Fails

### Problem 1: VLM Reads Linearly
The VLM reads text left-to-right (or right-to-left for Arabic) but doesn't understand the **columnar structure** where one line has multiple independent fields.

Example line: `تاريخ الانتهاء : / / ١٤هـ | تاريخ الإصدار : / / ١٤هـ | رقم التشغيل : ............`

VLM might output: `تاريخ الانتهاء : / / ١٤هـ تاريخ الإصدار : / / ١٤هـ رقم التشغيل`
Missing the structure!

### Problem 2: Placeholder Confusion
`............` placeholders might be:
- Interpreted as actual content
- Merged with adjacent text
- Skipped entirely

### Problem 3: Natural Prompt Too Generic
Current prompt says "extract all fields" but doesn't explain:
- How to handle multiple fields per line
- What section headers look like
- How to handle empty vs filled fields

### Problem 4: Parser Expects One Field Per Line
Current parser processes line-by-line, but this form has multiple fields per line separated by visual boundaries (table cells).

---

## Proposed Solution

### 1. **Specialized Table-Aware Prompt**

```
You are reading an Arabic form that is structured as a TABLE.

CRITICAL: Each ROW may contain MULTIPLE fields separated by vertical lines or spaces.

For EACH field you see:
- Write: field_label: value
- If empty (just dots .... or empty space): field_label: [فارغ]
- If date placeholder (/ / ١٤هـ): field_label: [فارغ]

Section headers are in bordered boxes. Mark them as:
[SECTION] section_name

Read the form ROW BY ROW, extracting ALL fields from each row.

Example row with 3 fields:
"رقم التشغيل : ........ | تاريخ الإصدار : 1445/1/1 | تاريخ الانتهاء : / / ١٤هـ"

Should output:
رقم التشغيل: [فارغ]
تاريخ الإصدار: 1445/1/1
تاريخ الانتهاء: [فارغ]
```

### 2. **Enhanced Parser for Multi-Field Lines**

The parser should:
1. Split lines by common separators: `|`, `،`, multiple spaces, tab characters
2. Then parse each segment for label:value pairs
3. Handle Arabic placeholder patterns

### 3. **Section Detection**

Look for section header patterns:
- Text inside bordered boxes
- Bold/larger text
- Common section words: بيانات, عنوان, معلومات

---

## Complete Field List Expected from This Form

### بيانات عامة (General Data) - 13 fields
1. رقم التشغيل
2. تاريخ الإصدار
3. تاريخ الانتهاء
4. نوع النشاط
5. مدينة مزاولة النشاط
6. اسم المالك
7. تاريخ الميلاد
8. المؤهل
9. رقم بطاقة الأحوال
10. تاريخها
11. مصدرها
12. الحالة الاجتماعية
13. عدد من يعولهم

### عنوان المراسلات (Correspondence Address) - 10 fields
1. المدينة
2. الحي
3. رقم القطعة
4. ص.ب
5. رمز بريدي
6. هاتف
7. المشرع/الشارع
8. جوال
9. فاكس
10. بريد إلكتروني

### بيانات رخصة القيادة (Driving License) - 4 fields
1. رقمها
2. تاريخ الإصدار
3. تاريخ الانتهاء
4. مصدرها

### بيانات المركبة (Vehicle Data) - 15 fields
1. رقم اللوحة
2. رقم الاستمارة
3. نوع الوقود
4. رقم الهيكل
5. تاريخ الإصدار
6. عدد القاعد
7. سنة الصنع
8. تاريخ الانتهاء
9. قيمة المركبة
10. الشركة الصانعة
11. سعة المحرك
12. تاريخ صلاحيتها
13. طراز المركبة
14. عدد الاسطوانات
15. تاريخ تسجيلها في النشاط

### Applicant Section - 4 fields
1. اسم مقدم الطلب
2. صفته
3. توقيعه
4. التاريخ
5. الختم

**TOTAL: ~46 fields**

---

## Implementation Priority

1. **HIGH**: Update VLM prompt to be table-aware
2. **HIGH**: Add multi-field-per-line parsing
3. **MEDIUM**: Add section header detection
4. **MEDIUM**: Better empty field handling
5. **LOW**: Arabic numeral normalization

---

## Type 2: Checklist/Requirements Analysis

### Content Structure

**Section Headers:**
- `طلبات الإصدار الجديد :` (New Issuance Requirements)
- `طلبات التجديد :` (Renewal Requirements)

**List Items (Arabic Numerals):**
```
١- صورة الاستمارة .
٢- صورة شمسية ٤*٦ .
٣- كشف اللجنة .
```

### Key Patterns

| Pattern | Example | Handling |
|---------|---------|----------|
| Arabic numerals | ١، ٢، ٣، ٤، ٥، ٦، ٧ | Parse as list markers |
| List item separators | `-`, `–`, `.`, `)` | Recognize all variations |
| Trailing periods | `صورة الاستمارة .` | Strip trailing `.` |
| Section headers | `طلبات ... :` | Detect and group |
| Standalone items | No `:` in content | Store as list items |

### Parsing Rules

1. **Detect checklist sections** by keywords: طلبات، متطلبات، شروط
2. **Parse numbered items** with both Western and Arabic numerals
3. **Handle items without values** as standalone list entries
4. **Strip trailing punctuation** from list items
5. **Group items by section**

---

## Generalization Strategy

### 1. Document Detection
- If mostly `label: value` pairs → Form mode
- If numbered lists without values → Checklist mode
- If signature fields → Include signature detection

### 2. Prompt Adaptation
The prompt should handle ALL document types:
- Table forms (multiple fields per row)
- Checklists (numbered lists)
- Signature areas
- Mixed content

### 3. Parser Flexibility
- Multiple field patterns
- Arabic + Western numerals
- Empty value normalization
- Section grouping
- List item support

### 4. UI/UX Adaptation
- Forms: Display as cards/table
- Checklists: Display as bulleted lists
- Signatures: Show presence/absence indicators
