# Supabase: Optional Confidence Columns for `ocr_history` (SQL)

This app **already stores confidence payload inside** `ocr_history.settings.confidence` for compatibility.

If you want dedicated columns + indexing (as described in `CONFIDENCE_SCORING_IMPLEMENTATION.md`), run:

```sql
alter table public.ocr_history
  add column if not exists confidence_overall float,
  add column if not exists confidence_level text,
  add column if not exists confidence_image_quality float,
  add column if not exists confidence_token_logits float,
  add column if not exists confidence_text_quality float,
  add column if not exists confidence_per_line jsonb,
  add column if not exists confidence_per_word jsonb,
  add column if not exists quality_factors jsonb,
  add column if not exists low_confidence_areas jsonb,
  add column if not exists confidence_warnings text[],
  add column if not exists confidence_recommendations text[];

create index if not exists idx_ocr_history_confidence_overall on public.ocr_history(confidence_overall);
create index if not exists idx_ocr_history_confidence_level on public.ocr_history(confidence_level);
```

## Notes
- After you add columns, we can update the backend to **write both**:
  - `settings.confidence` (full payload)
  - these columns (for fast querying + filtering)


