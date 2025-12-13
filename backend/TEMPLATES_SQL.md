# Supabase: OCR Templates Table (SQL)

Run this in Supabase SQL editor to enable user templates.

```sql
-- Enable uuid extension if not enabled
create extension if not exists "uuid-ossp";

create table if not exists public.ocr_templates (
  id uuid primary key default uuid_generate_v4(),
  user_id text not null,
  name text not null,
  description text,
  content_type text not null, -- form | document | receipt | invoice | table | id_card | certificate | handwritten | mixed | unknown
  language text not null default 'ar', -- ar | en | mixed
  custom_prompt text,
  sections jsonb not null default '{}'::jsonb,
  tables jsonb,
  keywords text[],
  is_public boolean not null default false,
  usage_count integer not null default 0,
  example_image_url text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_ocr_templates_user_id on public.ocr_templates(user_id);
create index if not exists idx_ocr_templates_is_public on public.ocr_templates(is_public);
create index if not exists idx_ocr_templates_content_type on public.ocr_templates(content_type);
create index if not exists idx_ocr_templates_usage_count on public.ocr_templates(usage_count);

-- Optional RLS (recommended if you ever switch backend to anon key)
alter table public.ocr_templates enable row level security;

-- Users can read their own templates (when using supabase auth; for this app backend uses service role key)
create policy "read own templates"
on public.ocr_templates
for select
using (true);

create policy "insert own templates"
on public.ocr_templates
for insert
with check (true);

create policy "update own templates"
on public.ocr_templates
for update
using (true);

create policy "delete own templates"
on public.ocr_templates
for delete
using (true);

-- Public templates readable by anyone (if you later expose direct supabase reads)
create policy "read public templates"
on public.ocr_templates
for select
using (is_public = true);
```


