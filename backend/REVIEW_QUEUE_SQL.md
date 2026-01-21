# Supabase: Review Queue Table (SQL)

This SQL creates the `ocr_review_queue` table for human-in-the-loop OCR verification.

## Why This Table?

The OCR system routes low-confidence and suspicious extractions to a review queue where humans can:
- Verify extracted values
- Correct OCR errors
- Reject hallucinated values
- Build ground truth data for future improvements

## Create Table

```sql
-- ============================================================================
-- OCR REVIEW QUEUE TABLE
-- ============================================================================
-- Stores fields that need human review due to:
-- - Low confidence scores
-- - Suspicious/hallucination patterns
-- - Validation failures
-- - Unreadable markers

CREATE TABLE IF NOT EXISTS public.ocr_review_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Link to original OCR history
    ocr_history_id UUID REFERENCES public.ocr_history(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    
    -- Field-level data
    field_name TEXT NOT NULL,
    extracted_value TEXT,
    
    -- Confidence information
    confidence_score FLOAT,
    confidence_level TEXT,  -- 'high', 'medium', 'low', 'unreadable', 'empty'
    
    -- Why this field needs review
    review_reason TEXT,
    validation_status TEXT,  -- 'valid', 'invalid', 'suspicious', 'unchecked'
    validation_message TEXT,
    
    -- Hallucination detection
    is_suspicious BOOLEAN DEFAULT false,
    suspicion_score FLOAT,
    suspicion_reasons TEXT[],
    
    -- Visual context for reviewer
    region_image_url TEXT,  -- Cropped field image (optional)
    full_image_url TEXT,    -- Full document image
    
    -- Review status tracking
    status TEXT DEFAULT 'pending',  -- 'pending', 'approved', 'corrected', 'rejected', 'skipped'
    
    -- Review outcome
    reviewed_by TEXT,
    reviewed_at TIMESTAMP WITH TIME ZONE,
    corrected_value TEXT,
    reviewer_notes TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Query pending items for a user
CREATE INDEX IF NOT EXISTS idx_review_queue_user_status 
ON public.ocr_review_queue(user_id, status);

-- Query by confidence level
CREATE INDEX IF NOT EXISTS idx_review_queue_confidence 
ON public.ocr_review_queue(confidence_level);

-- Query by OCR history
CREATE INDEX IF NOT EXISTS idx_review_queue_history 
ON public.ocr_review_queue(ocr_history_id);

-- Query suspicious items
CREATE INDEX IF NOT EXISTS idx_review_queue_suspicious 
ON public.ocr_review_queue(is_suspicious) WHERE is_suspicious = true;

-- ============================================================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================================================

-- Enable RLS
ALTER TABLE public.ocr_review_queue ENABLE ROW LEVEL SECURITY;

-- Users can only see their own review items
CREATE POLICY "Users can view own review items"
ON public.ocr_review_queue FOR SELECT
TO authenticated
USING (user_id = auth.uid()::text);

-- Users can insert their own review items
CREATE POLICY "Users can insert own review items"
ON public.ocr_review_queue FOR INSERT
TO authenticated
WITH CHECK (user_id = auth.uid()::text);

-- Users can update their own review items
CREATE POLICY "Users can update own review items"
ON public.ocr_review_queue FOR UPDATE
TO authenticated
USING (user_id = auth.uid()::text);

-- ============================================================================
-- AUTO-UPDATE TIMESTAMP TRIGGER
-- ============================================================================

CREATE OR REPLACE FUNCTION update_review_queue_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_review_queue_timestamp
BEFORE UPDATE ON public.ocr_review_queue
FOR EACH ROW
EXECUTE FUNCTION update_review_queue_timestamp();
```

## Review Statistics View (Optional)

```sql
-- Create a view for review statistics
CREATE OR REPLACE VIEW public.review_statistics AS
SELECT 
    user_id,
    COUNT(*) FILTER (WHERE status = 'pending') as pending_count,
    COUNT(*) FILTER (WHERE status = 'approved') as approved_count,
    COUNT(*) FILTER (WHERE status = 'corrected') as corrected_count,
    COUNT(*) FILTER (WHERE status = 'rejected') as rejected_count,
    COUNT(*) as total_count,
    AVG(confidence_score) FILTER (WHERE status IN ('approved', 'corrected')) as avg_confidence_approved,
    COUNT(*) FILTER (WHERE is_suspicious) as suspicious_count
FROM public.ocr_review_queue
GROUP BY user_id;
```

## Usage Notes

1. **Pending Items**: Query with `status = 'pending'` to get items awaiting review
2. **Priority Sorting**: Sort by `confidence_score ASC` to review lowest confidence first
3. **Batch Operations**: Can update multiple items at once with the same `corrected_value`
4. **Analytics**: Use corrected values to track OCR accuracy and build training data
