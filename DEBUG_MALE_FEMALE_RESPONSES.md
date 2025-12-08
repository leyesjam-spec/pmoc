# Debugging: Why Male/Female Responses Are Empty

## Problem
The system is showing:
- `DEBUG - Received male_responses: 0 items`
- `DEBUG - Received female_responses: 0 items`
- `WARNING: Using questionnaire_responses as fallback (separate male/female responses not available)`

## Root Cause Analysis

### Data Flow
1. **Database Query** → `couple_responses` table
2. **Build response_map** → Key: `category_id_question_id_sub_question_id` → Value: `{male: val, female: val}`
3. **Build meai_questions structure** → From `question_assessment` and `sub_question_assessment` tables
4. **Loop through meai_questions** → Match keys with response_map → Populate arrays
5. **Return arrays** → Send to Python service

### Possible Issues

#### Issue 1: Empty response_map
**Symptom:** `response_map` has 0 keys
**Cause:** No data in `couple_responses` table for the `access_id`
**Check:** Look for log: `"ERROR - response_map is EMPTY!"`

#### Issue 2: Key Mismatch
**Symptom:** `response_map` has keys, but loop can't find matches
**Cause:** Keys built differently in response_map vs. loop
**Check:** Compare:
- response_map keys: `"1_2_3"` (category_question_sub)
- Loop keys: `"1_2_3"` (should match)

**Common mismatch causes:**
- `sub_question_id` is NULL in database → response_map uses `0`, loop should also use `0`
- `sub_question_id` in database doesn't match `sub_question_id` in `meai_questions` structure

#### Issue 3: Respondent Field Mismatch
**Symptom:** `response_map` has data, but `male_count = 0` and `female_count = 0`
**Cause:** `respondent` field in database doesn't match expected values
**Check:** Look for log: `"DEBUG - Unexpected respondent value"`
**Expected values:** `'male'` or `'female'` (case-insensitive)

#### Issue 4: Loop Doesn't Execute
**Symptom:** `loop_iterations = 0`
**Cause:** `meai_questions` structure is empty
**Check:** Look for log: `"ERROR - Loop did not execute!"`

## Debugging Steps

### Step 1: Check PHP Error Logs
Look for these debug messages in order:

1. **Database Query:**
   ```
   DEBUG - Total responses from database: X for access_id: Y
   ```

2. **Response Map Building:**
   ```
   DEBUG - Processed X male responses and Y female responses into response_map
   DEBUG - Total response_map keys: Z
   ```

3. **Loop Execution:**
   ```
   DEBUG - Loop executed X times
   DEBUG - Found responses for X keys
   DEBUG - Total missing keys: Y
   ```

4. **Array Counts:**
   ```
   DEBUG - Array counts IMMEDIATELY after building loop:
   DEBUG -   male_responses: X
   DEBUG -   female_responses: Y
   ```

### Step 2: Check Key Matching
If arrays are empty but response_map has keys, check:
```
DEBUG - Missing response keys (first 5): [...]
DEBUG - response_map keys sample: [...]
```

Compare the keys - they should match exactly.

### Step 3: Check Database Data
Run this SQL query to verify data exists:
```sql
SELECT 
    cr.category_id, 
    cr.question_id, 
    cr.sub_question_id,
    cr.respondent,
    cr.response,
    COUNT(*) as count
FROM couple_responses cr
WHERE cr.access_id = 'YOUR_ACCESS_ID'
GROUP BY cr.category_id, cr.question_id, cr.sub_question_id, cr.respondent
ORDER BY cr.category_id, cr.question_id, cr.sub_question_id;
```

**Expected:** Should return rows with `respondent` = 'male' or 'female'

### Step 4: Verify Respondent Field Values
Check if respondent field has unexpected values:
```sql
SELECT DISTINCT respondent, COUNT(*) 
FROM couple_responses 
WHERE access_id = 'YOUR_ACCESS_ID'
GROUP BY respondent;
```

**Expected:** Should show 'male' and 'female' (case may vary)

## Common Fixes

### Fix 1: Empty response_map
**Problem:** No data in database
**Solution:** Ensure couple has completed questionnaire and responses are saved

### Fix 2: Key Mismatch
**Problem:** sub_question_id mismatch
**Solution:** Ensure sub_question_id in database matches sub_question_id in meai_questions structure

### Fix 3: Respondent Field
**Problem:** Respondent field has wrong values
**Solution:** Update database to use 'male' or 'female' (case-insensitive)

### Fix 4: Arrays Reset
**Problem:** Arrays are built but then reset
**Solution:** Check for code that resets arrays after the loop

## Current Status

The code now includes extensive debugging that will show:
- ✅ Total rows fetched from database
- ✅ Response map key count
- ✅ Loop iteration count
- ✅ Found vs missing keys
- ✅ Array counts at each step

**Next Steps:**
1. Run an analysis
2. Check PHP error logs for the debug messages above
3. Identify which step is failing
4. Apply the appropriate fix

