# Couple Responses Table Structure and Relationships

## Table: `couple_responses`

### Key Fields:
- `access_id` (INT) - Links responses from both partners (male and female share the same access_id)
- `category_id` (INT) - Question category
- `question_id` (INT) - Question ID
- `sub_question_id` (INT, nullable) - Sub-question ID (NULL for standalone questions)
- `respondent` (VARCHAR) - **CRITICAL**: Flags whether response is from 'male' or 'female'
- `response` (VARCHAR/TEXT) - The actual response value
- `reason` (TEXT, optional) - Reason for the response

### Relationship:
```
couple_profile (access_id, sex='Male' or 'Female')
    ↓
couple_responses (access_id, respondent='male' or 'female')
```

**Important:**
- Both male and female profiles share the same `access_id`
- The `respondent` field in `couple_responses` distinguishes between male and female responses
- For each question, there should be TWO rows: one with `respondent='male'` and one with `respondent='female'`

### Example Data Structure:
```
access_id | category_id | question_id | sub_question_id | respondent | response
----------|-------------|-------------|-----------------|------------|----------
48        | 1           | 1           | NULL            | male       | agree
48        | 1           | 1           | NULL            | female     | neutral
48        | 1           | 2           | 1               | male       | disagree
48        | 1           | 2           | 1               | female     | agree
```

## How the Code Uses This:

1. **Query all responses for an access_id:**
   ```sql
   SELECT response, category_id, question_id, sub_question_id, respondent
   FROM couple_responses
   WHERE access_id = ?
   ORDER BY category_id, question_id, COALESCE(sub_question_id, 0), respondent
   ```

2. **Build response_map:**
   - Key: `{category_id}_{question_id}_{sub_question_id}` (e.g., "1_2_0" or "1_2_1")
   - Value: `{'male': response_value, 'female': response_value}`

3. **Separate into arrays:**
   - `male_responses[]` - All male responses in question order
   - `female_responses[]` - All female responses in question order
   - `questionnaire_responses[]` - Combined/averaged responses

## Potential Issues:

1. **Respondent field values:**
   - Expected: 'male' or 'female' (case-insensitive)
   - Code checks: `strtolower($respondent) === 'male'` or `'female'`
   - If database has 'Male', 'Female', 'MALE', 'FEMALE', etc., they should still work due to case-insensitive check

2. **Missing responses:**
   - If only one partner has responded, arrays will be incomplete
   - Code should handle this by using defaults (value 3 = neutral)

3. **Key mismatch:**
   - Keys must match exactly when building `response_map` vs when looking up responses
   - NULL `sub_question_id` should be converted to `0` for key building

## Debugging Queries:

### Check respondent values:
```sql
SELECT DISTINCT respondent, COUNT(*) as count
FROM couple_responses
WHERE access_id = ?
GROUP BY respondent;
```

### Check if both male and female responses exist:
```sql
SELECT 
    category_id, 
    question_id, 
    sub_question_id,
    COUNT(DISTINCT respondent) as respondent_count,
    GROUP_CONCAT(DISTINCT respondent) as respondents
FROM couple_responses
WHERE access_id = ?
GROUP BY category_id, question_id, sub_question_id
HAVING respondent_count < 2;
```

### Sample responses for an access_id:
```sql
SELECT response, category_id, question_id, sub_question_id, respondent
FROM couple_responses
WHERE access_id = ?
ORDER BY category_id, question_id, COALESCE(sub_question_id, 0), respondent
LIMIT 10;
```

