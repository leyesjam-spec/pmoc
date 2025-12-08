# Understanding Features vs Samples in Machine Learning

## The Confusion
You asked: "Why does 82 features together with 500 synthetic couples and 22 database couples when training?"

This is a common confusion between **FEATURES** (columns) and **SAMPLES** (rows)!

**Note:** The feature count has been updated from 82 to 135 features based on recent improvements to the model.

## Simple Analogy: Excel Spreadsheet

Think of your training data like an Excel spreadsheet:

```
| Couple ID | male_age | female_age | age_gap | ... | Q1 | Q2 | Q3 | ... | Q59 | alignment | conflict | ... | risk_level |
|-----------|----------|------------|---------|-----|----|----|----|-----|-----|-----------|----------|-----|------------|
| 1         | 30       | 28         | 2       | ... | 4  | 3  | 2  | ... | 4   | 0.8       | 0.1      | ... | Low        |
| 2         | 35       | 32         | 3       | ... | 3  | 2  | 4  | ... | 3   | 0.5       | 0.4      | ... | Medium     |
| 3         | 25       | 27         | 2       | ... | 2  | 2  | 3  | ... | 2   | 0.3       | 0.6      | ... | High       |
| ...       | ...      | ...        | ...     | ... | ...| ...| ...| ... | ... | ...       | ...      | ... | ...        |
| 522       | 40       | 38         | 2       | ... | 4  | 4  | 4  | ... | 4   | 0.9       | 0.05     | ... | Low        |
```

- **82 FEATURES** = 82 COLUMNS (the characteristics of each couple)
- **522 SAMPLES** = 522 ROWS (the individual couples)

## Breaking Down the Features (Updated Structure)

### 1. Basic Demographic Features (11 features)
```
1.  male_age
2.  female_age
3.  age_gap (calculated: |male_age - female_age|)
4.  years_living_together (0 for non-Living In couples)
5.  education_level
6.  income_level
7.  education_income_diff (calculated: |education - income|)
8.  is_single (1 or 0)
9.  is_living_in (1 or 0)
10. is_separated_divorced (1 or 0)
11. employment_status (0=Unemployed, 1=Employed, 2=Self-employed)
```

**Note:** `years_living_together` is only meaningful when `civil_status` is "Living In", but we keep both features as they provide different information (relationship type vs. duration).

### 2. Questionnaire Responses (118 features - 59 male + 59 female)
Each answerable question has separate responses from both partners:
```
12-70.   Male responses (Q1_male, Q2_male, ..., Q59_male)
71-129.  Female responses (Q1_female, Q2_female, ..., Q59_female)
```

**Why separate?** This captures individual partner perspectives, allowing the model to detect differences in how each partner responds to the same questions.

### 3. Personalized Features (6 features)
```
130. alignment_score (overall agreement between partners)
131. conflict_ratio (frequency of disagreements)
132. category_1_alignment (alignment for category 1)
133. category_2_alignment (alignment for category 2)
134. category_3_alignment (alignment for category 3)
135. category_4_alignment (alignment for category 4)
```

**Total: 11 + 118 + 6 = 135 features**

**Changes from previous version:**
- ✅ Removed: `children` feature
- ✅ Removed: `male_avg_response`, `female_avg_response`
- ✅ Removed: `male_agree_ratio`, `male_disagree_ratio`, `female_agree_ratio`, `female_disagree_ratio`
- ✅ Added: `employment_status` feature
- ✅ Changed: Questionnaire responses from 59 combined to 118 separate (59 male + 59 female)

## What the Model Learns

The ML model learns patterns like:
- "Couples with large age gaps AND many disagreements → High risk"
- "Couples with high alignment AND low conflict → Low risk"
- "Question 15 (about finances) is a strong predictor of risk"

## The Training Process

```
INPUT (X): 522 couples × 135 features each
         ↓
    ML Model learns patterns
         ↓
OUTPUT (y): 522 risk levels (Low/Medium/High)
```

**After training:**
- The model can predict risk for NEW couples
- You give it 135 features for a new couple
- It outputs: "This couple is High risk with 85% confidence"

## Why 522 Couples?

- **22 real couples** from your database
- **500 synthetic couples** generated based on real patterns
- **Total: 522 training samples**

More samples = better model (up to a point). 522 is a good starting point, but you can add more real couples as they come in!

## Summary

- **135 FEATURES** = What information we know about each couple (columns)
- **522 SAMPLES** = How many couples we're training on (rows)
- The model learns: "Given these 135 features, what's the risk level?"

Think of it like teaching someone to recognize cats:
- **Features** = "has fur, has whiskers, has tail, has 4 legs..." (135 characteristics)
- **Samples** = "I'll show you 522 pictures of cats" (522 examples)

