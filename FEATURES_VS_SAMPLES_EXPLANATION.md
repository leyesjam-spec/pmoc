# Understanding Features vs Samples in Machine Learning

## The Confusion
You asked: "Why does 82 features together with 500 synthetic couples and 22 database couples when training?"

This is a common confusion between **FEATURES** (columns) and **SAMPLES** (rows)!

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

## Breaking Down the 82 Features

### 1. Basic Demographic Features (11 features)
```
1.  male_age
2.  female_age
3.  age_gap (calculated: |male_age - female_age|)
4.  years_living_together
5.  children
6.  education_level
7.  income_level
8.  education_income_diff (calculated: |education - income|)
9.  is_single (1 or 0)
10. is_living_in (1 or 0)
11. is_separated_divorced (1 or 0)
```

### 2. Questionnaire Responses (~59 features)
Each answerable question (main question or sub-question) becomes one feature:
```
12. Q1_response (2=disagree, 3=neutral, 4=agree)
13. Q2_response
14. Q3_response
...
70. Q59_response (or however many questions you have)
```

### 3. Personalized Features (12 features)
```
71. alignment_score
72. conflict_ratio
73. male_avg_response
74. female_avg_response
75. category_1_alignment
76. category_2_alignment
77. category_3_alignment
78. category_4_alignment
79. male_agree_ratio
80. male_disagree_ratio
81. female_agree_ratio
82. female_disagree_ratio
```

**Total: 11 + 59 + 12 = 82 features**

## What the Model Learns

The ML model learns patterns like:
- "Couples with large age gaps AND many disagreements → High risk"
- "Couples with high alignment AND low conflict → Low risk"
- "Question 15 (about finances) is a strong predictor of risk"

## The Training Process

```
INPUT (X): 522 couples × 82 features each
         ↓
    ML Model learns patterns
         ↓
OUTPUT (y): 522 risk levels (Low/Medium/High)
```

**After training:**
- The model can predict risk for NEW couples
- You give it 82 features for a new couple
- It outputs: "This couple is High risk with 85% confidence"

## Why 522 Couples?

- **22 real couples** from your database
- **500 synthetic couples** generated based on real patterns
- **Total: 522 training samples**

More samples = better model (up to a point). 522 is a good starting point, but you can add more real couples as they come in!

## Summary

- **82 FEATURES** = What information we know about each couple (columns)
- **522 SAMPLES** = How many couples we're training on (rows)
- The model learns: "Given these 82 features, what's the risk level?"

Think of it like teaching someone to recognize cats:
- **Features** = "has fur, has whiskers, has tail, has 4 legs..." (82 characteristics)
- **Samples** = "I'll show you 522 pictures of cats" (522 examples)

