# Understanding the 59 Questionnaire Responses

## What Are the 59 Responses?

The **59** represents the total number of **answerable questions** in the MEAI (Marriage Expectations and Inventory) questionnaire. This is the count of all questions that couples must answer, whether they are:
- **Standalone questions** (main questions without sub-questions)
- **Sub-questions** (questions that are part of a larger question)

## How Are They Structured?

The MEAI questionnaire is organized into **4 categories**:

1. **Marriage And Relationship**
2. **Responsible Parenthood**
3. **Planning The Family**
4. **Maternal Neonatal Child Health And Nutrition**

Each category contains multiple questions, and some questions have sub-questions. The **59** is the total count of all answerable items across all categories.

### Example Structure:

```
Category 1: Marriage And Relationship
├── Question 1: "How do you feel about..."
│   ├── Sub-question 1.1: "In terms of communication..."
│   ├── Sub-question 1.2: "In terms of trust..."
│   └── Sub-question 1.3: "In terms of respect..."
├── Question 2: "What are your expectations about..."
│   └── (Standalone - no sub-questions, counts as 1)
└── Question 3: "How important is..."
    ├── Sub-question 3.1: "Financial stability..."
    └── Sub-question 3.2: "Emotional support..."

... and so on across all 4 categories
```

**Total: 59 answerable items** (counting each sub-question and standalone question)

## How Are They Used in the ML Model?

### 1. **Data Collection**
- Each partner (male and female) answers all 59 questions independently
- Responses are on a scale: **2 = Disagree**, **3 = Neutral**, **4 = Agree**

### 2. **Feature Creation**
The 59 responses become **118 features** in the ML model:
- **59 features** from male partner's responses
- **59 features** from female partner's responses

**Why separate?** This allows the model to:
- Detect differences in how partners view the same issues
- Identify areas of disagreement or alignment
- Predict relationship risk based on partner perspectives

### 3. **What the Model Learns**

The ML model uses these 59 responses to identify patterns like:

- **Question 15** (about finances) might be a strong predictor of risk
- **Questions 20-25** (about communication) might indicate relationship stability
- **Large disagreements** on specific questions might signal high risk
- **High agreement** across categories might indicate low risk

### 4. **Personalized Features**

The 59 responses are also used to calculate **personalized features**:

- **Alignment Score**: How well partners agree across all 59 questions
- **Conflict Ratio**: How often partners disagree significantly
- **Category Alignments**: Agreement level for each of the 4 categories

## Example: How 59 Becomes 118 Features

```
Couple's Responses:

Male Partner:
Q1: 4 (Agree)
Q2: 3 (Neutral)
Q3: 2 (Disagree)
...
Q59: 4 (Agree)

Female Partner:
Q1: 4 (Agree)
Q2: 2 (Disagree)  ← Disagreement detected!
Q3: 3 (Neutral)
...
Q59: 4 (Agree)

ML Model Features:
Features 12-70:   [4, 3, 2, ..., 4]  ← Male responses (59 features)
Features 71-129:  [4, 2, 3, ..., 4]  ← Female responses (59 features)
```

The model sees that on Q2, the male partner is neutral (3) but the female partner disagrees (2). This **disagreement** is a signal that the model can use to predict risk.

## Why 59 Specifically?

The number **59** is **dynamically calculated** from your database:

1. The system queries the `question_assessment` and `sub_question_assessment` tables
2. It counts all answerable items (questions with sub-questions count the sub-questions, standalone questions count as 1)
3. The total across all 4 categories = **59**

If you add or remove questions in the database, this number will automatically adjust!

## Summary

- **59** = Total answerable questions in the MEAI questionnaire
- **59 × 2** = **118 features** (male + female responses)
- Used to predict relationship risk and identify areas of concern
- Dynamically calculated from your database structure
- Each response helps the model understand the couple's relationship dynamics

The model learns: *"Given how both partners answered these 59 questions, what's their relationship risk level?"*

