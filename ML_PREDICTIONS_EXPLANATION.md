# What the Machine Learning Models Predict

## Overview

The system uses **TWO separate ML models** that work together to provide comprehensive relationship risk assessment and counseling recommendations.

---

## Model 1: Risk Level Prediction Model

### What It Predicts

**Output:** Relationship Risk Level
- **Low Risk** (Class 0)
- **Medium Risk** (Class 1)  
- **High Risk** (Class 2)

### Model Type
- **Algorithm:** Random Forest Classifier
- **Type:** Classification (3 classes)
- **Output Format:** Single class prediction + probability distribution

### Input Features (135 total)
1. **11 Demographic Features:**
   - male_age, female_age, age_gap, years_living_together
   - education_level, income_level, education_income_diff
   - is_single, is_living_in, is_separated_divorced, employment_encoded

2. **118 Response Features:**
   - 59 male partner responses (Q1_male, Q2_male, ..., Q59_male)
   - 59 female partner responses (Q1_female, Q2_female, ..., Q59_female)

3. **6 Personalized Features:**
   - alignment_score (overall agreement between partners)
   - conflict_ratio (frequency of disagreements)
   - category_1_alignment, category_2_alignment, category_3_alignment, category_4_alignment

### What It Learns

The model learns patterns like:
- "Couples with large age gaps (15+ years) AND many disagreements → High Risk"
- "Couples with high alignment (85%+) AND low conflict (<10%) → Low Risk"
- "Education-income mismatch (diff ≥3) combined with high conflict → Medium/High Risk"
- "Question 15 (about finances) disagreement is a strong predictor of High Risk"

### Output Details

```python
# Prediction
risk_prediction = risk_model.predict(features)  # Returns: 0, 1, or 2
risk_level = ['Low', 'Medium', 'High'][risk_prediction]

# Confidence (Probability)
risk_probs = risk_model.predict_proba(features)  # Returns: [0.85, 0.10, 0.05]
ml_confidence = max(risk_probs)  # Highest probability = confidence level
```

**Example:**
- Prediction: `2` (High Risk)
- Probabilities: `[0.15, 0.20, 0.65]` (15% Low, 20% Medium, 65% High)
- Confidence: `0.65` (65% confident it's High Risk)

---

## Model 2: Category Scores Prediction Model

### What It Predicts

**Output:** Counseling Priority Scores for each MEAI Category (4 scores)
- **Category 1:** Marriage And Relationship (score: 0.0 - 1.0)
- **Category 2:** Responsible Parenthood (score: 0.0 - 1.0)
- **Category 3:** Planning The Family (score: 0.0 - 1.0)
- **Category 4:** Maternal Neonatal Child Health And Nutrition (score: 0.0 - 1.0)

### Model Type
- **Algorithm:** Multi-Output Random Forest Regressor
- **Type:** Regression (4 continuous outputs)
- **Output Format:** Array of 4 scores [0.0 - 1.0]

### Input Features (Same 135 features as Risk Model)

The category model uses the **same 135 features** as the risk model to predict which specific MEAI categories need the most counseling attention.

### What It Learns

The model learns which categories are most problematic based on:
- **Response patterns:** Which questions/categories show the most disagreement
- **Demographic factors:** How age, education, income affect specific category needs
- **Personalized features:** How alignment/conflict in specific categories relates to counseling needs

**Example Patterns:**
- "Couples with financial disagreements → High score in Marriage And Relationship category"
- "Couples with children + low alignment in Responsible Parenthood → High score in Responsible Parenthood"
- "Young couples with low education → High score in Planning The Family category"

### Output Details

```python
# Prediction
category_scores = category_model.predict(features)  # Returns: [0.45, 0.72, 0.31, 0.58]
# Scores are clipped to [0.0, 1.0]

# Interpretation
for category, score in zip(MEAI_CATEGORIES, category_scores):
    if score > 0.6:      # 60-100%
        priority = 'High'
    elif score > 0.3:    # 30-60%
        priority = 'Moderate'
    else:                # 0-30%
        priority = 'Low'
```

**Example:**
- Category Scores: `[0.45, 0.72, 0.31, 0.58]`
- Interpretation:
  - Marriage And Relationship: 0.45 (Moderate Priority)
  - Responsible Parenthood: 0.72 (High Priority) ← **Focus here**
  - Planning The Family: 0.31 (Low Priority)
  - Maternal Neonatal Child Health: 0.58 (Moderate Priority)

---

## How They Work Together

### Step 1: Risk Assessment
```
Input: 135 features (demographics + responses + personalized)
  ↓
Risk Model predicts: "High Risk" (65% confidence)
```

### Step 2: Category Prioritization
```
Input: Same 135 features
  ↓
Category Model predicts: [0.45, 0.72, 0.31, 0.58]
  ↓
Interpretation: "Responsible Parenthood needs most attention (72%)"
```

### Step 3: Hybrid Decision
```
Risk Model: "High Risk"
Actual Calculation (from responses): "Low Risk" (8% disagreement)
  ↓
Hybrid Logic: Choose appropriate risk level
  ↓
Final Output: "High Risk" (ML model overrides due to demographic factors)
```

### Step 4: Recommendations
```
Risk Level: High Risk
Category Scores: [0.45, 0.72, 0.31, 0.58]
  ↓
Generate personalized recommendations:
  - "High priority: Address Responsible Parenthood (72% score)"
  - "Focus on relationship foundations (45% score)"
  - "Develop family planning strategy (31% score)"
```

---

## Complete Prediction Flow

```
┌─────────────────────────────────────────────────────────┐
│ INPUT: Couple Profile + MEAI Responses                  │
│ - Demographics (age, education, income, etc.)           │
│ - 59 male responses + 59 female responses               │
│ - Personalized features (alignment, conflict)           │
└─────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │ Feature Engineering            │
        │ (135 features total)          │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │ MODEL 1: Risk Prediction      │
        │ Random Forest Classifier      │
        │ Output: Low/Medium/High Risk  │
        │ + Confidence (probabilities)  │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │ MODEL 2: Category Scores      │
        │ Multi-Output Regressor        │
        │ Output: 4 scores [0.0-1.0]    │
        │ (one per MEAI category)       │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │ Hybrid Decision Logic         │
        │ Combines ML prediction with   │
        │ actual response calculation    │
        └───────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │ OUTPUT:                      │
        │ - Final Risk Level            │
        │ - Category Priorities         │
        │ - Recommendations             │
        │ - Risk Reasoning              │
        │ - Counseling Reasoning        │
        └───────────────────────────────┘
```

---

## Training Data

### Risk Model Training
- **Input (X):** 522 couples × 135 features
- **Output (y):** 522 risk levels (Low/Medium/High)
- **Training Method:** Grid Search with Cross-Validation
- **Evaluation:** Accuracy score (typically 85-95% on training data)

### Category Model Training
- **Input (X):** 522 couples × 135 features
- **Output (y):** 522 couples × 4 category scores
- **Training Method:** Grid Search with Cross-Validation
- **Evaluation:** R² score (coefficient of determination)

### Data Sources
- **22 real couples** from database (labeled by disagreement ratio)
- **500 synthetic couples** (generated with balanced risk distribution)
- **Total: 522 training samples**

---

## Key Differences: Risk Model vs Category Model

| Aspect | Risk Model | Category Model |
|--------|-----------|----------------|
| **Type** | Classification | Regression |
| **Output** | 1 class (Low/Medium/High) | 4 continuous scores (0.0-1.0) |
| **Purpose** | Overall relationship risk | Specific counseling priorities |
| **Algorithm** | Random Forest Classifier | Multi-Output Random Forest Regressor |
| **Confidence** | Probability distribution | R² score (model fit) |
| **Use Case** | "Is this couple at risk?" | "Which areas need counseling?" |

---

## Example: Complete Prediction

### Input Couple:
- **Demographics:** Male 45, Female 30, Age gap 15, Education Level 4, Income Level 1
- **Responses:** 90% alignment, 8% conflict
- **Features:** 135 total features

### Model 1 Prediction (Risk):
```
Risk Model Output:
  - Prediction: High Risk (class 2)
  - Probabilities: [0.15, 0.20, 0.65]
  - Confidence: 65%
```

### Model 2 Prediction (Categories):
```
Category Model Output:
  - Marriage And Relationship: 0.45 (Moderate)
  - Responsible Parenthood: 0.38 (Low)
  - Planning The Family: 0.52 (Moderate)
  - Maternal Neonatal Child Health: 0.41 (Moderate)
```

### Hybrid Decision:
```
Actual Calculation: Low Risk (8% disagreement)
ML Prediction: High Risk (65% confidence)
  ↓
Decision: High Risk (ML overrides due to demographic factors)
Reason: Large age gap (15 years) + education-income mismatch (diff=3)
```

### Final Output:
```json
{
  "risk_level": "High",
  "ml_confidence": 0.65,
  "category_scores": [0.45, 0.38, 0.52, 0.41],
  "focus_categories": [
    {"name": "Planning The Family", "score": 0.52, "priority": "Moderate"},
    {"name": "Marriage And Relationship", "score": 0.45, "priority": "Moderate"},
    {"name": "Maternal Neonatal Child Health", "score": 0.41, "priority": "Moderate"},
    {"name": "Responsible Parenthood", "score": 0.38, "priority": "Low"}
  ],
  "risk_reasoning": "🔴 HIGH RISK CLASSIFICATION: ...",
  "recommendations": [
    "High priority: Address demographic factors (age gap, education-income mismatch)...",
    "Focus on Planning The Family category (52% score)..."
  ]
}
```

---

## Summary

### What the ML Models Predict:

1. **Risk Level Model:**
   - ✅ Predicts: **Low, Medium, or High Risk**
   - ✅ Provides: **Confidence level** (probability)
   - ✅ Uses: **All 135 features** (demographics + responses + personalized)

2. **Category Scores Model:**
   - ✅ Predicts: **4 counseling priority scores** (one per MEAI category)
   - ✅ Provides: **Specific areas needing attention**
   - ✅ Uses: **Same 135 features** as risk model

### How They're Used:

- **Risk Level:** Overall relationship health assessment
- **Category Scores:** Specific counseling focus areas
- **Together:** Comprehensive risk assessment + actionable recommendations

### Key Insight:

The models don't just predict risk—they **learn patterns** from training data to identify:
- Which demographic factors correlate with risk
- Which response patterns indicate problems
- Which MEAI categories need the most attention
- How to combine all factors for accurate predictions

