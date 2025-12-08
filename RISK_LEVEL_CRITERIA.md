# Risk Level Classification Criteria

## Overview
Risk levels are determined based on the **Weighted Disagreement Ratio** calculated from couple responses to the MEAI questionnaire.

## How Risk is Calculated

### Step 1: Count Disagreements
For each question, we count:
- **Question Disagreements**: When either partner disagrees with the question (response = 2)
- **Partner Disagreements**: When partners disagree with each other
  - Significant disagreement: Partners differ by 2+ points (e.g., one says 2, other says 4)
  - Minor disagreement: Partners differ by 1 point (e.g., one says 3, other says 4)
- **Neutral Responses**: When either partner is neutral (response = 3)

### Step 2: Calculate Weighted Disagreement Count
```
Total Disagreement = max(Question Disagreements, Partner Disagreements) + (Neutral Count × 0.3)
```

**Why use the maximum?** To avoid double-counting. We count either:
- When partners disagree with the questions, OR
- When partners disagree with each other
(whichever is higher)

**Why weight neutrals?** Neutral responses (30% weight) may indicate uncertainty or unresolved issues.

### Step 3: Calculate Disagreement Ratio
```
Disagreement Ratio = Total Weighted Disagreement / Total Questions
```

### Step 4: Classify Risk Level

| Risk Level | Disagreement Ratio | Meaning |
|------------|-------------------|---------|
| **Low Risk** | ≤ 15% (0.15) | Healthy relationship with minimal disagreements |
| **Medium Risk** | 15% - 35% (0.15 - 0.35) | Some concerns requiring proactive counseling |
| **High Risk** | > 35% (0.35) | Significant challenges requiring immediate attention |

## Current Thresholds

```python
if disagree_ratio > 0.35:      # 35%+ = High Risk
    risk_level = 'High'
elif disagree_ratio > 0.15:   # 15-35% = Medium Risk
    risk_level = 'Medium'
else:                          # <15% = Low Risk
    risk_level = 'Low'
```

## Examples

### Example 1: Low Risk Couple
- **Total Questions**: 59
- **Question Disagreements**: 3 (partners disagree with 3 questions)
- **Partner Disagreements**: 2 (partners disagree with each other on 2 questions)
- **Neutral Responses**: 5
- **Calculation**:
  - Max(3, 2) = 3
  - Weighted neutrals = 5 × 0.3 = 1.5
  - Total = 3 + 1.5 = 4.5
  - **Disagreement Ratio** = 4.5 / 59 = **7.6%**
- **Result**: **Low Risk** (7.6% < 15%)

### Example 2: Medium Risk Couple
- **Total Questions**: 59
- **Question Disagreements**: 8
- **Partner Disagreements**: 10
- **Neutral Responses**: 12
- **Calculation**:
  - Max(8, 10) = 10
  - Weighted neutrals = 12 × 0.3 = 3.6
  - Total = 10 + 3.6 = 13.6
  - **Disagreement Ratio** = 13.6 / 59 = **23.1%**
- **Result**: **Medium Risk** (15% < 23.1% < 35%)

### Example 3: High Risk Couple
- **Total Questions**: 59
- **Question Disagreements**: 15
- **Partner Disagreements**: 18
- **Neutral Responses**: 10
- **Calculation**:
  - Max(15, 18) = 18
  - Weighted neutrals = 10 × 0.3 = 3.0
  - Total = 18 + 3.0 = 21.0
  - **Disagreement Ratio** = 21.0 / 59 = **35.6%**
- **Result**: **High Risk** (35.6% > 35%)

## What Each Risk Level Means

### Low Risk (≤ 15% Disagreement)
- **Characteristics**:
  - High alignment between partners (typically > 70%)
  - Low conflict ratio (typically < 15%)
  - Partners mostly agree on relationship values and expectations
- **Recommendation**: Preventive counseling to maintain relationship health
- **Counseling Focus**: Relationship building, communication skills, maintaining strengths

### Medium Risk (15-35% Disagreement)
- **Characteristics**:
  - Moderate alignment (typically 50-70%)
  - Moderate conflict (typically 15-35%)
  - Some areas of concern requiring attention
- **Recommendation**: Proactive, focused counseling
- **Counseling Focus**: Address specific areas of concern, conflict resolution, communication improvement

### High Risk (> 35% Disagreement)
- **Characteristics**:
  - Low alignment (typically < 50%)
  - High conflict (typically > 35%)
  - Significant disagreements across multiple areas
- **Recommendation**: Intensive, structured counseling
- **Counseling Focus**: Core relationship issues, intensive conflict resolution, fundamental compatibility

## Additional Factors Considered

The ML model also considers:
- **Demographics**: Age, age gap, education, income, civil status
- **Alignment Score**: How similar partners' responses are (0-1 scale)
- **Conflict Ratio**: Frequency of partner disagreements (0-1 scale)
- **Category Scores**: Disagreement levels in specific MEAI categories

## Two Risk Calculation Methods

### Method 1: Actual Risk (Response-Based Only)
- **Uses**: Only questionnaire responses
- **Calculates**: Weighted disagreement ratio from responses
- **Thresholds**: Low ≤15%, Medium 15-35%, High >35%
- **Purpose**: Direct measure of relationship disagreement

### Method 2: ML Model Prediction (Response + Demographics)
- **Uses**: 135 features total:
  - **11 Demographic Features**: Age, age gap, education, income, civil status, employment
  - **118 Response Features**: 59 male responses + 59 female responses
  - **6 Personalized Features**: Alignment score, conflict ratio, 4 category alignments
- **Purpose**: Learns patterns from training data (real + synthetic couples)
- **Can consider**: Demographics that might affect risk (e.g., large age gaps, education mismatches)

## Hybrid Decision Logic

The system uses a hybrid approach combining both methods:
1. **Calculate actual risk** from disagreement ratio (responses only)
2. **Get ML model prediction** (responses + demographics + personalized features)
3. **Choose the appropriate risk level**:
   - If actual = Low AND alignment > 70% AND conflict < 15% → Trust actual (Low)
     - *Reason: Responses show healthy relationship, ignore demographic-based ML prediction*
   - If actual = High → Trust actual (High)
     - *Reason: High disagreement is clear indicator, more reliable than ML*
   - If ML suggests higher risk than actual → Use ML
     - *Reason: ML might catch patterns demographics reveal*
   - Otherwise → Use actual calculation
     - *Reason: Response-based calculation is more reliable*

**Why Hybrid?**
- **Actual calculation** (responses only) is transparent and directly measures disagreement
- **ML model** (responses + demographics) can learn complex patterns but might be biased
- **Combining both** gives the best of both worlds: transparency + pattern recognition

## Response Values

- **2 = Disagree**: Partner disagrees with the question
- **3 = Neutral**: Partner is neutral/unsure
- **4 = Agree**: Partner agrees with the question

## Notes

- Thresholds were adjusted from previous versions to be more sensitive:
  - Previous: High >50%, Medium >20%, Low <20%
  - Current: High >35%, Medium >15%, Low <15%
- This catches more couples with moderate disagreement as Medium/High risk
- The system prioritizes catching potential issues early rather than missing them

