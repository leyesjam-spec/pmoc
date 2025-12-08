# Education and Income in the ML Risk Assessment System

## Overview

The `education` and `monthly_income` fields from the couple profile are converted to numeric levels (`education_level` and `income_level`) and used as **demographic features** in the Machine Learning model to assess relationship risk.

---

## 1. Education Level Mapping

### How Education is Converted

The system maps text-based education values to numeric levels (0-4):

| Education Text | Numeric Level | Description |
|---------------|---------------|-------------|
| No Education, Pre School, Elementary Level, Elementary Graduate | **0** | Basic/No Education |
| High School Level, High School Graduate, Junior HS Level, Junior HS Graduate, Senior HS Level, Senior HS Graduate, ALS | **1** | High School Level |
| College Level, Vocational/Technical | **2** | Some College/Vocational |
| College Graduate | **3** | College Degree |
| Post Graduate | **4** | Advanced Degree |

### Default Value
- If education is missing or invalid: **Level 2** (College Level)

---

## 2. Income Level Mapping

### How Monthly Income is Converted

The system maps monthly income ranges to numeric levels (0-3):

| Monthly Income Range | Numeric Level | Description |
|---------------------|---------------|-------------|
| ₱5,000 below, ₱5,999-9,999 | **0** | Low Income |
| ₱10,000-14,999, ₱15,000-19,999 | **1** | Lower-Middle Income |
| ₱20,000-24,999 | **2** | Middle Income |
| ₱25,000 above | **3** | Upper-Middle/High Income |

### Default Value
- If income is missing or invalid: **Level 1** (₱10,000-14,999)

---

## 3. How They Are Used in the ML Model

### A. As Direct Features

Both `education_level` and `income_level` are included as **separate features** in the ML model:

```
Feature #5: education_level (0-4)
Feature #6: income_level (0-3)
```

**Total: 2 direct features**

### B. As Compatibility Indicator

The system calculates **Education-Income Compatibility**:

```python
education_income_diff = abs(education_level - income_level)
```

This measures the **mismatch** between education and income levels.

**Example:**
- Education Level: 4 (Post Graduate)
- Income Level: 1 (₱10,000-14,999)
- **Difference: 3** → Large mismatch (highly educated but low income)

**Interpretation:**
- **Difference = 0-1**: Compatible (education matches income expectations)
- **Difference = 2**: Moderate mismatch (may indicate underemployment or career transition)
- **Difference ≥ 3**: Significant mismatch (may indicate:
  - Underemployment (highly educated but low income)
  - Career change or transition
  - Economic challenges
  - Potential source of relationship stress)

This difference is included as:
```
Feature #7: education_income_diff (0-4)
```

**Total: 1 compatibility feature**

---

## 4. Why Education and Income Matter

### A. Socioeconomic Compatibility

Research shows that **socioeconomic differences** can affect relationship stability:

1. **Education Mismatch:**
   - Different education levels may indicate different:
     - Life goals and aspirations
     - Communication styles
     - Problem-solving approaches
     - Social circles and values

2. **Income Mismatch:**
   - Income differences can lead to:
     - Financial stress and disagreements
     - Power imbalances
     - Different spending habits and priorities
     - Stress about financial security

3. **Education-Income Mismatch:**
   - When education doesn't match income, it may indicate:
     - Underemployment (frustration, career dissatisfaction)
     - Career transitions (uncertainty, stress)
     - Economic challenges (financial stress)
     - Unrealized potential (resentment, dissatisfaction)

### B. How the ML Model Uses Them

The ML model learns patterns from training data:

1. **Direct Impact:**
   - The model learns if certain education/income combinations correlate with higher risk
   - Example: Large education-income mismatch (difference ≥ 3) might correlate with Medium/High risk

2. **Interaction with Other Factors:**
   - Education/income interact with:
     - Age gap (older couples with education mismatch may have different challenges)
     - Civil status (divorced couples with income mismatch may face additional stress)
     - Response patterns (demographics may explain why responses show certain patterns)

3. **Pattern Recognition:**
   - The model identifies patterns like:
     - "High education + Low income + High age gap = Higher risk"
     - "Compatible education-income + Low conflict = Lower risk"

---

## 5. In Risk Reasoning Output

When generating risk reasoning, the system explains education and income factors:

### Example Output:

```
👥 DEMOGRAPHIC FACTORS:
   • ⚠️ Education-Income Mismatch: 3 level difference
     → Education level: 4, Income level: 1
     → Significant differences may indicate compatibility challenges
     → This demographic factor may contribute to relationship stress
```

### Interpretation:

- **⚠️ Warning (Difference > 2):**
  - Significant mismatch detected
  - May contribute to relationship challenges
  - Explained in risk reasoning

- **Moderate (Difference = 2):**
  - Some mismatch, but manageable
  - Noted but not flagged as major concern

- **✅ Compatible (Difference ≤ 1):**
  - Education and income are aligned
  - Positive indicator for relationship stability

---

## 6. Feature Engineering Details

### Complete Feature Set (11 Demographic Features):

1. `male_age` - Male partner's age
2. `female_age` - Female partner's age
3. `age_gap` - Absolute difference in ages
4. `years_living_together` - Years cohabiting (0 if not Living In)
5. **`education_level`** - Education level (0-4) ← **Education**
6. **`income_level`** - Income level (0-3) ← **Income**
7. **`education_income_diff`** - Compatibility indicator (0-4) ← **Compatibility**
8. `is_single` - One-hot: Single status
9. `is_living_in` - One-hot: Living In status
10. `is_separated_divorced` - One-hot: Separated/Divorced/Widowed
11. `employment_encoded` - Employment status (0=Unemployed, 1=Employed, 2=Self-employed)

**Total: 11 demographic features** (including education and income)

---

## 7. Example Scenarios

### Scenario 1: Compatible Education-Income
- **Education:** College Graduate (Level 3)
- **Income:** ₱25,000 above (Level 3)
- **Difference:** 0
- **Impact:** ✅ Positive indicator, suggests socioeconomic compatibility

### Scenario 2: Moderate Mismatch
- **Education:** Post Graduate (Level 4)
- **Income:** ₱20,000-24,999 (Level 2)
- **Difference:** 2
- **Impact:** ⚠️ Moderate concern, may indicate career transition or underemployment

### Scenario 3: Significant Mismatch
- **Education:** Post Graduate (Level 4)
- **Income:** ₱10,000-14,999 (Level 1)
- **Difference:** 3
- **Impact:** ⚠️⚠️ High concern, significant mismatch may contribute to relationship stress

### Scenario 4: Reverse Mismatch
- **Education:** High School Graduate (Level 1)
- **Income:** ₱25,000 above (Level 3)
- **Difference:** 2
- **Impact:** ⚠️ Moderate concern, may indicate different career paths or opportunities

---

## 8. Important Notes

### ⚠️ Limitations

1. **Not Deterministic:**
   - Education/income alone don't determine risk
   - They are **one factor** among many (responses, alignment, conflict, etc.)

2. **Context Matters:**
   - A mismatch might be acceptable if:
     - Partners are in career transition
     - One partner is supporting the other's education
     - Partners have discussed and accepted the difference

3. **Response-Based Factors Are Primary:**
   - The system prioritizes **actual relationship responses** over demographics
   - If responses show high alignment and low conflict, demographics have less impact

### ✅ Best Practices

1. **Complete Data:**
   - Always provide accurate education and income data
   - Missing data defaults to middle values, which may not reflect reality

2. **Interpretation:**
   - Use education/income as **context**, not the sole determinant
   - Consider them alongside response-based factors (alignment, conflict)

3. **Counseling Focus:**
   - If education-income mismatch is flagged, counselors can:
     - Discuss financial goals and expectations
     - Address career aspirations and satisfaction
     - Explore how income differences affect the relationship

---

## 9. Summary

| Aspect | Details |
|--------|---------|
| **Education Levels** | 0-4 (No Education to Post Graduate) |
| **Income Levels** | 0-3 (₱5,000 below to ₱25,000 above) |
| **Compatibility Metric** | `abs(education_level - income_level)` |
| **ML Features** | 3 features total (education, income, difference) |
| **Purpose** | Assess socioeconomic compatibility and potential stress factors |
| **Impact** | One factor among many; responses are primary indicators |
| **Reasoning Display** | Explained in risk reasoning output with warnings for mismatches |

---

## 10. Code References

- **Mapping:** `ml_model/service.py` lines 1060-1070
- **Feature Engineering:** `ml_model/service.py` lines 1387-1413, 2283-2328
- **Risk Reasoning:** `ml_model/service.py` lines 2794-2808
- **PHP Mapping:** `ml_model/ml_api.php` lines 988-1013

