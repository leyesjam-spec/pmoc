# How Counseling Recommendation Priority is Calculated

## Overview

Yes, the **Counseling Recommendation Priority** (the percentage shown, e.g., "93.4% Priority") **follows and increases** with the Risk Level, but it's also adjusted by category scores.

---

## Calculation Formula

### Step 1: Base Priority from Risk Level

The system starts with a base priority percentage based on the Risk Level:

```javascript
if (riskLevel === 'High') {
    priorityPercentage = 85;  // High risk = 85% base priority
} else if (riskLevel === 'Medium') {
    priorityPercentage = 60;   // Medium risk = 60% base priority
} else {
    priorityPercentage = 25;   // Low risk = 25% base priority
}
```

**Base Priority by Risk Level:**
- **High Risk** → **85%** base priority
- **Medium Risk** → **60%** base priority
- **Low Risk** → **25%** base priority

### Step 2: Adjustment Based on Category Scores

The base priority is then **adjusted upward** based on the highest MEAI category score:

```javascript
const maxCategoryScore = Math.max(...focusCategories.map(cat => cat.score));
const categoryAdjustment = maxCategoryScore * 20;  // Scale to 0-20%
priorityPercentage = Math.min(95, priorityPercentage + categoryAdjustment);
```

**Category Adjustment:**
- Takes the **highest category score** (0.0 to 1.0)
- Multiplies by 20 to get adjustment (0% to 20%)
- Adds to base priority
- **Capped at 95% maximum**

---

## Examples

### Example 1: High Risk + High Category Scores
```
Risk Level: High Risk
Base Priority: 85%

Highest Category Score: 0.72 (72%)
Category Adjustment: 0.72 × 20 = 14.4%

Final Priority: 85% + 14.4% = 99.4% → Capped at 95%
Result: 95% Priority
```

### Example 2: High Risk + Moderate Category Scores
```
Risk Level: High Risk
Base Priority: 85%

Highest Category Score: 0.42 (42%)
Category Adjustment: 0.42 × 20 = 8.4%

Final Priority: 85% + 8.4% = 93.4%
Result: 93.4% Priority
```

### Example 3: Medium Risk + High Category Scores
```
Risk Level: Medium Risk
Base Priority: 60%

Highest Category Score: 0.65 (65%)
Category Adjustment: 0.65 × 20 = 13%

Final Priority: 60% + 13% = 73%
Result: 73% Priority
```

### Example 4: Low Risk + Low Category Scores
```
Risk Level: Low Risk
Base Priority: 25%

Highest Category Score: 0.15 (15%)
Category Adjustment: 0.15 × 20 = 3%

Final Priority: 25% + 3% = 28%
Result: 28% Priority
```

---

## Priority Ranges

### High Priority (>70%)
- **Base:** High Risk (85%) OR Medium Risk (60%) + high category scores
- **Recommendation:** "Intensive counseling program recommended"
- **Color:** Red (danger)

### Moderate Priority (40-70%)
- **Base:** Medium Risk (60%) OR Low Risk (25%) + moderate category scores
- **Recommendation:** "Structured counseling sessions recommended"
- **Color:** Yellow (warning)

### Low Priority (<40%)
- **Base:** Low Risk (25%) + low category scores
- **Recommendation:** "Preventive counseling recommended"
- **Color:** Green (success)

---

## Relationship Summary

| Risk Level | Base Priority | Typical Range | With High Categories |
|------------|---------------|---------------|---------------------|
| **High Risk** | 85% | 85-95% | 90-95% |
| **Medium Risk** | 60% | 60-80% | 70-85% |
| **Low Risk** | 25% | 25-45% | 30-50% |

---

## Key Points

### ✅ Yes, Priority Follows Risk Level:
1. **High Risk** always starts at **85%** (high priority)
2. **Medium Risk** always starts at **60%** (moderate priority)
3. **Low Risk** always starts at **25%** (low priority)

### ✅ But It's Also Adjusted:
- **Category scores can increase** the priority (up to +20%)
- **Maximum cap** is 95% (prevents unrealistic priorities)
- **Final priority** reflects both risk level AND specific category needs

### ✅ Why This Makes Sense:
- **High Risk couples** need intensive counseling (high priority)
- **But even High Risk couples** may have different category needs
- **Category scores** fine-tune the priority based on specific areas needing attention

---

## Visual Example

```
High Risk (85% base)
    +
Highest Category Score: 0.42 (42%)
    × 20 = 8.4% adjustment
    =
93.4% Priority
    ↓
"Intensive counseling program recommended"
```

---

## Summary

**Yes, the counseling recommendation priority DOES follow and increase with the risk level:**

- **High Risk** → **85% base** (can go up to 95%)
- **Medium Risk** → **60% base** (can go up to 80%)
- **Low Risk** → **25% base** (can go up to 45%)

**The priority is calculated as:**
```
Priority = Base Priority (from Risk Level) + Category Adjustment (0-20%)
```

**This ensures:**
- High Risk couples get high priority counseling recommendations
- But specific category needs can fine-tune the exact priority level
- The system provides a comprehensive, nuanced counseling recommendation

