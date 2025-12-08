# What Does the Risk Level Represent?

## Overview

The **Risk Level** (Low/Medium/High) shown in the Relationship Assessment is a **HYBRID assessment** that combines:

1. **Response-Based Analysis** (Questionnaire responses)
2. **Demographic Analysis** (Age, education, income, civil status, etc.)
3. **ML Pattern Recognition** (Learned patterns from training data)

---

## How It's Calculated

### Step 1: Response-Based Risk Calculation

**What it measures:** Direct disagreement between partners based on their MEAI questionnaire responses.

**How it works:**
- Counts when partners disagree with questions OR disagree with each other
- Calculates weighted disagreement ratio
- Classifies based on thresholds:
  - **High Risk:** >35% disagreement
  - **Medium Risk:** 20-35% disagreement  
  - **Low Risk:** ≤20% disagreement

**Example:**
- 25% disagreement ratio → **Medium Risk** (response-based)

### Step 2: ML Model Prediction

**What it measures:** Risk prediction using machine learning that considers ALL factors.

**Input Features (135 total):**

#### 1. Demographic Features (11 features)
- Age, age gap, education, income, civil status, employment
- These describe **who the couple is** (demographics)

#### 2. Response Features (118 features - 59 male + 59 female)

**Important:** These are **SEPARATE features**, not a comparison!

**What they represent:**
- **59 Male Responses:** The male partner's individual answers to all 59 MEAI questions
  - Each question gets one value: 2 (Disagree), 3 (Neutral), or 4 (Agree)
  - Example: `[4, 4, 3, 2, 4, 3, ...]` (59 values total)
  
- **59 Female Responses:** The female partner's individual answers to the same 59 MEAI questions
  - Each question gets one value: 2 (Disagree), 3 (Neutral), or 4 (Agree)
  - Example: `[4, 3, 4, 2, 4, 4, ...]` (59 values total)

**Why separate (not combined)?**

The purpose is to allow the ML model to learn **individual response patterns** and **how they interact**, not just compare them directly.

**What the ML model can learn from separate features:**

1. **Individual Response Styles:**
   - "When male partner has many 2s (disagrees frequently) → Higher risk"
   - "When female partner has many 3s (neutral frequently) → May indicate communication avoidance"
   - "When male partner has mostly 4s (very positive) → May indicate optimism or denial"

2. **Pattern Interactions:**
   - "Male partner: Many 2s (disagrees) **AND** Female partner: Many 2s (disagrees) → High Risk"
   - "Male partner: Mostly 4s (positive) **BUT** Female partner: Many 2s (negative) → Asymmetry indicates risk"
   - "Both partners: Similar patterns (both mostly 4s or both mostly 2s) → Lower risk"

3. **Question-Specific Patterns:**
   - "Question 15 (finances): Male=2, Female=4 → Financial disagreement pattern"
   - "Question 20 (communication): Male=3, Female=3 → Both neutral, may indicate avoidance"
   - "Question 30 (intimacy): Male=4, Female=2 → Intimacy mismatch pattern"

4. **Response Distribution Patterns:**
   - "Male: 40% are 2s, 30% are 3s, 30% are 4s → Mixed/uncertain response style"
   - "Female: 10% are 2s, 20% are 3s, 70% are 4s → Generally positive"
   - "Both: Similar distributions → Better alignment"

**How disagreement is calculated (separate process):**

The **disagreement/alignment** is calculated separately as personalized features:
- The system compares male_responses[i] vs female_responses[i] for each question
- This comparison creates: `alignment_score`, `conflict_ratio`, and `category_alignments`
- These become 6 additional features that summarize the relationship dynamics

#### 3. Personalized Features (6 features)
- **alignment_score:** Overall agreement between partners (0-1 scale)
- **conflict_ratio:** Frequency of disagreements (0-1 scale)
- **category_alignments:** Alignment for each of the 4 MEAI categories (4 values)

**What the ML model learns:**
- Patterns like "Large age gap + male has many disagreements + female has many disagreements → High Risk"
- "High education but low income + both partners disagree on financial questions → Medium/High Risk"
- "Both partners have high agreement scores + low conflict ratio → Low Risk"
- "Male partner is very positive but female partner is neutral/negative → Potential communication issues"

**Visual Example:**

```
MEAI Question 1: "I feel comfortable discussing finances with my partner"
  Male Response: 4 (Agree)
  Female Response: 2 (Disagree)
  → This becomes: male_responses[0] = 4, female_responses[0] = 2

MEAI Question 2: "We agree on family planning goals"
  Male Response: 4 (Agree)
  Female Response: 4 (Agree)
  → This becomes: male_responses[1] = 4, female_responses[1] = 4

... (continues for all 59 questions)

Final Feature Array:
[11 demographic features] + 
[male_responses: 59 values] + 
[female_responses: 59 values] + 
[6 personalized features] = 135 total features
```

**Key Point: Two Different Uses of the Responses**

The 59 male + 59 female responses are used in **TWO different ways**:

#### A. As Separate Features (118 features) - Individual Pattern Learning

**Purpose:** Learn patterns from each partner's **individual response style** and how they interact.

**What the model learns:**
- **Individual patterns:**
  - "Male partner's response style: Many 2s (disagrees) → Risk indicator"
  - "Female partner's response style: Many 3s (neutral) → Communication issue indicator"
  
- **Interaction patterns:**
  - "Male has pattern X **AND** Female has pattern Y → Risk level Z"
  - "Both have similar patterns → Lower risk"
  - "Asymmetric patterns (one positive, one negative) → Higher risk"

- **Question-specific patterns:**
  - "Question 15: Male=2, Female=4 → Financial disagreement"
  - "Question 20: Male=3, Female=3 → Both neutral, avoidance pattern"
  - The model learns which specific questions are most predictive

**Why this is powerful:**
- The model can learn: "When male partner disagrees on questions 10-15 (finances) AND female partner disagrees on questions 20-25 (communication) → High Risk"
- It captures **individual perspectives** and **how they combine** to create risk
- It's not just "they disagree" - it's "WHO disagrees on WHAT" and "HOW they disagree"

**Example:**
```
Male responses:  [4, 4, 2, 2, 4, 3, 2, 4, ...]
                 ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
                 Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8

Female responses: [4, 3, 4, 2, 2, 4, 4, 2, ...]
                   ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
                   Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8

ML Model learns:
- Q3: Male disagrees (2), Female agrees (4) → Disagreement on this topic
- Q4: Both disagree (2, 2) → Both unhappy with this
- Q5: Male agrees (4), Female disagrees (2) → Disagreement on this topic
- Pattern: Many disagreements across multiple questions → High Risk
```

#### B. As Comparison (Creates 6 Personalized Features) - Relationship Dynamics

**Purpose:** Create summary metrics that describe the **relationship dynamics**.

- The system **compares** male_responses[i] vs female_responses[i] for each question
- This comparison creates the **personalized features**:
  - `alignment_score`: Overall agreement (0-1)
  - `conflict_ratio`: Frequency of disagreements (0-1)
  - `category_alignments`: Alignment per MEAI category (4 values)

**Why both are needed:**
- **Separate features (118):** Capture individual patterns and question-specific details
- **Comparison features (6):** Capture overall relationship dynamics
- **Together:** Provide comprehensive view of both individual perspectives AND relationship health

**How Disagreement is Calculated:**

```python
# This happens BEFORE feeding to ML model
alignment_score = 0
conflict_count = 0

for each question i (0 to 58):
    male_resp = male_responses[i]      # e.g., 4 (Agree)
    female_resp = female_responses[i]  # e.g., 2 (Disagree)
    
    # Calculate difference
    difference = abs(male_resp - female_resp)  # |4 - 2| = 2
    
    # Count conflicts
    if difference >= 2:
        conflict_count += 1  # Significant disagreement
    elif difference == 1:
        conflict_count += 0.5  # Minor disagreement
    
    # Calculate alignment (how close they are)
    alignment_score += (4 - difference) / 4  # (4-2)/4 = 0.5

# Final personalized features
alignment_score = alignment_score / 59  # Average alignment
conflict_ratio = conflict_count / 59    # Conflict frequency
```

**Result:**
- These personalized features (alignment_score, conflict_ratio, category_alignments) are **also fed to the ML model** as 6 additional features
- So the ML model gets:
  - **Raw responses** (118 features: 59 male + 59 female) - individual patterns
  - **Comparison results** (6 features: alignment, conflict, category alignments) - relationship dynamics

**Complete Example:**

```
Couple Profile:
  - Demographics: Male 45, Female 30, Age gap 15, Education mismatch

MEAI Responses:
  Male:   [4, 4, 2, 2, 3, 4, 2, 4, 3, 2, ...] (59 values)
  Female: [4, 3, 2, 4, 2, 2, 4, 2, 4, 2, ...] (59 values)

Comparison (Creates Personalized Features):
  - Question 1: Male=4, Female=4 → Agree (difference=0)
  - Question 2: Male=4, Female=3 → Minor disagreement (difference=1)
  - Question 3: Male=2, Female=2 → Both disagree (difference=0, but both disagree)
  - Question 4: Male=2, Female=4 → Major disagreement (difference=2)
  ...
  Result: alignment_score = 0.35, conflict_ratio = 0.45

ML Model Input (135 features):
  [11 demographics] + 
  [59 male responses] + 
  [59 female responses] + 
  [6 personalized features] = 135 features

ML Model Analysis:
  - Sees: Large age gap (15 years)
  - Sees: Male has many 2s (disagrees frequently)
  - Sees: Female has many 2s (disagrees frequently)
  - Sees: Low alignment (0.35), High conflict (0.45)
  - Pattern learned: "Large age gap + both partners disagree frequently + low alignment → High Risk"

ML Model Prediction: **High Risk**
```

**Summary:**
- **59 male responses** = Male partner's individual answers (used as separate features)
- **59 female responses** = Female partner's individual answers (used as separate features)
- **Comparison** = Calculates alignment/conflict (becomes personalized features)
- **ML Model** = Uses BOTH the raw responses AND the comparison results to predict risk

---

## Why Separate Features Are Powerful

### The Problem with Just Comparing:

If we only compared responses (e.g., "they disagree on 20 questions"), we'd lose important information:

**Lost Information:**
- ❌ Which partner disagrees more?
- ❌ What topics do they disagree on?
- ❌ Are disagreements concentrated in specific areas?
- ❌ What's each partner's overall response style?

### The Power of Separate Features:

By keeping them separate, the ML model can learn:

**Individual Patterns:**
- "Male partner disagrees on financial questions (Q10-15) → Financial stress indicator"
- "Female partner is neutral on communication questions (Q20-25) → Communication avoidance"

**Combined Patterns:**
- "Male disagrees on finances **AND** Female disagrees on communication → High Risk"
- "Both partners disagree on same topics → Mutual dissatisfaction → High Risk"
- "One partner positive, one negative → Asymmetry → Medium Risk"

**Question-Specific Insights:**
- "Question 15 (finances) disagreement is a strong predictor"
- "Questions 20-25 (communication) alignment indicates stability"

**Response Style Analysis:**
- "Male: 70% positive (4s), 20% neutral (3s), 10% negative (2s) → Generally optimistic"
- "Female: 30% positive (4s), 40% neutral (3s), 30% negative (2s) → More cautious/concerned"
- "Asymmetric styles → Potential communication mismatch"

### Real-World Example:

```
Couple A:
  Male: [4, 4, 4, 4, 4, 4, ...] (mostly agrees)
  Female: [2, 2, 2, 2, 2, 2, ...] (mostly disagrees)
  → Model learns: "One very positive, one very negative → High Risk"

Couple B:
  Male: [4, 4, 2, 2, 4, 4, ...] (mixed)
  Female: [4, 4, 2, 2, 4, 4, ...] (mixed, similar pattern)
  → Model learns: "Both have similar patterns → Lower Risk"

Couple C:
  Male: [3, 3, 3, 3, 3, 3, ...] (mostly neutral)
  Female: [3, 3, 3, 3, 3, 3, ...] (mostly neutral)
  → Model learns: "Both neutral → May indicate avoidance/uncertainty → Medium Risk"
```

**The model learns these patterns automatically from training data!**

---

## Direct Answer: What's the Purpose?

**Q: If it's not a comparison, what's the purpose of having 59 male + 59 female responses as separate features?**

**A: To learn individual response patterns and how they interact, not just whether they agree or disagree.**

### The Purpose:

1. **Learn Individual Response Styles:**
   - Each partner has their own way of responding
   - Some are always positive (many 4s)
   - Some are always negative (many 2s)
   - Some are uncertain (many 3s)
   - The model learns: "This response style pattern → indicates risk"

2. **Learn Pattern Interactions:**
   - Not just "they disagree" but "WHO disagrees on WHAT"
   - "Male disagrees on finances + Female disagrees on communication → High Risk"
   - "Both disagree on same topics → Mutual dissatisfaction → High Risk"
   - "One positive, one negative → Asymmetry → Medium Risk"

3. **Learn Question-Specific Patterns:**
   - Which specific questions are most predictive?
   - "Question 15 (finances) disagreement → Strong risk indicator"
   - "Questions 20-25 (communication) alignment → Stability indicator"

4. **Learn Response Distributions:**
   - How many 2s, 3s, 4s does each partner have?
   - "Male: 70% positive → Optimistic"
   - "Female: 30% positive → Concerned"
   - "Asymmetric distributions → Communication mismatch"

### Why Not Just Compare?

If we only compared (e.g., "they disagree on 20 questions"), we'd lose:
- ❌ Which partner disagrees more
- ❌ What topics they disagree on
- ❌ Individual response styles
- ❌ Question-specific patterns

### Why Both Separate AND Comparison?

- **Separate features (118):** Capture individual patterns and details
- **Comparison features (6):** Capture overall relationship dynamics
- **Together:** Complete picture of both individual perspectives AND relationship health

**Think of it like this:**
- **Separate features** = "What does each partner think individually?"
- **Comparison features** = "How well do they agree overall?"
- **ML Model** = "Based on both, what's the risk level?"

---

## Direct Answer to Your Question

**Q: Does it compare the 59 male and 59 female responses? Does it measure disagreement in their responses?**

**A: Yes, but in TWO ways:**

### 1. As Separate Features (118 features)
- The ML model receives **each partner's responses separately**
- Male: 59 individual values `[4, 4, 3, 2, 4, ...]`
- Female: 59 individual values `[4, 3, 4, 2, 4, ...]`
- **Purpose:** Learn patterns from each partner's individual response style
- **Not compared here** - just fed as separate features

### 2. As Comparison (Creates 6 Personalized Features)
- The system **compares** male_responses[i] vs female_responses[i] for each question
- Calculates: alignment_score, conflict_ratio, category_alignments
- **Purpose:** Measure relationship dynamics (agreement/disagreement)
- **Compared here** - creates summary metrics

### Both Are Used Together

The ML model gets:

- **Raw individual responses** (118 features) → Learns individual patterns
- **Comparison results** (6 features) → Learns relationship dynamics
- **Demographics** (11 features) → Learns demographic patterns

**Total: 135 features** that together help predict risk level

**Example:**
```
Question 5: "We agree on financial decisions"
  Male: 4 (Agree)
  Female: 2 (Disagree)
  
Used as:
  1. Separate features: male_responses[4] = 4, female_responses[4] = 2
  2. Comparison: difference = |4-2| = 2 → contributes to conflict_ratio
```

Both pieces of information help the ML model make better predictions!

### Step 3: Hybrid Decision Logic

The system **intelligently combines** both methods:

```
┌─────────────────────────────────────────┐
│ Response-Based: Medium Risk (25%)      │
│ ML Prediction: High Risk               │
│                                         │
│ Hybrid Decision Logic:                 │
│ → If responses show Low Risk AND        │
│   high alignment/low conflict:          │
│   Trust response-based (ignore ML)      │
│                                         │
│ → If responses show High Risk:         │
│   Trust response-based (more reliable) │
│                                         │
│ → If ML suggests higher risk:           │
│   Use ML (catches demographic patterns)│
│                                         │
│ → Otherwise: Use response-based        │
└─────────────────────────────────────────┘
         ↓
   Final Risk Level
```

---

## What This Means

### The Risk Level is NOT just:
- ❌ Only questionnaire responses
- ❌ Only demographic factors
- ❌ Only ML prediction

### The Risk Level IS:
- ✅ **Hybrid assessment** combining responses + demographics
- ✅ **Intelligent decision** that prioritizes the most reliable indicator
- ✅ **Comprehensive evaluation** considering all relationship factors

---

## Why Hybrid Approach?

### Benefits:

1. **Prevents False Positives:**
   - If responses show healthy relationship (high alignment, low conflict)
   - But demographics suggest risk (large age gap, education mismatch)
   - System trusts the **actual relationship health** (responses)

2. **Catches Hidden Patterns:**
   - If responses show moderate disagreement
   - But demographics indicate high risk (e.g., divorced + large age gap)
   - ML model can identify patterns that response-only analysis might miss

3. **Balanced Assessment:**
   - Response-based: Direct measure of relationship health
   - ML-based: Pattern recognition from similar couples
   - Hybrid: Best of both worlds

---

## Example Scenarios

### Scenario 1: Response-Based Wins
```
Response-Based: Low Risk (8% disagreement, 90% alignment)
ML Prediction: High Risk (large age gap, education mismatch)
Final Decision: Low Risk
Reason: Actual relationship health (responses) overrides demographic concerns
```

### Scenario 2: ML Prediction Wins
```
Response-Based: Medium Risk (25% disagreement)
ML Prediction: High Risk (demographic patterns indicate higher risk)
Final Decision: High Risk
Reason: ML catches patterns that response-only analysis misses
```

### Scenario 3: Both Agree
```
Response-Based: High Risk (42% disagreement)
ML Prediction: High Risk
Final Decision: High Risk
Reason: Both methods agree - clear high risk situation
```

---

## Where You See It

### 1. View ML Recommendations Page (`view_ml_recommendations.php`)
- **Card Title:** "Relationship Assessment"
- **Badge:** "Low/Medium/High Risk"
- **Description:** Shows it's a hybrid assessment
- **Detailed Reasoning:** Explains which method was used and why

### 2. ML Dashboard (`ml_dashboard.php`)
- **Table Column:** "Risk Level (Hybrid: Responses + Demographics)"
- **Summary Cards:** Counts of Low/Medium/High Risk couples
- **Filter Buttons:** Filter by risk level

---

## Key Takeaway

**The Risk Level is a comprehensive, hybrid assessment that considers:**
- ✅ **What couples say** (questionnaire responses)
- ✅ **Who they are** (demographics)
- ✅ **How they compare** (alignment, conflict)
- ✅ **What patterns suggest** (ML learned patterns)

**It's not just one factor - it's an intelligent combination of all factors to provide the most accurate relationship risk assessment.**

---

## Technical Details

### Response-Based Calculation:
- Uses: Questionnaire responses only
- Measures: Disagreement ratio (weighted)
- Thresholds: High >35%, Medium >20%, Low ≤20%

### ML Model Prediction:
- Uses: 135 features (demographics + responses + personalized)
- Algorithm: Random Forest Classifier
- Trained on: 522 couples (22 real + 500 synthetic)

### Hybrid Decision:
- Logic: Smart selection between response-based and ML prediction
- Priority: Response-based for clear cases, ML for pattern recognition
- Result: Most accurate risk assessment

---

## Summary

**Risk Level = Hybrid Assessment of:**
1. Questionnaire responses (disagreement analysis)
2. Demographic factors (age, education, income, civil status)
3. ML pattern recognition (learned from similar couples)

**It's the overall relationship risk level that considers everything about the couple to provide the most accurate assessment.**

