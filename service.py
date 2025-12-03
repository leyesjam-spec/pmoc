# ML Service for Couple Counseling Recommendations
"""
Machine Learning Service
Uses Random Forest models for couple counseling risk assessment and recommendations
"""

import os
import json
import pickle
import threading
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import class_weight

# Import imbalanced-learn with error handling
try:
    from imblearn.over_sampling import SMOTE  # type: ignore
    from imblearn.combine import SMOTETomek  # type: ignore
    IMBALANCED_LEARN_AVAILABLE = True
    print("[OK] imbalanced-learn imported successfully - SMOTE features enabled")
except ImportError as e:
    IMBALANCED_LEARN_AVAILABLE = False
    SMOTE = None  # type: ignore
    SMOTETomek = None  # type: ignore
    print(f"Warning: imbalanced-learn not available. SMOTE features will be disabled. Error: {e}")
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

def calculate_personalized_features_flask(questionnaire_responses, male_responses, female_responses):
    """Calculate personalized features in Flask service when not provided by PHP API"""
    
    # If we have separate male/female responses, use them
    if male_responses and female_responses and len(male_responses) > 0 and len(female_responses) > 0:
        # Use the separate responses
        all_responses = male_responses + female_responses
    else:
        # questionnaire_responses should be a flat array of all responses
        # We need to split them properly - the PHP API should ideally send them separately
        all_responses = questionnaire_responses
        
        # Since we don't have separate male/female responses, we need to calculate them
        # This is a fallback - the PHP API should ideally send them separately
        if len(questionnaire_responses) >= 2:
            mid_point = len(questionnaire_responses) // 2
            male_responses = questionnaire_responses[:mid_point]
            female_responses = questionnaire_responses[mid_point:]
        else:
            male_responses = questionnaire_responses
            female_responses = questionnaire_responses
    
    # Calculate overall alignment score
    alignment_score = 0
    conflict_count = 0
    weighted_conflict_sum = 0.0
    total_questions = min(len(male_responses), len(female_responses))
    
    for i in range(total_questions):
        male_resp = male_responses[i] if i < len(male_responses) else 3
        female_resp = female_responses[i] if i < len(female_responses) else 3
        
        # Calculate alignment (how close their responses are)
        difference = abs(male_resp - female_resp)
        alignment_score += (4 - difference) / 4  # 0-1 scale
        
        # Count conflicts (responses that are very different)
        if difference >= 2:
            conflict_count += 1
            weighted_conflict_sum += 1.0
        elif difference == 1:
            # Identify which side of neutral
            if (male_resp == 3 and female_resp == 2) or (male_resp == 2 and female_resp == 3):
                weighted_conflict_sum += 0.6
            elif (male_resp == 4 and female_resp == 3) or (male_resp == 3 and female_resp == 4):
                weighted_conflict_sum += 0.4
            else:
                weighted_conflict_sum += 0.5
    
    alignment_score = alignment_score / total_questions if total_questions > 0 else 0.5
    conflict_ratio = (weighted_conflict_sum / total_questions) if total_questions > 0 else 0
    
    # Calculate partner averages
    male_avg = sum(male_responses) / len(male_responses) if male_responses else 3.0
    female_avg = sum(female_responses) / len(female_responses) if female_responses else 3.0
    
    # NEW: Category-specific alignment scores (4 features, one per MEAI category)
    category_alignments = []
    for category_id in range(1, len(MEAI_CATEGORIES) + 1):
        # Get question IDs for this category
        category_question_ids = [qid for qid, cid in MEAI_QUESTION_MAPPING.items() if cid == category_id]
        
        if not category_question_ids:
            category_alignments.append(0.5)  # Default if no questions
            continue
        
        # Calculate alignment for questions in this category only
        category_alignment = 0
        category_question_count = 0
        
        for qid in category_question_ids:
            # Find response index (qid is 1-indexed, responses are 0-indexed)
            resp_idx = qid - 1
            if resp_idx < len(male_responses) and resp_idx < len(female_responses):
                male_resp = male_responses[resp_idx]
                female_resp = female_responses[resp_idx]
                difference = abs(male_resp - female_resp)
                category_alignment += (4 - difference) / 4
                category_question_count += 1
        
        category_alignment = category_alignment / category_question_count if category_question_count > 0 else 0.5
        category_alignments.append(category_alignment)
    
    # NEW: Extreme response counts (4 features)
    male_agree_count = sum(1 for r in male_responses if r == 4)  # Agree = 4
    male_disagree_count = sum(1 for r in male_responses if r == 2)  # Disagree = 2
    female_agree_count = sum(1 for r in female_responses if r == 4)
    female_disagree_count = sum(1 for r in female_responses if r == 2)
    
    # Convert to ratios for better scaling
    total_male = len(male_responses) if male_responses else 1
    total_female = len(female_responses) if female_responses else 1
    male_agree_ratio = male_agree_count / total_male
    male_disagree_ratio = male_disagree_count / total_male
    female_agree_ratio = female_agree_count / total_female
    female_disagree_ratio = female_disagree_count / total_female
    
    return {
        'alignment_score': alignment_score,
        'conflict_ratio': conflict_ratio,
        'male_avg_response': male_avg,
        'female_avg_response': female_avg,
        'total_conflicts': conflict_count,
        # NEW: Category-specific alignments (4 features)
        'category_alignments': category_alignments,
        # NEW: Extreme response ratios (4 features)
        'male_agree_ratio': male_agree_ratio,
        'male_disagree_ratio': male_disagree_ratio,
        'female_agree_ratio': female_agree_ratio,
        'female_disagree_ratio': female_disagree_ratio
    }

# Global variables for models
ml_models = {
    'risk_model': None,
    'category_model': None,
    'risk_encoder': None
}

# Training status tracking
training_status = {
    'in_progress': False,
    'progress': 0,
    'message': '',
    'error': None,
    'thread': None  # Track the training thread
}
training_lock = threading.Lock()

# MEAI Categories - dynamically loaded from database question_category table
# These 4 categories are used for ML predictions and recommendations
MEAI_CATEGORIES = []

# MEAI Questions and Sub-questions - dynamically loaded from database
MEAI_QUESTIONS = {}  # {category_id: {question_id: {text, sub_questions: []}}}
MEAI_QUESTION_MAPPING = {}  # {question_id: category_id}

# ============================================================================
# DATA VALIDATION FUNCTIONS
# ============================================================================

def validate_couple_data(couple_profile, questionnaire_responses, male_responses=None, female_responses=None):
    """Validate input data before training/prediction"""
    errors = []
    warnings = []
    
    # Validate ages
    male_age = couple_profile.get('male_age', 0)
    female_age = couple_profile.get('female_age', 0)
    
    if not (18 <= male_age <= 100):
        errors.append(f"Invalid male age: {male_age} (must be 18-100)")
    if not (18 <= female_age <= 100):
        errors.append(f"Invalid female age: {female_age} (must be 18-100)")
    
    # Check age gap (warning if too large)
    age_gap = abs(male_age - female_age)
    if age_gap > 30:
        warnings.append(f"Large age gap detected: {age_gap} years")
    
    # Validate education and income levels
    education_level = couple_profile.get('education_level', 0)
    income_level = couple_profile.get('income_level', 0)
    
    if not (0 <= education_level <= 4):
        errors.append(f"Invalid education level: {education_level} (must be 0-4)")
    if not (0 <= income_level <= 4):
        errors.append(f"Invalid income level: {income_level} (must be 0-4)")
    
    # Validate years living together
    years_together = couple_profile.get('years_living_together', 0)
    if years_together < 0:
        errors.append(f"Invalid years living together: {years_together} (cannot be negative)")
    if years_together > 50:
        warnings.append(f"Unusually high years living together: {years_together}")
    
    # Validate children count
    children = couple_profile.get('children', 0)
    if children < 0:
        errors.append(f"Invalid children count: {children} (cannot be negative)")
    if children > 10:
        warnings.append(f"Unusually high children count: {children}")
    
    # Validate questionnaire responses
    if questionnaire_responses:
        expected_count = len(MEAI_QUESTION_MAPPING) if MEAI_QUESTION_MAPPING else 31
        if len(questionnaire_responses) != expected_count:
            errors.append(f"Expected {expected_count} responses, got {len(questionnaire_responses)}")
        
        # Check for invalid response values
        invalid_responses = [r for r in questionnaire_responses if r not in [2, 3, 4]]
        if invalid_responses:
            errors.append(f"Invalid response values found: {set(invalid_responses)} (must be 2, 3, or 4)")
        
        # Check for edge case: all responses the same
        unique_responses = set(questionnaire_responses)
        if len(unique_responses) == 1:
            warnings.append("All questionnaire responses are identical - may indicate data quality issue")
        
        # Check for too many neutral responses (potential issue)
        neutral_count = sum(1 for r in questionnaire_responses if r == 3)
        neutral_ratio = neutral_count / len(questionnaire_responses) if questionnaire_responses else 0
        if neutral_ratio > 0.8:
            warnings.append(f"High proportion of neutral responses: {neutral_ratio:.1%} (may indicate uncertainty)")
    
    # Validate male/female responses if provided
    if male_responses and female_responses:
        if len(male_responses) != len(female_responses):
            errors.append(f"Mismatched response counts: male={len(male_responses)}, female={len(female_responses)}")
        
        # Check for identical responses (edge case)
        if male_responses == female_responses:
            warnings.append("Male and female responses are identical - may indicate data quality issue")
    
    # Validate civil status
    civil_status = couple_profile.get('civil_status', 'Single')
    valid_statuses = ['Single', 'Living In', 'Separated', 'Divorced', 'Widowed']
    if civil_status not in valid_statuses:
        warnings.append(f"Unusual civil status: {civil_status} (expected: {valid_statuses})")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def validate_training_data(X, y_risk, y_categories):
    """Validate training data before model training"""
    errors = []
    warnings = []
    
    # Check for missing values
    if np.isnan(X).any():
        nan_count = np.isnan(X).sum()
        errors.append(f"Found {nan_count} NaN values in feature matrix")
    
    if np.isinf(X).any():
        inf_count = np.isinf(X).sum()
        errors.append(f"Found {inf_count} infinite values in feature matrix")
    
    # Check for empty data
    if len(X) == 0:
        errors.append("Training data is empty")
    
    if len(y_risk) == 0:
        errors.append("Risk labels are empty")
    
    if len(y_categories) == 0:
        errors.append("Category labels are empty")
    
    # Check data shapes
    if len(X) != len(y_risk):
        errors.append(f"Mismatched data shapes: X has {len(X)} samples, y_risk has {len(y_risk)}")
    
    if len(X) != len(y_categories):
        errors.append(f"Mismatched data shapes: X has {len(X)} samples, y_categories has {len(y_categories)}")
    
    # Check for class imbalance
    unique_risks, counts = np.unique(y_risk, return_counts=True)
    if len(unique_risks) < 3:
        warnings.append(f"Only {len(unique_risks)} risk classes found (expected 3) - synthetic samples will be added")
    
    # Check for severe imbalance (this is expected before SMOTE)
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    if imbalance_ratio > 5:
        warnings.append(f"Class imbalance detected before SMOTE: ratio {imbalance_ratio:.1f}:1 (SMOTE will balance this)")
    
    # Check for outliers in features (using IQR method)
    for i in range(min(10, X.shape[1])):  # Check first 10 features
        Q1 = np.percentile(X[:, i], 25)
        Q3 = np.percentile(X[:, i], 75)
        IQR = Q3 - Q1
        if IQR > 0:
            outliers = np.sum((X[:, i] < Q1 - 3*IQR) | (X[:, i] > Q3 + 3*IQR))
            if outliers > len(X) * 0.05:  # More than 5% outliers
                warnings.append(f"Feature {i} has {outliers} potential outliers ({outliers/len(X):.1%})")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def load_categories_from_db():
    """Load MEAI categories from database question_category table"""
    global MEAI_CATEGORIES
    try:
        import pymysql
        
        # Database connection (same as PHP)
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='',
            database='u520834156_DBpmoc25',
            charset='utf8mb4'
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT category_name FROM question_category ORDER BY category_id ASC")
        rows = cursor.fetchall()
        
        # Extract category names and simplify them
        # Expected format: "MARRIAGE EXPECTATIONS AND INVENTORY ON [CATEGORY NAME]"
        MEAI_CATEGORIES = []
        for row in rows:
            full_name = row[0]
            
            # Split on " ON " to extract category name
            if ' ON ' in full_name:
                # Get the part after " ON "
                short_name = full_name.split(' ON ', 1)[1].strip()
                # Convert from ALL CAPS to Title Case
                short_name = short_name.title()
                MEAI_CATEGORIES.append(short_name)
            else:
                # Fallback: use full name if format is unexpected
                MEAI_CATEGORIES.append(full_name.title())
        
        conn.close()
        print(f"Loaded {len(MEAI_CATEGORIES)} MEAI categories from database")
        return True
    except Exception as e:
        print(f"Error loading categories from database: {e}")
        # Fallback to hardcoded categories
        MEAI_CATEGORIES = [
            'Marriage And Relationship',
            'Responsible Parenthood',
            'Planning The Family',
            'Maternal Neonatal Child Health And Nutrition'
        ]
        print("Using fallback categories")
        return False

def load_questions_from_db():
    """Load MEAI questions and sub-questions from database"""
    global MEAI_QUESTIONS, MEAI_QUESTION_MAPPING
    try:
        import pymysql
        
        # Database connection
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='',
            database='u520834156_DBpmoc25',
            charset='utf8mb4'
        )
        
        cursor = conn.cursor()
        
        # Load questions with sub-questions
        query = """
        SELECT 
            qa.category_id,
            qa.question_id,
            qa.question_text,
            sqa.sub_question_id,
            sqa.sub_question_text
        FROM question_assessment qa
        LEFT JOIN sub_question_assessment sqa ON qa.question_id = sqa.question_id
        ORDER BY qa.category_id ASC, qa.question_id ASC, sqa.sub_question_id ASC
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Initialize structure
        MEAI_QUESTIONS = {}
        MEAI_QUESTION_MAPPING = {}
        
        for row in rows:
            category_id, question_id, question_text, sub_question_id, sub_question_text = row
            
            # Initialize category if not exists
            if category_id not in MEAI_QUESTIONS:
                MEAI_QUESTIONS[category_id] = {}
            
            # Initialize question if not exists
            if question_id not in MEAI_QUESTIONS[category_id]:
                MEAI_QUESTIONS[category_id][question_id] = {
                    'text': question_text,
                    'sub_questions': []
                }
            
            # Add sub-question if exists
            if sub_question_text:
                MEAI_QUESTIONS[category_id][question_id]['sub_questions'].append(sub_question_text)
        
        # Build mapping for answerable questions only
        question_counter = 1
        for cat_id, cat_questions in MEAI_QUESTIONS.items():
            for q_id, q_data in cat_questions.items():
                if q_data['sub_questions']:
                    # Question has sub-questions, map each sub-question
                    for sub_idx in range(len(q_data['sub_questions'])):
                        MEAI_QUESTION_MAPPING[question_counter] = cat_id
                        question_counter += 1
                else:
                    # Standalone question, map it
                    MEAI_QUESTION_MAPPING[question_counter] = cat_id
                    question_counter += 1
        
        conn.close()
        
        # Count answerable questions only (standalone main questions + sub-questions)
        total_answerable_questions = 0
        for cat_questions in MEAI_QUESTIONS.values():
            for q in cat_questions.values():
                if q['sub_questions']:
                    # Question has sub-questions, count only the sub-questions
                    total_answerable_questions += len(q['sub_questions'])
                else:
                    # Standalone question, count it
                    total_answerable_questions += 1
        
        print(f"Loaded {total_answerable_questions} answerable questions from database")
        print(f"Questions by category:")
        for cat_id, cat_questions in MEAI_QUESTIONS.items():
            answerable_count = 0
            for q in cat_questions.values():
                if q['sub_questions']:
                    answerable_count += len(q['sub_questions'])
                else:
                    answerable_count += 1
            print(f"  Category {cat_id}: {answerable_count} answerable questions")
        
        return True
        
    except Exception as e:
        print(f"Error loading questions from database: {e}")
        # Fallback: create basic structure
        MEAI_QUESTIONS = {
            1: {1: {'text': 'Marriage and Relationship Question', 'sub_questions': []}},
            2: {2: {'text': 'Responsible Parenthood Question', 'sub_questions': []}},
            3: {3: {'text': 'Planning The Family Question', 'sub_questions': []}},
            4: {4: {'text': 'Maternal Neonatal Child Health Question', 'sub_questions': []}}
        }
        MEAI_QUESTION_MAPPING = {1: 1, 2: 2, 3: 3, 4: 4}
        print("Using fallback question structure")
        return False

def generate_synthetic_data_based_on_real_couples(num_couples, real_couples_data):
    """Generate synthetic couples based on patterns from real couples"""
    np.random.seed(42)
    
    if not real_couples_data:
        print("No real couples data available, using generic synthetic data")
        return generate_synthetic_data(num_couples)
    
    print(f"Generating {num_couples} synthetic couples based on {len(real_couples_data)} real couples")
    
    # Extract patterns from real couples
    real_ages = [(row['male_age'], row['female_age']) for row in real_couples_data]
    real_civil_status = [row['civil_status'] for row in real_couples_data]
    real_education = [row['education_level'] for row in real_couples_data]
    real_income = [row['income_level'] for row in real_couples_data]
    real_children = [row['children'] for row in real_couples_data]
    real_years_together = [row['years_living_together'] for row in real_couples_data]
    real_responses = [row['questionnaire_responses'] for row in real_couples_data]
    real_risk_levels = [row['risk_level'] for row in real_couples_data]
    
    # Calculate statistics from real data
    male_ages = [age[0] for age in real_ages]
    female_ages = [age[1] for age in real_ages]
    age_gaps = [abs(age[0] - age[1]) for age in real_ages]
    
    # Ensure we generate samples for all 3 risk classes
    # Allocate: 30% Low, 40% Medium, 30% High
    num_low = int(num_couples * 0.30)
    num_medium = int(num_couples * 0.40)
    num_high = num_couples - num_low - num_medium
    
    data = []
    
    # Generate couples for all risk classes in one loop
    for i in range(num_couples):
        # Sample from real couple patterns with some variation
        base_couple_idx = np.random.randint(0, len(real_couples_data))
        base_couple = real_couples_data[base_couple_idx]
        
        # Generate ages based on real patterns with variation
        male_age = int(np.random.normal(np.mean(male_ages), np.std(male_ages)))
        female_age = int(np.random.normal(np.mean(female_ages), np.std(female_ages)))
        
        # Ensure realistic age ranges
        male_age = max(18, min(80, male_age))
        female_age = max(18, min(80, female_age))
        
        # Age gap based on real patterns
        real_age_gap = abs(male_age - female_age)
        if real_age_gap > np.percentile(age_gaps, 90):  # Large age gap
            # Keep the large gap but adjust ages
            if male_age > female_age:
                female_age = max(18, male_age - real_age_gap)
            else:
                male_age = max(18, female_age - real_age_gap)
        
        # Sample other attributes from real couples with variation
        civil_status = np.random.choice(real_civil_status)
        
        # Years living together based on civil status and age
        if civil_status == 'Living In':
            years_living_together = np.random.randint(1, max(1, int(np.mean(real_years_together)) + 5))
        else:
            years_living_together = 0
        
        # Children based on real patterns
        has_past_children = np.random.choice([True, False], p=[0.3, 0.7]) if np.random.random() < 0.4 else False
        if has_past_children:
            children = np.random.choice(real_children) if real_children else np.random.randint(1, 3)
        else:
            children = 0
        
        # Education and income based on real patterns
        education_level = np.random.choice(real_education)
        income_level = np.random.choice(real_income)
        
        # Generate questionnaire responses based on real patterns
        base_responses = base_couple['questionnaire_responses']
        questionnaire_responses = []
        
        # Determine target risk level for this couple
        if i < num_low:
            target_risk = 'Low'
            target_disagree_ratio = np.random.uniform(0.0, 0.15)  # Low risk: 0-15% disagree
        elif i < num_low + num_medium:
            target_risk = 'Medium'
            target_disagree_ratio = np.random.uniform(0.15, 0.30)  # Medium risk: 15-30% disagree
        else:
            target_risk = 'High'
            target_disagree_ratio = np.random.uniform(0.30, 0.60)  # High risk: 30-60% disagree
        
        # Generate responses to match target disagree ratio
        total_questions = len(base_responses)
        target_disagree_count = int(total_questions * target_disagree_ratio)
        target_agree_count = int(total_questions * (1 - target_disagree_ratio) * 0.6)  # 60% of remaining are agree
        target_neutral_count = total_questions - target_disagree_count - target_agree_count
        
        # Create response array
        response_array = [2] * target_disagree_count + [4] * target_agree_count + [3] * target_neutral_count
        np.random.shuffle(response_array)
        
        # Apply some variation based on real patterns
        for j, base_response in enumerate(base_responses):
            if j < len(response_array):
                # Blend target response with base pattern (70% target, 30% base)
                if np.random.random() < 0.3:
                    variation = np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
                    new_response = max(2, min(4, base_response + variation))
                else:
                    new_response = response_array[j]
                questionnaire_responses.append(new_response)
            else:
                questionnaire_responses.append(response_array[j % len(response_array)])
        
        # Calculate actual risk level based on response patterns
        disagree_count = sum(1 for r in questionnaire_responses if r == 2)
        disagree_ratio = disagree_count / len(questionnaire_responses)
        
        if disagree_ratio > 0.30:
            risk_level = 'High'
        elif disagree_ratio > 0.15:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        # Generate category scores based on actual question-category mapping
        category_scores = []
        
        for category_id in range(1, len(MEAI_CATEGORIES) + 1):
            # Get questions for this category
            category_question_ids = [qid for qid, cid in MEAI_QUESTION_MAPPING.items() if cid == category_id]
            
            if not category_question_ids:
                category_scores.append(0.5)  # Default score if no questions
                continue
            
            # Get responses for questions in this category
            category_responses = []
            for qid in category_question_ids:
                # Find response index for this question
                if qid <= len(questionnaire_responses):
                    category_responses.append(questionnaire_responses[qid - 1])  # question_id is 1-indexed
            
            if not category_responses:
                category_scores.append(0.5)  # Default score if no responses
                continue
            
            # Calculate disagreement ratio for this category
            cat_disagree_ratio = sum(1 for r in category_responses if r == 2) / len(category_responses)
            
            # Convert to 0-1 score (higher disagreement = higher score)
            category_score = min(1.0, cat_disagree_ratio * 2)  # Scale up disagreement
            category_scores.append(category_score)
        
        data.append({
            'male_age': male_age,
            'female_age': female_age,
            'civil_status': civil_status,
            'years_living_together': years_living_together,
            'past_children': has_past_children,
            'children': children,
            'education_level': education_level,
            'income_level': income_level,
            'questionnaire_responses': questionnaire_responses,
            'risk_level': risk_level,
            'category_scores': category_scores
        })
    
    return data

def generate_synthetic_data(num_couples=500):
    """Generate realistic synthetic couple data for training (fallback method)"""
    np.random.seed(42)
    
    data = []
    
    # Ensure we generate samples for all 3 risk classes
    # Allocate: 30% Low, 40% Medium, 30% High
    num_low = int(num_couples * 0.30)
    num_medium = int(num_couples * 0.40)
    num_high = num_couples - num_low - num_medium
    
    # Define realistic couple profiles with different risk patterns
    couple_profiles = [
        # Young couples (18-25) - often higher risk due to immaturity
        {'age_range': (18, 25), 'risk_bias': 'high', 'weight': 0.15},
        # Young adults (25-30) - moderate risk, learning phase
        {'age_range': (25, 30), 'risk_bias': 'medium', 'weight': 0.25},
        # Mature couples (30-40) - lower risk, more stable
        {'age_range': (30, 40), 'risk_bias': 'low', 'weight': 0.30},
        # Established couples (40-50) - very low risk, experienced
        {'age_range': (40, 50), 'risk_bias': 'low', 'weight': 0.20},
        # Older couples (50+) - mixed, some very stable, some with issues
        {'age_range': (50, 70), 'risk_bias': 'medium', 'weight': 0.10}
    ]
    
    for i in range(num_couples):
        # Select couple profile based on weights
        profile = np.random.choice(couple_profiles, p=[p['weight'] for p in couple_profiles])
        min_age, max_age = profile['age_range']
        risk_bias = profile['risk_bias']
        
        # Generate ages with realistic age gaps
        male_age = np.random.randint(min_age, max_age + 1)
        
        # Age gap patterns: most couples have 0-5 year gap, some have larger gaps
        age_gap_options = [
            (0, 2),    # Same age: 40%
            (1, 3),    # Small gap: 30%
            (2, 5),    # Medium gap: 20%
            (5, 15),   # Large gap: 8%
            (15, 25)   # Very large gap: 2%
        ]
        age_gap_weights = [0.40, 0.30, 0.20, 0.08, 0.02]
        
        age_gap_range = age_gap_options[np.random.choice(len(age_gap_options), p=age_gap_weights)]
        age_gap = np.random.randint(age_gap_range[0], age_gap_range[1] + 1)
        
        # Female age based on male age and gap
        if np.random.random() < 0.5:  # 50% chance female is younger
            female_age = max(18, male_age - age_gap)
        else:  # 50% chance female is older
            female_age = min(80, male_age + age_gap)
        
        # Civil status based on age and risk profile
        if risk_bias == 'high':
            civil_status_options = ['Single', 'Single', 'Living In', 'Separated', 'Divorced']
        elif risk_bias == 'low':
            civil_status_options = ['Single', 'Living In', 'Living In', 'Widowed']
        else:  # medium
            civil_status_options = ['Single', 'Living In', 'Widowed', 'Separated']
        
        civil_status = np.random.choice(civil_status_options)
        
        # Years living together based on civil status and age
        if civil_status == 'Living In':
            if male_age < 25:
                years_living_together = np.random.randint(1, 5)  # Young couples, shorter time
            elif male_age < 40:
                years_living_together = np.random.randint(1, 15)  # Mature couples, longer time
            else:
                years_living_together = np.random.randint(5, 25)  # Older couples, very long time
        else:
            years_living_together = 0
        
        # Past children based on age and civil status
        if male_age > 25 and civil_status in ['Living In', 'Widowed', 'Divorced']:
            has_past_children = np.random.choice([True, False], p=[0.4, 0.6])
        else:
            has_past_children = np.random.choice([True, False], p=[0.1, 0.9])
        
        if has_past_children:
            if male_age < 30:
                children = np.random.randint(1, 3)  # Young parents, fewer children
            else:
                children = np.random.randint(1, 5)  # Older parents, more children
        else:
            children = 0
        
        # Education levels based on age (older = more likely higher education)
        if male_age < 25:
            education_level = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.2, 0.4, 0.2, 0.1])
        elif male_age < 40:
            education_level = np.random.choice([0, 1, 2, 3, 4], p=[0.05, 0.1, 0.3, 0.4, 0.15])
        else:
            education_level = np.random.choice([0, 1, 2, 3, 4], p=[0.05, 0.05, 0.2, 0.5, 0.2])
        
        # Income levels based on education and age
        if education_level >= 3:  # Higher education
            income_level = np.random.choice([2, 3, 4], p=[0.2, 0.5, 0.3])
        elif education_level >= 2:  # Medium education
            income_level = np.random.choice([1, 2, 3, 4], p=[0.1, 0.4, 0.4, 0.1])
        else:  # Lower education
            income_level = np.random.choice([0, 1, 2, 3], p=[0.2, 0.4, 0.3, 0.1])
        
        # Generate questionnaire responses (3-option scale: agree/neutral/disagree)
        # Use dynamic question count from database
        total_questions = len(MEAI_QUESTION_MAPPING) if MEAI_QUESTION_MAPPING else 31  # Fallback to 31
        questionnaire_responses = np.random.randint(2, 5, total_questions)  # 2=disagree, 3=neutral, 4=agree
        
        # Determine target risk level for this couple based on allocation
        if i < num_low:
            target_risk = 'Low'
            target_disagree_ratio = np.random.uniform(0.0, 0.15)  # Low risk: 0-15% disagree
        elif i < num_low + num_medium:
            target_risk = 'Medium'
            target_disagree_ratio = np.random.uniform(0.15, 0.30)  # Medium risk: 15-30% disagree
        else:
            target_risk = 'High'
            target_disagree_ratio = np.random.uniform(0.30, 0.60)  # High risk: 30-60% disagree
        
        # Generate responses based on target risk level and couple characteristics
        base_disagree_prob = target_disagree_ratio
        base_agree_prob = (1 - target_disagree_ratio) * 0.6  # 60% of remaining are agree
        
        # Adjust based on age gap (larger gaps = more disagreements)
        age_gap = abs(male_age - female_age)
        if age_gap > 10:
            base_disagree_prob = min(0.8, base_disagree_prob + 0.1)
            base_agree_prob = max(0.1, base_agree_prob - 0.05)
        elif age_gap > 5:
            base_disagree_prob = min(0.8, base_disagree_prob + 0.05)
            base_agree_prob = max(0.1, base_agree_prob - 0.02)
        
        # Adjust based on education mismatch
        education_diff = abs(education_level - income_level)
        if education_diff > 2:
            base_disagree_prob = min(0.8, base_disagree_prob + 0.05)
            base_agree_prob = max(0.1, base_agree_prob - 0.02)
        
        # Adjust based on civil status
        if civil_status in ['Separated', 'Divorced']:
            base_disagree_prob = min(0.8, base_disagree_prob + 0.1)
            base_agree_prob = max(0.1, base_agree_prob - 0.05)
        elif civil_status == 'Living In' and years_living_together > 10:
            base_disagree_prob = max(0.05, base_disagree_prob - 0.05)
            base_agree_prob = min(0.8, base_agree_prob + 0.05)
        
        # Ensure probabilities are valid
        base_disagree_prob = max(0.05, min(0.8, base_disagree_prob))
        base_agree_prob = max(0.1, min(0.8, base_agree_prob))
        base_neutral_prob = 1.0 - base_disagree_prob - base_agree_prob
        
        # Generate responses
        questionnaire_responses = np.random.choice(
            [2, 3, 4],  # disagree, neutral, agree
            total_questions,
            p=[base_disagree_prob, base_neutral_prob, base_agree_prob]
        )
        
        # Calculate risk level based on actual response patterns
        disagree_count = sum(1 for r in questionnaire_responses if r == 2)
        disagree_ratio = disagree_count / len(questionnaire_responses)
        
        if disagree_ratio > 0.30:
            risk_level = 'High'
        elif disagree_ratio > 0.15:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        # Generate category scores based on actual question-category mapping
        category_scores = []
        
        for category_id in range(1, len(MEAI_CATEGORIES) + 1):
            # Get questions for this category
            category_question_ids = [qid for qid, cid in MEAI_QUESTION_MAPPING.items() if cid == category_id]
            
            if not category_question_ids:
                category_scores.append(0.5)  # Default score if no questions
                continue
            
            # Get responses for questions in this category
            category_responses = []
            for qid in category_question_ids:
                # Find response index for this question
                if qid <= len(questionnaire_responses):
                    category_responses.append(questionnaire_responses[qid - 1])  # question_id is 1-indexed
            
            if not category_responses:
                category_scores.append(0.5)  # Default score if no responses
                continue
            
            # Calculate disagreement ratio for this category
            cat_disagree_ratio = sum(1 for r in category_responses if r == 2) / len(category_responses)
            
            # Convert to 0-1 score (higher disagreement = higher score)
            category_score = min(1.0, cat_disagree_ratio * 2)  # Scale up disagreement
            category_scores.append(category_score)
        
        data.append({
            'male_age': male_age,
            'female_age': female_age,
            'civil_status': civil_status,
            'years_living_together': years_living_together,
            'past_children': has_past_children,
            'children': children,
            'education_level': education_level,
            'income_level': income_level,
            'questionnaire_responses': questionnaire_responses.tolist(),
            'risk_level': risk_level,
            'category_scores': category_scores
        })
    
    return data

def load_real_couples_for_training():
    """Load real couples from database for ML training"""
    try:
        import pymysql
        
        # Database connection
        conn = pymysql.connect(
            host='localhost',
            user='root',
            password='',
            database='u520834156_DBpmoc25',
            charset='utf8mb4'
        )
        
        cursor = conn.cursor()
        
        # Get all couples with their profiles and responses
        query = """
        SELECT 
            cp.access_id,
            MAX(CASE WHEN cp.sex = 'Male' THEN cp.first_name END) as male_name,
            MAX(CASE WHEN cp.sex = 'Female' THEN cp.first_name END) as female_name,
            MAX(CASE WHEN cp.sex = 'Male' THEN cp.age END) as male_age,
            MAX(CASE WHEN cp.sex = 'Female' THEN cp.age END) as female_age,
            MAX(cp.civil_status) as civil_status,
            MAX(cp.years_living_together) as years_living_together,
            MAX(cp.past_children) as past_children,
            MAX(cp.past_children_count) as children,
            MAX(cp.education) as education,
            MAX(cp.monthly_income) as monthly_income
        FROM couple_profile cp
        GROUP BY cp.access_id
        HAVING COUNT(DISTINCT cp.sex) = 2
        """
        
        cursor.execute(query)
        couples = cursor.fetchall()
        
        if not couples:
            print("No couples found in database")
            return []
        
        print(f"Found {len(couples)} real couples for training")
        
        # Get MEAI responses for each couple
        training_data = []
        
        # Education and income mapping (same as PHP)
        education_mapping = {
            'No Education': 0, 'Pre School': 0, 'Elementary Level': 0, 'Elementary Graduate': 0,
            'High School Level': 1, 'High School Graduate': 1, 'Junior HS Level': 1, 'Junior HS Graduate': 1,
            'Senior HS Level': 1, 'Senior HS Graduate': 1, 'College Level': 2, 'College Graduate': 3,
            'Vocational/Technical': 2, 'ALS': 1, 'Post Graduate': 4
        }
        
        income_mapping = {
            '5000 below': 0, '5999-9999': 0, '10000-14999': 1, '15000-19999': 1,
            '20000-24999': 2, '25000 above': 3
        }
        
        for couple in couples:
            access_id, male_name, female_name, male_age, female_age, civil_status, years_living_together, past_children, children, education, monthly_income = couple
            
            # Get MEAI responses for this couple from couple_responses table
            # Get responses ordered by category, question, sub-question, then respondent
            response_query = """
            SELECT cr.category_id, cr.question_id, cr.sub_question_id, cr.respondent, cr.response
            FROM couple_responses cr
            WHERE cr.access_id = %s
            ORDER BY cr.category_id, cr.question_id, COALESCE(cr.sub_question_id, 0), cr.respondent
            """
            cursor.execute(response_query, (access_id,))
            responses = cursor.fetchall()
            
            if len(responses) < 20:  # Need minimum responses
                continue
                
            # Build response map: (category_id, question_id, sub_question_id) -> {male: val, female: val}
            response_map = {}
            for category_id, question_id, sub_question_id, respondent, response in responses:
                key = (category_id, question_id, sub_question_id)
                if key not in response_map:
                    response_map[key] = {'male': None, 'female': None}
                
                # Convert response to numeric (2=disagree, 3=neutral, 4=agree)
                if response == 'agree':
                    resp_value = 4
                elif response == 'neutral':
                    resp_value = 3
                else:  # disagree
                    resp_value = 2
                
                if respondent.lower() == 'male':
                    response_map[key]['male'] = resp_value
                else:
                    response_map[key]['female'] = resp_value
            
            # Build questionnaire_responses array matching MEAI_QUESTION_MAPPING structure
            # MEAI_QUESTION_MAPPING maps sequential IDs (1, 2, 3...) to category_ids
            # We need to match responses using the MEAI_QUESTIONS structure
            questionnaire_responses = []
            
            # Build a lookup: (category_id, question_id, sub_question_id) -> response value
            # IMPORTANT: Don't just average - consider partner disagreements as risk indicators
            response_lookup = {}
            for key, resp_data in response_map.items():
                category_id, question_id, sub_question_id = key
                male_resp = resp_data.get('male')
                female_resp = resp_data.get('female')
                
                # Consider partner disagreements as indicators of conflict
                if male_resp is not None and female_resp is not None:
                    # If partners disagree (difference >= 2), this indicates conflict
                    # Use the more negative response (lower value) to capture the disagreement
                    if abs(male_resp - female_resp) >= 2:  # Significant disagreement
                        # Use the lower (more negative) response to reflect the conflict
                        avg_resp = min(male_resp, female_resp)
                    else:
                        # Minor difference - use average
                        avg_resp = round((male_resp + female_resp) / 2)
                    response_lookup[key] = avg_resp
                elif male_resp is not None:
                    response_lookup[key] = male_resp
                elif female_resp is not None:
                    response_lookup[key] = female_resp
                else:
                    response_lookup[key] = 3  # Default to neutral
            
            # Build responses in the order defined by MEAI_QUESTION_MAPPING
            # Iterate through categories and questions in the same order as MEAI_QUESTION_MAPPING
            for cat_id in sorted(MEAI_QUESTIONS.keys()):
                cat_questions = MEAI_QUESTIONS[cat_id]
                for q_id in sorted(cat_questions.keys()):
                    q_data = cat_questions[q_id]
                    
                    if q_data['sub_questions']:
                        # Question has sub-questions - map each sub-question
                        for sub_idx, sub_q_text in enumerate(q_data['sub_questions']):
                            # sub_question_id in database is 1-indexed, sub_idx is 0-indexed
                            sub_q_id = sub_idx + 1
                            key = (cat_id, q_id, sub_q_id)
                            
                            if key in response_lookup:
                                questionnaire_responses.append(response_lookup[key])
                            else:
                                questionnaire_responses.append(3)  # Default to neutral
                    else:
                        # Standalone question - no sub-question
                        key = (cat_id, q_id, None)
                        
                        if key in response_lookup:
                            questionnaire_responses.append(response_lookup[key])
                        else:
                            questionnaire_responses.append(3)  # Default to neutral
            
            # Get total expected responses (includes both main questions AND sub-questions)
            # MEAI_QUESTION_MAPPING maps each answerable question (standalone or sub-question) to a sequential ID
            total_expected_responses = len(MEAI_QUESTION_MAPPING)
            
            # Pad or truncate to expected number of responses
            while len(questionnaire_responses) < total_expected_responses:
                questionnaire_responses.append(3)  # Default to neutral
            questionnaire_responses = questionnaire_responses[:total_expected_responses]
            
            # Calculate risk level based on actual responses
            # NOTE: This is a heuristic for LABELING training data only
            # The ML model will learn from this labeled data and make predictions
            # For actual predictions, the ML model is used (see analyze() function)
            # More disagreements = higher risk
            # Also count neutral responses (3) as potential issues if they're from disagreements
            disagree_count = sum(1 for r in questionnaire_responses if r == 2)
            neutral_count = sum(1 for r in questionnaire_responses if r == 3)
            # Count neutral as partial disagreement (they might indicate unresolved issues)
            weighted_disagree_count = disagree_count + (neutral_count * 0.3)
            disagree_ratio = weighted_disagree_count / len(questionnaire_responses)
            
            # Adjusted thresholds to be more sensitive to issues
            if disagree_ratio > 0.25:  # Lowered from 0.3 to catch more high-risk cases
                risk_level = 'High'
            elif disagree_ratio > 0.10:  # Lowered from 0.15 to catch more medium-risk cases
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            # Generate category scores based on actual question-category mapping
            category_scores = []
            
            for category_id in range(1, len(MEAI_CATEGORIES) + 1):
                # Get questions for this category
                category_question_ids = [qid for qid, cid in MEAI_QUESTION_MAPPING.items() if cid == category_id]
                
                if not category_question_ids:
                    category_scores.append(0.5)  # Default score if no questions
                    continue
                
                # Get responses for questions in this category
                category_responses = []
                for qid in category_question_ids:
                    # Find response index for this question
                    # Note: This assumes responses are ordered by question_id
                    if qid <= len(questionnaire_responses):
                        category_responses.append(questionnaire_responses[qid - 1])  # question_id is 1-indexed
                
                if not category_responses:
                    category_scores.append(0.5)  # Default score if no responses
                    continue
                
                # Calculate disagreement ratio for this category
                # Count both disagreements and neutrals (which may indicate unresolved issues)
                cat_disagree_count = sum(1 for r in category_responses if r == 2)
                cat_neutral_count = sum(1 for r in category_responses if r == 3)
                # Weight neutrals as partial disagreements
                weighted_disagree_count = cat_disagree_count + (cat_neutral_count * 0.3)
                cat_disagree_ratio = weighted_disagree_count / len(category_responses) if len(category_responses) > 0 else 0
                
                # Convert to 0-1 score (higher disagreement = higher score)
                # Use a better scaling function to capture more nuance
                category_score = min(1.0, cat_disagree_ratio * 2.5)  # Increased from 2.0 to 2.5 for better sensitivity
                category_scores.append(category_score)
            
            # Map education and income to numeric levels
            education_level = education_mapping.get(education, 2) if education else 2
            income_level = income_mapping.get(monthly_income, 1) if monthly_income else 1
            
            # Convert ages to integers (age is stored as varchar)
            try:
                male_age = int(float(str(male_age).strip())) if male_age else 30
                female_age = int(float(str(female_age).strip())) if female_age else 30
            except (ValueError, TypeError):
                male_age = 30
                female_age = 30
            
            # Convert past_children from varchar ('Yes'/'No') to boolean
            past_children_bool = False
            if past_children:
                past_children_str = str(past_children).strip().lower()
                past_children_bool = past_children_str in ['yes', '1', 'true']
            
            # Convert years_living_together from varchar to int
            years_together_int = 0
            if years_living_together:
                try:
                    years_together_int = int(float(str(years_living_together).strip()))
                except (ValueError, TypeError):
                    years_together_int = 0
            
            training_data.append({
                'male_age': male_age,
                'female_age': female_age,
                'civil_status': civil_status or 'Single',
                'years_living_together': years_together_int,
                'past_children': past_children_bool,
                'children': int(children) if children else 0,
                'education_level': education_level,
                'income_level': income_level,
                'questionnaire_responses': questionnaire_responses,
                'risk_level': risk_level,
                'category_scores': category_scores
            })
        
        conn.close()
        print(f"Loaded {len(training_data)} real couples for training")
        return training_data
        
    except Exception as e:
        print(f"Error loading real couples: {e}")
        return []

def train_ml_models():
    """Train machine learning models"""
    print("Training ML models...")
    
    # Update progress: Loading questions and categories
    with training_lock:
        training_status['progress'] = 15
        training_status['message'] = 'Loading questions and categories...'
    
    # Ensure questions are loaded before training
    if not MEAI_CATEGORIES:
        load_categories_from_db()
    if not MEAI_QUESTIONS:
        load_questions_from_db()
    
    # Update progress: Loading real couples
    with training_lock:
        training_status['progress'] = 20
        training_status['message'] = 'Loading real couples from database...'
    
    # Load real couples from database for training
    real_couples_data = load_real_couples_for_training()
    
    # Always generate synthetic data (500 couples)
    # If we have real couples, use them to inform the synthetic generation
    if not real_couples_data:
        print("No real couples found, using generic synthetic data")
        synthetic_data = generate_synthetic_data(500)
        data = synthetic_data
    else:
        print(f"Found {len(real_couples_data)} real couples")
        print(f"Generating 500 synthetic couples based on real couple patterns")
        # Generate 500 synthetic couples based on real couple patterns
        synthetic_data = generate_synthetic_data_based_on_real_couples(500, real_couples_data)
        
        # Combine real couples with synthetic data
        print(f"Combining {len(real_couples_data)} real couples with {len(synthetic_data)} synthetic couples")
        data = real_couples_data + synthetic_data
        print(f"Total training data: {len(data)} couples (real + synthetic)")
    
    df = pd.DataFrame(data)
    
    # Update progress: Preparing features
    with training_lock:
        training_status['progress'] = 25
        training_status['message'] = 'Preparing features and encoding data...'
    
    # Prepare features
    X = []
    y_risk = []
    y_categories = []
    
    total_rows = len(df)
    for idx, row in df.iterrows():
        # Update progress during feature preparation (25-35%)
        if idx % 50 == 0:
            progress = 25 + int((idx / total_rows) * 10)
            with training_lock:
                training_status['progress'] = progress
                training_status['message'] = f'Preparing features... ({idx}/{total_rows} couples)'
        # NEW: Calculate age gap
        age_gap = abs(row['male_age'] - row['female_age'])
        
        # NEW: Calculate education/income compatibility
        education_income_diff = abs(row['education_level'] - row['income_level'])
        
        # NEW: Civil status encoding (one-hot: 3 features)
        civil_status = row.get('civil_status', 'Single')
        is_single = 1 if civil_status == 'Single' else 0
        is_living_in = 1 if civil_status == 'Living In' else 0
        is_separated_divorced = 1 if civil_status in ['Separated', 'Divorced', 'Widowed'] else 0
        
        # Combine all features (same structure as analysis)
        features = [
            row['male_age'],
            row['female_age'],
            age_gap,  # NEW
            row['years_living_together'],
            row['children'],
            row['education_level'],
            row['income_level'],
            education_income_diff,  # NEW
            is_single,  # NEW
            is_living_in,  # NEW
            is_separated_divorced  # NEW
        ]
        
        # Add questionnaire responses (ensure it's a flat list)
        questionnaire_responses = row['questionnaire_responses']
        if isinstance(questionnaire_responses, np.ndarray):
            features.extend(questionnaire_responses.tolist())
        else:
            features.extend(questionnaire_responses)
        
        # Add personalized features (synthetic for training)
        # Generate synthetic personalized features based on risk level
        if row['risk_level'] == 'High':
            # High risk: low alignment, high conflict
            alignment_score = np.random.uniform(0.2, 0.5)
            conflict_ratio = np.random.uniform(0.3, 0.7)
            # Category alignments: lower for high risk
            category_alignments = [np.random.uniform(0.2, 0.5) for _ in range(4)]
            # More extreme responses for high risk
            male_agree_ratio = np.random.uniform(0.1, 0.3)
            male_disagree_ratio = np.random.uniform(0.3, 0.5)
            female_agree_ratio = np.random.uniform(0.1, 0.3)
            female_disagree_ratio = np.random.uniform(0.3, 0.5)
        elif row['risk_level'] == 'Low':
            # Low risk: high alignment, low conflict
            alignment_score = np.random.uniform(0.6, 0.9)
            conflict_ratio = np.random.uniform(0.0, 0.2)
            # Category alignments: higher for low risk
            category_alignments = [np.random.uniform(0.6, 0.9) for _ in range(4)]
            # More balanced responses for low risk
            male_agree_ratio = np.random.uniform(0.4, 0.6)
            male_disagree_ratio = np.random.uniform(0.0, 0.2)
            female_agree_ratio = np.random.uniform(0.4, 0.6)
            female_disagree_ratio = np.random.uniform(0.0, 0.2)
        else:  # Medium
            # Medium risk: mixed patterns
            alignment_score = np.random.uniform(0.4, 0.7)
            conflict_ratio = np.random.uniform(0.1, 0.4)
            # Category alignments: mixed
            category_alignments = [np.random.uniform(0.3, 0.7) for _ in range(4)]
            # Moderate extreme responses
            male_agree_ratio = np.random.uniform(0.2, 0.5)
            male_disagree_ratio = np.random.uniform(0.1, 0.3)
            female_agree_ratio = np.random.uniform(0.2, 0.5)
            female_disagree_ratio = np.random.uniform(0.1, 0.3)
        
        # Add personalized features to match analysis structure
        personalized_features = [
            alignment_score,
            conflict_ratio,
            np.random.uniform(2.5, 3.5),  # male_avg_response
            np.random.uniform(2.5, 3.5),  # female_avg_response
            # NEW: Category-specific alignments (4 features)
            *category_alignments,
            # NEW: Extreme response ratios (4 features)
            male_agree_ratio,
            male_disagree_ratio,
            female_agree_ratio,
            female_disagree_ratio
        ]
        
        features.extend(personalized_features)
        
        X.append(features)
        
        # Risk level encoding
        risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        y_risk.append(risk_mapping[row['risk_level']])
        
        # Category scores (ensure it's a flat list)
        category_scores = row['category_scores']
        if isinstance(category_scores, np.ndarray):
            y_categories.append(category_scores.tolist())
        elif isinstance(category_scores, list):
            y_categories.append(category_scores)
        else:
            # Convert to list if it's not already
            y_categories.append(list(category_scores))
    
    # Convert to numpy arrays AFTER the loop completes
    X = np.array(X)
    y_risk = np.array(y_risk)
    y_categories = np.array(y_categories)
    
    # Ensure all 3 risk classes are present
    unique_risks = np.unique(y_risk)
    if len(unique_risks) < 3:
        print(f"Warning: Only {len(unique_risks)} risk classes found (expected 3). Adding synthetic samples to ensure all classes are represented...")
        
        # Find missing risk classes
        all_risk_classes = {0: 'Low', 1: 'Medium', 2: 'High'}
        missing_classes = [rc for rc in all_risk_classes.keys() if rc not in unique_risks]
        
        # Generate synthetic samples for missing classes
        total_questions = len(MEAI_QUESTION_MAPPING) if MEAI_QUESTION_MAPPING else 31
        synthetic_samples = []
        synthetic_risks = []
        synthetic_categories = []
        
        for missing_class in missing_classes:
            risk_name = all_risk_classes[missing_class]
            # Generate 10 synthetic samples for each missing class
            for _ in range(10):
                # Generate ages
                male_age = np.random.randint(25, 50)
                female_age = np.random.randint(23, 48)
                age_gap = abs(male_age - female_age)
                
                # Generate other attributes
                civil_status = np.random.choice(['Single', 'Living In', 'Widowed'])
                years_living_together = np.random.randint(0, 10) if civil_status == 'Living In' else 0
                children = np.random.randint(0, 3)
                education_level = np.random.randint(0, 4)
                income_level = np.random.randint(0, 4)
                education_income_diff = abs(education_level - income_level)
                
                # Civil status encoding
                is_single = 1 if civil_status == 'Single' else 0
                is_living_in = 1 if civil_status == 'Living In' else 0
                is_separated_divorced = 1 if civil_status in ['Separated', 'Divorced', 'Widowed'] else 0
                
                # Generate questionnaire responses based on risk level
                if risk_name == 'High':
                    disagree_ratio = np.random.uniform(0.30, 0.60)
                elif risk_name == 'Medium':
                    disagree_ratio = np.random.uniform(0.15, 0.30)
                else:  # Low
                    disagree_ratio = np.random.uniform(0.0, 0.15)
                
                disagree_count = int(total_questions * disagree_ratio)
                agree_count = int(total_questions * (1 - disagree_ratio) * 0.6)
                neutral_count = total_questions - disagree_count - agree_count
                
                questionnaire_responses = [2] * disagree_count + [4] * agree_count + [3] * neutral_count
                np.random.shuffle(questionnaire_responses)
                
                # Generate category scores based on risk level
                if risk_name == 'High':
                    category_scores = [np.random.uniform(0.5, 1.0) for _ in range(len(MEAI_CATEGORIES))]
                elif risk_name == 'Medium':
                    category_scores = [np.random.uniform(0.3, 0.7) for _ in range(len(MEAI_CATEGORIES))]
                else:  # Low
                    category_scores = [np.random.uniform(0.0, 0.5) for _ in range(len(MEAI_CATEGORIES))]
                
                # Build features
                features = [
                    male_age, female_age, age_gap, years_living_together, children,
                    education_level, income_level, education_income_diff,
                    is_single, is_living_in, is_separated_divorced
                ]
                
                # Add personalized features
                if risk_name == 'High':
                    alignment_score = np.random.uniform(0.2, 0.5)
                    conflict_ratio = np.random.uniform(0.3, 0.7)
                    category_alignments = [np.random.uniform(0.2, 0.5) for _ in range(4)]
                    male_agree_ratio = np.random.uniform(0.1, 0.3)
                    male_disagree_ratio = np.random.uniform(0.3, 0.5)
                    female_agree_ratio = np.random.uniform(0.1, 0.3)
                    female_disagree_ratio = np.random.uniform(0.3, 0.5)
                elif risk_name == 'Low':
                    alignment_score = np.random.uniform(0.6, 0.9)
                    conflict_ratio = np.random.uniform(0.0, 0.2)
                    category_alignments = [np.random.uniform(0.6, 0.9) for _ in range(4)]
                    male_agree_ratio = np.random.uniform(0.4, 0.6)
                    male_disagree_ratio = np.random.uniform(0.0, 0.2)
                    female_agree_ratio = np.random.uniform(0.4, 0.6)
                    female_disagree_ratio = np.random.uniform(0.0, 0.2)
                else:  # Medium
                    alignment_score = np.random.uniform(0.4, 0.7)
                    conflict_ratio = np.random.uniform(0.1, 0.4)
                    category_alignments = [np.random.uniform(0.3, 0.7) for _ in range(4)]
                    male_agree_ratio = np.random.uniform(0.2, 0.5)
                    male_disagree_ratio = np.random.uniform(0.1, 0.3)
                    female_agree_ratio = np.random.uniform(0.2, 0.5)
                    female_disagree_ratio = np.random.uniform(0.1, 0.3)
                
                personalized_features = [
                    alignment_score, conflict_ratio,
                    np.random.uniform(2.5, 3.5),  # male_avg_response
                    np.random.uniform(2.5, 3.5),  # female_avg_response
                    *category_alignments,
                    male_agree_ratio, male_disagree_ratio,
                    female_agree_ratio, female_disagree_ratio
                ]
                
                features.extend(personalized_features)
                features.extend(questionnaire_responses)
                
                synthetic_samples.append(features)
                synthetic_risks.append(missing_class)
                synthetic_categories.append(category_scores)
        
        # Add synthetic samples to training data
        if synthetic_samples:
            X_synthetic = np.array(synthetic_samples)
            y_risk_synthetic = np.array(synthetic_risks)
            y_categories_synthetic = np.array(synthetic_categories)
            
            X = np.vstack([X, X_synthetic])
            y_risk = np.concatenate([y_risk, y_risk_synthetic])
            y_categories = np.vstack([y_categories, y_categories_synthetic])
            
            print(f"Added {len(synthetic_samples)} synthetic samples to ensure all risk classes are represented")
            print(f"Final class distribution: {np.bincount(y_risk)}")
    
    # Update progress: Validating data
    with training_lock:
        training_status['progress'] = 35
        training_status['message'] = 'Validating training data...'
    
    # Validate training data
    print("Validating training data...")
    validation_result = validate_training_data(X, y_risk, y_categories)
    
    if not validation_result['valid']:
        print("ERROR: Training data validation failed:")
        for error in validation_result['errors']:
            print(f"  - {error}")
        return False
    
    if validation_result['warnings']:
        print("WARNINGS during validation:")
        for warning in validation_result['warnings']:
            print(f"  - {warning}")
    
    print(f"Training with {X.shape[1]} features: {X.shape[0]} samples")
    
    # Track original data composition for logging
    num_real_couples = len(real_couples_data) if real_couples_data else 0
    num_synthetic_couples = 500  # Always 500 synthetic couples
    print(f"Data composition: {num_real_couples} real couples + {num_synthetic_couples} synthetic couples = {X.shape[0]} total")
    print(f"Class distribution before SMOTE: {np.bincount(y_risk)}")
    
    # Apply SMOTE to the COMBINED dataset (real + synthetic couples)
    # This ensures balanced classes while preserving patterns from real data
    if not IMBALANCED_LEARN_AVAILABLE:
        print("Skipping SMOTE - imbalanced-learn not available")
    else:
        # Update progress: Applying SMOTE
        with training_lock:
            training_status['progress'] = 40
            training_status['message'] = 'Applying SMOTE for class balancing...'
        
        print("Applying SMOTE to combined dataset (real + synthetic couples) for class balancing...")
        try:
            # Use SMOTETomek (combines SMOTE oversampling with Tomek undersampling)
            smote_tomek = SMOTETomek(random_state=42, n_jobs=-1)
            X_resampled, y_risk_resampled = smote_tomek.fit_resample(X, y_risk)
            
            # For category scores, we need to match the resampled indices
            # SMOTETomek returns original samples + synthetic samples
            # We'll use the original samples and generate synthetic category scores for new samples
            original_size = len(X)
            resampled_size = len(X_resampled)
            
            if resampled_size > original_size:
                # New synthetic samples were added - generate matching category scores
                y_categories_resampled = np.zeros((resampled_size, y_categories.shape[1]))
                y_categories_resampled[:original_size] = y_categories
                
                # For synthetic samples, generate category scores based on their risk level
                for i in range(original_size, resampled_size):
                    risk_class = y_risk_resampled[i]
                    if risk_class == 2:  # High risk
                        y_categories_resampled[i] = np.random.uniform(0.5, 1.0, y_categories.shape[1])
                    elif risk_class == 1:  # Medium risk
                        y_categories_resampled[i] = np.random.uniform(0.3, 0.7, y_categories.shape[1])
                    else:  # Low risk
                        y_categories_resampled[i] = np.random.uniform(0.0, 0.5, y_categories.shape[1])
            else:
                # Some samples were removed (Tomek links) - keep matching ones
                # This is a simplified approach - in practice, we'd track which samples were kept
                y_categories_resampled = y_categories[:resampled_size]
            
            print(f"After SMOTE: {X_resampled.shape[0]} samples (was {X.shape[0]})")
            print(f"Class distribution after SMOTE: {np.bincount(y_risk_resampled)}")
            
            X = X_resampled
            y_risk = y_risk_resampled
            y_categories = y_categories_resampled
        except Exception as e:
            print(f"SMOTE failed (using original data): {e}")
            print("This is normal if you have very few samples or all samples in one class")
            # Continue with original data if SMOTE fails
    
    # Check for class imbalance
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_risk),
        y=y_risk
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class distribution: {np.bincount(y_risk)}")
    print(f"Class weights: {class_weight_dict}")
    
    # Update progress: Training risk model
    with training_lock:
        training_status['progress'] = 50
        training_status['message'] = 'Tuning hyperparameters for risk model...'
    
    # Hyperparameter tuning for risk model
    print("Tuning hyperparameters for risk model...")
    risk_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', class_weight_dict]
    }
    
    risk_base_model = RandomForestClassifier(random_state=42)
    risk_grid_search = GridSearchCV(
        risk_base_model,
        risk_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Update progress during risk model training (50-70%)
    with training_lock:
        training_status['progress'] = 60
        training_status['message'] = 'Training risk model (this may take a few minutes)...'
    
    risk_grid_search.fit(X, y_risk)
    risk_model = risk_grid_search.best_estimator_
    print(f"Best risk model params: {risk_grid_search.best_params_}")
    print(f"Best risk model CV score: {risk_grid_search.best_score_:.3f}")
    
    # Update progress: Training category model
    with training_lock:
        training_status['progress'] = 70
        training_status['message'] = 'Tuning hyperparameters for category model...'
    
    # Hyperparameter tuning for category model
    print("Tuning hyperparameters for category model...")
    category_param_grid = {
        'estimator__n_estimators': [100, 200],
        'estimator__max_depth': [10, 15, None],
        'estimator__min_samples_split': [2, 5]
    }
    
    category_base_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    category_grid_search = GridSearchCV(
        category_base_model,
        category_param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Update progress during category model training (70-85%)
    with training_lock:
        training_status['progress'] = 80
        training_status['message'] = 'Training category model (this may take a few minutes)...'
    
    category_grid_search.fit(X, y_categories)
    category_model = category_grid_search.best_estimator_
    print(f"Best category model params: {category_grid_search.best_params_}")
    print(f"Best category model CV score: {category_grid_search.best_score_:.3f}")
    
    # Update progress: Cross-validation
    with training_lock:
        training_status['progress'] = 85
        training_status['message'] = 'Evaluating models with cross-validation...'
    
    # Cross-validation evaluation
    risk_cv_scores = cross_val_score(risk_model, X, y_risk, cv=5, scoring='accuracy')
    print(f"Risk model CV accuracy: {risk_cv_scores.mean():.3f} (+/- {risk_cv_scores.std() * 2:.3f})")
    
    # Create risk encoder
    risk_encoder = LabelEncoder()
    risk_encoder.fit(['Low', 'Medium', 'High'])
    
    # Update progress: Saving models
    with training_lock:
        training_status['progress'] = 90
        training_status['message'] = 'Saving trained models to disk...'
        
        # Save models
    ml_models['risk_model'] = risk_model
    ml_models['category_model'] = category_model
    ml_models['risk_encoder'] = risk_encoder
    
    # Save to files
    try:
        with open('risk_model.pkl', 'wb') as f:
            pickle.dump(risk_model, f)
        
        with open('category_model.pkl', 'wb') as f:
            pickle.dump(category_model, f)
        
        with open('risk_encoder.pkl', 'wb') as f:
            pickle.dump(risk_encoder, f)
        
        # Update progress: Complete
        with training_lock:
            training_status['progress'] = 95
            training_status['message'] = 'Models saved successfully!'
        
        print("ML models trained and saved successfully")
        return True
    except Exception as e:
        print(f"Error saving models: {e}")
        # Models are still in memory, so training was successful
        # Just couldn't save to disk
        return True  # Still return True since models are loaded in memory

def load_ml_models():
    """Load pre-trained ML models"""
    try:
        if os.path.exists('risk_model.pkl'):
            with open('risk_model.pkl', 'rb') as f:
                ml_models['risk_model'] = pickle.load(f)
        
        if os.path.exists('category_model.pkl'):
            with open('category_model.pkl', 'rb') as f:
                ml_models['category_model'] = pickle.load(f)
        
        if os.path.exists('risk_encoder.pkl'):
            with open('risk_encoder.pkl', 'rb') as f:
                ml_models['risk_encoder'] = pickle.load(f)
        
        print("ML models loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading ML models: {e}")
        return False


def generate_ml_recommendations(couple_profile, risk_level, category_scores):
    """Generate ML-based counseling recommendations using model predictions"""
    
    # Map category scores to specific counseling topics based on ML predictions
    category_priorities = sorted(
        zip(MEAI_CATEGORIES, category_scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    recommendations = []
    focus_categories = []
    
    # Extract profile details for context-aware recommendations
    civil_status = couple_profile.get('civil_status', 'Single')
    years_together = couple_profile.get('years_living_together', 0)
    has_children = couple_profile.get('past_children', False)
    children_count = couple_profile.get('children', 0)
    male_age = couple_profile.get('male_age', 30)
    female_age = couple_profile.get('female_age', 30)
    
    # Process each category based on ML prediction strength
    # Four-level priority system: 0-20%, 20-40%, 40-70%, 70-100%
    for category, score in category_priorities:
        if score > 0.2:  # Show categories above 20%
            # Determine priority level based on score ranges
            if score > 0.7:  # 70-100%
                priority_level = 'Critical'
            elif score > 0.4:  # 40-70%
                priority_level = 'High'
            elif score > 0.2:  # 20-40%
                priority_level = 'Moderate'
            else:  # 0-20% (shouldn't reach here due to if condition)
                priority_level = 'Low'
            
            focus_categories.append({
                'name': category,
                'score': float(score),
                'priority': priority_level
            })
            
            # Generate recommendations based on ML-predicted MEAI category needs
            # Match categories regardless of exact case/formatting
            category_lower = category.lower()
            
            if 'marriage' in category_lower and 'relationship' in category_lower:
                if score > 0.7:
                    recommendations.append(f"High priority: Strengthen marriage expectations and relationship foundations - ML analysis indicates significant development needs")
                recommendations.append(f"Focus on partnership quality, mutual understanding, and marriage preparation based on MEAI assessment")
                
            elif 'responsible' in category_lower and 'parenthood' in category_lower:
                if has_children:
                    recommendations.append(f"Address responsible parenting with {children_count} {'child' if children_count == 1 else 'children'} - ML analysis suggests focused attention needed")
                recommendations.append(f"Strengthen family planning knowledge, shared parental responsibilities, and informed decision-making - ML-identified priority")
                    
            elif 'planning' in category_lower and 'family' in category_lower:
                recommendations.append(f"Develop comprehensive family planning strategy and reproductive health awareness - ML model indicates this requires attention")
                recommendations.append(f"Focus on family size decisions, spacing, and contraceptive knowledge based on ML predictions")
                
            elif 'maternal' in category_lower or 'neonatal' in category_lower or 'child health' in category_lower:
                recommendations.append(f"Prioritize maternal and child health education - ML analysis highlights importance for your family's wellbeing")
                if has_children:
                    recommendations.append(f"Address nutrition and health needs for existing children while planning for future")
                else:
                    recommendations.append(f"Prepare knowledge on prenatal care, newborn health, and child nutrition for future family planning")
    
    # Add risk-based recommendations
    if risk_level == 'High':
        recommendations.insert(0, f"URGENT: ML risk assessment indicates high priority intervention needed - recommend immediate counseling")
    elif risk_level == 'Medium':
        recommendations.insert(0, f"ML analysis suggests moderate intervention - proactive counseling recommended")
    else:
        recommendations.insert(0, f"ML prediction shows positive relationship indicators - continue strengthening current patterns")
    
    # Add demographic-contextual insights from ML
    age_gap = abs(male_age - female_age)
    if age_gap > 10:
        recommendations.append(f"Consider age difference dynamics ({age_gap} years) in relationship planning - ML factor analysis")
    
    if civil_status == 'Living In' and years_together > 5:
        recommendations.append(f"Long-term cohabitation ({years_together} years) - ML suggests discussing future commitment clarity")
    elif civil_status in ['Widowed', 'Separated', 'Divorced']:
        recommendations.append(f"Previous relationship experience ({civil_status}) - ML indicates importance of healing and fresh start focus")
    
    return {
        'recommendations': recommendations[:8],  # Top 8 ML-driven recommendations
        'focus_categories': focus_categories,
        'model_type': 'Machine Learning',
        'prediction_confidence': float(np.mean([score for _, score in category_priorities[:3]])),
        'analysis_method': 'Random Forest Counseling Topics Model'
    }


@app.route('/status', methods=['GET'])
def status():
    """Check service status"""
    ml_trained = all(model is not None for model in ml_models.values())
    
    return jsonify({
        'status': 'success',
        'service': 'Counseling Topics Service',
        'ml_trained': ml_trained
    })

def train_models_async():
    """Train ML models in background thread"""
    import sys
    import traceback
    
    with training_lock:
        training_status['in_progress'] = True
        training_status['progress'] = 0
        training_status['message'] = 'Starting training...'
        training_status['error'] = None
    
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        
        with training_lock:
            training_status['progress'] = 10
            training_status['message'] = 'Loading data and preparing features...'
        
        success = train_ml_models()
        
        sys.stdout.flush()
        sys.stderr.flush()
        
        with training_lock:
            if success:
                training_status['in_progress'] = False
                training_status['progress'] = 100
                training_status['message'] = 'Training completed successfully!'
            else:
                training_status['in_progress'] = False
                training_status['progress'] = 0
                training_status['message'] = 'Training failed'
                training_status['error'] = 'Training failed - check server logs for details'
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Training error: {error_trace}", file=sys.stderr)
        sys.stderr.flush()
        
        with training_lock:
            training_status['in_progress'] = False
            training_status['progress'] = 0
            training_status['message'] = 'Training error occurred'
            training_status['error'] = str(e)

@app.route('/train', methods=['POST'])
def train():
    """Start training ML models asynchronously"""
    with training_lock:
        # Check if training is actually running (thread alive check)
        if training_status['in_progress']:
            thread = training_status.get('thread')
            if thread and thread.is_alive():
                return jsonify({
                    'status': 'error',
                    'message': 'Training is already in progress. Please wait for it to complete.'
                }), 400
            else:
                # Thread is dead but status says in_progress - reset it
                print("Warning: Training status was stuck, resetting...")
                training_status['in_progress'] = False
                training_status['progress'] = 0
                training_status['message'] = ''
                training_status['error'] = None
                training_status['thread'] = None
        
        # Reset status
        training_status['in_progress'] = True
        training_status['progress'] = 0
        training_status['message'] = 'Starting training...'
        training_status['error'] = None
        
        # Start training in background thread
        thread = threading.Thread(target=train_models_async, daemon=True)
        training_status['thread'] = thread
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': 'Training started in background',
            'training_started': True
        })

@app.route('/training_status', methods=['GET'])
def get_training_status():
    """Get current training status"""
    with training_lock:
        return jsonify({
            'status': 'success',
            'in_progress': training_status['in_progress'],
            'progress': training_status['progress'],
            'message': training_status['message'],
            'error': training_status['error']
        })

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze couple and generate recommendations"""
    try:
        data = request.get_json()
        
        # Extract couple profile with conditional field handling
        couple_profile = {
            'male_age': data.get('male_age', 30),
            'female_age': data.get('female_age', 30),
            'civil_status': data.get('civil_status', 'Single'),
            'years_living_together': data.get('years_living_together', 0),  # Only for "Living In" status
            'past_children': data.get('past_children', False),
            'children': data.get('children', 0),  # Only if past_children is True
            'education_level': data.get('education_level', 2),
            'income_level': data.get('income_level', 2)
        }
        
        # Handle conditional fields based on civil status
        if couple_profile['civil_status'] != 'Living In':
            couple_profile['years_living_together'] = 0
        
        # Handle conditional fields based on past children
        if not couple_profile['past_children']:
            couple_profile['children'] = 0
        
        # Extract questionnaire responses (dynamic count based on actual questions)
        total_questions = len(MEAI_QUESTION_MAPPING) if MEAI_QUESTION_MAPPING else 31  # Fallback to 31
        questionnaire_responses = data.get('questionnaire_responses', [3] * total_questions)
        
        # PERSONALIZED FEATURES: Extract relationship dynamics
        personalized_features = data.get('personalized_features', {})
        male_responses = data.get('male_responses', [])
        female_responses = data.get('female_responses', [])
        
        # DEBUG: Log what we received
        print(f"DEBUG - Received male_responses: {len(male_responses) if male_responses else 0} items")
        print(f"DEBUG - Received female_responses: {len(female_responses) if female_responses else 0} items")
        print(f"DEBUG - Received questionnaire_responses: {len(questionnaire_responses) if questionnaire_responses else 0} items")
        
        # If personalized features are not provided, calculate them
        if not personalized_features or len(personalized_features) == 0:
            personalized_features = calculate_personalized_features_flask(questionnaire_responses, male_responses, female_responses)
        
        # Validate input data before prediction
        validation_result = validate_couple_data(
            couple_profile, 
            questionnaire_responses, 
            male_responses, 
            female_responses
        )
        
        if not validation_result['valid']:
            return jsonify({
                'status': 'error',
                'message': 'Data validation failed: ' + '; '.join(validation_result['errors'])
            })
        
        if validation_result['warnings']:
            print("WARNINGS during data validation:")
            for warning in validation_result['warnings']:
                print(f"  - {warning}")
        
        # NEW: Calculate age gap
        age_gap = abs(couple_profile['male_age'] - couple_profile['female_age'])
        
        # NEW: Calculate education/income compatibility
        education_income_diff = abs(couple_profile['education_level'] - couple_profile['income_level'])
        
        # NEW: Civil status encoding (one-hot: 3 features)
        civil_status = couple_profile.get('civil_status', 'Single')
        is_single = 1 if civil_status == 'Single' else 0
        is_living_in = 1 if civil_status == 'Living In' else 0
        is_separated_divorced = 1 if civil_status in ['Separated', 'Divorced', 'Widowed'] else 0
        
        # Prepare features for ML models (original + personalized)
        features = [
            couple_profile['male_age'],
            couple_profile['female_age'],
            age_gap,  # NEW
            couple_profile['years_living_together'],
            couple_profile['children'],
            couple_profile['education_level'],
            couple_profile['income_level'],
            education_income_diff,  # NEW
            is_single,  # NEW
            is_living_in,  # NEW
            is_separated_divorced  # NEW
        ] + questionnaire_responses
        
        # Add personalized features
        personalized_feature_values = [
            personalized_features.get('alignment_score', 0.5),
            personalized_features.get('conflict_ratio', 0.0),
            personalized_features.get('male_avg_response', 3.0),
            personalized_features.get('female_avg_response', 3.0),
            # NEW: Category-specific alignments (4 features)
            *personalized_features.get('category_alignments', [0.5, 0.5, 0.5, 0.5]),
            # NEW: Extreme response ratios (4 features)
            personalized_features.get('male_agree_ratio', 0.0),
            personalized_features.get('male_disagree_ratio', 0.0),
            personalized_features.get('female_agree_ratio', 0.0),
            personalized_features.get('female_disagree_ratio', 0.0)
        ]
        
        features.extend(personalized_feature_values)
        
        features_array = np.array(features).reshape(1, -1)
        print(f"Analysis with {len(features)} features: {features_array.shape}")
        
        # Predict risk level using ML model only (no heuristic adjustments)
        if ml_models['risk_model'] is not None:
            risk_prediction = ml_models['risk_model'].predict(features_array)[0]
            risk_levels = ['Low', 'Medium', 'High']
            risk_level = risk_levels[risk_prediction]
            print(f"DEBUG - ML risk prediction: {risk_level} (index: {risk_prediction})")
        else:
            return jsonify({
                'status': 'error',
                'message': 'Risk model not loaded. Train or load models first.'
            })
        
        # ML confidence based solely on model probabilities
        risk_probs = ml_models['risk_model'].predict_proba(features_array)[0]
        ml_confidence = float(np.clip(np.max(risk_probs), 0.0, 1.0))
        
        # Predict category scores with personalized adjustments
        if ml_models['category_model'] is not None:
            category_scores = ml_models['category_model'].predict(features_array)[0]
            category_scores = np.clip(category_scores, 0.0, 1.0)
        else:
            return jsonify({
                'status': 'error',
                'message': 'Category model not loaded. Train or load models first.'
            })
        
        # Use raw category scores without risk-level clamping to reflect true discrepancies
        
        # Format focus categories for response - SHOW ALL CATEGORIES
        # Three-level priority system: 0-30%, 30-60%, 60-100%
        focus_categories = []
        print(f"Processing {len(MEAI_CATEGORIES)} categories: {MEAI_CATEGORIES}")
        print(f"Category scores: {category_scores}")
        
        for cat, score in zip(MEAI_CATEGORIES, category_scores):
            # Show ALL categories (not just above 20%)
            # Determine priority level based on 3-level system
            if score > 0.6:  # 60-100%
                priority_level = 'High'
            elif score > 0.3:  # 30-60%
                priority_level = 'Moderate'
            else:  # 0-30%
                priority_level = 'Low'
                
            print(f"Category: {cat}, Score: {score:.3f}, Priority: {priority_level}")
            
            focus_categories.append({
                'name': cat,
                'score': float(score),
                'priority': priority_level
            })
        
        print(f"Generated {len(focus_categories)} focus categories")
        
        # Generate specific reasoning based on actual couple features
        risk_reasoning = generate_risk_reasoning(couple_profile, personalized_features, risk_level)
        counseling_reasoning = generate_counseling_reasoning(focus_categories, category_scores, ml_confidence)
        
        # Generate personalized recommendations
        personalized_recommendations = generate_personalized_recommendations(
            risk_level, category_scores, focus_categories, 
            personalized_features, male_responses, female_responses, couple_profile
        )
        
        return jsonify({
            'status': 'success',
            'couple_id': data.get('couple_id', 'unknown'),
            'risk_level': risk_level,
            'category_scores': category_scores.tolist(),
            'focus_categories': sorted(focus_categories, key=lambda x: x['score'], reverse=True),
            'recommendations': personalized_recommendations,
            'ml_confidence': ml_confidence,  # Dynamic confidence based on risk level
            'risk_reasoning': risk_reasoning,
            'counseling_reasoning': counseling_reasoning,
            'analysis_method': 'Random Forest Counseling Topics with Personalized Features',
            'generated_at': pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Analysis error: {str(e)}'
        })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Counseling Topics Service',
        'version': '1.0.0'
    })

def generate_personalized_recommendations(risk_level, category_scores, focus_categories, personalized_features, male_responses, female_responses, couple_profile=None):
    """Generate natural language recommendations using NLG engine"""
    try:
        # Import NLG engine
        from nlg_recommendation_engine import NLGRecommendationEngine
        
        # Initialize NLG engine
        nlg_engine = NLGRecommendationEngine()
        
        # Generate natural language recommendations
        recommendations = nlg_engine.generate_natural_recommendations(
            risk_level=risk_level,
            category_scores=category_scores,
            focus_categories=focus_categories,
            personalized_features=personalized_features,
            male_responses=male_responses,
            female_responses=female_responses,
            couple_profile=couple_profile or {}
        )
        
        return recommendations
        
    except ImportError:
        # Fallback to original rule-based system if NLG engine not available
        return generate_rule_based_recommendations(risk_level, category_scores, focus_categories, personalized_features, male_responses, female_responses)
    except Exception as e:
        print(f"NLG Error: {e}")
        # Fallback to original rule-based system
        return generate_rule_based_recommendations(risk_level, category_scores, focus_categories, personalized_features, male_responses, female_responses)

def generate_rule_based_recommendations(risk_level, category_scores, focus_categories, personalized_features, male_responses, female_responses):
    """Fallback rule-based recommendation generation"""
    recommendations = []
    
    # Extract personalized features
    alignment_score = personalized_features.get('alignment_score', 0.5)
    conflict_ratio = personalized_features.get('conflict_ratio', 0.0)
    male_avg = personalized_features.get('male_avg_response', 3.0)
    female_avg = personalized_features.get('female_avg_response', 3.0)
    
    # ENHANCED PERSONALIZATION: Analyze actual response patterns
    male_agree_count = sum(1 for r in male_responses if r >= 4)
    female_agree_count = sum(1 for r in female_responses if r >= 4)
    male_disagree_count = sum(1 for r in male_responses if r <= 2)
    female_disagree_count = sum(1 for r in female_responses if r <= 2)
    
    # Calculate unique couple dynamics
    total_responses = len(male_responses)
    male_positive_ratio = male_agree_count / total_responses if total_responses > 0 else 0
    female_positive_ratio = female_agree_count / total_responses if total_responses > 0 else 0
    couple_optimism = (male_positive_ratio + female_positive_ratio) / 2
    
    # 1. ENHANCED PERSONALIZED ALIGNMENT RECOMMENDATIONS
    if alignment_score < 0.3:
        recommendations.append(f"🚨 CRITICAL ALIGNMENT: Only {int(alignment_score * 100)}% agreement detected - immediate relationship counseling required")
    elif alignment_score < 0.5:
        recommendations.append(f"⚠️ SIGNIFICANT DISAGREEMENT: {int((1-alignment_score) * 100)}% disagreement on key issues - structured communication therapy needed")
    elif alignment_score < 0.7:
        recommendations.append(f"🔄 MODERATE ALIGNMENT: {int(alignment_score * 100)}% agreement - focus on understanding different perspectives")
    else:
        recommendations.append(f"✅ STRONG ALIGNMENT: {int(alignment_score * 100)}% agreement - continue building on shared values and goals")
    
    # 1.5. COUPLE-SPECIFIC OPTIMISM ANALYSIS
    if couple_optimism > 0.7:
        recommendations.append(f"🌟 EXCELLENT HARMONY: {int(couple_optimism * 100)}% positive responses - maintain current healthy communication patterns")
    elif couple_optimism > 0.5:
        recommendations.append(f"😊 GOOD HARMONY: {int(couple_optimism * 100)}% positive responses - good foundation with room for growth")
    elif couple_optimism > 0.3:
        recommendations.append(f"😐 MODERATE HARMONY: {int(couple_optimism * 100)}% positive responses - focus on building shared positive perspectives")
    else:
        recommendations.append(f"😟 CONCERNING HARMONY: Only {int(couple_optimism * 100)}% positive responses - intensive counseling needed to address underlying concerns")
    
    # 2. DYNAMIC CONFLICT-SPECIFIC RECOMMENDATIONS
    if conflict_ratio > 0.5:
        recommendations.append(f"💥 HIGH CONFLICT: {int(conflict_ratio * 100)}% of responses show major disagreement - intensive conflict resolution counseling required")
    elif conflict_ratio > 0.3:
        recommendations.append(f"⚡ MODERATE CONFLICT: {int(conflict_ratio * 100)}% disagreement detected - mediation and communication skills training recommended")
    elif conflict_ratio > 0.1:
        recommendations.append(f"🤝 MINOR CONFLICTS: {int(conflict_ratio * 100)}% disagreement - focus on conflict prevention strategies")
    else:
        recommendations.append(f"🎯 EXCELLENT HARMONY: Only {int(conflict_ratio * 100)}% disagreement - maintain current healthy communication patterns")
    
    # 3. PARTNER-SPECIFIC ANALYSIS (based on average responses)
    # Check for significant differences in partner responses
    avg_difference = abs(male_avg - female_avg)
    if avg_difference > 0.5 and alignment_score < 0.8:
        if male_avg > female_avg:
            recommendations.append(f"👨 PARTNER DIFFERENCES: Male partner shows {male_avg:.1f} vs female {female_avg:.1f} average - ensure balanced decision-making and equal voice")
        else:
            recommendations.append(f"👩 PARTNER DIFFERENCES: Female partner shows {female_avg:.1f} vs male {male_avg:.1f} average - ensure balanced decision-making and equal voice")
    elif avg_difference <= 0.3:
        recommendations.append(f"🤝 BALANCED PARTNERSHIP: Similar response averages ({male_avg:.1f} vs {female_avg:.1f}) - excellent relationship equality")
    
    # 4.5. PARTNER-SPECIFIC RESPONSE PATTERN ANALYSIS
    if male_positive_ratio > 0.7:
        recommendations.append(f"👨 MALE POSITIVE: Male partner shows {int(male_positive_ratio * 100)}% positive responses - excellent engagement and optimism")
    elif male_positive_ratio < 0.3:
        recommendations.append(f"👨 MALE CONCERNS: Male partner shows only {int(male_positive_ratio * 100)}% positive responses - individual counseling recommended")
    
    if female_positive_ratio > 0.7:
        recommendations.append(f"👩 FEMALE POSITIVE: Female partner shows {int(female_positive_ratio * 100)}% positive responses - excellent engagement and optimism")
    elif female_positive_ratio < 0.3:
        recommendations.append(f"👩 FEMALE CONCERNS: Female partner shows only {int(female_positive_ratio * 100)}% positive responses - individual counseling recommended")
    
    # 5. DYNAMIC RISK-LEVEL PERSONALIZED RECOMMENDATIONS
    if risk_level == 'High':
        recommendations.append(f"🔴 HIGH RISK PROFILE: Intensive counseling required - focus on core relationship issues, communication, and conflict resolution")
        if conflict_ratio > 0.4:
            recommendations.append(f"💥 CRISIS INTERVENTION: {int(conflict_ratio * 100)}% conflict rate - immediate mediation or specialized counseling required")
    elif risk_level == 'Medium':
        recommendations.append(f"🟡 MEDIUM RISK PROFILE: Proactive counseling recommended - address identified issues before they escalate into major problems")
    else:
        recommendations.append(f"🟢 LOW RISK PROFILE: Preventive counseling - maintain healthy relationship patterns and continue building strong foundations")
    
    # 6. DYNAMIC CATEGORY-SPECIFIC PERSONALIZED RECOMMENDATIONS
    for category in focus_categories:
        score = category['score']
        name = category['name']
        
        if score > 0.7:
            if 'Marriage' in name:
                recommendations.append(f"💕 CRITICAL MARRIAGE FOCUS: {name} at {int(score * 100)}% - immediate relationship foundation counseling required")
            elif 'Family' in name:
                recommendations.append(f"👶 CRITICAL FAMILY PLANNING: {name} at {int(score * 100)}% - intensive family planning and parenting preparation needed")
            elif 'Health' in name:
                recommendations.append(f"🏥 CRITICAL HEALTH FOCUS: {name} at {int(score * 100)}% - immediate health and wellness counseling required")
        elif score > 0.5:
            if 'Marriage' in name:
                recommendations.append(f"💕 HIGH MARRIAGE PRIORITY: {name} at {int(score * 100)}% - relationship foundation counseling recommended")
            elif 'Family' in name:
                recommendations.append(f"👶 HIGH FAMILY PRIORITY: {name} at {int(score * 100)}% - family planning counseling recommended")
            elif 'Health' in name:
                recommendations.append(f"🏥 HIGH HEALTH PRIORITY: {name} at {int(score * 100)}% - health and wellness counseling recommended")
        elif score > 0.3:
            if 'Marriage' in name:
                recommendations.append(f"💕 MODERATE MARRIAGE FOCUS: {name} at {int(score * 100)}% - relationship development sessions")
            elif 'Family' in name:
                recommendations.append(f"👶 MODERATE FAMILY FOCUS: {name} at {int(score * 100)}% - family planning education")
            elif 'Health' in name:
                recommendations.append(f"🏥 MODERATE HEALTH FOCUS: {name} at {int(score * 100)}% - health awareness sessions")
    
    
    return recommendations

def generate_risk_reasoning(couple_profile, personalized_features, risk_level):
    """Generate specific reasoning for risk level based on actual couple features"""
    reasoning_parts = []
    
    # Age difference analysis
    male_age = couple_profile.get('male_age', 30)
    female_age = couple_profile.get('female_age', 30)
    age_gap = abs(male_age - female_age)
    
    if age_gap > 10:
        reasoning_parts.append(f"Significant age gap ({age_gap} years) between partners")
    elif age_gap > 5:
        reasoning_parts.append(f"Moderate age gap ({age_gap} years) between partners")
    else:
        reasoning_parts.append(f"Minimal age gap ({age_gap} years) between partners")
    
    # Civil status analysis
    civil_status = couple_profile.get('civil_status', 'Single')
    if civil_status == 'Living In':
        years_together = couple_profile.get('years_living_together', 0)
        if years_together > 5:
            reasoning_parts.append(f"Long-term cohabitation ({years_together} years) with established patterns")
        elif years_together > 0:
            reasoning_parts.append(f"Recent cohabitation ({years_together} years) with developing patterns")
    elif civil_status in ['Widowed', 'Separated', 'Divorced']:
        reasoning_parts.append(f"Previous relationship experience ({civil_status}) affecting current dynamics")
    
    # Children analysis
    has_children = couple_profile.get('past_children', False)
    children_count = couple_profile.get('children', 0)
    if has_children and children_count > 0:
        reasoning_parts.append(f"Parenting experience with {children_count} child{'ren' if children_count > 1 else ''}")
    
    # Relationship dynamics analysis
    alignment_score = personalized_features.get('alignment_score', 0.5)
    conflict_ratio = personalized_features.get('conflict_ratio', 0.0)
    
    if alignment_score > 0.7:
        reasoning_parts.append(f"High alignment ({int(alignment_score * 100)}%) in MEAI responses")
    elif alignment_score < 0.4:
        reasoning_parts.append(f"Low alignment ({int(alignment_score * 100)}%) indicating significant disagreements")
    else:
        reasoning_parts.append(f"Moderate alignment ({int(alignment_score * 100)}%) with some areas of agreement")
    
    if conflict_ratio > 0.3:
        reasoning_parts.append(f"High conflict patterns ({int(conflict_ratio * 100)}% disagreement rate)")
    elif conflict_ratio < 0.1:
        reasoning_parts.append(f"Low conflict patterns ({int(conflict_ratio * 100)}% disagreement rate)")
    
    # Education and income compatibility
    education_level = couple_profile.get('education_level', 2)
    income_level = couple_profile.get('income_level', 2)
    
    if abs(education_level - income_level) > 2:
        reasoning_parts.append("Significant education-income mismatch affecting compatibility")
    else:
        reasoning_parts.append("Compatible education and income levels")
    
    # Combine reasoning based on risk level
    if risk_level == 'High':
        return f"High risk assessment based on: {'; '.join(reasoning_parts[:4])}"
    elif risk_level == 'Medium':
        return f"Medium risk assessment based on: {'; '.join(reasoning_parts[:3])}"
    else:
        return f"Low risk assessment based on: {'; '.join(reasoning_parts[:3])}"

def generate_counseling_reasoning(focus_categories, category_scores, ml_confidence):
    """Generate specific reasoning for counseling recommendation based on MEAI categories"""
    reasoning_parts = []
    
    # Analyze specific MEAI categories
    high_priority_categories = [cat for cat in focus_categories if cat['score'] > 0.6]
    moderate_priority_categories = [cat for cat in focus_categories if 0.3 < cat['score'] <= 0.6]
    low_priority_categories = [cat for cat in focus_categories if cat['score'] <= 0.3]
    
    if high_priority_categories:
        category_names = [cat['name'] for cat in high_priority_categories]
        reasoning_parts.append(f"Critical needs in: {', '.join(category_names[:2])}")
    
    if moderate_priority_categories:
        category_names = [cat['name'] for cat in moderate_priority_categories]
        reasoning_parts.append(f"Development areas in: {', '.join(category_names[:2])}")
    
    if low_priority_categories:
        category_names = [cat['name'] for cat in low_priority_categories]
        reasoning_parts.append(f"Strong areas in: {', '.join(category_names[:2])}")
    
    # Add confidence-based reasoning
    if ml_confidence > 0.6:
        reasoning_parts.append(f"High confidence ({int(ml_confidence * 100)}%) in assessment accuracy")
    elif ml_confidence > 0.3:
        reasoning_parts.append(f"Moderate confidence ({int(ml_confidence * 100)}%) in assessment accuracy")
    else:
        reasoning_parts.append(f"Conservative confidence ({int(ml_confidence * 100)}%) in assessment accuracy")
    
    # Combine reasoning
    if len(reasoning_parts) > 3:
        return f"Counseling recommendation based on: {'; '.join(reasoning_parts[:3])}"
    else:
        return f"Counseling recommendation based on: {'; '.join(reasoning_parts)}"

if __name__ == '__main__':
    print("Starting Counseling Topics Service...")
    
    # Load MEAI categories from database
    load_categories_from_db()
    print(f"MEAI Categories: {MEAI_CATEGORIES}")
    
    # Load MEAI questions and sub-questions from database
    load_questions_from_db()
    
    # Load existing models if available
    load_ml_models()
    
    print("Service ready!")
    print("Counseling Topics Models: Available" if all(model is not None for model in ml_models.values()) else "Counseling Topics Models: Training needed")
    print(f"Analysis Method: Random Forest Counseling Topics with {len(MEAI_CATEGORIES)} MEAI categories")
    
    # Heroku compatibility: use PORT environment variable if available
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0' if os.environ.get('PORT') else '127.0.0.1'
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host=host, port=port, debug=debug)
