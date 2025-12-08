
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

# Ensure MEAI questions are loaded on app startup (for Heroku/gunicorn)
@app.before_request
def ensure_questions_loaded():
    """Ensure MEAI questions are loaded before processing requests"""
    global MEAI_QUESTIONS, MEAI_QUESTION_MAPPING, MEAI_CATEGORIES
    if not MEAI_QUESTIONS or len(MEAI_QUESTIONS) == 0:
        print("MEAI_QUESTIONS not loaded, loading from database...")
        if not MEAI_CATEGORIES or len(MEAI_CATEGORIES) == 0:
            load_categories_from_db()
        load_questions_from_db()
        print(f"Loaded {len(MEAI_QUESTIONS)} categories with questions")

def get_db_config():
    """
    Auto-detect local vs remote database configuration
    Returns database connection parameters based on environment
    
    Detection priority:
    1. DB_HOST environment variable (explicit override)
    2. Heroku/production environment variables
    3. Default to localhost for development
    """
    # Check for explicit database host override
    db_host = os.getenv('DB_HOST', '').lower()
    if db_host:
        if db_host in ['localhost', '127.0.0.1']:
            return {
                'host': 'localhost',
                'user': 'root',
                'password': '',
                'database': 'u520834156_DBpmoc25',
                'charset': 'utf8mb4'
            }
        else:
            # Use remote with custom host
            return {
                'host': db_host,
                'user': os.getenv('DB_USER', 'u520834156_userPmoc'),
                'password': os.getenv('DB_PASSWORD', 'NzkN5arIO7@'),
                'database': os.getenv('DB_NAME', 'u520834156_DBpmoc25'),
                'charset': 'utf8mb4'
            }
    
    # Check for environment variable (most reliable)
    env = os.getenv('FLASK_ENV', '').lower()
    is_production = os.getenv('ENVIRONMENT', '').lower() == 'production'
    
    # Check if running on Heroku or production-like environment
    is_heroku = 'DYNO' in os.environ  # Heroku sets DYNO variable
    
    # If explicitly set to production or Heroku, use remote
    if is_production or is_heroku or env == 'production':
        return {
            'host': 'srv1322.hstgr.io',
            'user': 'u520834156_userPmoc',
            'password': 'NzkN5arIO7@',
            'database': 'u520834156_DBpmoc25',
            'charset': 'utf8mb4'
        }
    
    # Default to local for development (localhost)
    return {
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'database': 'u520834156_DBpmoc25',
        'charset': 'utf8mb4'
    }

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
        
        # Count conflicts using same logic as actual_disagree_ratio
        # This ensures conflict_ratio matches the disagreement calculation
        question_disagree = 1 if (male_resp == 2 or female_resp == 2) else 0
        if difference >= 2:
            partner_disagree = 1.0
        elif difference == 1:
            partner_disagree = 0.5
        else:
            partner_disagree = 0.0
        
        # Use max to avoid double counting (same as actual_disagree_ratio)
        weighted_conflict_sum += max(question_disagree, partner_disagree)
    
    alignment_score = alignment_score / total_questions if total_questions > 0 else 0.5
    # Add weighted neutrals (30% weight) to match actual_disagree_ratio
    neutral_count = sum(1 for i in range(total_questions) 
                       if (male_responses[i] == 3 or female_responses[i] == 3))
    conflict_ratio = ((weighted_conflict_sum + (neutral_count * 0.3)) / total_questions) if total_questions > 0 else 0
    
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
        'total_conflicts': conflict_count,
        # Category-specific alignments (4 features, one per MEAI category)
        'category_alignments': category_alignments
        # REMOVED: male_avg_response, female_avg_response, male_agree_ratio, male_disagree_ratio, female_agree_ratio, female_disagree_ratio
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
    
    # REMOVED: children validation (children feature removed)
    
    # Validate questionnaire responses
    if questionnaire_responses:
        # Calculate expected count from MEAI_QUESTIONS structure (more reliable than MEAI_QUESTION_MAPPING)
        expected_count = None
        received_count = len(questionnaire_responses)
        
        if MEAI_QUESTIONS and len(MEAI_QUESTIONS) > 0:
            expected_count = 0
            total_questions_in_structure = 0
            for cat_id, cat_questions in MEAI_QUESTIONS.items():
                for q_id, q_data in cat_questions.items():
                    total_questions_in_structure += 1
                    if q_data.get('sub_questions') and len(q_data['sub_questions']) > 0:
                        # Question has sub-questions, count only the sub-questions
                        expected_count += len(q_data['sub_questions'])
                    elif not q_data.get('sub_questions') or len(q_data.get('sub_questions', [])) == 0:
                        # Standalone question, count it
                        expected_count += 1
            print(f"DEBUG - MEAI_QUESTIONS: {len(MEAI_QUESTIONS)} categories, {total_questions_in_structure} questions, {expected_count} answerable items")
            print(f"DEBUG - Calculated expected_count from MEAI_QUESTIONS: {expected_count}")
            
            # If calculated count is 0 or very small, something is wrong - don't validate
            if expected_count == 0 or expected_count < 10:
                print(f"WARNING: Calculated expected_count ({expected_count}) seems wrong, skipping validation")
                expected_count = None
        elif MEAI_QUESTION_MAPPING and len(MEAI_QUESTION_MAPPING) > 0:
            # Fallback to mapping if questions structure not available
            mapping_count = len(MEAI_QUESTION_MAPPING)
            print(f"DEBUG - MEAI_QUESTION_MAPPING has {mapping_count} entries")
            
            # Only use mapping if it has a reasonable number of entries (should be 59, not 4)
            if mapping_count >= 50:  # Reasonable threshold for answerable questions
                expected_count = mapping_count
                print(f"DEBUG - Using MEAI_QUESTION_MAPPING count: {expected_count}")
            else:
                print(f"WARNING: MEAI_QUESTION_MAPPING count ({mapping_count}) seems too low, skipping validation")
                expected_count = None
        
        # If we still don't have an expected count, skip validation (allow any count)
        # This prevents false validation errors when questions aren't loaded
        # Also, if we receive 59 responses (the known correct count), accept it even if expected_count is wrong
        if expected_count is not None and received_count != expected_count:
            # Special case: if we receive 59 responses (known correct count), accept it
            # This handles cases where MEAI_QUESTIONS isn't loaded correctly
            if received_count == 59:
                print(f"INFO: Received 59 responses (known correct count), accepting despite expected_count={expected_count}")
            else:
                errors.append(f"Expected {expected_count} responses, got {received_count}")
                print(f"ERROR - Validation failed: Expected {expected_count}, got {received_count}")
        elif expected_count is None:
            print(f"INFO: Could not determine expected count, accepting {received_count} responses (validation skipped)")
            # If we receive 59, that's the known correct count, so log it
            if received_count == 59:
                print(f"INFO: 59 responses received - this matches the expected number of answerable questions")
        
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
        
        # Database connection - Auto-detect local vs remote
        db_config = get_db_config()
        conn = pymysql.connect(**db_config)
        
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
        
        # Database connection - Auto-detect local vs remote
        db_config = get_db_config()
        conn = pymysql.connect(**db_config)
        
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
    # Allocate: 33% Low, 34% Medium, 33% High (more balanced distribution)
    num_low = int(num_couples * 0.33)
    num_medium = int(num_couples * 0.34)
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
        # Target ranges adjusted to match thresholds: High >0.35, Medium >0.20, Low ≤0.20
        if i < num_low:
            target_risk = 'Low'
            target_disagree_ratio = np.random.uniform(0.0, 0.20)  # Low risk: 0-20% disagree
        elif i < num_low + num_medium:
            target_risk = 'Medium'
            target_disagree_ratio = np.random.uniform(0.20, 0.35)  # Medium risk: 20-35% disagree
        else:
            target_risk = 'High'
            target_disagree_ratio = np.random.uniform(0.35, 0.60)  # High risk: 35-60% disagree
        
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
        
        # Use same thresholds as training: High >0.35, Medium >0.20, Low ≤0.20
        if disagree_ratio > 0.35:
            risk_level = 'High'
        elif disagree_ratio > 0.20:
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
        
        # CRITICAL: Generate separate male_responses and female_responses
        # For synthetic data, we'll generate similar but slightly different responses
        # to simulate real couple dynamics
        male_responses = questionnaire_responses.copy()
        female_responses = questionnaire_responses.copy()
        
        # Add some variation between partners (simulate real couple differences)
        for j in range(len(questionnaire_responses)):
            # 30% chance of partner disagreement (difference of 1)
            if np.random.random() < 0.3:
                if np.random.random() < 0.5:
                    # Male slightly more positive/negative
                    male_responses[j] = max(2, min(4, male_responses[j] + np.random.choice([-1, 1])))
                else:
                    # Female slightly more positive/negative
                    female_responses[j] = max(2, min(4, female_responses[j] + np.random.choice([-1, 1])))
            # 10% chance of significant disagreement (difference of 2)
            elif np.random.random() < 0.1:
                if np.random.random() < 0.5:
                    male_responses[j] = max(2, min(4, male_responses[j] + np.random.choice([-2, 2])))
                else:
                    female_responses[j] = max(2, min(4, female_responses[j] + np.random.choice([-2, 2])))
        
        data.append({
            'male_age': male_age,
            'female_age': female_age,
            'civil_status': civil_status,
            'years_living_together': years_living_together,
            'past_children': has_past_children,
            'children': children,
            'education_level': education_level,
            'income_level': income_level,
            'questionnaire_responses': questionnaire_responses,  # Keep for backward compatibility
            'male_responses': male_responses,  # CRITICAL: Separate male responses (59 features)
            'female_responses': female_responses,  # CRITICAL: Separate female responses (59 features)
            'risk_level': risk_level,
            'category_scores': category_scores
        })
    
    return data

def generate_synthetic_data(num_couples=500):
    """Generate realistic synthetic couple data for training (fallback method)"""
    np.random.seed(42)
    
    data = []
    
    # Ensure we generate samples for all 3 risk classes
    # Allocate: 33% Low, 34% Medium, 33% High (more balanced distribution)
    num_low = int(num_couples * 0.33)
    num_medium = int(num_couples * 0.34)
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
        # Target ranges adjusted to match thresholds: High >0.35, Medium >0.20, Low ≤0.20
        if i < num_low:
            target_risk = 'Low'
            target_disagree_ratio = np.random.uniform(0.0, 0.20)  # Low risk: 0-20% disagree
        elif i < num_low + num_medium:
            target_risk = 'Medium'
            target_disagree_ratio = np.random.uniform(0.20, 0.35)  # Medium risk: 20-35% disagree
        else:
            target_risk = 'High'
            target_disagree_ratio = np.random.uniform(0.35, 0.60)  # High risk: 35-60% disagree
        
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
        # Use same thresholds as training: High >0.35, Medium >0.20, Low ≤0.20
        disagree_count = sum(1 for r in questionnaire_responses if r == 2)
        disagree_ratio = disagree_count / len(questionnaire_responses)
        
        if disagree_ratio > 0.35:
            risk_level = 'High'
        elif disagree_ratio > 0.20:
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
        
        # CRITICAL: Generate separate male_responses and female_responses
        # For synthetic data, we'll generate similar but slightly different responses
        # to simulate real couple dynamics
        male_responses = questionnaire_responses.copy()
        female_responses = questionnaire_responses.copy()
        
        # Add some variation between partners (simulate real couple differences)
        for j in range(len(questionnaire_responses)):
            # 30% chance of partner disagreement (difference of 1)
            if np.random.random() < 0.3:
                if np.random.random() < 0.5:
                    # Male slightly more positive/negative
                    male_responses[j] = max(2, min(4, male_responses[j] + np.random.choice([-1, 1])))
                else:
                    # Female slightly more positive/negative
                    female_responses[j] = max(2, min(4, female_responses[j] + np.random.choice([-1, 1])))
            # 10% chance of significant disagreement (difference of 2)
            elif np.random.random() < 0.1:
                if np.random.random() < 0.5:
                    male_responses[j] = max(2, min(4, male_responses[j] + np.random.choice([-2, 2])))
                else:
                    female_responses[j] = max(2, min(4, female_responses[j] + np.random.choice([-2, 2])))
        
        data.append({
            'male_age': male_age,
            'female_age': female_age,
            'civil_status': civil_status,
            'years_living_together': years_living_together,
            'past_children': has_past_children,
            'children': children,
            'education_level': education_level,
            'income_level': income_level,
            'questionnaire_responses': questionnaire_responses.tolist(),  # Keep for backward compatibility
            'male_responses': male_responses.tolist() if isinstance(male_responses, np.ndarray) else male_responses,  # CRITICAL: Separate male responses (59 features)
            'female_responses': female_responses.tolist() if isinstance(female_responses, np.ndarray) else female_responses,  # CRITICAL: Separate female responses (59 features)
            'risk_level': risk_level,
            'category_scores': category_scores
        })
    
    return data

def load_real_couples_for_training():
    """Load real couples from database for ML training"""
    try:
        import pymysql
        
        # Database connection - Auto-detect local vs remote
        db_config = get_db_config()
        conn = pymysql.connect(**db_config)
        
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
            
            # CRITICAL: Build separate male_responses and female_responses arrays
            # This matches the structure used in analysis (118 features: 59 male + 59 female)
            male_responses_array = []
            female_responses_array = []
            questionnaire_responses = []  # Still build combined for backward compatibility
            
            # Build responses in the order defined by MEAI_QUESTIONS structure
            # Iterate through categories and questions in the same order
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
                            
                            # Get separate male and female responses from response_map
                            male_resp = response_map.get(key, {}).get('male', 3)  # Default to neutral
                            female_resp = response_map.get(key, {}).get('female', 3)  # Default to neutral
                            
                            # Add to separate arrays
                            male_responses_array.append(male_resp if male_resp is not None else 3)
                            female_responses_array.append(female_resp if female_resp is not None else 3)
                            
                            # Also build combined questionnaire_responses (for backward compatibility)
                            if male_resp is not None and female_resp is not None:
                                # Consider partner disagreements as indicators of conflict
                                if abs(male_resp - female_resp) >= 2:  # Significant disagreement
                                    avg_resp = min(male_resp, female_resp)
                                else:
                                    avg_resp = round((male_resp + female_resp) / 2)
                                questionnaire_responses.append(avg_resp)
                            elif male_resp is not None:
                                questionnaire_responses.append(male_resp)
                            elif female_resp is not None:
                                questionnaire_responses.append(female_resp)
                            else:
                                questionnaire_responses.append(3)  # Default to neutral
                    else:
                        # Standalone question - no sub-question
                        key = (cat_id, q_id, None)
                        
                        # Get separate male and female responses from response_map
                        male_resp = response_map.get(key, {}).get('male', 3)
                        female_resp = response_map.get(key, {}).get('female', 3)
                        
                        # Add to separate arrays
                        male_responses_array.append(male_resp if male_resp is not None else 3)
                        female_responses_array.append(female_resp if female_resp is not None else 3)
                        
                        # Also build combined questionnaire_responses
                        if male_resp is not None and female_resp is not None:
                            if abs(male_resp - female_resp) >= 2:
                                avg_resp = min(male_resp, female_resp)
                            else:
                                avg_resp = round((male_resp + female_resp) / 2)
                            questionnaire_responses.append(avg_resp)
                        elif male_resp is not None:
                            questionnaire_responses.append(male_resp)
                        elif female_resp is not None:
                            questionnaire_responses.append(female_resp)
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
            agree_count = sum(1 for r in questionnaire_responses if r == 4)
            # Count neutral as partial disagreement (they might indicate unresolved issues)
            weighted_disagree_count = disagree_count + (neutral_count * 0.3)
            disagree_ratio = weighted_disagree_count / len(questionnaire_responses) if len(questionnaire_responses) > 0 else 0
            
            # DIAGNOSTIC: Log risk calculation details for real couples
            print(f"DEBUG - Real Couple Risk Calculation:")
            print(f"  Total responses: {len(questionnaire_responses)}")
            print(f"  Agree (4): {agree_count}, Neutral (3): {neutral_count}, Disagree (2): {disagree_count}")
            print(f"  Weighted disagree count: {weighted_disagree_count:.2f}")
            print(f"  Disagree ratio: {disagree_ratio:.3f} ({disagree_ratio*100:.1f}%)")
            
            # BALANCED THRESHOLDS: Optimal for counseling and prevention
            # High >0.35 (35%), Medium >0.20 (20%), Low ≤0.20 (20%)
            # This balances sensitivity (catching couples who need help) with specificity (avoiding over-classification)
            if disagree_ratio > 0.35:  # 35%+ weighted disagreement = High risk
                risk_level = 'High'
                print(f"  → Risk Level: HIGH (disagree_ratio {disagree_ratio:.3f} > 0.35)")
            elif disagree_ratio > 0.20:  # 20-35% weighted disagreement = Medium risk
                risk_level = 'Medium'
                print(f"  → Risk Level: MEDIUM (disagree_ratio {disagree_ratio:.3f} > 0.20)")
            else:  # ≤20% weighted disagreement = Low risk
                risk_level = 'Low'
                print(f"  → Risk Level: LOW (disagree_ratio {disagree_ratio:.3f} <= 0.20)")
            
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
                'questionnaire_responses': questionnaire_responses,  # Keep for backward compatibility
                'male_responses': male_responses_array,  # CRITICAL: Separate male responses (59 features)
                'female_responses': female_responses_array,  # CRITICAL: Separate female responses (59 features)
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
        
        # DIAGNOSTIC: Check risk level distribution
        from collections import Counter
        real_risk_dist = Counter([c['risk_level'] for c in real_couples_data])
        synthetic_risk_dist = Counter([c['risk_level'] for c in synthetic_data])
        total_risk_dist = Counter([c['risk_level'] for c in data])
        
        print(f"\n=== RISK LEVEL DISTRIBUTION ===")
        print(f"Real couples: {dict(real_risk_dist)}")
        print(f"Synthetic couples: {dict(synthetic_risk_dist)}")
        print(f"Total training data: {dict(total_risk_dist)}")
        print(f"===============================\n")
        
        # WARNING: If all real couples are Low Risk, this might bias the model
        if len(real_risk_dist) == 1 and 'Low' in real_risk_dist:
            print(f"⚠️  WARNING: All {len(real_couples_data)} real couples are classified as LOW RISK!")
            print(f"⚠️  This may bias the model towards predicting Low Risk for similar couples.")
            print(f"⚠️  Consider reviewing the risk calculation thresholds or couple responses.\n")
    
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
        
        # NEW: Encode employment status (use male partner's employment status)
        employment_status = row.get('employment_status', 'Unemployed')
        if employment_status == 'Employed':
            employment_encoded = 1
        elif employment_status == 'Self-employed':
            employment_encoded = 2
        else:  # Unemployed or unknown
            employment_encoded = 0
        
        # Combine all features (same structure as analysis)
        features = [
            row['male_age'],
            row['female_age'],
            age_gap,
            row['years_living_together'],
            row['education_level'],
            row['income_level'],
            education_income_diff,
            is_single,
            is_living_in,
            is_separated_divorced,
            employment_encoded  # NEW: Employment status
            # REMOVED: children feature
        ]
        
        # CRITICAL: Use separate male_responses + female_responses (118 features: 59 + 59)
        # This is REQUIRED for training - no fallback to questionnaire_responses
        # Training data must include separate male_responses and female_responses
        if 'male_responses' not in row or 'female_responses' not in row:
            print(f"ERROR - Training data missing male_responses or female_responses!")
            print(f"ERROR - Available keys: {list(row.keys())}")
            raise ValueError("Training data must include separate male_responses and female_responses (from respondent field)")
        
        male_responses = row['male_responses']
        female_responses = row['female_responses']
        
        # Validate arrays exist and have correct length
        if not male_responses or len(male_responses) == 0:
            raise ValueError(f"male_responses is empty for training sample")
        if not female_responses or len(female_responses) == 0:
            raise ValueError(f"female_responses is empty for training sample")
        
        expected_count = len(MEAI_QUESTION_MAPPING) if MEAI_QUESTION_MAPPING else 59
        if len(male_responses) != expected_count:
            raise ValueError(f"male_responses length ({len(male_responses)}) does not match expected ({expected_count})")
        if len(female_responses) != expected_count:
            raise ValueError(f"female_responses length ({len(female_responses)}) does not match expected ({expected_count})")
        
        # Add separate responses (118 features total: 59 male + 59 female)
        if isinstance(male_responses, np.ndarray):
            features.extend(male_responses.tolist())
        else:
            features.extend(male_responses)
        if isinstance(female_responses, np.ndarray):
            features.extend(female_responses.tolist())
        else:
            features.extend(female_responses)
        
        print(f"DEBUG - Training: Added {len(male_responses)} male + {len(female_responses)} female = {len(male_responses) + len(female_responses)} response features")
        
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
        
        # Add personalized features to match analysis structure (6 features)
        personalized_features = [
            alignment_score,
            conflict_ratio,
            # Category-specific alignments (4 features, one per MEAI category)
            *category_alignments
            # REMOVED: male_avg_response, female_avg_response, male_agree_ratio, male_disagree_ratio, female_agree_ratio, female_disagree_ratio
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
                
                # CRITICAL: Generate separate male_responses and female_responses
                # Start with questionnaire_responses, then add variation
                male_responses_synth = questionnaire_responses.copy()
                female_responses_synth = questionnaire_responses.copy()
                
                # Add variation between partners (simulate real couple differences)
                for j in range(len(questionnaire_responses)):
                    # 30% chance of partner disagreement (difference of 1)
                    if np.random.random() < 0.3:
                        if np.random.random() < 0.5:
                            male_responses_synth[j] = max(2, min(4, male_responses_synth[j] + np.random.choice([-1, 1])))
                        else:
                            female_responses_synth[j] = max(2, min(4, female_responses_synth[j] + np.random.choice([-1, 1])))
                    # 10% chance of significant disagreement (difference of 2)
                    elif np.random.random() < 0.1:
                        if np.random.random() < 0.5:
                            male_responses_synth[j] = max(2, min(4, male_responses_synth[j] + np.random.choice([-2, 2])))
                        else:
                            female_responses_synth[j] = max(2, min(4, female_responses_synth[j] + np.random.choice([-2, 2])))
                
                # Build features (11 demographic features - REMOVED: children)
                features = [
                    male_age, female_age, age_gap, years_living_together,
                    education_level, income_level, education_income_diff,
                    is_single, is_living_in, is_separated_divorced,
                    0  # employment_encoded (default to Unemployed for synthetic)
                ]
                
                # Add separate male_responses and female_responses (118 features: 59 + 59)
                features.extend(male_responses_synth)
                features.extend(female_responses_synth)
                
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
                    # REMOVED: male_agree_ratio, male_disagree_ratio, female_agree_ratio, female_disagree_ratio
                
                personalized_features = [
                    alignment_score,
                    conflict_ratio,
                    *category_alignments  # 4 category alignments
                    # REMOVED: male_avg_response, female_avg_response, agree/disagree ratios
                ]
                
                features.extend(personalized_features)
                
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
    class_dist_before = np.bincount(y_risk)
    print(f"Class distribution before SMOTE: {class_dist_before}")
    
    # Check if SMOTE is needed (only apply if significant imbalance exists)
    # Synthetic data is already balanced (33% Low, 34% Medium, 33% High)
    # SMOTE is only needed if real couples create severe imbalance
    max_class = np.max(class_dist_before)
    min_class = np.min(class_dist_before[class_dist_before > 0])  # Ignore zero classes
    imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
    
    # Only apply SMOTE if imbalance ratio > 1.5 (50% difference between classes)
    # This prevents unnecessary processing when data is already reasonably balanced
    should_apply_smote = imbalance_ratio > 1.5 and IMBALANCED_LEARN_AVAILABLE
    
    if not IMBALANCED_LEARN_AVAILABLE:
        print("Skipping SMOTE - imbalanced-learn not available")
    elif not should_apply_smote:
        print(f"Skipping SMOTE - data is already reasonably balanced (imbalance ratio: {imbalance_ratio:.2f} < 1.5)")
        print(f"  - Synthetic data provides balanced distribution (33/34/33)")
        print(f"  - Current distribution is acceptable for training")
    else:
        # Update progress: Applying SMOTE
        with training_lock:
            training_status['progress'] = 40
            training_status['message'] = 'Applying SMOTE for class balancing...'
        
        print(f"Applying SMOTE to combined dataset (imbalance ratio: {imbalance_ratio:.2f} > 1.5)")
        print(f"  - Real couples created imbalance, synthetic data is already balanced")
        print(f"  - SMOTE will balance overall distribution while preserving real data patterns")
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
    
    # Save to files - use ml_model folder (where this script is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        risk_model_path = os.path.join(script_dir, 'risk_model.pkl')
        category_model_path = os.path.join(script_dir, 'category_model.pkl')
        risk_encoder_path = os.path.join(script_dir, 'risk_encoder.pkl')
        
        with open(risk_model_path, 'wb') as f:
            pickle.dump(risk_model, f)
        
        with open(category_model_path, 'wb') as f:
            pickle.dump(category_model, f)
        
        with open(risk_encoder_path, 'wb') as f:
            pickle.dump(risk_encoder, f)
        
        # Update progress: Complete
        with training_lock:
            training_status['progress'] = 95
            training_status['message'] = 'Models saved successfully!'
        
        print(f"ML models trained and saved successfully to {script_dir}")
        return True
    except Exception as e:
        print(f"Error saving models: {e}")
        # Models are still in memory, so training was successful
        # Just couldn't save to disk
        return True  # Still return True since models are loaded in memory

def load_ml_models():
    """Load pre-trained ML models from ml_model folder"""
    # Get the directory where this script is located (ml_model folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    risk_model_path = os.path.join(script_dir, 'risk_model.pkl')
    category_model_path = os.path.join(script_dir, 'category_model.pkl')
    risk_encoder_path = os.path.join(script_dir, 'risk_encoder.pkl')
    
    try:
        if os.path.exists(risk_model_path):
            with open(risk_model_path, 'rb') as f:
                ml_models['risk_model'] = pickle.load(f)
            print(f"Loaded risk_model.pkl from {script_dir}")
        else:
            print(f"Warning: risk_model.pkl not found in {script_dir}")
        
        if os.path.exists(category_model_path):
            with open(category_model_path, 'rb') as f:
                ml_models['category_model'] = pickle.load(f)
            print(f"Loaded category_model.pkl from {script_dir}")
        else:
            print(f"Warning: category_model.pkl not found in {script_dir}")
        
        if os.path.exists(risk_encoder_path):
            with open(risk_encoder_path, 'rb') as f:
                ml_models['risk_encoder'] = pickle.load(f)
            print(f"Loaded risk_encoder.pkl from {script_dir}")
        else:
            print(f"Warning: risk_encoder.pkl not found in {script_dir}")
        
        if ml_models.get('risk_model') and ml_models.get('category_model') and ml_models.get('risk_encoder'):
            print("All ML models loaded successfully")
            return True
        else:
            print("Error: Not all models were loaded")
            return False
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
        
        # CRITICAL DEBUG: Log raw received data structure
        if data:
            print(f"DEBUG - Raw data keys received: {list(data.keys())}")
            print(f"DEBUG - Raw data has 'male_responses' key: {'male_responses' in data}")
            print(f"DEBUG - Raw data has 'female_responses' key: {'female_responses' in data}")
            if 'male_responses' in data:
                print(f"DEBUG - Raw male_responses value type: {type(data['male_responses'])}, length: {len(data['male_responses']) if isinstance(data['male_responses'], (list, tuple)) else 'N/A'}")
            if 'female_responses' in data:
                print(f"DEBUG - Raw female_responses value type: {type(data['female_responses'])}, length: {len(data['female_responses']) if isinstance(data['female_responses'], (list, tuple)) else 'N/A'}")
        
        # Extract couple profile with conditional field handling
        couple_profile = {
            'male_age': data.get('male_age', 30),
            'female_age': data.get('female_age', 30),
            'civil_status': data.get('civil_status', 'Single'),
            'years_living_together': data.get('years_living_together', 0),  # Only for "Living In" status
            'education_level': data.get('education_level', 2),
            'income_level': data.get('income_level', 2),
            'employment_status': data.get('employment_status', 'Unemployed')  # NEW: Employment status
            # REMOVED: past_children, children
        }
        
        # Handle conditional fields based on civil status
        if couple_profile['civil_status'] != 'Living In':
            couple_profile['years_living_together'] = 0
        
        # Extract questionnaire responses (dynamic count based on actual questions)
        total_questions = len(MEAI_QUESTION_MAPPING) if MEAI_QUESTION_MAPPING else 31  # Fallback to 31
        questionnaire_responses = data.get('questionnaire_responses', [3] * total_questions)
        
        # PERSONALIZED FEATURES: Extract relationship dynamics
        personalized_features = data.get('personalized_features', {})
        male_responses = data.get('male_responses', [])
        female_responses = data.get('female_responses', [])
        
        # CRITICAL DEBUG: Check what we actually received
        print(f"DEBUG - Received data keys: {list(data.keys())}")
        print(f"DEBUG - male_responses type: {type(male_responses)}, length: {len(male_responses) if isinstance(male_responses, (list, tuple)) else 'N/A'}")
        print(f"DEBUG - female_responses type: {type(female_responses)}, length: {len(female_responses) if isinstance(female_responses, (list, tuple)) else 'N/A'}")
        
        # DEBUG: Log what we received
        print(f"DEBUG - Received male_responses: {len(male_responses) if male_responses else 0} items")
        print(f"DEBUG - Received female_responses: {len(female_responses) if female_responses else 0} items")
        print(f"DEBUG - Received questionnaire_responses: {len(questionnaire_responses) if questionnaire_responses else 0} items")
        
        # CRITICAL: REQUIRE male_responses and female_responses from respondent field
        # These MUST come from the couple_responses table with respondent='male' or 'female'
        if not male_responses or len(male_responses) == 0:
            print(f"ERROR - male_responses is empty or None. Type: {type(male_responses)}, Value: {male_responses}")
            return jsonify({
                'status': 'error',
                'message': 'male_responses is required and must not be empty. Data must come from couple_responses table with respondent="male".'
            }), 400
            
        if not female_responses or len(female_responses) == 0:
            print(f"ERROR - female_responses is empty or None. Type: {type(female_responses)}, Value: {female_responses}")
            return jsonify({
                'status': 'error',
                'message': 'female_responses is required and must not be empty. Data must come from couple_responses table with respondent="female".'
            }), 400
        
        # Validate that arrays are lists/tuples
        if not isinstance(male_responses, (list, tuple)):
            return jsonify({
                'status': 'error',
                'message': f'male_responses must be a list/array, got {type(male_responses)}'
            }), 400
            
        if not isinstance(female_responses, (list, tuple)):
            return jsonify({
                'status': 'error',
                'message': f'female_responses must be a list/array, got {type(female_responses)}'
            }), 400
        
        # Validate that arrays have the expected length (should be 59)
        expected_count = len(MEAI_QUESTION_MAPPING) if MEAI_QUESTION_MAPPING else 59
        if len(male_responses) != expected_count:
            print(f"ERROR - male_responses length ({len(male_responses)}) does not match expected ({expected_count})")
            return jsonify({
                'status': 'error',
                'message': f'male_responses must have {expected_count} items (one per answerable question), got {len(male_responses)}'
            }), 400
            
        if len(female_responses) != expected_count:
            print(f"ERROR - female_responses length ({len(female_responses)}) does not match expected ({expected_count})")
            return jsonify({
                'status': 'error',
                'message': f'female_responses must have {expected_count} items (one per answerable question), got {len(female_responses)}'
            }), 400
        
        # Validate that arrays match each other in length
        if len(male_responses) != len(female_responses):
            return jsonify({
                'status': 'error',
                'message': f'male_responses ({len(male_responses)} items) and female_responses ({len(female_responses)} items) must have the same length'
            }), 400
        
        # CRITICAL: Verify arrays are not all zeros or all the same value (data quality check)
        if all(r == 0 for r in male_responses) or all(r == male_responses[0] for r in male_responses if len(male_responses) > 0):
            print(f"WARNING - male_responses appears to have low variance (all values are {male_responses[0] if len(male_responses) > 0 else 'N/A'})")
        if all(r == 0 for r in female_responses) or all(r == female_responses[0] for r in female_responses if len(female_responses) > 0):
            print(f"WARNING - female_responses appears to have low variance (all values are {female_responses[0] if len(female_responses) > 0 else 'N/A'})")
        
        print(f"DEBUG - Validation passed: male_responses={len(male_responses)} items, female_responses={len(female_responses)} items")
        
        # Calculate actual expected count for debugging
        if MEAI_QUESTIONS:
            actual_expected = 0
            for cat_questions in MEAI_QUESTIONS.values():
                for q_data in cat_questions.values():
                    if q_data.get('sub_questions'):
                        actual_expected += len(q_data['sub_questions'])
                    else:
                        actual_expected += 1
            print(f"DEBUG - MEAI_QUESTIONS loaded: {len(MEAI_QUESTIONS)} categories, {actual_expected} answerable questions")
        else:
            print(f"DEBUG - MEAI_QUESTIONS not loaded!")
        print(f"DEBUG - MEAI_QUESTION_MAPPING size: {len(MEAI_QUESTION_MAPPING)}")
        
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
        
        # NEW: Encode employment status (use male partner's employment status)
        # Employed=1, Self-employed=2, Unemployed=0
        employment_status = couple_profile.get('employment_status', 'Unemployed')
        if employment_status == 'Employed':
            employment_encoded = 1
        elif employment_status == 'Self-employed':
            employment_encoded = 2
        else:  # Unemployed or unknown
            employment_encoded = 0
        
        # Prepare features for ML models
        # FEATURE BREAKDOWN (Total: 135 features):
        #   1. Demographic features: 11
        #      - male_age, female_age, age_gap, years_living_together
        #      - education_level, income_level, education_income_diff
        #      - is_single, is_living_in, is_separated_divorced, employment_encoded
        #   2. Questionnaire responses: 118 (59 male + 59 female)
        #      - male_responses: 59 features (from respondent='male' in couple_responses)
        #      - female_responses: 59 features (from respondent='female' in couple_responses)
        #   3. Personalized features: 6
        #      - alignment_score, conflict_ratio
        #      - category_alignments: 4 features (one per MEAI category)
        
        # Basic demographic features (11 features)
        features = [
            couple_profile['male_age'],
            couple_profile['female_age'],
            age_gap,
            couple_profile['years_living_together'],  # 0 for non-Living In couples
            couple_profile['education_level'],
            couple_profile['income_level'],
            education_income_diff,
            is_single,
            is_living_in,
            is_separated_divorced,
            employment_encoded  # NEW: Employment status
            # REMOVED: children feature
        ]
        
        print(f"DEBUG - Step 1: Added {len(features)} demographic features")
        
        # CRITICAL: Use separate male_responses + female_responses (118 features total)
        # This gives us 59 + 59 = 118 features instead of 59 combined
        # REQUIRED: male_responses and female_responses must come from respondent field
        # We already validated they exist above, so we can safely use them here
        print(f"DEBUG - Step 2: Using male_responses ({len(male_responses)} items) and female_responses ({len(female_responses)} items) from respondent field")
        print(f"DEBUG -   male_responses first 3: {male_responses[:3] if len(male_responses) >= 3 else male_responses}")
        print(f"DEBUG -   female_responses first 3: {female_responses[:3] if len(female_responses) >= 3 else female_responses}")
        
        # CRITICAL: Always use separate male_responses + female_responses (118 features total)
        # This is REQUIRED - no fallback to questionnaire_responses
        features.extend(male_responses)
        features.extend(female_responses)
        
        print(f"DEBUG - Step 2: After adding responses, total features = {len(features)} (11 demographic + {len(male_responses)} male + {len(female_responses)} female)")
        
        # Add personalized features (6 features: alignment_score, conflict_ratio, 4 category_alignments)
        personalized_feature_values = [
            personalized_features.get('alignment_score', 0.5),
            personalized_features.get('conflict_ratio', 0.0),
            # Category-specific alignments (4 features, one per MEAI category)
            *personalized_features.get('category_alignments', [0.5, 0.5, 0.5, 0.5])
            # REMOVED: male_avg_response, female_avg_response, male_agree_ratio, male_disagree_ratio, female_agree_ratio, female_disagree_ratio
        ]
        
        features.extend(personalized_feature_values)
        
        # CRITICAL: Verify feature count
        expected_features = 11 + len(male_responses) + len(female_responses) + 6  # 11 demographic + 118 responses + 6 personalized
        actual_features = len(features)
        
        print(f"DEBUG - Feature count breakdown:")
        print(f"DEBUG -   Demographic features: 11")
        print(f"DEBUG -   Male responses: {len(male_responses)}")
        print(f"DEBUG -   Female responses: {len(female_responses)}")
        print(f"DEBUG -   Personalized features: 6")
        print(f"DEBUG -   Expected total: {expected_features}")
        print(f"DEBUG -   Actual total: {actual_features}")
        
        if actual_features != expected_features:
            print(f"ERROR - Feature count mismatch! Expected {expected_features}, got {actual_features}")
            print(f"ERROR - This suggests male_responses or female_responses are not being used correctly")
        
        features_array = np.array(features).reshape(1, -1)
        print(f"Analysis with {len(features)} features: {features_array.shape}")
        
        # HYBRID APPROACH: Calculate actual risk level from disagreement ratio AND use ML prediction
        # This helps catch cases where the model might be biased
        # Calculate actual disagreement ratio from male/female responses (more accurate)
        # Count disagreements: when partners disagree with the question OR when partners disagree with each other
        total_questions = min(len(male_responses), len(female_responses))
        partner_disagree_count = 0  # When partners disagree with each other
        question_disagree_count = 0  # When either partner disagrees with the question (response = 2)
        neutral_count = 0
        
        for i in range(total_questions):
            male_resp = male_responses[i] if i < len(male_responses) else 3
            female_resp = female_responses[i] if i < len(female_responses) else 3
            
            # Count when either partner disagrees with the question
            if male_resp == 2 or female_resp == 2:
                question_disagree_count += 1
            
            # Count when partners disagree with each other (significant difference)
            if abs(male_resp - female_resp) >= 2:  # Significant disagreement between partners
                partner_disagree_count += 1
            elif abs(male_resp - female_resp) == 1:  # Minor disagreement
                partner_disagree_count += 0.5
            
            # Count neutrals (either partner is neutral)
            if male_resp == 3 or female_resp == 3:
                neutral_count += 1
        
        # Combined disagreement: question disagreements + partner disagreements + weighted neutrals
        # Use the maximum of question disagreement or partner disagreement to avoid double counting
        total_disagree_count = max(question_disagree_count, partner_disagree_count) + (neutral_count * 0.3)
        actual_disagree_ratio = total_disagree_count / total_questions if total_questions > 0 else 0
        
        # Calculate actual risk level using same thresholds as training
        # Thresholds: High >0.35 (35%), Medium >0.20 (20%), Low ≤0.20 (20%)
        if actual_disagree_ratio > 0.35:
            actual_risk_level = 'High'
        elif actual_disagree_ratio > 0.20:
            actual_risk_level = 'Medium'
        else:
            actual_risk_level = 'Low'
        
        print(f"DEBUG - Actual Risk Calculation:")
        print(f"  Question disagreements: {question_disagree_count}, Partner disagreements: {partner_disagree_count:.1f}, Neutrals: {neutral_count}")
        print(f"  Total weighted disagree count: {total_disagree_count:.2f}")
        print(f"  Weighted disagree ratio: {actual_disagree_ratio:.3f} ({actual_disagree_ratio*100:.1f}%)")
        print(f"  Actual risk level: {actual_risk_level}")
        
        # Predict risk level using ML model
        if ml_models['risk_model'] is not None:
            risk_prediction = ml_models['risk_model'].predict(features_array)[0]
            risk_levels = ['Low', 'Medium', 'High']
            ml_risk_level = risk_levels[risk_prediction]
            print(f"DEBUG - ML risk prediction: {ml_risk_level} (index: {risk_prediction})")
            
            # ML confidence based solely on model probabilities
            risk_probs = ml_models['risk_model'].predict_proba(features_array)[0]
            ml_confidence = float(np.clip(np.max(risk_probs), 0.0, 1.0))
            print(f"DEBUG - ML probabilities: Low={risk_probs[0]:.3f}, Medium={risk_probs[1]:.3f}, High={risk_probs[2]:.3f}")
            
            # HYBRID DECISION: Smart risk level selection
            # Trust actual calculation when it shows Low Risk with high alignment/low conflict
            # Trust ML model when actual calculation shows High Risk (more reliable)
            # This prevents false High Risk predictions when data shows healthy relationship
            risk_level_priority = {'Low': 0, 'Medium': 1, 'High': 2}
            
            # Get personalized features to check alignment and conflict
            alignment_score = personalized_features.get('alignment_score', 0.5)
            conflict_ratio = personalized_features.get('conflict_ratio', 0.0)
            
            # If actual calculation shows Low Risk AND we have high alignment/low conflict, trust it
            # This prevents ML model from incorrectly predicting High Risk based on demographics
            if actual_risk_level == 'Low' and alignment_score > 0.7 and conflict_ratio < 0.15:
                risk_level = actual_risk_level
                print(f"DEBUG - Using ACTUAL risk level ({actual_risk_level}) over ML prediction ({ml_risk_level})")
                print(f"DEBUG - Reason: High alignment ({alignment_score:.1%}) and low conflict ({conflict_ratio:.1%}) indicate Low Risk")
            # If actual calculation shows High Risk, trust it (more reliable than ML for high risk)
            elif actual_risk_level == 'High':
                risk_level = actual_risk_level
                print(f"DEBUG - Using ACTUAL risk level ({actual_risk_level}) over ML prediction ({ml_risk_level})")
                print(f"DEBUG - Reason: Actual disagreement ratio ({actual_disagree_ratio:.1%}) indicates High Risk")
            # If ML suggests higher risk than actual, use ML (might catch patterns actual calculation misses)
            elif risk_level_priority[ml_risk_level] > risk_level_priority[actual_risk_level]:
                risk_level = ml_risk_level
                print(f"DEBUG - Using ML risk level ({ml_risk_level}) over actual ({actual_risk_level})")
                print(f"DEBUG - Reason: ML model suggests higher risk, may catch patterns not in disagreement ratio")
            # Otherwise, use actual calculation (more reliable for Low/Medium)
            else:
                risk_level = actual_risk_level
                print(f"DEBUG - Using ACTUAL risk level ({actual_risk_level}) over ML prediction ({ml_risk_level})")
                print(f"DEBUG - Reason: Actual calculation is more reliable for this risk level")
        else:
            return jsonify({
                'status': 'error',
                'message': 'Risk model not loaded. Train or load models first.'
            })
        
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
        # Pass actual_disagree_ratio, ml_risk_level, and actual_risk_level for detailed reasoning
        # These variables are defined in the hybrid decision logic above
        try:
            risk_reasoning = generate_risk_reasoning(
                couple_profile, 
                personalized_features, 
                risk_level,
                actual_disagree_ratio=actual_disagree_ratio,
                ml_risk_level=ml_risk_level,
                actual_risk_level=actual_risk_level
            )
        except NameError:
            # Fallback if variables not defined (shouldn't happen, but safe)
            risk_reasoning = generate_risk_reasoning(
                couple_profile, 
                personalized_features, 
                risk_level
            )
        counseling_reasoning = generate_counseling_reasoning(focus_categories, category_scores, ml_confidence)
        
        # Generate personalized recommendations
        personalized_recommendations = generate_personalized_recommendations(
            risk_level, category_scores, focus_categories, 
            personalized_features, male_responses, female_responses, couple_profile
        )
        
        return jsonify({
            'status': 'success',
            'couple_id': data.get('couple_id', 'unknown'),
            'risk_level': risk_level,  # Final hybrid risk level
            'actual_risk_level': actual_risk_level,  # Response-based risk level (male vs female comparison)
            'actual_disagree_ratio': float(actual_disagree_ratio),  # Disagreement ratio percentage
            'ml_risk_level': ml_risk_level,  # ML model prediction
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

def generate_risk_reasoning(couple_profile, personalized_features, risk_level, actual_disagree_ratio=None, ml_risk_level=None, actual_risk_level=None):
    """Generate detailed reasoning for risk level based on actual couple features"""
    reasoning_parts = []
    
    # PRIMARY REASONING: Why this risk level?
    if risk_level == 'High':
        reasoning_parts.append("🔴 HIGH RISK CLASSIFICATION:")
        if actual_disagree_ratio and actual_disagree_ratio > 0.35:
            reasoning_parts.append(f"   • Response Analysis: {actual_disagree_ratio*100:.1f}% weighted disagreement ratio (threshold: >35%)")
            reasoning_parts.append(f"   • This indicates significant disagreements across multiple MEAI categories")
        reasoning_parts.append("   • Requires immediate, intensive counseling intervention")
    elif risk_level == 'Medium':
        reasoning_parts.append("🟡 MEDIUM RISK CLASSIFICATION:")
        if actual_disagree_ratio and 0.20 < actual_disagree_ratio <= 0.35:
            reasoning_parts.append(f"   • Response Analysis: {actual_disagree_ratio*100:.1f}% weighted disagreement ratio (threshold: 20-35%)")
            reasoning_parts.append(f"   • This indicates moderate concerns requiring proactive attention")
        reasoning_parts.append("   • Proactive counseling recommended to address identified issues")
    else:  # Low
        reasoning_parts.append("🟢 LOW RISK CLASSIFICATION:")
        if actual_disagree_ratio and actual_disagree_ratio <= 0.20:
            reasoning_parts.append(f"   • Response Analysis: {actual_disagree_ratio*100:.1f}% weighted disagreement ratio (threshold: ≤20%)")
            reasoning_parts.append(f"   • This indicates healthy relationship with minimal disagreements")
        reasoning_parts.append("   • Preventive counseling recommended to maintain relationship health")
    
    # DECISION SOURCE
    if actual_risk_level and ml_risk_level:
        if actual_risk_level != ml_risk_level:
            reasoning_parts.append(f"\n📊 DECISION SOURCE: Hybrid Analysis (Methods Disagree)")
            reasoning_parts.append(f"   • Response-Based Calculation: {actual_risk_level} Risk")
            if actual_disagree_ratio is not None:
                reasoning_parts.append(f"     → Disagreement ratio: {actual_disagree_ratio*100:.1f}%")
            alignment_score = personalized_features.get('alignment_score', 0.5)
            conflict_ratio = personalized_features.get('conflict_ratio', 0.0)
            reasoning_parts.append(f"     → Alignment: {int(alignment_score * 100)}%, Conflict: {int(conflict_ratio * 100)}%")
            reasoning_parts.append(f"   • ML Model Prediction: {ml_risk_level} Risk")
            reasoning_parts.append(f"     → Based on demographic patterns and learned relationships")
            reasoning_parts.append(f"   • Final Decision: {risk_level} Risk")
            if risk_level == actual_risk_level:
                reasoning_parts.append(f"     → Chosen: Response-based calculation (more reliable for this case)")
            else:
                reasoning_parts.append(f"     → Chosen: ML model prediction (demographic factors indicate risk)")
        else:
            reasoning_parts.append(f"\n📊 DECISION SOURCE: Both Methods Agree")
            reasoning_parts.append(f"   • Response-Based Calculation: {actual_risk_level} Risk")
            reasoning_parts.append(f"   • ML Model Prediction: {ml_risk_level} Risk")
            reasoning_parts.append(f"   • Final Decision: {risk_level} Risk (consensus)")
    
    # DEMOGRAPHIC FACTORS ANALYSIS
    reasoning_parts.append("\n👥 DEMOGRAPHIC FACTORS:")
    
    # Age difference analysis
    male_age = couple_profile.get('male_age', 30)
    female_age = couple_profile.get('female_age', 30)
    age_gap = abs(male_age - female_age)
    
    if age_gap > 10:
        reasoning_parts.append(f"   • ⚠️ Significant age gap: {age_gap} years (Male: {male_age}, Female: {female_age})")
        reasoning_parts.append(f"     → Large age gaps can indicate different life stages and values")
        if risk_level == 'High':
            reasoning_parts.append(f"     → This demographic factor may contribute to relationship challenges")
    elif age_gap > 5:
        reasoning_parts.append(f"   • Moderate age gap: {age_gap} years (Male: {male_age}, Female: {female_age})")
    else:
        reasoning_parts.append(f"   • ✅ Minimal age gap: {age_gap} years (similar life stages)")
    
    # Civil status analysis
    civil_status = couple_profile.get('civil_status', 'Single')
    if civil_status == 'Living In':
        years_together = couple_profile.get('years_living_together', 0)
        if years_together > 5:
            reasoning_parts.append(f"   • Long-term cohabitation: {years_together} years")
            reasoning_parts.append(f"     → Established relationship patterns, may have unresolved issues")
        elif years_together > 0:
            reasoning_parts.append(f"   • Recent cohabitation: {years_together} years")
            reasoning_parts.append(f"     → Still developing relationship patterns")
    elif civil_status in ['Widowed', 'Separated', 'Divorced']:
        reasoning_parts.append(f"   • ⚠️ Previous relationship experience: {civil_status}")
        reasoning_parts.append(f"     → May affect current relationship dynamics and require healing focus")
        if risk_level in ['Medium', 'High']:
            reasoning_parts.append(f"     → This factor may contribute to relationship challenges")
    else:
        reasoning_parts.append(f"   • ✅ Single status: No previous relationship complications")
    
    # Education and income compatibility
    education_level = couple_profile.get('education_level', 2)
    income_level = couple_profile.get('income_level', 2)
    education_income_diff = abs(education_level - income_level)
    
    if education_income_diff > 2:
        reasoning_parts.append(f"   • ⚠️ Education-Income Mismatch: {education_income_diff} level difference")
        reasoning_parts.append(f"     → Education level: {education_level}, Income level: {income_level}")
        reasoning_parts.append(f"     → Significant differences may indicate compatibility challenges")
        if risk_level in ['Medium', 'High']:
            reasoning_parts.append(f"     → This demographic factor may contribute to relationship stress")
    elif education_income_diff > 1:
        reasoning_parts.append(f"   • Moderate education-income difference: {education_income_diff} levels")
    else:
        reasoning_parts.append(f"   • ✅ Compatible education-income levels: Similar socioeconomic status")
    
    # Employment status
    employment_status = couple_profile.get('employment_status', 'Unemployed')
    if employment_status == 'Unemployed':
        reasoning_parts.append(f"   • ⚠️ Employment Status: {employment_status}")
        reasoning_parts.append(f"     → Unemployment may contribute to financial stress and relationship challenges")
    elif employment_status == 'Self-employed':
        reasoning_parts.append(f"   • Employment Status: {employment_status} (variable income)")
    else:
        reasoning_parts.append(f"   • ✅ Employment Status: {employment_status} (stable income)")
    
    # RESPONSE-BASED FACTORS ANALYSIS
    reasoning_parts.append("\n💬 RESPONSE-BASED FACTORS:")
    
    # Relationship dynamics analysis
    alignment_score = personalized_features.get('alignment_score', 0.5)
    conflict_ratio = personalized_features.get('conflict_ratio', 0.0)
    
    # Detect conflicts between risk level and response-based indicators
    has_conflict = False
    if risk_level == 'High' and alignment_score > 0.7 and conflict_ratio < 0.15:
        has_conflict = True
        reasoning_parts.append(f"   ⚠️ CONFLICT DETECTED: High Risk classification despite positive response indicators")
        reasoning_parts.append(f"   • ✅ High Alignment: {int(alignment_score * 100)}% agreement between partners")
        reasoning_parts.append(f"   • ✅ Low Conflict: {int(conflict_ratio * 100)}% disagreement rate")
        reasoning_parts.append(f"   • Response-based calculation suggests: Low Risk")
        if ml_risk_level and ml_risk_level == 'High' and actual_risk_level and actual_risk_level == 'Low':
            reasoning_parts.append(f"   • ML Model Prediction: High Risk (based on demographic patterns)")
            reasoning_parts.append(f"   • Explanation: Despite good alignment and low conflict, demographic factors")
            reasoning_parts.append(f"     (age gap, civil status, education-income mismatch, employment) indicate")
            reasoning_parts.append(f"     potential relationship challenges that may not be immediately apparent")
            reasoning_parts.append(f"     in current responses but could surface over time.")
    elif risk_level == 'Low' and alignment_score < 0.4 and conflict_ratio > 0.3:
        has_conflict = True
        reasoning_parts.append(f"   ⚠️ CONFLICT DETECTED: Low Risk classification despite concerning response indicators")
        reasoning_parts.append(f"   • ⚠️ Low Alignment: {int(alignment_score * 100)}% agreement between partners")
        reasoning_parts.append(f"   • ⚠️ High Conflict: {int(conflict_ratio * 100)}% disagreement rate")
        reasoning_parts.append(f"   • Response-based calculation suggests: High Risk")
        if ml_risk_level and ml_risk_level == 'Low' and actual_risk_level and actual_risk_level == 'High':
            reasoning_parts.append(f"   • ML Model Prediction: Low Risk (based on demographic patterns)")
            reasoning_parts.append(f"   • Explanation: Despite current disagreements, demographic factors suggest")
            reasoning_parts.append(f"     potential for relationship stability. However, current response patterns")
            reasoning_parts.append(f"     indicate immediate attention may be needed.")
    
    if not has_conflict:
        # Normal display without conflict
        if alignment_score > 0.7:
            reasoning_parts.append(f"   • ✅ High Alignment: {int(alignment_score * 100)}% agreement between partners")
            reasoning_parts.append(f"     → Partners share similar values and perspectives")
            if risk_level == 'Low':
                reasoning_parts.append(f"     → This strong alignment supports Low Risk classification")
        elif alignment_score < 0.4:
            reasoning_parts.append(f"   • ⚠️ Low Alignment: {int(alignment_score * 100)}% agreement between partners")
            reasoning_parts.append(f"     → Significant differences in values and perspectives")
            if risk_level in ['Medium', 'High']:
                reasoning_parts.append(f"     → This low alignment contributes to {risk_level} Risk classification")
        else:
            reasoning_parts.append(f"   • Moderate Alignment: {int(alignment_score * 100)}% agreement")
            reasoning_parts.append(f"     → Some areas of agreement, some areas of difference")
        
        if conflict_ratio > 0.3:
            reasoning_parts.append(f"   • ⚠️ High Conflict: {int(conflict_ratio * 100)}% disagreement rate")
            reasoning_parts.append(f"     → Frequent disagreements between partners")
            if risk_level in ['Medium', 'High']:
                reasoning_parts.append(f"     → This high conflict rate is a primary factor for {risk_level} Risk")
        elif conflict_ratio > 0.1:
            reasoning_parts.append(f"   • Moderate Conflict: {int(conflict_ratio * 100)}% disagreement rate")
            reasoning_parts.append(f"     → Some disagreements, manageable with communication skills")
        else:
            reasoning_parts.append(f"   • ✅ Low Conflict: {int(conflict_ratio * 100)}% disagreement rate")
            reasoning_parts.append(f"     → Minimal disagreements, healthy communication patterns")
            if risk_level == 'Low':
                reasoning_parts.append(f"     → This low conflict supports Low Risk classification")
    
    # Category-specific analysis
    category_alignments = personalized_features.get('category_alignments', [0.5, 0.5, 0.5, 0.5])
    if len(category_alignments) >= 4 and MEAI_CATEGORIES:
        reasoning_parts.append("\n📋 MEAI CATEGORY ANALYSIS:")
        for i, (category, alignment) in enumerate(zip(MEAI_CATEGORIES, category_alignments)):
            if alignment < 0.4:
                reasoning_parts.append(f"   • ⚠️ {category}: {int(alignment * 100)}% alignment (Low - needs attention)")
            elif alignment > 0.7:
                reasoning_parts.append(f"   • ✅ {category}: {int(alignment * 100)}% alignment (High - strong agreement)")
            else:
                reasoning_parts.append(f"   • {category}: {int(alignment * 100)}% alignment (Moderate)")
    
    # SUMMARY
    reasoning_parts.append(f"\n📝 SUMMARY:")
    disagree_ratio_text = f"{actual_disagree_ratio*100:.1f}%" if actual_disagree_ratio is not None else "calculated from responses"
    
    # Check for conflicts to provide clearer summary
    alignment_score = personalized_features.get('alignment_score', 0.5)
    conflict_ratio = personalized_features.get('conflict_ratio', 0.0)
    has_conflict_summary = False
    if risk_level == 'High' and alignment_score > 0.7 and conflict_ratio < 0.15:
        has_conflict_summary = True
    elif risk_level == 'Low' and alignment_score < 0.4 and conflict_ratio > 0.3:
        has_conflict_summary = True
    
    if risk_level == 'High':
        reasoning_parts.append(f"   The {risk_level} Risk classification is based on:")
        if has_conflict_summary:
            reasoning_parts.append(f"   • ⚠️ ML Model Prediction (demographic-based) overrides response-based calculation")
            reasoning_parts.append(f"   • Despite good alignment ({int(alignment_score * 100)}%) and low conflict ({int(conflict_ratio * 100)}%),")
            reasoning_parts.append(f"     demographic factors indicate potential relationship challenges")
            if actual_disagree_ratio is not None and actual_disagree_ratio <= 0.20:
                reasoning_parts.append(f"   • Response-based calculation: Low Risk ({disagree_ratio_text} disagreement)")
            reasoning_parts.append(f"   • Primary factors: Demographic patterns (age gap, civil status, education-income mismatch)")
        else:
            if actual_disagree_ratio is not None:
                reasoning_parts.append(f"   • High disagreement ratio: {disagree_ratio_text} (threshold: >35%)")
            reasoning_parts.append(f"   • ML model pattern recognition indicating significant relationship challenges")
            reasoning_parts.append(f"   • Demographic factors (age gap, civil status, education-income mismatch) may contribute")
            reasoning_parts.append(f"   • Low alignment and/or high conflict patterns detected")
    elif risk_level == 'Medium':
        reasoning_parts.append(f"   The {risk_level} Risk classification is based on:")
        if actual_disagree_ratio is not None:
            reasoning_parts.append(f"   • Moderate disagreement ratio: {disagree_ratio_text} (threshold: 20-35%)")
        reasoning_parts.append(f"   • ML model identifying some areas of concern")
        reasoning_parts.append(f"   • Some demographic factors may contribute to challenges")
        reasoning_parts.append(f"   • Moderate alignment and/or conflict patterns detected")
    else:  # Low
        reasoning_parts.append(f"   The {risk_level} Risk classification is based on:")
        if has_conflict_summary:
            reasoning_parts.append(f"   • ⚠️ Response-based calculation (high alignment/low conflict) overrides ML prediction")
            reasoning_parts.append(f"   • Despite concerning demographic factors, current relationship indicators")
            reasoning_parts.append(f"     (alignment: {int(alignment_score * 100)}%, conflict: {int(conflict_ratio * 100)}%) show healthy patterns")
            if actual_disagree_ratio is not None:
                reasoning_parts.append(f"   • Low disagreement ratio: {disagree_ratio_text} (threshold: ≤20%)")
            reasoning_parts.append(f"   • Note: Monitor relationship as demographic factors may still pose challenges")
        else:
            if actual_disagree_ratio is not None:
                reasoning_parts.append(f"   • Low disagreement ratio: {disagree_ratio_text} (threshold: ≤20%)")
            reasoning_parts.append(f"   • High alignment and low conflict patterns")
            reasoning_parts.append(f"   • Healthy relationship indicators despite any demographic factors")
    
    return "\n".join(reasoning_parts)

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
    
    app.run(host='127.0.0.1', port=5000, debug=True)
