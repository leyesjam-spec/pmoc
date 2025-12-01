# ML Service for Couple Counseling Recommendations
"""
Machine Learning Service
Uses Random Forest models for couple counseling risk assessment and recommendations
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
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
        # We need to split them properly - the PHP API should send them as separate arrays
        # For now, assume questionnaire_responses is already the combined responses
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
    
    # Calculate alignment score
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
        # Asymmetric partial weights:
        # - 4 vs 2: 1.0 (strong conflict)
        # - 3 vs 2: 0.6 (neutral vs disagree)
        # - 4 vs 3: 0.4 (agree vs neutral)
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
                # Fallback for unexpected scales: treat as mild
                weighted_conflict_sum += 0.5
    
    alignment_score = alignment_score / total_questions if total_questions > 0 else 0.5
    conflict_ratio = (weighted_conflict_sum / total_questions) if total_questions > 0 else 0
    
    # Calculate partner averages
    male_avg = sum(male_responses) / len(male_responses) if male_responses else 3.0
    female_avg = sum(female_responses) / len(female_responses) if female_responses else 3.0
    
    # Calculate consistency (inverse of variance)
    def calculate_consistency(responses):
        if len(responses) < 2:
            return 1.0
        mean = sum(responses) / len(responses)
        variance = sum((r - mean) ** 2 for r in responses) / len(responses)
        return max(0, 1 - (variance / 4))  # 0-1 scale, higher = more consistent
    
    male_consistency = calculate_consistency(male_responses)
    female_consistency = calculate_consistency(female_responses)
    
    # Calculate power balance as a ratio
    power_balance = male_avg / female_avg if female_avg > 0 else 1.0
    
    # Calculate response variance
    response_variance = 0
    if len(all_responses) > 1:
        mean = sum(all_responses) / len(all_responses)
        response_variance = sum((r - mean) ** 2 for r in all_responses) / len(all_responses)
    
    return {
        'alignment_score': alignment_score,
        'conflict_ratio': conflict_ratio,
        'male_avg_response': male_avg,
        'female_avg_response': female_avg,
        'male_consistency': male_consistency,
        'female_consistency': female_consistency,
        'power_balance': power_balance,
        'response_variance': response_variance,
        'total_conflicts': conflict_count
    }

# Global variables for models
ml_models = {
    'risk_model': None,
    'category_model': None,
    'risk_encoder': None
}

# MEAI Categories - dynamically loaded from database question_category table
# These 4 categories are used for ML predictions and recommendations
MEAI_CATEGORIES = []

# MEAI Questions and Sub-questions - dynamically loaded from database
MEAI_QUESTIONS = {}  # {category_id: {question_id: {text, sub_questions: []}}}
MEAI_QUESTION_MAPPING = {}  # {question_id: category_id}

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
    
    data = []
    
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
        
        for j, base_response in enumerate(base_responses):
            # Add some variation to responses while maintaining patterns
            variation = np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
            new_response = max(2, min(4, base_response + variation))
            questionnaire_responses.append(new_response)
        
        # Calculate risk level based on response patterns
        disagree_count = sum(1 for r in questionnaire_responses if r == 2)
        disagree_ratio = disagree_count / len(questionnaire_responses)
        
        if disagree_ratio > 0.3:
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
        
        # Generate responses based on risk bias and couple characteristics
        if risk_bias == 'high':
            # High risk: more disagreements, conflicts
            base_disagree_prob = 0.4
            base_agree_prob = 0.3
        elif risk_bias == 'low':
            # Low risk: more agreements, harmony
            base_disagree_prob = 0.1
            base_agree_prob = 0.6
        else:  # medium
            # Medium risk: balanced responses
            base_disagree_prob = 0.2
            base_agree_prob = 0.4
        
        # Adjust based on age gap (larger gaps = more disagreements)
        age_gap = abs(male_age - female_age)
        if age_gap > 10:
            base_disagree_prob += 0.2
            base_agree_prob -= 0.1
        elif age_gap > 5:
            base_disagree_prob += 0.1
            base_agree_prob -= 0.05
        
        # Adjust based on education mismatch
        education_diff = abs(education_level - income_level)
        if education_diff > 2:
            base_disagree_prob += 0.1
            base_agree_prob -= 0.05
        
        # Adjust based on civil status
        if civil_status in ['Separated', 'Divorced']:
            base_disagree_prob += 0.2
            base_agree_prob -= 0.1
        elif civil_status == 'Living In' and years_living_together > 10:
            base_disagree_prob -= 0.1
            base_agree_prob += 0.1
        
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
        
        if disagree_ratio > 0.35:
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
            MAX(cp.children) as children,
            MAX(cp.education_level) as education_level,
            MAX(cp.income_level) as income_level
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
        for couple in couples:
            access_id, male_name, female_name, male_age, female_age, civil_status, years_living_together, past_children, children, education_level, income_level = couple
            
            # Get MEAI responses for this couple
            response_query = """
            SELECT qr.question_id, qr.response, qr.reason
            FROM question_response qr
            WHERE qr.access_id = %s
            ORDER BY qr.question_id
            """
            cursor.execute(response_query, (access_id,))
            responses = cursor.fetchall()
            
            if len(responses) < 50:  # Need minimum responses
                continue
                
            # Convert responses to numeric format (2=disagree, 3=neutral, 4=agree)
            questionnaire_responses = []
            for _, response, _ in responses:
                if response == 'agree':
                    questionnaire_responses.append(4)
                elif response == 'neutral':
                    questionnaire_responses.append(3)
                else:  # disagree
                    questionnaire_responses.append(2)
            
            # Get total expected responses (main questions only, not sub-questions)
            total_expected_responses = len(MEAI_QUESTION_MAPPING)
            
            # Pad or truncate to expected number of responses
            while len(questionnaire_responses) < total_expected_responses:
                questionnaire_responses.append(3)  # Default to neutral
            questionnaire_responses = questionnaire_responses[:total_expected_responses]
            
            # Calculate risk level based on actual responses (simple heuristic)
            # More disagreements = higher risk
            disagree_count = sum(1 for r in questionnaire_responses if r == 2)
            disagree_ratio = disagree_count / len(questionnaire_responses)
            
            if disagree_ratio > 0.3:
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
                    # Note: This assumes responses are ordered by question_id
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
            
            training_data.append({
                'male_age': male_age or 30,
                'female_age': female_age or 30,
                'civil_status': civil_status or 'Single',
                'years_living_together': years_living_together or 0,
                'past_children': bool(past_children),
                'children': children or 0,
                'education_level': education_level or 2,
                'income_level': income_level or 2,
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
    
    # Ensure questions are loaded before training
    if not MEAI_CATEGORIES:
        load_categories_from_db()
    if not MEAI_QUESTIONS:
        load_questions_from_db()
    
    # Load real couples from database for training
    real_couples_data = load_real_couples_for_training()
    if not real_couples_data:
        print("No real couples found, using generic synthetic data as fallback")
        data = generate_synthetic_data(500)
    else:
        print(f"Found {len(real_couples_data)} real couples, generating 500 synthetic couples based on their patterns")
        # Generate 500 synthetic couples based on real couple patterns
        data = generate_synthetic_data_based_on_real_couples(500, real_couples_data)
    
    df = pd.DataFrame(data)
    
    # Prepare features
    X = []
    y_risk = []
    y_categories = []
    
    for _, row in df.iterrows():
        # Combine all features (same structure as analysis)
        features = [
            row['male_age'],
            row['female_age'],
            row['years_living_together'],
            row['children'],
            row['education_level'],
            row['income_level']
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
            power_balance = np.random.uniform(1.0, 2.5)
        elif row['risk_level'] == 'Low':
            # Low risk: high alignment, low conflict
            alignment_score = np.random.uniform(0.6, 0.9)
            conflict_ratio = np.random.uniform(0.0, 0.2)
            power_balance = np.random.uniform(0.0, 0.8)
        else:  # Medium
            # Medium risk: mixed patterns
            alignment_score = np.random.uniform(0.4, 0.7)
            conflict_ratio = np.random.uniform(0.1, 0.4)
            power_balance = np.random.uniform(0.5, 1.5)
        
        # Add personalized features to match analysis structure
        personalized_features = [
            alignment_score,
            conflict_ratio,
            np.random.uniform(2.5, 3.5),  # male_avg_response
            np.random.uniform(2.5, 3.5),  # female_avg_response
            np.random.uniform(0.3, 0.8),  # male_consistency
            np.random.uniform(0.3, 0.8),  # female_consistency
            power_balance,
            np.random.uniform(0.5, 2.0)   # response_variance
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
    
    print(f"Training with {X.shape[1]} features: {X.shape[0]} samples")
    
    # Train risk prediction model
    risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
    risk_model.fit(X, y_risk)
    
    # Train category prediction model (multi-output regression)
    category_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    category_model.fit(X, y_categories)
    
    # Create risk encoder
    risk_encoder = LabelEncoder()
    risk_encoder.fit(['Low', 'Medium', 'High'])
        
        # Save models
    ml_models['risk_model'] = risk_model
    ml_models['category_model'] = category_model
    ml_models['risk_encoder'] = risk_encoder
    
    # Save to files
    with open('risk_model.pkl', 'wb') as f:
        pickle.dump(risk_model, f)
    
    with open('category_model.pkl', 'wb') as f:
        pickle.dump(category_model, f)
    
    with open('risk_encoder.pkl', 'wb') as f:
        pickle.dump(risk_encoder, f)
    
    print("ML models trained and saved successfully")
    return True

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

@app.route('/train', methods=['POST'])
def train():
    """Train ML models"""
    try:
        success = train_ml_models()
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Models trained successfully',
                'couples_count': 500,
                'training_method': 'Real couples + Synthetic based on real patterns'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Training failed'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Training error: {str(e)}'
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
        
        # Prepare features for ML models (original + personalized)
        features = [
            couple_profile['male_age'],
            couple_profile['female_age'],
            couple_profile['years_living_together'],
            couple_profile['children'],
            couple_profile['education_level'],
            couple_profile['income_level']
        ] + questionnaire_responses
        
        # Add personalized features
        personalized_feature_values = [
            personalized_features.get('alignment_score', 0.5),
            personalized_features.get('conflict_ratio', 0.0),
            personalized_features.get('male_avg_response', 3.0),
            personalized_features.get('female_avg_response', 3.0),
            personalized_features.get('male_consistency', 0.5),
            personalized_features.get('female_consistency', 0.5),
            personalized_features.get('power_balance', 0.0),
            personalized_features.get('response_variance', 0.0)
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
    male_consistency = personalized_features.get('male_consistency', 0.5)
    female_consistency = personalized_features.get('female_consistency', 0.5)
    power_balance = personalized_features.get('power_balance', 0.0)
    response_variance = personalized_features.get('response_variance', 0.0)
    
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
    
    # 3. DYNAMIC PARTNER-SPECIFIC RECOMMENDATIONS (only show if there's actual concern)
    # Only show power imbalance if there's significant difference AND low alignment
    if (power_balance > 1.5 or power_balance < 0.3) and alignment_score < 0.8:
        if male_avg > female_avg:
            recommendations.append(f"👨 MALE DOMINANCE: Male partner shows {male_avg:.1f} vs female {female_avg:.1f} average - ensure balanced decision-making and equal voice")
        else:
            recommendations.append(f"👩 FEMALE DOMINANCE: Female partner shows {female_avg:.1f} vs male {male_avg:.1f} average - ensure balanced decision-making and equal voice")
    elif power_balance >= 0.7 and power_balance <= 1.3:
        recommendations.append(f"🤝 BALANCED PARTNERSHIP: {power_balance:.1f} power balance - excellent relationship equality")
    # Skip moderate imbalance messages to avoid confusion
    
    # 4. ENHANCED PARTNER-SPECIFIC ANALYSIS
    # Male partner analysis
    if male_consistency < 0.3:
        recommendations.append(f"👨 MALE INCONSISTENCY: {int(male_consistency * 100)}% consistency - male partner needs individual counseling to clarify values and goals")
    elif male_consistency < 0.6:
        recommendations.append(f"👨 MALE UNCERTAINTY: {int(male_consistency * 100)}% consistency - male partner may benefit from values clarification sessions")
    else:
        recommendations.append(f"👨 MALE CLARITY: {int(male_consistency * 100)}% consistency - male partner shows clear values and goals")
    
    # Female partner analysis
    if female_consistency < 0.3:
        recommendations.append(f"👩 FEMALE INCONSISTENCY: {int(female_consistency * 100)}% consistency - female partner needs individual counseling to clarify values and goals")
    elif female_consistency < 0.6:
        recommendations.append(f"👩 FEMALE UNCERTAINTY: {int(female_consistency * 100)}% consistency - female partner may benefit from values clarification sessions")
    else:
        recommendations.append(f"👩 FEMALE CLARITY: {int(female_consistency * 100)}% consistency - female partner shows clear values and goals")
    
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
    
    # 7. DYNAMIC RESPONSE PATTERN RECOMMENDATIONS
    if response_variance > 2.5:
        recommendations.append(f"📊 COMPLEX DYNAMICS: High variance ({response_variance:.1f}) suggests complex relationship patterns - comprehensive assessment and specialized counseling needed")
    elif response_variance > 1.5:
        recommendations.append(f"📊 VARIED RESPONSES: Moderate variance ({response_variance:.1f}) indicates diverse perspectives - structured communication training recommended")
    elif response_variance > 0.5:
        recommendations.append(f"📊 BALANCED DIVERSITY: Healthy variance ({response_variance:.1f}) shows good relationship complexity - continue current approach")
    else:
        recommendations.append(f"📊 CONSISTENT PATTERNS: Low variance ({response_variance:.1f}) indicates stable relationship dynamics - maintain current healthy patterns")
    
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
