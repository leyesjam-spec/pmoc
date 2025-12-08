# Natural Language Generation (NLG) Recommendation Engine
"""
Advanced NLG system for generating human-like, personalized counseling recommendations
Replaces rule-based templates with natural language generation
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Any

class NLGRecommendationEngine:
    """Natural Language Generation engine for counseling recommendations"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.personality_traits = self._initialize_personality_traits()
        self.counseling_styles = self._initialize_counseling_styles()
    
    def _initialize_templates(self) -> Dict[str, List[str]]:
        """Initialize NLG templates for different recommendation types"""
        return {
            'alignment_critical': [
                "Based on your responses, there appears to be significant misalignment in core values and expectations. This level of disagreement ({alignment}%) suggests that both partners may benefit from intensive relationship counseling to explore fundamental compatibility and shared vision for the future.",
                "Your assessment reveals substantial differences in key relationship areas ({alignment}% alignment). This pattern often indicates underlying communication barriers that require professional intervention to help both partners understand and bridge these important gaps.",
                "The analysis shows concerning levels of disagreement ({alignment}% alignment) across multiple relationship dimensions. This suggests the need for immediate, structured counseling to address fundamental compatibility issues before moving forward with marriage plans."
            ],
            'alignment_moderate': [
                "Your responses indicate moderate alignment ({alignment}%) with some areas of strong agreement and others requiring attention. This is actually quite common and suggests a healthy foundation with specific areas for growth and development.",
                "The assessment reveals a balanced relationship dynamic ({alignment}% alignment) where you share many values while having some natural differences. This presents an excellent opportunity for targeted counseling to strengthen your partnership.",
                "Your responses show moderate alignment ({alignment}%) which indicates a relationship with both strengths and growth areas. This is an ideal situation for proactive counseling to build on your strengths while addressing areas of difference."
            ],
            'alignment_strong': [
                "Excellent news! Your responses demonstrate strong alignment ({alignment}%) across most relationship areas. This indicates a solid foundation of shared values and mutual understanding that will serve you well in marriage.",
                "Your assessment reveals remarkable compatibility ({alignment}% alignment) in core relationship areas. This strong foundation suggests you're well-prepared for marriage, though continued growth and communication will always be valuable.",
                "The analysis shows outstanding alignment ({alignment}%) in your relationship values and expectations. This level of compatibility indicates you're building on a very strong foundation for a successful marriage."
            ],
            'conflict_high': [
                "Your responses indicate significant conflict patterns ({conflict}% disagreement rate) that suggest underlying tensions requiring immediate attention. Professional mediation and conflict resolution counseling would be highly beneficial to address these patterns before they escalate.",
                "The assessment reveals concerning conflict levels ({conflict}% disagreement) across multiple areas. This pattern often indicates communication breakdowns that need structured intervention to help both partners develop healthier conflict resolution skills.",
                "Your responses show high conflict indicators ({conflict}% disagreement rate) that suggest the relationship would benefit from intensive conflict resolution counseling. Addressing these patterns now can prevent future relationship difficulties."
            ],
            'conflict_moderate': [
                "Your responses indicate moderate conflict levels ({conflict}% disagreement) which is actually quite normal in relationships. This suggests some areas where communication skills and conflict resolution strategies could be strengthened through targeted counseling.",
                "The assessment shows manageable conflict patterns ({conflict}% disagreement) with room for improvement in how you handle differences. This is an ideal situation for learning effective communication and conflict resolution skills.",
                "Your responses reveal moderate conflict indicators ({conflict}% disagreement) that suggest opportunities for growth in how you navigate differences together. This is a perfect scenario for proactive relationship skill development."
            ],
            'conflict_low': [
                "Your responses show excellent conflict management patterns ({conflict}% disagreement rate), indicating strong communication and problem-solving skills. This is a wonderful foundation that suggests you handle differences in healthy, constructive ways.",
                "The assessment reveals very low conflict indicators ({conflict}% disagreement), suggesting you have developed effective ways of managing differences. This strong foundation will serve you well in maintaining a healthy marriage.",
                "Your responses demonstrate outstanding conflict resolution patterns ({conflict}% disagreement rate), indicating mature communication skills and mutual respect. This level of harmony is a significant strength in your relationship."
            ],
            'power_imbalance': [
                "The analysis reveals a significant power imbalance in your relationship ({power_balance:.1f} ratio), with one partner's voice being more dominant in decision-making. This pattern can lead to resentment and communication breakdowns, making it important to work on balanced partnership dynamics.",
                "Your responses indicate a notable power imbalance ({power_balance:.1f} ratio) that may be affecting relationship equality. Addressing this through counseling can help ensure both partners feel heard and valued in the relationship.",
                "The assessment shows concerning power dynamics ({power_balance:.1f} ratio) that suggest one partner may be dominating decision-making processes. This imbalance can undermine relationship satisfaction and should be addressed through focused counseling."
            ],
            'power_balanced': [
                "Your responses indicate a healthy power balance ({power_balance:.1f} ratio) in your relationship, suggesting both partners have equal voice in decision-making. This balanced dynamic is a significant strength that contributes to relationship satisfaction and longevity.",
                "The analysis reveals excellent partnership dynamics ({power_balance:.1f} ratio) with balanced decision-making processes. This equality in your relationship is a wonderful foundation for a successful marriage.",
                "Your responses demonstrate healthy power sharing ({power_balance:.1f} ratio), indicating mutual respect and equal partnership. This balanced approach to decision-making is a key strength in your relationship."
            ],
            'consistency_issues': [
                "The assessment reveals some inconsistency in {partner}'s responses ({consistency}% consistency), which may indicate uncertainty or conflicting feelings about certain relationship aspects. Individual counseling could help clarify values and goals.",
                "Your responses show some inconsistency patterns ({consistency}% consistency) that suggest {partner} may benefit from additional reflection and clarification of personal values and relationship expectations.",
                "The analysis indicates some response inconsistency ({consistency}% consistency) that may reflect {partner}'s internal conflict or uncertainty. This is a common experience that can be addressed through supportive counseling."
            ],
            'consistency_strong': [
                "Your responses demonstrate excellent consistency ({consistency}% consistency), indicating clear values and well-thought-out perspectives on relationship matters. This clarity is a significant strength in your partnership.",
                "The assessment reveals strong consistency ({consistency}% consistency) in your responses, suggesting you have a clear understanding of your values and relationship expectations. This clarity is a wonderful foundation for marriage.",
                "Your responses show remarkable consistency ({consistency}% consistency), indicating thoughtful reflection and clear communication about your relationship needs and expectations. This level of clarity is a real strength."
            ],
            'risk_high': [
                "Based on the comprehensive analysis, your relationship shows indicators of high risk that require immediate attention. The combination of factors suggests that intensive counseling would be highly beneficial to address underlying issues before they become more serious.",
                "The assessment reveals multiple risk factors that indicate your relationship would benefit from immediate, structured intervention. Professional counseling can help you address these concerns and build a stronger foundation.",
                "Your responses indicate several areas of concern that suggest high-risk patterns in your relationship. Addressing these through professional counseling now can prevent future difficulties and help you build a healthier partnership."
            ],
            'risk_medium': [
                "The analysis indicates moderate risk factors in your relationship that suggest proactive counseling would be beneficial. Addressing these areas now can prevent future problems and strengthen your partnership.",
                "Your responses reveal some areas of concern that indicate moderate risk patterns. This is actually an ideal time for counseling to address these issues before they become more serious.",
                "The assessment shows moderate risk indicators that suggest your relationship would benefit from focused counseling to address specific areas of concern and build on your strengths."
            ],
            'risk_low': [
                "The analysis reveals a low-risk relationship profile with strong foundations and healthy patterns. This suggests you're well-prepared for marriage, though continued growth and communication will always be valuable.",
                "Your responses indicate a low-risk relationship with excellent compatibility and communication patterns. This strong foundation suggests you're building a relationship that can weather life's challenges together.",
                "The assessment shows a low-risk profile with healthy relationship dynamics. This indicates you have a solid foundation for a successful marriage, with opportunities for continued growth and development."
            ]
        }
    
    def _initialize_personality_traits(self) -> Dict[str, List[str]]:
        """Initialize personality-based language variations"""
        return {
            'empathetic': [
                "I understand this may feel overwhelming, but remember that every relationship has areas for growth.",
                "It's completely normal to have differences in relationships - what matters is how you work through them together.",
                "Many couples face similar challenges, and the fact that you're seeking help shows your commitment to each other."
            ],
            'encouraging': [
                "You've already taken an important step by participating in this assessment - that shows real commitment to your relationship.",
                "Every challenge in your relationship is also an opportunity for growth and deeper connection.",
                "The fact that you're both here, working on your relationship, speaks volumes about your dedication to each other."
            ],
            'professional': [
                "The assessment data provides valuable insights into your relationship dynamics and areas for development.",
                "Based on the comprehensive analysis, there are specific areas where targeted intervention would be most beneficial.",
                "The evaluation reveals important patterns that can guide your counseling and relationship development process."
            ]
        }
    
    def _initialize_counseling_styles(self) -> Dict[str, List[str]]:
        """Initialize different counseling approach recommendations"""
        return {
            'communication': [
                "Focus on developing active listening skills and expressing needs clearly",
                "Practice 'I' statements and avoid blame language during difficult conversations",
                "Learn to identify and communicate your emotional needs effectively"
            ],
            'conflict_resolution': [
                "Develop structured approaches to handling disagreements before they escalate",
                "Learn to take breaks during heated discussions and return with cooler heads",
                "Practice finding win-win solutions that address both partners' core needs"
            ],
            'values_clarification': [
                "Spend time exploring and sharing your individual values and life goals",
                "Identify areas where your values align and where they differ",
                "Work together to create shared values and vision for your future"
            ],
            'intimacy_building': [
                "Focus on emotional intimacy and vulnerability in your relationship",
                "Develop rituals and practices that strengthen your emotional connection",
                "Learn to express love and appreciation in ways that resonate with your partner"
            ]
        }
    
    def generate_natural_recommendations(self, 
                                       risk_level: str, 
                                       category_scores: List[float], 
                                       focus_categories: List[Dict], 
                                       personalized_features: Dict, 
                                       male_responses: List[int], 
                                       female_responses: List[int],
                                       couple_profile: Dict) -> List[str]:
        """Generate natural language recommendations using NLG"""
        
        recommendations = []
        
        # Extract key metrics
        alignment_score = personalized_features.get('alignment_score', 0.5)
        conflict_ratio = personalized_features.get('conflict_ratio', 0.0)
        male_avg = personalized_features.get('male_avg_response', 3.0)
        female_avg = personalized_features.get('female_avg_response', 3.0)
        
        # Calculate additional metrics
        male_agree_count = sum(1 for r in male_responses if r >= 4)
        female_agree_count = sum(1 for r in female_responses if r >= 4)
        total_responses = len(male_responses)
        male_positive_ratio = male_agree_count / total_responses if total_responses > 0 else 0
        female_positive_ratio = female_agree_count / total_responses if total_responses > 0 else 0
        couple_optimism = (male_positive_ratio + female_positive_ratio) / 2
        
        # 1. Generate alignment-based recommendations using actual data
        if alignment_score < 0.3:
            template = random.choice(self.templates['alignment_critical'])
            recommendations.append(template.format(alignment=int(alignment_score * 100)))
        elif alignment_score < 0.7:
            template = random.choice(self.templates['alignment_moderate'])
            recommendations.append(template.format(alignment=int(alignment_score * 100)))
        else:
            template = random.choice(self.templates['alignment_strong'])
            recommendations.append(template.format(alignment=int(alignment_score * 100)))
        
        # 2. Generate conflict-based recommendations using actual data
        if conflict_ratio > 0.5:
            template = random.choice(self.templates['conflict_high'])
            recommendations.append(template.format(conflict=int(conflict_ratio * 100)))
        elif conflict_ratio > 0.1:
            template = random.choice(self.templates['conflict_moderate'])
            recommendations.append(template.format(conflict=int(conflict_ratio * 100)))
        else:
            template = random.choice(self.templates['conflict_low'])
            recommendations.append(template.format(conflict=int(conflict_ratio * 100)))
        
        # 3. Generate partner difference recommendations based on average responses
        avg_difference = abs(male_avg - female_avg)
        if avg_difference > 0.5 and alignment_score < 0.8:
            # Significant difference in partner responses - only mention if it's a concern
            recommendations.append(f"Your responses show notable differences between partners (male: {male_avg:.1f}, female: {female_avg:.1f}). This suggests the need for balanced communication and decision-making processes.")
        # Removed: Generic "balanced partnership dynamics" message - not actionable for counseling
        
        # 5. Generate risk-based recommendations
        if risk_level == 'High':
            template = random.choice(self.templates['risk_high'])
            recommendations.append(template)
        elif risk_level == 'Medium':
            template = random.choice(self.templates['risk_medium'])
            recommendations.append(template)
        else:
            template = random.choice(self.templates['risk_low'])
            recommendations.append(template)
        
        # 6. Generate category-specific recommendations
        for category in focus_categories:
            if category['score'] > 0.6:
                recommendations.append(self._generate_category_specific_nlg(category, couple_profile))
        
        # 7. Generate personalized counseling approach
        recommendations.append(self._generate_counseling_approach_nlg(
            risk_level, alignment_score, conflict_ratio, couple_profile
        ))
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def _generate_category_specific_nlg(self, category: Dict, couple_profile: Dict) -> str:
        """Generate natural language for specific MEAI categories"""
        category_name = category['name'].lower()
        score = category['score']
        
        if 'marriage' in category_name and 'relationship' in category_name:
            if score > 0.7:
                return f"Your responses indicate significant challenges in marriage expectations and relationship foundations ({int(score * 100)}% priority). This suggests the need for intensive relationship counseling to address core compatibility issues and build stronger partnership foundations."
            else:
                return f"Your responses show some areas for development in marriage expectations and relationship foundations ({int(score * 100)}% priority). This presents an opportunity for focused counseling to strengthen your partnership and ensure you're both aligned on key relationship values."
        
        elif 'responsible' in category_name and 'parenthood' in category_name:
            has_children = couple_profile.get('past_children', False)
            children_count = couple_profile.get('children', 0)
            
            if score > 0.7:
                if has_children:
                    return f"Your responses indicate significant concerns about responsible parenthood ({int(score * 100)}% priority), particularly important given your experience with {children_count} child{'ren' if children_count > 1 else ''}. Intensive parenting counseling would be highly beneficial to address these concerns."
                else:
                    return f"Your responses show significant concerns about responsible parenthood ({int(score * 100)}% priority). This suggests the need for comprehensive parenting preparation and education before starting a family."
            else:
                return f"Your responses indicate some areas for development in responsible parenthood ({int(score * 100)}% priority). This is an excellent opportunity for parenting education and preparation to ensure you're both ready for the responsibilities of parenthood."
        
        elif 'planning' in category_name and 'family' in category_name:
            if score > 0.7:
                return f"Your responses reveal significant concerns about family planning ({int(score * 100)}% priority). This suggests the need for comprehensive family planning education and counseling to ensure you're both prepared for important reproductive health decisions."
            else:
                return f"Your responses show some areas for development in family planning ({int(score * 100)}% priority). This presents an opportunity for education and counseling to ensure you're both informed and aligned on family planning decisions."
        
        elif 'maternal' in category_name or 'neonatal' in category_name or 'child health' in category_name:
            if score > 0.7:
                return f"Your responses indicate significant concerns about maternal and child health ({int(score * 100)}% priority). This suggests the need for comprehensive health education and counseling to ensure you're both prepared for the health aspects of family life."
            else:
                return f"Your responses show some areas for development in maternal and child health awareness ({int(score * 100)}% priority). This presents an opportunity for health education and counseling to ensure you're both informed about important health considerations."
        
        return f"Your responses indicate {int(score * 100)}% priority for {category['name']} development, suggesting this area would benefit from focused attention and counseling."
    
    def _generate_counseling_approach_nlg(self, risk_level: str, alignment_score: float, 
                                        conflict_ratio: float, couple_profile: Dict) -> str:
        """Generate natural language counseling approach recommendation"""
        
        approaches = []
        
        if risk_level == 'High':
            approaches.append("intensive, structured counseling")
        elif risk_level == 'Medium':
            approaches.append("proactive, focused counseling")
        else:
            approaches.append("preventive, relationship-building counseling")
        
        if alignment_score < 0.5:
            approaches.append("values clarification and alignment work")
        
        if conflict_ratio > 0.3:
            approaches.append("conflict resolution and communication skills training")
        
        
        # Add demographic considerations
        age_gap = abs(couple_profile.get('male_age', 30) - couple_profile.get('female_age', 30))
        if age_gap > 10:
            approaches.append("age difference dynamics counseling")
        
        civil_status = couple_profile.get('civil_status', 'Single')
        if civil_status in ['Widowed', 'Separated', 'Divorced']:
            approaches.append("previous relationship healing and fresh start focus")
        
        if len(approaches) > 1:
            approach_text = ", ".join(approaches[:-1]) + f", and {approaches[-1]}"
        else:
            approach_text = approaches[0] if approaches else "general relationship counseling"
        
        return f"Based on your unique relationship profile, I recommend a counseling approach that focuses on {approach_text}. This tailored approach will address your specific needs while building on your relationship strengths."
    
    def generate_empathic_intro(self, couple_profile: Dict) -> str:
        """Generate an empathic introduction to the recommendations"""
        male_name = couple_profile.get('male_name', 'the male partner')
        female_name = couple_profile.get('female_name', 'the female partner')
        
        intros = [
            f"Thank you, {male_name} and {female_name}, for participating in this comprehensive relationship assessment. Your responses provide valuable insights into your relationship dynamics and areas for growth.",
            f"Based on your responses, {male_name} and {female_name}, I can see that you both care deeply about your relationship and are committed to building a strong foundation together.",
            f"Your participation in this assessment, {male_name} and {female_name}, demonstrates your dedication to understanding and improving your relationship. This commitment is the first step toward building an even stronger partnership."
        ]
        
        return random.choice(intros)
    
    def generate_encouraging_conclusion(self, risk_level: str) -> str:
        """Generate an encouraging conclusion based on risk level"""
        if risk_level == 'High':
            conclusions = [
                "Remember that every relationship faces challenges, and seeking help is a sign of strength, not weakness. With commitment and professional guidance, you can build a stronger foundation together.",
                "While the assessment reveals some areas of concern, these challenges also present opportunities for growth and deeper connection. Your willingness to address these issues shows your commitment to each other.",
                "The road ahead may require effort and dedication, but many couples have successfully navigated similar challenges and built stronger relationships as a result. You have the power to create positive change."
            ]
        elif risk_level == 'Medium':
            conclusions = [
                "Your relationship shows both strengths and areas for growth, which is completely normal. The fact that you're addressing these areas proactively shows your commitment to building a strong partnership.",
                "Every relationship has areas for development, and your willingness to work on these together is a wonderful sign of your dedication to each other. This proactive approach will serve you well.",
                "Your relationship has a solid foundation with specific areas for improvement. This balanced profile suggests you're well-positioned for growth and development together."
            ]
        else:
            conclusions = [
                "Your relationship shows excellent compatibility and healthy patterns. Continue building on these strengths while remaining open to continued growth and development together.",
                "You have a wonderful foundation for a successful marriage. Your strong compatibility and communication patterns suggest you're well-prepared for the journey ahead.",
                "Your relationship demonstrates healthy dynamics and strong compatibility. This solid foundation will serve you well as you continue to grow and develop together."
            ]
        
        return random.choice(conclusions)
