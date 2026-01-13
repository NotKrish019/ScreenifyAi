
import re
from typing import Dict, List, Set, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Import Gemini Service
from gemini_service import gemini_service

class HybridScorer:
    """
    Hybrid Scoring Engine:
    - 50% Statistical (Keyword Match + Cosine Similarity)
    - 50% AI (Gemini Analysis)
    """
    
    def __init__(self):
        # Improved Vectorizer: Captures phrases (e.g. "machine learning") and ignores common noise
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Capture Bigrams
            max_features=5000
        )

    def score_resume(self, resume_text: str, jd_text: str, resume_name: str = "Candidate") -> Dict:
        # Better text cleaning that preserves C++, C#, .NET
        clean_resume = self._clean_text(resume_text)
        clean_jd = self._clean_text(jd_text)
        
        # --- PART 1: Statistical Scoring (Max 50) ---
        stat_score, stat_details = self._calculate_statistical_score(clean_resume, clean_jd)
        
        # --- PART 2: Gemini AI Scoring (Max 50) ---
        # We pass the original text to AI for full context
        ai_result = gemini_service.evaluate_candidate(resume_text, jd_text)
        ai_score = ai_result.get("ai_score", 0)
        
        # --- Final Compilation ---
        final_score = stat_score + ai_score
        
        return {
            "final_score": round(final_score, 1),
            "breakdown": {
                "statistical_score": round(stat_score, 1),
                "ai_score": round(ai_score, 1),
                "stat_details": stat_details,
                "ai_details": ai_result
            },
            "match_category": self._get_category(final_score, stat_score, ai_score)
        }

    def _calculate_statistical_score(self, resume_text: str, jd_text: str) -> Tuple[float, Dict]:
        """
        Calculates score based on 50% Keyword Overlap + 50% Cosine Similarity.
        Total max output = 50.
        """
        try:
            # 1. Cosine Similarity (0-1) 
            # We fit on both to ensure vocabulary covers both docs
            tfidf_matrix = self.vectorizer.fit_transform([jd_text, resume_text])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Boost Semantic Score: 
            # Raw cosine is often low (0.2-0.4) for decent matches. We scale it up.
            # A score of 0.3 should be considered "good" (~70% points).
            # Formula: min(1.0, cosine * 2.5) * 25
            similarity_points = min(1.0, cosine_sim * 2.5) * 25.0
            
            # 2. Keyword Extraction & Overlap
            jd_keywords = self._extract_top_keywords(jd_text, top_n=30)
            resume_words_set = set(self._tokenize(resume_text))
            
            matched_keywords = [k for k in jd_keywords if k in resume_words_set]
            
            if len(jd_keywords) > 0:
                raw_ratio = len(matched_keywords) / len(jd_keywords)
            else:
                raw_ratio = 0
                
            # Boost Keyword Score: 
            # Matching 50% of top JD words is EXCELLENT. 
            # Formula: min(1.0, ratio * 2.0) * 25
            overlap_points = min(1.0, raw_ratio * 2.0) * 25.0
            
            total_stat_score = similarity_points + overlap_points
            
            return total_stat_score, {
                "cosine_similarity": round(cosine_sim, 2),
                "similarity_points": round(similarity_points, 1),
                "overlap_ratio": round(raw_ratio, 2),
                "overlap_points": round(overlap_points, 1),
                "jd_keywords": jd_keywords[:10], # Show top 10 for brevity in debug
                "matched_keywords": matched_keywords
            }
            
        except Exception as e:
            print(f"Statistical scoring error: {e}")
            return 0.0, {"error": str(e)}

    def _extract_top_keywords(self, text: str, top_n: int = 30) -> List[str]:
        # Blocklist: Common words in JDs that are NOT actual skills
        NON_SKILL_WORDS = {
            # Generic JD phrases
            'description', 'job', 'position', 'role', 'candidate', 'seeking', 
            'looking', 'required', 'requirements', 'qualifications', 'responsibilities',
            'experience', 'years', 'year', 'intern', 'internship', 'opportunity',
            'team', 'work', 'working', 'environment', 'company', 'organization',
            'ability', 'skills', 'skill', 'strong', 'excellent', 'good', 'great',
            'knowledge', 'understanding', 'familiar', 'familiarity', 'preferred',
            # General actions
            'develop', 'development', 'create', 'build', 'building', 'design',
            'implement', 'implementation', 'manage', 'management', 'support',
            'collaborate', 'communication', 'problem', 'solving', 'analytical',
            # Locations / time
            'location', 'remote', 'onsite', 'hybrid', 'full', 'time', 'part',
            # Resume common words
            'resume', 'cv', 'cover', 'letter', 'application', 'apply',
            # Misc noise
            'etc', 'including', 'includes', 'include', 'related', 'various',
            'using', 'used', 'use', 'will', 'would', 'should', 'must', 'can',
            'able', 'need', 'needs', 'have', 'has', 'having', 'like', 
            'well', 'new', 'multiple', 'provide', 'ensure', 'help', 
            'based', 'backend', 'frontend', 'end'  # Generic, not specific tech
        }
        
        words = self._tokenize(text)
        filtered = [
            w for w in words 
            if w not in ENGLISH_STOP_WORDS 
            and w not in NON_SKILL_WORDS
            and len(w) > 2
        ]
        return [word for word, count in Counter(filtered).most_common(top_n)]

    def _tokenize(self, text: str) -> List[str]:
        # Improved regex to keep C++, C#, .NET etc.
        # Find words that might contain +, #, . inside them
        return re.findall(r'\b[a-zA-Z][a-zA-Z0-9\+\#\.]*\b', text.lower())

    def _clean_text(self, text: str) -> str:
        # Keep alphanumeric and specific skill chars
        text = re.sub(r'[^a-zA-Z0-9\s\+\#\.]', ' ', text)
        return ' '.join(text.split())

    def _get_category(self, final: float, stat: float, ai: float) -> str:
        if final >= 80: return "Top Tier ⭐"
        if final >= 65: return "Strong Match"
        if final >= 45: return "Potential Fit"
        return "Weak Match"

hybrid_scorer = HybridScorer()
