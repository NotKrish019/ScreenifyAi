"""
NLP Engine Module
Handles text preprocessing, tokenization, and linguistic analysis.
Uses NLTK for core NLP operations.
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Set, Dict, Optional, Tuple
from collections import Counter

# NLTK imports
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK datasets."""
    datasets = [
        'punkt',
        'stopwords', 
        'wordnet',
        'averaged_perceptron_tagger',
        'omw-1.4'
    ]
    for dataset in datasets:
        try:
            nltk.download(dataset, quiet=True)
        except Exception as e:
            logging.warning(f"Failed to download NLTK dataset {dataset}: {e}")

# Initialize NLTK data
download_nltk_data()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import config
from config import DATASETS_DIR, JOB_ROLES_PATH, SKILLS_DATASET_PATH


class NLPEngine:
    """
    NLP Engine for text preprocessing and analysis.
    Provides methods for cleaning, tokenizing, and extracting features from text.
    """
    
    def __init__(self):
        """Initialize NLP engine with required resources."""
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Load stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        # Add custom stopwords relevant to resumes
        self.custom_stopwords = {
            'resume', 'cv', 'curriculum', 'vitae', 'page', 'phone', 
            'email', 'address', 'date', 'reference', 'available',
            'upon', 'request', 'objective', 'summary'
        }
        self.stop_words.update(self.custom_stopwords)
        
        # Load datasets
        self.job_roles = self._load_job_roles()
        self.skills_data = self._load_skills_dataset()
        self.all_skills = self._compile_all_skills()
        
        logger.info("NLP Engine initialized successfully")
    
    def _load_job_roles(self) -> Dict:
        """Load job roles dataset."""
        try:
            with open(JOB_ROLES_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data.get('job_roles', {}))} job roles")
            return data
        except Exception as e:
            logger.error(f"Failed to load job roles: {e}")
            return {"job_roles": {}, "experience_levels": {}, "education_keywords": {}}
    
    def _load_skills_dataset(self) -> Dict:
        """Load skills dataset."""
        try:
            with open(SKILLS_DATASET_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} skill categories")
            return data
        except Exception as e:
            logger.error(f"Failed to load skills dataset: {e}")
            return {}
    
    def _compile_all_skills(self) -> Set[str]:
        """Compile a set of all skills from the dataset."""
        all_skills = set()
        for category, data in self.skills_data.items():
            if isinstance(data, dict) and 'skills' in data:
                all_skills.update([s.lower() for s in data['skills']])
        return all_skills
    
    def preprocess(self, text: str, 
                   lowercase: bool = True,
                   remove_stopwords: bool = True,
                   lemmatize: bool = True,
                   remove_special_chars: bool = True) -> str:
        """
        Preprocess text with various NLP techniques.
        
        Args:
            text: Input text to preprocess
            lowercase: Convert to lowercase
            remove_stopwords: Remove stop words
            lemmatize: Apply lemmatization
            remove_special_chars: Remove special characters
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Lowercase
        if lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses (but remember them for later)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep alphanumeric and basic punctuation
        if remove_special_chars:
            text = re.sub(r'[^\w\s\-\.\+\#]', ' ', text)
        
        # Remove numbers that are standalone (keep those attached to words)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Remove stopwords
        if remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in self.stop_words]
        
        # Lemmatize
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        
        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 1]
        
        return ' '.join(tokens)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        try:
            return word_tokenize(text.lower())
        except:
            return text.lower().split()
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """
        Extract skills from text based on the skills dataset.
        Uses strict word boundary matching to avoid false positives.
        
        Args:
            text: Text to extract skills from
            
        Returns:
            Dictionary with skill categories and matched skills
        """
        import re
        
        text_lower = text.lower()
        
        # Skills that need special handling due to being short or ambiguous
        # These require explicit context like "R language", "R programming", etc.
        ambiguous_skills = {
            'r': [r'\br\s+programming', r'\br\s+language', r'\br\s+studio', r'rstudio', 
                  r'r\s+statistical', r'programming\s+in\s+r\b', r'\br\s+for\s+data'],
            'c': [r'\bc\s+programming', r'\bc\s+language', r'c/c\+\+', r'\bc\s+and\s+c\+\+',
                  r'programming\s+in\s+c\b', r'\bc\s+development'],
            'go': [r'\bgolang\b', r'\bgo\s+programming', r'\bgo\s+language', r'written\s+in\s+go\b',
                   r'\bgo\s+development', r'\bgo\s+developer'],
        }
        
        extracted = {}
        
        for category, data in self.skills_data.items():
            if not isinstance(data, dict) or 'skills' not in data:
                continue
                
            category_skills = []
            for skill in data['skills']:
                skill_lower = skill.lower()
                skill_found = False
                
                # Handle ambiguous/short skills with special patterns
                if skill_lower in ambiguous_skills:
                    patterns = ambiguous_skills[skill_lower]
                    for pattern in patterns:
                        if re.search(pattern, text_lower):
                            skill_found = True
                            break
                else:
                    # For normal skills, use word boundary matching
                    # Escape special regex characters in skill name
                    escaped_skill = re.escape(skill_lower)
                    
                    # Create pattern with word boundaries
                    # Allow for variations like .js, -js for frameworks
                    pattern = r'\b' + escaped_skill + r'\b'
                    
                    # Also check for common variations
                    variations = [pattern]
                    
                    # Add .js variation for JS frameworks
                    if skill_lower.endswith('js'):
                        base = skill_lower[:-2]
                        variations.append(r'\b' + re.escape(base) + r'\.js\b')
                    
                    # Check all patterns
                    for pat in variations:
                        if re.search(pat, text_lower):
                            skill_found = True
                            break
                
                if skill_found:
                    category_skills.append(skill)
            
            if category_skills:
                extracted[category] = list(set(category_skills))
        
        return extracted
    
    def extract_skills_flat(self, text: str) -> List[str]:
        """Extract skills as a flat list."""
        skills_dict = self.extract_skills(text)
        all_skills = []
        for skills in skills_dict.values():
            all_skills.extend(skills)
        return list(set(all_skills))
    
    def detect_job_role(self, text: str) -> Tuple[str, float]:
        """
        Detect the job role from text.
        
        Args:
            text: Job description or resume text
            
        Returns:
            Tuple of (detected_role, confidence_score)
        """
        text_lower = text.lower()
        best_match = None
        best_score = 0
        
        # Combine standard and additional job roles
        standard_roles = self.job_roles.get('job_roles', {})
        additional_roles = self.job_roles.get('additional_job_roles', {})
        
        # Iterate through all available roles
        all_roles = {**standard_roles, **additional_roles}
        
        for role_key, role_data in all_roles.items():
            # Skip invalid entries
            if not isinstance(role_data, dict) or 'title' not in role_data:
                continue
                
            score = 0
            
            # Check title
            if role_data['title'].lower() in text_lower:
                score += 10
            
            # Check aliases
            for alias in role_data.get('aliases', []):
                if alias.lower() in text_lower:
                    score += 5
                    break
            
            # Check core skills
            for skill in role_data.get('core_skills', []):
                if skill.lower() in text_lower:
                    score += 2
            
            # Check experience keywords
            for keyword in role_data.get('experience_keywords', []):
                if keyword.lower() in text_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = role_data['title']
        
        # Normalize confidence score (0-1)
        confidence = min(best_score / 30, 1.0)
        
        return best_match or "General", confidence
    
    def detect_experience_level(self, text: str) -> Tuple[str, Optional[int]]:
        """
        Detect experience level from text.
        
        Args:
            text: Resume or job description text
            
        Returns:
            Tuple of (level_name, years_if_detected)
        """
        text_lower = text.lower()
        
        # Try to extract years of experience
        year_patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)',
            r'(?:experience|exp)(?:\s+of)?\s*[:;]?\s*(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:in|of|working)',
        ]
        
        years = None
        for pattern in year_patterns:
            match = re.search(pattern, text_lower)
            if match:
                years = int(match.group(1))
                break
        
        # Determine level based on years or keywords
        experience_levels = self.job_roles.get('experience_levels', {})
        
        if years is not None:
            for level, data in experience_levels.items():
                min_years, max_years = data.get('years_range', [0, 100])
                if min_years <= years <= max_years:
                    return level.capitalize(), years
        
        # Fall back to keyword matching
        for level, data in experience_levels.items():
            for keyword in data.get('keywords', []):
                if keyword.lower() in text_lower:
                    return level.capitalize(), years
        
        return "Not specified", years
    
    def extract_keywords(self, text: str, top_n: int = 20) -> List[Tuple[str, int]]:
        """
        Extract top keywords from text.
        
        Args:
            text: Text to extract keywords from
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, frequency) tuples
        """
        # Preprocess
        processed = self.preprocess(text)
        tokens = self.tokenize(processed)
        
        # Filter to meaningful words
        meaningful_tokens = [
            t for t in tokens 
            if len(t) > 2 and t.isalpha()
        ]
        
        # Count frequencies
        freq = Counter(meaningful_tokens)
        
        return freq.most_common(top_n)
    
    def calculate_text_similarity_keywords(self, text1: str, text2: str) -> float:
        """
        Calculate similarity based on keyword overlap.
        Simple Jaccard similarity for quick comparison.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        tokens1 = set(self.tokenize(self.preprocess(text1)))
        tokens2 = set(self.tokenize(self.preprocess(text2)))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union)
    
    def get_skill_categories(self) -> List[str]:
        """Get list of skill categories."""
        return list(self.skills_data.keys())
    
    def is_ready(self) -> bool:
        """Check if NLP engine is ready."""
        return bool(self.skills_data) and bool(self.job_roles)


# Global NLP engine instance
nlp_engine = NLPEngine()