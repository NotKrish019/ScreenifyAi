
# backend/intern_skill_config.py
"""
FIXED SKILL CONFIGURATION FOR DATA SCIENCE INTERNSHIP
This file defines the EXACT skills used for scoring - no JD parsing variance.
"""

from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class SkillConfig:
    """Immutable skill configuration for consistent scoring."""
    
    # ═══════════════════════════════════════════════════════════════════════
    # CORE SKILLS - FIXED LIST (6 SKILLS)
    # These are the ONLY skills that count toward the 55% core weight.
    # The denominator is ALWAYS 6, preventing score compression.
    # ═══════════════════════════════════════════════════════════════════════
    
    CORE_SKILLS: List[str] = None
    
    # ═══════════════════════════════════════════════════════════════════════
    # PREFERRED SKILLS - SUPPLEMENTARY (7 SKILLS)
    # These contribute to the 20% preferred weight.
    # ═══════════════════════════════════════════════════════════════════════
    
    PREFERRED_SKILLS: List[str] = None
    
    # ═══════════════════════════════════════════════════════════════════════
    # SEMANTIC MAPPINGS - What counts as "implicit" evidence of a skill
    # ═══════════════════════════════════════════════════════════════════════
    
    SEMANTIC_INDICATORS: Dict[str, Set[str]] = None
    
    def __post_init__(self):
        """Initialize with default Data Science Intern configuration."""
        
        self.CORE_SKILLS = [
            "python",
            "machine learning",
            "data analysis", 
            "pandas",
            "numpy",
            "scikit-learn"
        ]
        
        self.PREFERRED_SKILLS = [
            "sql",
            "git",
            "tensorflow",
            "statistics",
            "data visualization",
            "jupyter",
            "deep learning"
        ]
        
        # Semantic indicators: if resume has ANY of these, 
        # grant 0.5 credit for the parent skill
        self.SEMANTIC_INDICATORS = {
            "machine learning": {
                "random forest", "decision tree", "xgboost", "lightgbm",
                "gradient boosting", "logistic regression", "linear regression",
                "svm", "support vector", "naive bayes", "knn", "k-nearest",
                "classification", "regression", "clustering", "k-means",
                "supervised learning", "unsupervised learning", "model training",
                "hyperparameter", "cross validation", "train test split",
                "feature engineering", "feature selection", "ml pipeline",
                "ensemble", "boosting", "bagging",
                # FIX: Deep Learning implies Machine Learning
                "deep learning", "neural network", "tensorflow", "pytorch", "keras",
                "cnn", "rnn", "lstm", "transformer", "bert", "gpt", "llm",
                "computer vision", "nlp", "natural language processing"
            },
            
            "deep learning": {
                "neural network", "tensorflow", "pytorch", "keras",
                "cnn", "rnn", "lstm", "transformer", "attention",
                "backpropagation", "activation function", "dropout",
                "convolutional", "recurrent", "autoencoder", "gan"
            },
            
            "data analysis": {
                "eda", "exploratory data analysis", "data cleaning",
                "data preprocessing", "data wrangling", "data manipulation",
                "missing values", "outlier detection", "statistical analysis",
                "descriptive statistics", "data exploration", "insights",
                "correlation", "distribution analysis"
            },
            
            "python": {
                "pandas", "numpy", "scipy", "matplotlib", "seaborn",
                "jupyter", "ipython", "anaconda", "pip", "conda",
                "python3", "py"
            },
            
            "data visualization": {
                "matplotlib", "seaborn", "plotly", "bokeh", "tableau",
                "power bi", "charts", "graphs", "dashboard", "ggplot"
            },
            
            "statistics": {
                "probability", "hypothesis testing", "a/b testing",
                "p-value", "confidence interval", "t-test", "chi-square",
                "anova", "regression analysis", "correlation", "variance",
                "standard deviation", "mean", "median", "distribution",
                "normal distribution", "statistical significance"
            },
            
            "sql": {
                "mysql", "postgresql", "sqlite", "database", "query",
                "queries", "joins", "select", "where", "group by"
            },
            
            "git": {
                "github", "gitlab", "bitbucket", "version control",
                "repository", "commit", "branch", "merge", "pull request"
            },
            
            "nlp": {
                "natural language processing", "text mining", "sentiment analysis",
                "tokenization", "lemmatization", "nltk", "spacy", "word2vec",
                "tfidf", "tf-idf", "cosine similarity", "text classification",
                "named entity recognition", "ner"
            }
        }
    
    @property
    def core_count(self) -> int:
        """Fixed denominator for core skills."""
        return len(self.CORE_SKILLS)
    
    @property
    def preferred_count(self) -> int:
        """Fixed denominator for preferred skills."""
        return len(self.PREFERRED_SKILLS)


# Singleton instance
INTERN_CONFIG = SkillConfig()


class SkillMatcher:
    """
    Matches resume skills against fixed skill buckets.
    Provides explicit and semantic match detection.
    """
    
    def __init__(self, config: SkillConfig = None):
        self.config = config or INTERN_CONFIG
        
        # Pre-compile lowercase versions for matching
        self._core_lower = {s.lower() for s in self.config.CORE_SKILLS}
        self._pref_lower = {s.lower() for s in self.config.PREFERRED_SKILLS}
        
        # Flatten semantic indicators for quick lookup
        self._build_semantic_index()
    
    def _build_semantic_index(self):
        """Build reverse index: indicator → parent skill."""
        self._indicator_to_skill = {}
        for skill, indicators in self.config.SEMANTIC_INDICATORS.items():
            for indicator in indicators:
                self._indicator_to_skill[indicator.lower()] = skill.lower()
    
    def match_skills(self, resume_skills: Set[str]) -> Dict:
        """
        Match resume skills against core and preferred buckets.
        
        Returns:
            {
                "core": {
                    "explicit": ["python", "pandas"],
                    "semantic": ["machine learning"],  # Inferred from RF, SVM
                    "missing": ["numpy"],
                    "explicit_count": 2,
                    "semantic_count": 1,
                    "total_credit": 2.5,  # 2 × 1.0 + 1 × 0.5
                    "max_credit": 6.0,
                    "percentage": 41.7
                },
                "preferred": {...}
            }
        """
        resume_lower = {s.lower().strip() for s in resume_skills}
        
        core_result = self._match_bucket(
            resume_lower, 
            self._core_lower,
            "core"
        )
        
        pref_result = self._match_bucket(
            resume_lower,
            self._pref_lower,
            "preferred"
        )
        
        return {
            "core": core_result,
            "preferred": pref_result
        }
    
    def _match_bucket(
        self, 
        resume_skills: Set[str], 
        bucket_skills: Set[str],
        bucket_name: str
    ) -> Dict:
        """Match against a single skill bucket."""
        
        explicit_matches = []
        semantic_matches = []
        missing = []
        
        for skill in bucket_skills:
            # Check explicit match
            if self._has_explicit_match(skill, resume_skills):
                explicit_matches.append(skill)
            # Check semantic match
            elif self._has_semantic_match(skill, resume_skills):
                semantic_matches.append(skill)
            else:
                missing.append(skill)
        
        # Calculate credit
        explicit_count = len(explicit_matches)
        semantic_count = len(semantic_matches)
        # FIX: MODE A - Inclusive Semantic Credit (0.9)
        total_credit = explicit_count * 1.0 + semantic_count * 0.9
        max_credit = len(bucket_skills)
        
        percentage = (total_credit / max_credit * 100) if max_credit > 0 else 0
        
        return {
            "explicit": explicit_matches,
            "semantic": semantic_matches,
            "missing": missing,
            "explicit_count": explicit_count,
            "semantic_count": semantic_count,
            "total_credit": total_credit,
            "max_credit": max_credit,
            "percentage": round(percentage, 1)
        }
    
    def _has_explicit_match(self, skill: str, resume_skills: Set[str]) -> bool:
        """Check if skill is explicitly mentioned."""
        skill_lower = skill.lower()
        
        # Direct match
        if skill_lower in resume_skills:
            return True
        
        # Partial match (skill contained in resume skill)
        for rs in resume_skills:
            if skill_lower in rs or rs in skill_lower:
                return True
        
        # Handle common variations
        variations = self._get_variations(skill_lower)
        for var in variations:
            if var in resume_skills:
                return True
        
        return False
    
    def _has_semantic_match(self, skill: str, resume_skills: Set[str]) -> bool:
        """Check if skill is semantically indicated by other skills."""
        skill_lower = skill.lower()
        
        # Get semantic indicators for this skill
        indicators = self.config.SEMANTIC_INDICATORS.get(skill_lower, set())
        
        if not indicators:
            return False
        
        # Check if any indicator is present in resume
        indicators_lower = {ind.lower() for ind in indicators}
        
        for resume_skill in resume_skills:
            # Check if resume skill matches any indicator
            if resume_skill in indicators_lower:
                return True
            
            # Check partial matches
            for indicator in indicators_lower:
                if indicator in resume_skill or resume_skill in indicator:
                    return True
        
        return False
    
    def _get_variations(self, skill: str) -> Set[str]:
        """Get common variations of a skill name."""
        variations = {skill}
        
        # Common variations
        variation_map = {
            "scikit-learn": {"sklearn", "scikit learn", "scikitlearn"},
            "machine learning": {"ml", "machinelearning"},
            "deep learning": {"dl", "deeplearning"},
            "data analysis": {"data analytics", "dataanalysis"},
            "data visualization": {"data viz", "dataviz", "visualization"},
            "tensorflow": {"tf", "tensor flow"},
            "pytorch": {"torch"},
            "numpy": {"np"},
            "pandas": {"pd"},
            "natural language processing": {"nlp"},
            "computer vision": {"cv"},
            "jupyter": {"jupyter notebook", "jupyter notebooks"},
            "github": {"git hub"},
            "postgresql": {"postgres"},
        }
        
        if skill in variation_map:
            variations.update(variation_map[skill])
        
        # Reverse lookup
        for canonical, vars in variation_map.items():
            if skill in vars:
                variations.add(canonical)
                variations.update(vars)
        
        return variations
