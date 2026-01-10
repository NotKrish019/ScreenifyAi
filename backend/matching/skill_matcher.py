"""
Skill matching module that compares JD requirements with Resume skills.
"""

from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
import logging

from utils.data_loader import data_loader
from matching.skill_extractor import skill_extractor
from config.weights_config import SKILL_WEIGHTS

logger = logging.getLogger(__name__)

@dataclass
class SkillMatchResult:
    """Result of skill matching."""
    score: float
    matched_core: List[str]
    missing_core: List[str]
    matched_preferred: List[str]
    extra_relevant: List[str]
    skill_coverage: float


class SkillMatcher:
    """Match skills between JD and Resume."""
    
    def __init__(self):
        self.data = data_loader
        self.extractor = skill_extractor
    
    def match_skills(
        self,
        jd_skills: List[str],
        resume_skills: List[str],
        categorize: bool = True
    ) -> SkillMatchResult:
        """Match resume skills against JD requirements."""
        
        # Normalize
        jd_normalized = set(self.extractor.normalize_skill_list(jd_skills))
        resume_normalized = set(self.extractor.normalize_skill_list(resume_skills))
        
        # Calculate matches
        matched = list(jd_normalized.intersection(resume_normalized))
        missing = list(jd_normalized - resume_normalized)
        extra = list(resume_normalized - jd_normalized)
        
        # Filter extra for relevance
        extra_relevant = [
            skill for skill in extra
            if self._is_skill_relevant(skill)
        ]
        
        # Calculate scores
        if jd_normalized:
            coverage = len(matched) / len(jd_normalized)
            weighted_score = self._calculate_weighted_score(
                matched, missing, extra_relevant, len(jd_skills)
            )
        else:
            coverage = 1.0
            weighted_score = 1.0
        
        return SkillMatchResult(
            score=weighted_score,
            matched_core=matched[:20],  # Limit
            missing_core=missing[:10],
            matched_preferred=[],
            extra_relevant=extra_relevant[:10],
            skill_coverage=coverage
        )
    
    def _is_skill_relevant(self, skill: str) -> bool:
        """Check if extra skill is relevant."""
        skill_info = self.data.get_skill_info(skill)
        if skill_info:
            return skill_info.get('weight', 0) >= 0.5
        return False
    
    def _calculate_weighted_score(
        self,
        matched: List[str],
        missing: List[str],
        extra_relevant: List[str],
        total_required: int
    ) -> float:
        """Calculate weighted skill match score."""
        if total_required == 0:
            return 1.0
        
        # Base score from matches
        score = len(matched) / total_required
        
        # Penalty for missing (up to -20%)
        penalty = min(0.2, len(missing) * 0.05)
        
        # Bonus for extra relevant (up to +10%)
        bonus = min(0.1, len(extra_relevant) * 0.02)
        
        return max(0, min(1.0, score - penalty + bonus))


# Singleton
skill_matcher = SkillMatcher()
