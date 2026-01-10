"""
Role detection module that identifies the most relevant role
from a Job Description using job_roles.json.
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

from utils.data_loader import data_loader
from matching.text_normalizer import text_normalizer
from matching.skill_extractor import skill_extractor

logger = logging.getLogger(__name__)

class RoleDetector:
    """Detect and match job roles from JD text."""
    
    def __init__(self):
        self.data = data_loader
        self._build_role_patterns()
    
    def _build_role_patterns(self):
        """Build patterns for role detection."""
        self.role_titles = []
        
        for role_key, role_data in self.data._job_roles.items():
            title = role_data.get('title', role_key)
            aliases = role_data.get('aliases', [])
            
            self.role_titles.append((title, role_key))
            for alias in aliases:
                self.role_titles.append((alias, role_key))
        
        # Sort by length (longest first)
        self.role_titles.sort(key=lambda x: len(x[0]), reverse=True)
    
    def detect_role(self, jd_text: str) -> Dict:
        """
        Detect the primary role from a Job Description.
        
        Returns:
            {
                "detected_role_key": "software_engineer",
                "detected_role_title": "Software Engineer",
                "confidence": 0.85,
                "match_reasons": [...],
                "role_data": { ... }
            }
        """
        normalized_jd = text_normalizer.normalize_text(jd_text).lower()
        
        # Score each role
        role_scores = {}
        
        for role_key, role_data in self.data._job_roles.items():
            score, reasons = self._score_role_match(
                normalized_jd, jd_text, role_key, role_data
            )
            role_scores[role_key] = {
                "score": score,
                "reasons": reasons,
                "role_data": role_data
            }
        
        # Sort by score
        sorted_roles = sorted(
            role_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        if not sorted_roles or sorted_roles[0][1]['score'] < 0.1:
            return self._create_unknown_role_response(jd_text)
        
        # Get top role
        top_role_key, top_role_info = sorted_roles[0]
        
        # Calculate confidence
        if len(sorted_roles) > 1:
            score_gap = top_role_info['score'] - sorted_roles[1][1]['score']
            confidence = min(0.95, 0.5 + score_gap)
        else:
            confidence = 0.9 if top_role_info['score'] > 0.5 else 0.6
        
        return {
            "detected_role_key": top_role_key,
            "detected_role_title": top_role_info['role_data'].get('title', top_role_key),
            "confidence": round(confidence, 2),
            "match_reasons": top_role_info['reasons'],
            "role_data": top_role_info['role_data']
        }
    
    def _score_role_match(
        self,
        normalized_jd: str,
        original_jd: str,
        role_key: str,
        role_data: Dict
    ) -> Tuple[float, List[str]]:
        """Score how well a JD matches a specific role."""
        score = 0.0
        reasons = []
        
        # Title match (30%)
        title = role_data.get('title', '').lower()
        aliases = [a.lower() for a in role_data.get('aliases', [])]
        
        if title and title in normalized_jd:
            score += 0.30
            reasons.append(f"Title match: {title}")
        else:
            for alias in aliases:
                if alias in normalized_jd:
                    score += 0.25
                    reasons.append(f"Alias match: {alias}")
                    break
        
        # Skill match (35%)
        jd_skills = set(skill_extractor.extract_skills_flat(original_jd))
        role_skills = set(skill_extractor.normalize_skill_list(
            role_data.get('core_skills', [])
        ))
        
        if role_skills:
            overlap = len(jd_skills.intersection(role_skills))
            skill_score = overlap / len(role_skills)
            score += skill_score * 0.35
            if skill_score > 0.3:
                reasons.append(f"Skill overlap: {skill_score:.0%}")
        
        # Keywords (35%)
        keywords = role_data.get('keywords', [])
        if keywords:
            matched = sum(1 for kw in keywords if kw.lower() in normalized_jd)
            kw_score = matched / len(keywords)
            score += kw_score * 0.35
        
        return score, reasons
    
    def _create_unknown_role_response(self, jd_text: str) -> Dict:
        """Response when no role detected."""
        skills = skill_extractor.extract_skills_flat(jd_text)
        
        return {
            "detected_role_key": "unknown",
            "detected_role_title": "General Position",
            "confidence": 0.3,
            "match_reasons": ["No strong role match found"],
            "role_data": {
                "core_skills": skills[:10],
                "secondary_skills": skills[10:20]
            }
        }


# Singleton
role_detector = RoleDetector()
