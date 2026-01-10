"""
Skill extraction module that identifies and normalizes skills
from JDs and resumes using the skills_dataset.json.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import logging

from utils.data_loader import data_loader
from matching.text_normalizer import text_normalizer

logger = logging.getLogger(__name__)

class SkillExtractor:
    """Extract and normalize skills from text."""
    
    def __init__(self):
        self.data = data_loader
        self._build_skill_patterns()
    
    def _build_skill_patterns(self):
        """Build regex patterns for skill matching."""
        # Get all skill names and aliases
        all_skills = []
        
        # Iterate through the skill lookup
        for normalized, info in self.data._skill_lookup.items():
            canonical = info['canonical']
            if canonical not in all_skills:
                all_skills.append(canonical)
        
        # Sort by length (longest first) for greedy matching
        all_skills.sort(key=len, reverse=True)
        
        # Build pattern (escape special regex characters)
        escaped = [re.escape(skill) for skill in all_skills]
        self.skill_pattern = re.compile(
            r'\b(' + '|'.join(escaped) + r')\b',
            re.IGNORECASE
        )
    
    def extract_skills(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract all skills from text.
        
        Args:
            text: Input text (JD or Resume)
            
        Returns:
            Dictionary with categorized skills
        """
        if not text:
            return {}
        
        # Normalize text
        normalized_text = text_normalizer.normalize_text(text)
        
        # Track found skills
        found_skills = defaultdict(list)
        skill_counts = defaultdict(int)
        seen_canonical = set()
        
        # Pattern matching against known skills
        matches = self.skill_pattern.finditer(normalized_text)
        for match in matches:
            skill_text = match.group(1)
            skill_info = self.data.get_skill_info(skill_text)
            
            if skill_info:
                canonical = skill_info['canonical']
                skill_counts[canonical] += 1
                
                if canonical not in seen_canonical:
                    seen_canonical.add(canonical)
                    found_skills[skill_info['category']].append({
                        "name": skill_text,
                        "canonical": canonical,
                        "weight": skill_info['weight'],
                        "count": 0
                    })
        
        # Update counts
        for category, skills in found_skills.items():
            for skill in skills:
                skill['count'] = skill_counts[skill['canonical']]
        
        return dict(found_skills)
    
    def extract_skills_flat(self, text: str) -> List[str]:
        """Extract skills as a flat list of canonical names."""
        categorized = self.extract_skills(text)
        flat = []
        for skills in categorized.values():
            for skill in skills:
                flat.append(skill['canonical'])
        return flat
    
    def normalize_skill_list(self, skills: List[str]) -> List[str]:
        """Normalize a list of skill names to their canonical forms."""
        normalized = []
        seen = set()
        
        for skill in skills:
            canonical = self.data.get_canonical_skill(skill)
            if canonical and canonical not in seen:
                normalized.append(canonical)
                seen.add(canonical)
            elif not canonical and skill not in seen:
                normalized.append(skill)
                seen.add(skill)
        
        return normalized


# Singleton instance
skill_extractor = SkillExtractor()
