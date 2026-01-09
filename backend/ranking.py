"""
Ranking Module
Handles the ranking logic and result generation for resume screening.
"""

import random
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from config import SCORE_THRESHOLDS, SUMMARY_TEMPLATES
from models import ResumeResult, FitCategory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CandidateScore:
    """Data class for holding candidate scoring data."""
    resume_id: str
    resume_name: str
    original_text: str
    preprocessed_text: str
    similarity_data: Dict


class RankingEngine:
    """
    Engine for ranking and scoring candidates based on similarity results.
    Generates final results with rankings, fit categories, and summaries.
    """
    
    def __init__(self):
        """Initialize the ranking engine."""
        logger.info("Ranking Engine initialized")
    
    def determine_fit_category(self, score: float) -> FitCategory:
        """
        Determine fit category based on score.
        
        Args:
            score: Match score (0-100)
            
        Returns:
            FitCategory enum value
        """
        if score >= SCORE_THRESHOLDS['high']:
            return FitCategory.HIGH
        elif score >= SCORE_THRESHOLDS['medium']:
            return FitCategory.MEDIUM
        else:
            return FitCategory.LOW
    
    def extract_candidate_info(self, resume_text: str) -> Dict:
        """
        Extract actual candidate information from resume text.
        """
        import re
        lines = resume_text.split('\n')
        info = {
            'name': None,
            'title': None,
            'experience_years': None,
            'companies': [],
            'education': [],
            'location': None
        }
        
        # Extract name (usually first non-empty line)
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) < 50 and not any(x in line.lower() for x in ['email', 'phone', '@', 'http', 'github']):
                # Check if it looks like a name (mostly letters and spaces)
                if re.match(r'^[A-Za-z\s\.\-]+$', line):
                    info['name'] = line.title()
                    break
        
        # Extract title (often second line or after name)
        title_keywords = ['developer', 'engineer', 'manager', 'analyst', 'designer', 'architect', 'consultant', 'specialist']
        for line in lines[1:10]:
            line_lower = line.lower().strip()
            if any(kw in line_lower for kw in title_keywords) and len(line) < 60:
                info['title'] = line.strip()
                break
        
        # Extract experience years
        exp_patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience[:\s]+(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s+in\s+'
        ]
        for pattern in exp_patterns:
            match = re.search(pattern, resume_text.lower())
            if match:
                info['experience_years'] = int(match.group(1))
                break
        
        # Extract companies
        company_patterns = [
            r'(?:at|@)\s+([A-Z][A-Za-z\s&]+(?:Inc|LLC|Corp|Ltd|Technologies|Systems|Solutions|Company)?)',
            r'\|\s+([A-Z][A-Za-z\s&]+)\s+\|',
            r'([A-Z][A-Za-z\s]+(?:Technologies|Systems|Solutions|Labs|Inc))'
        ]
        companies_found = set()
        for pattern in company_patterns:
            matches = re.findall(pattern, resume_text)
            for m in matches[:5]:
                if len(m) > 3 and len(m) < 40:
                    companies_found.add(m.strip())
        info['companies'] = list(companies_found)[:3]
        
        # Extract education
        edu_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college', 'institute', 'b.s.', 'm.s.', 'b.tech', 'm.tech', 'mba']
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in edu_keywords):
                edu_line = line.strip()
                if len(edu_line) > 10 and len(edu_line) < 100:
                    info['education'].append(edu_line)
        info['education'] = info['education'][:2]
        
        return info
    
    def generate_summary(self, 
                         score: float, 
                         matched_skills: List[str], 
                         missing_skills: List[str],
                         skill_breakdown: Dict,
                         resume_text: str = None,
                         resume_name: str = None) -> str:
        """
        Generate a detailed, personalized summary for the candidate.
        Actually reads the resume content to extract specific information.
        """
        fit = self.determine_fit_category(score)
        parts = []
        
        # Extract actual candidate info from resume
        candidate_info = {}
        if resume_text:
            candidate_info = self.extract_candidate_info(resume_text)
        
        # Build personalized summary
        name = candidate_info.get('name')
        title = candidate_info.get('title')
        exp_years = candidate_info.get('experience_years')
        companies = candidate_info.get('companies', [])
        education = candidate_info.get('education', [])
        
        # Opening with candidate identity
        if name:
            if title:
                parts.append(f"{name} is a {title.lower()}")
            else:
                parts.append(f"{name}")
            
            if exp_years:
                parts[-1] += f" with {exp_years}+ years of experience."
            elif companies:
                parts[-1] += f" with experience at {companies[0]}."
            else:
                parts[-1] += "."
        else:
            # Fallback if name not found
            candidate_name = resume_name.replace('.pdf', '').replace('.docx', '').replace('resume_', '').replace('_', ' ').title() if resume_name else "This candidate"
            parts.append(f"{candidate_name}")
            if exp_years:
                parts[-1] += f" has {exp_years}+ years of experience."
            else:
                parts[-1] += "."
        
        # Key skills
        if matched_skills and len(matched_skills) >= 3:
            top_skills = matched_skills[:4]
            parts.append(f"Demonstrates strong proficiency in {', '.join(top_skills[:-1])} and {top_skills[-1]}.")
        elif matched_skills:
            parts.append(f"Has skills in {', '.join(matched_skills)}.")
        
        # Education highlight
        if education:
            # Simplify education mention
            edu_short = education[0]
            if 'master' in edu_short.lower() or 'm.s.' in edu_short.lower():
                parts.append("Holds a Master's degree.")
            elif 'bachelor' in edu_short.lower() or 'b.s.' in edu_short.lower():
                parts.append("Has a Bachelor's degree in a relevant field.")
        
        # Fit-based recommendation
        if fit == FitCategory.HIGH:
            if missing_skills and len(missing_skills) <= 2:
                parts.append(f"Minor gaps in {', '.join(missing_skills[:2])} but overall excellent fit.")
            parts.append("Highly recommended for interview.")
        elif fit == FitCategory.MEDIUM:
            if missing_skills:
                parts.append(f"Would benefit from experience in {', '.join(missing_skills[:2])}.")
            parts.append("Consider for technical screening.")
        else:
            if missing_skills:
                parts.append(f"Lacks key skills: {', '.join(missing_skills[:3])}.")
            parts.append("May require significant training or better suited for a different role.")
        
        return ' '.join(parts)
    
    def rank_candidates(self, 
                        candidates: List[CandidateScore],
                        job_title: Optional[str] = None) -> List[ResumeResult]:
        """
        Rank candidates based on their scores and generate final results.
        
        Args:
            candidates: List of CandidateScore objects
            job_title: Detected job title for context
            
        Returns:
            Sorted list of ResumeResult objects
        """
        results = []
        
        for candidate in candidates:
            sim_data = candidate.similarity_data
            
            # Get the final score
            final_score = sim_data.get('final_score', 0)
            
            # Determine fit category
            fit = self.determine_fit_category(final_score)
            
            # Generate summary with actual resume content
            summary = self.generate_summary(
                final_score,
                sim_data.get('matched_skills', []),
                sim_data.get('missing_skills', []),
                sim_data.get('skill_breakdown', {}),
                resume_text=candidate.original_text,
                resume_name=candidate.resume_name
            )
            
            # Create result object
            result = ResumeResult(
                rank=1,  # ✅ Must be >= 1 for Pydantic validation (will be updated after sorting)
                id=candidate.resume_id,
                resume_name=candidate.resume_name,
                match_score=int(round(final_score)),
                fit=fit,
                matched_skills=sim_data.get('matched_skills', [])[:8],  # Limit to 8
                missing_skills=sim_data.get('missing_skills', [])[:5],  # Limit to 5
                summary=summary,
                skill_breakdown=sim_data.get('skill_breakdown', {})
            )
            
            results.append(result)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.match_score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(results, 1):
            result.rank = i
        
        logger.info(f"Ranked {len(results)} candidates")
        
        return results
    
    def get_ranking_stats(self, results: List[ResumeResult]) -> Dict:
        """
        Get statistics about the ranking results.
        
        Args:
            results: List of ranked results
            
        Returns:
            Dictionary with ranking statistics
        """
        if not results:
            return {}
        
        scores = [r.match_score for r in results]
        
        fit_counts = {
            'high': sum(1 for r in results if r.fit == FitCategory.HIGH),
            'medium': sum(1 for r in results if r.fit == FitCategory.MEDIUM),
            'low': sum(1 for r in results if r.fit == FitCategory.LOW)
        }
        
        # Collect all matched skills across candidates
        all_matched_skills = []
        for r in results:
            all_matched_skills.extend(r.matched_skills)
        
        skill_frequency = {}
        for skill in all_matched_skills:
            skill_frequency[skill] = skill_frequency.get(skill, 0) + 1
        
        # Sort skills by frequency
        common_skills = sorted(
            skill_frequency.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return {
            'total_candidates': len(results),
            'average_score': round(sum(scores) / len(scores), 1),
            'highest_score': max(scores),
            'lowest_score': min(scores),
            'score_range': max(scores) - min(scores),
            'fit_distribution': fit_counts,
            'common_matched_skills': [s[0] for s in common_skills],
            'recommendation': self._generate_recommendation(results, fit_counts)
        }
    
    def _generate_recommendation(self, 
                                  results: List[ResumeResult], 
                                  fit_counts: Dict) -> str:
        """
        Generate a recommendation based on the results.
        
        Args:
            results: Ranked results
            fit_counts: Distribution of fit categories
            
        Returns:
            Recommendation string
        """
        if fit_counts['high'] >= 2:
            return f"Strong candidate pool with {fit_counts['high']} excellent matches. Consider scheduling interviews with top {min(3, fit_counts['high'])} candidates."
        elif fit_counts['high'] >= 1:
            return f"One standout candidate identified. Consider also reviewing the {fit_counts['medium']} medium-fit candidates for potential."
        elif fit_counts['medium'] >= 2:
            return f"No perfect matches, but {fit_counts['medium']} candidates show good potential. Consider skills-based interview assessments."
        else:
            return "Limited matches found. Consider expanding search criteria or revisiting job requirements."
    
    def compare_candidates(self, 
                           result1: ResumeResult, 
                           result2: ResumeResult) -> Dict:
        """
        Compare two candidates side by side.
        
        Args:
            result1: First candidate result
            result2: Second candidate result
            
        Returns:
            Comparison dictionary
        """
        skills1 = set(result1.matched_skills)
        skills2 = set(result2.matched_skills)
        
        return {
            'score_difference': result1.match_score - result2.match_score,
            'common_skills': list(skills1 & skills2),
            'unique_to_first': list(skills1 - skills2),
            'unique_to_second': list(skills2 - skills1),
            'recommendation': (
                f"{result1.resume_name} is recommended" 
                if result1.match_score > result2.match_score 
                else f"{result2.resume_name} is recommended"
            )
        }


# Global ranking engine instance
ranking_engine = RankingEngine()