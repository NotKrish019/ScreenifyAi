"""
NLP-Based Resume Analysis Engine
Uses TF-IDF, cosine similarity, and smart extraction - no external AI needed
"""

import re
import logging
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResumeAnalyzer:
    """Smart NLP-based resume analyzer."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        
        # Common tech skills for matching
        self.tech_skills = {
            'python', 'javascript', 'typescript', 'java', 'c++', 'c#', 'ruby', 'go', 'rust', 'php', 'swift', 'kotlin',
            'react', 'angular', 'vue', 'svelte', 'next.js', 'nuxt', 'html', 'css', 'sass', 'tailwind',
            'node.js', 'nodejs', 'express', 'django', 'flask', 'fastapi', 'spring', 'rails',
            'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'sql', 'nosql', 'dynamodb',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'ci/cd',
            'git', 'linux', 'agile', 'scrum', 'rest', 'graphql', 'microservices', 'api'
        }
    
    def extract_name(self, text: str) -> str:
        """Extract candidate name from resume."""
        lines = text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) < 40:
                # Skip lines with common non-name content
                skip_words = ['email', 'phone', '@', 'http', 'github', 'linkedin', 'resume', 'cv', 'objective', 'summary']
                if not any(w in line.lower() for w in skip_words):
                    if re.match(r'^[A-Za-z\s\.\-]+$', line):
                        return line.title()
        return "Unknown Candidate"
    
    def extract_title(self, text: str) -> str:
        """Extract job title from resume."""
        title_patterns = [
            r'(?i)(software\s+(?:developer|engineer))',
            r'(?i)(full[\s\-]?stack\s+(?:developer|engineer))',
            r'(?i)(front[\s\-]?end\s+(?:developer|engineer))',
            r'(?i)(back[\s\-]?end\s+(?:developer|engineer))',
            r'(?i)(data\s+(?:scientist|analyst|engineer))',
            r'(?i)(devops\s+engineer)',
            r'(?i)(product\s+manager)',
            r'(?i)(senior\s+\w+\s*(?:developer|engineer)?)',
            r'(?i)(junior\s+\w+\s*(?:developer|engineer)?)',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).title()
        return "Professional"
    
    def extract_experience_years(self, text: str) -> int:
        """Extract years of experience."""
        patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience[:\s]+(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s+in\s+',
        ]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        
        # Count years from job dates
        year_pattern = r'\b(20\d{2})\s*[-–]\s*(20\d{2}|present|current)\b'
        matches = re.findall(year_pattern, text.lower())
        if matches:
            total_years = 0
            for start, end in matches:
                start_year = int(start)
                end_year = 2025 if end in ['present', 'current'] else int(end)
                total_years += max(0, end_year - start_year)
            return min(total_years, 20)  # Cap at 20
        return 0
    
    def extract_education(self, text: str) -> str:
        """Extract education level."""
        text_lower = text.lower()
        if 'phd' in text_lower or 'doctor' in text_lower:
            return "PhD"
        elif any(x in text_lower for x in ["master's", 'masters', 'm.s.', 'mba', 'm.tech']):
            return "Master's Degree"
        elif any(x in text_lower for x in ["bachelor's", 'bachelors', 'b.s.', 'b.tech', 'b.e.']):
            return "Bachelor's Degree"
        return ""
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text."""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.tech_skills:
            # Use word boundaries for accurate matching
            pattern = r'\b' + re.escape(skill.replace('.', r'\.')) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill)
        
        return found_skills
    
    def calculate_similarity(self, resume_text: str, jd_text: str) -> float:
        """Calculate TF-IDF cosine similarity between resume and JD."""
        try:
            tfidf_matrix = self.vectorizer.fit_transform([jd_text, resume_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def analyze_resume(self, resume_text: str, job_description: str, resume_name: str = "Candidate") -> Dict:
        """
        Analyze a resume using NLP techniques.
        """
        # Extract candidate info
        name = self.extract_name(resume_text)
        if name == "Unknown Candidate":
            name = resume_name.replace('.pdf', '').replace('.docx', '').replace('resume_', '').replace('_', ' ').title()
        
        title = self.extract_title(resume_text)
        exp_years = self.extract_experience_years(resume_text)
        education = self.extract_education(resume_text)
        
        # Extract skills
        resume_skills = set(self.extract_skills(resume_text))
        jd_skills = set(self.extract_skills(job_description))
        
        matched_skills = list(resume_skills & jd_skills)
        missing_skills = list(jd_skills - resume_skills)
        
        # Calculate match score
        tfidf_score = self.calculate_similarity(resume_text, job_description)
        skill_score = len(matched_skills) / max(len(jd_skills), 1) if jd_skills else 0.5
        
        # Combined score (60% skills, 40% text similarity)
        match_score = int((skill_score * 0.6 + tfidf_score * 0.4) * 100)
        match_score = max(20, min(95, match_score))  # Clamp between 20-95
        
        # Determine fit level
        if match_score >= 70:
            fit_level = "High"
        elif match_score >= 50:
            fit_level = "Medium"
        else:
            fit_level = "Low"
        
        # Generate personalized summary
        summary = self._generate_summary(name, title, exp_years, education, matched_skills, missing_skills, fit_level)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(fit_level, matched_skills, missing_skills, exp_years)
        
        logger.info(f"✅ Analyzed: {name} - {match_score}% match ({fit_level})")
        
        return {
            "candidate_name": name,
            "current_title": title,
            "experience_years": exp_years,
            "match_score": match_score,
            "fit_level": fit_level,
            "matched_skills": matched_skills[:8],
            "missing_skills": missing_skills[:5],
            "summary": summary,
            "recommendation": recommendation
        }
    
    def _generate_summary(self, name: str, title: str, exp_years: int, education: str, 
                          matched_skills: List[str], missing_skills: List[str], fit_level: str) -> str:
        """Generate a personalized summary."""
        parts = []
        
        # Opening
        if exp_years > 0:
            parts.append(f"{name} is a {title.lower()} with {exp_years}+ years of experience.")
        else:
            parts.append(f"{name} is a {title.lower()}.")
        
        # Skills highlight
        if matched_skills:
            top_skills = matched_skills[:4]
            if len(top_skills) >= 3:
                parts.append(f"Demonstrates proficiency in {', '.join(top_skills[:-1])} and {top_skills[-1]}.")
            else:
                parts.append(f"Has experience with {', '.join(top_skills)}.")
        
        # Education
        if education:
            parts.append(f"Holds a {education}.")
        
        # Gaps
        if fit_level != "High" and missing_skills:
            parts.append(f"Could strengthen skills in {', '.join(missing_skills[:2])}.")
        
        return ' '.join(parts)
    
    def _generate_recommendation(self, fit_level: str, matched_skills: List[str], 
                                  missing_skills: List[str], exp_years: int) -> str:
        """Generate hiring recommendation."""
        if fit_level == "High":
            if exp_years >= 5:
                return "Strong candidate for senior role. Recommend immediate interview. Demonstrated expertise aligns well with job requirements."
            else:
                return "Excellent match for the role. Proceed to technical interview. Shows strong potential and relevant skill set."
        elif fit_level == "Medium":
            if len(matched_skills) >= 4:
                return f"Decent skill match. Consider for screening call. May need training in {', '.join(missing_skills[:2])} to fully meet requirements."
            else:
                return "Partial alignment with requirements. Could be a good fit for junior role or with additional training."
        else:
            if exp_years > 0:
                return "Limited match with current requirements. May be better suited for a different role or team."
            else:
                return "Minimal skill overlap. Consider for entry-level positions or internship if applicable."


# Singleton instance
analyzer = ResumeAnalyzer()


def analyze_resume(resume_text: str, job_description: str, resume_name: str = "Candidate") -> Dict:
    """Analyze a single resume."""
    return analyzer.analyze_resume(resume_text, job_description, resume_name)


def batch_analyze_resumes(resumes: List[Dict], job_description: str) -> List[Dict]:
    """Analyze multiple resumes."""
    results = []
    
    for i, resume in enumerate(resumes):
        logger.info(f"📊 Analyzing resume {i+1}/{len(resumes)}: {resume.get('name', 'Unknown')}")
        
        result = analyze_resume(
            resume_text=resume.get('text', ''),
            job_description=job_description,
            resume_name=resume.get('name', 'Unknown')
        )
        result['file_name'] = resume.get('name', 'Unknown')
        results.append(result)
    
    # Sort by match score
    results.sort(key=lambda x: x.get('match_score', 0), reverse=True)
    
    # Add ranks
    for i, result in enumerate(results):
        result['rank'] = i + 1
    
    return results
