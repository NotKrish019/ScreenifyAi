"""
AI Service Module
Handles AI-powered resume analysis using Ollama local LLM.
Provides intelligent, context-aware resume screening.
"""

import logging
import json
import requests
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:1b"  # Fast, lightweight model


class AIService:
    """
    AI Service for intelligent resume analysis using Ollama.
    """
    
    def __init__(self):
        """Initialize the AI service."""
        self.model = MODEL_NAME
        self.base_url = OLLAMA_BASE_URL
        self._check_ollama()
        logger.info(f"AI Service initialized with model: {self.model}")
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            logger.warning("Ollama not running. Start with: ollama serve")
            return False
    
    def _call_ollama(self, prompt: str, max_tokens: int = 500) -> str:
        """Call Ollama API with a prompt."""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.3,
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return ""
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return ""
    
    def analyze_resume(self, resume_text: str, job_description: str) -> Dict:
        """
        Analyze a resume against a job description using AI.
        
        Args:
            resume_text: The extracted text from the resume
            job_description: The job description text
            
        Returns:
            Dictionary with analysis results
        """
        # Truncate if too long
        resume_text = resume_text[:4000] if len(resume_text) > 4000 else resume_text
        job_description = job_description[:2000] if len(job_description) > 2000 else job_description
        
        prompt = f"""You are an expert HR recruiter analyzing a resume. Be specific and accurate.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Analyze this resume and respond ONLY in this exact JSON format (no other text):
{{
    "match_score": <number 0-100>,
    "fit": "<High/Medium/Low>",
    "skills_found": ["skill1", "skill2", "skill3"],
    "skills_missing": ["skill1", "skill2"],
    "experience_years": <number or null>,
    "summary": "<2-3 sentence detailed summary about this specific candidate, their background, strengths, and fit for this role>",
    "recommendation": "<specific hiring recommendation for this candidate>"
}}

Be accurate - only list skills that are EXPLICITLY mentioned in the resume. Do not guess or assume skills."""

        response = self._call_ollama(prompt, max_tokens=600)
        
        # Parse JSON response
        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                result = json.loads(json_str)
                return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response: {e}")
        
        # Fallback if AI fails
        return {
            "match_score": 50,
            "fit": "Medium",
            "skills_found": [],
            "skills_missing": [],
            "experience_years": None,
            "summary": "Analysis in progress. Please try again.",
            "recommendation": "Pending review."
        }
    
    def extract_skills_ai(self, text: str) -> List[str]:
        """
        Extract skills from text using AI.
        More accurate than keyword matching.
        """
        text = text[:3000] if len(text) > 3000 else text
        
        prompt = f"""Extract ONLY the technical and professional skills explicitly mentioned in this text.
Return as a JSON array of strings. Only include skills that are clearly stated.

TEXT:
{text}

Respond ONLY with a JSON array like: ["Python", "React", "AWS"]"""

        response = self._call_ollama(prompt, max_tokens=200)
        
        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except:
            pass
        
        return []
    
    def generate_candidate_summary(self, 
                                   resume_text: str, 
                                   job_description: str,
                                   matched_skills: List[str],
                                   match_score: int) -> str:
        """
        Generate a detailed, natural language summary for a candidate.
        """
        resume_text = resume_text[:3000] if len(resume_text) > 3000 else resume_text
        
        prompt = f"""You are a senior HR recruiter writing a brief but detailed assessment.

JOB REQUIREMENTS:
{job_description[:1000]}

CANDIDATE RESUME:
{resume_text}

MATCHED SKILLS: {', '.join(matched_skills[:10]) if matched_skills else 'None identified'}
MATCH SCORE: {match_score}%

Write a 2-3 sentence professional summary about this specific candidate. Include:
1. Their apparent experience level and background
2. Key strengths relevant to this role
3. Any concerns or gaps
4. Your hiring recommendation

Be specific to THIS candidate. Do not use generic phrases."""

        response = self._call_ollama(prompt, max_tokens=250)
        
        if response and len(response) > 20:
            # Clean up the response
            response = response.replace('\n', ' ').strip()
            # Remove any thinking tags if present
            if '<think>' in response:
                end_think = response.find('</think>')
                if end_think != -1:
                    response = response[end_think + 8:].strip()
            return response
        
        return f"Candidate shows {match_score}% match. Further review recommended."

    def improve_job_description(self, jd_text: str) -> Dict:
        """
        Analyze and suggest improvements for a job description.
        """
        jd_text = jd_text[:2000] if len(jd_text) > 2000 else jd_text
        
        prompt = f"""You are an expert HR consultant. Analyze this Job Description.

JOB DESCRIPTION:
{jd_text}

Provide a constructive critique and an improved version.
Respond ONLY in this JSON format:
{{
    "strengths": ["string", "string"],
    "weaknesses": ["string", "string"],
    "suggestions": ["string", "string"],
    "improved_version": "Full improved text..."
}}"""

        response = self._call_ollama(prompt, max_tokens=800)
        
        try:
            # Find JSON in response (handle potential preamble text)
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except Exception as e:
            logger.error(f"JD Improvement failed: {e}")
            
        return {
            "strengths": [],
            "weaknesses": ["Analysis failed"],
            "suggestions": ["Please try again"],
            "improved_version": jd_text
        }

    def compare_candidates(self, candidates: List[Dict], job_description: str) -> Dict:
        """
        Compare multiple candidates side-by-side.
        """
        job_description = job_description[:1000] if len(job_description) > 1000 else job_description
        
        # Format candidate data for prompt
        candidates_text = ""
        for c in candidates:
            # Summarize resume text to save tokens
            text_snippet = c.get('original_text', '')[:1000]
            candidates_text += f"\n--- CANDIDATE: {c.get('name')} ---\n{text_snippet}\n"

        prompt = f"""Compare these candidates for the following role.

JOB:
{job_description}

CANDIDATES:
{candidates_text}

Compare them based on skills, experience, and fit.
Respond ONLY in this JSON format:
{{
    "summary": "Brief comparative summary of the group...",
    "best_candidate": "Name of best fit",
    "comparison_table": [
        {{
            "name": "Candidate Name",
            "pros": ["pro1", "pro2"],
            "cons": ["con1", "con2"],
            "verdict": "Short verdict"
        }}
    ]
}}"""

        response = self._call_ollama(prompt, max_tokens=1000)
        
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            
        return {
            "summary": "Comparison could not be generated.",
            "best_candidate": "Unknown",
            "comparison_table": []
        }


# Singleton instance
ai_service = AIService()
