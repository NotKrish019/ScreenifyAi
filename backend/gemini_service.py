
import os
import json
import logging
import google.generativeai as genai
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables FIRST (critical for import-time initialization)
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables. AI features will be disabled.")
            self.model = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                logger.info("Gemini Service initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.model = None

    def evaluate_candidate(self, resume_text: str, jd_text: str) -> Dict:
        """
        Analyzes a resume against a job description using Gemini.
        Returns a dictionary with score (0-50) and analysis.
        """
        if not self.model:
            return {
                "ai_score": 0,
                "reason": "Gemini API Key missing. Please add GEMINI_API_KEY to your .env file.",
                "analysis": "AI analysis unavailable."
            }

        prompt = f"""
        You are an expert Technical Recruiter. Your task is to evaluate a candidate's resume against a provided Job Description.
        
        **Job Description:**
        {jd_text[:8000]}... (truncated)
        
        **Candidate Resume:**
        {resume_text[:8000]}... (truncated)
        
        **Analysis Required:**
        1. **Years of Experience**: Does the candidate meet the required experience?
        2. **Skills Match**: Does the candidate have the SPECIFIC or EQUIVALENT skills requested?
        3. **Role Fit**: Is the candidate's background relevant to this role?
        
        **Output Format:**
        Provide a JSON response with the following structure:
        {{
            "score": <number between 0 and 50>,
            "reason": "<short justification for the score>",
            "analysis": {{
                "years_of_experience": "<extracted value>",
                "key_skills_found": ["<skill1>", "<skill2>"],
                "missing_critical_skills": ["<skill1>"],
                "fit_summary": "<2-3 sentences on overall fit>"
            }}
        }}
        
        The 'score' must be a numeric integer between 0 and 50.
        **Scoring Guide**:
        - 40-50: Strong Match (Has most skills & relevant experience).
        - 25-39: Potential Match (Has transferrable skills or some gaps).
        - 0-24: Low Match.
        
        Be analytical but reasonable. Recognize transferrable skills.
        """

        try:
            response = self.model.generate_content(prompt)
            # Clean response to get JSON
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:-3]
            elif text.startswith("```"):
                text = text[3:-3]
            
            data = json.loads(text)
            
            # Defensive coding
            score = data.get("score", 0)
            if score > 50: score = 50 # Cap at 50 as per requirement (50% of total)
            
            return {
                "ai_score": score,
                "reason": data.get("reason", "Analysis complete"),
                "analysis": data.get("analysis", {})
            }

        except Exception as e:
            logger.error(f"Gemini Analysis Failed: {e}")
            return {
                "ai_score": 0,
                "reason": "AI Analysis Failed",
                "analysis": {"error": str(e)}
            }
            
gemini_service = GeminiService()
