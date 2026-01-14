
import os
import json
import logging
from google import genai
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables FIRST (critical for import-time initialization)
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


import random
import itertools

class GeminiService:
    def __init__(self):
        # Load all available keys
        self.api_keys = []
        
        # Method 1: Check for comma-separated list in GEMINI_API_KEY
        main_key = os.getenv("GEMINI_API_KEY", "")
        if "," in main_key:
            self.api_keys.extend([k.strip() for k in main_key.split(",") if k.strip()])
        elif main_key:
            self.api_keys.append(main_key)
            
        # Method 2: Check for numbered keys (e.g. GEMINI_API_KEY_1 or GEMINI_API_KEY1)
        i = 1
        while True:
            # Try with underscore first
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            # If not found, try without underscore
            if not key:
                key = os.getenv(f"GEMINI_API_KEY{i}")
                
            if key:
                self.api_keys.append(key)
                i += 1
            else:
                # Stop if we hit a gap? Or assume strict sequence. 
                # Let's check a reasonable range to be safe or just stop. 
                # Usually users might skip, but let's assume sequence for now or check up to 5.
                if i > 20: break # Safety break
                
                # Check if next one exists just in case (e.g. key_1 and key_3)
                # Try both formats for next I
                next_key = os.getenv(f"GEMINI_API_KEY_{i+1}") or os.getenv(f"GEMINI_API_KEY{i+1}")
                if not next_key: break
                i += 1
                continue

        # Remove duplicates
        self.api_keys = list(set(self.api_keys))
        
        if not self.api_keys:
            logger.warning("No GEMINI_API_KEYs found in environment variables. AI features will be disabled.")
            self.key_cycle = None
        else:
            logger.info(f"Gemini Service initialized with {len(self.api_keys)} API keys.")
            # Use a cycle iterator for round-robin distribution
            random.shuffle(self.api_keys) # Shuffle once to randomize start order
            self.key_cycle = itertools.cycle(self.api_keys)

    def _get_client(self):
        """
        Returns a new client instance using the next API key in the rotation.
        """
        if not self.key_cycle:
            return None
            
        api_key = next(self.key_cycle)
        try:
            # logger.debug(f"Using API Key: ...{api_key[-5:]}")
            return genai.Client(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to create Gemini client: {e}")
            return None

    def evaluate_candidate(self, resume_text: str, jd_text: str) -> Dict:
        """
        Analyzes a resume against a job description using Gemini.
        Returns a dictionary with score (0-50) and analysis.
        """
        client = self._get_client()
        if not client:
            return {
                "ai_score": 0,
                "reason": "Gemini API Keys missing. Please check your .env file.",
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
            # Using the new Client API
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            
            # Clean response to get JSON
            text = response.text.strip()
            
            # Remove markdown code blocks if present
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
                
            text = text.strip()
            
            # Find JSON start and end if mixed with text
            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1:
                text = text[start_idx:end_idx+1]
            
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON. Raw output: {text}")
                # Fallback structure
                return {
                    "ai_score": 0,
                    "reason": "AI response was not valid JSON",
                    "analysis": {"raw_output": text}
                }
            
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
            

    def improve_jd(self, jd_text: str) -> Dict:
        """
        Improves a Job Description using a rotated API Key.
        """
        client = self._get_client()
        if not client:
             return {
                 "improved_version": jd_text,
                 "analysis": {"error": "Gemini API Keys missing or invalid."}
             }

        try:
            prompt = f"""
            Act as an expert HR Specialist. Review and improve this Job Description.
            
            **Original JD:**
            {jd_text}
            
            **Tasks:**
            1. Identify strengths and weaknesses.
            2. Suggest 3 concrete improvements.
            3. Rewrite the JD to be more professional, inclusive, and clear.
            
            **Output JSON:**
            {{
                "strengths": ["..."],
                "weaknesses": ["..."],
                "suggestions": ["..."],
                "improved_version": "..."
            }}
            """
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            
            # Parse JSON logic similar to evaluate
            text = response.text.strip()
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            text = text.strip()
            
            # Simple brace extraction
            s = text.find("{")
            e = text.rfind("}")
            if s != -1 and e != -1: text = text[s:e+1]
            
            return json.loads(text)

        except Exception as e:
            logger.error(f"JD Improvement Failed: {e}")
            return {
                "improved_version": jd_text,
                "analysis": {"error": str(e)}
            }
            

    def compare_candidates_ai(self, c1_text: str, c2_text: str, jd_text: str) -> Dict:
        """
        Compares two candidates against a JD using Gemini AI.
        Returns a dictionary with winner and detailed comparison.
        """
        client = self._get_client()
        if not client:
            return {"error": "AI Service unavailable"}

        try:
            prompt = f"""
            Act as an expert Technical Recruiter. Compare two candidates for a specific Job Description.
            
            **Job Description:**
            {jd_text[:5000]}
            
            **Candidate A:**
            {c1_text[:5000]}
            
            **Candidate B:**
            {c2_text[:5000]}
            
            **Task:**
            1. Analyze both candidates' strengths and weaknesses relative to the JD.
            2. Determine the better fit.
            3. Identify **SHARED SKILLS** (what both have).
            4. Identify **UNIQUE STRENGTHS** for EACH candidate (what one has but the other doesn't).
            5. Explain WHY one is better than the other.
            
            **Output JSON:**
            {{
                "winner": "Candidate A" or "Candidate B",
                "summary": "Short summary of the decision (2-3 sentences).",
                "comparison_points": [
                    "Point 1: Skill comparison...",
                    "Point 2: Experience comparison...",
                    "Point 3: Unique value add..."
                ],
                "shared_skills": ["Skill 1", "Skill 2"],
                "candidate_a_unique": ["Unique 1", "Unique 2"],
                "candidate_b_unique": ["Unique 3", "Unique 4"],
                "recommendation": "Final hiring recommendation."
            }}
            """
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            
            # Parse JSON
            text = response.text.strip()
            if text.startswith("```json"): text = text[7:]
            if text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            text = text.strip()
            
            # Simple brace extraction
            s = text.find("{")
            e = text.rfind("}")
            if s != -1 and e != -1: text = text[s:e+1]
            
            return json.loads(text)

        except Exception as e:
            logger.error(f"Candidate Comparison Failed: {e}")
            return {"error": str(e)}

gemini_service = GeminiService()
