
import os
import requests
import json
import logging
import re
from typing import Dict, List, Optional
from openai_service import analyzer as nlp_analyzer

logger = logging.getLogger(__name__)

class AIService:
    """
    Central AI Service that delegates to the best available engine.
    1. Tries External LLM (OpenAI) if API key is present.
    2. Falls back to "Smart Local NLP" (the regex/tfidf engine) if no key.
    """
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        
        self.has_llm = bool(self.openai_key or self.gemini_key)
        
        if self.openai_key:
            logger.info("🟢 External LLM (OpenAI) Integrated")
            self.provider = "openai"
        elif self.gemini_key:
            logger.info("🟢 External LLM (Gemini) Integrated")
            self.provider = "gemini"
        else:
            logger.info("🟡 No API Key found. Using Smart Local NLP Engine.")
            self.provider = "local"

    def improve_job_description(self, current_jd: str) -> Dict:
        """Improve JD using LLM or Local Logic."""
        if self.has_llm:
            return self._improve_jd_llm(current_jd)
        return self._improve_jd_local(current_jd)

    def compare_candidates(self, candidates: List[Dict], jd_text: str) -> Dict:
        """Compare candidates using LLM or Local Logic."""
        if self.has_llm:
            return self._compare_llm(candidates, jd_text)
        return self._compare_local(candidates, jd_text)

    def analyze_resume(self, text, jd, name):
        """Standard analysis with optional Gemini enhancement."""
        # Get base analysis from local NLP (fast and reliable)
        result = nlp_analyzer.analyze_resume(text, jd, name)
        
        # Enhance with Gemini insights if available
        if self.has_llm and self.provider == "gemini":
            try:
                gemini_insights = self._get_gemini_insights(text, jd, name, result)
                if gemini_insights:
                    result['explanation'] = gemini_insights
            except Exception as e:
                logger.warning(f"Gemini insights failed, using local: {e}")
        
        return result

    def _get_gemini_insights(self, resume_text: str, jd_text: str, name: str, analysis: Dict) -> Dict:
        """Get AI-powered insights from Gemini for a candidate."""
        try:
            prompt = f"""You are a senior HR consultant with 20 years of experience. Provide a DETAILED and INSIGHTFUL analysis of this candidate.

CANDIDATE: {name}
MATCH SCORE: {analysis.get('match_score', 0)}%
MATCHED SKILLS: {', '.join(analysis.get('matched_skills', [])[:15])}
MISSING SKILLS: {', '.join(analysis.get('missing_skills', [])[:8])}

JOB DESCRIPTION:
{jd_text[:1200]}

RESUME CONTENT:
{resume_text[:1500]}

Write a comprehensive analysis in this EXACT JSON format. Be DETAILED and SPECIFIC - don't give generic responses:

{{
  "summary": "Write a detailed 4-5 sentence paragraph about this candidate. Start with their overall profile and experience level. Then discuss their key technical strengths and how they align with the role. Mention any notable projects or achievements. Finally, give your professional assessment of their potential fit. Make it read like a real HR consultant's evaluation - insightful, specific, and actionable.",
  "strengths": [
    "Detailed strength 1 - explain WHY this is valuable for the role (2 sentences)",
    "Detailed strength 2 - connect their experience to job requirements (2 sentences)", 
    "Detailed strength 3 - highlight a unique advantage they bring (2 sentences)"
  ],
  "concerns": [
    "Specific concern or gap - explain the impact and how it could be addressed",
    "Another area needing attention with context"
  ],
  "tips": [
    "Specific interview question to ask them and why",
    "What to probe deeper on during screening",
    "Red flag to watch for OR green flag to confirm"
  ],
  "verdict": "STRONG HIRE / WORTH INTERVIEWING / NEEDS MORE REVIEW / PASS - with a brief justification"
}}

IMPORTANT: Don't just restate the skills list. Provide genuine insight like a real HR expert would. Be specific to THIS candidate and THIS job."""

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_key}"
            headers = {"Content-Type": "application/json"}
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            
            r = requests.post(url, headers=headers, json=payload, timeout=15)
            if r.status_code == 200:
                content = r.json()['candidates'][0]['content']['parts'][0]['text']
                # Parse JSON from response
                import json
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    parsed = json.loads(json_match.group())
                    logger.info(f"✨ Gemini insights generated for {name}")
                    return parsed
            else:
                logger.warning(f"Gemini API returned {r.status_code}")
            return None
        except Exception as e:
            logger.warning(f"Gemini insight generation error: {e}")
            return None

    def generate_full_report(self, candidates: List[Dict], jd_text: str) -> Dict:
        """Generate a complete AI-written report for all candidates."""
        if self.has_llm:
            return self._generate_report_llm(candidates, jd_text)
        return self._generate_report_local(candidates, jd_text)

    def _generate_report_llm(self, candidates: List[Dict], jd_text: str) -> Dict:
        """Use Gemini to write a full natural language report."""
        try:
            # Build candidate summaries for the prompt
            candidate_info = ""
            for i, c in enumerate(candidates):
                candidate_info += f"""
Candidate {i+1}: {c.get('resume_name', c.get('candidate_name', 'Unknown'))}
- Match Score: {c.get('match_score', 0)}%
- Fit Level: {c.get('fit', 'Unknown')}
- Matched Skills: {', '.join(c.get('matched_skills', [])[:10])}
- Missing Skills: {', '.join(c.get('missing_skills', [])[:10])}
- Summary: {c.get('summary', 'No summary')}
"""

            prompt = f"""
You are an expert HR consultant writing a professional hiring report. 

JOB DESCRIPTION:
{jd_text[:1500]}

CANDIDATES ANALYZED:
{candidate_info}

Write a comprehensive, professional hiring report in the following format. Be detailed, insightful, and write in natural paragraphs (not bullet points). Each section should be 2-3 sentences minimum.

For EACH candidate, write:
1. **Executive Summary**: A brief overview of who they are and their overall fit
2. **Strengths Analysis**: What makes them a good candidate? Be specific about their skills
3. **Areas of Concern**: What are the gaps or weaknesses? Be honest but professional
4. **Hire Recommendation**: Should we hire them? Give a clear YES/NO/MAYBE with detailed reasoning
5. **Interview Focus Areas**: What should we ask them about in an interview?

End with a **Final Ranking** section that ranks all candidates and explains who to prioritize.

Write naturally like a human HR consultant, not like a copy-paste of data. Be insightful and add value beyond just restating the numbers.
"""

            if self.provider == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_key}"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}]
                }
                r = requests.post(url, headers=headers, json=payload)
                if r.status_code != 200:
                    logger.error(f"Gemini Report Error: {r.text}")
                    r.raise_for_status()
                content = r.json()['candidates'][0]['content']['parts'][0]['text']
                return {"success": True, "report": content}
            else:
                # OpenAI fallback
                headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are an expert HR consultant writing professional hiring reports."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 2000
                }
                r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                r.raise_for_status()
                content = r.json()['choices'][0]['message']['content']
                return {"success": True, "report": content}

        except Exception as e:
            logger.error(f"Report Generation Failed: {e}")
            return self._generate_report_local(candidates, jd_text)

    def _generate_report_local(self, candidates: List[Dict], jd_text: str) -> Dict:
        """Generate a basic report without LLM."""
        report = "# AI Resume Screening Report\n\n"
        report += f"## Job Description Summary\n{jd_text[:300]}...\n\n"
        report += "## Candidate Analysis\n\n"
        
        for i, c in enumerate(candidates):
            name = c.get('resume_name', c.get('candidate_name', 'Unknown'))
            score = c.get('match_score', 0)
            
            report += f"### {i+1}. {name}\n\n"
            report += f"**Match Score:** {score}%\n\n"
            report += f"**Summary:** {c.get('summary', 'No summary available.')}\n\n"
            report += f"**Key Skills:** {', '.join(c.get('matched_skills', [])[:5]) or 'None identified'}\n\n"
            report += f"**Skill Gaps:** {', '.join(c.get('missing_skills', [])[:5]) or 'None identified'}\n\n"
            
            if score >= 75:
                report += "**Recommendation:** ✅ STRONG HIRE - This candidate shows excellent alignment with the role requirements.\n\n"
            elif score >= 50:
                report += "**Recommendation:** ⚠️ CONDITIONAL HIRE - Consider for the role but may need additional training or assessment.\n\n"
            else:
                report += "**Recommendation:** ❌ NOT RECOMMENDED - Significant gaps exist between candidate skills and job requirements.\n\n"
            
            report += "---\n\n"
        
        return {"success": True, "report": report}
        
    # --- Local Implementations (The "Smart NLP" Fallback) ---
    
    def _improve_jd_local(self, text: str) -> Dict:
        """Simulate JD improvement locally."""
        # Simple expansion logic
        improved = text
        
        # 1. Structure check
        if "Responsibilities" not in text:
            improved += "\n\nResponsibilities:\n- Design and develop scalable applications\n- Collaborate with cross-functional teams"
        if "Requirements" not in text:
            improved += "\n\nRequirements:\n- Proven experience in relevant technologies\n- Strong problem-solving skills"
            
        # 2. Extract Tech skills to ensure they are highlighted
        skills = nlp_analyzer.extract_skills(text)
        if skills:
            improved += f"\n\nTechnical Stack:\n- {', '.join(skills)}"
            
        return {
            "improved_version": improved,
            "strengths": ["Clear core requirements", "Includes technical keywords"],
            "weaknesses": ["Could be more structured" if len(text) < 200 else "Minor formatting improvements needed"],
            "suggestions": ["Add more specific deliverables", "Clarify years of experience required"]
        }

    def _compare_local(self, candidates: List[Dict], jd_text: str) -> Dict:
        """Compare candidates locally using existing analysis data."""
        if len(candidates) < 2:
            return {}
            
        c1 = candidates[0]
        c2 = candidates[1]
        
        # Use existing analysis if available, otherwise analyze
        if 'matched_skills' in c1:
            s1 = set(c1['matched_skills'])
            score1 = c1.get('match_score', 0)
            name1 = c1.get('candidate_name', c1.get('resume_name', 'Candidate 1'))
        else:
            a1 = nlp_analyzer.analyze_resume(c1.get('original_text', ''), jd_text, c1.get('name', 'C1'))
            s1 = set(a1['matched_skills'])
            score1 = a1['match_score']
            name1 = a1['candidate_name']

        if 'matched_skills' in c2:
            s2 = set(c2['matched_skills'])
            score2 = c2.get('match_score', 0)
            name2 = c2.get('candidate_name', c2.get('resume_name', 'Candidate 2'))
        else:
            a2 = nlp_analyzer.analyze_resume(c2.get('original_text', ''), jd_text, c2.get('name', 'C2'))
            s2 = set(a2['matched_skills'])
            score2 = a2['match_score']
            name2 = a2['candidate_name']
        
        common = list(s1 & s2)
        unique_1 = list(s1 - s2)
        unique_2 = list(s2 - s1)
        
        winner = name1 if score1 >= score2 else name2
        
        return {
            "comparison": {
                "common_skills": common,
                "unique_to_first": unique_1,
                "unique_to_second": unique_2,
                "c1_name": name1,
                "c2_name": name2,
                "recommendation": f"Based on the analysis, **{winner}** is the stronger candidate ({score1}% vs {score2}%). They share {len(common)} skills, but {winner} demonstrates better alignment with the core requirements."
            }
        }

    # --- LLM Implementations ---
    
    def _improve_jd_llm(self, text: str) -> Dict:
        """Call External LLM to improve JD."""
        try:
            if self.provider == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_key}"
                headers = {"Content-Type": "application/json"}
                prompt = "You are an expert HR AI. Improve this job description. Return ONLY valid JSON with keys: improved_version (string), strengths (list of strings), weaknesses (list of strings), suggestions (list of strings). Text: " + text
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}]
                }
                r = requests.post(url, headers=headers, json=payload)
                if r.status_code != 200:
                     logger.error(f"Gemini Improve Error: {r.text}")
                     r.raise_for_status()
                content = r.json()['candidates'][0]['content']['parts'][0]['text']
            else:
                # OpenAI Logic
                headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "You are an expert HR application. Improve the JD and return JSON with keys: improved_version, strengths (list), weaknesses (list), suggestions (list)."},
                        {"role": "user", "content": text}
                    ]
                }
                r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                r.raise_for_status()
                content = r.json()['choices'][0]['message']['content']

            # Parse JSON from content (handle potential markdown wrapping)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                 content = content.split("```")[1].split("```")[0]
            
            return json.loads(content)
        except Exception as e:
            logger.error(f"LLM Call Failed ({self.provider}): {e}")
            return self._improve_jd_local(text) # Fallback

    def _compare_llm(self, candidates, jd_text):
        """Call External LLM to compare candidates."""
        try:
            c1, c2 = candidates[0], candidates[1]
            
            # Prepare names
            n1 = c1.get('candidate_name', c1.get('resume_name', 'Candidate 1'))
            n2 = c2.get('candidate_name', c2.get('resume_name', 'Candidate 2'))
            
            prompt = f"""
            Compare these two candidates for the Job Description.
            
            JD: {jd_text[:800]}...
            
            Candidate 1 ({n1}): Score {c1.get('match_score', '?')}%. Skills: {', '.join(c1.get('matched_skills', []))}
            Summary: {c1.get('summary', 'N/A')}
            
            Candidate 2 ({n2}): Score {c2.get('match_score', '?')}%. Skills: {', '.join(c2.get('matched_skills', []))}
            Summary: {c2.get('summary', 'N/A')}
            
            Return JSON with keys: comparison (object with keys: common_skills (list of strings), unique_to_first (list of strings - skills {n1} has that {n2} doesn't), unique_to_second (list of strings), recommendation (string - clear winner declaration and why)).
            """
            
            if self.provider == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.gemini_key}"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}]
                }
                r = requests.post(url, headers=headers, json=payload)
                if r.status_code != 200:
                    logger.error(f"Gemini API Error: {r.status_code} - {r.text}")
                    r.raise_for_status()
                content = r.json()['candidates'][0]['content']['parts'][0]['text']
            else:
                 # OpenAI Compare Logic placeholder
                 return self._compare_local(candidates, jd_text)

            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                 content = content.split("```")[1].split("```")[0]
            
            return json.loads(content)
        except Exception as e:
             logger.error(f"LLM Compare Failed: {e}")
             return self._compare_local(candidates, jd_text)

# Singleton
ai_service = AIService()
