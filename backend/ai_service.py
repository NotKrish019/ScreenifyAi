
# backend/ai_service.py
"""
AI SERVICE ADAPTER
Routes requests to the new Hybrid Scorer (Gemini + Keywords).
"""

import traceback
from typing import Dict, List
from hybrid_scorer import hybrid_scorer

class AIAdapter:
    def __init__(self):
        self.scorer = hybrid_scorer

    def analyze_resume(self, resume_text: str, jd_text: str, name: str = "Unknown") -> Dict:
        try:
            # Call the Hybrid Scorer
            result = self.scorer.score_resume(resume_text, jd_text, name)
            
            # Extract data for frontend
            final_score = result["final_score"]
            breakdown = result["breakdown"]
            
            # Construct a summary from the AI and Stat details
            ai_data = breakdown["ai_details"].get("analysis", {})
            stat_data = breakdown["stat_details"]
            
            # Safe access to AI fields
            years_exp = ai_data.get("years_of_experience", "N/A")
            fit_summary = ai_data.get("fit_summary", "Analysis pending.")
            ai_reason = breakdown["ai_details"].get("reason", "")
            
            # Keyword highlights
            matched_kws = stat_data.get("matched_keywords", [])
            missing_stats = list(set(stat_data.get("jd_keywords", [])) - set(matched_kws))
            
            # Combine summaries
            full_summary = (
                f"{fit_summary}\n\n"
                f"**Reasoning:** {ai_reason}\n"
                f"**Experience:** {years_exp}"
            )

            return {
                "candidate_name": name,
                "match_score": int(final_score),
                "fit_level": result["match_category"],
                # Combine explicit keywords found + AI detected skills if available
                "matched_skills": matched_kws + ai_data.get("key_skills_found", []),
                "missing_skills": missing_stats[:5] + ai_data.get("missing_critical_skills", []),
                "summary": full_summary,
                "recommendation": "Interview" if final_score > 70 else "Review",
                "experience_years": 0, # Frontend can parse string above if needed, or we keep 0 for safety
                "ai_breakdown": breakdown # Pass full details for potential debugging
            }

        except Exception as e:
            print(f"Error in analyze_resume: {e}")
            traceback.print_exc()
            return {
                "candidate_name": name,
                "match_score": 0,
                "fit_level": "Error",
                "matched_skills": [],
                "missing_skills": [],
                "summary": f"Analysis Error: {str(e)}",
                "recommendation": "Review manually"
            }

    def improve_job_description(self, text: str) -> Dict:
        return {"improved_text": text, "suggestions": ["Feature temporarily disabled in Hybrid Mode."]}

    def compare_candidates(self, candidates: List[Dict], jd_text: str) -> Dict:
        # Simple comparison logic
        if len(candidates) < 2: 
            return {"comparison": {"winner": "N/A", "summary": "Select 2 candidates"}}
            
        c1 = candidates[0]
        c2 = candidates[1]
        
        winner = c1 if c1['match_score'] > c2['match_score'] else c2
        
        return {
            "comparison": {
                "winner": winner['candidate_name'],
                "summary": f"{winner['candidate_name']} scores higher ({winner['match_score']}) based on the hybrid analysis."
            }
        }

    def generate_full_report(self, results: List[Dict], jd_text: str) -> Dict:
        return {"report": "Report generation requires full context."}

ai_service = AIAdapter()
