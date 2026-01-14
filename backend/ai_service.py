
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
        # Import here to avoid circular dependency if any
        from gemini_service import gemini_service
        return gemini_service.improve_jd(text)

    def compare_candidates(self, candidates: List[Dict], jd_text: str) -> Dict:
        if len(candidates) < 2: 
            return {"comparison": {"winner": "N/A", "summary": "Select 2 candidates"}}
            
        c1 = candidates[0]
        c2 = candidates[1]
        
        # We need the original text ideally, but if not available, we use what we have
        # The 'original_text' might not be in the 'candidates' dict passed here if it was stripped
        # However, checking main.py, 'app_state.resumes' has the text. 
        # But here we receive processed 'candidates' results.
        # We might need to fetch the text or rely on summaries. 
        # For now, let's assume we can get text or use the summary/skills we have.
        
        # Actually, let's try to grab the text if passed, or fall back to score comparison
        # NOTE: In a real app, passing full text here is heavy. 
        # We will retrieve text from app_state in main.py before calling this, OR
        # we update main.py to pass text.
        
        # Let's assume passed candidates list has 'original_text' or we can't do deep AI comparison.
        # If not, we fall back to simple logic.
        
        c1_text = c1.get('original_text', '') or c1.get('summary', '')
        c2_text = c2.get('original_text', '') or c2.get('summary', '')
        
        from gemini_service import gemini_service
        ai_result = gemini_service.compare_candidates_ai(c1_text, c2_text, jd_text)
        
        if "error" in ai_result:
             # Fallback to score
             winner = c1 if c1['match_score'] > c2['match_score'] else c2
             return {
                "comparison": {
                    "winner": winner['candidate_name'],
                    "summary": f"{winner['candidate_name']} scores higher ({winner['match_score']}) based on scoring models. (AI Comparison Failed: {ai_result['error']})",
                    "details": [],
                    "shared_skills": [],
                    "c1_unique": [],
                    "c2_unique": [],
                    "recommendation": "Review manually due to AI error."
                }
             }

        # Map "Candidate A" / "Candidate B" back to names
        winner_name = c1['candidate_name'] if "Candidate A" in ai_result.get("winner", "") else c2['candidate_name']
        
        return {
            "comparison": {
                "winner": winner_name,
                "summary": ai_result.get("summary", ""),
                "details": ai_result.get("comparison_points", []),
                "recommendation": ai_result.get("recommendation", ""),
                "shared_skills": ai_result.get("shared_skills", []),
                "c1_unique": ai_result.get("candidate_a_unique", []),
                "c2_unique": ai_result.get("candidate_b_unique", [])
            }
        }

    def generate_full_report(self, results: List[Dict], jd_text: str) -> Dict:
        return {"report": "Report generation requires full context."}

ai_service = AIAdapter()
