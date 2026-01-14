"""
AI Resume Screening System - Simplified Backend
Focused on reliability and direct TF-IDF matching.
"""

import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Local imports
# We define simple models here to avoid dependency on the old models.py if it causes issues
class JobDescriptionText(BaseModel):
    text: str

# Configs
UPLOAD_DIR = Path("uploads")
JD_UPLOAD_DIR = UPLOAD_DIR / "jd"
RESUME_UPLOAD_DIR = UPLOAD_DIR / "resumes"

for d in [JD_UPLOAD_DIR, RESUME_UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix Path for Render/Vercel (adds 'backend' to sys.path)
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Logic Imports
from resume_parser import resume_parser
from ai_service import ai_service
import firestore_db

# State
class AppState:
    def __init__(self):
        self.jd_text = None
        self.resumes = [] # List of dicts
        self.results = []
        self.analysis_complete = False
    
    def reset(self):
        self.jd_text = None
        self.resumes = []
        self.results = []
        self.analysis_complete = False

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cleanup on start
    for d in [JD_UPLOAD_DIR, RESUME_UPLOAD_DIR]:
        for f in d.glob("*"):
            try: f.unlink()
            except: pass
    logger.info("System Ready (Simplified Mode)")
    yield
    app_state.reset()

app = FastAPI(title="Resume Screener Lite", lifespan=lifespan)

# STRICT CORS - The User's main complaint
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

# SERVE FRONTEND (Added for Render/Deployment)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Calculate paths relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent # Root of repo
FRONTEND_DIR = BASE_DIR / "frontend"

# Mount static assets (css, js, etc.)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def read_index():
    return FileResponse(FRONTEND_DIR / "index.html")

# --- Endpoints ---

# --- Endpoints ---

@app.post("/upload-jd-text")
def upload_jd_text(jd: JobDescriptionText):
    app_state.jd_text = jd.text
    logger.info("JD Text Uploaded")
    return {"success": True}

@app.post("/upload-jd-file")
def upload_jd_file(file: UploadFile = File(...)):
    """Upload a JD file (PDF, DOCX, TXT) and extract text from it."""
    file_id = str(uuid.uuid4())[:8]
    file_path = JD_UPLOAD_DIR / f"{file_id}_{file.filename}"
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Extract text using the resume parser (works for any document)
    text, success = resume_parser.extract_text(str(file_path))
    
    if success and text.strip():
        # Append to existing JD text (allows combining file + text)
        if app_state.jd_text:
            app_state.jd_text = app_state.jd_text + "\n\n" + text
        else:
            app_state.jd_text = text
        logger.info(f"JD file uploaded and extracted: {file.filename} ({len(text)} chars)")
        return {"success": True, "extracted_text": text, "filename": file.filename}
    else:
        logger.error(f"Failed to extract text from JD file: {file.filename}")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from {file.filename}. Please try a different file format.")

@app.post("/upload-resumes")
def upload_resumes(files: List[UploadFile] = File(...)):
    uploaded = []
    for file in files:
        file_id = str(uuid.uuid4())[:8]
        file_path = RESUME_UPLOAD_DIR / f"{file_id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        # Extract immediately
        text, success = resume_parser.extract_text(str(file_path))
        if success:
            app_state.resumes.append({
                "id": file_id,
                "name": file.filename,
                "path": str(file_path),
                "original_text": text
            })
            uploaded.append({"id": file_id, "name": file.filename})
            logger.info(f"Uploaded and extracted: {file.filename}")
        else:
            logger.error(f"Failed to extract text: {file.filename}")
            
    return {"success": True, "uploaded": uploaded}

@app.delete("/resumes/{resume_id}")
def remove_resume(resume_id: str):
    # Filter list
    app_state.resumes = [r for r in app_state.resumes if r['id'] != resume_id]
    return {"success": True}

@app.post("/analyze")
def analyze():
    # Defensive checks
    if not app_state.jd_text:
        logger.error("Analyze called but JD is missing")
        raise HTTPException(status_code=400, detail="Job Description is missing. Please upload it first.")
    
    if not app_state.resumes:
        logger.error("Analyze called but Resumes are missing")
        raise HTTPException(status_code=400, detail="No resumes found. Please upload resumes first.")
        
    try:
        logger.info(f"Analyzing {len(app_state.resumes)} resumes against JD ({len(app_state.jd_text)} chars)...")
        
        # Run matching logic using AI Service
        # This will use local NLP by default, but is structured to support LLM if configured
        
        # We need to process each resume
        results = []
        for resume in app_state.resumes:
             # Use ai_service for analysis
             analysis = ai_service.analyze_resume(
                 resume.get('original_text', ''), 
                 app_state.jd_text, 
                 resume.get('name', 'Unknown')
             )
             
             # Enrich with ID
             analysis['id'] = resume['id']
             analysis['resume_name'] = resume['name']
             
             # Format explanation for frontend compatibility
             analysis['explanation'] = {
                 "summary": analysis.get('summary', ''),
                 "strengths": [f"Proficient in {s}" for s in analysis.get('matched_skills', [])[:3]],
                 "tips": [analysis.get('recommendation', '')]
             }
             
             # Format fit to match frontend expectation (capitalized)
             analysis['fit'] = analysis.get('fit_level', 'Low')
             
             results.append(analysis)
        
        # Sort results
        results.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Add rank
        for i, r in enumerate(results):
            r['rank'] = i + 1

        if not results:
             logger.warning("Matcher returned empty results")
             
        app_state.results = results
        app_state.analysis_complete = True
        logger.info("Analysis completed successfully")
        

        # --- Save to History ---
        try:
            history_id = firestore_db.save_analysis_result(app_state.jd_text, results)
            if history_id:
                logger.info(f"Analysis saved to history ({history_id})")
        except Exception as db_err:
            logger.error(f"Failed to save history: {db_err}")
            # Don't fail the request if DB fails

        return {"success": True, "count": len(results)}
        
    except Exception as e:
        logger.error(f"CRITICAL ANALYSIS ERROR: {str(e)}", exc_info=True)
        # Return a 500 that the frontend can parse, rather than a hard crash
        raise HTTPException(status_code=500, detail=f"Server Analysis Failed: {str(e)}")

# Include the new extension endpoints
# (Integrated directly below to avoid circular imports)




@app.get("/history")
def get_history_list():
    """Get list of past analyses"""
    return firestore_db.get_history_list()

@app.get("/history/{analysis_id}")
def get_history_detail(analysis_id: str):
    """Get full detail of a past analysis"""
    data = firestore_db.get_analysis_detail(analysis_id)
    if not data:
        raise HTTPException(status_code=404, detail="History item not found")
    return data

@app.get("/results")
def get_results(use_ai: bool = False):
    return {"results": app_state.results}

@app.post("/reset")
def reset():
    app_state.reset()
    return {"success": True}
    
@app.get("/health")
def health():
    return {"status": "ok"}
@app.post("/improve-jd")
def improve_jd(payload: Optional[JobDescriptionText] = None):
    # Allow passing text directly, or fallback to stored text
    text_to_process = None
    
    if payload and payload.text:
        text_to_process = payload.text
        # Optional: update state too?
        app_state.jd_text = payload.text
    elif app_state.jd_text:
        text_to_process = app_state.jd_text
        
    if not text_to_process:
         raise HTTPException(status_code=400, detail="No JD provided. Please enter text or upload a file.")
    
    try:
        # Use our new AI service which handles LLM or Local fallback automatically
        improved_data = ai_service.improve_job_description(text_to_process)
        return {"success": True, "analysis": improved_data}
    except Exception as e:
        logger.error(f"JD Improve Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare")
def compare_candidates(payload: dict):
    # Payload: {"candidate_ids": ["id1", "id2"]}
    ids = payload.get("candidate_ids", [])
    if len(ids) != 2:
        raise HTTPException(status_code=400, detail="Select exactly 2 candidates")
        
    # Find candidates
    c1 = next((r for r in app_state.results if r['id'] == ids[0]), None)
    c2 = next((r for r in app_state.results if r['id'] == ids[1]), None)
    
    if not c1 or not c2:
        # Fallback to resumes list if results aren't populated yet? 
        # Actually comparison usually happens AFTER analysis.
        # But let's check app_state.resumes just in case we need raw text.
        c1_raw = next((r for r in app_state.resumes if r['id'] == ids[0]), {})
        c2_raw = next((r for r in app_state.resumes if r['id'] == ids[1]), {})
        
        # Merge basic info if found in results
        if c1: c1_raw.update(c1)
        if c2: c2_raw.update(c2)
        
        c1 = c1_raw
        c2 = c2_raw

    if not c1 or not c2:
         raise HTTPException(status_code=404, detail="Candidates not found")

    # Ensure we have the text for AI analysis
    # We first check if 'original_text' is in the result object (it might not be if filtered)
    # If not, we look it up in app_state.resumes
    
    def get_resume_text(cand_id, cand_obj):
        if 'original_text' in cand_obj and cand_obj['original_text']: 
            return cand_obj['original_text']
        # Look up in raw resumes
        raw = next((r for r in app_state.resumes if r['id'] == cand_id), None)
        if raw and 'original_text' in raw:
            return raw['original_text']
        # Fallback to summary if text is totally gone
        return cand_obj.get('summary', '')

    c1['original_text'] = get_resume_text(ids[0], c1)
    c2['original_text'] = get_resume_text(ids[1], c2)

    try:
        result = ai_service.compare_candidates([c1, c2], app_state.jd_text)
        
        # Inject existing scores and names for the frontend visualization
        result['comparison']['c1_score'] = c1.get('match_score', 0)
        result['comparison']['c2_score'] = c2.get('match_score', 0)
        # Prefer names from result if available, else from comparison
        result['comparison']['c1_name'] = c1.get('resume_name', result['comparison'].get('c1_name', 'Candidate 1'))
        result['comparison']['c2_name'] = c2.get('resume_name', result['comparison'].get('c2_name', 'Candidate 2'))
        
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Comparison Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-report")
def generate_report():
    """Generate a full AI-written report for all analyzed candidates."""
    if not app_state.results:
        raise HTTPException(status_code=400, detail="No analysis results available. Please run analysis first.")
    
    if not app_state.jd_text:
        raise HTTPException(status_code=400, detail="No job description available.")
    
    try:
        # Use the AI service to generate a comprehensive report
        report_data = ai_service.generate_full_report(app_state.results, app_state.jd_text)
        return report_data
    except Exception as e:
        logger.error(f"Report Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
