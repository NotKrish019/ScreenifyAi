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

# Logic Imports
from resume_parser import resume_parser
from simple_matcher import simple_matcher

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
    allow_credentials=True, # Changed back to True but with * is sometimes risky in production but fine for localhost uvicorn usually. 
                            # actually the user's error was "Access-Control-Allow-Origin" header missing.
                            # Starlette/FastAPI handles this well with ["*"] usually.
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---

@app.post("/upload-jd-text")
def upload_jd_text(jd: JobDescriptionText):
    app_state.jd_text = jd.text
    logger.info("JD Text Uploaded")
    return {"success": True}

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
            uploaded.append(file.filename)
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
        
        # Run matching logic
        results = simple_matcher.rank_resumes(app_state.jd_text, app_state.resumes)
        
        if not results:
             logger.warning("Matcher returned empty results")
             
        app_state.results = results
        app_state.analysis_complete = True
        logger.info("Analysis completed successfully")
        
        return {"success": True, "count": len(results)}
        
    except Exception as e:
        logger.error(f"CRITICAL ANALYSIS ERROR: {str(e)}", exc_info=True)
        # Return a 500 that the frontend can parse, rather than a hard crash
        raise HTTPException(status_code=500, detail=f"Server Analysis Failed: {str(e)}")

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