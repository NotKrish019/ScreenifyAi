"""
AI Resume Screening System - New Simplified Backend
Uses OpenAI for intelligent resume analysis
"""

import os
import uuid
import logging
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# PDF parsing
import pdfplumber
from docx import Document

# OpenAI service
from openai_service import analyze_resume, batch_analyze_resumes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Directories
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# App state
class AppState:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.job_description: Optional[str] = None
        self.resumes: List[dict] = []
        self.results: Optional[List[dict]] = None

app_state = AppState()


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 AI Resume Screening System starting...")
    # Clear old uploads
    for f in UPLOAD_DIR.iterdir():
        try:
            f.unlink()
        except:
            pass
    logger.info("✅ System ready!")
    yield
    logger.info("👋 Shutting down...")


# FastAPI app
app = FastAPI(
    title="AI Resume Screening API",
    description="OpenAI-powered resume analysis",
    version="2.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- PDF/DOCX Parsing ---
def extract_text(file_path: Path) -> str:
    """Extract text from PDF or DOCX file."""
    ext = file_path.suffix.lower()
    
    try:
        if ext == '.pdf':
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        
        elif ext in ['.docx', '.doc']:
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        
        elif ext == '.txt':
            return file_path.read_text()
        
        else:
            return ""
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""


# --- Pydantic Models ---
class JobDescription(BaseModel):
    text: str


# --- API Endpoints ---
@app.get("/")
async def root():
    return {"status": "running", "version": "2.0", "engine": "OpenAI GPT-4o-mini"}


@app.post("/upload-jd")
async def upload_job_description(jd: JobDescription):
    """Upload job description text."""
    if not jd.text or len(jd.text) < 50:
        raise HTTPException(400, "Job description too short")
    
    app_state.job_description = jd.text
    logger.info(f"📝 JD uploaded: {len(jd.text)} chars")
    
    return {"success": True, "length": len(jd.text)}


@app.post("/upload-resumes")
async def upload_resumes(files: List[UploadFile] = File(...)):
    """Upload resume files (PDF, DOCX, TXT)."""
    
    if len(files) + len(app_state.resumes) > 10:
        raise HTTPException(400, f"Maximum 10 resumes. Current: {len(app_state.resumes)}")
    
    uploaded = []
    errors = []
    
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in ['.pdf', '.docx', '.txt']:
            errors.append(f"{file.filename}: Invalid format")
            continue
        
        # Save file
        file_id = str(uuid.uuid4())[:8]
        file_path = UPLOAD_DIR / f"{file_id}{ext}"
        
        content = await file.read()
        file_path.write_bytes(content)
        
        # Extract text
        text = extract_text(file_path)
        if not text or len(text) < 100:
            errors.append(f"{file.filename}: Could not extract text")
            file_path.unlink()
            continue
        
        # Store
        resume_data = {
            "id": file_id,
            "name": file.filename,
            "text": text,
            "path": str(file_path)
        }
        app_state.resumes.append(resume_data)
        uploaded.append({"id": file_id, "name": file.filename})
        logger.info(f"📄 Uploaded: {file.filename}")
    
    return {
        "success": True,
        "uploaded": uploaded,
        "errors": errors,
        "total": len(app_state.resumes)
    }


@app.get("/resumes")
async def get_resumes():
    """Get list of uploaded resumes."""
    return {
        "resumes": [{"id": r["id"], "name": r["name"]} for r in app_state.resumes],
        "total": len(app_state.resumes)
    }


@app.delete("/resumes/{resume_id}")
async def remove_resume(resume_id: str):
    """Remove a resume by ID."""
    for i, r in enumerate(app_state.resumes):
        if r["id"] == resume_id:
            # Delete file
            try:
                Path(r["path"]).unlink()
            except:
                pass
            app_state.resumes.pop(i)
            return {"success": True}
    
    raise HTTPException(404, "Resume not found")


@app.post("/analyze")
async def analyze():
    """Run AI analysis on all resumes."""
    
    if not app_state.job_description:
        raise HTTPException(400, "No job description uploaded")
    
    if len(app_state.resumes) == 0:
        raise HTTPException(400, "No resumes uploaded")
    
    logger.info(f"🤖 Starting AI analysis of {len(app_state.resumes)} resumes...")
    
    # Run AI analysis
    results = batch_analyze_resumes(
        resumes=[{"text": r["text"], "name": r["name"]} for r in app_state.resumes],
        job_description=app_state.job_description
    )
    
    app_state.results = results
    logger.info(f"✅ Analysis complete! Top score: {results[0]['match_score']}%")
    
    return {
        "success": True,
        "total_analyzed": len(results),
        "top_candidate": results[0]["candidate_name"] if results else None
    }


@app.get("/results")
async def get_results():
    """Get analysis results."""
    
    if not app_state.results:
        raise HTTPException(404, "No results. Run /analyze first.")
    
    return {
        "success": True,
        "total": len(app_state.results),
        "results": app_state.results
    }


@app.post("/reset")
async def reset():
    """Reset all data."""
    # Clear files
    for r in app_state.resumes:
        try:
            Path(r["path"]).unlink()
        except:
            pass
    
    app_state.reset()
    logger.info("🔄 System reset")
    
    return {"success": True, "message": "All data cleared"}


# Run with: uvicorn app:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
