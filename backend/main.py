"""
AI Resume Screening System - FastAPI Backend
Main entry point for the application.

This system:
1. Accepts job descriptions (text or file)
2. Accepts exactly 5 resume files (PDF/DOCX)
3. Extracts and preprocesses text
4. Computes similarity using TF-IDF and cosine similarity
5. Ranks candidates and returns structured results
"""

import os
import uuid
import shutil
import logging
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Local imports
from config import (
    CORS_ORIGINS, API_TITLE, API_DESCRIPTION, API_VERSION,
    JD_UPLOAD_DIR, RESUME_UPLOAD_DIR, MAX_RESUMES, ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_MB
)
from models import (
    JobDescriptionText, ResumeResult, AnalysisResponse,
    UploadResponse, ErrorResponse, HealthResponse
)
from resume_parser import resume_parser
from nlp_engine import nlp_engine
from similarity_engine import similarity_engine
from ranking import ranking_engine, CandidateScore
from ai_service import ai_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Application State ---
class AppState:
    """Global application state management."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset application state."""
        self.jd_text: Optional[str] = None
        self.jd_preprocessed: Optional[str] = None
        self.jd_analysis: Optional[dict] = None
        self.resumes: List[dict] = []
        self.results: Optional[List[ResumeResult]] = None
        self.analysis_complete: bool = False

app_state = AppState()


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    logger.info("Starting AI Resume Screening System...")
    
    # Verify NLP engine is ready
    if not nlp_engine.is_ready():
        logger.warning("NLP Engine datasets not fully loaded")
    else:
        logger.info("NLP Engine ready with all datasets")
    
    # Clear old uploads
    for directory in [JD_UPLOAD_DIR, RESUME_UPLOAD_DIR]:
        if directory.exists():
            for file in directory.iterdir():
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete {file}: {e}")
    
    logger.info("System ready to accept requests")
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Resume Screening System...")
    app_state.reset()


# --- FastAPI App ---
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helper Functions ---
def validate_file_extension(filename: str) -> bool:
    """Validate file has allowed extension."""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def generate_file_id() -> str:
    """Generate unique file ID."""
    return str(uuid.uuid4())[:8]


def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    """Save uploaded file to destination."""
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination
    finally:
        upload_file.file.close()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


# --- API Endpoints ---

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "upload_jd": "/upload-jd",
            "upload_jd_text": "/upload-jd-text",
            "upload_resumes": "/upload-resumes",
            "analyze": "/analyze",
            "results": "/results",
            "reset": "/reset"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        nlp_ready=nlp_engine.is_ready(),
        datasets_loaded=bool(nlp_engine.skills_data and nlp_engine.job_roles)
    )


@app.post("/upload-jd", response_model=UploadResponse, tags=["Upload"])
async def upload_job_description_file(file: UploadFile = File(...)):
    """
    Upload a job description file (PDF, DOCX, or TXT).
    
    The file content will be extracted and stored for analysis.
    """
    logger.info(f"Received JD file upload: {file.filename}")
    
    # Validate file extension
    if not validate_file_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Generate unique filename
    file_id = generate_file_id()
    ext = Path(file.filename).suffix.lower()
    new_filename = f"jd_{file_id}{ext}"
    file_path = JD_UPLOAD_DIR / new_filename
    
    try:
        # Save file
        save_upload_file(file, file_path)
        logger.info(f"JD file saved: {file_path}")
        
        # Extract text
        extracted_text, success = resume_parser.extract_text(str(file_path))
        
        if not success or not extracted_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract text from the uploaded file"
            )
        
        # Preprocess text
        preprocessed = nlp_engine.preprocess(extracted_text)
        
        # Store in state
        app_state.jd_text = extracted_text
        app_state.jd_preprocessed = preprocessed
        app_state.results = None  # Reset results
        app_state.analysis_complete = False
        
        logger.info(f"JD processed successfully: {len(extracted_text)} chars")
        
        return UploadResponse(
            success=True,
            message="Job description uploaded and processed successfully",
            filename=file.filename,
            file_id=file_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing JD file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )


@app.post("/upload-jd-text", response_model=UploadResponse, tags=["Upload"])
async def upload_job_description_text(jd: JobDescriptionText):
    """
    Upload job description as plain text.
    
    Minimum 50 characters required.
    """
    logger.info(f"Received JD text upload: {len(jd.text)} chars")
    
    if len(jd.text.strip()) < 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job description must be at least 50 characters"
        )
    
    try:
        # Preprocess text
        preprocessed = nlp_engine.preprocess(jd.text)
        
        # Store in state
        app_state.jd_text = jd.text
        app_state.jd_preprocessed = preprocessed
        app_state.results = None
        app_state.analysis_complete = False
        
        logger.info("JD text processed successfully")
        
        return UploadResponse(
            success=True,
            message="Job description text received and processed",
            filename="text_input",
            file_id=generate_file_id()
        )
        
    except Exception as e:
        logger.error(f"Error processing JD text: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing text: {str(e)}"
        )


@app.post("/upload-resumes", tags=["Upload"])
async def upload_resumes(files: List[UploadFile] = File(...)):
    """
    Upload resume files (PDF or DOCX).
    
    Exactly 5 resumes are required for analysis.
    Files can be uploaded in multiple batches until 5 are received.
    """
    logger.info(f"Received {len(files)} resume file(s)")
    
    # Check if adding these would exceed limit
    current_count = len(app_state.resumes)
    if current_count + len(files) > MAX_RESUMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot add {len(files)} resumes. Current: {current_count}, Maximum: {MAX_RESUMES}"
        )
    
    uploaded = []
    errors = []
    
    for file in files:
        # Validate extension
        if not validate_file_extension(file.filename):
            errors.append(f"{file.filename}: Invalid file type")
            continue
        
        # Generate unique ID and filename
        file_id = generate_file_id()
        ext = Path(file.filename).suffix.lower()
        new_filename = f"resume_{file_id}{ext}"
        file_path = RESUME_UPLOAD_DIR / new_filename
        
        try:
            # Save file
            save_upload_file(file, file_path)
            
            # Extract text
            extracted_text, success = resume_parser.extract_text(str(file_path))
            
            if not success or not extracted_text.strip():
                errors.append(f"{file.filename}: Could not extract text")
                file_path.unlink()  # Remove failed file
                continue
            
            # Validate content
            is_valid, validation_msg = resume_parser.validate_resume_content(extracted_text)
            if not is_valid:
                errors.append(f"{file.filename}: {validation_msg}")
                file_path.unlink()
                continue
            
            # Preprocess
            preprocessed = nlp_engine.preprocess(extracted_text)
            
            # Get file size
            file_size = file_path.stat().st_size
            
            # Store resume data
            resume_data = {
                "id": file_id,
                "name": file.filename,
                "size": format_file_size(file_size),
                "path": str(file_path),
                "original_text": extracted_text,
                "preprocessed_text": preprocessed,
                "status": "ready"
            }
            
            app_state.resumes.append(resume_data)
            uploaded.append({
                "id": file_id,
                "name": file.filename,
                "size": format_file_size(file_size),
                "status": "ready"
            })
            
            logger.info(f"Resume processed: {file.filename}")
            
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")
            logger.error(f"Error processing resume {file.filename}: {str(e)}")
    
    # Reset results when resumes change
    app_state.results = None
    app_state.analysis_complete = False
    
    return {
        "success": len(uploaded) > 0,
        "message": f"Uploaded {len(uploaded)} resume(s). Total: {len(app_state.resumes)}/{MAX_RESUMES}",
        "uploaded": uploaded,
        "errors": errors if errors else None,
        "total_resumes": len(app_state.resumes),
        "required": MAX_RESUMES,
        "ready_for_analysis": len(app_state.resumes) == MAX_RESUMES
    }


@app.delete("/remove-resume/{resume_id}", tags=["Upload"])
async def remove_resume(resume_id: str):
    """Remove a specific resume by ID."""
    
    for i, resume in enumerate(app_state.resumes):
        if resume["id"] == resume_id:
            # Remove file
            try:
                Path(resume["path"]).unlink()
            except:
                pass
            
            # Remove from state
            app_state.resumes.pop(i)
            app_state.results = None
            app_state.analysis_complete = False
            
            return {
                "success": True,
                "message": f"Resume {resume['name']} removed",
                "total_resumes": len(app_state.resumes)
            }
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Resume with ID {resume_id} not found"
    )


@app.get("/resumes", tags=["Upload"])
async def get_uploaded_resumes():
    """Get list of currently uploaded resumes."""
    return {
        "resumes": [
            {
                "id": r["id"],
                "name": r["name"],
                "size": r["size"],
                "status": r["status"]
            }
            for r in app_state.resumes
        ],
        "total": len(app_state.resumes),
        "required": MAX_RESUMES,
        "ready_for_analysis": len(app_state.resumes) == MAX_RESUMES and app_state.jd_text is not None
    }


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_resumes():
    """
    Analyze all uploaded resumes against the job description.
    
    Requires:
    - Job description to be uploaded
    - Exactly 5 resumes to be uploaded
    
    Returns ranked results with match scores and analysis.
    """
    logger.info("Starting resume analysis...")
    
    # Validate prerequisites
    if not app_state.jd_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No job description uploaded. Please upload a JD first."
        )
    
    if len(app_state.resumes) != MAX_RESUMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Exactly {MAX_RESUMES} resumes required. Current: {len(app_state.resumes)}"
        )
    
    try:
        # Set job description in similarity engine
        app_state.jd_analysis = similarity_engine.set_job_description(
            app_state.jd_text,
            app_state.jd_preprocessed
        )
        
        logger.info(f"JD Analysis: {app_state.jd_analysis}")
        
        # Prepare resume data for batch processing
        resume_data = [
            {
                "original": r["original_text"],
                "preprocessed": r["preprocessed_text"]
            }
            for r in app_state.resumes
        ]
        
        # Compute similarities for all resumes
        similarity_results = similarity_engine.batch_compute_similarity(resume_data)
        
        # Create candidate scores
        candidates = []
        for i, resume in enumerate(app_state.resumes):
            candidate = CandidateScore(
                resume_id=resume["id"],
                resume_name=resume["name"],
                original_text=resume["original_text"],
                preprocessed_text=resume["preprocessed_text"],
                similarity_data=similarity_results[i]
            )
            candidates.append(candidate)
        
        # Rank candidates
        ranked_results = ranking_engine.rank_candidates(
            candidates,
            job_title=app_state.jd_analysis.get("detected_role")
        )
        
        # Get ranking statistics
        stats = ranking_engine.get_ranking_stats(ranked_results)
        
        # Store results
        app_state.results = ranked_results
        app_state.analysis_complete = True
        
        logger.info(f"Analysis complete. Top score: {ranked_results[0].match_score}%")
        
        # Convert results to dict for response
        results_dict = [
            {
                "rank": r.rank,
                "id": r.id,
                "resume_name": r.resume_name,
                "match_score": r.match_score,
                "fit": r.fit.value,
                "matched_skills": r.matched_skills,
                "missing_skills": r.missing_skills,
                "summary": r.summary,
                "skill_breakdown": r.skill_breakdown
            }
            for r in ranked_results
        ]
        
        return AnalysisResponse(
            success=True,
            message="Analysis completed successfully",
            job_title_detected=app_state.jd_analysis.get("detected_role"),
            total_candidates=len(ranked_results),
            results=ranked_results,
            analysis_metadata={
                "jd_skills_count": len(app_state.jd_analysis.get("skills_required", [])),
                "experience_level": app_state.jd_analysis.get("experience_level"),
                "statistics": stats
            }
        )
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get("/results", tags=["Analysis"])
async def get_results(use_ai: bool = False):
    """
    Get the latest analysis results.
    
    Set use_ai=true for AI-enhanced summaries (slower but more detailed).
    Default is fast mode without AI.
    """
    if not app_state.analysis_complete or not app_state.results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis results available. Please run /analyze first."
        )
    
    # Get statistics
    stats = ranking_engine.get_ranking_stats(app_state.results)
    
    # Fast mode - return results as-is
    if not use_ai:
        return {
            "success": True,
            "job_title_detected": app_state.jd_analysis.get("detected_role") if app_state.jd_analysis else None,
            "total_candidates": len(app_state.results),
            "results": [
                {
                    "rank": r.rank,
                    "id": r.id,
                    "resume_name": r.resume_name,
                    "match_score": r.match_score,
                    "fit": r.fit.value,
                    "matched_skills": r.matched_skills,
                    "missing_skills": r.missing_skills,
                    "summary": r.summary
                }
                for r in app_state.results
            ],
            "statistics": stats,
            "ai_enhanced": False
        }
    
    # AI mode - generate enhanced summaries
    enhanced_results = []
    for r in app_state.results:
        resume_data = next((res for res in app_state.resumes if res["id"] == r.id), None)
        ai_summary = r.summary
        ai_skills = r.matched_skills
        
        if resume_data and app_state.jd_text:
            try:
                ai_result = ai_service.analyze_resume(
                    resume_data["original_text"],
                    app_state.jd_text
                )
                if ai_result.get("summary"):
                    ai_summary = ai_result["summary"]
                if ai_result.get("skills_found"):
                    ai_skills = ai_result["skills_found"]
            except Exception as e:
                logger.warning(f"AI analysis failed for {r.resume_name}: {e}")
        
        enhanced_results.append({
            "rank": r.rank,
            "id": r.id,
            "resume_name": r.resume_name,
            "match_score": r.match_score,
            "fit": r.fit.value,
            "matched_skills": ai_skills,
            "missing_skills": r.missing_skills,
            "summary": ai_summary
        })
    
    return {
        "success": True,
        "job_title_detected": app_state.jd_analysis.get("detected_role") if app_state.jd_analysis else None,
        "total_candidates": len(app_state.results),
        "results": enhanced_results,
        "statistics": stats,
        "ai_enhanced": True
    }


@app.post("/reset", tags=["System"])
async def reset_system():
    """
    Reset the system state.
    
    Clears all uploaded files and analysis results.
    """
    logger.info("Resetting system state...")
    
    # Clear uploaded files
    for directory in [JD_UPLOAD_DIR, RESUME_UPLOAD_DIR]:
        if directory.exists():
            for file in directory.iterdir():
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Could not delete {file}: {e}")
    
    # Reset state
    app_state.reset()
    
    logger.info("System reset complete")
    
    return {
        "success": True,
        "message": "System reset successfully. All data cleared."
    }


@app.get("/jd-status", tags=["Upload"])
async def get_jd_status():
    """Get current job description status and analysis."""
    if not app_state.jd_text:
        return {
            "uploaded": False,
            "message": "No job description uploaded"
        }
    
    return {
        "uploaded": True,
        "text_length": len(app_state.jd_text),
        "analysis": app_state.jd_analysis
    }


# --- Error Handlers ---
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# --- Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )