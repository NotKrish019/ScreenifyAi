"""
Pydantic models for request/response validation.
Defines the data structures used throughout the API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class FitCategory(str, Enum):
    """Enum for candidate fit categories."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class JobDescriptionText(BaseModel):
    """Model for job description text input."""
    text: str = Field(..., min_length=50, description="Job description text content")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "We are looking for a Senior Software Engineer with 5+ years of experience in Python, React, and AWS..."
            }
        }


class ResumeInfo(BaseModel):
    """Model for basic resume information."""
    id: str = Field(..., description="Unique identifier for the resume")
    name: str = Field(..., description="Original filename of the resume")
    size: str = Field(..., description="File size in human-readable format")
    status: str = Field(default="ready", description="Processing status")


class ResumeResult(BaseModel):
    """Model for individual resume screening result."""
    rank: int = Field(..., ge=1, le=5, description="Ranking position (1-5)")
    id: str = Field(..., description="Unique identifier for the resume")
    resume_name: str = Field(..., description="Original filename")
    match_score: int = Field(..., ge=0, le=100, description="Match percentage score")
    fit: FitCategory = Field(..., description="Fit category (High/Medium/Low)")
    matched_skills: List[str] = Field(..., description="List of matched skills")
    missing_skills: List[str] = Field(default=[], description="List of missing required skills")
    summary: str = Field(..., description="AI-generated summary of the candidate")
    skill_breakdown: Optional[dict] = Field(default=None, description="Detailed skill category breakdown")


class AnalysisResponse(BaseModel):
    """Model for complete analysis response."""
    success: bool = Field(..., description="Whether analysis completed successfully")
    message: str = Field(..., description="Status message")
    job_title_detected: Optional[str] = Field(None, description="Detected job title from JD")
    total_candidates: int = Field(..., description="Number of candidates analyzed")
    results: List[ResumeResult] = Field(..., description="Ranked list of resume results")
    analysis_metadata: Optional[dict] = Field(None, description="Additional analysis information")


class UploadResponse(BaseModel):
    """Model for file upload response."""
    success: bool
    message: str
    filename: Optional[str] = None
    file_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Model for error responses."""
    success: bool = False
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str
    version: str
    nlp_ready: bool
    datasets_loaded: bool


class CompareRequest(BaseModel):
    """Model for candidate comparison request."""
    candidate_ids: List[str] = Field(..., min_items=2, description="List of candidate IDs to compare")