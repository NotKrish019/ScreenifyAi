"""
Configuration settings for the AI Resume Screening System.
Contains all constants, paths, and configurable parameters.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Upload directories
UPLOAD_DIR = BASE_DIR / "uploads"
JD_UPLOAD_DIR = UPLOAD_DIR / "jd"
RESUME_UPLOAD_DIR = UPLOAD_DIR / "resumes"

# Dataset paths
DATASETS_DIR = BASE_DIR / "datasets"
JOB_ROLES_PATH = DATASETS_DIR / "job_roles.json"
SKILLS_DATASET_PATH = DATASETS_DIR / "skills_dataset.json"

# Create directories if they don't exist
for directory in [UPLOAD_DIR, JD_UPLOAD_DIR, RESUME_UPLOAD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File constraints
MAX_RESUMES = 5
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}
MAX_FILE_SIZE_MB = 10

# Scoring thresholds
SCORE_THRESHOLDS = {
    "high": 75,      # >= 75% is High Fit
    "medium": 50,    # 50-74% is Medium Fit
    "low": 0         # < 50% is Low Fit
}

# TF-IDF Configuration
TFIDF_CONFIG = {
    "max_features": 5000,
    "ngram_range": (1, 2),      # Unigrams and bigrams
    "min_df": 1,                # Minimum document frequency
    "max_df": 0.95,             # Maximum document frequency
    "sublinear_tf": True,       # Apply sublinear tf scaling
}

# Skill matching weights
SKILL_WEIGHTS = {
    "exact_match": 1.0,
    "partial_match": 0.5,
    "category_match": 0.3,
}

# Summary generation templates
SUMMARY_TEMPLATES = {
    "high": [
        "Excellent candidate with strong alignment to role requirements. Demonstrates comprehensive skill coverage and relevant experience.",
        "Outstanding match for the position. Key qualifications and technical skills closely align with job requirements.",
        "Highly qualified candidate showing exceptional fit. Strong technical background with relevant industry experience."
    ],
    "medium": [
        "Good candidate with relevant skills and experience. Some areas may benefit from additional development.",
        "Solid match for the role with transferable skills. Background shows potential for growth in this position.",
        "Promising candidate with applicable experience. Core competencies align well with several key requirements."
    ],
    "low": [
        "Candidate shows some relevant skills but may require additional training for this specific role.",
        "Partial alignment with role requirements. Consider for positions with different skill emphasis.",
        "Limited match with current requirements. May be suitable for entry-level or adjacent positions."
    ]
}

# CORS settings
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "*"  # Allow all origins for development
]

# API settings
API_TITLE = "AI Resume Screening System"
API_DESCRIPTION = """
A sophisticated AI-powered resume screening system that:
- Extracts text from PDF/DOCX resumes
- Analyzes job descriptions
- Matches candidates to roles using NLP
- Ranks resumes by fit score
"""
API_VERSION = "1.0.0"