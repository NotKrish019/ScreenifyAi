# This __init__.py allows 'from config import X' to work with the config/ directory
# It re-exports everything from the parent backend/config.py file

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import from parent config.py and re-export
try:
    # Try direct import from parent directory's config.py
    import importlib.util
    config_path = os.path.join(parent_dir, 'config.py')
    spec = importlib.util.spec_from_file_location("backend_config", config_path)
    backend_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend_config)
    
    # Re-export everything
    SCORE_THRESHOLDS = backend_config.SCORE_THRESHOLDS
    SUMMARY_TEMPLATES = backend_config.SUMMARY_TEMPLATES
    BASE_DIR = backend_config.BASE_DIR
    UPLOAD_DIR = backend_config.UPLOAD_DIR
    JD_UPLOAD_DIR = backend_config.JD_UPLOAD_DIR
    RESUME_UPLOAD_DIR = backend_config.RESUME_UPLOAD_DIR
    DATASETS_DIR = backend_config.DATASETS_DIR
    JOB_ROLES_PATH = backend_config.JOB_ROLES_PATH
    SKILLS_DATASET_PATH = backend_config.SKILLS_DATASET_PATH
    MAX_RESUMES = backend_config.MAX_RESUMES
    ALLOWED_EXTENSIONS = backend_config.ALLOWED_EXTENSIONS
    MAX_FILE_SIZE_MB = backend_config.MAX_FILE_SIZE_MB
    TFIDF_CONFIG = backend_config.TFIDF_CONFIG
    SKILL_WEIGHTS = backend_config.SKILL_WEIGHTS
    CORS_ORIGINS = backend_config.CORS_ORIGINS
    API_TITLE = backend_config.API_TITLE
    API_DESCRIPTION = backend_config.API_DESCRIPTION
    API_VERSION = backend_config.API_VERSION
except Exception as e:
    print(f"Error loading backend config: {e}")
    # Fallback defaults
    SCORE_THRESHOLDS = {"high": 75, "medium": 50, "low": 0}
    SUMMARY_TEMPLATES = {"high": [], "medium": [], "low": []}
