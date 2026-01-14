import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import logging
import os
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

# Global client
db = None

def init_firestore():
    global db
    if db:
        return db

    try:
        # Check for service account file (relative to this file)
        base_path = os.path.dirname(os.path.abspath(__file__))
        cred_path = os.path.join(base_path, "serviceAccountKey.json")
        
        if not os.path.exists(cred_path):
            logger.warning(f"Firestore warning: {cred_path} not found. History features will be disabled.")
            return None

        # Initialize app
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        
        db = firestore.client()
        logger.info("Firestore initialized successfully.")
        return db
    except Exception as e:
        logger.error(f"Failed to initialize Firestore: {e}")
        return None

def save_analysis_result(jd_text, results):
    """
    Save the analysis results to Firestore.
    """
    db = init_firestore()
    if not db:
        return None

    try:
        # Create a summary object
        timestamp = datetime.now()
        
        # Identify top candidate
        top_candidate = "None"
        if results and len(results) > 0:
            top_candidate = results[0].get('resume_name', 'Unknown')
            
        doc_data = {
            "timestamp": timestamp,
            "jd_preview": jd_text[:200] + "..." if len(jd_text) > 200 else jd_text,
            "candidate_count": len(results),
            "top_candidate": top_candidate,
            "full_results": results  # Save full results for detail view
        }
        
        # Add to 'history' collection
        update_time, doc_ref = db.collection('history').add(doc_data)
        logger.info(f"Analysis saved to history with ID: {doc_ref.id}")
        return doc_ref.id
    except Exception as e:
        logger.error(f"Error saving to Firestore: {e}")
        return None

def get_history_list(limit=20):
    """
    Get a list of past analyses (lightweight view).
    """
    db = init_firestore()
    if not db:
        return []

    try:
        # Order by timestamp desc
        docs = db.collection('history').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit).stream()
        
        history = []
        for doc in docs:
            data = doc.to_dict()
            # Convert timestamp to string
            if 'timestamp' in data:
                data['timestamp'] = data['timestamp'].isoformat()
                
            # Exclude full results for the list view to save bandwidth
            if 'full_results' in data:
                del data['full_results']
                
            data['id'] = doc.id
            history.append(data)
            
        return history
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return []

def get_analysis_detail(analysis_id):
    """
    Get full details of a specific analysis.
    """
    db = init_firestore()
    if not db:
        return None
        
    try:
        doc_ref = db.collection('history').document(analysis_id)
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            if 'timestamp' in data:
                data['timestamp'] = data['timestamp'].isoformat()
            return data
        else:
            return None
    except Exception as e:
        logger.error(f"Error fetching detail: {e}")
        return None
