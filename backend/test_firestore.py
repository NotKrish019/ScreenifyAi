import firebase_admin
from firebase_admin import credentials, firestore
import os
import sys

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def test_connection():
    print("🔥 Testing Firestore Connection...")
    
    # Use absolute path relative to this script
    base_path = os.path.dirname(os.path.abspath(__file__))
    key_path = os.path.join(base_path, "serviceAccountKey.json")
    
    # 1. Check if file exists
    if not os.path.exists(key_path):
        print(f"{RED}❌ Error: '{key_path}' not found in the current directory.{RESET}")
        print("   Please download the private key from Firebase Console -> Project Settings -> Service Accounts.")
        print("   Rename it to 'serviceAccountKey.json' and place it in this 'backend' folder.")
        return

    try:
        # 2. Try to initialize
        if not firebase_admin._apps:
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)
            
        db = firestore.client()
        
        # 3. Try to write a test document
        print("   Attempting to write test document...")
        doc_ref = db.collection('test_connection').document('ping')
        doc_ref.set({
            'message': 'Hello from AI Resume Screener!',
            'timestamp': firestore.SERVER_TIMESTAMP
        })
        
        print(f"{GREEN}✅ SUCCESS! Connected to Firestore and wrote data.{RESET}")
        print("   You are ready to use the History feature.")
        
        # Cleanup
        doc_ref.delete()
        
    except Exception as e:
        print(f"{RED}❌ Connection Failed.{RESET}")
        print(f"   Error details: {e}")

if __name__ == "__main__":
    test_connection()
