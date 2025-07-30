#!/usr/bin/env python3
"""
SafeInbox Email Phishing Detection API Server
Run this script to start the Flask API server for your Chrome extension.
"""

import os
import sys
from flask_email_api import app, load_model

def check_files():
    """Check if required files exist"""
    
    if not os.path.exists('optimized_phishing_model.pkl'):
        print("ERROR: Missing optimized_phishing_model.pkl")
        print("Please make sure you have:")
        print("1. Trained your model using the hyperparameter tuning script")
        print("2. Saved the model as 'optimized_phishing_model.pkl'")
        return False
    
    return True

def main():
    """Start the API server"""
    
    print("SafeInbox Email Phishing Detection API")
    print("=" * 40)
    
    # Check if model file exists
    if not check_files():
        sys.exit(1)
    
    # Try to load the model
    print("Loading model...")
    if not load_model():
        print("ERROR: Failed to load model")
        sys.exit(1)
    
    print("Model loaded successfully")
    print("Starting API server...")
    print("")
    print("API Endpoints:")
    print("  - POST http://localhost:5001/predict")
    print("  - GET  http://localhost:5001/health")
    print("")
    print("Chrome Extension Configuration:")
    print("  Email Phishing API Endpoint: http://localhost:5001/predict")
    print("")
    print("IMPORTANT: Set this URL in your Chrome extension settings!")
    print("-" * 40)
    
    try:
        # Start the Flask server
        app.run(host='0.0.0.0', port=5001, debug=False)
    except KeyboardInterrupt:
        print("\nAPI server stopped")
    except Exception as e:
        print(f"ERROR: {e}")

# Run the server
if __name__ == "__main__":
    main() 