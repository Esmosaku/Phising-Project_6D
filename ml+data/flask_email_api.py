#!/usr/bin/env python3
"""
SafeInbox Email Phishing Detection API
Flask API for the Chrome extension that detects phishing emails using machine learning.
"""

import os
import re
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
from datetime import datetime
from bs4 import BeautifulSoup
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModel
import torch

app = Flask(__name__)

# Enable CORS for all routes and origins (including Chrome extensions)
CORS(app, origins=["chrome-extension://*", "http://localhost:*", "https://localhost:*"])

# Global variables for model and scaler
model = None
scaler = None
bert_tokenizer = None
bert_model = None

def load_model():
    """Load the trained model and scaler"""
    global model, scaler, bert_tokenizer, bert_model
    
    try:
        # Load the pickled model and scaler
        with open('optimized_phishing_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            
        model = model_data['model']
        scaler = model_data['scaler']
        
        print("ML model and scaler loaded successfully")
        
        # Load BERT model and tokenizer
        print("Loading BERT model...")
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert_model = AutoModel.from_pretrained("bert-base-uncased")
        bert_model.eval()
        
        print("BERT model loaded successfully")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def count_keywords(text):
    """Count occurrences of suspicious keywords in text"""
    keywords = ["urgent", "verify your account", "click here", "login now", "password reset",
                "account suspended", "update your information", "confirm your identity",
                "secure your account", "action required"]
    text = text.lower()
    counts = {}
    for word in keywords:
        counts["count_" + word.replace(" ", "_")] = len(re.findall(r'\b' + word + r'\b', text))
    return counts

def check_greeting(text):
    """Check for generic greetings in first 200 characters"""
    greetings = ["dear customer", "dear user", "hello sir", "hello madam", "dear client"]
    first_bit = text.lower()[:200]
    for greeting in greetings:
        if greeting in first_bit:
            return 1
    return 0

def get_sentiment(text):
    """Compute sentiment polarity and subjectivity"""
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def persuasion_cues(text):
    """Count gain and loss persuasion phrases"""
    good_phrases = ["win", "prize", "bonus", "reward"]
    bad_phrases = ["lose", "suspended", "locked", "expired"]
    text = text.lower()
    good_count = 0
    bad_count = 0
    for phrase in good_phrases:
        good_count += len(re.findall(r'\b' + phrase + r'\b', text))
    for phrase in bad_phrases:
        bad_count += len(re.findall(r'\b' + phrase + r'\b', text))
    return good_count, bad_count

def get_lengths(subject, body):
    """Get subject and body lengths"""
    return len(subject), len(body.split())

def count_html_tags(text):
    """Count HTML tags in text"""
    soup = BeautifulSoup(text, 'html.parser')
    return len(soup.find_all())

def count_urls(text):
    """Count URLs in text"""
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    return len(re.findall(url_pattern, text))

def count_attachments(text):
    """Count attachment references"""
    return text.lower().count("content-disposition: attachment")

def count_exclamation(text):
    """Count exclamation marks"""
    return text.count("!")

def get_bert_embeddings(text, max_length=512):
    """Generate BERT embeddings for a single text"""
    global bert_tokenizer, bert_model
    
    # Tokenize and encode the text
    inputs = bert_tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=max_length, 
        padding=True
    )
    
    # Generate embeddings
    with torch.no_grad():
        outputs = bert_model(**inputs)
        # Use the [CLS] token embedding (first token) as the sentence representation
        embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
    
    return embedding

def extract_features(subject, body, from_email):
    """Extract features from email text using the exact same method as training"""
    
    # Combine subject and body for analysis
    combined_text = subject + ' ' + body
    
    features = {}
    
    # Extract the same engineered features as training (in the same order)
    features.update(count_keywords(combined_text))
    features['generic_greeting'] = check_greeting(body)
    polarity, subjectivity = get_sentiment(body)
    features['polarity'] = polarity
    features['subjectivity'] = subjectivity
    good_count, bad_count = persuasion_cues(body)
    features['good_phrases'] = good_count
    features['bad_phrases'] = bad_count
    sub_len, body_len = get_lengths(subject, body)
    features['subject_length'] = sub_len
    features['body_length'] = body_len
    features['html_tags'] = count_html_tags(body)
    features['url_count'] = count_urls(body)
    features['attachment_count'] = count_attachments(body)
    features['exclamation_count'] = count_exclamation(combined_text)
    
    # Create engineered features DataFrame (same as training)
    features_df = pd.DataFrame([features])
    
    # Generate BERT embeddings for the combined text
    bert_embedding = get_bert_embeddings(combined_text)
    
    # Create BERT DataFrame with same column names as training
    bert_columns = [f'bert_dim_{i}' for i in range(len(bert_embedding))]
    bert_df = pd.DataFrame([bert_embedding], columns=bert_columns)
    
    # Combine engineered features with BERT embeddings (same as training)
    final_df = pd.concat([features_df, bert_df], axis=1)
    
    print(f"Extracted {len(features_df.columns)} engineered features + {len(bert_df.columns)} BERT features = {len(final_df.columns)} total features")
    print(f"Feature columns: {list(final_df.columns)[:5]}... + bert_dim_0 to bert_dim_{len(bert_embedding)-1}")
    
    return final_df

def make_prediction(features_df):
    """Make prediction using the loaded model"""
    global model, scaler
    
    # Scale the features (DataFrame with column names)
    scaled_features = scaler.transform(features_df)
    
    # Get prediction and probability
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0]
    confidence = max(probability)
    
    return int(prediction), float(confidence)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None and scaler is not None,
        'bert_loaded': bert_model is not None and bert_tokenizer is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if an email is phishing"""
    try:
        # Get email data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        subject = data.get('subject', '')
        body = data.get('body', '')
        from_email = data.get('from', '')
        
        if not subject and not body:
            return jsonify({'error': 'Subject or body is required'}), 400
        
        # Extract features
        features = extract_features(subject, body, from_email)
        
        # Make prediction
        prediction, confidence = make_prediction(features)
        
        # Return result
        result = {
            'isPhishing': bool(prediction),
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Start the app
if __name__ == '__main__':
    print("SafeInbox Email Phishing Detection API")
    print("=====================================")
    
    if load_model():
        print("Model loaded successfully")
        print("Starting Flask server...")
        print("Available endpoints:")
        print("- POST /predict - Single email prediction")
        print("- GET /health - Health check")
        print("")
        print("Your Chrome extension should use: http://localhost:5001/predict")
        
        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        print("Failed to load model")
        print("Make sure optimized_phishing_model.pkl exists in this directory") 