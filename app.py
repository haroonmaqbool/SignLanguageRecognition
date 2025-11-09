"""
Sign Language Recognition - Web Application Module
================================================

Project: Sign Language Recognition System
Course: Introduction to Artificial Intelligence (COMP-360)
Institution: Forman Christian College
Team: Haroon, Saria, Azmeer
Instructor: [Instructor Name]

Description:
This module provides a Flask web application for sign language recognition.
Users can upload images or use webcam for real-time detection. The app
supports both CNN and LSTM models and provides an intuitive web interface
for testing the trained models.

Features:
- Image upload and prediction
- Real-time webcam detection
- Model selection (CNN/LSTM)
- Confidence score display
- Hand landmark visualization
- Responsive web interface
- Batch processing support

Requirements:
- Flask, OpenCV, MediaPipe, NumPy
- TensorFlow/Keras for model loading
- Trained models from train_model.py

Author: AI Coding Assistant
Date: 2024
"""

# Step 1 - Import Required Libraries
from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os
import base64
import io
from PIL import Image
import time
import json

# Step 2 - Initialize Flask Application
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Step 3 - Global Variables
class SignLanguageApp:
    """
    Main application class for sign language recognition.
    """
    
    def __init__(self):
        self.models = {}
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.current_model = None
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """
        Load trained CNN and LSTM models.
        """
        print("📥 Loading trained models...")
        
        model_files = {
            'CNN': 'models/cnn_final.h5',
            'LSTM': 'models/lstm_final.h5'
        }
        
        for model_name, model_path in model_files.items():
            try:
                if os.path.exists(model_path):
                    self.models[model_name] = load_model(model_path)
                    print(f"✅ {model_name} model loaded successfully!")
                else:
                    print(f"⚠️  {model_name} model not found at {model_path}")
            except Exception as e:
                print(f"❌ Error loading {model_name} model: {e}")
        
        # Set default model
        if self.models:
            self.current_model = list(self.models.keys())[0]
            print(f"🎯 Default model set to: {self.current_model}")
    
    def extract_landmarks(self, image):
        """
        Extract hand landmarks from an image.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            numpy.ndarray or None: Extracted landmarks or None if no hand detected
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            # Get the first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks, dtype=np.float32)
        
        return None
    
    def predict_gesture(self, landmarks, model_name=None):
        """
        Predict sign language gesture from landmarks.
        
        Args:
            landmarks: Hand landmarks array
            model_name: Name of model to use (optional)
            
        Returns:
            dict: Prediction results
        """
        if landmarks is None:
            return {
                'prediction': None,
                'confidence': 0.0,
                'error': 'No hand detected in image'
            }
        
        model_to_use = model_name or self.current_model
        
        if model_to_use not in self.models:
            return {
                'prediction': None,
                'confidence': 0.0,
                'error': f'Model {model_to_use} not available'
            }
        
        try:
            # Reshape landmarks for model input
            landmarks_reshaped = landmarks.reshape(1, -1)
            
            # Make prediction
            prediction = self.models[model_to_use].predict(landmarks_reshaped, verbose=0)
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            
            predicted_letter = self.alphabet[predicted_class_idx]
            
            # Get top 3 predictions
            top_indices = np.argsort(prediction[0])[-3:][::-1]
            top_predictions = []
            for idx in top_indices:
                top_predictions.append({
                    'letter': self.alphabet[idx],
                    'confidence': float(prediction[0][idx])
                })
            
            return {
                'prediction': predicted_letter,
                'confidence': confidence,
                'top_predictions': top_predictions,
                'model_used': model_to_use,
                'error': None
            }
            
        except Exception as e:
            return {
                'prediction': None,
                'confidence': 0.0,
                'error': f'Prediction error: {str(e)}'
            }
    
    def draw_landmarks_on_image(self, image, landmarks):
        """
        Draw hand landmarks on the image.
        
        Args:
            image: Input image
            landmarks: Hand landmarks
            
        Returns:
            numpy.ndarray: Image with landmarks drawn
        """
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            # Draw landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return image

# Initialize application
sign_lang_app = SignLanguageApp()

# Step 4 - Flask Routes
@app.route('/')
def index():
    """
    Main page route.
    """
    return render_template('index.html', 
                         available_models=list(sign_lang_app.models.keys()),
                         current_model=sign_lang_app.current_model)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict sign language from uploaded image.
    """
    try:
        # Get uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get model selection
        model_name = request.form.get('model', sign_lang_app.current_model)
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Resize image if too large
        height, width = image.shape[:2]
        if height > 1000 or width > 1000:
            scale = min(1000/height, 1000/width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Extract landmarks
        landmarks = sign_lang_app.extract_landmarks(image)
        
        # Make prediction
        result = sign_lang_app.predict_gesture(landmarks, model_name)
        
        # Draw landmarks on image if requested
        draw_landmarks = request.form.get('draw_landmarks', 'false').lower() == 'true'
        if draw_landmarks and landmarks is not None:
            image = sign_lang_app.draw_landmarks_on_image(image, landmarks)
        
        # Convert image to base64 for display
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        result['image_with_landmarks'] = image_base64
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/set_model', methods=['POST'])
def set_model():
    """
    Set the current model for predictions.
    """
    try:
        data = request.get_json()
        model_name = data.get('model')
        
        if model_name not in sign_lang_app.models:
            return jsonify({'error': f'Model {model_name} not available'}), 400
        
        sign_lang_app.current_model = model_name
        return jsonify({'message': f'Model set to {model_name}', 'current_model': model_name})
        
    except Exception as e:
        return jsonify({'error': f'Error setting model: {str(e)}'}), 500

@app.route('/models')
def get_models():
    """
    Get available models.
    """
    return jsonify({
        'available_models': list(sign_lang_app.models.keys()),
        'current_model': sign_lang_app.current_model
    })

@app.route('/health')
def health_check():
    """
    Health check endpoint.
    """
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(sign_lang_app.models),
        'available_models': list(sign_lang_app.models.keys())
    })

# Step 5 - Error Handlers
@app.errorhandler(413)
def too_large(e):
    """
    Handle file too large error.
    """
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """
    Handle 404 errors.
    """
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """
    Handle 500 errors.
    """
    return jsonify({'error': 'Internal server error'}), 500

# Step 6 - Main Application
def create_templates():
    """
    Create HTML templates for the web application.
    """
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Main template
    #html_content = # Find this function in your app.py (around line 340):
# Find the create_templates() function in your app.py (around line 340)
# Replace the entire html_content variable with this COMPLETE code:

def create_templates():
    """
    Create HTML templates for the web application.
    """
    import os
    os.makedirs('templates', exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Language Recognition</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0a0e27;
            color: #e4e4e7;
            min-height: 100vh;
            overflow-x: hidden;
        }

        .bg-animation {
            position: fixed;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            z-index: -1;
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        }

        .bg-animation::before {
            content: '';
            position: absolute;
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, transparent 70%);
            top: -250px;
            right: -250px;
            animation: float 20s infinite ease-in-out;
        }

        .bg-animation::after {
            content: '';
            position: absolute;
            width: 400px;
            height: 400px;
            background: radial-gradient(circle, rgba(168, 85, 247, 0.08) 0%, transparent 70%);
            bottom: -200px;
            left: -200px;
            animation: float 15s infinite ease-in-out reverse;
        }

        @keyframes float {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            33% { transform: translate(30px, -30px) rotate(120deg); }
            66% { transform: translate(-20px, 20px) rotate(240deg); }
        }

        .landing-page {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            text-align: center;
            padding: 40px 20px;
            position: relative;
        }

        .hand-icon {
            font-size: 8rem;
            margin-bottom: 30px;
            animation: wave 2s infinite ease-in-out;
            filter: drop-shadow(0 0 30px rgba(99, 102, 241, 0.5));
        }

        @keyframes wave {
            0%, 100% { transform: rotate(0deg); }
            25% { transform: rotate(20deg); }
            75% { transform: rotate(-20deg); }
        }

        .landing-page h1 {
            font-size: 4rem;
            font-weight: 800;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: fadeInDown 1s ease-out;
        }

        .landing-page .subtitle {
            font-size: 1.4rem;
            margin-bottom: 20px;
            color: #a1a1aa;
            max-width: 700px;
            animation: fadeInUp 1s ease-out 0.2s backwards;
        }

        .landing-page .team-info {
            font-size: 0.95rem;
            color: #71717a;
            margin-bottom: 50px;
            animation: fadeInUp 1s ease-out 0.4s backwards;
        }

        .feature-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            animation: fadeInUp 1s ease-out 0.6s backwards;
        }

        .feature-card {
            background: rgba(39, 39, 42, 0.5);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }

        .feature-card:hover {
            transform: translateY(-5px);
            border-color: rgba(99, 102, 241, 0.5);
            box-shadow: 0 10px 40px rgba(99, 102, 241, 0.2);
        }

        .feature-card .icon {
            font-size: 2.5rem;
            margin-bottom: 15px;
        }

        .feature-card h3 {
            font-size: 1.1rem;
            margin-bottom: 10px;
            color: #e4e4e7;
        }

        .feature-card p {
            font-size: 0.9rem;
            color: #a1a1aa;
            line-height: 1.5;
        }

        .try-now-btn {
            padding: 20px 60px;
            font-size: 1.3rem;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            color: white;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            font-weight: 700;
            transition: all 0.3s ease;
            box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
            animation: fadeInUp 1s ease-out 0.8s backwards;
            position: relative;
            overflow: hidden;
        }

        .try-now-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }

        .try-now-btn:hover::before {
            left: 100%;
        }

        .try-now-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(99, 102, 241, 0.5);
        }

        .app-page {
            display: none;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
            animation: fadeIn 0.5s ease-out;
        }

        .app-page.active {
            display: block;
        }

        .top-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(39, 39, 42, 0.5);
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(99, 102, 241, 0.2);
        }

        .back-btn {
            padding: 12px 30px;
            background: rgba(99, 102, 241, 0.1);
            color: #6366f1;
            border: 1px solid #6366f1;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .back-btn:hover {
            background: #6366f1;
            color: white;
            transform: translateX(-5px);
        }

        .header h2 {
            font-size: 1.8rem;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .main-content {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 30px;
        }

        .camera-panel {
            background: rgba(39, 39, 42, 0.5);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(99, 102, 241, 0.2);
            backdrop-filter: blur(10px);
        }

        .camera-container {
            position: relative;
            border-radius: 15px;
            overflow: hidden;
            background: #18181b;
            margin-bottom: 25px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        }

        #video {
            width: 100%;
            height: auto;
            display: block;
            min-height: 400px;
        }

        .camera-overlay {
            position: absolute;
            top: 15px;
            right: 15px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 0.9rem;
            font-weight: 600;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .model-selector {
            margin: 25px 0;
            text-align: center;
        }

        .model-selector label {
            color: #a1a1aa;
            font-weight: 600;
            margin-right: 15px;
        }

        .model-selector select {
            padding: 12px 25px;
            background: rgba(39, 39, 42, 0.8);
            color: #e4e4e7;
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 25px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .model-selector select:hover {
            border-color: #6366f1;
        }

        .controls {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .control-btn {
            padding: 14px 35px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .control-btn::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.2);
            transform: translate(-50%, -50%);
            transition: width 0.5s, height 0.5s;
        }

        .control-btn:active::before {
            width: 300px;
            height: 300px;
        }

        .start-btn {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            box-shadow: 0 5px 20px rgba(16, 185, 129, 0.3);
        }

        .start-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
        }

        .stop-btn {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            box-shadow: 0 5px 20px rgba(239, 68, 68, 0.3);
        }

        .stop-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(239, 68, 68, 0.4);
        }

        .clear-btn {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white;
            box-shadow: 0 5px 20px rgba(245, 158, 11, 0.3);
        }

        .clear-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(245, 158, 11, 0.4);
        }

        .output-panel {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .prediction-box {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(168, 85, 247, 0.15) 100%);
            border: 1px solid rgba(99, 102, 241, 0.3);
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }

        .prediction-box::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200px;
            height: 200px;
            background: radial-gradient(circle, rgba(99, 102, 241, 0.1) 0%, transparent 70%);
            animation: rotate 10s linear infinite;
        }

        @keyframes rotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .prediction-box h3 {
            font-size: 1rem;
            color: #a1a1aa;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .current-letter {
            font-size: 5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 20px 0;
            position: relative;
            z-index: 1;
            animation: pulse 2s infinite ease-in-out;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }

        .confidence {
            font-size: 1.1rem;
            color: #71717a;
            font-weight: 600;
        }

        .confidence-bar {
            width: 100%;
            height: 8px;
            background: rgba(39, 39, 42, 0.5);
            border-radius: 10px;
            margin-top: 15px;
            overflow: hidden;
            position: relative;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
            border-radius: 10px;
            transition: width 0.5s ease;
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
        }

        .sentence-box {
            background: rgba(39, 39, 42, 0.5);
            border: 1px solid rgba(99, 102, 241, 0.2);
            padding: 30px;
            border-radius: 20px;
            backdrop-filter: blur(10px);
            min-height: 200px;
            flex: 1;
        }

        .sentence-box h3 {
            font-size: 1rem;
            color: #a1a1aa;
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 2px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .sentence-box h3::before {
            content: '✍️';
            font-size: 1.5rem;
        }

        .sentence-display {
            font-size: 1.8rem;
            color: #e4e4e7;
            line-height: 1.8;
            word-wrap: break-word;
            min-height: 100px;
            font-weight: 500;
            letter-spacing: 0.5px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 20px;
        }

        .stat-card {
            background: rgba(39, 39, 42, 0.5);
            border: 1px solid rgba(99, 102, 241, 0.2);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }

        .stat-card .value {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .stat-card .label {
            font-size: 0.9rem;
            color: #71717a;
            margin-top: 5px;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 1200px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 768px) {
            .landing-page h1 {
                font-size: 2.5rem;
            }
            .hand-icon {
                font-size: 5rem;
            }
            .feature-cards {
                grid-template-columns: 1fr;
            }
            .current-letter {
                font-size: 3.5rem;
            }
            .sentence-display {
                font-size: 1.4rem;
            }
            .top-bar {
                flex-direction: column;
                gap: 15px;
            }
            .stats-grid {
                grid-template-columns: 1fr;
            }
        }

        ::-webkit-scrollbar {
            width: 10px;
        }
        ::-webkit-scrollbar-track {
            background: #18181b;
        }
        ::-webkit-scrollbar-thumb {
            background: #6366f1;
            border-radius: 5px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #a855f7;
        }
    </style>
</head>
<body>
    <div class="bg-animation"></div>
    <div id="landingPage" class="landing-page">
        <div class="hand-icon">🤟</div>
        <h1>Sign Language Recognition</h1>
        <p class="subtitle">Transform hand gestures into text in real-time using AI-powered deep learning technology</p>
        <p class="team-info">Team: Haroon, Saria, Azmeer | COMP-360 | Forman Christian College</p>
        <div class="feature-cards">
            <div class="feature-card">
                <div class="icon">🎥</div>
                <h3>Real-Time Detection</h3>
                <p>Instant gesture recognition using your webcam</p>
            </div>
            <div class="feature-card">
                <div class="icon">🧠</div>
                <h3>AI Powered</h3>
                <p>Advanced CNN & LSTM neural networks</p>
            </div>
            <div class="feature-card">
                <div class="icon">✍️</div>
                <h3>Text Generation</h3>
                <p>Automatic sentence building from gestures</p>
            </div>
            <div class="feature-card">
                <div class="icon">⚡</div>
                <h3>High Accuracy</h3>
                <p>Trained on thousands of sign language samples</p>
            </div>
        </div>
        <button class="try-now-btn" onclick="showApp()">Try Now →</button>
    </div>
    <div id="appPage" class="app-page">
        <div class="top-bar">
            <button class="back-btn" onclick="showLanding()">← Back</button>
            <div class="header"><h2>Real-Time Detection Studio</h2></div>
            <div></div>
        </div>
        <div class="main-content">
            <div class="camera-panel">
                <div class="camera-container">
                    <video id="video" autoplay></video>
                    <div class="camera-overlay" id="cameraStatus">📷 Camera Off</div>
                </div>
                <div class="model-selector">
                    <label for="modelSelect">AI Model:</label>
                    <select id="modelSelect">
                        {% for model in available_models %}
                        <option value="{{ model }}" {% if model == current_model %}selected{% endif %}>{{ model }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="controls">
                    <button class="control-btn start-btn" onclick="startCamera()">▶ Start</button>
                    <button class="control-btn stop-btn" onclick="stopCamera()">⏹ Stop</button>
                    <button class="control-btn clear-btn" onclick="clearSentence()">🗑 Clear</button>
                </div>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="value" id="totalLetters">0</div>
                        <div class="label">Letters Detected</div>
                    </div>
                    <div class="stat-card">
                        <div class="value" id="wordsCount">0</div>
                        <div class="label">Words Formed</div>
                    </div>
                </div>
            </div>
            <div class="output-panel">
                <div class="prediction-box">
                    <h3>Current Gesture</h3>
                    <div class="current-letter" id="currentLetter">-</div>
                    <div class="confidence" id="confidence">Confidence: 0%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" id="confidenceFill" style="width: 0%"></div>
                    </div>
                </div>
                <div class="sentence-box">
                    <h3>Generated Text</h3>
                    <div class="sentence-display" id="sentenceDisplay">Start making gestures to build your message...</div>
                </div>
            </div>
        </div>
    </div>
    <script>
        let video = document.getElementById('video');
        let currentSentence = '';
        let stream = null;
        let predictionInterval = null;
        let lastPrediction = '';
        let predictionCount = 0;
        let totalLettersDetected = 0;

        function showApp() {
            document.getElementById('landingPage').style.display = 'none';
            document.getElementById('appPage').classList.add('active');
        }

        function showLanding() {
            document.getElementById('appPage').classList.remove('active');
            document.getElementById('landingPage').style.display = 'flex';
            stopCamera();
        }

        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                video.srcObject = stream;
                document.getElementById('cameraStatus').textContent = '🔴 Live';
                document.getElementById('cameraStatus').style.background = 'rgba(239, 68, 68, 0.8)';
                startPredictionLoop();
            } catch (err) {
                alert('Error accessing camera: ' + err.message);
            }
        }

        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                video.srcObject = null;
                stream = null;
                document.getElementById('cameraStatus').textContent = '📷 Camera Off';
                document.getElementById('cameraStatus').style.background = 'rgba(0, 0, 0, 0.8)';
            }
            if (predictionInterval) {
                clearInterval(predictionInterval);
                predictionInterval = null;
            }
        }

        function clearSentence() {
            currentSentence = '';
            lastPrediction = '';
            predictionCount = 0;
            totalLettersDetected = 0;
            document.getElementById('sentenceDisplay').textContent = 'Start making gestures to build your message...';
            document.getElementById('totalLetters').textContent = '0';
            document.getElementById('wordsCount').textContent = '0';
        }

        function updateStats() {
            document.getElementById('totalLetters').textContent = totalLettersDetected;
            const words = currentSentence.trim().split(/\\s+/).filter(w => w.length > 0).length;
            document.getElementById('wordsCount').textContent = words;
        }

        function startPredictionLoop() {
            if (predictionInterval) clearInterval(predictionInterval);
            predictionInterval = setInterval(async () => {
                if (!stream) return;
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0);
                canvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('image', blob, 'frame.jpg');
                    formData.append('model', document.getElementById('modelSelect').value);
                    formData.append('draw_landmarks', 'false');
                    try {
                        const response = await fetch('/predict', { method: 'POST', body: formData });
                        const data = await response.json();
                        if (data.prediction) {
                            document.getElementById('currentLetter').textContent = data.prediction;
                            const confidencePercent = (data.confidence * 100).toFixed(1);
                            document.getElementById('confidence').textContent = `Confidence: ${confidencePercent}%`;
                            document.getElementById('confidenceFill').style.width = confidencePercent + '%';
                            if (data.confidence > 0.7) {
                                if (data.prediction === lastPrediction) {
                                    predictionCount++;
                                    if (predictionCount >= 3) {
                                        currentSentence += data.prediction;
                                        totalLettersDetected++;
                                        document.getElementById('sentenceDisplay').textContent = currentSentence || 'Start making gestures...';
                                        updateStats();
                                        predictionCount = 0;
                                        lastPrediction = '';
                                    }
                                } else {
                                    lastPrediction = data.prediction;
                                    predictionCount = 1;
                                }
                            }
                        } else if (data.error) {
                            document.getElementById('currentLetter').textContent = '?';
                            document.getElementById('confidence').textContent = data.error;
                            document.getElementById('confidenceFill').style.width = '0%';
                        }
                    } catch (err) {
                        console.error('Prediction error:', err);
                    }
                }, 'image/jpeg');
            }, 1000);
        }

        document.getElementById('modelSelect').addEventListener('change', (e) => {
            fetch('/set_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: e.target.value })
            }).then(response => response.json())
              .then(data => console.log('Model changed to:', data.current_model))
              .catch(err => console.error('Error changing model:', err));
        });
    </script>
</body>
</html>'''
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ HTML template created successfully!")

def main():
    """
    Main function to run the Flask application.
    """
    print("=" * 60)
    print("Sign Language Recognition - Web Application")
    print("=" * 60)
    print("Team: Haroon, Saria, Azmeer")
    print("Course: COMP-360 - Introduction to Artificial Intelligence")
    print("Institution: Forman Christian College")
    print("=" * 60)
    
    # Create templates
    create_templates()
    
    # Check if models are loaded
    if not sign_lang_app.models:
        print("❌ No trained models found!")
        print("Please run train_model.py first to train the models.")
        return
    
    print(f"✅ Web application initialized successfully!")
    print(f"📊 Available models: {list(sign_lang_app.models.keys())}")
    print(f"🎯 Current model: {sign_lang_app.current_model}")
    print(f"\n🌐 Starting Flask web server...")
    print(f"📱 Open your browser and go to: http://localhost:5000")
    print(f"🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    """
    Execute the Flask application when script is run directly.
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Web application stopped by user.")
        print("Exiting gracefully...")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your setup and try again.")

