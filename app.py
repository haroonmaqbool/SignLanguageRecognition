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
    html_content = # Find this function in your app.py (around line 340):
def create_templates():
    """
    Create HTML templates for the web application.
    """
    # Create templates directory
    os.makedirs('templates', exist_ok=True)
    
    # Main template - REPLACE THE html_content WITH THIS:
    html_content = '''
<!DOCTYPE html>
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
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }

        /* Landing Page Styles */
        .landing-page {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            text-align: center;
            padding: 20px;
            color: white;
        }

        .landing-page h1 {
            font-size: 3.5rem;
            margin-bottom: 20px;
            animation: fadeInDown 1s ease-out;
        }

        .landing-page p {
            font-size: 1.3rem;
            margin-bottom: 40px;
            max-width: 600px;
            animation: fadeInUp 1s ease-out;
        }

        .try-now-btn {
            padding: 18px 50px;
            font-size: 1.2rem;
            background: white;
            color: #667eea;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
            animation: pulse 2s infinite;
        }

        .try-now-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }

        /* App Page Styles */
        .app-page {
            display: none;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }

        .app-page.active {
            display: block;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }

        .header h2 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }

        .back-btn {
            padding: 10px 25px;
            background: rgba(255,255,255,0.2);
            color: white;
            border: 2px solid white;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }

        .back-btn:hover {
            background: white;
            color: #667eea;
        }

        .content-container {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }

        .camera-section {
            text-align: center;
            margin-bottom: 30px;
        }

        .camera-container {
            position: relative;
            max-width: 640px;
            margin: 0 auto 20px;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            background: #000;
        }

        #video {
            width: 100%;
            height: auto;
            display: block;
        }

        .camera-overlay {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
        }

        .model-selector {
            margin: 20px 0;
            text-align: center;
        }

        .model-selector select {
            padding: 10px 20px;
            border: 2px solid #667eea;
            border-radius: 25px;
            font-size: 1rem;
            margin: 0 10px;
            cursor: pointer;
        }

        .controls {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .control-btn {
            padding: 12px 30px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .start-btn {
            background: #667eea;
            color: white;
        }

        .start-btn:hover {
            background: #5568d3;
            transform: translateY(-2px);
        }

        .stop-btn {
            background: #ef4444;
            color: white;
        }

        .stop-btn:hover {
            background: #dc2626;
            transform: translateY(-2px);
        }

        .clear-btn {
            background: #f59e0b;
            color: white;
        }

        .clear-btn:hover {
            background: #d97706;
            transform: translateY(-2px);
        }

        .output-section {
            margin-top: 30px;
        }

        .prediction-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
            text-align: center;
        }

        .prediction-box h3 {
            font-size: 1.2rem;
            margin-bottom: 10px;
            opacity: 0.9;
        }

        .current-letter {
            font-size: 4rem;
            font-weight: bold;
            margin: 10px 0;
        }

        .confidence {
            font-size: 1rem;
            opacity: 0.8;
        }

        .sentence-box {
            background: #f3f4f6;
            padding: 25px;
            border-radius: 15px;
            min-height: 100px;
        }

        .sentence-box h3 {
            font-size: 1.2rem;
            color: #667eea;
            margin-bottom: 15px;
        }

        .sentence-display {
            font-size: 1.8rem;
            color: #1f2937;
            line-height: 1.6;
            word-wrap: break-word;
            min-height: 50px;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }

        /* Animations */
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

        @keyframes pulse {
            0%, 100% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.05);
            }
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Responsive */
        @media (max-width: 768px) {
            .landing-page h1 {
                font-size: 2.5rem;
            }

            .landing-page p {
                font-size: 1.1rem;
            }

            .header h2 {
                font-size: 2rem;
            }

            .current-letter {
                font-size: 3rem;
            }

            .sentence-display {
                font-size: 1.4rem;
            }
        }
    </style>
</head>
<body>
    <!-- Landing Page -->
    <div id="landingPage" class="landing-page">
        <h1>🤟 Sign Language Recognition</h1>
        <p>Transform hand gestures into text in real-time using AI-powered recognition technology</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Team: Haroon, Saria, Azmeer | COMP-360 | Forman Christian College</p>
        <button class="try-now-btn" onclick="showApp()">Try Now</button>
    </div>

    <!-- App Page -->
    <div id="appPage" class="app-page">
        <button class="back-btn" onclick="showLanding()">← Back to Home</button>
        
        <div class="header">
            <h2>Real-Time Sign Language Detection</h2>
        </div>

        <div class="content-container">
            <!-- Camera Section -->
            <div class="camera-section">
                <div class="camera-container">
                    <video id="video" autoplay></video>
                    <div class="camera-overlay" id="cameraStatus">📷 Camera Off</div>
                </div>

                <!-- Model Selector -->
                <div class="model-selector">
                    <label for="modelSelect"><strong>Select Model:</strong></label>
                    <select id="modelSelect">
                        {% for model in available_models %}
                        <option value="{{ model }}" {% if model == current_model %}selected{% endif %}>{{ model }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="controls">
                    <button class="control-btn start-btn" onclick="startCamera()">Start Camera</button>
                    <button class="control-btn stop-btn" onclick="stopCamera()">Stop Camera</button>
                    <button class="control-btn clear-btn" onclick="clearSentence()">Clear Text</button>
                </div>
            </div>

            <!-- Loading -->
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing gesture...</p>
            </div>

            <!-- Output Section -->
            <div class="output-section">
                <!-- Current Prediction -->
                <div class="prediction-box">
                    <h3>Current Gesture</h3>
                    <div class="current-letter" id="currentLetter">-</div>
                    <div class="confidence" id="confidence">Confidence: 0%</div>
                </div>

                <!-- Sentence Output -->
                <div class="sentence-box">
                    <h3>Generated Text</h3>
                    <div class="sentence-display" id="sentenceDisplay">Start making gestures...</div>
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

        // Show app page
        function showApp() {
            document.getElementById('landingPage').style.display = 'none';
            document.getElementById('appPage').classList.add('active');
        }

        // Show landing page
        function showLanding() {
            document.getElementById('appPage').classList.remove('active');
            document.getElementById('landingPage').style.display = 'flex';
            stopCamera();
        }

        // Start camera
        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480 } 
                });
                video.srcObject = stream;
                document.getElementById('cameraStatus').textContent = '🔴 Live';
                
                // Start prediction loop
                startPredictionLoop();
            } catch (err) {
                alert('Error accessing camera: ' + err.message);
            }
        }

        // Stop camera
        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                video.srcObject = null;
                stream = null;
                document.getElementById('cameraStatus').textContent = '📷 Camera Off';
            }
            
            if (predictionInterval) {
                clearInterval(predictionInterval);
                predictionInterval = null;
            }
        }

        // Clear sentence
        function clearSentence() {
            currentSentence = '';
            lastPrediction = '';
            predictionCount = 0;
            document.getElementById('sentenceDisplay').textContent = 'Start making gestures...';
        }

        // Start prediction loop
        function startPredictionLoop() {
            if (predictionInterval) {
                clearInterval(predictionInterval);
            }

            predictionInterval = setInterval(async () => {
                if (!stream) return;

                // Capture frame from video
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0);
                
                // Convert to blob
                canvas.toBlob(async (blob) => {
                    // Send to backend for prediction
                    const formData = new FormData();
                    formData.append('image', blob, 'frame.jpg');
                    formData.append('model', document.getElementById('modelSelect').value);
                    formData.append('draw_landmarks', 'false');

                    try {
                        const response = await fetch('/predict', {
                            method: 'POST',
                            body: formData
                        });

                        const data = await response.json();
                        
                        if (data.prediction) {
                            // Update current letter
                            document.getElementById('currentLetter').textContent = data.prediction;
                            document.getElementById('confidence').textContent = 
                                `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
                            
                            // Add to sentence (with stability check)
                            if (data.confidence > 0.7) {
                                if (data.prediction === lastPrediction) {
                                    predictionCount++;
                                    // Add letter if seen consistently (3 times)
                                    if (predictionCount >= 3) {
                                        currentSentence += data.prediction;
                                        document.getElementById('sentenceDisplay').textContent = currentSentence || 'Start making gestures...';
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
                        }
                    } catch (err) {
                        console.error('Prediction error:', err);
                    }
                }, 'image/jpeg');
                
            }, 1000); // Send frame every 1 second
        }

        // Model selection change
        document.getElementById('modelSelect').addEventListener('change', (e) => {
            fetch('/set_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model: e.target.value })
            }).then(response => response.json())
              .then(data => console.log('Model changed to:', data.current_model))
              .catch(err => console.error('Error changing model:', err));
        });
    </script>
</body>
</html>
    '''
    
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

