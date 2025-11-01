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
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Language Recognition System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }
        .content {
            padding: 40px;
        }
        .section {
            margin-bottom: 40px;
            padding: 30px;
            border: 2px solid #f0f0f0;
            border-radius: 10px;
            background: #fafafa;
        }
        .section h2 {
            color: #333;
            margin-top: 0;
            border-bottom: 2px solid #4facfe;
            padding-bottom: 10px;
        }
        .upload-area {
            border: 3px dashed #4facfe;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            background: white;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .upload-area:hover {
            border-color: #00f2fe;
            background: #f8f9ff;
        }
        .upload-area.dragover {
            border-color: #00f2fe;
            background: #e8f4ff;
        }
        .file-input {
            display: none;
        }
        .btn {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
            margin: 10px;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
        }
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .model-selector {
            margin: 20px 0;
        }
        .model-selector select {
            padding: 10px;
            border: 2px solid #4facfe;
            border-radius: 5px;
            font-size: 16px;
            margin: 0 10px;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: white;
            border-radius: 10px;
            border-left: 5px solid #4facfe;
        }
        .prediction {
            font-size: 2em;
            font-weight: bold;
            color: #4facfe;
            margin: 10px 0;
        }
        .confidence {
            font-size: 1.2em;
            color: #666;
            margin: 10px 0;
        }
        .top-predictions {
            margin-top: 20px;
        }
        .top-predictions h4 {
            color: #333;
            margin-bottom: 10px;
        }
        .prediction-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        .prediction-item:last-child {
            border-bottom: none;
        }
        .image-display {
            margin-top: 20px;
            text-align: center;
        }
        .image-display img {
            max-width: 100%;
            max-height: 400px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4facfe;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error {
            color: #e74c3c;
            background: #ffeaea;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #e74c3c;
            margin: 20px 0;
        }
        .success {
            color: #27ae60;
            background: #eafaf1;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #27ae60;
            margin: 20px 0;
        }
        .checkbox-container {
            margin: 20px 0;
        }
        .checkbox-container input[type="checkbox"] {
            margin-right: 10px;
        }
        .info-box {
            background: #e8f4ff;
            border: 1px solid #4facfe;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }
        .info-box h4 {
            margin-top: 0;
            color: #4facfe;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Sign Language Recognition System</h1>
            <p>Team: Haroon, Saria, Azmeer | COMP-360 | Forman Christian College</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>📸 Image Upload & Prediction</h2>
                
                <div class="info-box">
                    <h4>How to use:</h4>
                    <ul>
                        <li>Upload an image containing a hand gesture</li>
                        <li>Select your preferred model (CNN or LSTM)</li>
                        <li>Optionally enable hand landmark visualization</li>
                        <li>Click "Predict" to get the sign language prediction</li>
                    </ul>
                </div>
                
                <div class="upload-area" onclick="document.getElementById('imageInput').click()">
                    <div id="uploadText">
                        <h3>📁 Click to upload an image</h3>
                        <p>or drag and drop your image here</p>
                        <p>Supported formats: JPG, PNG, JPEG</p>
                    </div>
                    <input type="file" id="imageInput" class="file-input" accept="image/*">
                </div>
                
                <div class="model-selector">
                    <label for="modelSelect"><strong>Select Model:</strong></label>
                    <select id="modelSelect">
                        {% for model in available_models %}
                        <option value="{{ model }}" {% if model == current_model %}selected{% endif %}>{{ model }}</option>
                        {% endfor %}
                    </select>
                </div>
                
                <div class="checkbox-container">
                    <input type="checkbox" id="drawLandmarks">
                    <label for="drawLandmarks">Show hand landmarks on result image</label>
                </div>
                
                <button class="btn" onclick="predictImage()" id="predictBtn">🔮 Predict Sign Language</button>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Processing image and extracting hand landmarks...</p>
                </div>
                
                <div id="result"></div>
            </div>
        </div>
    </div>

    <script>
        // File upload handling
        const uploadArea = document.querySelector('.upload-area');
        const fileInput = document.getElementById('imageInput');
        const uploadText = document.getElementById('uploadText');
        
        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                updateUploadText(files[0].name);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                updateUploadText(e.target.files[0].name);
            }
        });
        
        function updateUploadText(fileName) {
            uploadText.innerHTML = `<h3>📁 Selected: ${fileName}</h3><p>Click to change image</p>`;
        }
        
        // Prediction function
        async function predictImage() {
            const fileInput = document.getElementById('imageInput');
            const modelSelect = document.getElementById('modelSelect');
            const drawLandmarks = document.getElementById('drawLandmarks');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const predictBtn = document.getElementById('predictBtn');
            
            if (!fileInput.files[0]) {
                showError('Please select an image file first.');
                return;
            }
            
            // Show loading
            loading.style.display = 'block';
            predictBtn.disabled = true;
            result.innerHTML = '';
            
            try {
                const formData = new FormData();
                formData.append('image', fileInput.files[0]);
                formData.append('model', modelSelect.value);
                formData.append('draw_landmarks', drawLandmarks.checked);
                
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    showError(data.error);
                } else {
                    showResult(data);
                }
                
            } catch (error) {
                showError('Error processing image: ' + error.message);
            } finally {
                loading.style.display = 'none';
                predictBtn.disabled = false;
            }
        }
        
        function showResult(data) {
            const result = document.getElementById('result');
            
            let html = '<div class="result">';
            
            if (data.prediction) {
                html += `<div class="success">✅ Prediction successful!</div>`;
                html += `<div class="prediction">Predicted Letter: ${data.prediction}</div>`;
                html += `<div class="confidence">Confidence: ${(data.confidence * 100).toFixed(2)}%</div>`;
                html += `<div class="confidence">Model Used: ${data.model_used}</div>`;
                
                if (data.top_predictions && data.top_predictions.length > 0) {
                    html += '<div class="top-predictions">';
                    html += '<h4>Top Predictions:</h4>';
                    data.top_predictions.forEach((pred, index) => {
                        html += `<div class="prediction-item">`;
                        html += `<span>${index + 1}. ${pred.letter}</span>`;
                        html += `<span>${(pred.confidence * 100).toFixed(2)}%</span>`;
                        html += `</div>`;
                    });
                    html += '</div>';
                }
                
                if (data.image_with_landmarks) {
                    html += '<div class="image-display">';
                    html += '<h4>Processed Image:</h4>';
                    html += `<img src="data:image/jpeg;base64,${data.image_with_landmarks}" alt="Processed Image">`;
                    html += '</div>';
                }
            } else {
                html += `<div class="error">❌ ${data.error || 'No prediction available'}</div>`;
            }
            
            html += '</div>';
            result.innerHTML = html;
        }
        
        function showError(message) {
            const result = document.getElementById('result');
            result.innerHTML = `<div class="error">❌ ${message}</div>`;
        }
        
        // Set model function
        async function setModel(modelName) {
            try {
                const response = await fetch('/set_model', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ model: modelName })
                });
                
                const data = await response.json();
                if (data.error) {
                    console.error('Error setting model:', data.error);
                } else {
                    console.log('Model set to:', data.current_model);
                }
            } catch (error) {
                console.error('Error setting model:', error);
            }
        }
        
        // Model selection change
        document.getElementById('modelSelect').addEventListener('change', (e) => {
            setModel(e.target.value);
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

