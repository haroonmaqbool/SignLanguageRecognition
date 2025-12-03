"""
======================================================
Sign Language Recognition - Camera Detection & Web Application
======================================================
This script implements both camera detection and Flask web application for sign language recognition.
Team: Haroon, Saria, Azmeer
Course: COMP-360 - Introduction to Artificial Intelligence
"""
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from pathlib import Path
import os
import base64
import tempfile
from gtts import gTTS

# Flask imports (optional - only needed for web app mode)
try:
    from flask import Flask, render_template, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️  Flask not available. Web app mode disabled. Install Flask to enable web app.")

# Constants
Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
Predictions = True # Set to True to enable live predictions
EnableSpaceGesture = True  # Set to True to enable space gesture detection (improved logic distinguishes B from space)

# Web app mode flag - set to True to run Flask web app, False for direct camera mode
WEB_APP_MODE = True  # Change to False to run direct camera mode

def is_space_gesture(landmarks):
    """
    Detect if hand gesture is a space gesture (open hand with ALL fingers extended).
    
    Space gesture characteristics:
    - ALL 4 fingers (index, middle, ring, pinky) extended
    - Thumb extended/pointing away from palm (not tucked)
    - Hand is open/flat (distinguishes from B which has thumb tucked against palm)
    
    Args:
        landmarks: Array of 63 values (21 landmarks × 3 coordinates)
    
    Returns:
        Boolean: True if gesture appears to be space
    """
    # Landmark indices (MediaPipe hand landmarks):
    # 0 = Wrist, 4 = Thumb tip, 8 = Index tip, 12 = Middle tip, 16 = Ring tip, 20 = Pinky tip
    # 3 = Thumb IP, 6 = Index PIP, 10 = Middle PIP, 14 = Ring PIP, 18 = Pinky PIP
    # 5 = Index MCP, 9 = Middle MCP, 13 = Ring MCP, 17 = Pinky MCP
    
    # Extract coordinates
    wrist_x = landmarks[0 * 3 + 0]
    wrist_y = landmarks[0 * 3 + 1]
    
    thumb_tip_x = landmarks[4 * 3 + 0]
    thumb_tip_y = landmarks[4 * 3 + 1]
    thumb_ip_x = landmarks[3 * 3 + 0]
    thumb_ip_y = landmarks[3 * 3 + 1]
    
    index_tip_y = landmarks[8 * 3 + 1]
    index_pip_y = landmarks[6 * 3 + 1]
    index_mcp_x = landmarks[5 * 3 + 0]
    index_mcp_y = landmarks[5 * 3 + 1]
    
    middle_tip_y = landmarks[12 * 3 + 1]
    middle_pip_y = landmarks[10 * 3 + 1]
    
    ring_tip_y = landmarks[16 * 3 + 1]
    ring_pip_y = landmarks[14 * 3 + 1]
    
    pinky_tip_y = landmarks[20 * 3 + 1]
    pinky_pip_y = landmarks[18 * 3 + 1]
    
    # Check if 4 fingers (index, middle, ring, pinky) are extended
    index_extended = index_tip_y < index_pip_y
    middle_extended = middle_tip_y < middle_pip_y
    ring_extended = ring_tip_y < ring_pip_y
    pinky_extended = pinky_tip_y < pinky_pip_y
    
    four_fingers_extended = index_extended and middle_extended and ring_extended and pinky_extended
    
    if not four_fingers_extended:
        return False  # Not space if 4 fingers aren't extended
    
    # Check if thumb is extended (not tucked)
    # Method 1: Thumb tip should be above thumb IP (extended upward)
    thumb_extended_up = thumb_tip_y < thumb_ip_y
    
    # Method 2: Thumb tip should be away from palm (distance from thumb tip to index MCP)
    # In B sign, thumb is tucked close to palm. In space, thumb is extended away.
    thumb_to_index_mcp_dist = np.sqrt(
        (thumb_tip_x - index_mcp_x)**2 + (thumb_tip_y - index_mcp_y)**2
    )
    # Normalized distance threshold (thumb extended if far from index MCP)
    thumb_away_from_palm = thumb_to_index_mcp_dist > 0.15  # Threshold for thumb being away
    
    # Method 3: Thumb tip x-position relative to wrist (for left hand, thumb extended = thumb tip to the left)
    # This is more reliable for detecting thumb extension
    thumb_to_left = thumb_tip_x < wrist_x  # Thumb extended to the left (away from palm)
    
    # Thumb is extended if it's extended upward AND away from palm
    thumb_extended = thumb_extended_up and (thumb_away_from_palm or thumb_to_left)
    
    # Space gesture: 4 fingers extended AND thumb extended (not tucked)
    return four_fingers_extended and thumb_extended

# Get script directory
Script_dir = Path(__file__).parent.absolute()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize model (only if Predictions is True)
model = None
models_dict = {}  # For web app mode - can store multiple models
if Predictions:
    try:
        model_path = Script_dir / "models" / "cnn_baseline.h5"
        if model_path.exists():
            print(f"Loading model from {model_path}")
            model = load_model(str(model_path))
            models_dict['CNN (Best)'] = model  # Store for web app with clearer name
            print("   Model loaded successfully!")
        else:
            print(f"  Model not found: {model_path}")
            print("   Please run train_model.py first to train the model.")
            print("   Running without predictions (hand tracking only).")
            Predictions = False
    except Exception as e:
        print(f"  Error loading model: {e}")
        print("   Running without predictions (hand tracking only).")
        Predictions = False

# Try loading additional models for web app
try:
    model_last_path = Script_dir / "models" / "cnn_last.h5"
    if model_last_path.exists():
        models_dict['CNN (Final)'] = load_model(str(model_last_path))
        print(f"   Additional model CNN (Final) loaded!")
except:
    pass

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.3,  # Lowered to 0.3 for better A sign detection
    min_tracking_confidence=0.3,   # Lowered to 0.3 for better tracking
    model_complexity=1  # Higher complexity for better accuracy
)

# ============================================================================
# DIRECT CAMERA MODE (Original skeleton logic - UNCHANGED)
# ============================================================================
def run_camera_mode():
    """Run the original camera detection mode - EXACTLY as skeleton."""
    # Initialize webcam
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("✗ Error: Could not open webcam.")
        return
    
    print("\n" + "=" * 60)
    print("Sign Language Recognition - Live Camera")
    print("=" * 60)
    print("Press 'q' to quit")
    if Predictions:
        print("Live prediction: ENABLED")
    else:
        print("Live prediction: DISABLED (set Predictions = True to enable)")
    print("=" * 60 + "\n")
    
    while webcam.isOpened():
        success, img = webcam.read()
        if not success:
            print("Error: Failed to read frame from webcam.")
            break
        
        # Convert BGR to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(img_rgb)
        
        # Convert back to BGR for OpenCV display
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
            
            # Live prediction (only if enabled and model is loaded)
            if Predictions and model is not None:
                # Extract landmarks from first hand
                first_hand = results.multi_hand_landmarks[0]
                
                # Check if it's a right hand using multi_handedness
                is_right_hand = False
                if results.multi_handedness:
                    handedness = results.multi_handedness[0]
                    if hasattr(handedness, 'classification') and len(handedness.classification) > 0:
                        is_right_hand = handedness.classification[0].label == 'Right'
                
                # Build feature vector (63 dimensions: 21 landmarks × 3 coordinates)
                landmarks_array = np.zeros(63)
                idx = 0
                for landmark in first_hand.landmark:
                    landmarks_array[idx] = landmark.x
                    landmarks_array[idx + 1] = landmark.y
                    landmarks_array[idx + 2] = landmark.z
                    idx += 3
                
                # Normalize to left-hand orientation (flip right hand x-coordinates)
                if is_right_hand:
                    for i in range(0, 63, 3):  # Every 3rd element is x-coordinate
                        landmarks_array[i] = 1.0 - landmarks_array[i]  # Flip x
                
                # Reshape for model input: (1, 63)
                landmarks_array = landmarks_array.reshape(1, 63)
                
                # Check if this is a space gesture BEFORE model prediction (only if enabled)
                if EnableSpaceGesture and is_space_gesture(landmarks_array[0]):
                    # Space gesture detected
                    text = "Predicted: SPACE (Space gesture detected)"
                    cv2.putText(
                        img,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),  # Yellow color for space
                        2,
                    )
                else:
                    # Make prediction for regular letters
                    preds = model.predict(landmarks_array, verbose=0)
                    predicted_class_idx = np.argmax(preds, axis=1)[0]
                    confidence = preds[0][predicted_class_idx]
                    predicted_letter = Alphabets[predicted_class_idx]
                    
                    # Overlay prediction on frame
                    text = f"Predicted: {predicted_letter} ({confidence:.2f})"
                    cv2.putText(
                        img,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),  # Green color for letters
                        2,
                    )
        else:
            # No hand detected
            if Predictions:
                cv2.putText(
                    img,
                    "No hand detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
        
        # Display frame
        cv2.imshow("Webcam - Sign Language Recognition", img)
        
        # Exit on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
    
    # Cleanup
    webcam.release()
    cv2.destroyAllWindows()
    print("\n Webcam released. Exiting...")

# ============================================================================
# WEB APP MODE (Flask routes and HTML template)
# ============================================================================
if FLASK_AVAILABLE and WEB_APP_MODE:
    # Initialize Flask Application
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    current_model_name = list(models_dict.keys())[0] if models_dict else None
    
    def extract_landmarks_from_image(image):
        """
        Extract landmarks from image - EXACTLY matching skeleton logic.
        """
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            # Extract landmarks from first hand
            first_hand = results.multi_hand_landmarks[0]
            
            # Check if it's a right hand using multi_handedness
            is_right_hand = False
            if results.multi_handedness:
                handedness = results.multi_handedness[0]
                if hasattr(handedness, 'classification') and len(handedness.classification) > 0:
                    is_right_hand = handedness.classification[0].label == 'Right'
            
            # Build feature vector (63 dimensions: 21 landmarks × 3 coordinates)
            landmarks_array = np.zeros(63)
            idx = 0
            for landmark in first_hand.landmark:
                landmarks_array[idx] = landmark.x
                landmarks_array[idx + 1] = landmark.y
                landmarks_array[idx + 2] = landmark.z
                idx += 3
            
            # Normalize to left-hand orientation (flip right hand x-coordinates)
            if is_right_hand:
                for i in range(0, 63, 3):  # Every 3rd element is x-coordinate
                    landmarks_array[i] = 1.0 - landmarks_array[i]  # Flip x
            
            return landmarks_array, first_hand
        
        return None, None
    
    def predict_gesture(landmarks_array, model_to_use=None):
        """
        Predict gesture from landmarks - EXACTLY matching skeleton logic.
        """
        if landmarks_array is None:
            return {
                'prediction': None,
                'confidence': 0.0,
                'error': 'No hand detected in image'
            }
        
        model_to_use = model_to_use or current_model_name
        if model_to_use not in models_dict:
            return {
                'prediction': None,
                'confidence': 0.0,
                'error': f'Model {model_to_use} not available'
            }
        
        try:
            # Reshape for model input: (1, 63)
            landmarks_array = landmarks_array.reshape(1, 63)
            
            # Check if this is a space gesture BEFORE model prediction (only if enabled)
            if EnableSpaceGesture and is_space_gesture(landmarks_array[0]):
                return {
                    'prediction': ' ',
                    'confidence': 0.95,
                    'top_predictions': [{'letter': ' ', 'confidence': 0.95}],
                    'model_used': model_to_use,
                    'error': None,
                    'is_space': True
                }
            
            # Make prediction for regular letters
            preds = models_dict[model_to_use].predict(landmarks_array, verbose=0)
            predicted_class_idx = np.argmax(preds, axis=1)[0]
            confidence = preds[0][predicted_class_idx]
            predicted_letter = Alphabets[predicted_class_idx]
            
            # Get top 3 predictions
            top_indices = np.argsort(preds[0])[-3:][::-1]
            top_predictions = []
            for idx in top_indices:
                top_predictions.append({
                    'letter': Alphabets[idx],
                    'confidence': float(preds[0][idx])
                })
            
            return {
                'prediction': predicted_letter,
                'confidence': float(confidence),
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
    
    def draw_landmarks_on_image(image, hand_landmarks):
        """Draw landmarks on image - matching skeleton style."""
        if hand_landmarks is None:
            return image
        
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )
        return annotated_image
    
    # Flask Routes
    @app.route('/')
    def index():
        """Main page route."""
        return render_template('index.html',
                             available_models=list(models_dict.keys()),
                             current_model=current_model_name)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Predict sign language from uploaded image."""
        try:
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
            
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No image file selected'}), 400
            
            # Get model selection
            model_name = request.form.get('model', current_model_name)
            
            # Read image
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({'error': 'Invalid image format'}), 400
            
            # Resize image if too large
            height, width = image.shape[:2]
            if height > 480 or width > 640:
                scale = min(640/width, 480/height)
                new_height = int(height * scale)
                new_width = int(width * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # Extract landmarks - EXACTLY matching skeleton logic
            landmarks_array, hand_landmarks_obj = extract_landmarks_from_image(image)
            
            # Make prediction - EXACTLY matching skeleton logic
            result = predict_gesture(landmarks_array, model_name)
            
            # Draw landmarks on image if requested
            draw_landmarks = request.form.get('draw_landmarks', 'false').lower() == 'true'
            if draw_landmarks and hand_landmarks_obj is not None:
                image = draw_landmarks_on_image(image, hand_landmarks_obj)
            
            # Convert image to base64 for display
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            _, buffer = cv2.imencode('.jpg', image, encode_param)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            result['image_with_landmarks'] = image_base64
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
    
    @app.route('/text-to-speech', methods=['POST'])
    def text_to_speech():
        """Convert text to speech and return audio file."""
        try:
            data = request.get_json()
            text = data.get('text', '').strip()
            
            if not text:
                return jsonify({'error': 'No text provided'}), 400
            
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_filename = fp.name
            
            # Generate speech using gTTS
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_filename)
            
            # Read the audio file and convert to base64
            with open(temp_filename, 'rb') as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Clean up temporary file
            os.unlink(temp_filename)
            
            return jsonify({
                'success': True,
                'audio': audio_base64,
                'text': text
            })
            
        except Exception as e:
            return jsonify({'error': f'Text-to-speech error: {str(e)}'}), 500
    
    @app.route('/set_model', methods=['POST'])
    def set_model():
        """Set the current model for predictions."""
        try:
            data = request.get_json()
            model_name = data.get('model')
            
            global current_model_name
            if model_name not in models_dict:
                return jsonify({'error': f'Model {model_name} not available'}), 400
            
            current_model_name = model_name
            return jsonify({'message': f'Model set to {model_name}', 'current_model': model_name})
            
        except Exception as e:
            return jsonify({'error': f'Error setting model: {str(e)}'}), 500
    
    @app.route('/models')
    def get_models():
        """Get available models."""
        return jsonify({
            'available_models': list(models_dict.keys()),
            'current_model': current_model_name
        })
    
    @app.route('/health')
    def health_check():
        """Health check endpoint."""
        return jsonify({
            'status': 'healthy',
            'models_loaded': len(models_dict),
            'available_models': list(models_dict.keys())
        })
    
    # Error Handlers
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Page not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({'error': 'Internal server error'}), 500
    
    # HTML Template Creation (full version from app.py)
    def create_templates():
        """Create HTML templates for the web application."""
        os.makedirs('templates', exist_ok=True)
        
        # Full HTML template matching app.py exactly
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Language Recognition</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
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
            background: linear-gradient(135deg, #0a1f1a 0%, #0d2818 50%, #061612 100%);
            overflow: hidden;
        }
        .landing-page {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            text-align: center;
            padding: 40px 20px;
        }
        .landing-page h1 {
            font-size: 4.5rem;
            font-weight: 900;
            margin-bottom: 25px;
            background: linear-gradient(135deg, #10b981 0%, #34d399 50%, #6ee7b7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .try-now-btn {
            padding: 22px 70px;
            font-size: 1.4rem;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            font-weight: 800;
            margin-top: 40px;
        }
        .app-page {
            display: none;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .app-page.active { display: block; }
        .camera-container {
            position: relative;
            border-radius: 15px;
            overflow: hidden;
            background: #18181b;
            margin-bottom: 25px;
        }
        #video, #displayCanvas {
            width: 100%;
            height: auto;
            display: block;
            min-height: 400px;
        }
        .prediction-box {
            background: rgba(39, 39, 42, 0.5);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 20px;
        }
        .current-letter {
            font-size: 5rem;
            font-weight: 800;
            color: #10b981;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="bg-animation"></div>
    <div id="landingPage" class="landing-page">
        <h1>Sign Language Recognition</h1>
        <p style="font-size: 1.5rem; margin-bottom: 20px; color: #9ca3af;">
            Transform hand gestures into text in real-time using AI-powered deep learning
        </p>
        <p style="font-size: 1rem; color: #6b7280; margin-bottom: 40px;">
            Team: Haroon, Saria, Azmeer | COMP-360 | Forman Christian College
        </p>
        <button class="try-now-btn" onclick="showApp()">Try Now →</button>
    </div>
    <div id="appPage" class="app-page">
        <div style="display: flex; justify-content: space-between; margin-bottom: 30px;">
            <button onclick="showLanding()" style="padding: 12px 30px; background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid #10b981; border-radius: 25px; cursor: pointer;">← Back</button>
            <h2 style="background: linear-gradient(135deg, #10b981 0%, #34d399 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Real-Time Detection Studio</h2>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 400px; gap: 30px;">
            <div style="background: rgba(39, 39, 42, 0.5); border-radius: 20px; padding: 30px;">
                <div class="camera-container">
                    <video id="video" autoplay playsinline></video>
                    <canvas id="displayCanvas" style="display: none;"></canvas>
                </div>
                <div style="text-align: center; margin: 25px 0;">
                    <label style="color: #9ca3af; margin-right: 15px;">AI Model:</label>
                    <select id="modelSelect" style="padding: 12px 25px; background: rgba(39, 39, 42, 0.8); color: #e4e4e7; border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 25px;">
                        {% for model in available_models %}
                        <option value="{{ model }}" {% if model == current_model %}selected{% endif %}>{{ model }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div style="display: flex; gap: 15px; justify-content: center;">
                    <button onclick="startCamera()" style="padding: 14px 35px; background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; border: none; border-radius: 25px; cursor: pointer;">▶ Start</button>
                    <button onclick="stopCamera()" style="padding: 14px 35px; background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white; border: none; border-radius: 25px; cursor: pointer;">⏹ Stop</button>
                    <button onclick="clearSentence()" style="padding: 14px 35px; background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white; border: none; border-radius: 25px; cursor: pointer;">🗑 Clear</button>
                </div>
            </div>
            <div>
                <div class="prediction-box">
                    <h3 style="font-size: 1rem; color: #9ca3af; margin-bottom: 15px;">Current Gesture</h3>
                    <div class="current-letter" id="currentLetter">-</div>
                    <div id="confidence" style="font-size: 1.1rem; color: #6b7280;">Confidence: 0%</div>
                </div>
                <div class="prediction-box">
                    <h3 style="font-size: 1rem; color: #9ca3af; margin-bottom: 20px;">Generated Text</h3>
                    <div id="sentenceDisplay" style="font-size: 1.8rem; color: #e4e4e7; min-height: 100px; margin-bottom: 20px;">Start making gestures...</div>
                    <button id="ttsButton" onclick="speakText()" disabled style="padding: 14px 35px; background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); color: white; border: none; border-radius: 25px; cursor: pointer;">🔊 Speak Text</button>
                </div>
            </div>
        </div>
    </div>
    <script>
        let video = document.getElementById('video');
        let displayCanvas = document.getElementById('displayCanvas');
        let displayCtx = displayCanvas.getContext('2d');
        let currentSentence = '';
        let stream = null;
        let predictionInterval = null;
        let lastPrediction = '';
        let predictionCount = 0;
        let currentAudio = null;
        let firstPredictionTime = null;  // Track when we first saw this prediction
        const MIN_CONFIDENCE = 0.85;  // Higher confidence threshold (85%)
        const REQUIRED_CONSECUTIVE = 6;  // Need 6 consecutive predictions (1.2 seconds at 200ms intervals)
        const MIN_TIME_MS = 1000;  // Minimum 1 second of stable prediction before accepting
        
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
                video.onloadedmetadata = () => {
                    displayCanvas.width = video.videoWidth;
                    displayCanvas.height = video.videoHeight;
                    video.style.display = 'none';
                    displayCanvas.style.display = 'block';
                    startPredictionLoop();
                };
            } catch (err) {
                alert('Error accessing camera: ' + err.message);
            }
        }
        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                video.srcObject = null;
                stream = null;
            }
            if (predictionInterval) {
                clearInterval(predictionInterval);
                predictionInterval = null;
            }
            video.style.display = 'block';
            displayCanvas.style.display = 'none';
        }
        function clearSentence() {
            currentSentence = '';
            lastPrediction = '';
            predictionCount = 0;
            firstPredictionTime = null;
            document.getElementById('sentenceDisplay').textContent = 'Start making gestures...';
            document.getElementById('ttsButton').disabled = true;
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
                    formData.append('draw_landmarks', 'true');
                    try {
                        const response = await fetch('/predict', { method: 'POST', body: formData });
                        const data = await response.json();
                        if (data.image_with_landmarks) {
                            const img = new Image();
                            img.onload = () => {
                                displayCtx.drawImage(img, 0, 0, displayCanvas.width, displayCanvas.height);
                            };
                            img.src = 'data:image/jpeg;base64,' + data.image_with_landmarks;
                        }
                        if (data.prediction) {
                            // Display "SPACE" for space gestures, otherwise show the letter
                            const displayText = (data.is_space || data.prediction === ' ') ? 'SPACE' : data.prediction;
                            document.getElementById('currentLetter').textContent = displayText;
                            const confidencePercent = (data.confidence * 100).toFixed(1);
                            document.getElementById('confidence').textContent = `Confidence: ${confidencePercent}%`;
                            
                            // Only consider predictions with high confidence
                            if (data.confidence >= MIN_CONFIDENCE) {
                                // Check if this is the same prediction as before
                                if (data.prediction === lastPrediction) {
                                    // Same prediction - increment counter
                                    predictionCount++;
                                    
                                    // Track when we first saw this prediction
                                    if (firstPredictionTime === null) {
                                        firstPredictionTime = Date.now();
                                    }
                                    
                                    // Check if we've met both requirements:
                                    // 1. Enough consecutive predictions
                                    // 2. Enough time has passed (verification period)
                                    const timeElapsed = Date.now() - firstPredictionTime;
                                    
                                    if (predictionCount >= REQUIRED_CONSECUTIVE && timeElapsed >= MIN_TIME_MS) {
                                        // Verified! Accept the letter
                                        currentSentence += data.prediction;
                                        document.getElementById('sentenceDisplay').textContent = currentSentence;
                                        document.getElementById('ttsButton').disabled = false;
                                        
                                        // Reset for next prediction
                                        predictionCount = 0;
                                        lastPrediction = '';
                                        firstPredictionTime = null;
                                    }
                                } else {
                                    // Different prediction - reset everything
                                    lastPrediction = data.prediction;
                                    predictionCount = 1;
                                    firstPredictionTime = Date.now();
                                }
                            } else {
                                // Confidence too low - reset if we were tracking something
                                if (lastPrediction !== '') {
                                    lastPrediction = '';
                                    predictionCount = 0;
                                    firstPredictionTime = null;
                                }
                            }
                        } else {
                            // No prediction - reset tracking
                            if (lastPrediction !== '') {
                                lastPrediction = '';
                                predictionCount = 0;
                                firstPredictionTime = null;
                            }
                        }
                    } catch (err) {
                        console.error('Prediction error:', err);
                    }
                }, 'image/jpeg');
            }, 200);
        }
        async function speakText() {
            const text = currentSentence.trim();
            if (!text) return;
            const ttsButton = document.getElementById('ttsButton');
            ttsButton.disabled = true;
            try {
                const response = await fetch('/text-to-speech', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });
                const data = await response.json();
                if (data.success) {
                    const audioBlob = new Blob([Uint8Array.from(atob(data.audio), c => c.charCodeAt(0))], { type: 'audio/mpeg' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    currentAudio = new Audio(audioUrl);
                    currentAudio.onended = () => {
                        ttsButton.disabled = false;
                        URL.revokeObjectURL(audioUrl);
                    };
                    await currentAudio.play();
                }
            } catch (err) {
                console.error('TTS error:', err);
            } finally {
                ttsButton.disabled = false;
            }
        }
        document.getElementById('modelSelect').addEventListener('change', (e) => {
            fetch('/set_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: e.target.value })
            });
        });
    </script>
</body>
</html>'''
        
        with open('templates/index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print("✅ HTML template created successfully!")
    
    def main_web_app():
        """Main function to run Flask web application."""
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
        if not models_dict:
            print("❌ No trained models found!")
            print("Please run train_model.py first to train the model.")
            return
        
        print(f"✅ Web application initialized successfully!")
        print(f"📊 Available models: {list(models_dict.keys())}")
        print(f"🎯 Current model: {current_model_name}")
        print(f"\n🌐 Starting Flask web server...")
        print(f"📱 Open your browser and go to: http://localhost:5000")
        print(f"🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Run Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    if WEB_APP_MODE and FLASK_AVAILABLE:
        try:
            main_web_app()
        except KeyboardInterrupt:
            print("\n\n⚠️  Web application stopped by user.")
            print("Exiting gracefully...")
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please check your setup and try again.")
    else:
        if WEB_APP_MODE and not FLASK_AVAILABLE:
            print("⚠️  Flask not available. Running in camera mode instead.")
        # Run direct camera mode (original skeleton behavior)
        run_camera_mode()
