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

# Flask imports 
try:
    from flask import Flask, render_template, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️  Flask not available. Web app mode disabled. Install Flask to enable web app.")

# Constants
Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # Will be updated based on loaded model
Predictions = True # Set to True to enable live predictions

# Web app mode flag - set to True to run Flask web app, False for direct camera mode
WEB_APP_MODE = True 

Script_dir = Path(__file__).parent.absolute()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize model 
model = None
models_dict = {}  # store multiple models if possible
if Predictions:
    try:
        model_path = Script_dir / "models" / "cnn_baseline.h5"
        if model_path.exists():
            print(f"Loading model from {model_path}")
            model = load_model(str(model_path))
            models_dict['CNN (Best)'] = model 
            
            # Auto-detect number of classes from model output shape
            num_classes = model.output.shape[1]
            print(f"   Model loaded successfully!")
            print(f"   Detected {num_classes} classes in model")
            
            # Update Alphabets list based on number of classes
            if num_classes == 29:
                # Model trained with A-Z + space + del + nothing
                Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [" ", "DEL", "NONE"]
                print(f"   Using 29 classes: A-Z + space + del + nothing")
            elif num_classes == 27:
                # Model trained with A-Z + space
                Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [" "]
                print(f"   Using 27 classes: A-Z + space")
            elif num_classes == 26:
                # Model trained with A-Z only
                Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                print(f"   Using 26 classes: A-Z only")
            else:
                # Unknown number of classes, use default
                print(f"   Warning: Unknown number of classes ({num_classes}), using default A-Z")
                Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:num_classes] if num_classes <= 26 else list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [" "] * (num_classes - 26)
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
        # Ensure Alphabets is updated if this model has different number of classes
        if models_dict['CNN (Final)'].output.shape[1] != len(Alphabets):
            num_classes = models_dict['CNN (Final)'].output.shape[1]
            if num_classes == 29:
                Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [" ", "DEL", "NONE"]
            elif num_classes == 27:
                Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [" "]
            elif num_classes == 26:
                Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
except:
    pass

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.3, 
    min_tracking_confidence=0.3,  
    model_complexity=1  
)

# ============================================================================
# DIRECT CAMERA MODE 
# ============================================================================
def run_camera_mode():
    """Run the camera detection mode """

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

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(img_rgb)
        
        # Convert back to BGR for OpenCV display
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Drawing hand landmarks
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

                landmarks_array = landmarks_array.reshape(1, 63)
                
                # Make prediction using model 
                preds = model.predict(landmarks_array, verbose=0)
                predicted_class_idx = np.argmax(preds, axis=1)[0]
                confidence = preds[0][predicted_class_idx]
                predicted_letter = Alphabets[predicted_class_idx]
                
                # Format display text 
                if predicted_letter == ' ':
                    display_text = "SPACE"
                elif predicted_letter == "DEL":
                    display_text = "DEL"
                elif predicted_letter == "NONE":
                    display_text = "NONE"
                else:
                    display_text = predicted_letter
                
                # Overlay prediction on frame
                text = f"Predicted: {display_text} ({confidence:.2f})"
                cv2.putText(
                    img,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),  # Green color for all predictions
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
    
    # Always use the best model (CNN (Final) if available, otherwise CNN (Best))
    if 'CNN (Final)' in models_dict:
        current_model_name = 'CNN (Final)'
    elif 'CNN (Best)' in models_dict:
        current_model_name = 'CNN (Best)'
    else:
        current_model_name = list(models_dict.keys())[0] if models_dict else None
    
    def extract_landmarks_from_image(image):
        """
        Extract landmarks from image 
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
        Predict gesture from landmarks 
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
            
            # Make prediction using model 
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
        """Draw landmarks on image"""
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
            
            # Always use the best model 
            model_name = current_model_name
            
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
            
            # Extract landmarks 
            landmarks_array, hand_landmarks_obj = extract_landmarks_from_image(image)
            
            # Make prediction 
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
    
    # HTML Template Creation (frontend design from app.py, backend logic preserved)
    def create_templates():
        """Create HTML templates for the web application."""
        os.makedirs('templates', exist_ok=True)
        
        # Frontend design copied from app.py, but keeping stricter prediction logic
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

        /* Animated Green Background with Particles */
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

        .bg-animation::before {
            content: '';
            position: absolute;
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(16, 185, 129, 0.15) 0%, transparent 70%);
            top: -300px;
            right: -300px;
            animation: float 25s infinite ease-in-out;
        }

        .bg-animation::after {
            content: '';
            position: absolute;
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, rgba(52, 211, 153, 0.1) 0%, transparent 70%);
            bottom: -250px;
            left: -250px;
            animation: float 20s infinite ease-in-out reverse;
        }

        /* Floating Particles */
        .particle {
            position: absolute;
            width: 4px;
            height: 4px;
            background: rgba(16, 185, 129, 0.6);
            border-radius: 50%;
            animation: particleFloat 15s infinite ease-in-out;
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.8);
        }

        .particle:nth-child(1) { left: 10%; top: 20%; animation-delay: 0s; animation-duration: 12s; }
        .particle:nth-child(2) { left: 20%; top: 80%; animation-delay: 2s; animation-duration: 15s; }
        .particle:nth-child(3) { left: 60%; top: 10%; animation-delay: 4s; animation-duration: 18s; }
        .particle:nth-child(4) { left: 80%; top: 70%; animation-delay: 6s; animation-duration: 14s; }
        .particle:nth-child(5) { left: 30%; top: 50%; animation-delay: 8s; animation-duration: 16s; }
        .particle:nth-child(6) { left: 70%; top: 40%; animation-delay: 3s; animation-duration: 13s; }
        .particle:nth-child(7) { left: 50%; top: 90%; animation-delay: 5s; animation-duration: 17s; }
        .particle:nth-child(8) { left: 90%; top: 30%; animation-delay: 7s; animation-duration: 11s; }

        @keyframes particleFloat {
            0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.3; }
            25% { transform: translate(50px, -50px) scale(1.5); opacity: 0.8; }
            50% { transform: translate(100px, 0) scale(1); opacity: 0.5; }
            75% { transform: translate(50px, 50px) scale(1.2); opacity: 0.7; }
        }

        @keyframes float {
            0%, 100% { transform: translate(0, 0) rotate(0deg); }
            33% { transform: translate(40px, -40px) rotate(120deg); }
            66% { transform: translate(-30px, 30px) rotate(240deg); }
        }

        /* Grid Pattern Overlay */
        .grid-overlay {
            position: fixed;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            z-index: -1;
            background-image: 
                linear-gradient(rgba(16, 185, 129, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(16, 185, 129, 0.03) 1px, transparent 1px);
            background-size: 50px 50px;
            animation: gridMove 20s linear infinite;
        }

        @keyframes gridMove {
            0% { transform: translate(0, 0); }
            100% { transform: translate(50px, 50px); }
        }

        /* Landing Page */
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

        /* Animated Hand Icon */
        .hand-container {
            position: relative;
            margin-bottom: 40px;
        }

        .hand-icon {
            font-size: 10rem;
            animation: handWave 3s infinite ease-in-out;
            filter: drop-shadow(0 0 40px rgba(16, 185, 129, 0.8));
            position: relative;
            z-index: 2;
        }

        .hand-glow {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 200px;
            height: 200px;
            background: radial-gradient(circle, rgba(16, 185, 129, 0.4) 0%, transparent 70%);
            border-radius: 50%;
            animation: pulse 3s infinite ease-in-out;
            z-index: 1;
        }

        @keyframes handWave {
            0%, 100% { transform: rotate(0deg) translateY(0); }
            10% { transform: rotate(14deg) translateY(-10px); }
            20% { transform: rotate(-8deg) translateY(0); }
            30% { transform: rotate(14deg) translateY(-10px); }
            40% { transform: rotate(-4deg) translateY(0); }
            50% { transform: rotate(10deg) translateY(-5px); }
            60% { transform: rotate(0deg) translateY(0); }
        }

        @keyframes pulse {
            0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.5; }
            50% { transform: translate(-50%, -50%) scale(1.3); opacity: 0.2; }
        }

        /* Title with Gradient and Animation */
        .landing-page h1 {
            font-size: 4.5rem;
            font-weight: 900;
            margin-bottom: 25px;
            background: linear-gradient(135deg, #10b981 0%, #34d399 50%, #6ee7b7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: titleGlow 3s ease-in-out infinite, fadeInDown 1s ease-out;
            text-shadow: 0 0 80px rgba(16, 185, 129, 0.5);
            letter-spacing: -2px;
        }

        @keyframes titleGlow {
            0%, 100% { filter: brightness(1); }
            50% { filter: brightness(1.3); }
        }

        .landing-page .subtitle {
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: #9ca3af;
            max-width: 750px;
            animation: fadeInUp 1s ease-out 0.2s backwards;
            line-height: 1.6;
        }

        .landing-page .team-info {
            font-size: 1rem;
            color: #6b7280;
            margin-bottom: 60px;
            animation: fadeInUp 1s ease-out 0.4s backwards;
            padding: 12px 30px;
            background: rgba(16, 185, 129, 0.1);
            border-radius: 30px;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        /* Feature Cards with 3D Effect */
        .feature-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 25px;
            max-width: 1000px;
            margin: 50px auto;
            padding: 0 20px;
            animation: fadeInUp 1s ease-out 0.6s backwards;
        }

        .feature-card {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(52, 211, 153, 0.05) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 20px;
            padding: 35px 25px;
            text-align: center;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }

        .feature-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.2), transparent);
            transition: left 0.5s;
        }

        .feature-card:hover::before {
            left: 100%;
        }

        .feature-card:hover {
            transform: translateY(-10px) scale(1.05);
            border-color: rgba(16, 185, 129, 0.6);
            box-shadow: 0 20px 60px rgba(16, 185, 129, 0.3);
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(52, 211, 153, 0.1) 100%);
        }

        .feature-card .icon {
            font-size: 3.5rem;
            margin-bottom: 20px;
            animation: iconBounce 2s infinite ease-in-out;
            display: inline-block;
        }

        .feature-card:nth-child(1) .icon { animation-delay: 0s; }
        .feature-card:nth-child(2) .icon { animation-delay: 0.2s; }
        .feature-card:nth-child(3) .icon { animation-delay: 0.4s; }
        .feature-card:nth-child(4) .icon { animation-delay: 0.6s; }

        @keyframes iconBounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }

        .feature-card h3 {
            font-size: 1.2rem;
            margin-bottom: 12px;
            color: #10b981;
            font-weight: 700;
        }

        .feature-card p {
            font-size: 0.95rem;
            color: #9ca3af;
            line-height: 1.6;
        }

        /* Try Now Button with Advanced Effects */
        .try-now-btn {
            padding: 22px 70px;
            font-size: 1.4rem;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            font-weight: 800;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 15px 40px rgba(16, 185, 129, 0.4);
            animation: fadeInUp 1s ease-out 0.8s backwards, buttonPulse 2s infinite;
            position: relative;
            overflow: hidden;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .try-now-btn::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }

        .try-now-btn:hover::before {
            width: 400px;
            height: 400px;
        }

        .try-now-btn:hover {
            transform: translateY(-5px) scale(1.05);
            box-shadow: 0 20px 60px rgba(16, 185, 129, 0.6);
        }

        .try-now-btn:active {
            transform: translateY(-2px) scale(1.02);
        }

        @keyframes buttonPulse {
            0%, 100% { box-shadow: 0 15px 40px rgba(16, 185, 129, 0.4); }
            50% { box-shadow: 0 15px 50px rgba(16, 185, 129, 0.6); }
        }

        /* Scroll Indicator */
        .scroll-indicator {
            position: absolute;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            animation: bounce 2s infinite;
        }

        .scroll-indicator::before {
            content: '↓';
            font-size: 2rem;
            color: #10b981;
            opacity: 0.7;
        }

        @keyframes bounce {
            0%, 100% { transform: translateX(-50%) translateY(0); }
            50% { transform: translateX(-50%) translateY(10px); }
        }

        /* App Page Styles */
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
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        .back-btn {
            padding: 12px 30px;
            background: rgba(16, 185, 129, 0.1);
            color: #10b981;
            border: 1px solid #10b981;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .back-btn:hover {
            background: #10b981;
            color: white;
            transform: translateX(-5px);
        }

        .header h2 {
            font-size: 1.8rem;
            background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
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
            border: 1px solid rgba(16, 185, 129, 0.2);
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

        #video, #displayCanvas {
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
            color: #9ca3af;
            font-weight: 600;
            margin-right: 15px;
        }

        .model-selector select {
            padding: 12px 25px;
            background: rgba(39, 39, 42, 0.8);
            color: #e4e4e7;
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 25px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .model-selector select:hover {
            border-color: #10b981;
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

        .start-btn {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            box-shadow: 0 5px 20px rgba(16, 185, 129, 0.3);
        }

        .start-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5);
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
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(52, 211, 153, 0.1) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
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
            background: radial-gradient(circle, rgba(16, 185, 129, 0.15) 0%, transparent 70%);
            animation: rotate 10s linear infinite;
        }

        @keyframes rotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .prediction-box h3 {
            font-size: 1rem;
            color: #9ca3af;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .current-letter {
            font-size: 5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 20px 0;
            position: relative;
            z-index: 1;
        }

        .confidence {
            font-size: 1.1rem;
            color: #6b7280;
            font-weight: 600;
        }

        .confidence-bar {
            width: 100%;
            height: 8px;
            background: rgba(39, 39, 42, 0.5);
            border-radius: 10px;
            margin-top: 15px;
            overflow: hidden;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
            border-radius: 10px;
            transition: width 0.3s ease;
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
        }

        .sentence-box {
            background: rgba(39, 39, 42, 0.5);
            border: 1px solid rgba(16, 185, 129, 0.2);
            padding: 30px;
            border-radius: 20px;
            backdrop-filter: blur(10px);
            min-height: 200px;
            flex: 1;
        }

        .sentence-box h3 {
            font-size: 1rem;
            color: #9ca3af;
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
            margin-bottom: 20px;
        }

        /* Text-to-Speech Button */
        .tts-button {
            padding: 14px 35px;
            background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 5px 20px rgba(139, 92, 246, 0.3);
            display: inline-flex;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
        }

        .tts-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5);
            background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
        }

        .tts-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .tts-button .icon {
            font-size: 1.2rem;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 20px;
        }

        .stat-card {
            background: rgba(39, 39, 42, 0.5);
            border: 1px solid rgba(16, 185, 129, 0.2);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }

        .stat-card .value {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .stat-card .label {
            font-size: 0.9rem;
            color: #6b7280;
            margin-top: 5px;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 1200px) {
            .main-content { grid-template-columns: 1fr; }
        }

        @media (max-width: 768px) {
            .landing-page h1 { font-size: 2.8rem; }
            .hand-icon { font-size: 6rem; }
            .feature-cards { grid-template-columns: 1fr; }
            .current-letter { font-size: 3.5rem; }
            .sentence-display { font-size: 1.4rem; }
            .top-bar { flex-direction: column; gap: 15px; }
            .stats-grid { grid-template-columns: 1fr; }
        }

        ::-webkit-scrollbar { width: 10px; }
        ::-webkit-scrollbar-track { background: #18181b; }
        ::-webkit-scrollbar-thumb { background: #10b981; border-radius: 5px; }
        ::-webkit-scrollbar-thumb:hover { background: #059669; }
    </style>
</head>
<body>
    <div class="bg-animation">
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
        <div class="particle"></div>
    </div>
    <div class="grid-overlay"></div>

    <div id="landingPage" class="landing-page">
        <div class="hand-container">
            <div class="hand-glow"></div>
            <div class="hand-icon">🤟</div>
        </div>
        <h1>Sign Language Recognition</h1>
        <p class="subtitle">Transform hand gestures into text in real-time using cutting-edge AI-powered deep learning technology</p>
        <p class="team-info">Team: Haroon, Saria, Azmeer | COMP-360 | Forman Christian College</p>
        
        <div class="feature-cards">
            <div class="feature-card">
                <div class="icon">🎥</div>
                <h3>Real-Time Detection</h3>
                <p>Instant gesture recognition using your webcam with millisecond response</p>
            </div>
            <div class="feature-card">
                <div class="icon">🧠</div>
                <h3>AI Powered</h3>
                <p>AI model that learns from thousands of hand gestures to recognize signs accurately</p>
            </div>
            <div class="feature-card">
                <div class="icon">✍️</div>
                <h3>Text Generation</h3>
                <p>Automatic sentence building from detected gestures</p>
            </div>
            <div class="feature-card">
                <div class="icon">⚡</div>
                <h3>High Accuracy</h3>
                <p>Trained on thousands of ASL samples for reliability</p>
            </div>
        </div>

        <button class="try-now-btn" onclick="showApp()">Try Now →</button>
        <div class="scroll-indicator"></div>
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
                    <video id="video" autoplay playsinline></video>
                    <canvas id="displayCanvas" style="display: none;"></canvas>
                    <div class="camera-overlay" id="cameraStatus">📷 Camera Off</div>
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
                    <button class="tts-button" id="ttsButton" onclick="speakText()" disabled>
                        <span class="icon">🔊</span>
                        <span>Speak Text</span>
                    </button>
                </div>
            </div>
        </div>
    </div>
    <script>
        // DOM Elements
        const video = document.getElementById('video');
        const displayCanvas = document.getElementById('displayCanvas');
        const displayCtx = displayCanvas.getContext('2d');
        
        // State variables
        let currentSentence = '';
        let stream = null;
        let predictionInterval = null;
        let animationFrameId = null;
        let lastPrediction = '';
        let predictionCount = 0;
        let totalLettersDetected = 0;
        let currentAudio = null;
        let isPredicting = false;
        let lastFrameWithLandmarks = null;
        let firstPredictionTime = null;  // Track when we first saw this prediction
        const MIN_CONFIDENCE = 0.85;  // Higher confidence threshold (85%)
        const REQUIRED_CONSECUTIVE = 6;  // Need 6 consecutive predictions
        const MIN_TIME_MS = 1000;  // Minimum 1 second of stable prediction before accepting

        function showApp() {
            document.getElementById('landingPage').style.display = 'none';
            document.getElementById('appPage').classList.add('active');
        }

        function showLanding() {
            document.getElementById('appPage').classList.remove('active');
            document.getElementById('landingPage').style.display = 'flex';
            stopCamera();
            if (currentAudio) {
                currentAudio.pause();
                currentAudio = null;
            }
        }

        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: { ideal: 640 }, 
                        height: { ideal: 480 },
                        frameRate: { ideal: 30 }
                    } 
                });
                video.srcObject = stream;
                
                // Wait for video to be ready
                video.onloadedmetadata = () => {
                    // Set canvas size to match video
                    displayCanvas.width = video.videoWidth;
                    displayCanvas.height = video.videoHeight;
                    
                    document.getElementById('cameraStatus').textContent = '🔴 Live';
                    document.getElementById('cameraStatus').style.background = 'rgba(239, 68, 68, 0.8)';
                    
                    // Start the smooth video display loop
                    startVideoLoop();
                    
                    // Start prediction loop (separate from display)
                    startPredictionLoop();
                };
            } catch (err) {
                alert('Error accessing camera: ' + err.message);
            }
        }

        function stopCamera() {
            // Stop prediction loop
            if (predictionInterval) {
                clearInterval(predictionInterval);
                predictionInterval = null;
            }
            
            // Stop animation frame
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }
            
            // Stop camera stream
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                video.srcObject = null;
                stream = null;
            }
            
            // Reset display
            video.style.display = 'block';
            displayCanvas.style.display = 'none';
            lastFrameWithLandmarks = null;
            
            document.getElementById('cameraStatus').textContent = '📷 Camera Off';
            document.getElementById('cameraStatus').style.background = 'rgba(0, 0, 0, 0.8)';
        }

        function clearSentence() {
            currentSentence = '';
            lastPrediction = '';
            predictionCount = 0;
            totalLettersDetected = 0;
            firstPredictionTime = null;
            document.getElementById('sentenceDisplay').textContent = 'Start making gestures to build your message...';
            document.getElementById('totalLetters').textContent = '0';
            document.getElementById('wordsCount').textContent = '0';
            document.getElementById('ttsButton').disabled = true;
            if (currentAudio) {
                currentAudio.pause();
                currentAudio = null;
            }
        }

        function updateStats() {
            document.getElementById('totalLetters').textContent = totalLettersDetected;
            const words = currentSentence.trim().split(/\\s+/).filter(w => w.length > 0).length;
            document.getElementById('wordsCount').textContent = words;
            document.getElementById('ttsButton').disabled = currentSentence.trim().length === 0;
        }

        // Smooth video display loop - runs at full frame rate
        function startVideoLoop() {
            // Video loop removed - we only draw frames from prediction results
            // This prevents the blinking caused by two loops fighting over the canvas
            video.style.display = 'none';
            displayCanvas.style.display = 'block';
        }

        // Prediction loop - handles both video display and predictions
        function startPredictionLoop() {
            if (predictionInterval) clearInterval(predictionInterval);
            
            // Run predictions every 200ms for smooth display with landmarks
            predictionInterval = setInterval(async () => {
                if (!stream || isPredicting) return;
                
                isPredicting = true;
                
                try {
                    // Create a temporary canvas to capture current frame
                    const tempCanvas = document.createElement('canvas');
                    tempCanvas.width = video.videoWidth;
                    tempCanvas.height = video.videoHeight;
                    const tempCtx = tempCanvas.getContext('2d');
                    tempCtx.drawImage(video, 0, 0);
                    
                    // Convert to blob
                    const blob = await new Promise(resolve => tempCanvas.toBlob(resolve, 'image/jpeg', 0.8));
                    
                    const formData = new FormData();
                    formData.append('image', blob, 'frame.jpg');
                    formData.append('draw_landmarks', 'true');
                    
                    const response = await fetch('/predict', { method: 'POST', body: formData });
                    const data = await response.json();
                    
                    // Update the display with landmarks
                    if (data.image_with_landmarks) {
                        const img = new Image();
                        img.onload = () => {
                            // Draw the frame with landmarks
                            displayCtx.drawImage(img, 0, 0, displayCanvas.width, displayCanvas.height);
                        };
                        img.src = 'data:image/jpeg;base64,' + data.image_with_landmarks;
                    }
                    
                    // Update prediction display
                    if (data.prediction) {
                        // Format display text (model handles space, del, nothing directly)
                        let displayText = data.prediction;
                        if (data.prediction === ' ') {
                            displayText = 'SPACE';
                        } else if (data.prediction === 'DEL') {
                            displayText = 'DEL';
                        } else if (data.prediction === 'NONE') {
                            displayText = 'NONE';
                        }
                        document.getElementById('currentLetter').textContent = displayText;
                        const confidencePercent = (data.confidence * 100).toFixed(1);
                        document.getElementById('confidence').textContent = `Confidence: ${confidencePercent}%`;
                        document.getElementById('confidenceFill').style.width = confidencePercent + '%';
                        
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
                                    // Verified! Handle the prediction
                                    if (data.prediction === 'DEL' || data.prediction === 'del') {
                                        // Delete gesture: remove last character
                                        if (currentSentence.length > 0) {
                                            currentSentence = currentSentence.slice(0, -1);
                                        }
                                    } else {
                                        // Regular letter/space: add to sentence
                                        currentSentence += data.prediction;
                                    }
                                    
                                    totalLettersDetected++;
                                    document.getElementById('sentenceDisplay').textContent = currentSentence || 'Start making gestures to build your message...';
                                    updateStats();
                                    
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
                    } else if (data.error) {
                        document.getElementById('currentLetter').textContent = '?';
                        document.getElementById('confidence').textContent = data.error;
                        document.getElementById('confidenceFill').style.width = '0%';
                    }
                } catch (err) {
                    console.error('Prediction error:', err);
                } finally {
                    isPredicting = false;
                }
            }, 200);  // 200ms = 5 predictions per second (smooth but not overwhelming)
        }

        async function speakText() {
            const text = currentSentence.trim();
            if (!text) {
                alert('No text to speak!');
                return;
            }

            const ttsButton = document.getElementById('ttsButton');
            ttsButton.disabled = true;
            ttsButton.innerHTML = '<span class="icon">⏳</span><span>Loading...</span>';

            try {
                const response = await fetch('/text-to-speech', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });

                const data = await response.json();

                if (data.success) {
                    if (currentAudio) {
                        currentAudio.pause();
                    }

                    const audioBlob = base64ToBlob(data.audio, 'audio/mpeg');
                    const audioUrl = URL.createObjectURL(audioBlob);
                    currentAudio = new Audio(audioUrl);
                    
                    currentAudio.onended = () => {
                        ttsButton.disabled = false;
                        ttsButton.innerHTML = '<span class="icon">🔊</span><span>Speak Text</span>';
                        URL.revokeObjectURL(audioUrl);
                    };

                    currentAudio.onerror = () => {
                        alert('Error playing audio');
                        ttsButton.disabled = false;
                        ttsButton.innerHTML = '<span class="icon">🔊</span><span>Speak Text</span>';
                    };

                    ttsButton.innerHTML = '<span class="icon">🔊</span><span>Playing...</span>';
                    await currentAudio.play();
                } else {
                    alert('Error: ' + (data.error || 'Failed to generate speech'));
                    ttsButton.disabled = false;
                    ttsButton.innerHTML = '<span class="icon">🔊</span><span>Speak Text</span>';
                }
            } catch (err) {
                console.error('TTS error:', err);
                alert('Failed to generate speech: ' + err.message);
                ttsButton.disabled = false;
                ttsButton.innerHTML = '<span class="icon">🔊</span><span>Speak Text</span>';
            }
        }

        function base64ToBlob(base64, mimeType) {
            const byteCharacters = atob(base64);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            return new Blob([byteArray], { type: mimeType });
        }

    </script>
</body>
</html>'''
        
        with open('templates/index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(" HTML template created successfully!")
    
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
            print(" No trained models found!")
            print("Please run train_model.py first to train the model.")
            return
        
        print(f" Web application initialized successfully!")
        print(f" Available models: {list(models_dict.keys())}")
        print(f" Current model: {current_model_name}")
        print(f"\n Starting Flask web server...")
        print(f" Open your browser and go to: http://localhost:5000")
        print(f" Press Ctrl+C to stop the server")
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
            print("\n\n  Web application stopped by user.")
            print("Exiting gracefully...")
        except Exception as e:
            print(f"\n An error occurred: {e}")
            print("Please check your setup and try again.")
    else:
        if WEB_APP_MODE and not FLASK_AVAILABLE:
            print("  Flask not available. Running in camera mode instead.")
        # Run direct camera mode 
        run_camera_mode()
