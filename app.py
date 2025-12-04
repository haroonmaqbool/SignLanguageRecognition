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
    from flask import Flask, render_template_string, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Web app mode disabled.")

Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
Predictions = True
WEB_APP_MODE = True 
Script_dir = Path(__file__).parent.absolute()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

model = None
models_dict = {}
if Predictions:
    try:
        model_path = Script_dir / "models" / "cnn_baseline.h5"
        if model_path.exists():
            print(f"Loading model from {model_path}")
            model = load_model(str(model_path))
            models_dict['CNN (Best)'] = model 
            num_classes = model.output.shape[1]
            print(f"   Model loaded successfully!")
            print(f"   Detected {num_classes} classes in model")
            if num_classes == 29:
                Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [" ", "DEL", "NONE"]
            elif num_classes == 27:
                Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [" "]
            elif num_classes == 26:
                Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        else:
            print(f"  Model not found: {model_path}")
            Predictions = False
    except Exception as e:
        print(f"  Error loading model: {e}")
        Predictions = False

try:
    model_last_path = Script_dir / "models" / "cnn_last.h5"
    if model_last_path.exists():
        models_dict['CNN (Final)'] = load_model(str(model_last_path))
        print(f"   Additional model CNN (Final) loaded!")
except:
    pass

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3, model_complexity=1)

def run_camera_mode():
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Sign Language Recognition - Live Camera")
    print("Press 'q' to quit")
    
    while webcam.isOpened():
        success, img = webcam.read()
        if not success:
            break
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if Predictions and model is not None:
                first_hand = results.multi_hand_landmarks[0]
                is_right_hand = False
                if results.multi_handedness:
                    handedness = results.multi_handedness[0]
                    if hasattr(handedness, 'classification') and len(handedness.classification) > 0:
                        is_right_hand = handedness.classification[0].label == 'Right'
                landmarks_array = np.zeros(63)
                idx = 0
                for landmark in first_hand.landmark:
                    landmarks_array[idx] = landmark.x
                    landmarks_array[idx + 1] = landmark.y
                    landmarks_array[idx + 2] = landmark.z
                    idx += 3
                if is_right_hand:
                    for i in range(0, 63, 3):
                        landmarks_array[i] = 1.0 - landmarks_array[i]
                landmarks_array = landmarks_array.reshape(1, 63)
                preds = model.predict(landmarks_array, verbose=0)
                predicted_class_idx = np.argmax(preds, axis=1)[0]
                confidence = preds[0][predicted_class_idx]
                predicted_letter = Alphabets[predicted_class_idx]
                display_text = "SPACE" if predicted_letter == ' ' else predicted_letter
                text = f"Predicted: {display_text} ({confidence:.2f})"
                cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            if Predictions:
                cv2.putText(img, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Webcam - Sign Language Recognition", img)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
    webcam.release()
    cv2.destroyAllWindows()

if FLASK_AVAILABLE and WEB_APP_MODE:
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    
    if 'CNN (Final)' in models_dict:
        current_model_name = 'CNN (Final)'
    elif 'CNN (Best)' in models_dict:
        current_model_name = 'CNN (Best)'
    else:
        current_model_name = list(models_dict.keys())[0] if models_dict else None
    
    def extract_landmarks_from_image(image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        if results.multi_hand_landmarks:
            first_hand = results.multi_hand_landmarks[0]
            is_right_hand = False
            if results.multi_handedness:
                handedness = results.multi_handedness[0]
                if hasattr(handedness, 'classification') and len(handedness.classification) > 0:
                    is_right_hand = handedness.classification[0].label == 'Right'
            landmarks_array = np.zeros(63)
            idx = 0
            for landmark in first_hand.landmark:
                landmarks_array[idx] = landmark.x
                landmarks_array[idx + 1] = landmark.y
                landmarks_array[idx + 2] = landmark.z
                idx += 3
            if is_right_hand:
                for i in range(0, 63, 3):
                    landmarks_array[i] = 1.0 - landmarks_array[i]
            return landmarks_array, first_hand
        return None, None
    
    def predict_gesture(landmarks_array, model_to_use=None):
        if landmarks_array is None:
            return {'prediction': None, 'confidence': 0.0, 'error': 'No hand detected in image'}
        model_to_use = model_to_use or current_model_name
        if model_to_use not in models_dict:
            return {'prediction': None, 'confidence': 0.0, 'error': f'Model {model_to_use} not available'}
        try:
            landmarks_array = landmarks_array.reshape(1, 63)
            preds = models_dict[model_to_use].predict(landmarks_array, verbose=0)
            predicted_class_idx = np.argmax(preds, axis=1)[0]
            confidence = preds[0][predicted_class_idx]
            predicted_letter = Alphabets[predicted_class_idx]
            top_indices = np.argsort(preds[0])[-3:][::-1]
            top_predictions = [{'letter': Alphabets[idx], 'confidence': float(preds[0][idx])} for idx in top_indices]
            return {'prediction': predicted_letter, 'confidence': float(confidence), 'top_predictions': top_predictions, 'model_used': model_to_use, 'error': None}
        except Exception as e:
            return {'prediction': None, 'confidence': 0.0, 'error': f'Prediction error: {str(e)}'}
    
    def draw_landmarks_on_image(image, hand_landmarks):
        if hand_landmarks is None:
            return image
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        return annotated_image

    HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SignSpeak - Sign Language Recognition</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700;9..40,800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        :root {
            --mint-50: #f0fdf4; --mint-100: #dcfce7; --mint-200: #bbf7d0; --mint-300: #86efac;
            --mint-400: #4ade80; --mint-500: #22c55e; --mint-600: #16a34a; --mint-700: #15803d;
            --teal-400: #2dd4bf; --teal-500: #14b8a6; --emerald-400: #34d399;
            --slate-50: #f8fafc; --slate-100: #f1f5f9; --slate-200: #e2e8f0; --slate-300: #cbd5e1;
            --slate-400: #94a3b8; --slate-500: #64748b; --slate-600: #475569; --slate-700: #334155;
            --slate-800: #1e293b; --slate-900: #0f172a;
        }
        html { scroll-behavior: smooth; }
        body { font-family: 'DM Sans', sans-serif; background: var(--slate-50); color: var(--slate-800); min-height: 100vh; overflow-x: hidden; }

        .animated-bg {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1;
            background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 25%, #f0fdfa 50%, #f5f3ff 75%, #fdf4ff 100%);
            background-size: 400% 400%;
            animation: gradientFlow 15s ease infinite;
        }
        @keyframes gradientFlow { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }

        .blob-container { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; overflow: hidden; pointer-events: none; }
        .blob { position: absolute; border-radius: 50%; filter: blur(80px); opacity: 0.6; animation: blobFloat 20s ease-in-out infinite; }
        .blob-1 { width: 600px; height: 600px; background: linear-gradient(135deg, rgba(134, 239, 172, 0.4), rgba(45, 212, 191, 0.3)); top: -200px; left: -100px; }
        .blob-2 { width: 500px; height: 500px; background: linear-gradient(135deg, rgba(52, 211, 153, 0.3), rgba(167, 139, 250, 0.2)); top: 50%; right: -150px; animation-delay: -5s; animation-duration: 25s; }
        .blob-3 { width: 400px; height: 400px; background: linear-gradient(135deg, rgba(74, 222, 128, 0.35), rgba(45, 212, 191, 0.25)); bottom: -100px; left: 30%; animation-delay: -10s; animation-duration: 22s; }
        .blob-4 { width: 350px; height: 350px; background: linear-gradient(135deg, rgba(134, 239, 172, 0.3), rgba(192, 132, 252, 0.2)); top: 30%; left: 50%; animation-delay: -7s; animation-duration: 18s; }
        @keyframes blobFloat { 0%, 100% { transform: translate(0, 0) scale(1) rotate(0deg); } 25% { transform: translate(30px, -50px) scale(1.1) rotate(5deg); } 50% { transform: translate(-20px, 30px) scale(0.95) rotate(-5deg); } 75% { transform: translate(40px, 20px) scale(1.05) rotate(3deg); } }

        .particles { position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; pointer-events: none; }
        .particle { position: absolute; width: 8px; height: 8px; background: linear-gradient(135deg, var(--mint-400), var(--teal-400)); border-radius: 50%; opacity: 0.4; animation: particleDrift 15s ease-in-out infinite; }
        .particle:nth-child(1) { left: 10%; top: 20%; } .particle:nth-child(2) { left: 20%; top: 70%; animation-delay: -2s; animation-duration: 18s; }
        .particle:nth-child(3) { left: 70%; top: 15%; animation-delay: -4s; animation-duration: 20s; } .particle:nth-child(4) { left: 85%; top: 60%; animation-delay: -6s; animation-duration: 16s; }
        .particle:nth-child(5) { left: 40%; top: 85%; animation-delay: -8s; animation-duration: 22s; } .particle:nth-child(6) { left: 60%; top: 40%; animation-delay: -3s; animation-duration: 17s; }
        @keyframes particleDrift { 0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.4; } 25% { transform: translate(20px, -30px) scale(1.2); opacity: 0.6; } 50% { transform: translate(-15px, 20px) scale(0.8); opacity: 0.3; } 75% { transform: translate(25px, 10px) scale(1.1); opacity: 0.5; } }

        .navbar { position: fixed; top: 0; left: 0; right: 0; z-index: 1000; padding: 20px 60px; display: flex; justify-content: space-between; align-items: center; background: rgba(255, 255, 255, 0.8); backdrop-filter: blur(20px); border-bottom: 1px solid rgba(134, 239, 172, 0.2); transition: all 0.3s ease; }
        .navbar.scrolled { padding: 15px 60px; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.05); }
        .logo { display: flex; align-items: center; gap: 12px; font-size: 1.5rem; font-weight: 700; color: var(--slate-800); }
        .logo-icon { width: 40px; height: 40px; background: linear-gradient(135deg, var(--mint-400), var(--teal-500)); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; }
        .nav-links { display: flex; align-items: center; gap: 40px; }
        .nav-links a { text-decoration: none; color: var(--slate-600); font-weight: 500; font-size: 0.95rem; transition: color 0.3s ease; position: relative; }
        .nav-links a::after { content: ''; position: absolute; bottom: -4px; left: 0; width: 0; height: 2px; background: linear-gradient(90deg, var(--mint-500), var(--teal-500)); transition: width 0.3s ease; }
        .nav-links a:hover { color: var(--mint-600); }
        .nav-links a:hover::after { width: 100%; }
        .nav-cta { padding: 12px 28px; background: linear-gradient(135deg, var(--mint-500), var(--teal-500)); color: white; border: none; border-radius: 50px; font-family: inherit; font-size: 0.95rem; font-weight: 600; cursor: pointer; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3); }
        .nav-cta:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(34, 197, 94, 0.4); }

        .hero { min-height: 100vh; display: flex; align-items: center; padding: 120px 60px 80px; position: relative; }
        .hero-content { flex: 1; max-width: 650px; }
        .hero-badge { display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px; background: rgba(134, 239, 172, 0.2); border: 1px solid rgba(134, 239, 172, 0.4); border-radius: 50px; font-size: 0.85rem; font-weight: 500; color: var(--mint-700); margin-bottom: 24px; animation: fadeInUp 0.8s ease forwards; }
        .hero-badge-dot { width: 8px; height: 8px; background: var(--mint-500); border-radius: 50%; animation: pulse 2s ease-in-out infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(1.2); } }
        .hero h1 { font-size: clamp(2.8rem, 5vw, 4.5rem); font-weight: 800; line-height: 1.1; color: var(--slate-900); margin-bottom: 24px; animation: fadeInUp 0.8s ease 0.1s forwards; opacity: 0; }
        .hero h1 .highlight { background: linear-gradient(135deg, var(--mint-500), var(--teal-500)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
        .hero-description { font-size: 1.2rem; line-height: 1.7; color: var(--slate-600); margin-bottom: 40px; animation: fadeInUp 0.8s ease 0.2s forwards; opacity: 0; }
        .hero-buttons { display: flex; gap: 16px; animation: fadeInUp 0.8s ease 0.3s forwards; opacity: 0; }
        .btn-primary { display: inline-flex; align-items: center; gap: 10px; padding: 18px 36px; background: linear-gradient(135deg, var(--mint-500), var(--teal-500)); color: white; border: none; border-radius: 16px; font-family: inherit; font-size: 1.05rem; font-weight: 600; cursor: pointer; transition: all 0.3s ease; box-shadow: 0 8px 30px rgba(34, 197, 94, 0.35); }
        .btn-primary:hover { transform: translateY(-3px); box-shadow: 0 12px 40px rgba(34, 197, 94, 0.45); }
        .btn-primary .arrow { transition: transform 0.3s ease; }
        .btn-primary:hover .arrow { transform: translateX(4px); }
        .btn-secondary { display: inline-flex; align-items: center; gap: 10px; padding: 18px 36px; background: white; color: var(--slate-700); border: 2px solid var(--slate-200); border-radius: 16px; font-family: inherit; font-size: 1.05rem; font-weight: 600; cursor: pointer; transition: all 0.3s ease; }
        .btn-secondary:hover { border-color: var(--mint-400); color: var(--mint-600); transform: translateY(-3px); box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08); }

        .hero-visual { flex: 1; display: flex; justify-content: center; align-items: center; position: relative; animation: fadeInRight 1s ease 0.4s forwards; opacity: 0; }
        @keyframes fadeInRight { from { opacity: 0; transform: translateX(50px); } to { opacity: 1; transform: translateX(0); } }
        .hero-cards { position: relative; width: 500px; height: 500px; }
        .floating-card { position: absolute; background: white; border-radius: 24px; padding: 24px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1); animation: cardFloat 6s ease-in-out infinite; border: 1px solid rgba(134, 239, 172, 0.2); }
        .floating-card-1 { top: 0; left: 50%; transform: translateX(-50%); }
        .floating-card-2 { top: 35%; left: 0; animation-delay: -2s; }
        .floating-card-3 { top: 35%; right: 0; animation-delay: -4s; }
        .floating-card-4 { bottom: 0; left: 50%; transform: translateX(-50%); animation-delay: -3s; }
        @keyframes cardFloat { 0%, 100% { transform: translateY(0) translateX(-50%); } 50% { transform: translateY(-15px) translateX(-50%); } }
        .floating-card-2, .floating-card-3 { animation-name: cardFloatSide; }
        @keyframes cardFloatSide { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-15px); } }
        .card-emoji { font-size: 2.5rem; margin-bottom: 12px; }
        .card-title { font-size: 1rem; font-weight: 700; color: var(--slate-800); margin-bottom: 4px; }
        .card-subtitle { font-size: 0.85rem; color: var(--slate-500); }
        .hero-stats { display: flex; gap: 50px; margin-top: 60px; animation: fadeInUp 0.8s ease 0.5s forwards; opacity: 0; }
        .stat-item { text-align: left; }
        .stat-value { font-size: 2.5rem; font-weight: 800; color: var(--mint-600); }
        .stat-label { font-size: 0.95rem; color: var(--slate-500); }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }

        .features { padding: 100px 60px; position: relative; }
        .section-header { text-align: center; max-width: 700px; margin: 0 auto 80px; }
        .section-badge { display: inline-flex; align-items: center; gap: 8px; padding: 8px 20px; background: rgba(134, 239, 172, 0.15); border-radius: 50px; font-size: 0.9rem; font-weight: 600; color: var(--mint-600); margin-bottom: 20px; }
        .section-header h2 { font-size: clamp(2rem, 4vw, 3rem); font-weight: 800; color: var(--slate-900); margin-bottom: 16px; }
        .section-header p { font-size: 1.15rem; color: var(--slate-500); line-height: 1.7; }
        .features-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 30px; max-width: 1400px; margin: 0 auto; }
        .feature-card { background: white; border-radius: 28px; padding: 40px 32px; border: 1px solid rgba(134, 239, 172, 0.15); transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1); position: relative; overflow: hidden; }
        .feature-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(90deg, var(--mint-400), var(--teal-400)); opacity: 0; transition: opacity 0.3s ease; }
        .feature-card:hover { transform: translateY(-8px); box-shadow: 0 30px 60px rgba(0, 0, 0, 0.1); border-color: rgba(134, 239, 172, 0.4); }
        .feature-card:hover::before { opacity: 1; }
        .feature-icon { width: 70px; height: 70px; background: linear-gradient(135deg, rgba(134, 239, 172, 0.2), rgba(45, 212, 191, 0.15)); border-radius: 20px; display: flex; align-items: center; justify-content: center; font-size: 2rem; margin-bottom: 24px; transition: transform 0.3s ease; }
        .feature-card:hover .feature-icon { transform: scale(1.1) rotate(5deg); }
        .feature-card h3 { font-size: 1.3rem; font-weight: 700; color: var(--slate-800); margin-bottom: 12px; }
        .feature-card p { font-size: 1rem; color: var(--slate-500); line-height: 1.7; }

        .how-it-works { padding: 100px 60px; background: linear-gradient(180deg, rgba(134, 239, 172, 0.05) 0%, transparent 100%); }
        .steps-container { display: flex; justify-content: center; gap: 60px; max-width: 1200px; margin: 0 auto; flex-wrap: wrap; }
        .step-card { flex: 1; min-width: 280px; max-width: 350px; text-align: center; position: relative; }
        .step-number { width: 60px; height: 60px; background: linear-gradient(135deg, var(--mint-500), var(--teal-500)); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; font-weight: 800; color: white; margin: 0 auto 24px; box-shadow: 0 10px 30px rgba(34, 197, 94, 0.3); }
        .step-card h3 { font-size: 1.25rem; font-weight: 700; color: var(--slate-800); margin-bottom: 12px; }
        .step-card p { font-size: 1rem; color: var(--slate-500); line-height: 1.6; }

        .team-section { padding: 100px 60px; text-align: center; }
        .team-card { display: inline-block; background: white; border-radius: 24px; padding: 40px 60px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.08); border: 1px solid rgba(134, 239, 172, 0.2); }
        .team-label { font-size: 0.85rem; font-weight: 600; color: var(--mint-600); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 16px; }
        .team-names { font-size: 1.5rem; font-weight: 700; color: var(--slate-800); margin-bottom: 12px; }
        .team-info { font-size: 1rem; color: var(--slate-500); }

        .cta-section { padding: 100px 60px; }
        .cta-box { max-width: 900px; margin: 0 auto; background: linear-gradient(135deg, var(--mint-500), var(--teal-500)); border-radius: 32px; padding: 80px 60px; text-align: center; position: relative; overflow: hidden; }
        .cta-box::before { content: ''; position: absolute; top: -50%; right: -50%; width: 100%; height: 100%; background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%); animation: ctaGlow 8s ease-in-out infinite; }
        @keyframes ctaGlow { 0%, 100% { transform: translate(0, 0); } 50% { transform: translate(-20px, 20px); } }
        .cta-box h2 { font-size: 2.5rem; font-weight: 800; color: white; margin-bottom: 16px; position: relative; z-index: 1; }
        .cta-box p { font-size: 1.15rem; color: rgba(255, 255, 255, 0.9); margin-bottom: 40px; position: relative; z-index: 1; }
        .cta-btn { display: inline-flex; align-items: center; gap: 12px; padding: 20px 44px; background: white; color: var(--mint-600); border: none; border-radius: 16px; font-family: inherit; font-size: 1.1rem; font-weight: 700; cursor: pointer; transition: all 0.3s ease; position: relative; z-index: 1; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2); }
        .cta-btn:hover { transform: translateY(-3px) scale(1.02); box-shadow: 0 15px 40px rgba(0, 0, 0, 0.25); }

        .footer { padding: 40px 60px; text-align: center; border-top: 1px solid rgba(134, 239, 172, 0.2); }
        .footer p { color: var(--slate-500); font-size: 0.95rem; }

        .app-page { display: none; min-height: 100vh; padding: 100px 40px 40px; }
        .app-page.active { display: block; }
        .app-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 40px; padding: 20px 30px; background: white; border-radius: 20px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05); border: 1px solid rgba(134, 239, 172, 0.15); }
        .back-btn { display: inline-flex; align-items: center; gap: 10px; padding: 14px 28px; background: rgba(134, 239, 172, 0.1); color: var(--mint-700); border: 1px solid rgba(134, 239, 172, 0.3); border-radius: 12px; font-family: inherit; font-size: 0.95rem; font-weight: 600; cursor: pointer; transition: all 0.3s ease; }
        .back-btn:hover { background: rgba(134, 239, 172, 0.2); transform: translateX(-4px); }
        .app-title { font-size: 1.5rem; font-weight: 700; color: var(--slate-800); }
        .app-grid { display: grid; grid-template-columns: 1fr 400px; gap: 30px; max-width: 1400px; margin: 0 auto; }
        .camera-section { background: white; border-radius: 24px; padding: 30px; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.06); border: 1px solid rgba(134, 239, 172, 0.15); }
        .camera-wrapper { position: relative; border-radius: 16px; overflow: hidden; background: var(--slate-100); margin-bottom: 30px; }
        #video, #displayCanvas { width: 100%; height: auto; display: block; min-height: 400px; background: var(--slate-900); }
        .camera-status { position: absolute; top: 16px; right: 16px; display: flex; align-items: center; gap: 8px; padding: 10px 20px; background: rgba(0, 0, 0, 0.6); backdrop-filter: blur(10px); border-radius: 50px; font-size: 0.85rem; font-weight: 500; color: white; }
        .status-dot { width: 8px; height: 8px; background: var(--slate-400); border-radius: 50%; transition: all 0.3s ease; }
        .status-dot.live { background: #ef4444; box-shadow: 0 0 10px #ef4444; animation: statusPulse 1.5s ease-in-out infinite; }
        @keyframes statusPulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .controls { display: flex; gap: 14px; justify-content: center; flex-wrap: wrap; margin-bottom: 30px; }
        .ctrl-btn { display: inline-flex; align-items: center; gap: 10px; padding: 16px 32px; font-family: inherit; font-size: 1rem; font-weight: 600; border: none; border-radius: 14px; cursor: pointer; transition: all 0.3s ease; }
        .ctrl-btn.start { background: linear-gradient(135deg, var(--mint-500), var(--teal-500)); color: white; box-shadow: 0 8px 25px rgba(34, 197, 94, 0.3); }
        .ctrl-btn.start:hover { transform: translateY(-3px); box-shadow: 0 12px 35px rgba(34, 197, 94, 0.4); }
        .ctrl-btn.stop { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; box-shadow: 0 8px 25px rgba(239, 68, 68, 0.3); }
        .ctrl-btn.stop:hover { transform: translateY(-3px); box-shadow: 0 12px 35px rgba(239, 68, 68, 0.4); }
        .ctrl-btn.clear { background: var(--slate-100); color: var(--slate-700); border: 1px solid var(--slate-200); }
        .ctrl-btn.clear:hover { background: var(--slate-200); transform: translateY(-3px); }
        .stats-row { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
        .stat-box { background: linear-gradient(135deg, rgba(134, 239, 172, 0.1), rgba(45, 212, 191, 0.05)); border: 1px solid rgba(134, 239, 172, 0.2); border-radius: 16px; padding: 24px; text-align: center; }
        .stat-box .value { font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, var(--mint-600), var(--teal-500)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
        .stat-box .label { font-size: 0.9rem; color: var(--slate-500); margin-top: 4px; }
        .output-section { display: flex; flex-direction: column; gap: 24px; }
        .prediction-card { background: white; border-radius: 24px; padding: 40px; text-align: center; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.06); border: 1px solid rgba(134, 239, 172, 0.15); position: relative; overflow: hidden; }
        .prediction-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; background: linear-gradient(90deg, var(--mint-400), var(--teal-400)); }
        .prediction-label { font-size: 0.8rem; font-weight: 600; color: var(--mint-600); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 20px; }
        .current-letter { font-size: 5rem; font-weight: 800; background: linear-gradient(135deg, var(--mint-500), var(--teal-500)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1; margin: 16px 0; }
        .confidence-text { font-size: 1rem; color: var(--slate-500); margin-bottom: 16px; }
        .confidence-bar { width: 100%; height: 8px; background: var(--slate-100); border-radius: 10px; overflow: hidden; }
        .confidence-fill { height: 100%; background: linear-gradient(90deg, var(--mint-400), var(--teal-400)); border-radius: 10px; transition: width 0.4s ease; }
        .sentence-card { flex: 1; background: white; border-radius: 24px; padding: 30px; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.06); border: 1px solid rgba(134, 239, 172, 0.15); display: flex; flex-direction: column; }
        .sentence-header { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; }
        .sentence-icon { font-size: 1.5rem; }
        .sentence-title { font-size: 0.8rem; font-weight: 600; color: var(--mint-600); text-transform: uppercase; letter-spacing: 2px; }
        .sentence-display { flex: 1; font-size: 1.4rem; font-weight: 500; color: var(--slate-800); line-height: 1.8; min-height: 100px; }
        .sentence-display.placeholder { color: var(--slate-400); font-style: italic; }
        .tts-btn { display: inline-flex; align-items: center; justify-content: center; gap: 10px; padding: 16px 32px; margin-top: 20px; background: linear-gradient(135deg, var(--mint-500), var(--teal-500)); color: white; border: none; border-radius: 14px; font-family: inherit; font-size: 1rem; font-weight: 600; cursor: pointer; transition: all 0.3s ease; box-shadow: 0 8px 25px rgba(34, 197, 94, 0.3); }
        .tts-btn:hover:not(:disabled) { transform: translateY(-3px); box-shadow: 0 12px 35px rgba(34, 197, 94, 0.4); }
        .tts-btn:disabled { opacity: 0.4; cursor: not-allowed; }

        @media (max-width: 1200px) { .app-grid { grid-template-columns: 1fr; } .hero { flex-direction: column; text-align: center; padding: 140px 40px 80px; } .hero-content { max-width: 100%; } .hero-buttons { justify-content: center; } .hero-stats { justify-content: center; } .hero-visual { margin-top: 60px; } .hero-cards { width: 350px; height: 400px; } }
        @media (max-width: 768px) { .navbar { padding: 15px 20px; } .nav-links { display: none; } .hero, .features, .how-it-works, .team-section, .cta-section { padding-left: 20px; padding-right: 20px; } .hero-buttons { flex-direction: column; } .btn-primary, .btn-secondary { width: 100%; justify-content: center; } .features-grid { grid-template-columns: 1fr; } .steps-container { flex-direction: column; align-items: center; } .controls { flex-direction: column; } .ctrl-btn { width: 100%; justify-content: center; } .stats-row { grid-template-columns: 1fr; } }
        .landing-page { display: block; }
        .landing-page.hidden { display: none; }
    </style>
</head>
<body>
    <div class="animated-bg"></div>
    <div class="blob-container"><div class="blob blob-1"></div><div class="blob blob-2"></div><div class="blob blob-3"></div><div class="blob blob-4"></div></div>
    <div class="particles"><div class="particle"></div><div class="particle"></div><div class="particle"></div><div class="particle"></div><div class="particle"></div><div class="particle"></div></div>

    <div id="landingPage" class="landing-page">
        <nav class="navbar" id="navbar">
            <div class="logo"><div class="logo-icon">🤟</div><span>SignSpeak</span></div>
            <div class="nav-links"><a href="#features">Features</a><a href="#how-it-works">How It Works</a><a href="#team">Team</a></div>
            <button class="nav-cta" onclick="showApp()">Try Now</button>
        </nav>

        <section class="hero">
            <div class="hero-content">
                <div class="hero-badge"><span class="hero-badge-dot"></span>AI-Powered Recognition</div>
                <h1>Translate <span class="highlight">Sign Language</span> to Text Instantly!</h1>
                <p class="hero-description">Break communication barriers with our cutting-edge AI. Transform hand gestures into text and speech in real-time — making sign language accessible to everyone.</p>
                <div class="hero-buttons">
                    <button class="btn-primary" onclick="showApp()"><span>Get Started</span><span class="arrow">→</span></button>
                    <button class="btn-secondary" onclick="document.getElementById('features').scrollIntoView({behavior: 'smooth'})"><span>Learn More</span></button>
                </div>
                <div class="hero-stats">
                    <div class="stat-item"><div class="stat-value">26+</div><div class="stat-label">ASL Letters</div></div>
                    <div class="stat-item"><div class="stat-value">Real-time</div><div class="stat-label">Detection</div></div>
                    <div class="stat-item"><div class="stat-value">97%+</div><div class="stat-label">Accuracy</div></div>
                </div>
            </div>
            <div class="hero-visual">
                <div class="hero-cards">
                    <div class="floating-card floating-card-1"><div class="card-emoji">🎥</div><div class="card-title">Real-Time Camera</div><div class="card-subtitle">Instant detection</div></div>
                    <div class="floating-card floating-card-2"><div class="card-emoji">🧠</div><div class="card-title">AI Model</div><div class="card-subtitle">Deep learning</div></div>
                    <div class="floating-card floating-card-3"><div class="card-emoji">🔊</div><div class="card-title">Text to Speech</div><div class="card-subtitle">Voice output</div></div>
                    <div class="floating-card floating-card-4"><div class="card-emoji">✍️</div><div class="card-title">Text Output</div><div class="card-subtitle">Build sentences</div></div>
                </div>
            </div>
        </section>

        <section class="features" id="features">
            <div class="section-header">
                <div class="section-badge">✨ Features</div>
                <h2>Everything You Need for Sign Language Translation</h2>
                <p>Our platform combines advanced AI with an intuitive interface to make sign language translation seamless and accessible.</p>
            </div>
            <div class="features-grid">
                <div class="feature-card"><div class="feature-icon">🎥</div><h3>Real-Time Detection</h3><p>Instant gesture recognition using your webcam with millisecond response time. No delays, just seamless translation.</p></div>
                <div class="feature-card"><div class="feature-icon">🧠</div><h3>AI-Powered Recognition</h3><p>Deep learning model trained on thousands of ASL samples for maximum accuracy and reliability.</p></div>
                <div class="feature-card"><div class="feature-icon">🔊</div><h3>Text to Speech</h3><p>Convert your translated text to natural speech with one click. Perfect for communication assistance.</p></div>
                <div class="feature-card"><div class="feature-icon">✍️</div><h3>Sentence Building</h3><p>Build complete sentences from detected character gestures. Includes space and delete support.</p></div>
                <div class="feature-card"><div class="feature-icon">👐</div><h3>Hand Tracking</h3><p>Advanced MediaPipe integration for precise hand landmark detection and visualization.</p></div>
                <div class="feature-card"><div class="feature-icon">⚡</div><h3>High Accuracy</h3><p>Confidence-based verification ensures only accurate predictions are registered. No false positives.</p></div>
            </div>
        </section>

        <section class="how-it-works" id="how-it-works">
            <div class="section-header">
                <div class="section-badge">🚀 How It Works</div>
                <h2>Three Simple Steps</h2>
                <p>Getting started is easy. Just follow these steps to begin translating sign language.</p>
            </div>
            <div class="steps-container">
                <div class="step-card"><div class="step-number">1</div><h3>Start Camera</h3><p>Click the start button to activate your webcam and begin the detection session.</p></div>
                <div class="step-card"><div class="step-number">2</div><h3>Make Gestures</h3><p>Show ASL hand signs to the camera. The AI will recognize and translate them in real-time.</p></div>
                <div class="step-card"><div class="step-number">3</div><h3>Get Output</h3><p>Watch as your gestures become text. Use text-to-speech to hear your message aloud.</p></div>
            </div>
        </section>

        <section class="team-section" id="team">
            <div class="section-header"><div class="section-badge">👥 Our Team</div><h2>Built with Passion</h2></div>
            <div class="team-card">
                <div class="team-label">Created By</div>
                <div class="team-names">Haroon, Saria & Azmeer</div>
                <div class="team-info">COMP-360 · Introduction to Artificial Intelligence<br>Forman Christian College</div>
            </div>
        </section>

        <section class="cta-section">
            <div class="cta-box">
                <h2>Ready to Break Barriers?</h2>
                <p>Start translating sign language to text and speech right now. No installation required.</p>
                <button class="cta-btn" onclick="showApp()"><span>Launch Application</span><span>→</span></button>
            </div>
        </section>

        <footer class="footer"><p>© 2025 SignSpeak · COMP-360 Project · Forman Christian College</p></footer>
    </div>

    <div id="appPage" class="app-page">
        <nav class="navbar">
            <div class="logo"><div class="logo-icon">🤟</div><span>SignSpeak</span></div>
            <div class="nav-links"></div>
            <button class="back-btn" onclick="showLanding()"><span>←</span><span>Back to Home</span></button>
        </nav>
        <div class="app-header"><div class="app-title">Detection Studio</div><div></div></div>
        <div class="app-grid">
            <div class="camera-section">
                <div class="camera-wrapper">
                    <video id="video" autoplay playsinline></video>
                    <canvas id="displayCanvas" style="display: none;"></canvas>
                    <div class="camera-status"><div class="status-dot" id="statusDot"></div><span id="statusText">Camera Off</span></div>
                </div>
                <div class="controls">
                    <button class="ctrl-btn start" onclick="startCamera()"><span>▶</span><span>Start Camera</span></button>
                    <button class="ctrl-btn stop" onclick="stopCamera()"><span>⏹</span><span>Stop</span></button>
                    <button class="ctrl-btn clear" onclick="clearSentence()"><span>🗑</span><span>Clear All</span></button>
                </div>
                <div class="stats-row">
                    <div class="stat-box"><div class="value" id="totalLetters">0</div><div class="label">Letters Detected</div></div>
                    <div class="stat-box"><div class="value" id="wordsCount">0</div><div class="label">Words Formed</div></div>
                </div>
            </div>
            <div class="output-section">
                <div class="prediction-card">
                    <div class="prediction-label">Current Gesture</div>
                    <div class="current-letter" id="currentLetter">—</div>
                    <div class="confidence-text" id="confidence">Confidence: 0%</div>
                    <div class="confidence-bar"><div class="confidence-fill" id="confidenceFill" style="width: 0%"></div></div>
                </div>
                <div class="sentence-card">
                    <div class="sentence-header"><span class="sentence-icon">✍️</span><span class="sentence-title">Generated Text</span></div>
                    <div class="sentence-display placeholder" id="sentenceDisplay">Start making gestures to build your message...</div>
                    <button class="tts-btn" id="ttsButton" onclick="speakText()" disabled><span>🔊</span><span>Speak Text</span></button>
                </div>
            </div>
        </div>
    </div>

    <script>
        window.addEventListener('scroll', () => { const navbar = document.getElementById('navbar'); if (window.scrollY > 50) { navbar.classList.add('scrolled'); } else { navbar.classList.remove('scrolled'); } });
        const video = document.getElementById('video');
        const displayCanvas = document.getElementById('displayCanvas');
        const displayCtx = displayCanvas.getContext('2d');
        let currentSentence = '', stream = null, predictionInterval = null, lastPrediction = '', predictionCount = 0, totalLettersDetected = 0, currentAudio = null, isPredicting = false, firstPredictionTime = null;
        const MIN_CONFIDENCE = 0.85, REQUIRED_CONSECUTIVE = 6, MIN_TIME_MS = 1000;

        function showApp() { document.getElementById('landingPage').classList.add('hidden'); document.getElementById('appPage').classList.add('active'); window.scrollTo(0, 0); }
        function showLanding() { document.getElementById('appPage').classList.remove('active'); document.getElementById('landingPage').classList.remove('hidden'); stopCamera(); if (currentAudio) { currentAudio.pause(); currentAudio = null; } window.scrollTo(0, 0); }

        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } } });
                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    displayCanvas.width = video.videoWidth; displayCanvas.height = video.videoHeight;
                    document.getElementById('statusText').textContent = 'Live'; document.getElementById('statusDot').classList.add('live');
                    video.style.display = 'none'; displayCanvas.style.display = 'block';
                    startPredictionLoop();
                };
            } catch (err) { alert('Error accessing camera: ' + err.message); }
        }

        function stopCamera() {
            if (predictionInterval) { clearInterval(predictionInterval); predictionInterval = null; }
            if (stream) { stream.getTracks().forEach(track => track.stop()); video.srcObject = null; stream = null; }
            video.style.display = 'block'; displayCanvas.style.display = 'none';
            document.getElementById('statusText').textContent = 'Camera Off'; document.getElementById('statusDot').classList.remove('live');
        }

        function clearSentence() {
            currentSentence = ''; lastPrediction = ''; predictionCount = 0; totalLettersDetected = 0; firstPredictionTime = null;
            const display = document.getElementById('sentenceDisplay'); display.textContent = 'Start making gestures to build your message...'; display.classList.add('placeholder');
            document.getElementById('totalLetters').textContent = '0'; document.getElementById('wordsCount').textContent = '0'; document.getElementById('ttsButton').disabled = true;
            if (currentAudio) { currentAudio.pause(); currentAudio = null; }
        }

        function updateStats() { document.getElementById('totalLetters').textContent = totalLettersDetected; const words = currentSentence.trim().split(/\\s+/).filter(w => w.length > 0).length; document.getElementById('wordsCount').textContent = words; document.getElementById('ttsButton').disabled = currentSentence.trim().length === 0; }

        function startPredictionLoop() {
            if (predictionInterval) clearInterval(predictionInterval);
            predictionInterval = setInterval(async () => {
                if (!stream || isPredicting) return;
                isPredicting = true;
                try {
                    const tempCanvas = document.createElement('canvas'); tempCanvas.width = video.videoWidth; tempCanvas.height = video.videoHeight;
                    const tempCtx = tempCanvas.getContext('2d'); tempCtx.drawImage(video, 0, 0);
                    const blob = await new Promise(resolve => tempCanvas.toBlob(resolve, 'image/jpeg', 0.8));
                    const formData = new FormData(); formData.append('image', blob, 'frame.jpg'); formData.append('draw_landmarks', 'true');
                    const response = await fetch('/predict', { method: 'POST', body: formData }); const data = await response.json();
                    if (data.image_with_landmarks) { const img = new Image(); img.onload = () => { displayCtx.drawImage(img, 0, 0, displayCanvas.width, displayCanvas.height); }; img.src = 'data:image/jpeg;base64,' + data.image_with_landmarks; }
                    if (data.prediction) {
                        let displayText = data.prediction; if (data.prediction === ' ') displayText = 'SPACE'; else if (data.prediction === 'DEL') displayText = 'DEL'; else if (data.prediction === 'NONE') displayText = 'NONE';
                        document.getElementById('currentLetter').textContent = displayText;
                        const confidencePercent = (data.confidence * 100).toFixed(1); document.getElementById('confidence').textContent = 'Confidence: ' + confidencePercent + '%'; document.getElementById('confidenceFill').style.width = confidencePercent + '%';
                        if (data.confidence >= MIN_CONFIDENCE) {
                            if (data.prediction === lastPrediction) { predictionCount++; if (firstPredictionTime === null) firstPredictionTime = Date.now(); const timeElapsed = Date.now() - firstPredictionTime;
                                if (predictionCount >= REQUIRED_CONSECUTIVE && timeElapsed >= MIN_TIME_MS) { if (data.prediction === 'DEL' || data.prediction === 'del') { if (currentSentence.length > 0) currentSentence = currentSentence.slice(0, -1); } else { currentSentence += data.prediction; }
                                    totalLettersDetected++; const display = document.getElementById('sentenceDisplay'); display.textContent = currentSentence || 'Start making gestures to build your message...'; display.classList.toggle('placeholder', !currentSentence); updateStats(); predictionCount = 0; lastPrediction = ''; firstPredictionTime = null; }
                            } else { lastPrediction = data.prediction; predictionCount = 1; firstPredictionTime = Date.now(); }
                        } else { if (lastPrediction !== '') { lastPrediction = ''; predictionCount = 0; firstPredictionTime = null; } }
                    } else if (data.error) { document.getElementById('currentLetter').textContent = '?'; document.getElementById('confidence').textContent = data.error; document.getElementById('confidenceFill').style.width = '0%'; }
                } catch (err) { console.error('Prediction error:', err); } finally { isPredicting = false; }
            }, 200);
        }

        async function speakText() {
            const text = currentSentence.trim(); if (!text) { alert('No text to speak!'); return; }
            const ttsButton = document.getElementById('ttsButton'); ttsButton.disabled = true; ttsButton.innerHTML = '<span>⏳</span><span>Loading...</span>';
            try {
                const response = await fetch('/text-to-speech', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: text }) }); const data = await response.json();
                if (data.success) { if (currentAudio) currentAudio.pause(); const audioBlob = base64ToBlob(data.audio, 'audio/mpeg'); const audioUrl = URL.createObjectURL(audioBlob); currentAudio = new Audio(audioUrl);
                    currentAudio.onended = () => { ttsButton.disabled = false; ttsButton.innerHTML = '<span>🔊</span><span>Speak Text</span>'; URL.revokeObjectURL(audioUrl); };
                    currentAudio.onerror = () => { alert('Error playing audio'); ttsButton.disabled = false; ttsButton.innerHTML = '<span>🔊</span><span>Speak Text</span>'; };
                    ttsButton.innerHTML = '<span>🔊</span><span>Playing...</span>'; await currentAudio.play();
                } else { alert('Error: ' + (data.error || 'Failed to generate speech')); ttsButton.disabled = false; ttsButton.innerHTML = '<span>🔊</span><span>Speak Text</span>'; }
            } catch (err) { console.error('TTS error:', err); alert('Failed to generate speech: ' + err.message); ttsButton.disabled = false; ttsButton.innerHTML = '<span>🔊</span><span>Speak Text</span>'; }
        }

        function base64ToBlob(base64, mimeType) { const byteCharacters = atob(base64); const byteNumbers = new Array(byteCharacters.length); for (let i = 0; i < byteCharacters.length; i++) { byteNumbers[i] = byteCharacters.charCodeAt(i); } const byteArray = new Uint8Array(byteNumbers); return new Blob([byteArray], { type: mimeType }); }
    </script>
</body>
</html>'''

    @app.route('/')
    def index():
        return render_template_string(HTML_TEMPLATE, available_models=list(models_dict.keys()), current_model=current_model_name)
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No image file selected'}), 400
            model_name = current_model_name
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return jsonify({'error': 'Invalid image format'}), 400
            height, width = image.shape[:2]
            if height > 480 or width > 640:
                scale = min(640/width, 480/height)
                image = cv2.resize(image, (int(width * scale), int(height * scale)))
            landmarks_array, hand_landmarks_obj = extract_landmarks_from_image(image)
            result = predict_gesture(landmarks_array, model_name)
            draw_landmarks = request.form.get('draw_landmarks', 'false').lower() == 'true'
            if draw_landmarks and hand_landmarks_obj is not None:
                image = draw_landmarks_on_image(image, hand_landmarks_obj)
            _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            result['image_with_landmarks'] = base64.b64encode(buffer).decode('utf-8')
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
    
    @app.route('/text-to-speech', methods=['POST'])
    def text_to_speech():
        try:
            data = request.get_json()
            text = data.get('text', '').strip()
            if not text:
                return jsonify({'error': 'No text provided'}), 400
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_filename = fp.name
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_filename)
            with open(temp_filename, 'rb') as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
            os.unlink(temp_filename)
            return jsonify({'success': True, 'audio': audio_base64, 'text': text})
        except Exception as e:
            return jsonify({'error': f'Text-to-speech error: {str(e)}'}), 500
    
    @app.route('/set_model', methods=['POST'])
    def set_model():
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
        return jsonify({'available_models': list(models_dict.keys()), 'current_model': current_model_name})
    
    @app.route('/health')
    def health_check():
        return jsonify({'status': 'healthy', 'models_loaded': len(models_dict), 'available_models': list(models_dict.keys())})
    
    @app.errorhandler(413)
    def too_large(e):
        return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Page not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({'error': 'Internal server error'}), 500
    
    def main_web_app():
        print("=" * 60)
        print("Sign Language Recognition - Web Application")
        print("=" * 60)
        print("Team: Haroon, Saria, Azmeer")
        print("Course: COMP-360 - Introduction to Artificial Intelligence")
        print("Institution: Forman Christian College")
        print("=" * 60)
        if not models_dict:
            print("No trained models found!")
            print("Please run train_model.py first to train the model.")
            return
        print(f"Web application initialized successfully!")
        print(f"Available models: {list(models_dict.keys())}")
        print(f"Current model: {current_model_name}")
        print(f"\nStarting Flask web server...")
        print(f"Open your browser and go to: http://localhost:5000")
        print(f"Press Ctrl+C to stop the server")
        print("=" * 60)
        app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    if WEB_APP_MODE and FLASK_AVAILABLE:
        try:
            main_web_app()
        except KeyboardInterrupt:
            print("\n\nWeb application stopped by user.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
    else:
        if WEB_APP_MODE and not FLASK_AVAILABLE:
            print("Flask not available. Running in camera mode instead.")
        run_camera_mode()