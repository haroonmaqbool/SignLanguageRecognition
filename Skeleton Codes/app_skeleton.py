# Web Application - Skeleton Code
# Course: COMP-360
# Team: Haroon, Saria, Azmeer

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Create Flask app
app = Flask(__name__)

# Global variables
models = {}  # Store loaded models
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def load_models():
    # Load CNN and LSTM models
    
    # TODO: Load models from models folder
    # models['cnn'] = load_model("models/cnn_final.h5")
    # models['lstm'] = load_model("models/lstm_final.h5")
    
    # TODO: Initialize MediaPipe Hands for extracting landmarks
    # mp_hands = mp.solutions.hands
    # hands = mp_hands.Hands(...)
    
    pass

def extract_landmarks(image):
    # Extract hand landmarks from an image
    
    # TODO: Convert image to RGB (MediaPipe needs RGB)
    
    # TODO: Process with MediaPipe
    
    # TODO: Extract landmarks if hand detected
    # Return array of landmarks, or None if no hand
    
    return None

def predict_sign(image, model_type='cnn'):
    # Predict sign language letter from image
    
    # TODO: Extract landmarks using extract_landmarks()
    
    # TODO: If landmarks found, reshape for model input
    
    # TODO: Use model to predict
    # prediction = models[model_type].predict(landmarks)
    
    # TODO: Get predicted letter and confidence
    # predicted_class = np.argmax(prediction[0])
    # letter = alphabet[predicted_class]
    # confidence = np.max(prediction[0])
    
    # TODO: Return letter and confidence
    
    return "A", 0.95  # placeholder

@app.route('/')
def index():
    # Home page
    
    # TODO: Render index.html template
    # return render_template('index.html')
    
    return "Sign Language Recognition App"

@app.route('/predict', methods=['POST'])
def predict():
    # Handle image upload and make prediction
    
    # TODO: Get uploaded image file from request
    # file = request.files['image']
    
    # TODO: Read image using OpenCV
    # Read bytes, convert to numpy array, then cv2.imdecode()
    
    # TODO: Get model type (cnn or lstm) from request
    
    # TODO: Call predict_sign() to get prediction
    
    # TODO: Return JSON with prediction and confidence
    # return jsonify({'prediction': letter, 'confidence': confidence})
    
    return jsonify({"message": "Prediction endpoint"})

def main():
    # Start the Flask app
    
    load_models()
    print("Starting Flask app on http://localhost:5000")
    app.run(debug=True, port=5000)

if __name__ == "__main__":
    main()
