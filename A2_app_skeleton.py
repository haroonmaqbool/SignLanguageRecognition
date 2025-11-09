"""
======================================================
Sign Language Recognition - Camera Detection
======================================================
This script implements the camera detection for sign language recognition.
Team: Haroon, Saria, Azmeer
Course: COMP-360 - Introduction to Artificial Intelligence
Institution: Forman Christian College
"""
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from pathlib import Path

# Constants
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
PREDICT_LIVE = True # Set to True to enable live predictions

# Get script directory
SCRIPT_DIR = Path(__file__).parent.absolute()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize model (only if PREDICT_LIVE is True)
model = None
if PREDICT_LIVE:
    try:
        model_path = SCRIPT_DIR / "models" / "cnn_baseline.h5"
        if model_path.exists():
            print(f"Loading model from {model_path}...")
            model = load_model(str(model_path))
            print("✓ Model loaded successfully!")
        else:
            print(f"⚠️  Model not found: {model_path}")
            print("   Please run train_model.py first to train the model.")
            print("   Running without predictions (hand tracking only).")
            PREDICT_LIVE = False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("   Running without predictions (hand tracking only).")
        PREDICT_LIVE = False

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Initialize webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("✗ Error: Could not open webcam.")
    exit(1)

print("\n" + "=" * 60)
print("Sign Language Recognition - Live Camera")
print("=" * 60)
print("Press 'q' to quit")
if PREDICT_LIVE:
    print("Live prediction: ENABLED")
else:
    print("Live prediction: DISABLED (set PREDICT_LIVE = True to enable)")
print("=" * 60 + "\n")

while webcam.isOpened():
    success, img = webcam.read()
    
    if not success:
        print("⚠️  Failed to read frame from webcam.")
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
        if PREDICT_LIVE and model is not None:
            # Extract landmarks from first hand
            first_hand = results.multi_hand_landmarks[0]
            
            # Build feature vector (63 dimensions: 21 landmarks × 3 coordinates)
            landmarks_array = np.zeros(63)
            idx = 0
            for landmark in first_hand.landmark:
                landmarks_array[idx] = landmark.x
                landmarks_array[idx + 1] = landmark.y
                landmarks_array[idx + 2] = landmark.z
                idx += 3
            
            # Reshape for model input: (1, 63)
            landmarks_array = landmarks_array.reshape(1, 63)
            
            # Make prediction
            preds = model.predict(landmarks_array, verbose=0)
            predicted_class_idx = np.argmax(preds, axis=1)[0]
            confidence = preds[0][predicted_class_idx]
            predicted_letter = CLASS_NAMES[predicted_class_idx]
            
            # Overlay prediction on frame
            text = f"Predicted: {predicted_letter} ({confidence:.2f})"
            cv2.putText(
                img,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
    else:
        # No hand detected
        if PREDICT_LIVE:
            cv2.putText(
                img,
                "No hand detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )
    
    # Display frame
    cv2.imshow("Webcam - Sign Language Recognition", img)
    
    # Exit on 'q' key press
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

# Cleanup
webcam.release()
cv2.destroyAllWindows()
print("\n✓ Webcam released. Exiting...")
