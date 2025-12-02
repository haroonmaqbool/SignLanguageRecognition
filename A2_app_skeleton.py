"""
======================================================
Sign Language Recognition - Camera Detection
======================================================
This script implements the camera detection for sign language recognition.
Team: Haroon, Saria, Azmeer
Course: COMP-360 - Introduction to Artificial Intelligence
"""
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from pathlib import Path

# Constants
Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
Predictions = True # Set to True to enable live predictions

def is_space_gesture(landmarks):
    """
    Detect if hand gesture is a space gesture (open hand with all fingers extended).
    
    Args:
        landmarks: Array of 63 values (21 landmarks × 3 coordinates)
    
    Returns:
        Boolean: True if gesture appears to be space
    """
    # Extract y-coordinates (vertical position)
    # Lower y = higher on screen, higher y = lower on screen
    thumb_tip_y = landmarks[4 * 3 + 1]  # Index 4, y-coordinate
    thumb_ip_y = landmarks[3 * 3 + 1]
    
    index_tip_y = landmarks[8 * 3 + 1]
    index_pip_y = landmarks[6 * 3 + 1]
    
    middle_tip_y = landmarks[12 * 3 + 1]
    middle_pip_y = landmarks[10 * 3 + 1]
    
    ring_tip_y = landmarks[16 * 3 + 1]
    ring_pip_y = landmarks[14 * 3 + 1]
    
    pinky_tip_y = landmarks[20 * 3 + 1]
    pinky_pip_y = landmarks[18 * 3 + 1]
    
    # Check if all fingertips are extended (tip y < PIP y means tip is above PIP = extended)
    thumb_extended = thumb_tip_y < thumb_ip_y
    index_extended = index_tip_y < index_pip_y
    middle_extended = middle_tip_y < middle_pip_y
    ring_extended = ring_tip_y < ring_pip_y
    pinky_extended = pinky_tip_y < pinky_pip_y
    
    # Space gesture: All 5 fingers extended (or at least 4 out of 5)
    fingers_extended = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
    
    # If 4 or 5 fingers are extended, likely a space gesture
    return fingers_extended >= 4

# Get script directory
Script_dir = Path(__file__).parent.absolute()

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize model (only if Predictions is True)
model = None
if Predictions:
    try:
        model_path = Script_dir / "models" / "cnn_baseline.h5"
        if model_path.exists():
            print(f"Loading model from {model_path}")
            model = load_model(str(model_path))
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

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.3,  # Lowered to 0.3 for better A sign detection
    min_tracking_confidence=0.3,   # Lowered to 0.3 for better tracking
    model_complexity=1  # Higher complexity for better accuracy
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
            
            # Check if this is a space gesture BEFORE model prediction
            if is_space_gesture(landmarks_array[0]):
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
