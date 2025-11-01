# Real-time Detection - Skeleton Code
# Course: COMP-360
# Team: Haroon, Saria, Azmeer

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

class SignLanguageDetector:
    
    def __init__(self, model_path, model_type="CNN"):
        # Initialize detector with model
        
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        # TODO: Load model using load_model()
        
        # TODO: Initialize MediaPipe Hands
        # self.mp_hands = mp.solutions.hands
        # self.hands = self.mp_hands.Hands(...)
        
        print(f"Detector ready with {model_type}")
    
    def extract_landmarks(self, frame):
        # Extract landmarks from video frame
        
        # TODO: Convert BGR to RGB (MediaPipe needs RGB)
        
        # TODO: Process frame with MediaPipe
        
        # TODO: Extract landmarks if hand detected
        # Return array of landmarks, or None
        
        return None
    
    def predict(self, landmarks):
        # Predict letter from landmarks
        
        # TODO: Reshape landmarks for model input
        
        # TODO: Make prediction
        
        # TODO: Get predicted class and confidence
        # Use np.argmax() and np.max()
        
        # TODO: Return letter and confidence
        
        return "A", 0.95  # placeholder
    
    def run(self):
        # Main loop for real-time detection
        
        # TODO: Initialize webcam
        # cap = cv2.VideoCapture(0)
        
        print("Press 'q' to quit")
        
        # TODO: Main loop
        # while True:
        #     ret, frame = cap.read()
        #     if not ret:
        #         break
        #     
        #     # Flip frame (mirror effect)
        #     frame = cv2.flip(frame, 1)
        #     
        #     # Extract landmarks
        #     landmarks = self.extract_landmarks(frame)
        #     
        #     # Make prediction if hand detected
        #     if landmarks is not None:
        #         letter, confidence = self.predict(landmarks)
        #         # Draw prediction on frame using cv2.putText()
        #     
        #     # Show frame
        #     cv2.imshow('Sign Language Detection', frame)
        #     
        #     # Quit on 'q' key
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        
        # TODO: Release webcam and close windows
        # cap.release()
        # cv2.destroyAllWindows()
        
        print("Stopped")

def main():
    print("Real-time Sign Language Detection")
    
    # TODO: Set model path
    # model_path = "models/cnn_final.h5"
    
    # TODO: Create detector and run
    # detector = SignLanguageDetector(model_path, "CNN")
    # detector.run()

if __name__ == "__main__":
    main()
