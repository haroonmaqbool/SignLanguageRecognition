"""
Sign Language Recognition - Real-time Detection Module
====================================================

Project: Sign Language Recognition System
Course: Introduction to Artificial Intelligence (COMP-360)
Institution: Forman Christian College
Team: Haroon, Saria, Azmeer
Instructor: [Instructor Name]

Description:
This module provides real-time sign language recognition using webcam input.
It captures video frames, extracts hand landmarks using MediaPipe, and predicts
sign language gestures using trained CNN/LSTM models. The system displays
real-time predictions with confidence scores and visual feedback.

Features:
- Real-time webcam capture using OpenCV
- Hand landmark extraction with MediaPipe
- Live prediction using trained models
- Confidence score display
- Visual feedback with bounding boxes
- Keyboard controls for model switching
- Performance monitoring

Requirements:
- OpenCV, MediaPipe, NumPy
- TensorFlow/Keras for model loading
- Trained models from train_model.py

Author: AI Coding Assistant
Date: 2024
"""

# Step 1 - Import Required Libraries
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
import os

class SignLanguageDetector:
    """
    Real-time sign language detection class.
    """
    
    def __init__(self, model_path, model_type="CNN"):
        """
        Initialize the sign language detector.
        
        Args:
            model_path (str): Path to the trained model
            model_type (str): Type of model ("CNN" or "LSTM")
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """
        Load the trained model for prediction.
        """
        print(f"📥 Loading {self.model_type} model from {self.model_path}...")
        
        try:
            self.model = load_model(self.model_path)
            print(f"✅ {self.model_type} model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def extract_landmarks(self, frame):
        """
        Extract hand landmarks from a frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            numpy.ndarray or None: Extracted landmarks or None if no hand detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Get the first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks, dtype=np.float32)
        
        return None
    
    def predict_gesture(self, landmarks):
        """
        Predict sign language gesture from landmarks.
        
        Args:
            landmarks: Hand landmarks array
            
        Returns:
            tuple: (predicted_class, confidence_score)
        """
        if self.model is None or landmarks is None:
            return None, 0.0
        
        try:
            # Reshape landmarks for model input
            landmarks_reshaped = landmarks.reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(landmarks_reshaped, verbose=0)
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            predicted_letter = self.alphabet[predicted_class_idx]
            
            return predicted_letter, confidence
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None, 0.0
    
    def draw_landmarks(self, frame, landmarks):
        """
        Draw hand landmarks on the frame.
        
        Args:
            frame: Input video frame
            landmarks: Hand landmarks
            
        Returns:
            numpy.ndarray: Frame with landmarks drawn
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Draw landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return frame
    
    def calculate_fps(self):
        """
        Calculate and update FPS.
        """
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run_detection(self):
        """
        Run real-time sign language detection.
        """
        print("🚀 Starting real-time sign language detection...")
        print("📋 Controls:")
        print("   • Press 'q' to quit")
        print("   • Press 's' to save current frame")
        print("   • Press 'h' to show/hide hand landmarks")
        print("   • Press 'c' to clear prediction history")
        print("=" * 50)
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Detection settings
        show_landmarks = True
        prediction_history = []
        max_history = 10
        
        # Detection parameters
        detection_confidence_threshold = 0.5
        smoothing_window = 5
        
        print("✅ Webcam initialized successfully!")
        print("🎥 Starting detection loop...")
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from webcam")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Calculate FPS
                self.calculate_fps()
                
                # Extract landmarks
                landmarks = self.extract_landmarks(frame)
                
                # Make prediction if landmarks are detected
                if landmarks is not None:
                    predicted_letter, confidence = self.predict_gesture(landmarks)
                    
                    if predicted_letter and confidence > detection_confidence_threshold:
                        # Add to prediction history for smoothing
                        prediction_history.append(predicted_letter)
                        if len(prediction_history) > max_history:
                            prediction_history.pop(0)
                        
                        # Get most frequent prediction in recent history
                        if len(prediction_history) >= smoothing_window:
                            from collections import Counter
                            most_common = Counter(prediction_history[-smoothing_window:]).most_common(1)[0]
                            final_prediction = most_common[0]
                        else:
                            final_prediction = predicted_letter
                        
                        # Draw prediction on frame
                        cv2.putText(frame, f"Prediction: {final_prediction}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Draw confidence bar
                        bar_width = int(confidence * 200)
                        cv2.rectangle(frame, (10, 90), (10 + bar_width, 110), (0, 255, 0), -1)
                        cv2.rectangle(frame, (10, 90), (210, 110), (255, 255, 255), 2)
                    else:
                        cv2.putText(frame, "No hand detected or low confidence", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "No hand detected", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw hand landmarks if enabled
                if show_landmarks:
                    frame = self.draw_landmarks(frame, landmarks)
                
                # Draw FPS
                cv2.putText(frame, f"FPS: {self.current_fps}", 
                          (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw model info
                cv2.putText(frame, f"Model: {self.model_type}", 
                          (frame.shape[1] - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw instructions
                cv2.putText(frame, "Press 'q' to quit, 'h' for landmarks, 's' to save", 
                          (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Sign Language Recognition - Real-time Detection', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("🛑 Quitting detection...")
                    break
                elif key == ord('h'):
                    show_landmarks = not show_landmarks
                    print(f"🔧 Hand landmarks: {'ON' if show_landmarks else 'OFF'}")
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    filename = f"captured_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"💾 Frame saved as {filename}")
                elif key == ord('c'):
                    prediction_history.clear()
                    print("🧹 Prediction history cleared")
        
        except KeyboardInterrupt:
            print("\n⚠️  Detection interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            print("✅ Detection stopped. Resources cleaned up.")

def main():
    """
    Main function to run real-time sign language detection.
    """
    print("=" * 60)
    print("Sign Language Recognition - Real-time Detection")
    print("=" * 60)
    print("Team: Haroon, Saria, Azmeer")
    print("Course: COMP-360 - Introduction to Artificial Intelligence")
    print("Institution: Forman Christian College")
    print("=" * 60)
    
    # Step 1 - Check for trained models
    print("🔍 Checking for trained models...")
    
    model_files = {
        'CNN': 'models/cnn_final.h5',
        'LSTM': 'models/lstm_final.h5'
    }
    
    available_models = {}
    for model_type, model_path in model_files.items():
        if os.path.exists(model_path):
            available_models[model_type] = model_path
            print(f"✅ {model_type} model found: {model_path}")
        else:
            print(f"⚠️  {model_type} model not found: {model_path}")
    
    if not available_models:
        print("❌ No trained models found!")
        print("Please run train_model.py first to train the models.")
        return
    
    # Step 2 - Select model
    print(f"\n📋 Available models: {list(available_models.keys())}")
    
    if len(available_models) == 1:
        selected_model = list(available_models.keys())[0]
        print(f"🎯 Using {selected_model} model (only option available)")
    else:
        print("🎯 Please select a model:")
        for i, model_type in enumerate(available_models.keys(), 1):
            print(f"   {i}. {model_type}")
        
        try:
            choice = int(input("Enter your choice (1-{}): ".format(len(available_models))))
            selected_model = list(available_models.keys())[choice - 1]
        except (ValueError, IndexError):
            print("⚠️  Invalid choice. Using first available model.")
            selected_model = list(available_models.keys())[0]
    
    # Step 3 - Initialize detector
    print(f"\n🚀 Initializing {selected_model} detector...")
    detector = SignLanguageDetector(available_models[selected_model], selected_model)
    
    if detector.model is None:
        print("❌ Failed to initialize detector. Exiting...")
        return
    
    # Step 4 - Run detection
    print(f"\n🎥 Starting real-time detection with {selected_model} model...")
    detector.run_detection()
    
    print("\n" + "=" * 60)
    print("🎉 Real-time detection session completed!")
    print("=" * 60)

if __name__ == "__main__":
    """
    Execute the real-time detection when script is run directly.
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user.")
        print("Exiting gracefully...")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your setup and try again.")

