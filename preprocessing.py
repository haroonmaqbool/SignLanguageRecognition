"""
Sign Language Recognition - Data Preprocessing Pipeline
======================================================

Project: Sign Language Recognition System
Course: Introduction to Artificial Intelligence (COMP-360)
Institution: Forman Christian College
Team: Haroon, Saria, Azmeer

Description:
This script implements the complete data preprocessing pipeline for a Sign Language 
Recognition system using Deep Learning (CNN/LSTM) and Computer Vision (MediaPipe).
The pipeline processes ASL alphabet images to extract hand landmarks and prepare
data for model training.

Features:
- Downloads ASL alphabet dataset from KaggleHub
- Extracts hand landmarks using MediaPipe
- Processes images (resize, color conversion)
- Splits data into train/test sets
- One-hot encodes labels
- Saves processed data as numpy arrays

Requirements:
- Python 3.7+
- OpenCV, MediaPipe, NumPy, Scikit-learn, TensorFlow/Keras
- KaggleHub for dataset access

Author: AI Coding Assistant
Date: 2024
"""

# Step 1 - Import Required Libraries
import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from pathlib import Path
import time

def main():
    """
    Main function to execute the complete data preprocessing pipeline.
    """
    print("=" * 60)
    print("Sign Language Recognition - Data Preprocessing Pipeline")
    print("=" * 60)
    print("Team: Haroon, Saria, Azmeer")
    print("Course: COMP-360 - Introduction to Artificial Intelligence")
    print("Institution: Forman Christian College")
    print("=" * 60)
    
    # Step 2 - Initialize MediaPipe Hands
    print("\n🔧 Initializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✅ MediaPipe Hands initialized successfully!")
    
    # Step 3 - Load Local Dataset
    print("\n📥 Loading ASL Alphabet Dataset from local directory...")
    
    # Common dataset directory names - adjust as needed
    possible_paths = [
        "archive/asl_alphabet_train/asl_alphabet_train",  # Your dataset location
        "asl_alphabet_train",
        "asl_alphabet_train/asl_alphabet_train", 
        "dataset/asl_alphabet_train",
        "data/asl_alphabet_train",
        "ASL_Alphabet_Dataset/asl_alphabet_train"
    ]
    
    train_path = None
    for path in possible_paths:
        if os.path.exists(path):
            train_path = path
            break
    
    if train_path is None:
        print("❌ Dataset not found in common locations!")
        print("📁 Please ensure your dataset is in one of these locations:")
        for path in possible_paths:
            print(f"   • {path}")
        print("\n💡 If your dataset is in a different location, please update the 'train_path' variable in the script.")
        print("📋 Expected structure: dataset_folder/asl_alphabet_train/A/, dataset_folder/asl_alphabet_train/B/, etc.")
        return
    
    print(f"✅ Dataset found at: {train_path}")
    
    # Verify the dataset structure
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    missing_letters = []
    for letter in alphabet:
        letter_path = os.path.join(train_path, letter)
        if not os.path.exists(letter_path):
            missing_letters.append(letter)
    
    if missing_letters:
        print(f"⚠️  Warning: Missing folders for letters: {missing_letters}")
        print("   The script will continue with available letters only.")
    else:
        print("✅ All letter folders (A-Z) found in dataset!")
    
    # Step 4 - Initialize Data Storage
    print("\n📊 Initializing data storage...")
    X = []  # Feature vectors (hand landmarks)
    y = []  # Labels (A-Z)
    
    # Define alphabet mapping (A-Z = 0-25)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label_mapping = {letter: idx for idx, letter in enumerate(alphabet)}
    
    print(f"✅ Data storage initialized for {len(alphabet)} classes")
    
    # Step 5 - Process Images and Extract Landmarks
    print("\n🖼️  Processing images and extracting hand landmarks...")
    print("This may take several minutes depending on dataset size...")
    
    start_time = time.time()
    processed_count = 0
    total_images = 0
    
    # Count total images first for progress tracking
    for letter in alphabet:
        letter_path = os.path.join(train_path, letter)
        if os.path.exists(letter_path):
            total_images += len([f for f in os.listdir(letter_path) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"📈 Total images to process: {total_images}")
    
    # Process each letter folder
    for letter_idx, letter in enumerate(alphabet):
        letter_path = os.path.join(train_path, letter)
        
        if not os.path.exists(letter_path):
            print(f"⚠️  Warning: Folder for letter '{letter}' not found, skipping...")
            continue
            
        print(f"\n📁 Processing letter '{letter}' ({letter_idx + 1}/{len(alphabet)})...")
        
        # Get all image files in the letter folder
        image_files = [f for f in os.listdir(letter_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        letter_processed = 0
        
        for img_file in image_files:
            img_path = os.path.join(letter_path, img_file)
            
            try:
                # Read image
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Resize image to 128x128
                image_resized = cv2.resize(image, (128, 128))
                
                # Convert BGR to RGB (MediaPipe expects RGB)
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = hands.process(image_rgb)
                
                # Extract hand landmarks
                if results.multi_hand_landmarks:
                    # Get the first (and only) hand
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Extract 21 landmarks (x, y, z coordinates)
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    
                    # Add to dataset
                    X.append(landmarks)
                    y.append(label_mapping[letter])
                    letter_processed += 1
                    processed_count += 1
                    
                    # Progress update every 100 images
                    if processed_count % 100 == 0:
                        elapsed_time = time.time() - start_time
                        progress = (processed_count / total_images) * 100
                        print(f"   📊 Progress: {processed_count}/{total_images} ({progress:.1f}%) - "
                              f"Time elapsed: {elapsed_time:.1f}s")
                
            except Exception as e:
                print(f"   ⚠️  Error processing {img_file}: {e}")
                continue
        
        print(f"   ✅ Processed {letter_processed} images for letter '{letter}'")
    
    # Close MediaPipe hands
    hands.close()
    
    # Step 6 - Convert to NumPy Arrays
    print(f"\n🔄 Converting data to NumPy arrays...")
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"✅ Data converted successfully!")
    print(f"   📊 Feature matrix shape: {X.shape}")
    print(f"   📊 Labels shape: {y.shape}")
    
    # Step 7 - One-Hot Encode Labels
    print(f"\n🏷️  One-hot encoding labels...")
    y_categorical = to_categorical(y, num_classes=26)
    
    print(f"✅ Labels one-hot encoded!")
    print(f"   📊 One-hot encoded labels shape: {y_categorical.shape}")
    
    # Step 8 - Split Data into Train/Test Sets
    print(f"\n✂️  Splitting data into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # Ensure balanced distribution
    )
    
    print(f"✅ Data split completed!")
    print(f"   📊 Training set: {X_train.shape[0]} samples")
    print(f"   📊 Test set: {X_test.shape[0]} samples")
    
    # Step 9 - Save Processed Data
    print(f"\n💾 Saving processed data...")
    
    # Create output directory if it doesn't exist
    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy arrays
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    print(f"✅ Data saved successfully in '{output_dir}' directory!")
    
    # Step 10 - Print Summary
    total_time = time.time() - start_time
    print(f"\n" + "=" * 60)
    print("🎉 PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"📊 Dataset Summary:")
    print(f"   • Total samples processed: {len(X)}")
    print(f"   • Feature dimensions: {X.shape[1]} (21 landmarks × 3 coordinates)")
    print(f"   • Number of classes: 26 (A-Z)")
    print(f"   • Training samples: {X_train.shape[0]}")
    print(f"   • Test samples: {X_test.shape[0]}")
    print(f"   • Processing time: {total_time:.2f} seconds")
    print(f"\n📁 Output Files:")
    print(f"   • X_train.npy - Training features")
    print(f"   • X_test.npy - Test features")
    print(f"   • y_train.npy - Training labels (one-hot encoded)")
    print(f"   • y_test.npy - Test labels (one-hot encoded)")
    
    print(f"\n🚀 Next Steps:")
    print(f"   • Use the processed data to train your CNN/LSTM model")
    print(f"   • The feature vectors contain 21 hand landmarks with x, y, z coordinates")
    print(f"   • Each landmark represents a specific point on the hand")
    print(f"   • Labels are one-hot encoded for multi-class classification")
    
    print(f"\n" + "=" * 60)
    print("✅ Preprocessing pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    """
    Execute the preprocessing pipeline when script is run directly.
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user.")
        print("Exiting gracefully...")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your setup and try again.")

