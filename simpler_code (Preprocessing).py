import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Path to test folder
DATA_DIR = 'archive/asl_alphabet_train/asl_alphabet_train'

# Storage for data and labels
data = []
labels = []

# Process each letter folder (A, B, C, etc.)
print("Processing images...")
for letter_folder in os.listdir(DATA_DIR):
    letter_path = os.path.join(DATA_DIR, letter_folder)
    
    # Skip if not a folder
    if not os.path.isdir(letter_path):
        continue
    
    # Process each image in the letter folder
    for img_path in os.listdir(letter_path):
        # Only process image files
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Read image
        img = cv2.imread(os.path.join(letter_path, img_path))
        if img is None:
            continue
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            data_aux = []
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract x and y coordinates
            x_coords = []
            y_coords = []
            for landmark in hand_landmarks.landmark:
                x_coords.append(landmark.x)
                y_coords.append(landmark.y)
            
            # Normalize coordinates (relative to min value)
            min_x = min(x_coords)
            min_y = min(y_coords)
            
            # Store normalized coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min_x)
                data_aux.append(y - min_y)
            
            # Label is the folder name (A, B, C, etc.)
            label = letter_folder
            
            data.append(data_aux)
            labels.append(label)
            print(f"Processed: {letter_folder}/{img_path} -> {label}")

# Save data
print(f"\nSaving {len(data)} samples...")
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

hands.close()
print("Done! Data saved to 'data.pickle'")
print(f"Data shape: {len(data)}")