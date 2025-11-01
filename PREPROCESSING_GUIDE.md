# Data Preprocessing Guide - Sign Language Recognition

## 📋 Overview

This guide explains the **Data Pre-processing Pipeline** for your Sign Language Recognition project. The preprocessing step is crucial as it prepares your raw images for model training.

## 🎯 What is Data Preprocessing?

Data preprocessing transforms raw images into a format that your deep learning models can understand. For sign language recognition, this involves:
1. **Loading images** from the dataset
2. **Extracting hand landmarks** using MediaPipe
3. **Converting to numerical format** (NumPy arrays)
4. **Splitting into train/test sets**
5. **Normalizing and encoding labels**

## 🔄 Preprocessing Pipeline Steps

### Step 1: Dataset Loading
- **Input**: Raw images from `archive/asl_alphabet_train/asl_alphabet_train/`
- **Structure**: Each letter (A-Z) has its own folder with ~3000 images
- **Task**: Find and verify the dataset location

### Step 2: Initialize MediaPipe
- **Purpose**: Extract hand landmarks (21 points per hand)
- **Each landmark**: 3 coordinates (x, y, z) = 63 features per image
- **Why MediaPipe**: Robust hand detection regardless of lighting/angle

### Step 3: Image Processing
For each image:
1. **Load image** using OpenCV (`cv2.imread`)
2. **Resize** to 128×128 pixels (standardization)
3. **Convert** BGR → RGB (MediaPipe requirement)
4. **Extract landmarks** using MediaPipe Hands

### Step 4: Feature Extraction
- Extract 21 landmarks × 3 coordinates = **63 features per image**
- Store as flat array: `[x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]`
- If no hand detected, skip the image

### Step 5: Label Encoding
- Map letters A-Z to numbers 0-25
- One-hot encode for multi-class classification
- Example: A → [1,0,0,...,0], B → [0,1,0,...,0]

### Step 6: Train/Test Split
- **80% training**, **20% testing**
- Stratified split (maintains class distribution)
- Random seed for reproducibility (42)

### Step 7: Save Processed Data
- Save as NumPy arrays (.npy files)
- Files created:
  - `X_train.npy` - Training features
  - `X_test.npy` - Test features  
  - `y_train.npy` - Training labels (one-hot)
  - `y_test.npy` - Test labels (one-hot)

## 📊 Data Flow Diagram

```
Raw Images (JPG files)
    ↓
[Load & Resize] → 128×128 RGB images
    ↓
[MediaPipe Extraction] → 21 landmarks × 3 coords = 63 features
    ↓
[Feature Array] → NumPy array (N samples × 63 features)
    ↓
[Label Encoding] → One-hot encoded (N samples × 26 classes)
    ↓
[Train/Test Split] → 80% train, 20% test
    ↓
[Save as .npy] → processed_data/ directory
```

## 🔍 Understanding the Output

After preprocessing, you'll have:
- **X_train**: Shape `(N_train, 63)` - Training features
- **X_test**: Shape `(N_test, 63)` - Test features
- **y_train**: Shape `(N_train, 26)` - Training labels (one-hot)
- **y_test**: Shape `(N_test, 26)` - Test labels (one-hot)

Where:
- `N_train` ≈ 80% of total images
- `N_test` ≈ 20% of total images
- 63 = 21 landmarks × 3 coordinates
- 26 = Number of classes (A-Z)

## 🛠️ Current Implementation Status

Your `preprocessing.py` already includes:
✅ Complete MediaPipe initialization
✅ Dataset path detection
✅ Image loading and processing
✅ Landmark extraction
✅ Train/test splitting
✅ Data saving functionality
✅ Progress tracking
✅ Error handling

## 📝 Key Functions in preprocessing.py

1. **`main()`**: Orchestrates the entire pipeline
2. **MediaPipe Hands**: Extracts hand landmarks
3. **Image processing loop**: Processes all images
4. **`train_test_split()`**: Splits data randomly
5. **`to_categorical()`**: One-hot encodes labels
6. **`np.save()`**: Saves processed arrays

## 🚀 Running the Preprocessing

```bash
# Navigate to project directory
cd Sign-Language-Recognition

# Run preprocessing
python preprocessing.py
```

**Expected Output:**
- Progress updates for each letter (A-Z)
- Processing statistics
- Saved files in `processed_data/` folder

## ⚠️ Common Issues & Solutions

### Issue 1: "Dataset not found"
**Solution**: Check your dataset path matches one in `possible_paths` list (line 70-77)

### Issue 2: "No hand detected"
**Solution**: Some images might not have visible hands. The script skips these automatically.

### Issue 3: Processing takes too long
**Solution**: This is normal! Processing 78,000 images takes time. Be patient or process a subset for testing.

## 📚 Next Steps After Preprocessing

1. **Verify processed data**:
   ```python
   import numpy as np
   X_train = np.load('processed_data/X_train.npy')
   print(f"Shape: {X_train.shape}")
   ```

2. **Train models** using `train_model.py`

3. **Evaluate models** using `evaluate_model.py`

## 💡 Tips for Better Preprocessing

1. **Data Quality**: Ensure images have clear hand gestures
2. **Landmark Quality**: MediaPipe works best with:
   - Good lighting
   - Clear background
   - Full hand visible
3. **Memory Management**: Process in batches for very large datasets
4. **Progress Tracking**: Monitor processing to catch errors early

---

**Note**: This preprocessing pipeline is specifically designed for static images. For real-time detection, see `realtime_detection.py` which processes video frames differently.



