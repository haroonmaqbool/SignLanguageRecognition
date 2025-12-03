# A2 Files - Quick Reference for Viva

## 📋 File Overview (One-Liner Each)

| File | Purpose |
|------|---------|
| **A2_preprocessing.py** | Converts raw images → hand landmarks → NumPy arrays |
| **A2_train_model.py** | Trains MLP/Conv1D neural network on landmarks |
| **A2_evaluate_model_skeleton.py** | Tests model, generates confusion matrix & metrics |
| **A2_app_skeleton.py** | Real-time webcam sign language recognition |
| **A2_check_dataset.py** | Validates dataset structure before preprocessing |

---

## 🔄 Pipeline Flow

```
Dataset Check → Preprocessing → Training → Evaluation → Real-time App
```

---

## 📊 Key Numbers to Remember

- **21 landmarks** per hand (MediaPipe)
- **3 coordinates** per landmark (x, y, z)
- **63 features** total (21 × 3)
- **26 classes** (A-Z)
- **80/20 split** (train/test)
- **64 batch size** (training)
- **20 epochs** (normal training)

---

## 🎯 A2_preprocessing.py - Key Points

**What it does:**
- Loads images from A-Z folders
- Uses MediaPipe to extract 21 hand landmarks
- Converts to 63-dimensional feature vectors
- One-hot encodes labels (A=0, B=1, ..., Z=25)
- Splits 80/20 train/test
- Saves as NumPy arrays

**Output:**
- `X_train.npy`, `X_test.npy` (features)
- `y_train.npy`, `y_test.npy` (one-hot labels)

**Why MediaPipe?**
- Pre-trained hand detection
- Standardized landmarks
- Robust to lighting/background
- Small feature space (63 vs 49,152 pixels)

---

## 🧠 A2_train_model.py - Key Points

**Model Architecture (MLP):**
```
Input(63) → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dense(26, Softmax)
```

**Training Config:**
- Optimizer: Adam (lr=0.001)
- Loss: Categorical cross-entropy
- Batch size: 64
- Epochs: 20 (normal) or 100 (overfit debug)

**Overfit Debug Mode:**
- Uses only 256 samples
- Should achieve >95% training accuracy
- Verifies model can learn (checks for bugs)

**Output:**
- `cnn_baseline.h5` (best model)
- `cnn_last.h5` (final model)

**Why Dropout?** Prevents overfitting (30% neurons disabled randomly)

**Why Softmax?** Converts scores to probability distribution

---

## 📈 A2_evaluate_model_skeleton.py - Key Points

**Metrics Calculated:**
1. **Accuracy**: Overall correctness
2. **Confusion Matrix**: Which letters are confused
3. **Classification Report**: Precision, Recall, F1 per class

**Process:**
1. Load test data and trained model
2. Make predictions: `model.predict(X_test)`
3. Convert probabilities to classes: `np.argmax()`
4. Calculate metrics
5. Generate confusion matrix heatmap
6. Save plot to `plots/`

**Why Confusion Matrix?** Shows which letters are commonly mistaken (e.g., I vs J)

---

## 🎥 A2_app_skeleton.py - Key Points

**Real-time Flow:**
1. Capture frame from webcam
2. Convert BGR → RGB (MediaPipe needs RGB)
3. Extract hand landmarks
4. Flatten to 63 features
5. Predict using model
6. Display prediction on frame

**Key Differences from Preprocessing:**
- `static_image_mode=False` (optimized for video)
- Processes frames continuously
- Draws landmarks and predictions on screen

**Why BGR→RGB?** OpenCV uses BGR, MediaPipe uses RGB

---

## ✅ A2_check_dataset.py - Key Points

**What it checks:**
- Dataset folder exists
- All 26 letter folders (A-Z) present
- Image count per letter
- Warns if <20 letters found

**Why important?** Catches dataset issues before preprocessing

---

## 🔑 Important Concepts

### Hand Landmarks
- 21 standardized points on hand
- Normalized coordinates (0-1 range)
- Invariant to hand size, image resolution

### One-Hot Encoding
- A = [1,0,0,...,0]
- B = [0,1,0,...,0]
- Z = [0,0,0,...,1]
- Required for categorical cross-entropy loss

### Train/Test Split
- 80% training (model learns)
- 20% testing (model evaluated)
- Stratified (balanced classes)
- Random state=42 (reproducible)

### Model Training
- Forward pass: Input → Model → Prediction
- Loss calculation: Compare prediction vs truth
- Backpropagation: Update weights
- Epoch: One full pass through training data

---

## ❓ Common Viva Questions

**Q: Why 63 dimensions?**
A: 21 landmarks × 3 coordinates (x, y, z) = 63 features

**Q: Why use MediaPipe instead of raw pixels?**
A: Smaller (63 vs 49,152), robust to background/lighting, standardized representation

**Q: What if hand not detected?**
A: In preprocessing: skip image. In app: show "No hand detected"

**Q: What is overfit debug mode?**
A: Uses 256 samples, trains 100 epochs. Verifies model can memorize (checks for bugs)

**Q: Why categorical cross-entropy?**
A: Standard loss for multi-class classification, works with softmax

**Q: How handle different hand sizes?**
A: MediaPipe normalizes to 0-1 range (relative coordinates), invariant to size

**Q: MLP vs Conv1D?**
A: MLP treats features as flat vector. Conv1D reshapes to (21,3) to learn spatial patterns

---

## 📁 File Dependencies

```
A2_check_dataset.py
  ↓
A2_preprocessing.py → processed_data/*.npy
  ↓
A2_train_model.py → models/*.h5
  ↓
A2_evaluate_model_skeleton.py → plots/*.png
  ↓
A2_app_skeleton.py → Real-time predictions
```

---

## 🎓 Viva Tips

1. **Know the numbers**: 21 landmarks, 63 features, 26 classes
2. **Understand the flow**: Check → Preprocess → Train → Evaluate → Deploy
3. **Explain MediaPipe**: Why use it, what it does, how it helps
4. **Model architecture**: Be able to draw/explain MLP structure
5. **Evaluation metrics**: Accuracy, confusion matrix, precision/recall
6. **Real-time challenges**: Speed, frame processing, visual feedback

---

**Good luck!** 🚀



