# Sign Language Recognition Project - Initial Implementation Review

**Review Date:** 2024  
**Reviewer:** AI Code Reviewer  
**Project Phase:** Initial Implementation & Debugging  
**Team:** Haroon, Saria, Azmeer  
**Course:** COMP-360 - Introduction to Artificial Intelligence

---

## Executive Summary

This review examines the **Initial Implementation & Debugging** phase of the Sign Language Recognition project. The project demonstrates a solid foundation with proper data preprocessing, model training, evaluation, and live prediction capabilities. However, several minor inconsistencies and potential runtime issues were identified that should be addressed before submission.

**Overall Status:** ⚠️ **Minor fixes needed before marking this phase complete.**

---

## 1. Structural Review

### 1.1 File Organization ✅
- **Status:** GOOD
- **Findings:**
  - All required scripts are present: `A2_preprocessing.py`, `A2_train_model.py`, `A2_evaluate_model_skeleton.py`, `A2_app_skeleton.py`
  - Directory structure is logical: `processed_data/`, `models/`, `plots/`
  - Supporting files (`A2_check_dataset.py`) are well-organized

### 1.2 Path References ⚠️
- **Status:** MOSTLY GOOD (with minor issues)
- **Findings:**
  - ✅ All scripts use `Path(__file__).parent.absolute()` for robust path resolution
  - ✅ Relative paths are correctly constructed using `Script_dir / "processed_data"`
  - ⚠️ **Issue:** Error messages reference `A1_preprocessing.py` but actual file is `A2_preprocessing.py`
    - **Location:** `A2_train_model.py:56`, `A2_evaluate_model_skeleton.py:45`
    - **Fix:** Update error messages to reference `A2_preprocessing.py`

### 1.3 Import Statements ✅
- **Status:** GOOD
- **Findings:**
  - All imports are present and correctly ordered
  - No circular dependencies detected
  - Required libraries: `numpy`, `tensorflow.keras`, `sklearn`, `cv2`, `mediapipe` are all imported

### 1.4 Model Input/Output Shapes ✅
- **Status:** CONSISTENT
- **Findings:**
  - Preprocessing outputs: `(N, 63)` feature vectors ✅
  - Model input: `(63,)` or `(None, 63)` ✅
  - Model output: `(26,)` one-hot encoded classes ✅
  - Label encoding: Consistent A-Z mapping (0-25) ✅

---

## 2. Functional Validation

### 2.1 Data Loading Pipeline ✅
- **Status:** GOOD
- **Findings:**
  - `A2_preprocessing.py` correctly:
    - Loads images from dataset
    - Extracts 21 landmarks × 3 coordinates = 63 features
    - One-hot encodes labels (26 classes)
    - Splits train/test (80/20) with stratification
    - Saves to `processed_data/` directory
  - Error handling is present for missing dataset

### 2.2 Model Training Pipeline ✅
- **Status:** GOOD (with minor issue)
- **Findings:**
  - `A2_train_model.py` correctly:
    - Loads preprocessed data
    - Performs comprehensive sanity checks
    - Builds MLP and Conv1D models as specified
    - Trains with correct hyperparameters (batch_size=64, epochs=20)
    - Saves models to `models/cnn_baseline.h5` and `models/cnn_last.h5`
  - ⚠️ **Issue:** Environment variable name inconsistency
    - **Current:** `Overfit = os.getenv('Overfit', '0')`
    - **Expected:** `OVERFIT_DEBUG = os.getenv('OVERFIT_DEBUG', '0')`
    - **Impact:** Users following documentation may use wrong env var name
    - **Fix:** Change to `OVERFIT_DEBUG` for consistency with requirements

### 2.3 Overfit Debug Mode ⚠️
- **Status:** FUNCTIONAL (with naming issue)
- **Findings:**
  - ✅ Correctly slices first 256 samples when enabled
  - ✅ Trains up to 100 epochs in debug mode
  - ✅ Checks for near-100% train accuracy
  - ✅ Warns if accuracy < 95%
  - ⚠️ **Issue:** Environment variable name should be `OVERFIT_DEBUG` not `Overfit`

### 2.4 Model Evaluation Pipeline ✅
- **Status:** GOOD
- **Findings:**
  - `A2_evaluate_model_skeleton.py` correctly:
    - Loads test data
    - Loads trained models (baseline and final)
    - Computes accuracy, confusion matrix, classification report
    - Plots confusion matrix with A-Z labels
    - Saves plots to `plots/` directory
  - All metrics are correctly calculated using sklearn

### 2.5 Live Prediction Pipeline ✅
- **Status:** GOOD (with minor issue)
- **Findings:**
  - `A2_app_skeleton.py` correctly:
    - Loads model when `Predictions = True`
    - Extracts MediaPipe landmarks per frame
    - Builds (63,) feature vector
    - Makes predictions and overlays on frame
    - Skips prediction if no hand detected
  - ⚠️ **Issue:** Missing `cv2.LINE_AA` parameter in `cv2.putText()` call
    - **Location:** Line 125
    - **Impact:** Minor - text rendering may be slightly less smooth
    - **Fix:** Add `cv2.LINE_AA` as last parameter

---

## 3. Debug & Fix Recommendations

### 3.1 Critical Issues (Must Fix)

#### Issue #1: File Name Reference Mismatch
- **Files:** `A2_train_model.py:56`, `A2_evaluate_model_skeleton.py:45`
- **Problem:** Error messages reference `A1_preprocessing.py` but file is `A2_preprocessing.py`
- **Fix:**
```python
# Current (line 56 in A2_train_model.py):
print("Please run A1_preprocessing.py first to generate processed data.")

# Should be:
print("Please run A2_preprocessing.py first to generate processed data.")
```

#### Issue #2: Duplicate Print Statement
- **File:** `A2_preprocessing.py:270`
- **Problem:** Print statement outside `if __name__ == "__main__"` block executes on import
- **Fix:** Remove line 270:
```python
# Delete this line (currently at line 270):
print("Preprocessing pipeline completed successfully!")
```

### 3.2 Minor Issues (Should Fix)

#### Issue #3: Environment Variable Naming
- **File:** `A2_train_model.py:29`
- **Problem:** Uses `Overfit` instead of `OVERFIT_DEBUG`
- **Fix:**
```python
# Current:
Overfit = os.getenv('Overfit', '0') == '1'

# Should be:
OVERFIT_DEBUG = os.getenv('OVERFIT_DEBUG', '0') == '1'
# Then update all references from 'Overfit' to 'OVERFIT_DEBUG'
```

#### Issue #4: Missing cv2.LINE_AA Parameter
- **File:** `A2_app_skeleton.py:125`
- **Problem:** Missing anti-aliasing parameter in text rendering
- **Fix:**
```python
# Current:
cv2.putText(
    img,
    text,
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2,
)

# Should be:
cv2.putText(
    img,
    text,
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2,
    cv2.LINE_AA  # Add this line
)
```

### 3.3 Code Quality Issues (Optional)

#### Issue #5: Variable Naming Consistency
- **Files:** Multiple
- **Problem:** Inconsistent use of `Alphabets` vs `ALPHABET` vs `CLASS_NAMES`
- **Current State:**
  - `A2_train_model.py`: Uses `Alphabets`
  - `A2_evaluate_model_skeleton.py`: Uses `Alphabets`
  - `A2_app_skeleton.py`: Uses `Alphabets`
  - `A2_evaluate_model.py`: Uses `ALPHABET`
- **Recommendation:** Standardize to `CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")` for consistency (as per requirements)

#### Issue #6: Print Statement Formatting
- **Files:** Multiple
- **Problem:** Inconsistent use of checkmarks/symbols in print statements
- **Current:** Mix of `✓`, `✗`, `⚠️`, and plain text
- **Recommendation:** Standardize to consistent format (either all symbols or all plain text)

---

## 4. Readability & Quality Check

### 4.1 Code Organization ✅
- **Status:** GOOD
- **Findings:**
  - Functions are logically organized
  - Clear separation of concerns
  - Appropriate use of constants at module level

### 4.2 Comments & Documentation ✅
- **Status:** GOOD
- **Findings:**
  - All files have proper header docstrings
  - Functions have docstrings explaining purpose
  - Inline comments are minimal but helpful
  - Team and course information is present

### 4.3 Error Handling ✅
- **Status:** GOOD
- **Findings:**
  - Try/except blocks are used appropriately
  - Meaningful error messages are provided
  - Graceful degradation (e.g., app runs without model if not found)

### 4.4 Beginner-Friendliness ✅
- **Status:** GOOD
- **Findings:**
  - No complex list comprehensions (as required)
  - Explicit loops are used
  - Variable names are descriptive
  - Code structure is clear and linear

---

## 5. Testing & Validation Checklist

### 5.1 Data Flow Validation

| Check | Status | Notes |
|-------|--------|-------|
| Preprocessing creates 63-dim features | ✅ | Correct: 21 landmarks × 3 coords |
| Labels are one-hot encoded (26 classes) | ✅ | Verified in sanity checks |
| Train/test split is 80/20 | ✅ | Confirmed in code |
| Data shapes are consistent | ✅ | All assertions pass |
| No NaN/Inf values | ✅ | Checked in sanity_checks() |

### 5.2 Model Training Validation

| Check | Status | Notes |
|-------|--------|-------|
| MLP model architecture matches spec | ✅ | Input(63) → Dense(128) → Dropout → Dense(64) → Dense(26) |
| Conv1D model architecture matches spec | ✅ | Reshape(21,3) → Conv1D → MaxPool → Flatten → Dense(64) → Dense(26) |
| Training uses correct hyperparameters | ✅ | batch_size=64, epochs=20, lr=1e-3 |
| ModelCheckpoint saves best model | ✅ | Monitors val_accuracy, saves to cnn_baseline.h5 |
| Final model saved as cnn_last.h5 | ✅ | Confirmed in code |
| Overfit debug mode works | ⚠️ | Functional but env var name inconsistent |

### 5.3 Evaluation Validation

| Check | Status | Notes |
|-------|--------|-------|
| Loads test data correctly | ✅ | From processed_data/X_test.npy |
| Loads trained models | ✅ | From models/cnn_baseline.h5 and cnn_last.h5 |
| Computes accuracy correctly | ✅ | Uses sklearn.accuracy_score |
| Generates confusion matrix | ✅ | With A-Z labels |
| Saves confusion matrix plot | ✅ | To plots/ directory |
| Prints classification report | ✅ | With class names A-Z |

### 5.4 Live Prediction Validation

| Check | Status | Notes |
|-------|--------|-------|
| Loads model when PREDICT_LIVE=True | ✅ | From models/cnn_baseline.h5 |
| Extracts landmarks correctly | ✅ | 21 landmarks × 3 coords = 63 features |
| Builds feature vector correctly | ✅ | Shape (1, 63) |
| Makes predictions | ✅ | Uses model.predict() |
| Overlays prediction on frame | ⚠️ | Missing cv2.LINE_AA parameter |
| Skips prediction if no hand | ✅ | Correctly implemented |
| Camera opens/closes safely | ✅ | Proper cleanup in finally block |

---

## 6. Final Checklist

| Check | Status | Notes / Fix Suggestions |
|-------|--------|------------------------|
| **Structural** |
| All scripts run independently | ✅ | No import errors detected |
| Path references are correct | ⚠️ | Fix: Update A1_ references to A2_ |
| Model shapes are consistent | ✅ | All shapes verified |
| Label encoding is consistent | ✅ | A-Z mapping is correct everywhere |
| **Functional** |
| Pipeline executes end-to-end | ✅ | Data → Train → Evaluate → Predict |
| Model checkpoints save correctly | ✅ | Verified in code |
| Overfit debug mode works | ⚠️ | Fix: Rename env var to OVERFIT_DEBUG |
| Evaluation computes metrics correctly | ✅ | All metrics verified |
| Live prediction runs safely | ⚠️ | Fix: Add cv2.LINE_AA parameter |
| **Code Quality** |
| Clear print statements | ✅ | Informative messages throughout |
| Proper error handling | ✅ | Try/except blocks present |
| Beginner-friendly code | ✅ | No list comprehensions, explicit loops |
| Constants defined properly | ⚠️ | Consider standardizing to CLASS_NAMES |
| **Documentation** |
| Functions have docstrings | ✅ | All functions documented |
| Header information present | ✅ | Team, course, institution listed |
| Comments are helpful | ✅ | Appropriate level of commenting |

---

## 7. Recommended Fixes Summary

### Must Fix (Before Submission):
1. ✅ Update error messages: `A1_preprocessing.py` → `A2_preprocessing.py` (2 locations)
2. ✅ Remove duplicate print statement in `A2_preprocessing.py:270`

### Should Fix (For Best Practice):
3. ✅ Rename environment variable: `Overfit` → `OVERFIT_DEBUG` in `A2_train_model.py`
4. ✅ Add `cv2.LINE_AA` parameter in `A2_app_skeleton.py:125`

### Optional (Code Quality):
5. Consider standardizing variable names: `Alphabets` → `CLASS_NAMES` for consistency
6. Consider standardizing print statement formatting

---

## 8. Conclusion

### Strengths:
- ✅ Solid implementation of all required features
- ✅ Comprehensive sanity checks in training script
- ✅ Proper error handling throughout
- ✅ Clear code organization and documentation
- ✅ Consistent data shapes and label encoding
- ✅ All required models (MLP, Conv1D) implemented correctly

### Areas for Improvement:
- ⚠️ Minor naming inconsistencies (file references, env vars)
- ⚠️ One duplicate print statement
- ⚠️ Missing optional parameter in OpenCV call

### Final Verdict:

**⚠️ Minor fixes needed before marking this phase complete.**

The project is **functionally complete** and demonstrates good understanding of the pipeline. The identified issues are minor and can be fixed quickly. After addressing the "Must Fix" items, the project will be ready for submission.

---

## 9. Quick Fix Guide

### Fix #1: Update File References
```bash
# In A2_train_model.py line 56:
sed -i 's/A1_preprocessing/A2_preprocessing/g' A2_train_model.py

# In A2_evaluate_model_skeleton.py line 45:
sed -i 's/A1_preprocessing/A2_preprocessing/g' A2_evaluate_model_skeleton.py
```

### Fix #2: Remove Duplicate Print
```python
# Delete line 270 in A2_preprocessing.py
# (the print statement outside if __name__ == "__main__")
```

### Fix #3: Update Environment Variable
```python
# In A2_train_model.py:
# Change line 29 from:
Overfit = os.getenv('Overfit', '0') == '1'
# To:
OVERFIT_DEBUG = os.getenv('OVERFIT_DEBUG', '0') == '1'
# Then update all references from 'Overfit' to 'OVERFIT_DEBUG' (lines 185, 192, 222, 259)
```

### Fix #4: Add cv2.LINE_AA
```python
# In A2_app_skeleton.py line 125, add cv2.LINE_AA as last parameter
cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
```

---

**Review Complete** ✅

