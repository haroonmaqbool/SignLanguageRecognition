## Sign Language Recognition System (COMP‑360 Project)
 
 > “Can we make a computer actually *read our hands*?”  
 
 **Course:** Introduction to Artificial Intelligence (COMP‑360)  
 **Institution:** Forman Christian College  
 **Team:** Haroon • Saria • Azmeer  
 
 **Idea in one line:**  
 Turn **ASL hand gestures** into **live text and speech** using **MediaPipe**, **CNN models**, and a custom **Flask web app**.
 
 We trained our models on the public **ASL Alphabet Dataset** (A–Z hand signs) from **Kaggle**, and then converted each image into hand‑landmark features using MediaPipe.

---

## What Our Project Can Do

| Feature | Description |
|---------|-------------|
| Real‑time ASL Detection | Reads your hand signs from a webcam and predicts the current letter. |
| Modern Web UI | Animated landing page and a “Real‑Time Detection Studio” for live use. |
| AI Models | CNN‑based models trained on ASL alphabet landmarks. |
| Sentence Builder | Stable predictions are appended to form full sentences. |
| Text‑to‑Speech | One‑click button to speak out the generated sentence using gTTS. |
| Hand Landmarks | MediaPipe landmarks drawn directly on the video feed for feedback. |

---

## Quick Demo — How It Feels to Use

1. Open the web app → a **landing page** with an animated hand (`🤟`) welcomes you.  
2. Click **“Try Now →”** → you enter the **Real‑Time Detection Studio**.  
3. Turn on your webcam → the app starts reading your hand signs letter by letter.  
4. The **current letter**, **confidence bar**, and **running sentence** update in real time.  
5. Hit **“Speak Text”** → your sentence is converted to **audio** using gTTS.  

> In simple words: you sign → our model predicts → the app writes it → and then speaks it.

---

## Tech Stack

- **Python 3**
- **Flask** – backend web framework
- **TensorFlow / Keras** – deep learning models (CNN)
- **MediaPipe Hands** – 3D hand landmark detection (21 points)
- **OpenCV** – image & video frame handling
- **NumPy, scikit‑learn** – data + evaluation
- **gTTS** – Google Text‑to‑Speech for audio output
- **HTML / CSS / Vanilla JS** – front‑end (all custom, no big CSS framework)

---

## Project Structure (High‑Level)

```text
SignLanguageRecognition-SLR/
├── app.py                 # Flask web app + real-time detection studio
├── preprocessing.py       # ASL dataset preprocessing & landmark extraction
├── train_model.py         # CNN model training
├── evaluate_model.py      # Model evaluation & plots
├── realtime_detection.py  # (Optional) standalone webcam script
├── models/                # Trained CNN models (.h5 files)
├── processed_data/        # Saved NumPy arrays (X_train, y_train, etc.)
├── plots/                 # Training curves & confusion matrices
├── reports/               # Classification reports
├── templates/
│   └── index.html         # Single-page UI (landing + studio)
├── requirements.txt
└── README.md
```

> Note: Some filenames (e.g. model names) may change as we experiment, but the overall structure stays the same.

---

## How to Run the Project

### 1️⃣ Set Up Environment

- Install **Python 3.8+**
- Make sure you have a **webcam** connected

```bash
# (Optional but recommended) create virtual environment
python -m venv slr_env

# Activate (Windows)
slr_env\Scripts\activate

# Activate (macOS / Linux)
source slr_env/bin/activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Prepare Data & Train Models (First Time Only)

```bash
# 1. Preprocess ASL dataset (download + landmarks + splits)
python preprocessing.py

# 2. Train CNN model(s)
python train_model.py

# 3. Evaluate and generate plots/reports
python evaluate_model.py
```

Make sure trained models (e.g. `cnn_baseline.h5`, `cnn_last.h5`) are inside the `models/` folder, because `app.py` expects them there.

### 4️⃣ Run the Web App

```bash
python app.py
```

Then open your browser and go to: `http://localhost:5000`

---

## How It Works (Short Version)

- **Step 1 – Detect the Hand**  
  We use **MediaPipe Hands** to detect a single hand and extract **21 landmarks** `(x, y, z)` → flattened into a **63‑dimensional vector**.

- **Step 2 – Normalize & Preprocess**  
  We normalize the landmarks and also **standardize left/right hands** so the model sees a consistent representation.

- **Step 3 – CNN Prediction**  
  The 63‑D vector is passed to a trained **CNN classifier** that outputs probabilities over **26 classes (A–Z)**.

- **Step 4 – UI Logic**  
  In `app.py`, we:
  - Capture frames from the webcam in the browser
  - Send each frame to `/predict` (Flask route)
  - Draw **landmarks** on top of the image on the server side
  - Send back both **prediction** and **image_with_landmarks** (base64)

- **Step 5 – Sentence + Speech**  
  - The front‑end adds stable predictions (seen multiple times) to a running **sentence**  
  - A separate `/text-to-speech` route uses **gTTS** to generate an **MP3** and returns it as base64  
  - The browser plays it directly without saving any files manually

---

## Web App Overview (What We Built in `app.py`)

- **Landing Page**
  - Big animated **🤟 hand icon**
  - Soft green **particle background** and **grid animation**
  - Our team & course info displayed
  - Four feature cards: Real‑time, AI‑powered, Text Generation, High Accuracy

- **Real‑Time Detection Studio**
  - Live camera feed with **status badge** (`📷 Camera Off` / `🔴 Live`)
  - **Model Selector** dropdown (e.g. `CNN`, `CNN_LAST`)
  - Controls: **Start**, **Stop**, **Clear**
  - Stats: **Letters Detected**, **Words Formed**
  - **Current Gesture** card:
    - Big letter
    - Confidence percentage
    - Animated progress bar
  - **Generated Text** card:
    - Running sentence from your signs
    - **“Speak Text”** button for TTS

This whole UI is rendered from a single `index.html` file that `app.py` creates in the `templates/` folder if it doesn’t exist.

---

## Model & Dataset Details

- **Dataset**
   - **Name:** ASL Alphabet Dataset (Kaggle)
   - **Classes:** 26 letters (A–Z), each represented by hand‑gesture images
   - Each image is resized and passed through MediaPipe to extract landmarks
  - Data saved as `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`

- **Model**
  - Input: 63‑D landmark vector
  - Architecture:
    - 1D convolution layers + BatchNorm + Dropout
    - Global pooling
    - Dense layers
    - Softmax over 26 classes

- **Evaluation**
  - Accuracy, Precision, Recall, F1‑score
  - Confusion matrices for each model
  - Training curves (loss & accuracy)

All plots and reports are saved under `plots/` and `reports/`.

---

## Common Issues & Fixes

- **“No trained models found!” in console**
  - Make sure you ran `train_model.py`
  - Check that `models/cnn_baseline.h5` (or similar) actually exists

- **Webcam not accessible in the browser**
  - Allow camera permissions for `http://localhost:5000`
  - Close other apps using the camera (Zoom, Teams, etc.)

- **Slow performance**
  - Use a smaller webcam resolution
  - Close extra programs
  - (Optional) Use a machine with a GPU for training

---

## Team

- **Haroon** – Model integration, backend logic, real‑time prediction loop  
- **Saria** – Dataset preprocessing, experiments, evaluation & reports  
- **Azmeer** – Front‑end UI/UX, text‑to‑speech integration, overall polishing  

*(Roles are approximate; we all helped each other out when things broke.)*

---

## Note

This project was built **for educational purposes** as part of **COMP‑360 (Introduction to Artificial Intelligence)** at **Forman Christian College**.  
