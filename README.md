# Sign Language Recognition System

## 🎓 Project Overview

**Course:** Introduction to Artificial Intelligence (COMP-360)  
**Institution:** Forman Christian College  
**Team:** Haroon, Saria, Azmeer  
**Instructor:** [Instructor Name]

This project implements a comprehensive **Sign Language Recognition System** using Deep Learning and Computer Vision techniques. The system can recognize American Sign Language (ASL) alphabet gestures and convert them into text, with support for both image upload and real-time webcam detection.

## 🚀 Features

- **Deep Learning Models**: CNN and LSTM architectures for gesture classification
- **Hand Landmark Extraction**: Using MediaPipe for robust hand detection
- **Real-time Detection**: Live webcam-based sign language recognition
- **Web Application**: Flask-based interface for easy interaction
- **Model Evaluation**: Comprehensive performance analysis and visualization
- **Multi-model Support**: Switch between CNN and LSTM models
- **Confidence Scoring**: Detailed prediction confidence and top predictions

## 📁 Project Structure

```
Sign Language Recognition/
├── preprocessing.py          # Data preprocessing and landmark extraction
├── train_model.py           # CNN and LSTM model training
├── evaluate_model.py        # Model evaluation and visualization
├── realtime_detection.py    # Real-time webcam detection
├── app.py                   # Flask web application
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── models/                 # Trained model files
│   ├── cnn_final.h5
│   └── lstm_final.h5
├── processed_data/         # Preprocessed dataset
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
├── plots/                  # Generated visualizations
│   ├── cnn_training_history.png
│   ├── lstm_training_history.png
│   ├── cnn_confusion_matrix.png
│   ├── lstm_confusion_matrix.png
│   └── model_comparison.png
├── reports/                # Evaluation reports
│   ├── cnn_classification_report.txt
│   └── lstm_classification_report.txt
└── templates/              # Web application templates
    └── index.html
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.7 or higher
- Webcam (for real-time detection)
- At least 4GB RAM (8GB recommended)
- GPU support (optional, for faster training)

### Step 1: Clone/Download Project

```bash
# If using git
git clone [repository-url]
cd sign-language-recognition

# Or download and extract the project files
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv sign_lang_env

# Activate virtual environment
# On Windows:
sign_lang_env\Scripts\activate
# On macOS/Linux:
source sign_lang_env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Complete Pipeline

```bash
# 1. Preprocess the dataset
python preprocessing.py

# 2. Train the models
python train_model.py

# 3. Evaluate the models
python evaluate_model.py

# 4. Run real-time detection (optional)
python realtime_detection.py

# 5. Start the web application
python app.py
```

## 📊 Usage Guide

### 1. Data Preprocessing (`preprocessing.py`)

This module downloads the ASL alphabet dataset and extracts hand landmarks:

```bash
python preprocessing.py
```

**Features:**
- Downloads dataset from KaggleHub
- Extracts 21 hand landmarks per image
- Resizes images to 128×128 pixels
- Splits data into train/test sets (80/20)
- Saves processed data as NumPy arrays

### 2. Model Training (`train_model.py`)

Trains both CNN and LSTM models for sign language classification:

```bash
python train_model.py
```

**Features:**
- Builds 1D CNN model for spatial feature extraction
- Builds LSTM model for temporal sequence processing
- Implements data augmentation and regularization
- Saves best and final model versions
- Generates training history plots

### 3. Model Evaluation (`evaluate_model.py`)

Comprehensive evaluation of trained models:

```bash
python evaluate_model.py
```

**Features:**
- Loads trained models and test data
- Generates confusion matrices
- Creates performance comparison charts
- Produces detailed classification reports
- Saves evaluation results and visualizations

### 4. Real-time Detection (`realtime_detection.py`)

Live webcam-based sign language recognition:

```bash
python realtime_detection.py
```

**Features:**
- Real-time webcam capture
- Live hand landmark extraction
- Instant gesture prediction
- Confidence score display
- Keyboard controls for interaction

**Controls:**
- `q`: Quit detection
- `h`: Toggle hand landmarks
- `s`: Save current frame
- `c`: Clear prediction history

### 5. Web Application (`app.py`)

Flask-based web interface for image upload and prediction:

```bash
python app.py
```

**Features:**
- Image upload and prediction
- Model selection (CNN/LSTM)
- Hand landmark visualization
- Confidence score display
- Responsive web interface

**Access:** Open your browser and go to `http://localhost:5000`

## 🧠 Technical Details

### Model Architecture

**CNN Model:**
- Input: 63-dimensional hand landmarks (21 points × 3 coordinates)
- Conv1D layers with BatchNormalization and Dropout
- Global Average Pooling
- Dense layers with regularization
- Output: 26 classes (A-Z)

**LSTM Model:**
- Input: Reshaped landmarks (21, 3)
- LSTM layers with dropout
- Dense layers for classification
- Output: 26 classes (A-Z)

### Hand Landmark Extraction

- Uses MediaPipe Hands solution
- Extracts 21 hand landmarks per image
- Each landmark has (x, y, z) coordinates
- Robust to hand orientation and lighting

### Dataset

- **Source**: KaggleHub - ASL Alphabet Dataset
- **Classes**: 26 letters (A-Z)
- **Images**: Hand gesture photos
- **Preprocessing**: Resize to 128×128, landmark extraction

## 📈 Performance Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **Training History**: Loss and accuracy curves

## 🔧 Troubleshooting

### Common Issues

1. **"No trained models found"**
   - Run `python train_model.py` first
   - Ensure models are saved in `models/` directory

2. **"Dataset not found"**
   - Run `python preprocessing.py` first
   - Check internet connection for KaggleHub download

3. **Webcam not working**
   - Ensure webcam is connected and not used by other applications
   - Check camera permissions

4. **Memory errors during training**
   - Reduce batch size in `train_model.py`
   - Use smaller model architectures
   - Close other applications

### Performance Optimization

- **GPU Support**: Install TensorFlow with GPU support for faster training
- **Batch Size**: Adjust batch size based on available memory
- **Model Complexity**: Reduce model size for faster inference

## 📚 Dependencies

See `requirements.txt` for complete list of dependencies:

- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Computer vision operations
- **MediaPipe**: Hand landmark extraction
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning utilities
- **Matplotlib/Seaborn**: Visualization
- **Flask**: Web application framework

## 🎯 Future Enhancements

- **PSL Support**: Add Pakistani Sign Language gestures
- **Word Recognition**: Extend to full words and phrases
- **Mobile App**: Develop mobile application
- **Real-time Translation**: Add text-to-speech functionality
- **Gesture Recording**: Allow users to record custom gestures

## 📄 License

This project is developed for educational purposes as part of the COMP-360 course at Forman Christian College.

## 👥 Team

- **Haroon** - [Role/Contribution]
- **Saria** - [Role/Contribution]  
- **Azmeer** - [Role/Contribution]

## 📞 Support

For questions or issues, please contact the development team or refer to the course instructor.

---

**Note**: This project is designed for educational purposes and demonstrates the application of deep learning and computer vision techniques in sign language recognition.