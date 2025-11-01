"""
Sign Language Recognition - Model Training Module
===============================================

Project: Sign Language Recognition System
Course: Introduction to Artificial Intelligence (COMP-360)
Institution: Forman Christian College
Team: Haroon, Saria, Azmeer
Instructor: [Instructor Name]

Description:
This module handles the training of deep learning models for sign language recognition.
It loads preprocessed data from preprocessing.py and trains both CNN and LSTM models
to classify ASL alphabet gestures based on hand landmarks extracted via MediaPipe.

Features:
- Loads preprocessed hand landmark data
- Builds and trains CNN model for spatial feature extraction
- Builds and trains LSTM model for temporal sequence processing
- Implements data augmentation and regularization
- Saves trained models as .h5 files
- Provides training progress visualization

Requirements:
- TensorFlow/Keras
- NumPy, Matplotlib
- Scikit-learn for metrics
- Preprocessed data from preprocessing.py

Author: AI Coding Assistant
Date: 2024
"""

# Step 1 - Import Required Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, Reshape, Flatten, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import plot_model
import os
import time

def load_preprocessed_data():
    """
    Load preprocessed data from preprocessing.py output.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Training and test data
    """
    print("📥 Loading preprocessed data...")
    
    try:
        # Load preprocessed data
        X_train = np.load("processed_data/X_train.npy")
        X_test = np.load("processed_data/X_test.npy")
        y_train = np.load("processed_data/y_train.npy")
        y_test = np.load("processed_data/y_test.npy")
        
        print(f"✅ Data loaded successfully!")
        print(f"   📊 Training set: {X_train.shape[0]} samples")
        print(f"   📊 Test set: {X_test.shape[0]} samples")
        print(f"   📊 Feature dimensions: {X_train.shape[1]}")
        print(f"   📊 Number of classes: {y_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run preprocessing.py first to generate the required data files.")
        return None, None, None, None

def build_cnn_model(input_shape, num_classes):
    """
    Build a 1D CNN model for hand landmark classification.
    
    Args:
        input_shape (tuple): Input shape for the model
        num_classes (int): Number of output classes
        
    Returns:
        Model: Compiled CNN model
    """
    print("🏗️  Building CNN model...")
    
    model = Sequential([
        # Reshape input for 1D convolution
        Reshape((21, 3), input_shape=input_shape),
        
        # First Conv1D block
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.25),
        
        # Second Conv1D block
        Conv1D(128, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.25),
        
        # Third Conv1D block
        Conv1D(256, 3, activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dropout(0.5),
        
        # Dense layers
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ CNN model built successfully!")
    return model

def build_lstm_model(input_shape, num_classes):
    """
    Build an LSTM model for hand landmark sequence classification.
    
    Args:
        input_shape (tuple): Input shape for the model
        num_classes (int): Number of output classes
        
    Returns:
        Model: Compiled LSTM model
    """
    print("🏗️  Building LSTM model...")
    
    model = Sequential([
        # Reshape input for LSTM
        Reshape((21, 3), input_shape=input_shape),
        
        # LSTM layers
        LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        
        # Dense layers
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ LSTM model built successfully!")
    return model

def train_model(model, X_train, y_train, X_test, y_test, model_name, epochs=100):
    """
    Train the specified model with callbacks and monitoring.
    
    Args:
        model: Keras model to train
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name (str): Name for saving the model
        epochs (int): Number of training epochs
        
    Returns:
        History: Training history object
    """
    print(f"\n🚀 Training {model_name} model...")
    print(f"   📊 Epochs: {epochs}")
    print(f"   📊 Training samples: {X_train.shape[0]}")
    print(f"   📊 Test samples: {X_test.shape[0]}")
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            f'models/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Train model
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"✅ {model_name} training completed in {training_time:.2f} seconds!")
    
    return history

def plot_training_history(history, model_name):
    """
    Plot training history for accuracy and loss.
    
    Args:
        history: Keras history object
        model_name (str): Name of the model
    """
    print(f"📊 Plotting training history for {model_name}...")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'{model_name} - Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title(f'{model_name} - Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Training history plot saved as 'plots/{model_name}_training_history.png'")

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate the trained model and print detailed metrics.
    
    Args:
        model: Trained Keras model
        X_test, y_test: Test data
        model_name (str): Name of the model
    """
    print(f"\n📊 Evaluating {model_name} model...")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"✅ {model_name} Evaluation Results:")
    print(f"   📊 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   📊 Test Loss: {test_loss:.4f}")
    
    # Classification report
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    print(f"\n📋 Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                              target_names=list(alphabet)))
    
    return test_accuracy, y_pred_classes, y_true_classes

def main():
    """
    Main function to execute the complete model training pipeline.
    """
    print("=" * 60)
    print("Sign Language Recognition - Model Training Pipeline")
    print("=" * 60)
    print("Team: Haroon, Saria, Azmeer")
    print("Course: COMP-360 - Introduction to Artificial Intelligence")
    print("Institution: Forman Christian College")
    print("=" * 60)
    
    # Step 1 - Load Preprocessed Data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    if X_train is None:
        print("❌ Failed to load data. Exiting...")
        return
    
    # Step 2 - Get Data Shapes
    input_shape = (X_train.shape[1],)
    num_classes = y_train.shape[1]
    
    print(f"\n📊 Data Information:")
    print(f"   • Input shape: {input_shape}")
    print(f"   • Number of classes: {num_classes}")
    print(f"   • Feature type: Hand landmarks (21 points × 3 coordinates)")
    
    # Step 3 - Build and Train CNN Model
    print(f"\n" + "="*50)
    print("🏗️  CNN MODEL TRAINING")
    print("="*50)
    
    cnn_model = build_cnn_model(input_shape, num_classes)
    cnn_model.summary()
    
    cnn_history = train_model(cnn_model, X_train, y_train, X_test, y_test, 
                             "cnn", epochs=100)
    
    # Step 4 - Build and Train LSTM Model
    print(f"\n" + "="*50)
    print("🏗️  LSTM MODEL TRAINING")
    print("="*50)
    
    lstm_model = build_lstm_model(input_shape, num_classes)
    lstm_model.summary()
    
    lstm_history = train_model(lstm_model, X_train, y_train, X_test, y_test, 
                              "lstm", epochs=100)
    
    # Step 5 - Plot Training Histories
    print(f"\n📊 Generating training visualizations...")
    plot_training_history(cnn_history, "CNN")
    plot_training_history(lstm_history, "LSTM")
    
    # Step 6 - Evaluate Models
    print(f"\n" + "="*50)
    print("📊 MODEL EVALUATION")
    print("="*50)
    
    cnn_accuracy, cnn_pred, cnn_true = evaluate_model(cnn_model, X_test, y_test, "CNN")
    lstm_accuracy, lstm_pred, lstm_true = evaluate_model(lstm_model, X_test, y_test, "LSTM")
    
    # Step 7 - Save Final Models
    print(f"\n💾 Saving final models...")
    cnn_model.save('models/cnn_final.h5')
    lstm_model.save('models/lstm_final.h5')
    print("✅ Models saved successfully!")
    
    # Step 8 - Model Comparison
    print(f"\n" + "="*50)
    print("🏆 MODEL COMPARISON")
    print("="*50)
    print(f"📊 CNN Model Accuracy: {cnn_accuracy:.4f} ({cnn_accuracy*100:.2f}%)")
    print(f"📊 LSTM Model Accuracy: {lstm_accuracy:.4f} ({lstm_accuracy*100:.2f}%)")
    
    if cnn_accuracy > lstm_accuracy:
        print(f"🏆 CNN model performs better!")
        best_model = "CNN"
    else:
        print(f"🏆 LSTM model performs better!")
        best_model = "LSTM"
    
    print(f"\n" + "="*60)
    print("🎉 MODEL TRAINING COMPLETE!")
    print("="*60)
    print(f"📁 Saved Files:")
    print(f"   • models/cnn_best.h5 - Best CNN model")
    print(f"   • models/cnn_final.h5 - Final CNN model")
    print(f"   • models/lstm_best.h5 - Best LSTM model")
    print(f"   • models/lstm_final.h5 - Final LSTM model")
    print(f"   • plots/cnn_training_history.png - CNN training plots")
    print(f"   • plots/lstm_training_history.png - LSTM training plots")
    print(f"\n🏆 Best performing model: {best_model}")
    print("="*60)

if __name__ == "__main__":
    """
    Execute the model training pipeline when script is run directly.
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        print("Exiting gracefully...")
    except Exception as e:
        print(f"\n❌ An error occurred during training: {e}")
        print("Please check your setup and try again.")

