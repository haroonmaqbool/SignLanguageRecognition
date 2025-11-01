"""
======================================================
Sign Language Recognition - Data Preprocessing 
======================================================
This script implements the model evaluation for sign language recognition.
Team: Haroon, Saria, Azmeer
Course: COMP-360 - Introduction to Artificial Intelligence
Institution: Forman Christian College
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import load_model

def load_test_data():
    # Load test data for evaluation
    
    # TODO: Load X_test.npy and y_test.npy from processed_data folder
    
    # TODO: Print how many test samples we have
    
    # TODO: Return X_test, y_test
    
    pass

def load_trained_models():
    # Load the trained CNN and LSTM models
    
    # TODO: Load models using load_model() from models folder
    # Example: cnn_model = load_model("models/cnn_final.h5")
    
    # TODO: Return both models
    
    pass

def evaluate_model(model, X_test, y_test, model_name):
    # Evaluate a model and calculate accuracy
    
    # TODO: Make predictions using model.predict(X_test)
    
    # TODO: Convert predictions from probabilities to class numbers
    # Use np.argmax() to get the class with highest probability
    # predicted_classes = np.argmax(predictions, axis=1)
    # true_classes = np.argmax(y_test, axis=1)
    
    # TODO: Calculate accuracy using accuracy_score()
    
    # TODO: Print confusion matrix and classification report (optional)
    
    print(f"{model_name} accuracy: ...")
    # return accuracy
    pass

def compare_models(cnn_accuracy, lstm_accuracy):
    # Compare which model is better
    
    # TODO: Print both accuracies
    
    # TODO: Say which one is better
    
    # TODO: Optional: Make a bar chart comparing them
    
    pass

def main():
    print("Evaluating models...")
    
    # TODO: Load test data
    # X_test, y_test = load_test_data()
    
    # TODO: Load models
    # cnn_model, lstm_model = load_trained_models()
    
    # TODO: Evaluate CNN
    # cnn_accuracy = evaluate_model(cnn_model, X_test, y_test, "CNN")
    
    # TODO: Evaluate LSTM
    # lstm_accuracy = evaluate_model(lstm_model, X_test, y_test, "LSTM")
    
    # TODO: Compare them
    # compare_models(cnn_accuracy, lstm_accuracy)
    
    print("Evaluation done!")

if __name__ == "__main__":
    main()
