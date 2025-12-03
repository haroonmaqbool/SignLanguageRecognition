"""
======================================================
Sign Language Recognition - Model Evaluation
======================================================
This script implements the model evaluation for sign language recognition.
Team: Haroon, Saria, Azmeer
Course: COMP-360 - Introduction to Artificial Intelligence
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from pathlib import Path
import traceback

# Constants - Will be auto-detected from model/data
Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # Default, will be updated
Num_Alphabets = 26  # Default, will be auto-detected

# Get script directory
Script_dir = Path(__file__).parent.absolute()


def load_test_data():
    """Load test data for evaluation."""
    global Alphabets, Num_Alphabets
    
    print("=" * 60)
    print("Loading test data")
    print("=" * 60)
    
    try:
        data_dir = Script_dir / "processed_data"
        X_test = np.load(str(data_dir / "X_test.npy"))
        y_test = np.load(str(data_dir / "y_test.npy"))
        
        # Auto-detect number of classes from data
        Num_Alphabets = y_test.shape[1]
        
        # Update Alphabets list based on number of classes
        if Num_Alphabets == 29:
            # Model trained with A-Z + space + del + nothing
            Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [" ", "DEL", "NONE"]
            print(f"   Detected 29 classes: A-Z + space + del + nothing")
        elif Num_Alphabets == 27:
            # Model trained with A-Z + space
            Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [" "]
            print(f"   Detected 27 classes: A-Z + space")
        elif Num_Alphabets == 26:
            # Model trained with A-Z only
            Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            print(f"   Detected 26 classes: A-Z only")
        else:
            # Unknown number, use default
            Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:Num_Alphabets] if Num_Alphabets <= 26 else list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [" "] * (Num_Alphabets - 26)
            print(f"   Warning: Unknown number of classes ({Num_Alphabets}), using default mapping")
        
        print(f"Test data loaded successfully!")
        print(f"Test samples: {X_test.shape[0]}")
        print(f"Feature dimensions: {X_test.shape[1]}")
        print(f"Number of classes: {Num_Alphabets}")
        
        return X_test, y_test
        
    except FileNotFoundError as e:
        print(f" Error: Could not find data file: {e}")
        print("Please run A2_preprocessing.py first to generate processed data.")
        return None, None
    except Exception as e:
        print(f" Error loading test data: {e}")
        return None, None


def load_trained_models():
    """Load the trained models and verify class count matches."""
    global Alphabets, Num_Alphabets
    print("\n" + "=" * 60)
    print("Loading trained models")
    print("=" * 60)
    
    models_dir = Script_dir / "models"
    models = {}
    
    # Load baseline model (required)
    baseline_path = models_dir / "cnn_baseline.h5"
    if baseline_path.exists():
        try:
            models['baseline'] = load_model(str(baseline_path))
            print(f" Loaded baseline model: {baseline_path}")
        except Exception as e:
            print(f" Error loading baseline model: {e}")
    else:
        print(f"  Baseline model not found: {baseline_path}")
        print("   Please run train_model.py first to train the model.")
    
    # Load final model (optional)
    final_path = models_dir / "cnn_last.h5"
    if final_path.exists():
        try:
            models['final'] = load_model(str(final_path))
            print(f" Loaded final model: {final_path}")
        except Exception as e:
            print(f" Error loading final model: {e}")
    else:
        print(f" Final model not found: {final_path} (optional)")
    
    return models


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and calculate accuracy, confusion matrix, and classification report."""
    print("\n" + "=" * 60)
    print(f"Evaluating {model_name} model")
    print("=" * 60)
    
    # Make predictions
    print("Making predictions")
    preds = model.predict(X_test, verbose=0)
    
    # Convert predictions from probabilities to class numbers
    predicted_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(true_classes, predicted_classes)
    print(f"\n {model_name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix plot")
    # Adjust figure size based on number of classes (larger for 29 classes)
    fig_size = (16, 14) if Num_Alphabets > 26 else (14, 12)
    plt.figure(figsize=fig_size)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name} ({Num_Alphabets} classes)', fontsize=16, fontweight='bold')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(Alphabets))
    # Adjust font size based on number of classes
    font_size = 7 if Num_Alphabets > 26 else 8
    plt.xticks(tick_marks, Alphabets, rotation=45, fontsize=font_size)
    plt.yticks(tick_marks, Alphabets, fontsize=font_size)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    annotation_font_size = 6 if Num_Alphabets > 26 else 8
    for i in range(len(Alphabets)):
        for j in range(len(Alphabets)):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=annotation_font_size)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    plots_dir = Script_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    print(f" Confusion matrix saved: {plot_path}")
    plt.close()
    
    # Print classification report
    print("\n" + "-" * 60)
    print("Classification Report:")
    print("-" * 60)
    report = classification_report(true_classes, predicted_classes, target_names=Alphabets)
    print(report)
    
    return accuracy


def main():
    """Main function to run evaluation."""
    print("=" * 60)
    print("Sign Language Recognition - Model Evaluation")
    print("=" * 60)
    print("Team: Haroon, Saria, Azmeer")
    print("Course: COMP-360 - Introduction to Artificial Intelligence")
    print("=" * 60)
    
    # Load test data
    X_test, y_test = load_test_data()
    if X_test is None:
        return
    
    # Load models
    models = load_trained_models()
    if not models:
        print("\n No models loaded. Exiting")
        return
    
    # Evaluate each model
    results = {}
    for model_key, model in models.items():
        model_display_name = f"{model_key.capitalize()} Model"
        accuracy = evaluate_model(model, X_test, y_test, model_display_name)
        results[model_key] = accuracy
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    for model_key, accuracy in results.items():
        print(f"{model_key.capitalize()} Model: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if len(results) > 1:
        best_model = max(results, key=results.get)
        print(f"\n Best performing model: {best_model.capitalize()} ({results[best_model]:.4f})")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Evaluation interrupted by user.")
        print("Exiting gracefully...")
    except Exception as e:
        print(f"\n An error occurred: {e}")
        traceback.print_exc()
