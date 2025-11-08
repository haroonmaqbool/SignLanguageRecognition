"""
======================================================
Sign Language Recognition - Model Evaluation
======================================================
This script implements the model evaluation for sign language recognition.
Team: Haroon, Saria, Azmeer
Course: COMP-360 - Introduction to Artificial Intelligence
Institution: Forman Christian College
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from tensorflow.keras.models import load_model
from pathlib import Path

# Constants
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUM_CLASSES = 26

# Get script directory
SCRIPT_DIR = Path(__file__).parent.absolute()

# Optional: Path to original images for misclassification visualization
# Set to None if images are not available
ORIGINAL_IMAGE_PATHS = None  # Can be set to a dictionary mapping indices to image paths


def load_test_data():
    """Load test data for evaluation."""
    print("=" * 60)
    print("Loading test data...")
    print("=" * 60)
    
    try:
        data_dir = SCRIPT_DIR / "processed_data"
        X_test = np.load(str(data_dir / "X_test.npy"))
        y_test = np.load(str(data_dir / "y_test.npy"))
        
        print(f"✓ Test data loaded successfully!")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Feature dimensions: {X_test.shape[1]}")
        print(f"  Number of classes: {y_test.shape[1]}")
        
        return X_test, y_test
        
    except FileNotFoundError as e:
        print(f"✗ Error: Could not find data file: {e}")
        print("Please run A1_preprocessing.py first to generate processed data.")
        return None, None
    except Exception as e:
        print(f"✗ Error loading test data: {e}")
        return None, None


def load_trained_models():
    """Load the trained models."""
    print("\n" + "=" * 60)
    print("Loading trained models...")
    print("=" * 60)
    
    models_dir = SCRIPT_DIR / "models"
    models = {}
    
    # Load baseline model (required)
    baseline_path = models_dir / "cnn_baseline.h5"
    if baseline_path.exists():
        try:
            models['baseline'] = load_model(str(baseline_path))
            print(f"✓ Loaded baseline model: {baseline_path}")
        except Exception as e:
            print(f"✗ Error loading baseline model: {e}")
    else:
        print(f"⚠️  Baseline model not found: {baseline_path}")
        print("   Please run train_model.py first to train the model.")
    
    # Load final model (optional)
    final_path = models_dir / "cnn_last.h5"
    if final_path.exists():
        try:
            models['final'] = load_model(str(final_path))
            print(f"✓ Loaded final model: {final_path}")
        except Exception as e:
            print(f"✗ Error loading final model: {e}")
    else:
        print(f"ℹ️  Final model not found: {final_path} (optional)")
    
    return models


def find_top_confusion_pairs(cm, top_k=3):
    """Find top-K confusion pairs (excluding diagonal)."""
    confusion_pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j:  # Exclude diagonal (correct predictions)
                confusion_pairs.append((i, j, cm[i, j]))
    
    # Sort by count (descending)
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    return confusion_pairs[:top_k]


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and calculate accuracy, confusion matrix, and classification report."""
    print("\n" + "=" * 60)
    print(f"Evaluating {model_name} model...")
    print("=" * 60)
    
    # Make predictions
    print("Making predictions...")
    preds = model.predict(X_test, verbose=0)
    
    # Convert predictions from probabilities to class numbers
    predicted_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(true_classes, predicted_classes)
    print(f"\n✓ {model_name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_classes, predicted_classes, labels=range(NUM_CLASSES), zero_division=0
    )
    
    # Save per-class metrics to CSV
    plots_dir = SCRIPT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    csv_path = plots_dir / "per_class_report.csv"
    
    print(f"\nSaving per-class metrics to CSV...")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
        for i in range(NUM_CLASSES):
            writer.writerow([CLASS_NAMES[i], precision[i], recall[i], f1[i], int(support[i])])
    print(f"✓ Per-class metrics saved: {csv_path}")
    
    # Find top-3 confusion pairs
    top_confusions = find_top_confusion_pairs(cm, top_k=3)
    print(f"\nTop-3 Confusion Pairs:")
    for idx, (true_class, pred_class, count) in enumerate(top_confusions, 1):
        print(f"  {idx}. True: {CLASS_NAMES[true_class]} → Predicted: {CLASS_NAMES[pred_class]} ({count} samples)")
    
    # Plot confusion matrix (counts)
    print("\nGenerating confusion matrix plot (counts)...")
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Counts) - {model_name}', fontsize=16, fontweight='bold')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save plot
    plot_path = plots_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix (counts) saved: {plot_path}")
    plt.close()
    
    # Plot normalized confusion matrix (percentages)
    print("\nGenerating normalized confusion matrix plot (percentages)...")
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    plt.figure(figsize=(14, 12))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Normalized) - {model_name}', fontsize=16, fontweight='bold')
    plt.colorbar(label='Percentage')
    
    # Add labels
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES)
    
    # Add text annotations (percentages)
    thresh = cm_normalized.max() / 2.0
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(j, i, format(cm_normalized[i, j], '.2f'),
                    horizontalalignment="center",
                    color="white" if cm_normalized[i, j] > thresh else "black",
                    fontsize=8)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save normalized plot
    plot_path_norm = plots_dir / f"confusion_matrix_normalized_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(str(plot_path_norm), dpi=300, bbox_inches='tight')
    print(f"✓ Normalized confusion matrix saved: {plot_path_norm}")
    plt.close()
    
    # Optional: Save misclassified examples grid
    if ORIGINAL_IMAGE_PATHS is not None and len(top_confusions) > 0:
        print("\nGenerating misclassified examples grid...")
        try:
            import cv2
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            axes = axes.flatten()
            plot_idx = 0
            
            for true_class, pred_class, count in top_confusions[:3]:
                # Find misclassified examples
                misclassified_indices = []
                for idx in range(len(true_classes)):
                    if true_classes[idx] == true_class and predicted_classes[idx] == pred_class:
                        misclassified_indices.append(idx)
                    if len(misclassified_indices) >= 16:
                        break
                
                # Plot up to 16 examples
                for idx in misclassified_indices[:16]:
                    if plot_idx >= 16:
                        break
                    if idx in ORIGINAL_IMAGE_PATHS:
                        img_path = ORIGINAL_IMAGE_PATHS[idx]
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            axes[plot_idx].imshow(img_rgb)
                            axes[plot_idx].set_title(f"True: {CLASS_NAMES[true_class]}, Pred: {CLASS_NAMES[pred_class]}")
                            axes[plot_idx].axis('off')
                            plot_idx += 1
            
            # Hide unused subplots
            for idx in range(plot_idx, 16):
                axes[idx].axis('off')
            
            plt.tight_layout()
            grid_path = plots_dir / f"misclassified_examples_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(str(grid_path), dpi=300, bbox_inches='tight')
            print(f"✓ Misclassified examples grid saved: {grid_path}")
            plt.close()
        except Exception as e:
            print(f"⚠️  Could not generate misclassified examples grid: {e}")
    
    # Print classification report
    print("\n" + "-" * 60)
    print("Classification Report:")
    print("-" * 60)
    report = classification_report(true_classes, predicted_classes, target_names=CLASS_NAMES)
    print(report)
    
    return accuracy


def main():
    """Main function to run evaluation."""
    print("=" * 60)
    print("Sign Language Recognition - Model Evaluation")
    print("=" * 60)
    print("Team: Haroon, Saria, Azmeer")
    print("Course: COMP-360 - Introduction to Artificial Intelligence")
    print("Institution: Forman Christian College")
    print("=" * 60)
    
    # Load test data
    X_test, y_test = load_test_data()
    if X_test is None:
        return
    
    # Load models
    models = load_trained_models()
    if not models:
        print("\n✗ No models loaded. Exiting...")
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
        print(f"\n🏆 Best performing model: {best_model.capitalize()} ({results[best_model]:.4f})")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user.")
        print("Exiting gracefully...")
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
        import traceback
        traceback.print_exc()
