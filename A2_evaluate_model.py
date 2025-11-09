"""
Sign Language Recognition - Model Evaluation Module
=================================================

Project: Sign Language Recognition System
Course: Introduction to Artificial Intelligence (COMP-360)
Institution: Forman Christian College
Team: Haroon, Saria, Azmeer
Instructor: [Instructor Name]

Description:
This module provides comprehensive evaluation and visualization of trained sign language
recognition models. It loads trained models, evaluates their performance on test data,
and generates detailed visualizations including confusion matrices, accuracy plots,
and classification reports.

Features:
- Loads trained CNN and LSTM models
- Evaluates model performance on test data
- Generates confusion matrices with heatmaps
- Creates accuracy comparison charts
- Displays detailed classification reports
- Saves evaluation results and visualizations

Requirements:
- TensorFlow/Keras
- NumPy, Matplotlib, Seaborn
- Scikit-learn for metrics
- Trained models from train_model.py

Author: AI Coding Assistant
Date: 2024
"""

# Step 1 - Import Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from tensorflow.keras.models import load_model
from pathlib import Path
import time

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def load_test_data():
    """
    Load preprocessed test data for evaluation.
    
    Returns:
        tuple: (X_test, y_test) - Test data and labels
    """
    print("📥 Loading test data...")
    
    try:
        X_test = np.load("processed_data/X_test.npy")
        y_test = np.load("processed_data/y_test.npy")
        
        print(f"✅ Test data loaded successfully!")
        print(f"   📊 Test samples: {X_test.shape[0]}")
        print(f"   📊 Feature dimensions: {X_test.shape[1]}")
        print(f"   📊 Number of classes: {y_test.shape[1]}")
        
        return X_test, y_test
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run preprocessing.py first to generate the required data files.")
        return None, None

def load_trained_models():
    """
    Load trained CNN and LSTM models.
    
    Returns:
        tuple: (cnn_model, lstm_model) - Loaded models
    """
    print("📥 Loading trained models...")
    
    models = {}
    model_files = {'CNN': 'models/cnn_final.h5', 'LSTM': 'models/lstm_final.h5'}
    
    for model_name, model_path in model_files.items():
        if Path(model_path).exists():
            try:
                models[model_name] = load_model(model_path)
                print(f"✅ {model_name} model loaded from {model_path}")
            except Exception as e:
                print(f"❌ Error loading {model_name} model: {e}")
        else:
            print(f"⚠️  {model_name} model not found at {model_path}")
            print(f"   Please run train_model.py first to train the models.")
    
    return models

def evaluate_single_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model and return detailed metrics.
    
    Args:
        model: Trained Keras model
        X_test, y_test: Test data and labels
        model_name (str): Name of the model
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print(f"\n📊 Evaluating {model_name} model...")
    
    # Get predictions
    start_time = time.time()
    y_pred_proba = model.predict(X_test, verbose=0)
    prediction_time = time.time() - start_time
    
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Test loss and accuracy from model evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    # Store results
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'prediction_time': prediction_time,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"✅ {model_name} evaluation completed!")
    print(f"   📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   📊 Precision: {precision:.4f}")
    print(f"   📊 Recall: {recall:.4f}")
    print(f"   📊 F1-Score: {f1:.4f}")
    print(f"   📊 Prediction Time: {prediction_time:.4f} seconds")
    
    return results

def plot_confusion_matrix(y_true, y_pred, model_name, alphabet=ALPHABET):
    """
    Plot and save confusion matrix for a model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name (str): Name of the model
        alphabet (str): String of class labels
    """
    print(f"📊 Generating confusion matrix for {model_name}...")
    
    # Create plots directory
    Path('plots').mkdir(exist_ok=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(alphabet),
                yticklabels=list(alphabet),
                cbar_kws={'label': 'Number of Samples'})
    
    plt.title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Add accuracy text
    accuracy = accuracy_score(y_true, y_pred)
    plt.figtext(0.5, 0.02, f'Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)', 
                ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name.lower()}_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Confusion matrix saved as 'plots/{model_name.lower()}_confusion_matrix.png'")

def plot_model_comparison(results_list):
    """
    Plot comparison charts for multiple models.
    
    Args:
        results_list (list): List of evaluation results dictionaries
    """
    print("📊 Generating model comparison charts...")
    
    # Create plots directory
    Path('plots').mkdir(exist_ok=True)
    
    # Extract data for plotting
    model_names = [r['model_name'] for r in results_list]
    accuracies = [r['accuracy'] for r in results_list]
    precisions = [r['precision'] for r in results_list]
    recalls = [r['recall'] for r in results_list]
    f1_scores = [r['f1_score'] for r in results_list]
    prediction_times = [r['prediction_time'] for r in results_list]
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    bars1 = ax1.bar(model_names, accuracies, color=['skyblue', 'lightcoral'])
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Precision, Recall, F1-Score comparison
    x = np.arange(len(model_names))
    width = 0.25
    
    ax2.bar(x - width, precisions, width, label='Precision', color='lightgreen')
    ax2.bar(x, recalls, width, label='Recall', color='orange')
    ax2.bar(x + width, f1_scores, width, label='F1-Score', color='purple')
    
    ax2.set_title('Precision, Recall, and F1-Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Prediction time comparison
    bars3 = ax3.bar(model_names, prediction_times, color=['gold', 'lightblue'])
    ax3.set_title('Prediction Time Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Time (seconds)', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, time_val in zip(bars3, prediction_times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{time_val:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    # Overall performance radar chart
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Normalize prediction time for radar chart (invert so higher is better)
    max_time = max(prediction_times)
    normalized_times = [1 - (t / max_time) for t in prediction_times]
    
    # Create radar chart data
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    
    for i, model_name in enumerate(model_names):
        values = [accuracies[i], precisions[i], recalls[i], f1_scores[i]]
        values += values[:1]  # Complete the circle
        ax4.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax4.fill(angles, values, alpha=0.25)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 1)
    ax4.set_title('Overall Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Model comparison charts saved as 'plots/model_comparison.png'")

def generate_classification_reports(results_list, alphabet=ALPHABET):
    """
    Generate and save detailed classification reports.
    
    Args:
        results_list (list): List of evaluation results dictionaries
        alphabet (str): String of class labels
    """
    print("📋 Generating detailed classification reports...")
    
    # Create reports directory
    Path('reports').mkdir(exist_ok=True)
    
    for results in results_list:
        model_name = results['model_name']
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        print(f"\n📊 {model_name} Classification Report:")
        print("=" * 50)
        
        # Generate classification report
        report = classification_report(y_true, y_pred, 
                                     target_names=list(alphabet),
                                     output_dict=True)
        
        # Print to console
        print(classification_report(y_true, y_pred, target_names=list(alphabet)))
        
        # Save to file
        with open(f'reports/{model_name.lower()}_classification_report.txt', 'w') as f:
            f.write(f"{model_name} Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(classification_report(y_true, y_pred, target_names=list(alphabet)))
            f.write(f"\n\nDetailed Metrics:\n")
            f.write(f"Overall Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1-Score: {results['f1_score']:.4f}\n")
            f.write(f"Prediction Time: {results['prediction_time']:.4f} seconds\n")
        
        print(f"✅ {model_name} report saved as 'reports/{model_name.lower()}_classification_report.txt'")

def main():
    """
    Main function to execute the complete model evaluation pipeline.
    """
    print("=" * 60)
    print("Sign Language Recognition - Model Evaluation Pipeline")
    print("=" * 60)
    print("Team: Haroon, Saria, Azmeer")
    print("Course: COMP-360 - Introduction to Artificial Intelligence")
    print("Institution: Forman Christian College")
    print("=" * 60)
    
    # Step 1 - Load Test Data
    X_test, y_test = load_test_data()
    
    if X_test is None:
        print("❌ Failed to load test data. Exiting...")
        return
    
    # Step 2 - Load Trained Models
    models = load_trained_models()
    
    if not models:
        print("❌ No trained models found. Please run train_model.py first.")
        return
    
    # Step 3 - Evaluate Each Model
    print(f"\n" + "="*50)
    print("📊 MODEL EVALUATION")
    print("="*50)
    
    results_list = []
    
    for model_name, model in models.items():
        results = evaluate_single_model(model, X_test, y_test, model_name)
        results_list.append(results)
    
    # Step 4 - Generate Confusion Matrices
    print(f"\n" + "="*50)
    print("📊 CONFUSION MATRICES")
    print("="*50)
    
    for results in results_list:
        plot_confusion_matrix(results['y_true'], results['y_pred'], 
                            results['model_name'])
    
    # Step 5 - Generate Comparison Charts
    print(f"\n" + "="*50)
    print("📊 MODEL COMPARISON")
    print("="*50)
    
    plot_model_comparison(results_list)
    
    # Step 6 - Generate Classification Reports
    print(f"\n" + "="*50)
    print("📋 CLASSIFICATION REPORTS")
    print("="*50)
    
    generate_classification_reports(results_list)
    
    # Step 7 - Summary
    print(f"\n" + "="*60)
    print("🎉 MODEL EVALUATION COMPLETE!")
    print("="*60)
    
    print(f"📊 Evaluation Summary:")
    for results in results_list:
        print(f"   • {results['model_name']}: {results['accuracy']:.4f} accuracy")
    
    # Find best model
    best_model = max(results_list, key=lambda x: x['accuracy'])
    print(f"\n🏆 Best performing model: {best_model['model_name']}")
    print(f"   📊 Accuracy: {best_model['accuracy']:.4f} ({best_model['accuracy']*100:.2f}%)")
    print(f"   📊 F1-Score: {best_model['f1_score']:.4f}")
    
    print(f"\n📁 Generated Files:")
    print(f"   • plots/cnn_confusion_matrix.png")
    print(f"   • plots/lstm_confusion_matrix.png")
    print(f"   • plots/model_comparison.png")
    print(f"   • reports/cnn_classification_report.txt")
    print(f"   • reports/lstm_classification_report.txt")
    
    print("="*60)

if __name__ == "__main__":
    """
    Execute the model evaluation pipeline when script is run directly.
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user.")
        print("Exiting gracefully...")
    except Exception as e:
        print(f"\n❌ An error occurred during evaluation: {e}")
        print("Please check your setup and try again.")