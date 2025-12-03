"""
======================================================
Sign Language Recognition - Model Training (Improved)
======================================================
This script implements improved models for sign language recognition.
Team: Haroon, Saria, Azmeer
Course: COMP-360 - Introduction to Artificial Intelligence
Institution: Forman Christian College
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Reshape, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
import traceback


# GPU Configuration
def configure_gpu():
    """Configure TensorFlow to use GPU if available."""
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU detected: {len(gpus)} GPU(s) available")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            return True
        except RuntimeError as e:
            print(f"⚠️  GPU configuration error: {e}")
            return False
    else:
        print("ℹ️  No GPU detected. Training will use CPU.")
        print("   For faster training, install TensorFlow with GPU support:")
        print("   pip install tensorflow[and-cuda]")
        return False


# Constants
Alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
Num_Alphabets = 26
Dimensions = 63  # 21 landmarks × 3 coordinates

# Model architecture flag
USE_CONV1D = False  # Set to True to use Conv1D instead of MLP

# Overfit debug mode (set via environment variable)
Overfit = (
    os.getenv('Overfit', '0') == '1' or 
    os.getenv('OVERFIT_DEBUG', '0') == '1' or
    os.getenv('overfit', '0') == '1'
)

if Overfit:
    print("⚠️  OVERFIT DEBUG MODE ENABLED")
    print("   Set via environment variable: Overfit=1 or OVERFIT_DEBUG=1")

# Get script directory
Script_dir = Path(__file__).parent.absolute()


def save_training_history(history, model_name):
    """Save training history to JSON file."""
    import json
    
    # Create logs directory
    logs_dir = Script_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Prepare history dictionary
    history_dict = {
        'train_accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'train_loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'epochs': len(history.history['accuracy']),
        'best_val_accuracy': float(max(history.history['val_accuracy'])),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1])
    }
    
    # Add learning rate history if available
    if 'lr' in history.history:
        history_dict['learning_rate'] = [float(x) for x in history.history['lr']]
    
    # Save to JSON
    log_path = logs_dir / f'{model_name}_history.json'
    with open(log_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"\n✅ Training history saved: {log_path}")
    print(f"   Epochs trained: {history_dict['epochs']}")
    print(f"   Best val accuracy: {history_dict['best_val_accuracy']:.4f}")
    
    return history_dict


def plot_training_history(history, model_name):
    """Plot and save training curves."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    
    # Create plots directory
    plots_dir = Script_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    ax1.plot(epochs_range, history.history['accuracy'], 
             label='Train Accuracy', linewidth=2, marker='o', markersize=3)
    ax1.plot(epochs_range, history.history['val_accuracy'], 
             label='Val Accuracy', linewidth=2, marker='s', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(epochs_range, history.history['loss'], 
             label='Train Loss', linewidth=2, marker='o', markersize=3)
    ax2.plot(epochs_range, history.history['val_loss'], 
             label='Val Loss', linewidth=2, marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = plots_dir / f"{model_name}_training_curves.png"
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved: {plot_path}")


def load_data():
    """Load preprocessed data from processed_data directory."""
    print("=" * 60)
    print("Loading preprocessed data")
    print("=" * 60)
    
    try:
        data_dir = Script_dir / "processed_data"
        X_train = np.load(str(data_dir / "X_train.npy"))
        X_test = np.load(str(data_dir / "X_test.npy"))
        y_train = np.load(str(data_dir / "y_train.npy"))
        y_test = np.load(str(data_dir / "y_test.npy"))
        
        print(f"✅ Data loaded successfully!")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find data file: {e}")
        print("Please run A2_preprocessing.py first to generate processed data.")
        return None, None, None, None
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None, None


def sanity_checks(X_train, X_test, y_train, y_test):
    """Perform sanity checks on the loaded data."""
    print("\n" + "=" * 60)
    print("Running sanity checks")
    print("=" * 60)
    
    # Check shapes
    print("\n1. Checking data shapes")
    assert len(X_train.shape) == 2, f"X_train should be 2D, got shape {X_train.shape}"
    assert len(X_test.shape) == 2, f"X_test should be 2D, got shape {X_test.shape}"
    assert X_train.shape[1] == Dimensions, f"X_train feature dim should be {Dimensions}, got {X_train.shape[1]}"
    assert X_test.shape[1] == Dimensions, f"X_test feature dim should be {Dimensions}, got {X_test.shape[1]}"
    assert y_train.shape[1] == Num_Alphabets, f"y_train should have {Num_Alphabets} classes, got {y_train.shape[1]}"
    assert y_test.shape[1] == Num_Alphabets, f"y_test should have {Num_Alphabets} classes, got {y_test.shape[1]}"
    print("   ✅ All shape assertions passed")
    
    # Check for NaN/Inf
    print("\n2. Checking for NaN and Inf values")
    if np.isnan(X_train).any():
        print("   ⚠️  WARNING: NaN values found in X_train!")
        return False
    if np.isnan(X_test).any():
        print("   ⚠️  WARNING: NaN values found in X_test!")
        return False
    if np.isinf(X_train).any():
        print("   ⚠️  WARNING: Inf values found in X_train!")
        return False
    if np.isinf(X_test).any():
        print("   ⚠️  WARNING: Inf values found in X_test!")
        return False
    print("   ✅ No NaN or Inf values found")
    
    # Check normalization
    print("\n3. Checking normalization...")
    train_mean = np.mean(X_train)
    train_std = np.std(X_train)
    print(f"   X_train mean: {train_mean:.6f} (should be ≈0)")
    print(f"   X_train std: {train_std:.6f} (should be ≈1)")
    
    if abs(train_mean) > 0.1:
        print("   ⚠️  WARNING: Data doesn't appear to be normalized (mean not near 0)")
        print("   Please re-run A2_preprocessing.py with normalization enabled")
    else:
        print("   ✅ Data appears to be normalized")
    
    # Check one-hot encoding
    print("\n4. Checking one-hot encoding...")
    train_sum = np.sum(y_train, axis=1)
    test_sum = np.sum(y_test, axis=1)
    if not np.allclose(train_sum, 1.0):
        print("   ⚠️  WARNING: y_train is not properly one-hot encoded!")
        return False
    if not np.allclose(test_sum, 1.0):
        print("   ⚠️  WARNING: y_test is not properly one-hot encoded!")
        return False
    print("   ✅ One-hot encoding verified")
    
    # Print class counts
    print("\n5. Class distribution:")
    train_classes = np.argmax(y_train, axis=1)
    test_classes = np.argmax(y_test, axis=1)
    
    print("   Training set class counts:")
    for i in range(min(5, Num_Alphabets)):  # Show first 5 classes
        count = np.sum(train_classes == i)
        print(f"      {Alphabets[i]}: {count}")
    print("      ...")
    
    print("\n✅ All sanity checks passed!")
    return True


def build_mlp_model(input_dim, Num_Alphabets):
    """Build improved MLP model with BatchNormalization and better regularization."""
    
    print("\n🏗️  Building improved MLP model")
    
    model = Sequential([
        # First hidden layer
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.4),
        
        # Second hidden layer
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        # Third hidden layer
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # Output layer
        Dense(Num_Alphabets, activation='softmax')
    ])
    
    print("✅ Model architecture:")
    print(f"   Input → Dense(256) → BN → Dropout(0.4)")
    print(f"   → Dense(128) → BN → Dropout(0.4)")
    print(f"   → Dense(64) → Dropout(0.3)")
    print(f"   → Output(26)")
    
    return model


def build_conv1d_model(input_dim, Num_Alphabets):
    """Build Conv1D model: Reshape (21,3) → Conv1D(64,3,relu) → MaxPool1D(2) → Flatten → Dense(64,relu) → Dense(26,softmax)"""
    print("\n🏗️  Building Conv1D model")
    model = Sequential([
        Reshape((21, 3), input_shape=(input_dim,)),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(Num_Alphabets, activation='softmax')
    ])
    return model


def train_model(model, X_train, y_train, X_test, y_test, model_name="cnn_baseline"):
    """Train the model with specified parameters."""
    print("\n" + "=" * 60)
    print(f"Training {model_name} model")
    print("=" * 60)
    
    # Split training data into train and validation
    from sklearn.model_selection import train_test_split as split
    
    X_train_split, X_val, y_train_split, y_val = split(
        X_train,
        y_train,
        test_size=0.2,  # 20% of training data for validation
        random_state=42,
        stratify=np.argmax(y_train, axis=1)  # Maintain class balance
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training: {X_train_split.shape[0]} samples ({X_train_split.shape[0]/X_train.shape[0]*100:.1f}% of train)")
    print(f"   Validation: {X_val.shape[0]} samples ({X_val.shape[0]/X_train.shape[0]*100:.1f}% of train)")
    print(f"   Test (held out): {X_test.shape[0]} samples")
    
    # Create models directory
    models_dir = Script_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n⚙️  Model compiled:")
    print(f"   Optimizer: Adam (lr=5e-4)")
    print(f"   Loss: categorical_crossentropy")
    print(f"   Metrics: accuracy")
    
    # Print model summary
    print("\n📋 Model architecture:")
    model.summary()
    
    # Setup callbacks
    checkpoint_path = str(models_dir / f"{model_name}.h5")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        mode='max',
        patience=15,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
        mode='max'
    )
    
    print(f"\n📋 Callbacks configured:")
    print(f"   ✅ ModelCheckpoint (save best model)")
    print(f"   ✅ EarlyStopping (patience=15)")
    print(f"   ✅ ReduceLROnPlateau (patience=5, factor=0.5)")
    
    # Training parameters
    batch_size = 32
    epochs = 200
    
    print(f"\n🎯 Training parameters:")
    print(f"   Batch size: {batch_size}")
    print(f"   Max epochs: {epochs}")
    print(f"   Early stopping will stop earlier if no improvement")
    
    if Overfit:
        print(f"\n⚠️  OVERFIT DEBUG MODE ACTIVE")
        print(f"   Using first 256 training samples only")
        X_train_final = X_train_split[:256]
        y_train_final = y_train_split[:256]
    else:
        X_train_final = X_train_split
        y_train_final = y_train_split
    
    # Train model
    print(f"\n🚀 Starting training...")
    history = model.fit(
        X_train_final,
        y_train_final,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping, reduce_lr],
        verbose=1
    )
    
    # Print final accuracies
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    print(f"Final training accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Final validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    # Overfit debug check
    if Overfit:
        if train_acc < 0.95:
            print("\n⚠️  WARNING: Overfit debug mode did not achieve near-100% train accuracy!")
            print("   This may indicate issues with preprocessing or labels.")
        else:
            print("\n✅ Overfit debug check passed: Model can memorize training data")
    
    # Save final model
    final_model_path = str(models_dir / "cnn_last.h5")
    model.save(final_model_path)
    print(f"\n✅ Model saved:")
    print(f"   Best model: {checkpoint_path}")
    print(f"   Final model: {final_model_path}")
    
    # Save training history
    save_training_history(history, model_name)
    
    # Plot training curves
    plot_training_history(history, model_name)
    
    return history


def main():
    """Main function to run the training pipeline."""
    print("=" * 60)
    print("Sign Language Recognition - Model Training")
    print("=" * 60)
    print("Team: Haroon, Saria, Azmeer")
    print("Course: COMP-360 - Introduction to Artificial Intelligence")
    print("=" * 60)
    
    # Configure GPU
    print("\n" + "=" * 60)
    print("Checking GPU availability...")
    print("=" * 60)
    gpu_available = configure_gpu()
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    if X_train is None:
        return
    
    # Sanity checks
    if not sanity_checks(X_train, X_test, y_train, y_test):
        print("\n❌ Sanity checks failed. Exiting")
        return
    
    # Build model
    input_dim = X_train.shape[1]
    Num_Alphabets_local = y_train.shape[1]
    
    if USE_CONV1D:
        model = build_conv1d_model(input_dim, Num_Alphabets_local)
        model_name = "cnn_baseline_conv1d"
    else:
        model = build_mlp_model(input_dim, Num_Alphabets_local)
        model_name = "cnn_baseline"
    
    # Train model
    history = train_model(model, X_train, y_train, X_test, y_test, model_name)
    
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("   1. Run: python A2_evaluate_model.py")
    print("   2. Run: python A2_app_skeleton.py")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        print("Exiting gracefully...")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        traceback.print_exc()