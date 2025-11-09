"""
======================================================
Sign Language Recognition - Model Training (Baseline)
======================================================
This script implements baseline models for sign language recognition.
Team: Haroon, Saria, Azmeer
Course: COMP-360 - Introduction to Artificial Intelligence
Institution: Forman Christian College
"""

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from pathlib import Path

# Constants
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUM_CLASSES = 26
FEATURE_DIM = 63  # 21 landmarks × 3 coordinates

# Model architecture flag
USE_CONV1D = False  # Set to True to use Conv1D instead of MLP

# Overfit debug mode (set via environment variable OVERFIT_DEBUG=1)
OVERFIT_DEBUG = os.getenv('OVERFIT_DEBUG', '0') == '1'

# Get script directory
SCRIPT_DIR = Path(__file__).parent.absolute()


def load_data():
    """Load preprocessed data from processed_data directory."""
    print("=" * 60)
    print("Loading preprocessed data...")
    print("=" * 60)
    
    try:
        data_dir = SCRIPT_DIR / "processed_data"
        X_train = np.load(str(data_dir / "X_train.npy"))
        X_test = np.load(str(data_dir / "X_test.npy"))
        y_train = np.load(str(data_dir / "y_train.npy"))
        y_test = np.load(str(data_dir / "y_test.npy"))
        
        print(f"✓ Data loaded successfully!")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
        
    except FileNotFoundError as e:
        print(f"✗ Error: Could not find data file: {e}")
        print("Please run A1_preprocessing.py first to generate processed data.")
        return None, None, None, None
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None, None, None, None


def sanity_checks(X_train, X_test, y_train, y_test):
    """Perform sanity checks on the loaded data."""
    print("\n" + "=" * 60)
    print("Running sanity checks...")
    print("=" * 60)
    
    # Check shapes
    print("\n1. Checking data shapes...")
    assert len(X_train.shape) == 2, f"X_train should be 2D, got shape {X_train.shape}"
    assert len(X_test.shape) == 2, f"X_test should be 2D, got shape {X_test.shape}"
    assert X_train.shape[1] == FEATURE_DIM, f"X_train feature dim should be {FEATURE_DIM}, got {X_train.shape[1]}"
    assert X_test.shape[1] == FEATURE_DIM, f"X_test feature dim should be {FEATURE_DIM}, got {X_test.shape[1]}"
    assert y_train.shape[1] == NUM_CLASSES, f"y_train should have {NUM_CLASSES} classes, got {y_train.shape[1]}"
    assert y_test.shape[1] == NUM_CLASSES, f"y_test should have {NUM_CLASSES} classes, got {y_test.shape[1]}"
    print("  ✓ All shape assertions passed")
    
    # Check for NaN/Inf
    print("\n2. Checking for NaN and Inf values...")
    if np.isnan(X_train).any():
        print("  ✗ WARNING: NaN values found in X_train!")
        return False
    if np.isnan(X_test).any():
        print("  ✗ WARNING: NaN values found in X_test!")
        return False
    if np.isinf(X_train).any():
        print("  ✗ WARNING: Inf values found in X_train!")
        return False
    if np.isinf(X_test).any():
        print("  ✗ WARNING: Inf values found in X_test!")
        return False
    print("  ✓ No NaN or Inf values found")
    
    # Check one-hot encoding
    print("\n3. Checking one-hot encoding...")
    train_sum = np.sum(y_train, axis=1)
    test_sum = np.sum(y_test, axis=1)
    if not np.allclose(train_sum, 1.0):
        print("  ✗ WARNING: y_train is not properly one-hot encoded!")
        return False
    if not np.allclose(test_sum, 1.0):
        print("  ✗ WARNING: y_test is not properly one-hot encoded!")
        return False
    print("  ✓ One-hot encoding verified")
    
    # Print class counts
    print("\n4. Class distribution:")
    train_classes = np.argmax(y_train, axis=1)
    test_classes = np.argmax(y_test, axis=1)
    
    print("  Training set class counts:")
    for i in range(NUM_CLASSES):
        count = np.sum(train_classes == i)
        print(f"    {CLASS_NAMES[i]}: {count}")
    
    print("  Test set class counts:")
    for i in range(NUM_CLASSES):
        count = np.sum(test_classes == i)
        print(f"    {CLASS_NAMES[i]}: {count}")
    
    print("\n✓ All sanity checks passed!")
    return True


def build_mlp_model(input_dim, num_classes):
    """Build MLP model: Input(63) → Dense(128,relu) → Dropout(0.3) → Dense(64,relu) → Dense(26,softmax)"""
    print("\nBuilding MLP model...")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model


def build_conv1d_model(input_dim, num_classes):
    """Build Conv1D model: Reshape (21,3) → Conv1D(64,3,relu) → MaxPool1D(2) → Flatten → Dense(64,relu) → Dense(26,softmax)"""
    print("\nBuilding Conv1D model...")
    model = Sequential([
        Reshape((21, 3), input_shape=(input_dim,)),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model


def train_model(model, X_train, y_train, X_test, y_test, model_name="cnn_baseline"):
    """Train the model with specified parameters."""
    print("\n" + "=" * 60)
    print(f"Training {model_name} model...")
    print("=" * 60)
    
    # Create models directory
    models_dir = SCRIPT_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\nModel architecture:")
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
    
    # Training parameters
    batch_size = 64
    epochs = 100 if OVERFIT_DEBUG else 20
    
    print(f"\nTraining parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Validation data: Test set ({X_test.shape[0]} samples)")
    
    if OVERFIT_DEBUG:
        print(f"\n⚠️  OVERFIT DEBUG MODE ACTIVE")
        print(f"  Using first 256 training samples only")
        X_train_debug = X_train[:256]
        y_train_debug = y_train[:256]
    else:
        X_train_debug = X_train
        y_train_debug = y_train
    
    # Train model
    history = model.fit(
        X_train_debug,
        y_train_debug,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint],
        verbose=1
    )
    
    # Print final accuracies
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    print(f"Final training accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Final test accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    
    # Overfit debug check
    if OVERFIT_DEBUG:
        if train_acc < 0.95:
            print("\n⚠️  WARNING: Overfit debug mode did not achieve near-100% train accuracy!")
            print("   This may indicate issues with preprocessing or labels.")
        else:
            print("\n✓ Overfit debug check passed: Model can memorize training data")
    
    # Save final model
    final_model_path = str(models_dir / "cnn_last.h5")
    model.save(final_model_path)
    print(f"\n✓ Model saved:")
    print(f"  Best model: {checkpoint_path}")
    print(f"  Final model: {final_model_path}")
    
    return history


def main():
    """Main function to run the training pipeline."""
    print("=" * 60)
    print("Sign Language Recognition - Baseline Model Training")
    print("=" * 60)
    print("Team: Haroon, Saria, Azmeer")
    print("Course: COMP-360 - Introduction to Artificial Intelligence")
    print("Institution: Forman Christian College")
    print("=" * 60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    if X_train is None:
        return
    
    # Sanity checks
    if not sanity_checks(X_train, X_test, y_train, y_test):
        print("\n✗ Sanity checks failed. Exiting...")
        return
    
    # Overfit debug mode: slice first 256 samples
    if OVERFIT_DEBUG:
        print("\n" + "=" * 60)
        print("⚠️  OVERFIT DEBUG MODE ENABLED")
        print("=" * 60)
        print("Using first 256 training samples for debugging...")
        X_train = X_train[:256]
        y_train = y_train[:256]
    
    # Build model
    input_dim = X_train.shape[1]
    num_classes = y_train.shape[1]
    
    if USE_CONV1D:
        model = build_conv1d_model(input_dim, num_classes)
        model_name = "cnn_baseline_conv1d"
    else:
        model = build_mlp_model(input_dim, num_classes)
        model_name = "cnn_baseline"
    
    # Train model
    history = train_model(model, X_train, y_train, X_test, y_test, model_name)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        print("Exiting gracefully...")
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
        import traceback
        traceback.print_exc()
