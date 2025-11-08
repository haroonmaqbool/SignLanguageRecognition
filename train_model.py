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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from pathlib import Path

# Constants
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUM_CLASSES = 26

# Configuration flags
USE_NORMALIZED = True  # Set to True to use processed_data_norm/, False for processed_data/
USE_CONV1D = False  # Set to True to use Conv1D instead of MLP
KEEP_Z = False  # Must match preprocessing setting (42 vs 63 features)

# Overfit debug mode (set via environment variable OVERFIT_DEBUG=1)
OVERFIT_DEBUG = os.getenv('OVERFIT_DEBUG', '0') == '1'

# Get script directory
SCRIPT_DIR = Path(__file__).parent.absolute()


def load_data():
    """Load preprocessed data from processed_data or processed_data_norm directory."""
    print("=" * 60)
    print("Loading preprocessed data...")
    print("=" * 60)
    
    # Choose data directory based on USE_NORMALIZED flag
    if USE_NORMALIZED:
        data_dir = SCRIPT_DIR / "processed_data_norm"
        print("Using normalized data from processed_data_norm/")
    else:
        data_dir = SCRIPT_DIR / "processed_data"
        print("Using original data from processed_data/")
    
    try:
        X_train = np.load(str(data_dir / "X_train.npy"))
        X_test = np.load(str(data_dir / "X_test.npy"))
        y_train = np.load(str(data_dir / "y_train.npy"))
        y_test = np.load(str(data_dir / "y_test.npy"))
        
        print(f"✓ Data loaded successfully!")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Feature dimensions: {X_train.shape[1]}")
        
        # Verify feature dimension matches KEEP_Z setting
        expected_dim = 42 if (USE_NORMALIZED and not KEEP_Z) else 63
        if X_train.shape[1] != expected_dim:
            print(f"⚠️  Warning: Feature dimension {X_train.shape[1]} doesn't match expected {expected_dim}")
            print(f"   (USE_NORMALIZED={USE_NORMALIZED}, KEEP_Z={KEEP_Z})")
        
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


def build_conv1d_model(input_dim, num_classes, keep_z=False):
    """Build Conv1D model: Reshape appropriately → Conv1D(64,3,relu) → MaxPool1D(2) → Flatten → Dense(64,relu) → Dense(26,softmax)"""
    print("\nBuilding Conv1D model...")
    if keep_z or input_dim == 63:
        reshape_shape = (21, 3)
    else:
        reshape_shape = (21, 2)
    
    model = Sequential([
        Reshape(reshape_shape, input_shape=(input_dim,)),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model


def compute_class_weights(y_train):
    """Compute class weights if imbalance >15%."""
    train_classes = np.argmax(y_train, axis=1)
    class_counts = np.zeros(NUM_CLASSES)
    for i in range(NUM_CLASSES):
        class_counts[i] = np.sum(train_classes == i)
    
    max_count = np.max(class_counts)
    min_count = np.min(class_counts)
    imbalance_ratio = (max_count - min_count) / max_count if max_count > 0 else 0.0
    
    if imbalance_ratio > 0.15:
        print(f"\n⚠️  Class imbalance detected: {imbalance_ratio*100:.1f}%")
        print("   Computing class weights...")
        total = np.sum(class_counts)
        class_weights = {}
        for i in range(NUM_CLASSES):
            if class_counts[i] > 0:
                class_weights[i] = total / (NUM_CLASSES * class_counts[i])
            else:
                class_weights[i] = 1.0
        return class_weights
    else:
        print(f"\n✓ Class distribution balanced (imbalance: {imbalance_ratio*100:.1f}%)")
        return None


def train_model(model, X_train, y_train, X_test, y_test, model_name="cnn_baseline"):
    """Train the model with specified parameters."""
    print("\n" + "=" * 60)
    print(f"Training {model_name} model...")
    print("=" * 60)
    
    # Create models and logs directories
    models_dir = SCRIPT_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    logs_dir = SCRIPT_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\nModel architecture:")
    model.summary()
    
    # Compute class weights if needed
    class_weights = compute_class_weights(y_train)
    
    # Setup callbacks
    checkpoint_path = str(models_dir / f"{model_name}.h5")
    callbacks_list = [
        ModelCheckpoint(
            checkpoint_path,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger(
            str(logs_dir / "experiments.csv"),
            append=True
        )
    ]
    
    # Training parameters
    batch_size = 64
    epochs = 100 if OVERFIT_DEBUG else 20
    
    print(f"\nTraining parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Validation data: Test set ({X_test.shape[0]} samples)")
    if class_weights is not None:
        print(f"  Class weights: ENABLED")
    
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
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )
    
    # Find best epoch and accuracy
    best_epoch = np.argmax(history.history['val_accuracy'])
    best_val_acc = history.history['val_accuracy'][best_epoch]
    best_train_acc = history.history['accuracy'][best_epoch]
    
    # Print compact training summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Best epoch: {best_epoch + 1}")
    print(f"Training accuracy at best epoch: {best_train_acc:.4f} ({best_train_acc*100:.2f}%)")
    
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"Final training accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"Final test accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    
    # Overfit debug check
    if OVERFIT_DEBUG:
        if final_train_acc < 0.95:
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
    print(f"  Training log: {logs_dir / 'experiments.csv'}")
    
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
        model = build_conv1d_model(input_dim, num_classes, keep_z=KEEP_Z)
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
