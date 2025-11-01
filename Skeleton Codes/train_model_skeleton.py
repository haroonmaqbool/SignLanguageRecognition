# Model Training - Skeleton Code
# Course: COMP-360
# Team: Haroon, Saria, Azmeer

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
import os

def load_preprocessed_data():
    # Load the data files that were saved by preprocessing.py
    
    # TODO: Load X_train.npy, X_test.npy, y_train.npy, y_test.npy from processed_data folder
    # Use: np.load("processed_data/X_train.npy")
    
    # TODO: Print the shapes to check if data loaded correctly
    
    # TODO: Return X_train, X_test, y_train, y_test
    
    pass

def build_cnn_model(input_shape, num_classes):
    # Build CNN model for classification
    # input_shape: shape of input data (like (63,) for 63 features)
    # num_classes: 26 for letters A-Z
    
    model = Sequential()
    
    # TODO: Add layers to build CNN
    # Need: Conv1D, MaxPooling, Dropout, Dense layers
    # Example: model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
    # Final layer should be Dense with num_classes and softmax
    
    # TODO: Compile model
    # optimizer = Adam, loss = 'categorical_crossentropy', metrics = ['accuracy']
    
    return model

def build_lstm_model(input_shape, num_classes):
    # Build LSTM model for classification
    # Same parameters as CNN: input_shape and num_classes
    
    model = Sequential()
    
    # TODO: Add LSTM layers
    # Use LSTM layer(s), can set return_sequences=True for first LSTM
    # Add Dense layers, Dropout if needed
    # Final layer: Dense with num_classes and softmax
    
    # TODO: Compile model (same as CNN)
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, model_name, epochs=50):
    # Train the model
    # model_name is used for saving (like "cnn" or "lstm")
    
    # TODO: Create models folder if needed
    # os.makedirs('models', exist_ok=True)
    
    # TODO: Train using model.fit()
    # Use X_train, y_train for training
    # Use X_test, y_test for validation (validation_data parameter)
    # Set epochs, batch_size=32
    
    # TODO: Save trained model
    # model.save(f'models/{model_name}_final.h5')
    
    print(f"Model {model_name} training done!")
    pass

def plot_training_history(history, model_name):
    # Plot accuracy and loss graphs (optional)
    
    # TODO: Plot accuracy: history.history['accuracy'] and history.history['val_accuracy']
    # TODO: Plot loss: history.history['loss'] and history.history['val_loss']
    # TODO: Save plot: plt.savefig(f'plots/{model_name}_training.png')
    
    pass

def main():
    print("Starting model training...")
    
    # TODO: Load data
    # X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # TODO: Get shapes
    # input_shape = (X_train.shape[1],)
    # num_classes = y_train.shape[1]
    
    # TODO: Build and train CNN
    # cnn = build_cnn_model(input_shape, num_classes)
    # train_model(cnn, X_train, y_train, X_test, y_test, "cnn")
    
    # TODO: Build and train LSTM
    # lstm = build_lstm_model(input_shape, num_classes)
    # train_model(lstm, X_train, y_train, X_test, y_test, "lstm")
    
    print("Done!")

if __name__ == "__main__":
    main()

