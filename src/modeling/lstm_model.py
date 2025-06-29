import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

class LSTMModel:
    def __init__(self, input_shape, units=64, dropout_rate=0.2):
        """
        Initialize LSTM model.
        
        Args:
            input_shape (tuple): Shape of input (window_size, n_channels)
            units (int): Number of LSTM units
            dropout_rate (float): Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.units = units
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    def build_model(self):
        """
        Build LSTM model.
        
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        model = Sequential([
            LSTM(self.units, input_shape=self.input_shape, return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(self.units // 2),
            Dropout(self.dropout_rate),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X, y, epochs=20, batch_size=32, validation_split=0.2, save_path="data/outputs/model.h5"):
        """
        Train the LSTM model.
        
        Args:
            X (np.ndarray): Input windows (n_windows, window_size, n_channels)
            y (np.ndarray): Labels (n_windows,)
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            validation_split (float): Fraction of data for validation
            save_path (str): Path to save model weights
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path)
        self.plot_history(history)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model.
        
        Args:
            X_test (np.ndarray): Test windows
            y_test (np.ndarray): Test labels
            
        Returns:
            tuple: Test loss, test accuracy
        """
        return self.model.evaluate(X_test, y_test, verbose=0)

    def plot_history(self, history, save_path="data/outputs/training_history.png"):
        """
        Plot training history.
        
        Args:
            history: Keras training history
            save_path (str): Path to save plot
        """
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    # Example usage
    data = np.load("data/processed/eeg_processed.npz")
    X, y = data['windows'], data['labels']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LSTMModel(input_shape=(X.shape[1], X.shape[2]))
    model.train(X_train, y_train)
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")