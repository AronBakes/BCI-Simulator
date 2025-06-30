import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

class MultiModalLSTMModel:
    """A class to build and train a multi-modal LSTM model for EEG, image, and text data.

    This class combines LSTM for time series (EEG) with dense layers for image and text features,
    targeting neural decoding tasks.
    """

    def __init__(self, lstm_input_shape, image_shape, text_shape, units=64, dropout_rate=0.2):
        """Initialize the MultiModalLSTMModel with input shapes and hyperparameters.

        Args:
            lstm_input_shape (tuple): Shape of EEG windows (window_size, n_channels).
            image_shape (tuple): Shape of flattened image features (n_features,).
            text_shape (tuple): Shape of text features (n_features,).
            units (int, optional): Number of LSTM units. Defaults to 64.
            dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.

        Attributes:
            lstm_input_shape (tuple): EEG input shape.
            image_shape (tuple): Image input shape.
            text_shape (tuple): Text input shape.
            units (int): Number of units.
            dropout_rate (float): Dropout rate.
            model (Model): Compiled Keras model.
        """
        self.lstm_input_shape = lstm_input_shape
        self.image_shape = image_shape
        self.text_shape = text_shape
        self.units = units
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    def build_model(self):
        """Build the multi-modal LSTM model architecture.

        Returns:
            Model: Compiled Keras model with LSTM, image, and text branches.

        Notes:
            Uses Concatenate to combine features before final prediction.
        """
        lstm_input = Input(shape=self.lstm_input_shape)
        x = LSTM(self.units, return_sequences=True)(lstm_input)
        x = Dropout(self.dropout_rate)(x)
        x = LSTM(self.units // 2)(x)
        lstm_output = Dropout(self.dropout_rate)(x)

        image_input = Input(shape=self.image_shape)
        image_flat = tf.keras.layers.Flatten()(image_input)
        image_dense = Dense(self.units // 2, activation='relu')(image_flat)

        text_input = Input(shape=self.text_shape)
        text_dense = Dense(self.units // 2, activation='relu')(text_input)

        combined = Concatenate()([lstm_output, image_dense, text_dense])
        x = Dense(32, activation='relu')(combined)
        output = Dense(1, activation='sigmoid')(x)  # Binary classification (e.g., high activity)

        model = Model(inputs=[lstm_input, image_input, text_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_lstm, X_image, X_text, y, epochs=20, batch_size=32, validation_split=0.2,
              save_path="data/outputs/model.keras"):
        """Train the multi-modal LSTM model.

        Args:
            X_lstm (np.ndarray): EEG windows with shape (n_samples, window_size, n_channels).
            X_image (np.ndarray): Image features with shape (n_samples, n_features).
            X_text (np.ndarray): Text features with shape (n_samples, n_features).
            y (np.ndarray): Labels with shape (n_samples,).
            epochs (int, optional): Number of training epochs. Defaults to 20.
            batch_size (int, optional): Batch size. Defaults to 32.
            validation_split (float, optional): Fraction of data for validation. Defaults to 0.2.
            save_path (str, optional): Path to save model weights as .keras file. Defaults to "data/outputs/model.keras".

        Notes:
            Uses EarlyStopping to prevent overfitting.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = self.model.fit(
            [X_lstm, X_image, X_text], y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save(save_path, save_format='keras')  # Explicitly save as .keras
        self.plot_history(history)

    def evaluate(self, X_lstm_test, X_image_test, X_text_test, y_test):
        """Evaluate the model on test data.

        Args:
            X_lstm_test (np.ndarray): Test EEG windows.
            X_image_test (np.ndarray): Test image features.
            X_text_test (np.ndarray): Test text features.
            y_test (np.ndarray): Test labels.

        Returns:
            tuple: (loss, accuracy) from model evaluation.

        Notes:
            Evaluates without verbose output.
        """
        return self.model.evaluate([X_lstm_test, X_image_test, X_text_test], y_test, verbose=0)

    def plot_history(self, history, save_path="data/outputs/training_history.png"):
        """Plot training history of loss and accuracy.

        Args:
            history: Keras training history object.
            save_path (str, optional): Path to save the plot. Defaults to "data/outputs/training_history.png".

        Notes:
            Saves two subplots (loss and accuracy) side by side.
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
    # Load preprocessed data
    eeg_data = np.load("data/processed/eeg_processed.npz")
    X_lstm, y = eeg_data['windows'], eeg_data['labels']
    
    image_data = np.load("data/raw/brain_images.npz")
    X_image = image_data['images']  # Use images directly (flatten if needed later)
    
    X_text = np.load("data/raw/text_data.npz")['features']
    
    # Align sample sizes (for simplicity, take min length)
    min_samples = min(X_lstm.shape[0], X_image.shape[0], X_text.shape[0])
    X_lstm, X_image, X_text, y = X_lstm[:min_samples], X_image[:min_samples], X_text[:min_samples], y[:min_samples]
    
    # Reshape for model
    X_image = X_image.reshape((X_image.shape[0], -1))  # Flatten images
    X_text = X_text.reshape((X_text.shape[0], -1))  # Ensure 2D
    
    # Split data
    X_lstm_train, X_lstm_test, X_image_train, X_image_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
        X_lstm, X_image, X_text, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    model = MultiModalLSTMModel(
        lstm_input_shape=(X_lstm.shape[1], X_lstm.shape[2]),
        image_shape=(X_image.shape[1],),
        text_shape=(X_text.shape[1],)
    )
    model.train(X_lstm_train, X_image_train, X_text_train, y_train)
    
    loss, accuracy = model.evaluate(X_lstm_test, X_image_test, X_text_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")