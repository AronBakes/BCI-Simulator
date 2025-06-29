import numpy as np
import os

class EEGPreprocessor:
    def __init__(self, window_size=256, step_size=128, threshold=1.5):
        """
        Initialize EEG data preprocessor.
        
        Args:
            window_size (int): Number of time points in each window (1 sec at 256 Hz)
            step_size (int): Step size for sliding window
            threshold (float): Amplitude threshold for binary labels
        """
        self.window_size = window_size
        self.step_size = step_size
        self.threshold = threshold

    def normalize(self, data):
        """
        Normalize EEG data to zero mean and unit variance.
        
        Args:
            data (np.ndarray): EEG data (n_samples, n_channels, time_points)
            
        Returns:
            np.ndarray: Normalized data
        """
        data = data - np.mean(data, axis=2, keepdims=True)
        data = data / (np.std(data, axis=2, keepdims=True) + 1e-8)
        return data

    def create_windows(self, data):
        """
        Create sliding windows for LSTM input.
        
        Args:
            data (np.ndarray): EEG data (n_samples, n_channels, time_points)
            
        Returns:
            np.ndarray: Windows (n_windows, window_size, n_channels)
            np.ndarray: Labels (n_windows,)
        """
        n_samples, n_channels, time_points = data.shape
        windows = []
        labels = []

        for i in range(n_samples):
            for j in range(0, time_points - self.window_size + 1, self.step_size):
                window = data[i, :, j:j + self.window_size].T  # Shape: (window_size, n_channels)
                windows.append(window)
                # Label: 1 if max amplitude exceeds threshold, else 0
                label = 1 if np.max(np.abs(window)) > self.threshold else 0
                labels.append(label)

        return np.array(windows), np.array(labels)

    def preprocess(self, data_path="data/raw/eeg_data.npy", save_path="data/processed/eeg_processed.npz"):
        """
        Preprocess EEG dataset and save results.
        
        Args:
            data_path (str): Path to raw EEG data
            save_path (str): Path to save processed data
            
        Returns:
            np.ndarray: Windows
            np.ndarray: Labels
        """
        data = np.load(data_path)
        data = self.normalize(data)
        windows, labels = self.create_windows(data)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, windows=windows, labels=labels)
        return windows, labels

if __name__ == "__main__":
    # Example usage
    preprocessor = EEGPreprocessor(window_size=256, step_size=128)
    windows, labels = preprocessor.preprocess()
    print(f"Windows shape: {windows.shape}, Labels shape: {labels.shape}")